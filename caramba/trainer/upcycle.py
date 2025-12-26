"""Model upcycling: converting pretrained models to new architectures.

Upcycling takes a pretrained model (like Llama) and trains it to use a new
architecture (like DBA attention) while preserving its learned knowledge.
We do this by distillation: the original model is the "teacher" and the
new architecture is the "student." The student learns to produce the same
outputs as the teacher, then we fine-tune on language modeling to recover
any lost performance.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from caramba.config.defaults import Defaults
from caramba.config.eval import EvalVerifyConfig
from caramba.config.group import Group
from caramba.config.kvcache import KVCachePolicyConfig, KVCachePolicyDecoupledConfig
from caramba.config.layer import AttentionMode
from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.run import Run
from caramba.config.train import TrainConfig, TrainPhase
from caramba.config.verify import CompareVerifyConfig, KVCacheVerifyConfig
from caramba.console import logger
from caramba.data.npy import NpyDataset
from caramba.eval.suite import assert_eval_thresholds, run_eval_verify
from caramba.layer.attention import AttentionLayer
from caramba.loader.checkpoint import CheckpointLoader
from caramba.loader.hf import HFLoader
from caramba.loader.llama_upcycle import LlamaUpcycle
from caramba.model import Model
from caramba.trainer.blockwise import BlockwiseConfig, BlockwiseTrainer
from caramba.trainer.compare import assert_thresholds, compare_teacher_student
from caramba.trainer.distill import DistillLoss
from caramba.trainer.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedStrategy,
)


def _make_teacher_model_config(model_config: ModelConfig) -> ModelConfig:
    """Create a teacher model config with standard attention throughout.

    The teacher must use the original Llama architecture (standard attention)
    so we can load the pretrained weights correctly. The student uses the
    new architecture (e.g., DBA) from the manifest.
    """

    def rewrite_node_dict(node_dict: dict[str, Any]) -> dict[str, Any]:
        """Recursively convert attention layers to standard mode."""
        if node_dict.get("type") == "AttentionLayer":
            node_dict = node_dict.copy()
            node_dict["mode"] = "standard"
            # Remove DBA-specific fields
            node_dict.pop("sem_dim", None)
            node_dict.pop("geo_dim", None)
            node_dict.pop("decoupled_gate", None)
            node_dict.pop("decoupled_gate_dynamic", None)
            return node_dict

        if "layers" in node_dict and isinstance(node_dict["layers"], list):
            node_dict = node_dict.copy()
            node_dict["layers"] = [
                rewrite_node_dict(layer) if isinstance(layer, dict) else layer
                for layer in node_dict["layers"]
            ]

        return node_dict

    teacher_data = model_config.model_dump()
    teacher_data["topology"] = rewrite_node_dict(teacher_data["topology"])
    return ModelConfig.model_validate(teacher_data)


class Upcycle:
    """Orchestrates the full upcycling pipeline: distillation, fine-tuning, verification.

    The pipeline has two main training phases:
    1. Blockwise distillation: Train each attention layer individually to match the teacher
    2. Global fine-tuning: Train the whole model on language modeling loss

    After training, verification checks that the student produces outputs similar
    to the teacher, catching training failures before expensive benchmarking.
    """

    # Bytes per element for different precision formats
    BYTES_PER_FP32: float = 4.0
    BYTES_PER_FP16: float = 2.0
    BYTES_PER_Q8_0: float = 1.0
    BYTES_PER_Q4_0: float = 0.625

    def __init__(
        self,
        manifest: Manifest,
        group: Group,
        train: TrainConfig,
        *,
        dist_config: DistributedConfig | None = None,
        defaults: Defaults | None = None,
        checkpoint_dir: Path | str | None = None,
        resume_from: Path | str | None = None,
    ) -> None:
        """Initialize the upcycling trainer.

        This loads both the teacher (original architecture) and student (new
        architecture), applies the pretrained weights, and sets up distributed
        training if configured.

        Args:
            manifest: Model architecture specification
            group: Experiment group with data paths and settings
            train: Training hyperparameters
            dist_config: Optional distributed training settings
            defaults: Optional global defaults (save frequency, etc.)
            checkpoint_dir: Where to save checkpoints
            resume_from: Path to resume training from a checkpoint
        """
        self.manifest = manifest
        self.group = group
        self.defaults = defaults

        self.save_every = defaults.save_every if defaults else 500
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("runs") / group.name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up distributed training if configured
        self.dist_config = dist_config
        self.dist_ctx: DistributedContext | None = None
        if dist_config is not None and dist_config.strategy != DistributedStrategy.NONE:
            self.dist_ctx = DistributedContext.init(dist_config)
            self.device = self.dist_ctx.device
        else:
            self.device = self.parse_device(train.device)

        self.device_name = str(self.device)
        self.dtype = self.parse_dtype(train.dtype)
        self.teacher: nn.Module
        self.student: nn.Module
        self.init_models(train)

        if resume_from is not None:
            self.load_checkpoint(Path(resume_from))

    def run(self, run: Run) -> None:
        """Execute a single training run (blockwise or global phase).

        After training completes, runs any configured verification steps.
        """
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")

        torch.manual_seed(run.seed)

        match run.train.phase:
            case TrainPhase.BLOCKWISE:
                self.run_blockwise(run)
            case TrainPhase.GLOBAL:
                self.run_global(run)
            case _:
                raise ValueError(f"Unsupported train phase: {run.train.phase}")

        self.verify(run)

    def verify(self, run: Run) -> None:
        """Run post-training verification to catch failures early.

        Verification is cheaper than full benchmarking, so we use it as a
        quick sanity check before proceeding to expensive evaluations.
        """
        cfg = run.verify
        if cfg is None:
            return

        logger.header("Verification")

        if isinstance(cfg, CompareVerifyConfig):
            self.verify_compare(run, cfg)
        elif isinstance(cfg, EvalVerifyConfig):
            self.verify_eval(run, cfg)
        elif isinstance(cfg, KVCacheVerifyConfig):
            self.verify_kvcache(run, cfg)

    def verify_compare(self, run: Run, cfg: CompareVerifyConfig) -> None:
        """Compare teacher and student outputs on sample batches.

        This measures how well the student learned to match the teacher.
        With fail_fast=False, violations are logged as warnings but don't
        stop the pipeline—useful for seeing benchmark results even when
        thresholds are exceeded.
        """
        logger.info(f"Running teacher/student comparison on {cfg.batches} batches...")

        train = self.require_train(run)
        batches = self.collect_compare_batches(
            batch_size=train.batch_size,
            block_size=train.block_size,
            count=cfg.batches,
        )

        result = compare_teacher_student(
            teacher=self.teacher,
            student=self.student,
            batches=batches,
            predicate=lambda _, m: isinstance(m, AttentionLayer),
            attention=cfg.attention,
            logits=cfg.logits,
        )
        logger.success(f"Comparison complete • {result.batches} batches verified")

        # Log metrics for visibility
        metrics: dict[str, str] = {}
        if result.attention_mean_l1 is not None:
            metrics["attention_mean_l1"] = f"{result.attention_mean_l1:.6f}"
        if result.attention_max_l1 is not None:
            metrics["attention_max_l1"] = f"{result.attention_max_l1:.6f}"
        if result.logits_mean_l1 is not None:
            metrics["logits_mean_l1"] = f"{result.logits_mean_l1:.6f}"
        if result.logits_max_l1 is not None:
            metrics["logits_max_l1"] = f"{result.logits_max_l1:.6f}"
        if metrics:
            logger.key_value(metrics)

        # Check thresholds
        violations = assert_thresholds(
            result=result,
            attention=cfg.attention,
            logits=cfg.logits,
            fail_fast=cfg.fail_fast,
        )

        if violations:
            for v in violations:
                logger.warning(f"Threshold exceeded: {v.message()}")
            logger.warning(
                f"Verification found {len(violations)} threshold violation(s), "
                "but fail_fast=False so continuing to benchmarks..."
            )

    def verify_eval(self, run: Run, cfg: EvalVerifyConfig) -> None:
        """Run a small behavioral evaluation suite.

        Tests basic model capabilities (completion, Q&A) to catch catastrophic
        failures that might not show up in loss metrics.
        """
        logger.info("Running behavioral evaluation suite...")

        summary = run_eval_verify(
            teacher=self.teacher,
            student=self.student,
            cfg=cfg,
            device=self.device,
        )

        logger.key_value({
            "Teacher accuracy": f"{summary.teacher_accuracy:.1%}",
            "Student accuracy": f"{summary.student_accuracy:.1%}",
        })
        logger.success("Evaluation complete")
        assert_eval_thresholds(summary=summary, thresholds=cfg.thresholds)

    def verify_kvcache(self, run: Run, cfg: KVCacheVerifyConfig) -> None:
        """Compare KV-cache memory between teacher and student.

        For DBA upcycling, we expect the student to use significantly less
        KV-cache memory due to the compressed attention dimensions.
        """
        logger.info("Analyzing KV-cache memory footprint...")

        teacher_bytes = self._estimate_model_kvcache_bytes(
            self.teacher, cfg.teacher, cfg.n_layers
        )
        student_bytes = self._estimate_model_kvcache_bytes_decoupled(
            self.student, cfg.student, cfg.n_layers
        )

        teacher_total = teacher_bytes * cfg.batch_size * cfg.max_seq_len
        student_total = student_bytes * cfg.batch_size * cfg.max_seq_len

        reduction = teacher_total / student_total if student_total > 0 else float("inf")

        logger.key_value({
            "Teacher KV-cache": f"{teacher_total / 1024 / 1024:.2f} MB",
            "Student KV-cache": f"{student_total / 1024 / 1024:.2f} MB",
            "Reduction": f"{reduction:.2f}x",
        })
        logger.success("KV-cache analysis complete")

        if cfg.min_reduction_ratio is not None and reduction < cfg.min_reduction_ratio:
            raise AssertionError(
                f"KV cache reduction {reduction:.2f}x is below minimum {cfg.min_reduction_ratio}x"
            )

    def _kind_to_bytes(self, kind: str) -> float:
        """Convert precision/quantization format to bytes per element."""
        kind_lower = kind.lower() if isinstance(kind, str) else str(kind.value).lower()
        if kind_lower == "fp32":
            return self.BYTES_PER_FP32
        elif kind_lower == "fp16":
            return self.BYTES_PER_FP16
        elif kind_lower == "q8_0":
            return self.BYTES_PER_Q8_0
        else:
            return self.BYTES_PER_Q4_0

    def _estimate_model_kvcache_bytes(
        self,
        model: nn.Module,
        policy: KVCachePolicyConfig,
        n_layers: int,
    ) -> int:
        """Estimate bytes per token for a standard model's KV cache."""
        k_dim = 0
        v_dim = 0
        for module in model.modules():
            if isinstance(module, AttentionLayer):
                k_dim = module.config.kv_heads * module.config.head_dim
                v_dim = k_dim
                break

        k_bytes = k_dim * self._kind_to_bytes(policy.k.kind.value)
        v_bytes = v_dim * self._kind_to_bytes(policy.v.kind.value)
        return int((k_bytes + v_bytes) * n_layers)

    def _estimate_model_kvcache_bytes_decoupled(
        self,
        model: nn.Module,
        policy: KVCachePolicyDecoupledConfig,
        n_layers: int,
    ) -> int:
        """Estimate bytes per token for a DBA model's KV cache.

        DBA has separate semantic and geometric key dimensions, which are
        typically much smaller than the original key dimension.
        """
        sem_dim = 0
        geo_dim = 0
        v_dim = 0
        for module in model.modules():
            if isinstance(module, AttentionLayer):
                cfg = module.config
                if cfg.mode == AttentionMode.DECOUPLED:
                    sem_dim = cfg.sem_dim or cfg.d_model
                    geo_dim = cfg.geo_dim or cfg.d_model
                    v_dim = cfg.v_dim
                else:
                    sem_dim = cfg.kv_heads * cfg.head_dim
                    geo_dim = 0
                    v_dim = sem_dim
                break

        k_sem_bytes = sem_dim * self._kind_to_bytes(policy.k_sem.kind.value)
        k_geo_bytes = geo_dim * self._kind_to_bytes(policy.k_geo.kind.value)
        v_bytes = v_dim * self._kind_to_bytes(policy.v.kind.value)
        return int((k_sem_bytes + k_geo_bytes + v_bytes) * n_layers)

    def collect_compare_batches(
        self, batch_size: int, block_size: int, count: int
    ) -> list[Tensor]:
        """Load a small set of batches for verification.

        We use the same data as training but don't shuffle, ensuring
        deterministic verification results.
        """
        path = Path(self.group.data)
        dataset = NpyDataset(str(path), block_size=block_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        batches: list[Tensor] = []
        for x, _ in loader:
            batches.append(x.to(device=self.device))
            if len(batches) >= count:
                break
        return batches

    def init_models(self, train: TrainConfig) -> None:
        """Build teacher and student models and load pretrained weights.

        The teacher uses standard Llama attention (matching the checkpoint),
        while the student uses the manifest's architecture (e.g., DBA).
        Both start with identical weights; the student's new parameters
        are initialized appropriately by the upcycle surgery.
        """
        if train.teacher_ckpt is None:
            raise ValueError("train.teacher_ckpt is required for upcycle.")

        logger.header("Model Initialization")

        # Load checkpoint
        logger.info(f"Loading teacher checkpoint: {train.teacher_ckpt}")
        ckpt_path = self.resolve_teacher_ckpt(train.teacher_ckpt)
        state_dict = CheckpointLoader().load(ckpt_path)
        logger.success(f"Loaded checkpoint with {len(state_dict)} keys")

        # Create teacher with standard attention
        logger.info("Building teacher model (standard attention)...")
        teacher_model_config = _make_teacher_model_config(self.manifest.model)
        self.teacher = Model(teacher_model_config).to(device=self.device, dtype=self.dtype)
        logger.success("Teacher model ready")

        # Create student with manifest's architecture
        logger.info("Building student model (manifest attention)...")
        self.student = Model(self.manifest.model).to(device=self.device, dtype=self.dtype)
        logger.success("Student model ready")

        # Apply pretrained weights
        logger.info("Applying weights to teacher...")
        LlamaUpcycle(self.teacher, state_dict).apply()

        logger.info("Applying upcycle surgery to student...")
        LlamaUpcycle(self.student, state_dict).apply()
        logger.success("Weight transfer complete")

        # Set up distributed training
        if self.dist_ctx is not None:
            logger.info("Wrapping student for distributed training...")
            self.student = self.dist_ctx.wrap_model(self.student)

        # Apply torch.compile for speed (PyTorch 2.0+)
        if train.compile_model and hasattr(torch, "compile"):
            if self.device.type == "cuda":
                logger.info("Compiling models with torch.compile...")
                try:
                    self.teacher = torch.compile(self.teacher, mode="reduce-overhead")  # type: ignore[assignment]
                    self.student = torch.compile(self.student, mode="reduce-overhead")  # type: ignore[assignment]
                    logger.success("torch.compile applied")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without: {e}")
            else:
                logger.warning(f"torch.compile not supported on {self.device.type}, skipping")

        self.teacher.eval()
        logger.success("Initialization complete")

    def _log(self, msg: str) -> None:
        """Log only from the main process in distributed mode."""
        if self.dist_ctx is not None:
            self.dist_ctx.log(msg)
        else:
            logger.info(msg)

    def save_checkpoint(
        self,
        run_id: str,
        phase: str,
        step: int,
        *,
        block_index: int | None = None,
        is_final: bool = False,
    ) -> Path:
        """Save the student model weights to disk.

        Checkpoints allow resuming training after interruption and provide
        snapshots for analysis. We save after each block completes and
        periodically during training.
        """
        if is_final:
            filename = f"{run_id}_{phase}_final.pt"
        elif block_index is not None:
            filename = f"{run_id}_{phase}_block{block_index}_step{step}.pt"
        else:
            filename = f"{run_id}_{phase}_step{step}.pt"

        path = self.checkpoint_dir / filename

        state = {
            "student_state_dict": self.student.state_dict(),
            "run_id": run_id,
            "phase": phase,
            "step": step,
            "block_index": block_index,
        }

        if self.dist_ctx is not None:
            self.dist_ctx.save_checkpoint(state, str(path))
        else:
            torch.save(state, path)

        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: Path) -> dict[str, object]:
        """Resume training from a saved checkpoint.

        Loads the student weights and returns metadata about where training
        left off (phase, step, block index).
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.student.load_state_dict(state["student_state_dict"])
        logger.success(
            f"Resumed from checkpoint: run={state.get('run_id')}, "
            f"phase={state.get('phase')}, step={state.get('step')}"
        )

        return {
            "run_id": state.get("run_id"),
            "phase": state.get("phase"),
            "step": state.get("step"),
            "block_index": state.get("block_index"),
        }

    def get_latest_checkpoint(self, run_id: str, phase: str) -> Path | None:
        """Find the most recent checkpoint for a run/phase.

        Checks for a final checkpoint first, then falls back to the latest
        intermediate checkpoint by modification time.
        """
        final_path = self.checkpoint_dir / f"{run_id}_{phase}_final.pt"
        if final_path.exists():
            return final_path

        pattern = f"{run_id}_{phase}_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def _unfreeze_all_parameters(self) -> None:
        """Re-enable gradients on all student parameters.

        Blockwise training freezes most parameters, so we need to unfreeze
        everything before global fine-tuning.
        """
        for param in self.student.parameters():
            param.requires_grad = True

    def run_blockwise(self, run: Run) -> None:
        """Run blockwise distillation: train each layer to match the teacher.

        This is the first training phase. We train one block (attention layer)
        at a time, freezing all others. This ensures stable inputs for each
        block during training.

        Supports two stopping criteria:
        1. Fixed steps: Train each block for exactly N steps
        2. Convergence: Train until loss drops below a target or patience runs out
        """
        train = self.require_train(run)
        loader = self.build_loader(train)
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=train.lr)

        blockwise_config = self._build_blockwise_config(train)

        trainer = BlockwiseTrainer(
            teacher=self.teacher,
            student=self.student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _, m: isinstance(m, AttentionLayer),
            config=blockwise_config,
        )

        self._log_optimization_settings(train)

        n_blocks = trainer.block_count()
        self.student.train()
        loader_iter = iter(loader)

        # Choose between convergence-based and fixed-step training
        if train.convergence_target is not None:
            target_loss = float(train.convergence_target)
            patience = train.convergence_patience
            max_steps = train.convergence_max_steps
            logger.header(
                "Blockwise Distillation (Convergence)",
                f"{n_blocks} blocks • target_loss={target_loss:.4f} • "
                f"patience={patience} • max_steps={max_steps}",
            )
            self._run_blockwise_convergence(
                run_id=run.id,
                trainer=trainer,
                loader=loader,
                loader_iter=loader_iter,
                n_blocks=n_blocks,
                target_loss=target_loss,
                patience=patience,
                max_steps=max_steps,
            )
        else:
            logger.header("Blockwise Distillation", f"{n_blocks} blocks × {run.steps} steps")
            self._run_blockwise_fixed(
                run_id=run.id,
                trainer=trainer,
                loader=loader,
                loader_iter=loader_iter,
                n_blocks=n_blocks,
                steps_per_block=run.steps,
            )

        self.save_checkpoint(run.id, "blockwise", step=0, is_final=True)

    def _run_blockwise_fixed(
        self,
        *,
        run_id: str,
        trainer: BlockwiseTrainer,
        loader: DataLoader[tuple[Tensor, Tensor]],
        loader_iter: Iterator[tuple[Tensor, Tensor]],
        n_blocks: int,
        steps_per_block: int,
    ) -> None:
        """Train each block for a fixed number of steps.

        Simple and predictable, but may over-train easy blocks and under-train
        difficult ones.
        """
        total_steps = n_blocks * steps_per_block
        global_step = 0

        with logger.progress_bar() as progress:
            overall_task = progress.add_task(
                f"Training {n_blocks} blocks...", total=total_steps
            )

            for block_index in range(n_blocks):
                loss = None

                for step in range(steps_per_block):
                    (x, _), loader_iter = self.next_batch(loader, loader_iter)
                    x = x.to(device=self.device)
                    loss = trainer.step(x, block_index=block_index)
                    global_step += 1

                    loss_val = float(loss) if loss is not None else 0.0
                    progress.update(
                        overall_task,
                        advance=1,
                        description=(
                            f"Block {block_index + 1}/{n_blocks} • "
                            f"step {step + 1}/{steps_per_block} • loss={loss_val:.4f}"
                        ),
                    )

                    if self.save_every > 0 and global_step % self.save_every == 0:
                        self.save_checkpoint(
                            run_id, "blockwise", global_step, block_index=block_index
                        )

                if loss is not None:
                    if self.dist_ctx is not None:
                        loss = self.dist_ctx.all_reduce(loss, op="avg")
                    logger.success(
                        f"Block {block_index + 1}/{n_blocks} complete • "
                        f"loss={float(loss):.6f}"
                    )

                self.save_checkpoint(
                    run_id, "blockwise", global_step, block_index=block_index
                )

    def _run_blockwise_convergence(
        self,
        *,
        run_id: str,
        trainer: BlockwiseTrainer,
        loader: DataLoader[tuple[Tensor, Tensor]],
        loader_iter: Iterator[tuple[Tensor, Tensor]],
        n_blocks: int,
        target_loss: float,
        patience: int,
        max_steps: int,
    ) -> None:
        """Train each block until it converges or patience runs out.

        More efficient than fixed steps because easy blocks finish quickly
        and difficult blocks get more training. We stop a block when:
        - Loss drops below target_loss (success)
        - No improvement for `patience` steps (stuck)
        - max_steps reached (budget exhausted)
        """
        total_steps_taken = 0

        for block_index in range(n_blocks):
            best_loss = float("inf")
            steps_without_improvement = 0
            step = 0
            loss: Tensor | None = None

            logger.info(
                f"Block {block_index + 1}/{n_blocks} starting (target={target_loss:.4f})..."
            )

            while step < max_steps:
                (x, _), loader_iter = self.next_batch(loader, loader_iter)
                x = x.to(device=self.device)
                loss = trainer.step(x, block_index=block_index)
                loss_val = float(loss)
                step += 1
                total_steps_taken += 1

                if loss_val < best_loss:
                    best_loss = loss_val
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if step % 50 == 0:
                    logger.info(
                        f"  Block {block_index + 1} • step {step} • "
                        f"loss={loss_val:.6f} • best={best_loss:.6f} • "
                        f"patience={patience - steps_without_improvement}"
                    )

                if self.save_every > 0 and total_steps_taken % self.save_every == 0:
                    self.save_checkpoint(
                        run_id, "blockwise", total_steps_taken, block_index=block_index
                    )

                if loss_val <= target_loss:
                    logger.success(
                        f"Block {block_index + 1}/{n_blocks} converged! "
                        f"loss={loss_val:.6f} ≤ target={target_loss:.4f} after {step} steps"
                    )
                    break

                if steps_without_improvement >= patience:
                    logger.warning(
                        f"Block {block_index + 1}/{n_blocks} stopped early "
                        f"(patience exhausted) • best_loss={best_loss:.6f} after {step} steps"
                    )
                    break
            else:
                final_loss = float(loss) if loss is not None else best_loss
                logger.warning(
                    f"Block {block_index + 1}/{n_blocks} reached max_steps={max_steps} • "
                    f"final_loss={final_loss:.6f}"
                )

            self.save_checkpoint(
                run_id, "blockwise", total_steps_taken, block_index=block_index
            )

        logger.success(f"Blockwise distillation complete • total_steps={total_steps_taken}")

    def run_global(self, run: Run) -> None:
        """Run global fine-tuning on language modeling loss.

        This is the second training phase. After blockwise distillation,
        we fine-tune the entire model end-to-end on next-token prediction.
        This recovers any performance lost during architecture conversion.
        """
        train = self.require_train(run)
        loader = self.build_loader(train)

        # Unfreeze all params (blockwise leaves most frozen)
        self._unfreeze_all_parameters()

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=train.lr)
        self.student.train()

        has_diffusion = self._has_diffusion_head()

        logger.header("Global Fine-tuning", f"{run.steps} steps")

        loader_iter = iter(loader)
        loss: Tensor | None = None

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=run.steps)

            for step in range(run.steps):
                (x, y), loader_iter = self.next_batch(loader, loader_iter)
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                ce_loss: Tensor
                diff_loss: Tensor | None = None

                if has_diffusion:
                    result = self.student.forward(x, return_features=True)  # type: ignore[call-arg]
                    features: Tensor = result[0]  # type: ignore[index]
                    logits: Tensor = result[1]  # type: ignore[index]
                    ce_loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]), y.reshape(-1)
                    )
                    diff_loss_val: Tensor = self.student.diffusion_loss(features, y)  # type: ignore[attr-defined]
                    diff_loss = diff_loss_val
                    diff_weight = self._get_diffusion_loss_weight()
                    loss = ce_loss + diff_weight * diff_loss_val
                else:
                    logits = self.student.forward(x)
                    ce_loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]), y.reshape(-1)
                    )
                    loss = ce_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_val = float(loss) if loss is not None else 0.0
                if has_diffusion and diff_loss is not None:
                    desc = (
                        f"Step {step + 1}/{run.steps} • loss={loss_val:.4f} "
                        f"(ce={float(ce_loss):.4f} diff={float(diff_loss):.4f})"
                    )
                else:
                    desc = f"Step {step + 1}/{run.steps} • loss={loss_val:.4f}"
                progress.update(task, advance=1, description=desc)

                current_step = step + 1
                if self.save_every > 0 and current_step % self.save_every == 0:
                    self.save_checkpoint(run.id, "global", current_step)

        if loss is not None:
            if self.dist_ctx is not None:
                loss = self.dist_ctx.all_reduce(loss.detach(), op="avg")
            logger.success(f"Global fine-tuning complete • final loss={float(loss):.6f}")

        self.save_checkpoint(run.id, "global", run.steps, is_final=True)

    def _has_diffusion_head(self) -> bool:
        """Check if the model has a diffusion head for hybrid training."""
        return (
            hasattr(self.student, "diffusion_head")
            and getattr(self.student, "diffusion_head", None) is not None
        )

    def _get_diffusion_loss_weight(self) -> float:
        """Get the diffusion loss weight from model config."""
        try:
            return float(self.student.config.diffusion_head.loss_weight)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return 0.10

    def _build_blockwise_config(self, train: TrainConfig) -> BlockwiseConfig:
        """Create BlockwiseConfig from training settings.

        Maps the user-facing TrainConfig options to the internal BlockwiseConfig
        used by the blockwise trainer.
        """
        amp_dtype = torch.float16
        if train.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

        return BlockwiseConfig(
            cache_teacher_outputs=train.cache_teacher_outputs,
            use_amp=train.use_amp,
            amp_dtype=amp_dtype,
            accumulation_steps=train.gradient_accumulation_steps,
        )

    def _log_optimization_settings(self, train: TrainConfig) -> None:
        """Display which optimizations are enabled for this run."""
        settings: dict[str, str] = {}
        if train.cache_teacher_outputs:
            settings["Teacher caching"] = "enabled"
        if train.use_amp:
            settings["Mixed precision"] = train.amp_dtype
        if train.gradient_accumulation_steps > 1:
            settings["Gradient accumulation"] = f"{train.gradient_accumulation_steps} steps"
        if train.num_workers > 0:
            settings["DataLoader workers"] = str(train.num_workers)
        if train.compile_model:
            settings["torch.compile"] = "enabled"

        if settings:
            logger.key_value(settings)

    def resolve_teacher_ckpt(self, ckpt: str) -> Path:
        """Resolve a checkpoint path or Hugging Face URI.

        Supports both local paths and hf:// URIs (e.g., hf://meta-llama/Llama-3.2-1B).
        """
        if ckpt.startswith("hf://"):
            return HFLoader(repo_id=ckpt[5:]).load()
        return Path(ckpt)

    def build_loader(self, train: TrainConfig) -> DataLoader[tuple[Tensor, Tensor]]:
        """Create a DataLoader with performance optimizations.

        Uses parallel data loading and memory pinning when configured,
        which can significantly speed up training.
        """
        path = Path(self.group.data)
        dataset = NpyDataset(str(path), block_size=train.block_size)

        use_pin_memory = train.pin_memory and self.device.type == "cuda"

        loader_kwargs: dict[str, object] = {
            "batch_size": train.batch_size,
            "shuffle": True,
            "drop_last": True,
            "num_workers": train.num_workers,
            "pin_memory": use_pin_memory,
        }

        if train.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        if self.dist_ctx is not None:
            return self.dist_ctx.wrap_dataloader(dataset, **loader_kwargs)  # type: ignore[arg-type]
        return DataLoader(dataset, **loader_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def next_batch(
        loader: DataLoader[tuple[Tensor, Tensor]],
        iterator: Iterator[tuple[Tensor, Tensor]],
    ) -> tuple[tuple[Tensor, Tensor], Iterator[tuple[Tensor, Tensor]]]:
        """Get the next batch, cycling the loader if exhausted.

        We train for more steps than the dataset has batches, so we
        restart the iterator when it runs out.
        """
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iter = iter(loader)
            return next(new_iter), new_iter

    def require_train(self, run: Run) -> TrainConfig:
        """Get the train config or raise if missing."""
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        return run.train

    @staticmethod
    def parse_device(device: str) -> torch.device:
        """Parse a device string like 'cuda', 'mps', or 'cpu'."""
        return torch.device(device)

    @staticmethod
    def parse_dtype(dtype: str) -> torch.dtype:
        """Parse a dtype string to a torch dtype."""
        match dtype:
            case "float32":
                return torch.float32
            case "float16":
                return torch.float16
            case "bfloat16":
                return torch.bfloat16
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")
