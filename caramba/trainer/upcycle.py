"""Model upcycling: converting pretrained models to new architectures.

Upcycling takes a pretrained model (like Llama) and trains it to use a new
architecture (like DBA attention) while preserving its learned knowledge.
We do this by distillation: the original model is the "teacher" and the
new architecture is the "student." The student learns to produce the same
outputs as the teacher, then we fine-tune on language modeling to recover
any lost performance.
"""
from __future__ import annotations

from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, cast

import re
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.optim import Optimizer

from caramba.config.defaults import Defaults
from caramba.config.eval import EvalVerifyConfig
from caramba.config.group import Group
from caramba.config.kvcache import KVCachePolicyConfig, KVCachePolicyDecoupledConfig
from caramba.config.layer import AttentionMode
from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.run import Run
from caramba.config.train import TrainConfig, TrainPhase
from caramba.config.verify import (
    CompareVerifyConfig,
    FidelityVerifyConfig,
    KVCacheVerifyConfig,
)
from caramba.console import logger
from caramba.data import build_token_dataset
from caramba.eval.suite import assert_eval_thresholds, run_eval_verify
from caramba.instrumentation import (
    LivePlotter,
    RunLogger,
    TensorBoardWriter,
    WandBWriter,
    generate_analysis_png,
)
from caramba.layer.attention import AttentionLayer
from caramba.loader.checkpoint import CheckpointLoader
from caramba.loader.hf import HFLoader
from caramba.loader.llama_upcycle import LlamaUpcycle
from caramba.model import Model
from caramba.runtime import RuntimePlan, load_plan, make_plan_key, save_plan
from caramba.trainer.blockwise import BlockwiseConfig, BlockwiseTrainer
from caramba.trainer.compare import assert_thresholds, compare_teacher_student
from caramba.trainer.fidelity import (
    assert_fidelity_thresholds,
    compute_short_context_fidelity,
)
from caramba.trainer.distill import DistillLoss
from caramba.trainer.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedStrategy,
)
from caramba.trainer.scheduler import LRSchedulerConfig, build_lr_scheduler


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
        self.run_logger = RunLogger(self.checkpoint_dir, filename="train.jsonl", enabled=True)
        self.tb_writer: TensorBoardWriter | None = None
        self.wandb_writer: WandBWriter | None = None
        self.live_plotter: LivePlotter | None = None
        self._resume_state: dict[str, object] | None = None

        # Set up distributed training if configured
        self.dist_config = dist_config
        self.dist_ctx: DistributedContext | None = None
        if dist_config is not None and dist_config.strategy != DistributedStrategy.NONE:
            self.dist_ctx = DistributedContext.init(dist_config)
            self.device = self.dist_ctx.device
        else:
            self.device = self.parse_device(train.device)

        self.device_name = str(self.device)
        self.runtime_plan = self._load_or_create_runtime_plan(train)
        self.dtype = self.parse_dtype(self.runtime_plan.dtype)
        self.teacher: nn.Module
        self.student: nn.Module
        self.init_models(train)

        if resume_from is not None:
            self._resume_state = self.load_checkpoint(Path(resume_from))

    def run(self, run: Run) -> None:
        """Execute a single training run (blockwise or global phase).

        After training completes, runs any configured verification steps.
        """
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")

        torch.manual_seed(run.seed)
        instrument = str(getattr(self.defaults, "instrument", "rich") if self.defaults else "rich")
        tokens = {t for t in re.split(r"[,+\s]+", instrument.lower()) if t}
        tb_enabled = ("tb" in tokens) or ("tensorboard" in tokens)
        live_enabled = ("live" in tokens) or ("plot" in tokens) or ("liveplot" in tokens)
        # Create a per-run TensorBoard writer directory.
        self.tb_writer = TensorBoardWriter(
            self.checkpoint_dir / "tb" / str(run.id),
            enabled=bool(tb_enabled),
            log_every=10,
        )
        # Create a per-run W&B writer (best-effort). Full behavior is refined by later todos.
        wandb_enabled = bool(
            self.defaults is not None
            and bool(getattr(self.defaults, "wandb", False))
            and str(getattr(self.defaults, "wandb_project", "") or "").strip() != ""
        )
        self.wandb_writer = WandBWriter(
            self.checkpoint_dir / "wandb",
            enabled=wandb_enabled,
            project=str(getattr(self.defaults, "wandb_project", "") if self.defaults else ""),
            entity=str(getattr(self.defaults, "wandb_entity", "") or "") or None,
            mode=str(getattr(self.defaults, "wandb_mode", "online") if self.defaults else "online"),
            run_name=f"{self.group.name}:{run.id}",
            group=self.group.name,
            tags=["caramba", "upcycle", str(run.train.phase.value)],
            config={
                "manifest": self.manifest.model_dump(),
                "defaults": self.defaults.model_dump() if self.defaults is not None else {},
                "group": self.group.model_dump(),
                "run": run.model_dump(),
                "train": run.train.model_dump() if run.train is not None else {},
            },
        )
        self.live_plotter = LivePlotter(
            enabled=bool(live_enabled),
            title=f"{self.group.name}:{run.id}",
            plot_every=10,
        )

        match run.train.phase:
            case TrainPhase.BLOCKWISE:
                self.run_blockwise(run)
            case TrainPhase.GLOBAL:
                self.run_global(run)
            case _:
                raise ValueError(f"Unsupported train phase: {run.train.phase}")

        self.verify(run)
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.wandb_writer is not None:
            self.wandb_writer.close()
            self.wandb_writer = None
        if self.live_plotter is not None:
            self.live_plotter.close()
            self.live_plotter = None
        # Optional post-run summary figure (best-effort).
        try:
            generate_analysis_png(
                self.checkpoint_dir / "train.jsonl",
                self.checkpoint_dir / f"{run.id}_analysis.png",
            )
        except Exception:
            pass
        # Best-effort structured event for downstream analysis.
        self.run_logger.log_event(
            type="run_complete",
            run_id=str(run.id),
            phase=str(run.train.phase.value),
            step=int(run.steps),
            data={},
        )

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
        elif isinstance(cfg, FidelityVerifyConfig):
            self.verify_fidelity(run, cfg)
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
            self.run_logger.log_metrics(
                run_id=str(run.id),
                phase="verify_compare",
                step=0,
                metrics=metrics,
            )
            if self.tb_writer is not None:
                self.tb_writer.log_scalars(
                    prefix="verify/compare",
                    step=0,
                    scalars={k: float(v) for k, v in metrics.items()},
                )
            if self.wandb_writer is not None:
                self.wandb_writer.log_scalars(
                    prefix="verify/compare",
                    step=0,
                    scalars={k: float(v) for k, v in metrics.items()},
                )

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
        self.run_logger.log_metrics(
            run_id=str(run.id),
            phase="verify_eval",
            step=0,
            metrics={
                "teacher_accuracy": float(summary.teacher_accuracy),
                "student_accuracy": float(summary.student_accuracy),
            },
        )
        if self.tb_writer is not None:
            self.tb_writer.log_scalars(
                prefix="verify/eval",
                step=0,
                scalars={
                    "teacher_accuracy": float(summary.teacher_accuracy),
                    "student_accuracy": float(summary.student_accuracy),
                },
            )
        if self.wandb_writer is not None:
            self.wandb_writer.log_scalars(
                prefix="verify/eval",
                step=0,
                scalars={
                    "teacher_accuracy": float(summary.teacher_accuracy),
                    "student_accuracy": float(summary.student_accuracy),
                },
            )
        logger.success("Evaluation complete")
        assert_eval_thresholds(summary=summary, thresholds=cfg.thresholds)

    def verify_fidelity(self, run: Run, cfg: FidelityVerifyConfig) -> None:
        """Check short-context delta NLL / PPL ratio between teacher and student.

        Why this exists:
        - Loss-based deltas are a strong, low-variance quality signal.
        - This is much cheaper than full benchmark suites, so we can gate early.
        """

        train = self.require_train(run)
        batch_size = int(cfg.batch_size) if cfg.batch_size is not None else int(train.batch_size)
        block_size = int(cfg.block_size) if cfg.block_size is not None else int(train.block_size)

        logger.info(
            f"Running short-context fidelity on {cfg.batches} batches "
            f"(split={cfg.split}, batch_size={batch_size}, block_size={block_size})..."
        )

        batches = self.collect_fidelity_batches(
            batch_size=batch_size,
            block_size=block_size,
            count=int(cfg.batches),
            split=str(cfg.split),
        )

        result = compute_short_context_fidelity(
            teacher=self.teacher,
            student=self.student,
            batches=batches,
        )

        metrics = {
            "teacher_nll": float(result.teacher_nll),
            "student_nll": float(result.student_nll),
            "delta_nll": float(result.delta_nll),
            "ppl_ratio": float(result.ppl_ratio),
            "tokens": float(result.tokens),
        }
        logger.key_value(
            {
                "teacher_nll": f"{result.teacher_nll:.6f}",
                "student_nll": f"{result.student_nll:.6f}",
                "delta_nll": f"{result.delta_nll:.6f}",
                "ppl_ratio": f"{result.ppl_ratio:.6f}",
                "tokens": str(result.tokens),
            }
        )
        self.run_logger.log_metrics(
            run_id=str(run.id),
            phase="verify_fidelity",
            step=0,
            metrics={k: float(v) for k, v in metrics.items()},
        )
        if self.tb_writer is not None:
            self.tb_writer.log_scalars(prefix="verify/fidelity", step=0, scalars=metrics)
        if self.wandb_writer is not None:
            self.wandb_writer.log_scalars(prefix="verify/fidelity", step=0, scalars=metrics)

        violations = assert_fidelity_thresholds(
            result=result,
            max_delta_nll=float(cfg.max_delta_nll) if cfg.max_delta_nll is not None else None,
            max_ppl_ratio=float(cfg.max_ppl_ratio) if cfg.max_ppl_ratio is not None else None,
            fail_fast=bool(cfg.fail_fast),
        )
        if violations:
            for v in violations:
                logger.warning(f"Threshold exceeded: {v.message()}")
            logger.warning(
                f"Verification found {len(violations)} threshold violation(s), "
                "but fail_fast=False so continuing to benchmarks..."
            )

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
        self.run_logger.log_metrics(
            run_id=str(run.id),
            phase="verify_kvcache",
            step=0,
            metrics={
                "teacher_kvcache_mb": float(teacher_total / 1024 / 1024),
                "student_kvcache_mb": float(student_total / 1024 / 1024),
                "reduction": float(reduction),
            },
        )
        if self.tb_writer is not None:
            self.tb_writer.log_scalars(
                prefix="verify/kvcache",
                step=0,
                scalars={
                    "teacher_kvcache_mb": float(teacher_total / 1024 / 1024),
                    "student_kvcache_mb": float(student_total / 1024 / 1024),
                    "reduction": float(reduction),
                },
            )
        if self.wandb_writer is not None:
            self.wandb_writer.log_scalars(
                prefix="verify/kvcache",
                step=0,
                scalars={
                    "teacher_kvcache_mb": float(teacher_total / 1024 / 1024),
                    "student_kvcache_mb": float(student_total / 1024 / 1024),
                    "reduction": float(reduction),
                },
            )
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
        dataset = build_token_dataset(path=path, block_size=int(block_size))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        batches: list[Tensor] = []
        for x, _ in loader:
            batches.append(x.to(device=self.device))
            if len(batches) >= count:
                break
        return batches

    def collect_fidelity_batches(
        self, *, batch_size: int, block_size: int, count: int, split: str
    ) -> list[tuple[Tensor, Tensor]]:
        """Load (x, y) batches for fidelity checks.

        Why this exists:
        - Fidelity checks need targets (y) to compute NLL.
        - We keep it deterministic by disabling shuffle.
        """

        path = Path(self.group.data)
        dataset = build_token_dataset(path=path, block_size=int(block_size))

        # Mirror the manifest-driven val split used by build_loaders().
        val_frac = float(getattr(self.defaults, "val_frac", 0.0)) if self.defaults else 0.0
        n = len(cast(Sized, dataset))
        n_val = int(n * val_frac) if val_frac > 0 else 0
        if val_frac > 0 and n_val <= 0 and n > 1:
            n_val = 1
        n_train = max(1, n - n_val) if n > 0 else 0

        use_val = False
        if split in ("val", "auto") and n_val > 0 and n_train > 0:
            use_val = True
        if split == "val" and not use_val:
            logger.warning("Requested split=val, but no val split is configured; falling back to train.")

        if use_val:
            ds = Subset(dataset, range(n_train, n_train + n_val))
        elif n_val > 0 and n_train > 0:
            ds = Subset(dataset, range(0, n_train))
        else:
            ds = dataset

        loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=True)
        batches: list[tuple[Tensor, Tensor]] = []
        for x, y in loader:
            batches.append((x.to(device=self.device), y.to(device=self.device)))
            if len(batches) >= int(count):
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

        # Apply torch.compile for speed (PyTorch 2.0+). This is intentionally conservative
        # because compile can have significant overhead on small models or non-CUDA devices.
        if bool(self.runtime_plan.compile) and hasattr(torch, "compile"):
            if self.device.type == "cuda":
                logger.info(
                    f"Compiling student with torch.compile (mode={self.runtime_plan.compile_mode})..."
                )
                try:
                    self.student = torch.compile(
                        self.student, mode=self.runtime_plan.compile_mode
                    )  # type: ignore[assignment]
                    logger.success("torch.compile applied")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without: {e}")
            else:
                logger.warning(f"torch.compile not supported on {self.device.type}, skipping")

        self.teacher.eval()
        logger.success("Initialization complete")

        # Optional activation checkpointing (core model path, not just FSDP).
        if bool(getattr(train, "activation_checkpointing", False)):
            threshold = float(getattr(train, "activation_checkpoint_threshold_mb", 0.0))
            for m in self.student.modules():
                if hasattr(m, "activation_checkpointing"):
                    try:
                        setattr(m, "activation_checkpointing", True)
                        setattr(m, "activation_checkpoint_threshold_mb", threshold)
                    except Exception:
                        pass

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
        block_step: int | None = None,
        global_step: int | None = None,
        optimizer: Optimizer | None = None,
        scheduler: object | None = None,
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
            "block_step": block_step,
            "global_step": global_step if global_step is not None else step,
        }
        if optimizer is not None:
            try:
                state["optimizer_state_dict"] = optimizer.state_dict()
            except Exception:
                pass
        if scheduler is not None:
            try:
                state["scheduler_state_dict"] = getattr(scheduler, "state_dict")()
            except Exception:
                pass

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
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint must be a dict, got {type(state).__name__}")
        self._validate_checkpoint_state(state)

        self.student.load_state_dict(state["student_state_dict"])
        logger.success(
            f"Resumed from checkpoint: run={state.get('run_id')}, "
            f"phase={state.get('phase')}, step={state.get('step')}"
        )
        # Return raw state so callers can restore optimizer/scheduler if present.
        return state

    @staticmethod
    def _validate_checkpoint_state(state: dict[str, object]) -> None:
        """Validate a checkpoint payload before applying it."""

        required = ["student_state_dict", "run_id", "phase", "step"]
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")
        if not isinstance(state.get("student_state_dict"), dict):
            raise TypeError("Checkpoint student_state_dict must be a dict")
        # Basic metadata sanity.
        _ = str(state.get("run_id"))
        _ = str(state.get("phase"))
        try:
            int(state.get("step"))  # type: ignore[arg-type]
        except Exception as e:
            raise TypeError("Checkpoint step must be int-like") from e

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
        loader, _val_loader = self.build_loaders(train)
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
        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(n_blocks * run.steps),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )
        self.student.train()
        loader_iter = iter(loader)

        # Optional resume (blockwise fixed-step mode only).
        resume_block_index = 0
        resume_block_step = 0
        resume_global_step = 0
        if self._resume_state is not None and str(self._resume_state.get("phase", "")) == "blockwise":
            if str(self._resume_state.get("run_id", "")) == str(run.id):
                resume_block_index = self._int_or(self._resume_state.get("block_index"), 0)
                resume_block_step = self._int_or(self._resume_state.get("block_step"), 0)
                resume_global_step = self._int_or(self._resume_state.get("global_step"), 0)
                try:
                    if "optimizer_state_dict" in self._resume_state:
                        optimizer.load_state_dict(self._resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
                    if scheduler is not None and "scheduler_state_dict" in self._resume_state:
                        scheduler.load_state_dict(self._resume_state["scheduler_state_dict"])  # type: ignore[arg-type]
                except Exception:
                    pass
        self.run_logger.log_event(
            type="phase_start",
            run_id=str(run.id),
            phase="blockwise",
            step=0,
            data={
                "n_blocks": int(n_blocks),
                "steps_per_block": int(run.steps),
                "convergence_target": (
                    None
                    if train.convergence_target is None
                    else float(train.convergence_target)
                ),
                "convergence_patience": int(train.convergence_patience),
                "convergence_max_steps": int(train.convergence_max_steps),
            },
        )

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
                lr_scheduler=scheduler,
                start_block_index=resume_block_index,
                start_block_step=resume_block_step,
                start_global_step=resume_global_step,
            )

        self.save_checkpoint(
            run.id,
            "blockwise",
            step=0,
            optimizer=optimizer,
            scheduler=scheduler,
            is_final=True,
        )

    def _run_blockwise_fixed(
        self,
        *,
        run_id: str,
        trainer: BlockwiseTrainer,
        loader: DataLoader[tuple[Tensor, Tensor]],
        loader_iter: Iterator[tuple[Tensor, Tensor]],
        n_blocks: int,
        steps_per_block: int,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None,
        start_block_index: int = 0,
        start_block_step: int = 0,
        start_global_step: int = 0,
    ) -> None:
        """Train each block for a fixed number of steps.

        Simple and predictable, but may over-train easy blocks and under-train
        difficult ones.
        """
        total_steps = n_blocks * steps_per_block
        global_step = int(start_global_step)

        with logger.progress_bar() as progress:
            overall_task = progress.add_task(
                f"Training {n_blocks} blocks...", total=total_steps
            )

            for block_index in range(int(start_block_index), n_blocks):
                loss = None
                step0 = int(start_block_step) if block_index == int(start_block_index) else 0
                for step in range(step0, steps_per_block):
                    (x, _), loader_iter = self.next_batch(loader, loader_iter)
                    x = x.to(device=self.device)
                    loss = trainer.step(x, block_index=block_index)
                    global_step += 1

                    loss_val = float(loss) if loss is not None else 0.0
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    lr = float(trainer.optimizer.param_groups[0].get("lr", 0.0))
                    self.run_logger.log_metrics(
                        run_id=str(run_id),
                        phase="blockwise",
                        step=int(global_step),
                        metrics={
                            "loss": float(loss_val),
                            "lr": float(lr),
                            "block_index": int(block_index),
                            "block_step": int(step + 1),
                            "steps_per_block": int(steps_per_block),
                            "n_blocks": int(n_blocks),
                        },
                    )
                    if self.tb_writer is not None:
                        self.tb_writer.log_scalars(
                            prefix="train/blockwise",
                            step=int(global_step),
                            scalars={
                                "loss": float(loss_val),
                                "lr": float(lr),
                                "block_index": float(block_index),
                            },
                        )
                    if self.wandb_writer is not None and (int(global_step) % 10 == 0):
                        self.wandb_writer.log_scalars(
                            prefix="train/blockwise",
                            step=int(global_step),
                            scalars={
                                "loss": float(loss_val),
                                "lr": float(lr),
                                "block_index": float(block_index),
                            },
                        )
                    if self.live_plotter is not None:
                        self.live_plotter.update(
                            step=int(global_step),
                            scalars={
                                "loss": float(loss_val),
                                "block_index": float(block_index),
                            },
                        )
                    self._maybe_log_histograms(step=int(global_step), prefix="blockwise")
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
                            run_id,
                            "blockwise",
                            global_step,
                            block_index=block_index,
                            block_step=int(step + 1),
                            global_step=int(global_step),
                            optimizer=trainer.optimizer,
                            scheduler=lr_scheduler,
                        )

                if loss is not None:
                    if self.dist_ctx is not None:
                        loss = self.dist_ctx.all_reduce(loss, op="avg")
                    logger.success(
                        f"Block {block_index + 1}/{n_blocks} complete • "
                        f"loss={float(loss):.6f}"
                    )
                    self.run_logger.log_event(
                        type="block_complete",
                        run_id=str(run_id),
                        phase="blockwise",
                        step=int(global_step),
                        data={"block_index": int(block_index), "loss": float(loss)},
                    )

                self.save_checkpoint(
                    run_id,
                    "blockwise",
                    global_step,
                    block_index=block_index,
                    block_step=int(steps_per_block),
                    global_step=int(global_step),
                    optimizer=trainer.optimizer,
                    scheduler=lr_scheduler,
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
                lr = float(trainer.optimizer.param_groups[0].get("lr", 0.0))

                if loss_val < best_loss:
                    best_loss = loss_val
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                self.run_logger.log_metrics(
                    run_id=str(run_id),
                    phase="blockwise",
                    step=int(total_steps_taken),
                    metrics={
                        "loss": float(loss_val),
                        "best_loss": float(best_loss),
                        "lr": float(lr),
                        "block_index": int(block_index),
                        "block_step": int(step),
                        "target_loss": float(target_loss),
                        "patience_left": int(patience - steps_without_improvement),
                    },
                )
                if self.tb_writer is not None:
                    self.tb_writer.log_scalars(
                        prefix="train/blockwise",
                        step=int(total_steps_taken),
                        scalars={
                            "loss": float(loss_val),
                            "best_loss": float(best_loss),
                            "target_loss": float(target_loss),
                            "lr": float(lr),
                            "block_index": float(block_index),
                        },
                    )
                if self.wandb_writer is not None and (int(total_steps_taken) % 10 == 0):
                    self.wandb_writer.log_scalars(
                        prefix="train/blockwise",
                        step=int(total_steps_taken),
                        scalars={
                            "loss": float(loss_val),
                            "best_loss": float(best_loss),
                            "target_loss": float(target_loss),
                            "lr": float(lr),
                            "block_index": float(block_index),
                        },
                    )
                if self.live_plotter is not None:
                    self.live_plotter.update(
                        step=int(total_steps_taken),
                        scalars={
                            "loss": float(loss_val),
                            "best_loss": float(best_loss),
                            "target_loss": float(target_loss),
                            "lr": float(lr),
                            "block_index": float(block_index),
                        },
                    )

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
        self.run_logger.log_event(
            type="phase_complete",
            run_id=str(run_id),
            phase="blockwise",
            step=int(total_steps_taken),
            data={"total_steps": int(total_steps_taken)},
        )

    def run_global(self, run: Run) -> None:
        """Run global fine-tuning on language modeling loss.

        This is the second training phase. After blockwise distillation,
        we fine-tune the entire model end-to-end on next-token prediction.
        This recovers any performance lost during architecture conversion.
        """
        train = self.require_train(run)
        loader, val_loader = self.build_loaders(train)

        # Unfreeze all params (blockwise leaves most frozen)
        self._unfreeze_all_parameters()

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=train.lr)
        self.student.train()

        has_diffusion = self._has_diffusion_head()
        use_amp = bool(train.use_amp) and self.device.type in ("cuda", "mps", "cpu")
        amp_dtype = self._resolve_amp_dtype(str(train.amp_dtype))
        scaler = None
        if use_amp and self.device.type == "cuda" and amp_dtype == torch.float16:
            # fp16 on CUDA benefits from GradScaler to avoid underflow.
            try:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                scaler = None

        logger.header("Global Fine-tuning", f"{run.steps} steps")
        self.run_logger.log_event(
            type="phase_start",
            run_id=str(run.id),
            phase="global",
            step=0,
            data={"steps": int(run.steps), "has_diffusion": bool(has_diffusion)},
        )

        loader_iter = iter(loader)
        loss: Tensor | None = None
        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=str(getattr(train, "scheduler", "none")),
                total_steps=int(run.steps),
                warmup_steps=int(getattr(train, "warmup_steps", 0)),
                min_lr_ratio=float(getattr(train, "min_lr_ratio", 0.0)),
            ),
        )

        # Optional resume (global phase).
        start_step = 0
        if self._resume_state is not None and str(self._resume_state.get("phase", "")) == "global":
            if str(self._resume_state.get("run_id", "")) == str(run.id):
                start_step = self._int_or(self._resume_state.get("step"), 0)
                try:
                    if "optimizer_state_dict" in self._resume_state:
                        optimizer.load_state_dict(self._resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
                    if scheduler is not None and "scheduler_state_dict" in self._resume_state:
                        scheduler.load_state_dict(self._resume_state["scheduler_state_dict"])  # type: ignore[arg-type]
                except Exception:
                    pass

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=run.steps)

            for step in range(int(start_step), run.steps):
                (x, y), loader_iter = self.next_batch(loader, loader_iter)
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                ce_loss: Tensor
                diff_loss: Tensor | None = None

                optimizer.zero_grad(set_to_none=True)
                autocast_enabled = bool(use_amp)
                try:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=amp_dtype,
                        enabled=autocast_enabled,
                    ):
                        loss, ce_loss, diff_loss = self._compute_loss(x, y, has_diffusion)
                except TypeError:
                    # Older torch versions: fallback to no autocast.
                    autocast_enabled = False
                    loss, ce_loss, diff_loss = self._compute_loss(x, y, has_diffusion)

                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_val = float(loss) if loss is not None else 0.0
                lr = float(optimizer.param_groups[0].get("lr", train.lr))
                metrics: dict[str, float] = {
                    "loss": float(loss_val),
                    "ce_loss": float(ce_loss),
                    "lr": float(lr),
                }
                if diff_loss is not None:
                    metrics["diff_loss"] = float(diff_loss)
                self.run_logger.log_metrics(
                    run_id=str(run.id),
                    phase="global",
                    step=int(step + 1),
                    metrics=metrics,
                )
                if self.tb_writer is not None:
                    self.tb_writer.log_scalars(
                        prefix="train/global",
                        step=int(step + 1),
                        scalars=metrics,
                    )
                if self.wandb_writer is not None and (int(step + 1) % 10 == 0):
                    self.wandb_writer.log_scalars(
                        prefix="train/global",
                        step=int(step + 1),
                        scalars=metrics,
                    )
                if self.live_plotter is not None:
                    self.live_plotter.update(step=int(step + 1), scalars=metrics)
                self._maybe_log_histograms(step=int(step + 1), prefix="global")

                # Periodic evaluation loss on held-out validation split (if configured).
                if (
                    val_loader is not None
                    and self.defaults is not None
                    and int(getattr(self.defaults, "eval_iters", 0)) > 0
                    and ((step + 1) % int(getattr(self.defaults, "eval_iters", 0)) == 0)
                ):
                    val_metrics = self._eval_global_loss(val_loader, max_batches=2)
                    self.run_logger.log_metrics(
                        run_id=str(run.id),
                        phase="eval_global",
                        step=int(step + 1),
                        metrics=val_metrics,
                    )
                    if self.tb_writer is not None:
                        self.tb_writer.log_scalars(
                            prefix="eval/global",
                            step=int(step + 1),
                            scalars=val_metrics,
                        )
                    if self.wandb_writer is not None:
                        self.wandb_writer.log_scalars(
                            prefix="eval/global",
                            step=int(step + 1),
                            scalars=val_metrics,
                        )
                    if self.live_plotter is not None:
                        self.live_plotter.update(step=int(step + 1), scalars=val_metrics)
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
                    self.save_checkpoint(
                        run.id,
                        "global",
                        current_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=int(current_step),
                    )

        if loss is not None:
            if self.dist_ctx is not None:
                loss = self.dist_ctx.all_reduce(loss.detach(), op="avg")
            logger.success(f"Global fine-tuning complete • final loss={float(loss):.6f}")
            self.run_logger.log_event(
                type="phase_complete",
                run_id=str(run.id),
                phase="global",
                step=int(run.steps),
                data={"final_loss": float(loss)},
            )

        self.save_checkpoint(
            run.id,
            "global",
            run.steps,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=int(run.steps),
            is_final=True,
        )

    def _has_diffusion_head(self) -> bool:
        """Check if the model has a diffusion head for hybrid training."""
        return (
            hasattr(self.student, "diffusion_head")
            and getattr(self.student, "diffusion_head", None) is not None
        )

    def _maybe_log_histograms(self, *, step: int, prefix: str) -> None:
        """Optionally log parameter/gradient histograms to TensorBoard.

        This is intentionally conservative: we only log a small number of tensors
        and only when the TensorBoard writer decides it is time to log.
        """

        if self.tb_writer is None:
            return
        if not self.tb_writer.should_log(int(step)):
            return

        max_params = 4
        logged = 0
        for name, p in self.student.named_parameters():
            if logged >= max_params:
                break
            if p is None:
                continue
            try:
                self.tb_writer.log_histogram(
                    name=f"histo/{prefix}/param/{name}",
                    step=int(step),
                    values=p.detach().float().cpu(),
                )
                if p.grad is not None:
                    self.tb_writer.log_histogram(
                        name=f"histo/{prefix}/grad/{name}",
                        step=int(step),
                        values=p.grad.detach().float().cpu(),
                    )
                logged += 1
            except Exception:
                # Histogram logging must never break training.
                return

    def _eval_global_loss(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]],
        *,
        max_batches: int = 2,
    ) -> dict[str, float]:
        """Compute a small validation loss estimate for the student model."""

        was_training = self.student.training
        self.student.eval()

        has_diffusion = self._has_diffusion_head()
        diff_weight = self._get_diffusion_loss_weight() if has_diffusion else 0.0

        total_loss = 0.0
        total_ce = 0.0
        total_diff = 0.0
        n = 0

        with torch.no_grad():
            for (x, y) in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                if has_diffusion:
                    result = self.student.forward(x, return_features=True)  # type: ignore[call-arg]
                    features: Tensor = result[0]  # type: ignore[index]
                    logits: Tensor = result[1]  # type: ignore[index]
                    ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
                    diff = self.student.diffusion_loss(features, y)  # type: ignore[attr-defined]
                    loss = ce + float(diff_weight) * diff
                    total_diff += float(diff)
                else:
                    logits = self.student.forward(x)
                    ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
                    loss = ce

                total_loss += float(loss)
                total_ce += float(ce)
                n += 1
                if n >= int(max_batches):
                    break

        # Distributed: average across processes.
        if self.dist_ctx is not None and n > 0:
            t = torch.tensor([total_loss, total_ce, total_diff, float(n)], device=self.device)
            t = self.dist_ctx.all_reduce(t, op="sum")
            total_loss = float(t[0].item())
            total_ce = float(t[1].item())
            total_diff = float(t[2].item())
            n = int(t[3].item())

        if was_training:
            self.student.train()

        denom = float(n) if n > 0 else 1.0
        metrics: dict[str, float] = {
            "val_loss": total_loss / denom,
            "val_ce_loss": total_ce / denom,
        }
        if has_diffusion:
            metrics["val_diff_loss"] = total_diff / denom
        return metrics

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype:
        """Resolve AMP dtype, including an 'auto' mode driven by device."""

        s = str(amp_dtype).lower()
        if s == "auto":
            if self.device.type == "cuda":
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        return torch.bfloat16
                except Exception:
                    pass
                return torch.float16
            if self.device.type == "mps":
                return torch.float16
            # CPU autocast works best with bf16.
            return torch.bfloat16

        if s == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def _get_diffusion_loss_weight(self) -> float:
        """Get the diffusion loss weight from model config."""
        try:
            return float(self.student.config.diffusion_head.loss_weight)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return 0.10

    def _compute_loss(
        self, x: Tensor, y: Tensor, has_diffusion: bool
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute forward pass and loss.

        Returns:
            (total_loss, ce_loss, diff_loss) where diff_loss is None if not using diffusion.
        """
        if has_diffusion:
            result = self.student.forward(x, return_features=True)  # type: ignore[call-arg]
            features: Tensor = result[0]  # type: ignore[index]
            logits: Tensor = result[1]  # type: ignore[index]
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), y.reshape(-1)
            )
            diff_loss_val: Tensor = self.student.diffusion_loss(features, y)  # type: ignore[attr-defined]
            diff_weight = self._get_diffusion_loss_weight()
            loss = ce_loss + diff_weight * diff_loss_val
            return loss, ce_loss, diff_loss_val
        else:
            logits = self.student.forward(x)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), y.reshape(-1)
            )
            return ce_loss, ce_loss, None

    def _build_blockwise_config(self, train: TrainConfig) -> BlockwiseConfig:
        """Create BlockwiseConfig from training settings.

        Maps the user-facing TrainConfig options to the internal BlockwiseConfig
        used by the blockwise trainer.
        """
        amp_str = str(train.amp_dtype).lower()
        if amp_str == "auto":
            if self.device.type == "cuda":
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        amp_dtype = torch.bfloat16
                    else:
                        amp_dtype = torch.float16
                except Exception:
                    amp_dtype = torch.float16
            elif self.device.type == "mps":
                amp_dtype = torch.float16
            else:
                amp_dtype = torch.bfloat16
        elif amp_str == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

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
        compile_setting = getattr(train, "compile_model", False)
        if bool(compile_setting) and str(compile_setting).lower() not in ("false", "0", "off", "disabled"):
            settings["torch.compile"] = str(compile_setting)
        if bool(getattr(train, "auto_batch_size", False)):
            settings["Auto batch size"] = (
                f"ref={int(getattr(train, 'auto_batch_ref_block_size', 512))} "
                f"min={int(getattr(train, 'auto_batch_min', 1))}"
            )

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
        dataset = build_token_dataset(path=path, block_size=int(train.block_size))

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

    def build_loaders(
        self, train: TrainConfig
    ) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]] | None]:
        """Create train/val DataLoaders from the same underlying token stream.

        Why this exists:
        - Many quality gates (and basic sanity checks) need a held-out validation loss.
        - We want the split behavior to be entirely driven by the manifest defaults.
        """

        path = Path(self.group.data)
        dataset = build_token_dataset(path=path, block_size=int(train.block_size))

        val_frac = float(getattr(self.defaults, "val_frac", 0.0)) if self.defaults else 0.0
        n = len(cast(Sized, dataset))
        n_val = int(n * val_frac) if val_frac > 0 else 0
        # Keep at least 1 validation example if val_frac > 0.
        if val_frac > 0 and n_val <= 0 and n > 1:
            n_val = 1
        n_train = max(1, n - n_val) if n > 0 else 0

        if n_val > 0 and n_train > 0:
            train_ds = Subset(dataset, range(0, n_train))
            val_ds = Subset(dataset, range(n_train, n_train + n_val))
        else:
            train_ds = dataset
            val_ds = None

        # Use persisted runtime plan decisions.
        batch_size = int(self.runtime_plan.batch_size)

        use_pin_memory = train.pin_memory and self.device.type == "cuda"
        loader_kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "drop_last": True,
            "num_workers": train.num_workers,
            "pin_memory": use_pin_memory,
        }
        if train.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        if self.dist_ctx is not None:
            train_loader = self.dist_ctx.wrap_dataloader(
                train_ds, shuffle=True, **loader_kwargs  # type: ignore[arg-type]
            )
            val_loader = (
                self.dist_ctx.wrap_dataloader(
                    val_ds, shuffle=False, **loader_kwargs  # type: ignore[arg-type]
                )
                if val_ds is not None
                else None
            )
            return train_loader, val_loader

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)  # type: ignore[arg-type]
        val_loader = (
            DataLoader(val_ds, shuffle=False, **loader_kwargs)  # type: ignore[arg-type]
            if val_ds is not None
            else None
        )
        return train_loader, val_loader

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

    def parse_dtype(self, dtype: str) -> torch.dtype:
        """Parse a dtype string to a torch dtype.

        Supports an "auto" mode that picks a sane default based on the device.
        """
        dt = str(dtype).lower()
        if dt == "auto":
            if self.device.type == "cuda":
                # Prefer bf16 on supported CUDA devices.
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        return torch.bfloat16
                except Exception:
                    pass
                return torch.float16
            if self.device.type == "mps":
                return torch.float16
            return torch.float32

        match dt:
            case "float32":
                return torch.float32
            case "float16":
                return torch.float16
            case "bfloat16":
                return torch.bfloat16
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _load_or_create_runtime_plan(self, train: TrainConfig) -> RuntimePlan:
        """Derive a runtime plan and persist it for reuse."""

        # Build a stable payload that excludes volatile fields like checkpoint paths.
        train_payload = train.model_dump()
        train_payload.pop("teacher_ckpt", None)
        payload: dict[str, Any] = {
            "device": str(self.device),
            "torch": str(getattr(torch, "__version__", "")),
            "model": self.manifest.model.model_dump(),
            "train": train_payload,
        }
        key = make_plan_key(payload)
        plan_path = self.checkpoint_dir / "plans" / f"{key}.json"
        existing = load_plan(plan_path)
        if existing is not None and existing.key == key:
            return existing

        # Resolve decisions.
        dtype_str = str(train.dtype).lower()
        if dtype_str == "auto":
            if self.device.type == "cuda":
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        dtype_str = "bfloat16"
                    else:
                        dtype_str = "float16"
                except Exception:
                    dtype_str = "float16"
            elif self.device.type == "mps":
                dtype_str = "float16"
            else:
                dtype_str = "float32"

        amp_dtype_str = str(train.amp_dtype).lower()
        if amp_dtype_str == "auto":
            if self.device.type == "cuda":
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        amp_dtype_str = "bfloat16"
                    else:
                        amp_dtype_str = "float16"
                except Exception:
                    amp_dtype_str = "float16"
            elif self.device.type == "mps":
                amp_dtype_str = "float16"
            else:
                amp_dtype_str = "bfloat16"

        # Batch size tuning decision.
        batch_size = int(train.batch_size)
        if bool(getattr(train, "auto_batch_size", False)):
            ref = int(getattr(train, "auto_batch_ref_block_size", 512))
            min_bs = int(getattr(train, "auto_batch_min", 1))
            if ref > 0:
                batch_size = max(min_bs, int(batch_size * (ref / float(train.block_size))))

        # Compile decision.
        compile_setting: object = getattr(train, "compile_model", False)
        compile_mode = str(getattr(train, "compile_mode", "reduce-overhead"))
        should_compile = False
        if isinstance(compile_setting, bool):
            should_compile = compile_setting
        else:
            s = str(compile_setting).strip().lower()
            if s == "auto":
                should_compile = self.device.type == "cuda"
            elif s in ("1", "true", "yes", "on"):
                should_compile = True
            else:
                should_compile = False

        plan = RuntimePlan(
            key=key,
            device=str(self.device),
            torch_version=str(getattr(torch, "__version__", "")),
            dtype=dtype_str,
            use_amp=bool(train.use_amp),
            amp_dtype=amp_dtype_str,
            batch_size=int(batch_size),
            compile=bool(should_compile),
            compile_mode=str(compile_mode),
        )
        try:
            save_plan(plan_path, plan, payload=payload)
        except Exception:
            pass
        return plan

    @staticmethod
    def _int_or(value: object, default: int = 0) -> int:
        """Best-effort int conversion for checkpoint metadata."""
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return int(default)
