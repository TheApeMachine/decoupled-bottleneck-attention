"""
upcycle provides the upcycle training loop.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from collections.abc import Iterator

from torch import Tensor, nn
from torch.utils.data import DataLoader

from caramba.config.group import Group
from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.run import Run
from caramba.config.topology import NodeConfig, TopologyConfig
from caramba.config.train import TrainConfig, TrainPhase
from caramba.config.verify import CompareVerifyConfig, KVCacheVerifyConfig
from caramba.config.kvcache import KVCachePolicyConfig, KVCachePolicyDecoupledConfig
from caramba.config.eval import EvalVerifyConfig
from caramba.data.npy import NpyDataset
from caramba.layer.attention import AttentionLayer
from caramba.loader.checkpoint import CheckpointLoader
from caramba.loader.hf import HFLoader
from caramba.loader.llama_upcycle import LlamaUpcycle
from caramba.model import Model
from caramba.trainer.blockwise import BlockwiseTrainer
from caramba.trainer.compare import assert_thresholds, compare_teacher_student
from caramba.trainer.distill import DistillLoss
from caramba.trainer.distributed import (
    DistributedContext,
    DistributedConfig,
    DistributedStrategy,
)
from caramba.eval.suite import assert_eval_thresholds, run_eval_verify
from caramba.config.layer import AttentionLayerConfig, AttentionMode


def _make_teacher_model_config(model_config: "ModelConfig") -> "ModelConfig":
    """
    Create a teacher model config by rewriting all attention layers to standard mode.

    The teacher uses original Llama-style attention, while the student uses DBA.
    This enables meaningful distillation where the teacher provides "correct" targets.
    """
    from typing import Any
    from caramba.config.model import ModelConfig

    def rewrite_node_dict(node_dict: dict[str, Any]) -> dict[str, Any]:
        """Recursively rewrite node dicts, converting attention to standard."""
        # Check if this is an attention layer config
        if node_dict.get("type") == "AttentionLayer":
            # Convert to standard attention
            node_dict = node_dict.copy()
            node_dict["mode"] = "standard"
            # Remove DBA-specific fields (they have defaults so can be omitted)
            node_dict.pop("sem_dim", None)
            node_dict.pop("geo_dim", None)
            node_dict.pop("decoupled_gate", None)
            node_dict.pop("decoupled_gate_dynamic", None)
            return node_dict

        # If it has layers, recurse into them
        if "layers" in node_dict and isinstance(node_dict["layers"], list):
            node_dict = node_dict.copy()
            node_dict["layers"] = [
                rewrite_node_dict(layer) if isinstance(layer, dict) else layer
                for layer in node_dict["layers"]
            ]

        return node_dict

    # Dump the entire config to a dict, rewrite, and re-validate
    teacher_data = model_config.model_dump()
    teacher_data["topology"] = rewrite_node_dict(teacher_data["topology"])
    return ModelConfig.model_validate(teacher_data)


class Upcycle:
    """Runs blockwise distillation and global fine-tuning.

    Supports distributed training via DDP or FSDP for scaling to larger models
    and multi-GPU setups. Use the `distributed` parameter in TrainConfig to enable.
    """

    def __init__(
        self,
        manifest: Manifest,
        group: Group,
        train: TrainConfig,
        *,
        dist_config: DistributedConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.group = group

        # Initialize distributed context if configured
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

    def run(self, run: Run) -> None:
        """Execute a single run phase."""
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
        """Run post-run verification steps."""
        cfg = run.verify
        if cfg is None:
            return
        if isinstance(cfg, CompareVerifyConfig):
            self.verify_compare(run, cfg)
        elif isinstance(cfg, EvalVerifyConfig):
            self.verify_eval(run, cfg)
        elif isinstance(cfg, KVCacheVerifyConfig):
            self.verify_kvcache(run, cfg)

    def verify_compare(self, run: Run, cfg: CompareVerifyConfig) -> None:
        """Compare teacher/student on a few batches."""
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
        print(f"verify: compare batches={result.batches}")
        assert_thresholds(result=result, attention=cfg.attention, logits=cfg.logits)

    def verify_eval(self, run: Run, cfg: EvalVerifyConfig) -> None:
        """Run a small behavioral evaluation suite."""
        summary = run_eval_verify(
            teacher=self.teacher,
            student=self.student,
            cfg=cfg,
            device=self.device,
        )
        print(f"verify: eval teacher_acc={summary.teacher_accuracy:.3f} student_acc={summary.student_accuracy:.3f}")
        assert_eval_thresholds(summary=summary, thresholds=cfg.thresholds)

    def verify_kvcache(self, run: Run, cfg: KVCacheVerifyConfig) -> None:
        """Estimate KV-cache memory for teacher vs student.

        Compares the memory footprint of KV caches between models,
        which is especially relevant for DBA upcycling where we expect
        significant memory reduction due to smaller sem/geo key dimensions.
        """
        # Extract dimensions from the actual models
        teacher_bytes_per_token = self._estimate_model_kvcache_bytes(
            self.teacher, cfg.teacher, cfg.n_layers
        )
        student_bytes_per_token = self._estimate_model_kvcache_bytes_decoupled(
            self.student, cfg.student, cfg.n_layers
        )

        # Total memory for full sequence
        teacher_total = teacher_bytes_per_token * cfg.batch_size * cfg.max_seq_len
        student_total = student_bytes_per_token * cfg.batch_size * cfg.max_seq_len

        reduction = teacher_total / student_total if student_total > 0 else float("inf")

        self._print(
            f"verify: kvcache teacher={teacher_total / 1024 / 1024:.2f}MB, "
            f"student={student_total / 1024 / 1024:.2f}MB, reduction={reduction:.2f}x"
        )

        # Check threshold if configured
        if cfg.min_reduction_ratio is not None and reduction < cfg.min_reduction_ratio:
            raise AssertionError(
                f"KV cache reduction {reduction:.2f}x is below minimum {cfg.min_reduction_ratio}x"
            )

    def _kind_to_bytes(self, kind: str) -> float:
        """Convert cache kind to approximate bytes per element."""
        kind_lower = kind.lower() if isinstance(kind, str) else str(kind.value).lower()
        if kind_lower == "fp32":
            return 4.0
        elif kind_lower == "fp16":
            return 2.0
        elif kind_lower == "q8_0":
            return 1.0  # int8 + scale overhead ~1 byte effective
        else:  # q4_0, nf4
            return 0.5 + 0.125  # 4-bit + scale overhead

    def _estimate_model_kvcache_bytes(
        self,
        model: nn.Module,
        policy: "KVCachePolicyConfig",
        n_layers: int,
    ) -> int:
        """Estimate bytes per token for a standard model's KV cache."""
        from caramba.config.kvcache import KVCachePolicyConfig

        # Find attention layer dimensions from model
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
        policy: "KVCachePolicyDecoupledConfig",
        n_layers: int,
    ) -> int:
        """Estimate bytes per token for a decoupled model's KV cache."""
        from caramba.config.kvcache import KVCachePolicyDecoupledConfig

        # Find attention layer dimensions from model
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
                    # Fallback for non-decoupled
                    sem_dim = cfg.kv_heads * cfg.head_dim
                    geo_dim = 0
                    v_dim = sem_dim
                break

        k_sem_bytes = sem_dim * self._kind_to_bytes(policy.k_sem.kind.value)
        k_geo_bytes = geo_dim * self._kind_to_bytes(policy.k_geo.kind.value)
        v_bytes = v_dim * self._kind_to_bytes(policy.v.kind.value)
        return int((k_sem_bytes + k_geo_bytes + v_bytes) * n_layers)

    def collect_compare_batches(self, batch_size: int, block_size: int, count: int) -> list[Tensor]:
        """Collect a small deterministic batch set."""
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
        """Build models and load teacher weights.

        The teacher uses standard attention (matching the original Llama checkpoint),
        while the student uses the manifest's attention config (typically DBA).
        This creates a meaningful distillation target where the teacher provides
        "correct" outputs and the student learns to match them with the new architecture.
        """
        if train.teacher_ckpt is None:
            raise ValueError("train.teacher_ckpt is required for upcycle.")

        ckpt_path = self.resolve_teacher_ckpt(train.teacher_ckpt)
        state_dict = CheckpointLoader().load(ckpt_path)
        self._print(f"upcycle: loaded state_dict keys={len(state_dict)}")

        # Create teacher with standard attention (original Llama architecture)
        teacher_model_config = _make_teacher_model_config(self.manifest.model)
        self.teacher = Model(teacher_model_config).to(device=self.device, dtype=self.dtype)

        # Create student with the manifest's attention config (e.g., DBA)
        self.student = Model(self.manifest.model).to(device=self.device, dtype=self.dtype)

        self._print("upcycle: teacher uses standard attention, student uses manifest attention")

        # Load weights into both models
        # Teacher: direct load (architecture matches checkpoint)
        LlamaUpcycle(self.teacher, state_dict).apply()
        # Student: upcycle surgery (transforms weights for new architecture)
        LlamaUpcycle(self.student, state_dict).apply()

        # Wrap student for distributed training (teacher stays unwrapped for eval)
        if self.dist_ctx is not None:
            self.student = self.dist_ctx.wrap_model(self.student)

        self.teacher.eval()
        self._print("upcycle: initialization complete")

    def _print(self, msg: str) -> None:
        """Print only from main process in distributed mode."""
        if self.dist_ctx is not None:
            self.dist_ctx.print(msg)
        else:
            print(msg)

    def _unfreeze_all_parameters(self) -> None:
        """Re-enable gradients on all student parameters.

        Call this before global fine-tuning to ensure all parameters
        are trainable (blockwise training freezes most of them).
        """
        for param in self.student.parameters():
            param.requires_grad = True

    def run_blockwise(self, run: Run) -> None:
        """Run blockwise distillation."""
        train = self.require_train(run)
        loader = self.build_loader(train)
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=train.lr)
        trainer = BlockwiseTrainer(
            teacher=self.teacher,
            student=self.student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _, m: isinstance(m, AttentionLayer),
        )

        self.student.train()
        loader_iter = iter(loader)
        for block_index in range(trainer.block_count()):
            loss = None
            for step in range(run.steps):
                (x, _), loader_iter = self.next_batch(loader, loader_iter)
                x = x.to(device=self.device)
                loss = trainer.step(x, block_index=block_index)

            # Reduce loss across processes for logging
            if loss is not None:
                if self.dist_ctx is not None:
                    loss = self.dist_ctx.all_reduce(loss, op="avg")
                self._print(f"blockwise block={block_index} loss={float(loss):.6f}")

    def run_global(self, run: Run) -> None:
        """Run global fine-tuning on next-token loss.

        This phase runs after blockwise distillation. We must re-enable
        gradients on all parameters since blockwise freezes all but the
        last trained block.

        If the model has a diffusion head enabled, this phase trains
        both the cross-entropy loss and the diffusion denoising loss.
        """
        train = self.require_train(run)
        loader = self.build_loader(train)

        # Re-enable gradients on all student parameters
        # (blockwise training leaves most params frozen)
        self._unfreeze_all_parameters()

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=train.lr)
        self.student.train()

        # Check if diffusion head is enabled
        has_diffusion = self._has_diffusion_head()

        loader_iter = iter(loader)
        loss: Tensor | None = None
        for step in range(run.steps):
            (x, y), loader_iter = self.next_batch(loader, loader_iter)
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            ce_loss: Tensor
            diff_loss: Tensor | None = None

            if has_diffusion:
                # Use return_features path for diffusion training
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

            # Periodic logging
            if step % 100 == 0:
                if self.dist_ctx is not None:
                    loss_reduced = self.dist_ctx.all_reduce(loss.detach(), op="avg")
                else:
                    loss_reduced = loss.detach()
                if has_diffusion and diff_loss is not None:
                    self._print(
                        f"global step={step} loss={float(loss_reduced):.6f} "
                        f"(ce={float(ce_loss):.4f} diff={float(diff_loss):.4f})"
                    )
                else:
                    self._print(f"global step={step} loss={float(loss_reduced):.6f}")

        if loss is not None:
            if self.dist_ctx is not None:
                loss = self.dist_ctx.all_reduce(loss.detach(), op="avg")
            self._print(f"global loss={float(loss):.6f}")

    def _has_diffusion_head(self) -> bool:
        """Check if the student model has a diffusion head enabled."""
        return (
            hasattr(self.student, "diffusion_head")
            and getattr(self.student, "diffusion_head", None) is not None
        )

    def _get_diffusion_loss_weight(self) -> float:
        """Get the diffusion loss weight from config."""
        if hasattr(self.student, "config"):
            cfg = getattr(self.student, "config", None)
            if cfg is not None and hasattr(cfg, "diffusion_head"):
                dh_cfg = getattr(cfg, "diffusion_head", None)
                if dh_cfg is not None and hasattr(dh_cfg, "loss_weight"):
                    return float(getattr(dh_cfg, "loss_weight", 0.10))
        return 0.10  # Default weight

    def resolve_teacher_ckpt(self, ckpt: str) -> Path:
        """Resolve a local path or hf:// URI."""
        if ckpt.startswith("hf://"):
            return HFLoader(repo_id=ckpt[5:]).load()
        return Path(ckpt)

    def build_loader(self, train: TrainConfig) -> DataLoader[tuple[Tensor, Tensor]]:
        """Build the data loader with distributed support."""
        path = Path(self.group.data)
        dataset = NpyDataset(str(path), block_size=train.block_size)

        if self.dist_ctx is not None:
            return self.dist_ctx.wrap_dataloader(
                dataset,
                batch_size=train.batch_size,
                shuffle=True,
                drop_last=True,
            )
        return DataLoader(dataset, batch_size=train.batch_size, shuffle=True, drop_last=True)

    @staticmethod
    def next_batch(
        loader: DataLoader[tuple[Tensor, Tensor]],
        iterator: Iterator[tuple[Tensor, Tensor]],
    ) -> tuple[tuple[Tensor, Tensor], Iterator[tuple[Tensor, Tensor]]]:
        """Return next batch, cycling if needed."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iter = iter(loader)
            return next(new_iter), new_iter

    def require_train(self, run: Run) -> TrainConfig:
        """Return the train config or raise."""
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        return run.train

    @staticmethod
    def parse_device(device: str) -> torch.device:
        """Parse device string."""
        return torch.device(device)

    @staticmethod
    def parse_dtype(dtype: str) -> torch.dtype:
        """Parse dtype string."""
        match dtype:
            case "float32": return torch.float32
            case "float16": return torch.float16
            case "bfloat16": return torch.bfloat16
            case _: raise ValueError(f"Unsupported dtype: {dtype}")
