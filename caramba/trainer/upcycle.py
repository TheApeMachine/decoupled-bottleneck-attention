"""
upcycle provides the upcycle training loop.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from caramba.config.group import Group
from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.run import Run
from caramba.config.topology import NodeConfig, TopologyConfig
from caramba.config.train import TrainConfig, TrainPhase
from caramba.config.verify import CompareVerifyConfig, KVCacheVerifyConfig
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
from caramba.eval.suite import assert_eval_thresholds, run_eval_verify
# TODO: Re-implement KV cache verification with real cache
# from caramba.trainer.kvcache_verify import estimate_kvcache_report


class Upcycle:
    """Runs blockwise distillation and global fine-tuning."""

    def __init__(
        self,
        manifest: Manifest,
        group: Group,
        train: TrainConfig,
    ) -> None:
        self.manifest = manifest
        self.group = group
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
        """Estimate KV-cache bytes for teacher vs student."""
        # TODO: Re-implement with real cache estimation
        print("verify: kvcache verification not yet implemented")

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
        """Build models and load teacher weights."""
        if train.teacher_ckpt is None:
            raise ValueError("train.teacher_ckpt is required for upcycle.")

        ckpt_path = self.resolve_teacher_ckpt(train.teacher_ckpt)
        state_dict = CheckpointLoader().load(ckpt_path)
        print(f"upcycle: loaded state_dict keys={len(state_dict)}")

        self.teacher = Model(self.manifest.model).to(device=self.device, dtype=self.dtype)
        self.student = Model(self.manifest.model).to(device=self.device, dtype=self.dtype)

        LlamaUpcycle(self.teacher, state_dict).apply()
        LlamaUpcycle(self.student, state_dict).apply()

        self.teacher.eval()
        print("upcycle: initialization complete")

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
            if loss is not None:
                print(f"blockwise block={block_index} loss={float(loss):.6f}")

    def run_global(self, run: Run) -> None:
        """Run global fine-tuning on next-token loss."""
        train = self.require_train(run)
        loader = self.build_loader(train)
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=train.lr)
        self.student.train()

        loader_iter = iter(loader)
        loss: Tensor | None = None
        for _ in range(run.steps):
            (x, y), loader_iter = self.next_batch(loader, loader_iter)
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            logits = self.student.forward(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if loss is not None:
            print(f"global loss={float(loss):.6f}")

    def resolve_teacher_ckpt(self, ckpt: str) -> Path:
        """Resolve a local path or hf:// URI."""
        if ckpt.startswith("hf://"):
            return HFLoader(repo_id=ckpt[5:]).load()
        return Path(ckpt)

    def build_loader(self, train: TrainConfig) -> DataLoader[tuple[Tensor, Tensor]]:
        """Build the data loader."""
        path = Path(self.group.data)
        dataset = NpyDataset(str(path), block_size=train.block_size)
        return DataLoader(dataset, batch_size=train.batch_size, shuffle=True, drop_last=True)

    @staticmethod
    def next_batch(
        loader: DataLoader[tuple[Tensor, Tensor]],
        iterator: object,
    ) -> tuple[tuple[Tensor, Tensor], object]:
        """Return next batch, cycling if needed."""
        try:
            return next(iterator), iterator  # type: ignore[arg-type]
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
