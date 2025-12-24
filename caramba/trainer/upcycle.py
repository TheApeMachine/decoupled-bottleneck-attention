"""
upcycle provides the upcycle training loop.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from typing_extensions import override

from caramba.config.group import Group
from caramba.config.layer import AttentionLayerConfig
from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.run import Run
from caramba.config.topology import (
    NodeConfig,
    TopologyConfig,
    _TopologyConfigBase,
)
from caramba.config.train import TrainConfig, TrainPhase
from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    LlamaAttentionWeightConfig,
)
from caramba.data.npy import NpyDataset
from caramba.layer.attention import Attention
from caramba.load.hf import HfDownload
from caramba.load.llama_loader import load_torch_state_dict
from caramba.load.llama_upcycle import LlamaUpcycle
from caramba.model.model import Model
from caramba.trainer.blockwise import BlockwiseTrainer
from caramba.trainer.distill import DistillLoss


class Upcycle:
    """
    Upcycle runs blockwise distillation and global fine-tuning.
    """
    def __init__(
        self,
        *,
        manifest: Manifest,
        group: Group,
        train: TrainConfig,
    ) -> None:
        """
        __init__ initializes the upcycle session.
        """
        self.manifest: Manifest = manifest
        self.group: Group = group
        self.device: torch.device = self._parse_device(train.device)
        self.device_name: str = str(self.device)
        self.dtype: torch.dtype = self._parse_dtype(train.dtype)
        self.teacher: Model
        self.student: Model
        self._init_models(train)
    def run(self, run: Run) -> None:
        """
        run executes a single run phase.
        """
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        if run.train.device != self.device_name:
            raise ValueError(
                f"Run {run.id} device mismatch: "
                f"{run.train.device} vs {self.device_name}"
            )
        if run.train.dtype != self._dtype_name():
            raise ValueError(
                f"Run {run.id} dtype mismatch: "
                f"{run.train.dtype} vs {self._dtype_name()}"
            )

        torch.manual_seed(int(run.seed))

        match run.train.phase:
            case TrainPhase.BLOCKWISE:
                self._run_blockwise(run)
            case TrainPhase.GLOBAL:
                self._run_global(run)
            case _:
                raise ValueError(f"Unsupported train phase: {run.train.phase}")
    def _init_models(self, train: TrainConfig) -> None:
        """
        _init_models builds models and loads teacher weights.
        """
        if train.teacher_ckpt is None:
            raise ValueError("train.teacher_ckpt is required for upcycle.")

        print(
            "upcycle: initializing models "
            f"(device={self.device_name}, dtype={self._dtype_name()})"
        )
        teacher_cfg = self._teacher_config(self.manifest.model)
        self.teacher = Model(teacher_cfg).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.student = Model(self.manifest.model).to(
            device=self.device,
            dtype=self.dtype,
        )

        print(f"upcycle: resolving teacher_ckpt={train.teacher_ckpt!r}")
        ckpt_path = self._resolve_teacher_ckpt(train.teacher_ckpt)
        print(f"upcycle: loading teacher state_dict from {str(ckpt_path)!r}")
        state_dict = load_torch_state_dict(ckpt_path)
        print(f"upcycle: loaded state_dict keys={len(state_dict)}")

        print("upcycle: applying teacher weights")
        LlamaUpcycle(model=self.teacher, state_dict=state_dict).apply()
        print("upcycle: applying student weights")
        LlamaUpcycle(model=self.student, state_dict=state_dict).apply()

        self.teacher.eval()
        print("upcycle: initialization complete")
    def _run_blockwise(self, run: Run) -> None:
        """
        _run_blockwise runs blockwise distillation.
        """
        train = self._require_train(run)
        loader = self._build_loader(train)
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(train.lr),
        )
        trainer = BlockwiseTrainer(
            teacher=self.teacher,
            student=self.student,
            optimizer=optimizer,
            loss=DistillLoss(),
            predicate=lambda _name, module: isinstance(module, Attention),
        )

        print(
            "upcycle: blockwise start "
            f"(blocks={trainer.block_count()}, steps={int(run.steps)})"
        )
        self.student.train()
        loader_iter = iter(loader)
        for block_index in range(trainer.block_count()):
            last_loss = None
            steps = int(run.steps)
            for step in range(steps):
                (x, _y), loader_iter = self._next_batch(loader, loader_iter)
                x = x.to(device=self.device)
                last_loss = trainer.step(x, block_index=block_index)
                if step == 0:
                    print(f"upcycle: blockwise block={block_index} step=0")
            if last_loss is not None:
                print(
                    f"blockwise block={block_index} "
                    f"loss={float(last_loss):.6f}"
                )
    def _run_global(self, run: Run) -> None:
        """
        _run_global runs global fine-tuning on next-token loss.
        """
        train = self._require_train(run)
        loader = self._build_loader(train)
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(train.lr),
        )
        self.student.train()

        loader_iter = iter(loader)
        last_loss: Tensor | None = None
        for _ in range(int(run.steps)):
            (x, y), loader_iter = self._next_batch(loader, loader_iter)
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            logits = self.student.forward(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                y.reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()
        if last_loss is not None:
            print(f"global loss={float(last_loss):.6f}")

    def _resolve_teacher_ckpt(self, ckpt: str | None) -> Path:
        """
        _resolve_teacher_ckpt resolves a local path or hf:// URI.
        """
        if ckpt is None or not ckpt:
            raise ValueError("train.teacher_ckpt is required for upcycle.")
        if ckpt.startswith("hf://"):
            return HfDownload.from_uri(ckpt).fetch()

        path = Path(ckpt)
        if not path.is_file():
            raise ValueError(f"teacher_ckpt not found: {path}")
        return path
    def _build_loader(
        self,
        train: TrainConfig,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        """
        _build_loader builds the data loader.
        """
        data_path = self.group.data
        if not data_path:
            raise ValueError("Group data path is empty.")

        path = Path(data_path)
        if not path.is_file():
            raise ValueError(f"Data path not found: {path}")

        dataset = NpyDataset(str(path), block_size=int(train.block_size))
        return DataLoader(
            dataset,
            batch_size=int(train.batch_size),
            shuffle=True,
            drop_last=True,
        )
    @staticmethod
    def _next_batch(
        loader: DataLoader[tuple[Tensor, Tensor]],
        iterator: object,
    ) -> tuple[tuple[Tensor, Tensor], object]:
        """
        _next_batch returns the next batch, cycling the loader if needed.
        """
        try:
            batch = next(iterator)  # type: ignore[arg-type]
            return batch, iterator
        except StopIteration:
            new_iter = iter(loader)
            batch = next(new_iter)
            return batch, new_iter
    def _teacher_config(self, model: ModelConfig) -> ModelConfig:
        """
        _teacher_config derives a teacher config from the student model.
        """
        self._found_decoupled = False
        topo = self._swap_topology(model.topology)
        if not self._found_decoupled:
            raise ValueError(
                "Cannot derive teacher config: no decoupled attention "
                "layers found."
            )
        return model.model_copy(update={"topology": topo})
    def _swap_topology(self, config: TopologyConfig) -> TopologyConfig:
        """
        _swap_topology replaces decoupled attention weights with Llama weights.
        """
        layers = [self._swap_node(node) for node in list(config.layers)]
        return config.model_copy(update={"layers": layers})
    def _swap_node(self, node: NodeConfig) -> NodeConfig:
        """
        _swap_node swaps a layer or topology node.
        """
        if isinstance(node, _TopologyConfigBase):
            return self._swap_topology(node)
        return self._swap_layer(node)
    def _swap_layer(self, node: NodeConfig) -> NodeConfig:
        """
        _swap_layer replaces decoupled attention weight configs.
        """
        if not isinstance(node, AttentionLayerConfig):
            return node
        weight = node.weight
        if isinstance(weight, DecoupledAttentionWeightConfig):
            self._found_decoupled = True
            llama = LlamaAttentionWeightConfig(
                d_model=int(weight.d_model),
                n_heads=int(weight.n_heads),
                n_kv_heads=int(weight.n_kv_heads),
                rope_base=float(weight.rope_base),
                rope_dim=int(weight.rope_dim),
                bias=bool(weight.bias),
            )
            return node.model_copy(update={"weight": llama})
        return node
    def _require_train(self, run: Run) -> TrainConfig:
        """
        _require_train returns the train config or raises.
        """
        if run.train is None:
            raise ValueError(f"Run {run.id} has no train config.")
        return run.train

    @staticmethod
    def _parse_device(device: str) -> torch.device:
        """
        _parse_device validates and returns a torch.device.
        """
        if not device:
            raise ValueError("device must be non-empty")
        return torch.device(device)

    @staticmethod
    def _parse_dtype(dtype: str) -> torch.dtype:
        """
        _parse_dtype converts a dtype string to torch.dtype.
        """
        match dtype:
            case "float32":
                return torch.float32
            case "float16":
                return torch.float16
            case "bfloat16":
                return torch.bfloat16
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _dtype_name(self) -> str:
        """
        _dtype_name returns the current dtype name.
        """
        if self.dtype is torch.float32:
            return "float32"
        if self.dtype is torch.float16:
            return "float16"
        if self.dtype is torch.bfloat16:
            return "bfloat16"
        raise ValueError(f"Unsupported dtype: {self.dtype}")
