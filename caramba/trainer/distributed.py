"""Distributed training support with DDP and FSDP.

This module provides utilities for multi-GPU training using:
- DistributedDataParallel (DDP): For data-parallel training across multiple GPUs
- FullyShardedDataParallel (FSDP): For sharding model parameters across GPUs

Usage:
    # Initialize distributed training
    ctx = DistributedContext.init()

    # Wrap model
    model = ctx.wrap_model(model, strategy="ddp")  # or "fsdp"

    # Create distributed data loader
    loader = ctx.wrap_dataloader(dataset, batch_size=32)

    # Training loop
    for batch in loader:
        ...

    # Cleanup
    ctx.cleanup()

References:
- PyTorch DDP: https://pytorch.org/docs/stable/distributed.html
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from caramba.console import logger as console_logger

if TYPE_CHECKING:
    from torch.nn.parallel import DistributedDataParallel
    from torch.distributed.fsdp import FullyShardedDataParallel


__all__ = [
    "DistributedStrategy",
    "DistributedConfig",
    "DistributedContext",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
]


class DistributedStrategy(Enum):
    """Distributed training strategy."""
    NONE = "none"      # Single GPU / CPU
    DDP = "ddp"        # DistributedDataParallel
    FSDP = "fsdp"      # FullyShardedDataParallel


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Strategy selection
    strategy: DistributedStrategy = DistributedStrategy.NONE

    # DDP options
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25
    ddp_gradient_as_bucket_view: bool = True

    # FSDP options (requires torch >= 2.0)
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True
    fsdp_backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST
    fsdp_activation_checkpointing: bool = False
    fsdp_limit_all_gathers: bool = True
    fsdp_use_orig_params: bool = True

    # Auto-wrap policy for FSDP (transformer layers to wrap)
    fsdp_min_num_params: int = 100_000
    fsdp_transformer_layer_cls: list[str] = field(default_factory=list)

    # Communication backend
    backend: str = "nccl"  # nccl for CUDA, gloo for CPU

    # Process group
    init_method: str | None = None  # Auto-detect from env vars

    # Gradient synchronization
    sync_batch_norms: bool = True
    find_unused_parameters: bool = False


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def _get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


class DistributedContext:
    """Context manager for distributed training."""

    def __init__(self, config: DistributedConfig | None = None) -> None:
        self.config = config or DistributedConfig()
        self._initialized = False
        self._local_rank = 0
        self._rank = 0
        self._world_size = 1
        self._device: torch.device = torch.device("cpu")

    @classmethod
    def init(
        cls,
        config: DistributedConfig | None = None,
        local_rank: int | None = None,
    ) -> "DistributedContext":
        """Initialize distributed training context.

        This should be called at the start of training. It will:
        1. Initialize the process group (if not already initialized)
        2. Set up the device for this process
        3. Return a context object for managing distributed training

        Args:
            config: Distributed configuration
            local_rank: Local rank override (usually auto-detected from env)

        Returns:
            DistributedContext instance
        """
        ctx = cls(config)

        if config is None or config.strategy == DistributedStrategy.NONE:
            # Single-GPU mode
            ctx._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ctx._initialized = True
            return ctx

        # Get rank information from environment
        ctx._local_rank = local_rank if local_rank is not None else _get_local_rank()

        backend = ctx.config.backend

        # Initialize process group if needed
        if not dist.is_initialized():
            init_method = ctx.config.init_method

            # Auto-detect init method from environment
            # Both with and without MASTER_ADDR, we use "env://" as torchrun sets it
            if init_method is None:
                init_method = "env://"

            dist.init_process_group(backend=backend, init_method=init_method)

        ctx._rank = dist.get_rank()
        ctx._world_size = dist.get_world_size()

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(ctx._local_rank)
            ctx._device = torch.device(f"cuda:{ctx._local_rank}")
        else:
            ctx._device = torch.device("cpu")

        ctx._initialized = True

        if is_main_process():
            console_logger.info(
                f"Distributed training initialized: "
                f"world_size={ctx._world_size}, backend={backend}"
            )

        return ctx

    @property
    def device(self) -> torch.device:
        """Get the device for this process."""
        return self._device

    @property
    def rank(self) -> int:
        """Get the rank of this process."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Get the total number of processes."""
        return self._world_size

    @property
    def local_rank(self) -> int:
        """Get the local rank (GPU index on this node)."""
        return self._local_rank

    @property
    def is_main(self) -> bool:
        """Check if this is the main process."""
        return self._rank == 0

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap a model for distributed training.

        Args:
            model: The model to wrap

        Returns:
            Wrapped model (DDP, FSDP, or original)
        """
        if self.config.strategy == DistributedStrategy.NONE:
            return model.to(self._device)

        # Move model to device
        model = model.to(self._device)

        # Sync batch norms if requested
        if self.config.sync_batch_norms:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.config.strategy == DistributedStrategy.DDP:
            return self._wrap_ddp(model)
        elif self.config.strategy == DistributedStrategy.FSDP:
            return self._wrap_fsdp(model)
        else:
            return model

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(
            model,
            device_ids=[self._local_rank] if torch.cuda.is_available() else None,
            output_device=self._local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.ddp_find_unused_parameters,
            bucket_cap_mb=self.config.ddp_bucket_cap_mb,
            gradient_as_bucket_view=self.config.ddp_gradient_as_bucket_view,
        )

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        """Wrap model with FullyShardedDataParallel."""
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                CPUOffload,
                BackwardPrefetch,
                MixedPrecision,
            )
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )
        except ImportError as e:
            raise ImportError(
                "FSDP requires PyTorch 2.0+. Please upgrade: pip install torch>=2.0"
            ) from e

        # Parse sharding strategy
        sharding_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_map.get(
            self.config.fsdp_sharding_strategy,
            ShardingStrategy.FULL_SHARD,
        )

        # Parse backward prefetch
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = prefetch_map.get(
            self.config.fsdp_backward_prefetch,
            BackwardPrefetch.BACKWARD_PRE,
        )

        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_cpu_offload else None

        # Mixed precision
        mixed_precision = None
        if self.config.fsdp_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # Auto-wrap policy
        # Note: In PyTorch 2.0+, these return partial functions that FSDP uses
        from functools import partial

        if self.config.fsdp_transformer_layer_cls:
            # Use transformer-specific wrapping
            layer_classes: set[type[nn.Module]] = set()
            for cls_name in self.config.fsdp_transformer_layer_cls:
                # Try to resolve class name - collect all matching module types
                for _name, module in model.named_modules():
                    if type(module).__name__ == cls_name:
                        layer_classes.add(type(module))
                        # Don't break - continue to find all matching types

            if layer_classes:
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=layer_classes,
                )
            else:
                auto_wrap_policy = partial(
                    size_based_auto_wrap_policy,
                    min_num_params=self.config.fsdp_min_num_params,
                )
        else:
            auto_wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.fsdp_min_num_params,
            )

        wrapped = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            backward_prefetch=backward_prefetch,
            auto_wrap_policy=auto_wrap_policy,
            limit_all_gathers=self.config.fsdp_limit_all_gathers,
            use_orig_params=self.config.fsdp_use_orig_params,
            device_id=self._local_rank if torch.cuda.is_available() else None,
        )

        # Enable activation checkpointing if requested
        if self.config.fsdp_activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
                checkpoint_wrapper,
                CheckpointImpl,
            )

            non_reentrant_wrapper = lambda module: checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

            apply_activation_checkpointing(
                wrapped,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
            )

        return wrapped

    def wrap_dataloader(
        self,
        dataset: Dataset[Any],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> DataLoader[Any]:
        """Create a distributed-aware DataLoader.

        Args:
            dataset: The dataset to load
            batch_size: Batch size per process (not global)
            shuffle: Whether to shuffle (handled by DistributedSampler)
            drop_last: Whether to drop incomplete batches
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory (recommended for CUDA)
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader with distributed sampler if needed
        """
        if self.config.strategy == DistributedStrategy.NONE:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                **kwargs,
            )

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            **kwargs,
        )

    def set_epoch(self, loader: DataLoader[Any], epoch: int) -> None:
        """Set the epoch for the distributed sampler (for proper shuffling)."""
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

    def all_reduce(
        self,
        tensor: Tensor,
        op: str = "sum",
    ) -> Tensor:
        """All-reduce a tensor across processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ("sum", "avg", "max", "min")

        Returns:
            Reduced tensor
        """
        if not dist.is_initialized():
            return tensor

        op_map = {
            "sum": dist.ReduceOp.SUM,
            "avg": dist.ReduceOp.SUM,  # Will divide after
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }
        reduce_op = op_map.get(op, dist.ReduceOp.SUM)

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=reduce_op)

        if op == "avg":
            tensor = tensor / self._world_size

        return tensor

    def broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
        """Broadcast a tensor from src to all processes."""
        if not dist.is_initialized():
            return tensor

        dist.broadcast(tensor, src=src)
        return tensor

    def barrier(self) -> None:
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only from the main process (deprecated, use log instead)."""
        if self.is_main:
            console_logger.log(" ".join(str(arg) for arg in args))

    def log(self, msg: str) -> None:
        """Log a message only from the main process using the unified logger."""
        if self.is_main:
            console_logger.info(msg)

    def save_checkpoint(
        self,
        state: dict[str, Any],
        path: str,
        *,
        only_main: bool = True,
    ) -> None:
        """Save a checkpoint (only from main process by default)."""
        if only_main and not self.is_main:
            return

        # For FSDP, need special handling to gather full state
        if self.config.strategy == DistributedStrategy.FSDP:
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig

                # Check if 'model' key contains an FSDP-wrapped module
                if 'model' in state and isinstance(state['model'], FSDP):
                    fsdp_model = state['model']
                    # Use full state dict for single-file checkpoint
                    full_state_dict_config = FullStateDictConfig(
                        offload_to_cpu=True,
                        rank0_only=True,
                    )
                    with FSDP.state_dict_type(
                        fsdp_model,
                        StateDictType.FULL_STATE_DICT,
                        full_state_dict_config,
                    ):
                        state['model'] = fsdp_model.state_dict()

                # Synchronize all ranks before saving
                self.barrier()

                # Only rank 0 saves the checkpoint
                if self.is_main:
                    torch.save(state, path)

                # Synchronize after save
                self.barrier()
                return
            except ImportError:
                # Fall back to regular save if FSDP imports fail
                pass

        torch.save(state, path)

    def __enter__(self) -> "DistributedContext":
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()
