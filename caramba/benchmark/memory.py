"""Memory benchmark: measuring KV-cache and peak memory usage.

For DBA upcycling, the key metric is KV-cache memory reduction. Standard
attention caches K and V tensors of size n_kv_heads × head_dim per token.
DBA caches semantic keys (sem_dim), geometric keys (geo_dim), and values
(v_dim)—typically much smaller total.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from caramba.benchmark.utils import get_model_vocab_size
from caramba.config.benchmark import MemoryBenchmarkConfig
from caramba.layer.attention import AttentionLayer, AttentionMode

logger = logging.getLogger(__name__)


@dataclass
class MemoryMeasurement:
    """Single memory measurement for a specific configuration."""

    seq_len: int
    batch_size: int
    peak_memory_mb: float
    kvcache_memory_mb: float
    model_memory_mb: float
    quantization: str


@dataclass
class KVCacheAnalysis:
    """Analysis of KV-cache memory usage.

    For standard attention, K and V both use n_kv_heads × head_dim.
    For DBA, the cache stores:
    - k_sem: semantic keys (sem_dim per layer)
    - k_geo: geometric keys (geo_dim per layer)
    - v: values (v_dim per layer)

    Byte estimates include quantization overhead:
    - fp16: 2 bytes/element
    - q8: 1 byte/element
    - q4: 0.625 bytes/element
    """

    model_name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    attention_mode: str
    bytes_per_token_fp16: float
    bytes_per_token_q8: float
    bytes_per_token_q4: float

    # DBA-specific dimensions
    sem_dim: int | None = None
    geo_dim: int | None = None
    v_dim: int | None = None
    bytes_per_token_dba_fp16: float | None = None
    bytes_per_token_dba_q8: float | None = None
    bytes_per_token_dba_q4: float | None = None


@dataclass
class MemoryResult:
    """Complete memory benchmark results for a model."""

    model_name: str
    measurements: list[MemoryMeasurement] = field(default_factory=list)
    kvcache_analysis: KVCacheAnalysis | None = None

    @property
    def peak_memory_mb(self) -> float:
        """Maximum peak memory across all measurements."""
        if not self.measurements:
            return 0.0
        return max(m.peak_memory_mb for m in self.measurements)


class MemoryBenchmark:
    """Measures memory usage including KV-cache analysis.

    Analyzes the model architecture to compute theoretical KV-cache sizes,
    then runs actual forward passes to measure peak memory.
    """

    def __init__(
        self, config: MemoryBenchmarkConfig, device: torch.device
    ) -> None:
        """Set up the benchmark with config and target device."""
        self.config = config
        self.device = device

    def run(self, model: nn.Module, model_name: str) -> MemoryResult:
        """Run the memory benchmark, measuring both theoretical and actual usage."""
        model.eval()
        result = MemoryResult(model_name=model_name)

        # Analyze KV-cache structure from model architecture
        result.kvcache_analysis = self._analyze_kvcache(model, model_name)

        # Measure actual memory usage
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                for quant in self.config.quantization_modes:
                    measurement = self._measure(
                        model=model,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        quantization=quant,
                    )
                    result.measurements.append(measurement)

        return result

    def _analyze_kvcache(self, model: nn.Module, model_name: str) -> KVCacheAnalysis:
        """Analyze KV-cache structure from model architecture.

        Inspects attention layers to determine cache dimensions and
        compute theoretical bytes per token for different precisions.
        """
        n_layers = 0
        n_kv_heads: int | None = None
        head_dim: int | None = None
        attention_mode: str | None = None
        sem_dim: int | None = None
        geo_dim: int | None = None
        v_dim: int | None = None

        for module in model.modules():
            if isinstance(module, AttentionLayer):
                n_layers += 1
                layer_n_kv_heads = module.n_kv_heads
                layer_head_dim = module.head_dim
                layer_mode = module.mode.value

                # Validate consistency across layers
                if n_kv_heads is None:
                    n_kv_heads = layer_n_kv_heads
                elif n_kv_heads != layer_n_kv_heads:
                    raise ValueError(
                        f"Inconsistent n_kv_heads: {n_kv_heads} vs {layer_n_kv_heads}"
                    )

                if head_dim is None:
                    head_dim = layer_head_dim
                elif head_dim != layer_head_dim:
                    raise ValueError(
                        f"Inconsistent head_dim: {head_dim} vs {layer_head_dim}"
                    )

                if attention_mode is None:
                    attention_mode = layer_mode
                elif attention_mode != layer_mode:
                    raise ValueError(
                        f"Inconsistent attention mode: {attention_mode} vs {layer_mode}"
                    )

                # Extract DBA dimensions
                if module.mode == AttentionMode.DECOUPLED:
                    cfg = module.config
                    layer_sem_dim = cfg.sem_dim
                    layer_geo_dim = cfg.geo_dim
                    layer_v_dim = cfg.v_dim

                    if sem_dim is None:
                        sem_dim = layer_sem_dim
                    elif sem_dim != layer_sem_dim:
                        raise ValueError(
                            f"Inconsistent sem_dim: {sem_dim} vs {layer_sem_dim}"
                        )

                    if geo_dim is None:
                        geo_dim = layer_geo_dim
                    elif geo_dim != layer_geo_dim:
                        raise ValueError(
                            f"Inconsistent geo_dim: {geo_dim} vs {layer_geo_dim}"
                        )

                    if v_dim is None:
                        v_dim = layer_v_dim
                    elif v_dim != layer_v_dim:
                        raise ValueError(
                            f"Inconsistent v_dim: {v_dim} vs {layer_v_dim}"
                        )

        # Use defaults if no attention layers found
        used_defaults = []
        if n_kv_heads is None:
            n_kv_heads = 0
            used_defaults.append("n_kv_heads=0")
        if head_dim is None:
            head_dim = 0
            used_defaults.append("head_dim=0")
        if attention_mode is None:
            attention_mode = "standard"
            used_defaults.append("attention_mode='standard'")

        if n_layers == 0 or used_defaults:
            logger.warning(
                "No attention layers detected; using defaults: %s",
                ", ".join(used_defaults) if used_defaults else "n_layers=0",
            )

        # Calculate bytes per token for standard attention
        kv_dim = n_kv_heads * head_dim
        bytes_fp16 = 2 * n_layers * kv_dim * 2.0
        bytes_q8 = 2 * n_layers * kv_dim * 1.0
        bytes_q4 = 2 * n_layers * kv_dim * 0.625

        # Calculate DBA bytes per token
        bytes_dba_fp16: float | None = None
        bytes_dba_q8: float | None = None
        bytes_dba_q4: float | None = None
        if sem_dim is not None and geo_dim is not None:
            actual_v_dim = v_dim if v_dim is not None else kv_dim
            dba_elements_per_token = n_layers * (sem_dim + geo_dim + actual_v_dim)
            bytes_dba_fp16 = float(dba_elements_per_token * 2.0)
            bytes_dba_q8 = float(dba_elements_per_token * 1.0)
            bytes_dba_q4 = float(dba_elements_per_token * 0.625)

        return KVCacheAnalysis(
            model_name=model_name,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            attention_mode=attention_mode,
            bytes_per_token_fp16=float(bytes_fp16),
            bytes_per_token_q8=float(bytes_q8),
            bytes_per_token_q4=float(bytes_q4),
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            v_dim=v_dim,
            bytes_per_token_dba_fp16=bytes_dba_fp16,
            bytes_per_token_dba_q8=bytes_dba_q8,
            bytes_per_token_dba_q4=bytes_dba_q4,
        )

    def _measure(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        quantization: str,
    ) -> MemoryMeasurement:
        """Measure actual memory usage for a specific configuration."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if self.device.type == "cuda":
            model_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            model_memory = 0.0

        vocab_size = self._get_vocab_size(model)

        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
            device=self.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            _ = model(input_ids)

        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_memory = 0.0

        kvcache_memory = self._estimate_kvcache_memory(
            model=model,
            batch_size=batch_size,
            seq_len=seq_len,
            quantization=quantization,
        )

        return MemoryMeasurement(
            seq_len=seq_len,
            batch_size=batch_size,
            peak_memory_mb=peak_memory,
            kvcache_memory_mb=kvcache_memory,
            model_memory_mb=model_memory,
            quantization=quantization,
        )

    def _estimate_kvcache_memory(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        quantization: str,
    ) -> float:
        """Estimate theoretical KV-cache memory usage."""
        n_layers = 0
        kv_dim = 0
        dba_k_dim: int | None = None
        dba_v_dim: int | None = None

        for module in model.modules():
            if isinstance(module, AttentionLayer):
                n_layers += 1
                kv_dim = module.n_kv_heads * module.head_dim

                if module.mode == AttentionMode.DECOUPLED:
                    cfg = module.config
                    if cfg.sem_dim and cfg.geo_dim:
                        dba_k_dim = cfg.sem_dim + cfg.geo_dim
                        dba_v_dim = cfg.v_dim

        bytes_per_elem = {
            "fp16": 2.0,
            "fp32": 4.0,
            "q8": 1.0,
            "q8_0": 1.0,
            "q4": 0.625,
            "q4_0": 0.625,
            "nf4": 0.625,
        }.get(quantization, 2.0)

        if dba_k_dim is not None:
            actual_v_dim = dba_v_dim if dba_v_dim is not None else kv_dim
            k_bytes = n_layers * batch_size * seq_len * dba_k_dim * bytes_per_elem
            v_bytes = n_layers * batch_size * seq_len * actual_v_dim * bytes_per_elem
            total_bytes = k_bytes + v_bytes
        else:
            total_bytes = (
                2 * n_layers * batch_size * seq_len * kv_dim * bytes_per_elem
            )

        return total_bytes / (1024 * 1024)

    def _get_vocab_size(self, model: nn.Module) -> int:
        """Get vocab size from model."""
        return get_model_vocab_size(model, default=32000)
