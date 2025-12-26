"""
memory provides memory profiling for language models.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

import torch
from torch import nn

from caramba.config.benchmark import MemoryBenchmarkConfig
from caramba.layer.attention import AttentionLayer, AttentionMode


@dataclass
class MemoryMeasurement:
    """Single memory measurement."""

    seq_len: int
    batch_size: int
    peak_memory_mb: float
    kvcache_memory_mb: float
    model_memory_mb: float
    quantization: str


@dataclass
class KVCacheAnalysis:
    """Analysis of KV-cache memory usage."""

    model_name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    attention_mode: str
    bytes_per_token_fp16: float
    bytes_per_token_q8: float
    bytes_per_token_q4: float

    # DBA-specific
    sem_dim: int | None = None
    geo_dim: int | None = None
    bytes_per_token_dba_fp16: float | None = None


@dataclass
class MemoryResult:
    """Result of a memory benchmark."""

    model_name: str
    measurements: list[MemoryMeasurement] = field(default_factory=list)
    kvcache_analysis: KVCacheAnalysis | None = None

    @property
    def peak_memory_mb(self) -> float:
        if not self.measurements:
            return 0.0
        return max(m.peak_memory_mb for m in self.measurements)


class MemoryBenchmark:
    """Measures memory usage including KV-cache."""

    def __init__(self, config: MemoryBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def run(self, model: nn.Module, model_name: str) -> MemoryResult:
        """Run memory benchmark on a model."""
        model.eval()
        result = MemoryResult(model_name=model_name)

        # Analyze KV-cache structure
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
        """Analyze KV-cache structure from model architecture."""
        n_layers = 0
        n_kv_heads: int | None = None
        head_dim: int | None = None
        attention_mode: str | None = None
        sem_dim: int | None = None
        geo_dim: int | None = None

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
                        f"Inconsistent n_kv_heads across layers: {n_kv_heads} vs {layer_n_kv_heads}"
                    )

                if head_dim is None:
                    head_dim = layer_head_dim
                elif head_dim != layer_head_dim:
                    raise ValueError(
                        f"Inconsistent head_dim across layers: {head_dim} vs {layer_head_dim}"
                    )

                if attention_mode is None:
                    attention_mode = layer_mode
                elif attention_mode != layer_mode:
                    raise ValueError(
                        f"Inconsistent attention mode across layers: {attention_mode} vs {layer_mode}"
                    )

                if module.mode == AttentionMode.DECOUPLED:
                    cfg = module.config
                    layer_sem_dim = cfg.sem_dim
                    layer_geo_dim = cfg.geo_dim

                    if sem_dim is None:
                        sem_dim = layer_sem_dim
                    elif sem_dim != layer_sem_dim:
                        raise ValueError(
                            f"Inconsistent sem_dim across layers: {sem_dim} vs {layer_sem_dim}"
                        )

                    if geo_dim is None:
                        geo_dim = layer_geo_dim
                    elif geo_dim != layer_geo_dim:
                        raise ValueError(
                            f"Inconsistent geo_dim across layers: {geo_dim} vs {layer_geo_dim}"
                        )

        # Use defaults if no attention layers found
        n_kv_heads = n_kv_heads or 0
        head_dim = head_dim or 0
        attention_mode = attention_mode or "standard"

        # Calculate bytes per token
        # Standard: 2 * n_layers * n_kv_heads * head_dim * dtype_size (K and V)
        kv_dim = n_kv_heads * head_dim
        bytes_fp16 = 2 * n_layers * kv_dim * 2  # 2 bytes for fp16
        bytes_q8 = 2 * n_layers * kv_dim * 1  # 1 byte for q8
        bytes_q4 = 2 * n_layers * kv_dim * 0.5  # 0.5 bytes for q4

        # DBA: reduced cache due to smaller key dimension
        bytes_dba_fp16: float | None = None
        if sem_dim is not None and geo_dim is not None:
            dba_k_dim = sem_dim + geo_dim
            bytes_dba_fp16 = float(n_layers * (dba_k_dim + kv_dim) * 2)  # sem+geo for K, full for V

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
            bytes_per_token_dba_fp16=bytes_dba_fp16,
        )

    def _measure(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        quantization: str,
    ) -> MemoryMeasurement:
        """Measure actual memory usage."""
        # Clear memory
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Get baseline model memory
        if self.device.type == "cuda":
            model_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            model_memory = 0.0  # Can't measure on CPU/MPS easily

        # Determine vocab size from model
        vocab_size = self._get_vocab_size(model)

        # Create input and run forward
        input_ids = torch.randint(
            0, vocab_size,
            (batch_size, seq_len),
            device=self.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            _ = model(input_ids)

        # Measure peak memory
        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_memory = 0.0

        # Estimate KV-cache memory (theoretical)
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
        """Estimate KV-cache memory usage."""
        n_layers = 0
        kv_dim = 0
        dba_k_dim: int | None = None

        for module in model.modules():
            if isinstance(module, AttentionLayer):
                n_layers += 1
                kv_dim = module.n_kv_heads * module.head_dim

                if module.mode == AttentionMode.DECOUPLED:
                    cfg = module.config
                    if cfg.sem_dim and cfg.geo_dim:
                        dba_k_dim = cfg.sem_dim + cfg.geo_dim

        # Bytes per element
        bytes_per_elem = {
            "fp16": 2.0,
            "fp32": 4.0,
            "q8": 1.0,
            "q4": 0.5,
        }.get(quantization, 2.0)

        if dba_k_dim is not None:
            # DBA: K uses sem+geo dims, V uses full kv_dim
            k_bytes = n_layers * batch_size * seq_len * dba_k_dim * bytes_per_elem
            v_bytes = n_layers * batch_size * seq_len * kv_dim * bytes_per_elem
            total_bytes = k_bytes + v_bytes
        else:
            # Standard: K and V both use kv_dim
            total_bytes = 2 * n_layers * batch_size * seq_len * kv_dim * bytes_per_elem

        return total_bytes / (1024 * 1024)  # MB

    def _get_vocab_size(self, model: nn.Module) -> int:
        """Get vocab size from model, with fallback to default."""
        # Try common config attributes
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):  # type: ignore[union-attr]
            return int(model.config.vocab_size)  # type: ignore[union-attr]

        # Try getting from embedding layer
        if hasattr(model, "get_input_embeddings"):
            embedding = model.get_input_embeddings()  # type: ignore[operator]
            if embedding is not None and hasattr(embedding, "num_embeddings"):
                return int(embedding.num_embeddings)  # type: ignore[union-attr]

        # Fallback to a reasonable default
        return 32000
