"""Optimizer module: quantization and optional Triton kernels.

This package provides low-level optimizations for inference:
- Quantization: q8_0, q4_0, nf4 formats for KV-cache compression
- Triton kernels: Fused decoupled attention for CUDA
- Runtime checks: Safe fallbacks when Triton isn't available
"""
from caramba.optimizer.quantizer import Quantizer

__all__ = ["Quantizer"]
