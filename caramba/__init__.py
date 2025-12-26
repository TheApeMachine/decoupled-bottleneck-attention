"""Caramba: A research platform for efficient AI.

Caramba provides tools for model architecture experimentation, with a focus on
Decoupled Bottleneck Attention (DBA) for KV-cache compression. The platform
supports upcycling pretrained models to new architectures and comprehensive
benchmarking.

Core workflows:
- Upcycling: Convert pretrained models (e.g., Llama) to DBA attention
- Training: Blockwise distillation + global fine-tuning
- Benchmarking: Perplexity, latency, memory profiling
- Artifact generation: Paper-ready CSV, charts, and LaTeX tables
"""
