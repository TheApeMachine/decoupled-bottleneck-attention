"""Checkpoint loading utilities for pretrained weights.

Loading pretrained weights is more complex than it sounds—checkpoints come
in different formats (PyTorch, safetensors, sharded), from different sources
(local files, Hugging Face Hub), and need to be mapped to potentially different
architectures (standard attention → DBA). This package handles all of that.
"""
from __future__ import annotations
