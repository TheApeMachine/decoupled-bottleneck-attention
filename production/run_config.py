"""Configuration classes for training and sampling runs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Optional


def _get(args: argparse.Namespace, name: str, default: Any) -> Any:
    try:
        return getattr(args, name)
    except AttributeError:
        return default


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return int(default)


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return float(default)


def _as_str(x: Any, default: str) -> str:
    try:
        s = str(x)
        return s
    except (ValueError, TypeError):
        return str(default)


def _as_opt_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class CommonRunConfig:
    """Common configuration shared by training and sampling runs."""

    out_dir: Optional[str]
    instrument: str
    live_plot: bool
    tb: bool
    wandb: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CommonRunConfig":
        """Create a CommonRunConfig from argparse.Namespace arguments."""
        return cls(
            out_dir=(_as_str(_get(args, "out_dir", None), "") or None),
            instrument=_as_str(_get(args, "instrument", "off"), "off"),
            live_plot=bool(_get(args, "live_plot", False)),
            tb=bool(_get(args, "tb", False)),
            wandb=bool(_get(args, "wandb", False)),
        )


@dataclass(frozen=True)
class SampleConfig(CommonRunConfig):
    """Configuration for model sampling/inference runs."""

    ckpt: str
    draft_ckpt: Optional[str]
    prompt_tokens: str
    tokenizer: str

    max_new_tokens: int
    temperature: float
    top_k: Optional[int]

    # KV controls
    kv_cache: str
    kv_qblock: int
    kv_residual: int
    kv_decode_block: int
    kv_fused: str

    # Optional heterogeneous overrides
    kv_cache_k: Optional[str]
    kv_cache_v: Optional[str]
    kv_cache_k_sem: Optional[str]
    kv_cache_k_geo: Optional[str]
    kv_qblock_k: Optional[int]
    kv_qblock_v: Optional[int]
    kv_qblock_k_sem: Optional[int]
    kv_qblock_k_geo: Optional[int]

    # Spec knobs
    spec_k: int
    spec_method: str
    spec_extra_token: bool
    spec_disable_below_accept: float

    # Expert override for decoupled cache policy
    kv_policy: Optional[str]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SampleConfig":
        common = CommonRunConfig.from_args(args)
        ckpt = _as_str(_get(args, "ckpt", ""), "")
        if not ckpt:
            ckpt = ""
        draft = _as_str(_get(args, "draft_ckpt", ""), "") or None
        return cls(
            **common.__dict__,
            ckpt=ckpt,
            draft_ckpt=draft,
            prompt_tokens=_as_str(_get(args, "prompt_tokens", "0"), "0"),
            tokenizer=_as_str(_get(args, "tokenizer", "raw"), "raw"),
            max_new_tokens=_as_int(_get(args, "max_new_tokens", 50), 50),
            temperature=_as_float(_get(args, "temperature", 1.0), 1.0),
            top_k=_as_opt_int(_get(args, "top_k", None)),
            kv_cache=_as_str(_get(args, "kv_cache", "fp16"), "fp16"),
            kv_qblock=_as_int(_get(args, "kv_qblock", 32), 32),
            kv_residual=_as_int(_get(args, "kv_residual", 0), 0),
            kv_decode_block=_as_int(_get(args, "kv_decode_block", 1024), 1024),
            kv_fused=_as_str(_get(args, "kv_fused", "auto"), "auto"),
            kv_cache_k=(_as_str(_get(args, "kv_cache_k", ""), "") or None),
            kv_cache_v=(_as_str(_get(args, "kv_cache_v", ""), "") or None),
            kv_cache_k_sem=(_as_str(_get(args, "kv_cache_k_sem", ""), "") or None),
            kv_cache_k_geo=(_as_str(_get(args, "kv_cache_k_geo", ""), "") or None),
            kv_qblock_k=_as_opt_int(_get(args, "kv_qblock_k", None)),
            kv_qblock_v=_as_opt_int(_get(args, "kv_qblock_v", None)),
            kv_qblock_k_sem=_as_opt_int(_get(args, "kv_qblock_k_sem", None)),
            kv_qblock_k_geo=_as_opt_int(_get(args, "kv_qblock_k_geo", None)),
            spec_k=_as_int(_get(args, "spec_k", 4), 4),
            spec_method=_as_str(_get(args, "spec_method", "reject_sampling"), "reject_sampling"),
            spec_extra_token=bool(_get(args, "spec_extra_token", False)),
            spec_disable_below_accept=_as_float(_get(args, "spec_disable_below_accept", 0.0), 0.0),
            kv_policy=(_as_str(_get(args, "kv_policy", ""), "") or None),
        )


@dataclass(frozen=True)
class TrainConfig(CommonRunConfig):
    """Configuration for model training runs."""

    data: str
    vocab_size: Optional[int]
    tokenizer: str
    data_format: str
    data_dtype: str
    val_frac: float

    # Model shape / architecture
    block: int
    layers: int
    n_head: int
    kv_head: Optional[int]
    d_model: int
    d_ff: int
    embed_dim: int

    attn_mode: str
    attn_dim: int
    sem_dim: int
    geo_dim: int

    no_decoupled_gate: bool
    no_rope: bool
    rope_base: float
    tie_qk: bool
    null_attn: bool
    no_learned_temp: bool
    mlp: str
    dropout: float

    # Training
    steps: int
    optimizer: str
    lr: float
    weight_decay: float
    lr_schedule: str
    warmup_steps: int
    min_lr: float

    # Optimizer knobs
    adam_eps: float
    adam_betas: str
    lion_betas: str
    opt_foreach: bool
    opt_fused: bool

    # Loop cadence
    eval_iters: int
    eval_every: int
    save_every: int
    log_every: int

    # Schedules (optional)
    seq_schedule: Optional[str]
    batch_schedule: Optional[str]
    batch_by_seq: Optional[str]

    # Batch defaults (can be auto-populated later)
    batch_size: int
    grad_accum: int

    # Resume
    resume: bool
    resume_path: Optional[str]
    resume_allow_config_mismatch: bool

    # Stability
    nan_policy: str
    nan_lr_decay: float
    grad_clip: float
    sync_timing: bool
    legacy_micro_steps: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        common = CommonRunConfig.from_args(args)
        return cls(
            **common.__dict__,
            data=_as_str(_get(args, "data", ""), ""),
            vocab_size=_as_opt_int(_get(args, "vocab_size", None)),
            tokenizer=_as_str(_get(args, "tokenizer", "raw"), "raw"),
            data_format=_as_str(_get(args, "data_format", "auto"), "auto"),
            data_dtype=_as_str(_get(args, "data_dtype", "int64"), "int64"),
            val_frac=_as_float(_get(args, "val_frac", 0.1), 0.1),
            block=_as_int(_get(args, "block", 0), 0),
            layers=_as_int(_get(args, "layers", 0), 0),
            n_head=_as_int(_get(args, "n_head", 0), 0),
            kv_head=_as_opt_int(_get(args, "kv_head", None)),
            d_model=_as_int(_get(args, "d_model", 0), 0),
            d_ff=_as_int(_get(args, "d_ff", 0), 0),
            embed_dim=_as_int(_get(args, "embed_dim", 0), 0),
            attn_mode=_as_str(_get(args, "attn_mode", ""), ""),
            attn_dim=_as_int(_get(args, "attn_dim", 0), 0),
            sem_dim=_as_int(_get(args, "sem_dim", 0), 0),
            geo_dim=_as_int(_get(args, "geo_dim", 0), 0),
            no_decoupled_gate=bool(_get(args, "no_decoupled_gate", False)),
            no_rope=bool(_get(args, "no_rope", False)),
            rope_base=_as_float(_get(args, "rope_base", 10000.0), 10000.0),
            tie_qk=bool(_get(args, "tie_qk", False)),
            null_attn=bool(_get(args, "null_attn", False)),
            no_learned_temp=bool(_get(args, "no_learned_temp", False)),
            mlp=_as_str(_get(args, "mlp", "swiglu"), "swiglu"),
            dropout=_as_float(_get(args, "dropout", 0.0), 0.0),
            steps=_as_int(_get(args, "steps", -1), -1),
            optimizer=_as_str(_get(args, "optimizer", "adamw"), "adamw"),
            lr=_as_float(_get(args, "lr", 3e-4), 3e-4),
            weight_decay=_as_float(_get(args, "weight_decay", 0.1), 0.1),
            lr_schedule=_as_str(_get(args, "lr_schedule", "cosine"), "cosine"),
            warmup_steps=_as_int(_get(args, "warmup_steps", 0), 0),
            min_lr=_as_float(_get(args, "min_lr", 0.0), 0.0),
            adam_eps=_as_float(_get(args, "adam_eps", 1e-8), 1e-8),
            adam_betas=_as_str(_get(args, "adam_betas", "0.9,0.95"), "0.9,0.95"),
            lion_betas=_as_str(_get(args, "lion_betas", "0.9,0.99"), "0.9,0.99"),
            opt_foreach=bool(_get(args, "opt_foreach", False)),
            opt_fused=bool(_get(args, "opt_fused", False)),
            eval_iters=_as_int(_get(args, "eval_iters", 20), 20),
            eval_every=_as_int(_get(args, "eval_every", 0), 0),
            save_every=_as_int(_get(args, "save_every", 0), 0),
            log_every=_as_int(_get(args, "log_every", 0), 0),
            seq_schedule=(_as_str(_get(args, "seq_schedule", ""), "") or None),
            batch_schedule=(_as_str(_get(args, "batch_schedule", ""), "") or None),
            batch_by_seq=(_as_str(_get(args, "batch_by_seq", ""), "") or None),
            batch_size=_as_int(_get(args, "batch_size", 0), 0),
            grad_accum=_as_int(_get(args, "grad_accum", 0), 0),
            resume=bool(_get(args, "resume", False)),
            resume_path=(_as_str(_get(args, "resume_path", ""), "") or None),
            resume_allow_config_mismatch=bool(_get(args, "resume_allow_config_mismatch", False)),
            nan_policy=_as_str(_get(args, "nan_policy", "reduce_lr"), "reduce_lr"),
            nan_lr_decay=_as_float(_get(args, "nan_lr_decay", 0.5), 0.5),
            grad_clip=_as_float(_get(args, "grad_clip", 0.0), 0.0),
            sync_timing=bool(_get(args, "sync_timing", False)),
            legacy_micro_steps=bool(_get(args, "legacy_micro_steps", False)),
        )
