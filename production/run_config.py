"""Configuration classes for training and sampling runs."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

# Python 3.12+ has `typing.override`; fall back to a no-op decorator.
try:  # pragma: no cover
    from typing import override
except ImportError:  # pragma: no cover
    try:
        from typing_extensions import override
    except ImportError:  # pragma: no cover
        _F = TypeVar("_F", bound=Callable[..., object])

        def override(f: _F) -> _F:
            return f

from production.selfopt_cache import as_str_object_dict


def _args_map(args: argparse.Namespace) -> dict[str, object]:
    # `argparse.Namespace` is dynamically typed (attributes are `Any`); treat it as a dict boundary.
    d = as_str_object_dict(args.__dict__)
    return {} if d is None else d


def _get(d: dict[str, object], name: str, default: object | None = None) -> object | None:
    v = d.get(str(name), default)
    return v


def _as_int(x: object, default: int) -> int:
    try:
        return int(str(x))
    except (ValueError, TypeError):
        return int(default)


def _as_float(x: object, default: float) -> float:
    try:
        return float(str(x))
    except (ValueError, TypeError):
        return float(default)


def _as_str(x: object, default: str) -> str:
    try:
        s = str(x)
        return s
    except (ValueError, TypeError):
        return str(default)


def _as_opt_int(x: object) -> int | None:
    if x is None:
        return None
    try:
        return int(str(x))
    except (ValueError, TypeError):
        pass

    try:
        v = float(str(x))
    except (ValueError, TypeError):
        return None

    return int(v) if v.is_integer() else None


@dataclass(frozen=True)
class CommonRunConfig:
    """Common configuration shared by training and sampling runs."""

    out_dir: str | None
    instrument: str
    live_plot: bool
    tb: bool
    wandb: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CommonRunConfig":
        """Create a CommonRunConfig from argparse.Namespace arguments."""
        d = _args_map(args)
        wandb_raw = _get(d, "wandb", False)
        wandb_enabled = True if wandb_raw is None else bool(wandb_raw)
        return cls(
            out_dir=(_as_str(_get(d, "out_dir", None), "") or None),
            instrument=_as_str(_get(d, "instrument", "off"), "off"),
            live_plot=bool(_get(d, "live_plot", False)),
            tb=bool(_get(d, "tb", False)),
            wandb=bool(wandb_enabled),
        )


@dataclass(frozen=True)
class SampleConfig(CommonRunConfig):
    """Configuration for model sampling/inference runs."""

    ckpt: str
    draft_ckpt: str | None
    prompt_tokens: str
    tokenizer: str

    max_new_tokens: int
    temperature: float
    top_k: int | None

    # KV controls
    kv_cache: str
    kv_qblock: int
    kv_residual: int
    kv_decode_block: int
    kv_fused: str

    # Optional heterogeneous overrides
    kv_cache_k: str | None
    kv_cache_v: str | None
    kv_cache_k_sem: str | None
    kv_cache_k_geo: str | None
    kv_qblock_k: int | None
    kv_qblock_v: int | None
    kv_qblock_k_sem: int | None
    kv_qblock_k_geo: int | None

    # Spec knobs
    spec_k: int
    spec_method: str
    spec_extra_token: bool
    spec_disable_below_accept: float

    # Expert override for decoupled cache policy
    kv_policy: str | None

    @classmethod
    @override
    def from_args(cls, args: argparse.Namespace) -> "SampleConfig":
        d = _args_map(args)
        common = CommonRunConfig.from_args(args)
        ckpt = _as_str(_get(d, "ckpt", ""), "")
        if not ckpt:
            ckpt = ""
        draft = _as_str(_get(d, "draft_ckpt", ""), "") or None
        return cls(
            out_dir=common.out_dir,
            instrument=common.instrument,
            live_plot=common.live_plot,
            tb=common.tb,
            wandb=common.wandb,
            ckpt=ckpt,
            draft_ckpt=draft,
            prompt_tokens=_as_str(_get(d, "prompt_tokens", "0"), "0"),
            tokenizer=_as_str(_get(d, "tokenizer", "raw"), "raw"),
            max_new_tokens=_as_int(_get(d, "max_new_tokens", 50), 50),
            temperature=_as_float(_get(d, "temperature", 1.0), 1.0),
            top_k=_as_opt_int(_get(d, "top_k", None)),
            kv_cache=_as_str(_get(d, "kv_cache", "fp16"), "fp16"),
            kv_qblock=_as_int(_get(d, "kv_qblock", 32), 32),
            kv_residual=_as_int(_get(d, "kv_residual", 0), 0),
            kv_decode_block=_as_int(_get(d, "kv_decode_block", 1024), 1024),
            kv_fused=_as_str(_get(d, "kv_fused", "auto"), "auto"),
            kv_cache_k=(_as_str(_get(d, "kv_cache_k", ""), "") or None),
            kv_cache_v=(_as_str(_get(d, "kv_cache_v", ""), "") or None),
            kv_cache_k_sem=(_as_str(_get(d, "kv_cache_k_sem", ""), "") or None),
            kv_cache_k_geo=(_as_str(_get(d, "kv_cache_k_geo", ""), "") or None),
            kv_qblock_k=_as_opt_int(_get(d, "kv_qblock_k", None)),
            kv_qblock_v=_as_opt_int(_get(d, "kv_qblock_v", None)),
            kv_qblock_k_sem=_as_opt_int(_get(d, "kv_qblock_k_sem", None)),
            kv_qblock_k_geo=_as_opt_int(_get(d, "kv_qblock_k_geo", None)),
            spec_k=_as_int(_get(d, "spec_k", 4), 4),
            spec_method=_as_str(_get(d, "spec_method", "reject_sampling"), "reject_sampling"),
            spec_extra_token=bool(_get(d, "spec_extra_token", False)),
            spec_disable_below_accept=_as_float(_get(d, "spec_disable_below_accept", 0.0), 0.0),
            kv_policy=(_as_str(_get(d, "kv_policy", ""), "") or None),
        )


@dataclass(frozen=True)
class TrainConfig(CommonRunConfig):
    """Configuration for model training runs."""

    data: str
    vocab_size: int | None
    tokenizer: str
    data_format: str
    data_dtype: str
    val_frac: float

    # Model shape / architecture
    block: int
    layers: int
    n_head: int
    kv_head: int | None
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

    # Optional diffusion head (adapter) knobs
    diffusion_head: bool
    diffusion_head_num_train_timesteps: int
    diffusion_head_num_infer_steps: int
    diffusion_head_time_embed_dim: int
    diffusion_head_mlp_mult: int
    diffusion_head_cfg_dropout_p: float
    diffusion_head_cfg_guidance_scale: float
    diffusion_head_scheduler: str
    diffusion_head_loss_weight: float

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
    seq_schedule: str | None
    batch_schedule: str | None
    batch_by_seq: str | None

    # Batch defaults (can be auto-populated later)
    batch_size: int
    grad_accum: int

    # Resume
    resume: bool
    resume_path: str | None
    resume_allow_config_mismatch: bool

    # Stability
    nan_policy: str
    nan_lr_decay: float
    grad_clip: float
    sync_timing: bool
    legacy_micro_steps: bool

    @classmethod
    @override
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        d = _args_map(args)
        common = CommonRunConfig.from_args(args)
        return cls(
            out_dir=common.out_dir,
            instrument=common.instrument,
            live_plot=common.live_plot,
            tb=common.tb,
            wandb=common.wandb,
            data=_as_str(_get(d, "data", ""), ""),
            vocab_size=_as_opt_int(_get(d, "vocab_size", None)),
            tokenizer=_as_str(_get(d, "tokenizer", "raw"), "raw"),
            data_format=_as_str(_get(d, "data_format", "auto"), "auto"),
            data_dtype=_as_str(_get(d, "data_dtype", "int64"), "int64"),
            val_frac=_as_float(_get(d, "val_frac", 0.1), 0.1),
            block=_as_int(_get(d, "block", 0), 0),
            layers=_as_int(_get(d, "layers", 0), 0),
            n_head=_as_int(_get(d, "n_head", 0), 0),
            kv_head=_as_opt_int(_get(d, "kv_head", None)),
            d_model=_as_int(_get(d, "d_model", 0), 0),
            d_ff=_as_int(_get(d, "d_ff", 0), 0),
            embed_dim=_as_int(_get(d, "embed_dim", 0), 0),
            attn_mode=_as_str(_get(d, "attn_mode", ""), ""),
            attn_dim=_as_int(_get(d, "attn_dim", 0), 0),
            sem_dim=_as_int(_get(d, "sem_dim", 0), 0),
            geo_dim=_as_int(_get(d, "geo_dim", 0), 0),
            no_decoupled_gate=bool(_get(d, "no_decoupled_gate", False)),
            no_rope=bool(_get(d, "no_rope", False)),
            rope_base=_as_float(_get(d, "rope_base", 10000.0), 10000.0),
            tie_qk=bool(_get(d, "tie_qk", False)),
            null_attn=bool(_get(d, "null_attn", False)),
            no_learned_temp=bool(_get(d, "no_learned_temp", False)),
            mlp=_as_str(_get(d, "mlp", "swiglu"), "swiglu"),
            dropout=_as_float(_get(d, "dropout", 0.0), 0.0),
            diffusion_head=bool(_get(d, "diffusion_head", False)),
            diffusion_head_num_train_timesteps=_as_int(_get(d, "diffusion_head_num_train_timesteps", 1000), 1000),
            diffusion_head_num_infer_steps=_as_int(_get(d, "diffusion_head_num_infer_steps", 12), 12),
            diffusion_head_time_embed_dim=_as_int(_get(d, "diffusion_head_time_embed_dim", 128), 128),
            diffusion_head_mlp_mult=_as_int(_get(d, "diffusion_head_mlp_mult", 4), 4),
            diffusion_head_cfg_dropout_p=_as_float(_get(d, "diffusion_head_cfg_dropout_p", 0.10), 0.10),
            diffusion_head_cfg_guidance_scale=_as_float(_get(d, "diffusion_head_cfg_guidance_scale", 1.5), 1.5),
            diffusion_head_scheduler=_as_str(_get(d, "diffusion_head_scheduler", "ddim"), "ddim"),
            diffusion_head_loss_weight=_as_float(_get(d, "diffusion_head_loss_weight", 0.10), 0.10),
            steps=_as_int(_get(d, "steps", -1), -1),
            optimizer=_as_str(_get(d, "optimizer", "adamw"), "adamw"),
            lr=_as_float(_get(d, "lr", 3e-4), 3e-4),
            weight_decay=_as_float(_get(d, "weight_decay", 0.1), 0.1),
            lr_schedule=_as_str(_get(d, "lr_schedule", "cosine"), "cosine"),
            warmup_steps=_as_int(_get(d, "warmup_steps", 0), 0),
            min_lr=_as_float(_get(d, "min_lr", 0.0), 0.0),
            adam_eps=_as_float(_get(d, "adam_eps", 1e-8), 1e-8),
            adam_betas=_as_str(_get(d, "adam_betas", "0.9,0.95"), "0.9,0.95"),
            lion_betas=_as_str(_get(d, "lion_betas", "0.9,0.99"), "0.9,0.99"),
            opt_foreach=bool(_get(d, "opt_foreach", False)),
            opt_fused=bool(_get(d, "opt_fused", False)),
            eval_iters=_as_int(_get(d, "eval_iters", 20), 20),
            eval_every=_as_int(_get(d, "eval_every", 0), 0),
            save_every=_as_int(_get(d, "save_every", 0), 0),
            log_every=_as_int(_get(d, "log_every", 0), 0),
            seq_schedule=(_as_str(_get(d, "seq_schedule", ""), "") or None),
            batch_schedule=(_as_str(_get(d, "batch_schedule", ""), "") or None),
            batch_by_seq=(_as_str(_get(d, "batch_by_seq", ""), "") or None),
            batch_size=_as_int(_get(d, "batch_size", 0), 0),
            grad_accum=_as_int(_get(d, "grad_accum", 0), 0),
            resume=bool(_get(d, "resume", False)),
            resume_path=(_as_str(_get(d, "resume_path", ""), "") or None),
            resume_allow_config_mismatch=bool(_get(d, "resume_allow_config_mismatch", False)),
            nan_policy=_as_str(_get(d, "nan_policy", "reduce_lr"), "reduce_lr"),
            nan_lr_decay=_as_float(_get(d, "nan_lr_decay", 0.5), 0.5),
            grad_clip=_as_float(_get(d, "grad_clip", 0.0), 0.0),
            sync_timing=bool(_get(d, "sync_timing", False)),
            legacy_micro_steps=bool(_get(d, "legacy_micro_steps", False)),
        )
