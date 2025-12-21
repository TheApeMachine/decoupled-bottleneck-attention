"""Human-readable run summary.

Why this exists:
- The CLI is intentionally minimal; we still want visibility into what was derived.
- A compact, stable printout makes debugging and experiment tracking easier.
"""

from __future__ import annotations

import os
from typing import cast


def _fmt_int(x: object) -> str:
    try:
        return f"{int(cast(int | float | str, x)):_d}"
    except (ValueError, TypeError):
        return "?"


def _fmt_float(x: object) -> str:
    try:
        return f"{float(cast(int | float | str, x)):.3g}"
    except (ValueError, TypeError):
        return "?"


def print_summary(*, args: object, device: object, cfg: object, n_total_tokens: int) -> None:
    """Why: show the key derived values up front so misconfigurations are obvious."""
    try:
        exp_s = str(getattr(args, "exp", None) or "")
        data_s = str(getattr(args, "data", None) or "")
        out_s = str(getattr(args, "out_dir", None) or "")

        ds_tok = getattr(args, "dataset_tokens", None)
        tp = getattr(args, "target_params", None)
        ds_src = getattr(args, "dataset_tokens_source", None)
        tp_src = getattr(args, "target_params_source", None)
        layers_src = getattr(args, "layers_source", None)
        exp_src = getattr(args, "exp_source", None)

        head_dim = None
        try:
            head_dim = int(getattr(cfg, "d_model")) // max(1, int(getattr(cfg, "n_head")))
        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
            head_dim = None

        print(
            f"[intent] device={str(device)} exp={exp_s or '?'} ({exp_src or '?'}) "
            f"data={os.path.basename(data_s) or '?'} "
            f"dataset_tokens={_fmt_int(ds_tok)} ({ds_src or '?'}) "
            f"target_params={_fmt_int(tp)} ({tp_src or '?'}) "
            f"layers={_fmt_int(getattr(cfg, 'n_layer', 0))} ({layers_src or '?'}) out_dir={out_s or '?'}",
            flush=True,
        )
        print(
            f"[data] n_tokens={_fmt_int(n_total_tokens)}",
            flush=True,
        )
        print(
            f"[model] block={_fmt_int(getattr(cfg, 'block_size', 0))} d_model={_fmt_int(getattr(cfg, 'd_model', 0))} "
            f"n_head={_fmt_int(getattr(cfg, 'n_head', 0))} head_dim={_fmt_int(head_dim)} "
            f"d_ff={_fmt_int(getattr(cfg, 'd_ff', 0))} embed_dim={_fmt_int(getattr(cfg, 'embed_dim', 0))}",
            flush=True,
        )
        print(
            f"[attn] mode={str(getattr(cfg, 'attn_mode', ''))} attn_dim={_fmt_int(getattr(cfg, 'attn_dim', 0))} "
            f"sem_dim={_fmt_int(getattr(cfg, 'sem_dim', 0))} geo_dim={_fmt_int(getattr(cfg, 'geo_dim', 0))} "
            f"rope={_fmt_int(int(bool(getattr(cfg, 'rope', False))))} tie_qk={_fmt_int(int(bool(getattr(cfg, 'tie_qk', False))))} "
            f"null_attn={_fmt_int(int(bool(getattr(cfg, 'null_attn', False))))}",
            flush=True,
        )
        print(
            f"[traincfg] steps={getattr(args, 'steps', None)} "
            f"lr={_fmt_float(getattr(args, 'lr', None))} "
            f"wd={_fmt_float(getattr(args, 'weight_decay', None))} "
            f"opt={str(getattr(args, 'optimizer', ''))} "
            f"sched={str(getattr(args, 'lr_schedule', ''))} "
            f"warmup={_fmt_int(getattr(args, 'warmup_steps', None))} "
            f"min_lr={_fmt_float(getattr(args, 'min_lr', None))}",
            flush=True,
        )
    except (OSError, UnicodeEncodeError, AttributeError, TypeError, ValueError):
        pass


