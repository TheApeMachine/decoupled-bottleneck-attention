"""Human-readable run summary.

Why this exists:
- The CLI is intentionally minimal; we still want visibility into what was derived.
- A compact, stable printout makes debugging and experiment tracking easier.
"""

from __future__ import annotations

import importlib
import os
import traceback
from typing import Protocol, runtime_checkable

from production.attention_impl.decoupled_attention_impl.triton_runtime import TRITON_AVAILABLE
from production.console import get_console, rich_enabled
from production.runtime_tuning import KVCachePolicy


def _get_int_attr(o: object, name: str, default: int) -> int:
    v = getattr(o, name, default)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return int(default)
    return int(default)


def _fmt_int(x: object) -> str:
    try:
        if isinstance(x, bool):
            return f"{int(x):_d}"
        if isinstance(x, int):
            return f"{int(x):_d}"
        if isinstance(x, float):
            return f"{int(x):_d}"
        if isinstance(x, str):
            return f"{int(x):_d}"
        return "?"
    except (ValueError, TypeError):
        return "?"


def _fmt_float(x: object) -> str:
    try:
        if isinstance(x, bool):
            return f"{float(int(x)):.3g}"
        if isinstance(x, (int, float)):
            return f"{float(x):.3g}"
        if isinstance(x, str):
            return f"{float(x):.3g}"
        return "?"
    except (ValueError, TypeError):
        return "?"


@runtime_checkable
class _RichTable(Protocol):
    def add_column(self, *args: object, **kwargs: object) -> object: ...

    def add_row(self, *args: object, **kwargs: object) -> object: ...


def _debug_enabled() -> bool:
    v = str(os.environ.get("SUMMARY_DEBUG", "")).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    v2 = str(os.environ.get("DEBUG", "")).strip().lower()
    return v2 in ("1", "true", "yes", "on")


def print_summary(*, args: object, device: object, cfg: object, n_total_tokens: int) -> None:
    """Why: show the key derived values up front so misconfigurations are obvious."""
    try:
        console = get_console()

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
            head_dim = _get_int_attr(cfg, "d_model", 0) // max(1, _get_int_attr(cfg, "n_head", 1))
        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
            head_dim = None

        kv_policy_s = "-"
        kv_policy_raw = getattr(args, "kv_policy", None)
        if kv_policy_raw is not None and str(kv_policy_raw).strip() != "":
            try:
                kv_policy_s = KVCachePolicy.parse(str(kv_policy_raw)).short()
            except (TypeError, ValueError):
                kv_policy_s = str(kv_policy_raw)

        null_enabled = bool(getattr(cfg, "null_attn", False))
        if not null_enabled:
            null_state = "Inactive"
        else:
            is_decoupled = str(getattr(cfg, "attn_mode", "")).strip().lower() == "decoupled"
            dev_type = str(getattr(device, "type", str(device))).strip().lower()
            fused_capable = bool(is_decoupled and TRITON_AVAILABLE and dev_type == "cuda")
            null_state = "Active (Fused-capable)" if fused_capable else "Active (Unfused-only)"

        # Prefer rich table if available (pretty + compact), otherwise plain text lines.
        if rich_enabled():
            try:
                mod = importlib.import_module("rich.table")
                table_ctor = getattr(mod, "Table", None)
                if callable(table_ctor):
                    t_obj = table_ctor(title="Run Summary", show_header=False)
                    if not isinstance(t_obj, _RichTable):
                        raise TypeError("rich.Table does not match expected interface")
                    t = t_obj
                    # `Table` is dynamic; keep calls best-effort and rely on runtime.
                    _ = t.add_column("Key", style="bold", no_wrap=True)
                    _ = t.add_column("Value")
                    _ = t.add_row(
                        "intent",
                        (
                            f"device={str(device)}  exp={exp_s or '?'} ({exp_src or '?'})  "
                            f"data={os.path.basename(data_s) or '?'}  out_dir={out_s or '?'}\n"
                            f"dataset_tokens={_fmt_int(ds_tok)} ({ds_src or '?'})  "
                            f"target_params={_fmt_int(tp)} ({tp_src or '?'})  "
                            f"layers={_fmt_int(getattr(cfg, 'n_layer', 0))} ({layers_src or '?'})"
                        ),
                    )
                    _ = t.add_row("data", f"n_tokens={_fmt_int(n_total_tokens)}")
                    _ = t.add_row(
                        "model",
                        (
                            f"block={_fmt_int(getattr(cfg, 'block_size', 0))}  "
                            f"d_model={_fmt_int(getattr(cfg, 'd_model', 0))}  "
                            f"n_head={_fmt_int(getattr(cfg, 'n_head', 0))}  "
                            f"head_dim={_fmt_int(head_dim)}  "
                            f"d_ff={_fmt_int(getattr(cfg, 'd_ff', 0))}  "
                            f"embed_dim={_fmt_int(getattr(cfg, 'embed_dim', 0))}"
                        ),
                    )
                    _ = t.add_row(
                        "attn",
                        (
                            f"mode={str(getattr(cfg, 'attn_mode', ''))}  "
                            f"attn_dim={_fmt_int(getattr(cfg, 'attn_dim', 0))}  "
                            f"sem_dim={_fmt_int(getattr(cfg, 'sem_dim', 0))}  "
                            f"geo_dim={_fmt_int(getattr(cfg, 'geo_dim', 0))}  "
                            f"rope={_fmt_int(int(bool(getattr(cfg, 'rope', False))))}  "
                            f"tie_qk={_fmt_int(int(bool(getattr(cfg, 'tie_qk', False))))}  "
                            f"null_attn={_fmt_int(int(bool(getattr(cfg, 'null_attn', False))))}"
                        ),
                    )
                    _ = t.add_row("kv_policy", kv_policy_s)
                    _ = t.add_row("null_attention", null_state)
                    _ = t.add_row(
                        "traincfg",
                        (
                            f"steps={_fmt_int(getattr(args, 'steps', None))}  "
                            f"lr={_fmt_float(getattr(args, 'lr', None))}  "
                            f"wd={_fmt_float(getattr(args, 'weight_decay', None))}  "
                            f"opt={str(getattr(args, 'optimizer', ''))}  "
                            f"sched={str(getattr(args, 'lr_schedule', ''))}  "
                            f"warmup={_fmt_int(getattr(args, 'warmup_steps', None))}  "
                            f"min_lr={_fmt_float(getattr(args, 'min_lr', None))}"
                        ),
                    )
                    console.print(t)
                    return
            except (ImportError, AttributeError, TypeError, ValueError):
                pass

        console.print(
            f"[intent] device={str(device)} exp={exp_s or '?'} ({exp_src or '?'})  dataset_tokens={_fmt_int(ds_tok)} ({ds_src or '?'}) target_params={_fmt_int(tp)} ({tp_src or '?'}) layers={_fmt_int(getattr(cfg, 'n_layer', 0))} ({layers_src or '?'}) out_dir={out_s or '?'} data={os.path.basename(data_s) or '?'}",
            flush=True,
        )
        console.print(f"[data] n_tokens={_fmt_int(n_total_tokens)}", flush=True)
        console.print(f"[model] block={_fmt_int(getattr(cfg, 'block_size', 0))} d_model={_fmt_int(getattr(cfg, 'd_model', 0))} n_head={_fmt_int(getattr(cfg, 'n_head', 0))} head_dim={_fmt_int(head_dim)} d_ff={_fmt_int(getattr(cfg, 'd_ff', 0))} embed_dim={_fmt_int(getattr(cfg, 'embed_dim', 0))}", flush=True)
        console.print(f"[attn] mode={str(getattr(cfg, 'attn_mode', ''))} attn_dim={_fmt_int(getattr(cfg, 'attn_dim', 0))} sem_dim={_fmt_int(getattr(cfg, 'sem_dim', 0))} geo_dim={_fmt_int(getattr(cfg, 'geo_dim', 0))} rope={_fmt_int(int(bool(getattr(cfg, 'rope', False))))} tie_qk={_fmt_int(int(bool(getattr(cfg, 'tie_qk', False))))} null_attn={_fmt_int(int(bool(getattr(cfg, 'null_attn', False))))}", flush=True)
        console.print(f"[kv] policy={kv_policy_s}", flush=True)
        console.print(f"[null] attention={null_state}", flush=True)
        console.print(f"[traincfg] steps={_fmt_int(getattr(args, 'steps', None))} lr={_fmt_float(getattr(args, 'lr', None))} wd={_fmt_float(getattr(args, 'weight_decay', None))} opt={str(getattr(args, 'optimizer', ''))} sched={str(getattr(args, 'lr_schedule', ''))} warmup={_fmt_int(getattr(args, 'warmup_steps', None))} min_lr={_fmt_float(getattr(args, 'min_lr', None))}", flush=True)
    except (OSError, UnicodeEncodeError, AttributeError, TypeError, ValueError) as e:
        try:
            console = get_console()
            console.print(f"[summary] Non-fatal error while printing run summary: {e!r}", flush=True)
            if _debug_enabled():
                console.print(traceback.format_exc(), flush=True)
        except (OSError, UnicodeEncodeError, AttributeError, TypeError, ValueError):
            return

