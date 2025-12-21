"""
Intent → derived architecture fields.

This module is the primary way we configure runs: we express high-level intent
(dataset scale / target parameter budget, and depth) and derive concrete
architecture fields (width, heads, MLP size, embedding width) for the runner.

Note: this runs early in the CLI pipeline (before tokenization/model build),
so it intentionally avoids depending on runtime-only facts like `vocab_size`.
"""
from __future__ import annotations

import argparse
import math
from typing import cast

from production.config_impl.infer import infer_dataset_tokens_from_path, infer_layers_from_out_dir

# Chinchilla-style default: optimal tokens ≈ 20 × params.
_TOKENS_PER_PARAM: float = 20.0

# GPT-style: d_ff = 4 × d_model.
_MLP_RATIO: int = 4

# Rough param model for GPT blocks (QKV+O + MLP) with MLP ratio 4: ~ 12 * d_model^2 per layer.
_PARAMS_PER_LAYER_COEFF: float = 12.0


def _round_up(x: float, multiple: int) -> int:
    """Why: keep dims aligned for matmul efficiency and divisibility constraints."""
    multiple = int(max(1, multiple))
    return int(math.ceil(float(x) / float(multiple)) * multiple)


def derive_target_params(dataset_tokens: int) -> int:
    """Why: convert dataset scale into a single parameter budget scalar."""
    return int(max(1.0, float(dataset_tokens) / float(_TOKENS_PER_PARAM)))


def derive_d_model(*, layers: int, target_params: int, multiple: int = 128, min_d_model: int = 256) -> int:
    """Why: choose width so the approximate parameter count matches the target budget."""
    layers = int(max(1, layers))
    target_params = int(max(1, target_params))
    raw = math.sqrt(float(target_params) / (float(_PARAMS_PER_LAYER_COEFF) * float(layers)))
    return int(max(int(min_d_model), _round_up(raw, multiple)))


def derive_n_head(d_model: int) -> int:
    """Why: pick a head count that makes head_dim "nice" while dividing d_model exactly."""
    d_model = int(max(1, d_model))
    want = max(1, int(round(d_model / 128.0)))
    for delta in range(0, 64):
        for cand in (want - delta, want + delta):
            if cand >= 1 and d_model % cand == 0:
                return int(cand)
    for cand in range(1, d_model + 1):
        if d_model % cand == 0:
            return int(cand)
    return 1


def _maybe_positive_int(value: object | None) -> int | None:
    """Parse a positive int from CLI-ish values (str/int/float)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        iv = int(value)
        return iv if iv > 0 else None
    if isinstance(value, str):
        try:
            iv = int(value)
        except ValueError:
            return None
        return iv if iv > 0 else None
    # Last-resort: try parsing from its string form.
    try:
        iv = int(str(value))
    except ValueError:
        return None
    return iv if iv > 0 else None


def apply_intent(args: argparse.Namespace) -> None:
    """Fill architecture fields from *intent* (exp + layers + dataset scale).

    Why:
    - Downstream model building needs concrete dims.
    - We keep CLI flags minimal by inferring missing values from out_dir and data naming.
    """
    # 1) Resolve dataset token scale (or explicit target params).
    target_params = _maybe_positive_int(cast(object, getattr(args, "target_params", None)))

    data_obj = cast(object, getattr(args, "data", None))
    data_path = str(data_obj) if data_obj is not None else None

    dataset_tokens = _maybe_positive_int(cast(object, getattr(args, "dataset_tokens", None)))

    if target_params is None and data_path is not None:
        dataset_tokens = (
            dataset_tokens if dataset_tokens is not None else infer_dataset_tokens_from_path(str(data_path))
        )
        if dataset_tokens is None:
            raise ValueError(
                f"Could not infer dataset token scale from data path {data_path!r}. "
                + "Use a name like `fineweb_20b.npy` or provide a sibling `.meta` with `tokens: ...`."
            )
        target_params = derive_target_params(int(dataset_tokens))

    # 2) Resolve layers.
    layers_raw = cast(object, getattr(args, "layers", None))
    if layers_raw in (None, 0, "0"):
        out_dir_obj = cast(object, getattr(args, "out_dir", None))
        out_dir_path = str(out_dir_obj) if out_dir_obj is not None else None

        layers_guess = infer_layers_from_out_dir(out_dir_path) if out_dir_path else None
        if layers_guess is None:
            if target_params is None:
                raise ValueError(
                    "Need either --layers, --size/target_params, or a dataset token scale to infer layers."
                )

            # Why: prefer ~128-dim heads by choosing depth that yields a clean divisor layout.
            if int(target_params) < 50_000_000:
                candidates = (2, 4, 6, 8, 12)
                min_d_model = 64
                multiple = 64
            elif int(target_params) < 500_000_000:
                candidates = (8, 12, 16, 20, 22, 24, 32)
                min_d_model = 256
                multiple = 128
            else:
                candidates = (12, 16, 20, 22, 24, 32, 48, 64, 96)
                min_d_model = 256
                multiple = 128

            best_L = int(candidates[0])
            best_score = float("inf")
            for cand in candidates:
                d_model = derive_d_model(
                    layers=int(cand),
                    target_params=int(target_params),
                    multiple=multiple,
                    min_d_model=min_d_model,
                )
                n_head = derive_n_head(int(d_model))
                head_dim = float(d_model) / max(1.0, float(n_head))
                score = abs(head_dim - 128.0) / 128.0
                if score < best_score:
                    best_score = score
                    best_L = int(cand)
            layers_guess = int(best_L)

        args.layers = int(layers_guess)

    # If we're missing data (e.g. sample mode), there's nothing else to infer here.
    if data_path is None:
        return

    if target_params is None:
        dtok = infer_dataset_tokens_from_path(str(data_path))
        if dtok is None:
            return
        target_params = derive_target_params(int(dtok))

    if _maybe_positive_int(cast(object, getattr(args, "d_model", None))) is None:
        layers = _maybe_positive_int(cast(object, getattr(args, "layers", None))) or 1
        args.d_model = derive_d_model(layers=layers, target_params=int(target_params))

    if _maybe_positive_int(cast(object, getattr(args, "n_head", None))) is None:
        d_model = _maybe_positive_int(cast(object, getattr(args, "d_model", None))) or 1
        args.n_head = derive_n_head(d_model)

    if _maybe_positive_int(cast(object, getattr(args, "d_ff", None))) is None:
        d_model = _maybe_positive_int(cast(object, getattr(args, "d_model", None))) or 1
        args.d_ff = int(_MLP_RATIO) * d_model

    if _maybe_positive_int(cast(object, getattr(args, "embed_dim", None))) is None:
        d_model = _maybe_positive_int(cast(object, getattr(args, "d_model", None))) or 1
        args.embed_dim = d_model


