"""Training-default derivations for the intent-first workflow.

Why this exists:
- Runner code expects a filled-out training config.
- We keep the CLI small by populating required fields here, using heuristics
  that are consistent and reproducible.
"""

from __future__ import annotations

import argparse
import os
from typing import cast

from production.optimizer.counts import CountCodec
from production.optimizer.heuristics import HeuristicPlanner
from production.optimizer.result_types import ResultTypeMapper
from production.optimizer.watch import OptimizerLike


class TrainingDefaultsDeriver:
    """Derive architecture + training defaults (only for train mode)."""

    @staticmethod
    def apply(o: OptimizerLike, *, args: argparse.Namespace) -> None:
        """Why: downstream training code needs these fields even when CLI doesn't expose them."""
        exp0 = o.get("exp")
        dtok_raw = o.get("dataset_tokens")
        dtok = int(cast(int, dtok_raw)) if dtok_raw is not None else None
        target_params = int(cast(int, o.get("target_params")))
        L = int(cast(int, o.get("layers")))  # pylint: disable=invalid-name

        if not hasattr(args, "tokenizer"):
            data_path = str(o.get("data") or "")
            base = os.path.basename(data_path).lower()
            tok = "tiktoken" if ("fineweb" in base or base.endswith(".tokens")) else "raw"
            o.set("tokenizer", tok)
        if not hasattr(args, "val_frac"):
            o.set("val_frac", 0.1)
        if not hasattr(args, "data_format"):
            o.set("data_format", "auto")
        if not hasattr(args, "data_dtype"):
            o.set("data_dtype", "int64")

        # Model fields are intentionally set to 0/"" so the model config can self-optimize.
        for name, val in [
            ("layers", int(L)),
            ("d_model", 0),
            ("n_head", 0),
            ("d_ff", 0),
            ("embed_dim", 0),
            ("attn_mode", ""),
            ("attn_dim", 0),
            ("sem_dim", 0),
            ("geo_dim", 0),
            ("kv_head", None),
            ("rope_base", 10000.0),
            ("tie_qk", False),
            ("null_attn", False),
            ("no_rope", False),
            ("no_decoupled_gate", False),
            ("no_learned_temp", False),
            ("mlp", "swiglu"),
            ("dropout", 0.0),
        ]:
            if not hasattr(args, name):
                o.set(name, val)

        # Block size selection is device- and scale-aware: keeps early runs feasible.
        if not hasattr(args, "block"):
            if str(o.get("device_type")) == "cuda":
                block = 2048
            elif str(o.get("device_type")) == "mps":
                block = 512 if target_params < 50_000_000 else 1024
            else:
                block = 32 if target_params < 50_000_000 else (256 if (dtok is None or int(dtok) <= 200_000_000) else 512)
            o.set("block", int(block))

        # Optimizer/lr defaults are scale-aware (keeps novice configs from blowing up).
        if not hasattr(args, "optimizer"):
            o.set("optimizer", "adamw")
        if not hasattr(args, "weight_decay"):
            o.set("weight_decay", 0.1)
        if not hasattr(args, "lr"):
            o.set("lr", float(HeuristicPlanner.derive_lr_from_params(target_params)))
        if not hasattr(args, "min_lr"):
            o.set("min_lr", float(cast(float, o.get("lr"))) * 0.1)
        if not hasattr(args, "lr_schedule"):
            o.set("lr_schedule", "cosine")
        if not hasattr(args, "warmup_steps"):
            o.set("warmup_steps", 0)
        if not hasattr(args, "adam_eps"):
            o.set("adam_eps", 1e-8)
        if not hasattr(args, "adam_betas"):
            o.set("adam_betas", "0.9,0.95")
        if not hasattr(args, "lion_betas"):
            o.set("lion_betas", "0.9,0.99")
        if not hasattr(args, "opt_foreach"):
            o.set("opt_foreach", False)
        if not hasattr(args, "opt_fused"):
            o.set("opt_fused", False)
        if not hasattr(args, "eval_iters"):
            o.set("eval_iters", 20)
        if not hasattr(args, "eval_every"):
            o.set("eval_every", 0)
        if not hasattr(args, "save_every"):
            o.set("save_every", 0)
        if not hasattr(args, "log_every"):
            o.set("log_every", 0)

        # Stable run-id naming keeps artifacts searchable without extra flags.
        if not o.get("out_dir") and exp0:
            rt = ResultTypeMapper.normalize(str(exp0))
            dtok_s = CountCodec.format(int(dtok)) if dtok is not None else "data"
            p_s = CountCodec.format(int(target_params))
            dev_s = str(o.get("device_type"))
            layers = int(L)  # pylint: disable=invalid-name
            seed = int(cast(int, o.get("seed")))
            run_id = f"{dev_s}_{dtok_s}_{p_s}_l{layers}_{rt}_s{seed}"
            o.set("out_dir", os.path.join("runs", run_id))


