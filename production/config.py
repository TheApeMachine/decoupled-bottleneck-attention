"""Production config helpers (public API).

Why this exists:
- Many modules import `production.config` directly; keep that path stable.
- The implementation is split into `production/config_impl/` so each concern is
  small and easy to reason about.
"""

from __future__ import annotations

from production.config_impl.device import pick_device
from production.config_impl.infer import infer_dataset_tokens_from_path, infer_layers_from_out_dir
from production.config_impl.intent import apply_intent, derive_d_model, derive_n_head, derive_target_params
from production.config_impl.out_dir import default_out_dir
from production.config_impl.presets import EXP_PRESETS, apply_exp_preset
from production.config_impl.seed import set_seed
from production.config_impl.time import now_iso

__all__ = [
    "EXP_PRESETS",
    "apply_exp_preset",
    "apply_intent",
    "default_out_dir",
    "derive_d_model",
    "derive_n_head",
    "derive_target_params",
    "infer_dataset_tokens_from_path",
    "infer_layers_from_out_dir",
    "now_iso",
    "pick_device",
    "set_seed",
]


