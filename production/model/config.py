"""
Model architecture configuration (self-optimizing).

Allows the model to auto-fit the task, which makes it very easy to
run on different kinds of hardware, without having to manage all
the configuration. This helps a lot while running experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Literal
from collections.abc import Mapping

import torch

# Chinchilla scaling: optimal tokens per parameter
CHINCHILLA_TOKENS_PER_PARAM: float = 20.0


class Mode(Enum):
    """
    Mode is equivalent to the attention layout of the model. This allows us to
    compare different attention architectures without duplicating code.
    """
    BASELINE = "baseline"
    GQA = "gqa"
    BOTTLENECK = "bottleneck"
    DECOUPLED = "decoupled"


@dataclass
class ModelConfig:
    """
    ModelConfig acts as the main entry point for the model configuration, while at the
    same time grouping some of the easier self-optimization steps.
    """

    device: torch.device | str | None = field(default_factory=lambda: torch.device("cpu"))

    vocab_size: int = 0
    block_size: int = 0

    # Head selection policy:
    # - "standard": prefer common head counts (more conventional + often faster on GPUs)
    # - "any": allow any divisor of d_model (max flexibility)
    head_policy: Literal["standard", "any"] = "standard"

    n_layer: int = 0
    n_head: int = 0
    kv_head: int | None = None  # for GQA: number of KV heads (defaults to n_head)
    d_model: int = 0
    dim_multiplier: int = 0
    d_ff: int = 0

    embed_dim: int = 0  # lexical bottleneck if < d_model

    # Canonical attention mode string (legacy-friendly): {"standard","gqa","bottleneck","decoupled"}.
    # We keep it stringly-typed because configs are serialized through JSON-like dicts and the
    # public surface (tests, harnesses, ckpts) expects strings.
    attn_mode: str = "bottleneck"
    attn_dim: int = 0
    head_dim: int = 0
    sem_dim: int = 0
    geo_dim: int = 0

    decoupled_gate: bool = True
    decoupled_gate_dynamic: bool = True

    rope: bool = True
    rope_base: float = 10000.0

    tie_qk: bool = False
    null_attn: bool = False
    learned_temp: bool = True

    mlp: Literal["swiglu", "gelu"] = "swiglu"
    dropout: float = 0.0

    train_long_seq_enabled: bool = True
    train_long_seq_threshold: int | None = None
    train_long_seq_mem_block: int | None = None
    train_long_seq_local_window: int | None = None
    train_long_seq_q_chunk: int | None = None
    train_long_seq_mem_summarizer: Literal["mean", "linear", "conv"] = "conv"

    # Optional diffusion head (embedding-space denoiser conditioned on x_small).
    diffusion_head: bool = False
    diffusion_head_num_train_timesteps: int = 1000
    diffusion_head_num_infer_steps: int = 12
    diffusion_head_time_embed_dim: int = 128
    diffusion_head_mlp_mult: int = 4
    diffusion_head_cfg_dropout_p: float = 0.10
    diffusion_head_cfg_guidance_scale: float = 1.5
    diffusion_head_scheduler: str = "ddim"  # "ddpm" | "ddim" | "dpm"
    diffusion_head_loss_weight: float = 0.10

    def __post_init__(self) -> None:
        # Normalize device inputs (some checkpoints store device as string).
        dev = self.device
        if dev is None:
            self.device = torch.device("cpu")
        else:
            try:
                self.device = torch.device(dev)
            except (TypeError, ValueError):
                self.device = torch.device("cpu")

        # Normalize mode strings / enum values.
        self.attn_mode = _normalize_attn_mode(self.attn_mode)

        hp = str(self.head_policy or "standard").strip().lower()
        if hp not in ("standard", "any"):
            hp = "standard"
        self.head_policy = hp

        mlp = str(self.mlp or "swiglu").strip().lower()
        self.mlp = "gelu" if mlp == "gelu" else "swiglu"

        # Keep dropout in a sane range.
        try:
            self.dropout = float(self.dropout)
        except (TypeError, ValueError):
            self.dropout = 0.0
        if not math.isfinite(self.dropout):
            self.dropout = 0.0
        self.dropout = float(min(max(self.dropout, 0.0), 1.0))

        try:
            self.train_long_seq_enabled = bool(self.train_long_seq_enabled)
        except (TypeError, ValueError):
            self.train_long_seq_enabled = True

        def _opt_int(v: object, *, default: int | None) -> int | None:
            if v is None:
                return default
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, int):
                return int(v)
            if isinstance(v, float):
                return int(v)
            if isinstance(v, str):
                s = v.strip()
                if s == "":
                    return default
                try:
                    return int(float(s))
                except ValueError:
                    return default
            return default

        self.train_long_seq_threshold = _opt_int(self.train_long_seq_threshold, default=None)
        if self.train_long_seq_threshold is not None:
            self.train_long_seq_threshold = int(max(0, int(self.train_long_seq_threshold)))

        self.train_long_seq_mem_block = _opt_int(self.train_long_seq_mem_block, default=None)
        if self.train_long_seq_mem_block is not None:
            self.train_long_seq_mem_block = int(max(1, int(self.train_long_seq_mem_block)))

        self.train_long_seq_local_window = _opt_int(self.train_long_seq_local_window, default=None)
        if self.train_long_seq_local_window is not None:
            self.train_long_seq_local_window = int(max(0, int(self.train_long_seq_local_window)))

        self.train_long_seq_q_chunk = _opt_int(self.train_long_seq_q_chunk, default=None)
        if self.train_long_seq_q_chunk is not None:
            self.train_long_seq_q_chunk = int(max(1, int(self.train_long_seq_q_chunk)))

        mem_sum = str(self.train_long_seq_mem_summarizer).strip().lower()
        if mem_sum == "conv":
            self.train_long_seq_mem_summarizer = "conv"
        elif mem_sum == "linear":
            self.train_long_seq_mem_summarizer = "linear"
        else:
            self.train_long_seq_mem_summarizer = "mean"

        # Diffusion head normalization (safe defaults, clamp ranges).
        try:
            self.diffusion_head = bool(self.diffusion_head)
        except (TypeError, ValueError):
            self.diffusion_head = False
        try:
            self.diffusion_head_num_train_timesteps = int(max(1, int(self.diffusion_head_num_train_timesteps)))
        except (TypeError, ValueError):
            self.diffusion_head_num_train_timesteps = 1000
        try:
            self.diffusion_head_num_infer_steps = int(max(1, int(self.diffusion_head_num_infer_steps)))
        except (TypeError, ValueError):
            self.diffusion_head_num_infer_steps = 12
        try:
            self.diffusion_head_time_embed_dim = int(max(8, int(self.diffusion_head_time_embed_dim)))
        except (TypeError, ValueError):
            self.diffusion_head_time_embed_dim = 128
        try:
            self.diffusion_head_mlp_mult = int(max(1, int(self.diffusion_head_mlp_mult)))
        except (TypeError, ValueError):
            self.diffusion_head_mlp_mult = 4
        try:
            self.diffusion_head_cfg_dropout_p = float(self.diffusion_head_cfg_dropout_p)
        except (TypeError, ValueError):
            self.diffusion_head_cfg_dropout_p = 0.10
        if (not math.isfinite(self.diffusion_head_cfg_dropout_p)) or self.diffusion_head_cfg_dropout_p < 0.0:
            self.diffusion_head_cfg_dropout_p = 0.0
        self.diffusion_head_cfg_dropout_p = float(min(self.diffusion_head_cfg_dropout_p, 1.0))
        try:
            self.diffusion_head_cfg_guidance_scale = float(self.diffusion_head_cfg_guidance_scale)
        except (TypeError, ValueError):
            self.diffusion_head_cfg_guidance_scale = 1.5
        if (not math.isfinite(self.diffusion_head_cfg_guidance_scale)) or self.diffusion_head_cfg_guidance_scale < 1.0:
            self.diffusion_head_cfg_guidance_scale = 1.0
        try:
            self.diffusion_head_loss_weight = float(self.diffusion_head_loss_weight)
        except (TypeError, ValueError):
            self.diffusion_head_loss_weight = 0.10
        if (not math.isfinite(self.diffusion_head_loss_weight)) or self.diffusion_head_loss_weight < 0.0:
            self.diffusion_head_loss_weight = 0.0
        self.diffusion_head_scheduler = str(self.diffusion_head_scheduler or "ddim").strip().lower() or "ddim"

        try:
            self.decoupled_gate = bool(self.decoupled_gate)
        except (TypeError, ValueError):
            self.decoupled_gate = True
        try:
            self.decoupled_gate_dynamic = bool(self.decoupled_gate_dynamic)
        except (TypeError, ValueError):
            self.decoupled_gate_dynamic = True

    @classmethod
    def from_dict(cls, cfg: Mapping[str, object], *, device: torch.device | None = None) -> "ModelConfig":
        """Load config from a mapping, ignoring unknown keys (ckpt back-compat)."""
        def _as_int(o: object, default: int) -> int:
            if isinstance(o, bool):
                return int(o)
            if isinstance(o, int):
                return int(o)
            if isinstance(o, float):
                return int(o)
            if isinstance(o, str):
                try:
                    return int(o.strip())
                except ValueError:
                    return int(default)
            return int(default)

        def _as_float(o: object, default: float) -> float:
            if isinstance(o, bool):
                return float(int(o))
            if isinstance(o, (int, float)):
                return float(o)
            if isinstance(o, str):
                try:
                    return float(o.strip())
                except ValueError:
                    return float(default)
            return float(default)

        def _as_bool(o: object, default: bool) -> bool:
            if isinstance(o, bool):
                return bool(o)
            if isinstance(o, int):
                return bool(o != 0)
            if isinstance(o, str):
                s = o.strip().lower()
                if s in ("1", "true", "t", "yes", "y", "on"):
                    return True
                if s in ("0", "false", "f", "no", "n", "off"):
                    return False
            return bool(default)

        def _as_device(o: object, default: torch.device) -> torch.device:
            if isinstance(o, torch.device):
                return o
            if isinstance(o, str):
                try:
                    return torch.device(o)
                except (TypeError, ValueError):
                    return default
            return default

        raw = dict(cfg)
        allowed = {f.name for f in fields(cls)}
        filtered: dict[str, object] = {str(k): v for k, v in raw.items() if str(k) in allowed}

        # Construct with known-typed values (no **kwargs of object-typed dict).
        inst = cls(device=device if device is not None else _as_device(filtered.get("device", "cpu"), torch.device("cpu")))

        for k, v in filtered.items():
            if k == "device":
                # device is handled during inst creation (and may be overridden by the arg).
                continue
            match k:
                case "vocab_size":
                    inst.vocab_size = _as_int(v, 0)
                case "block_size":
                    inst.block_size = _as_int(v, 0)
                case "head_policy":
                    hp = str(v).strip().lower()
                    inst.head_policy = "any" if hp == "any" else "standard"
                case "n_layer":
                    inst.n_layer = _as_int(v, 0)
                case "n_head":
                    inst.n_head = _as_int(v, 0)
                case "kv_head":
                    inst.kv_head = None if v is None else _as_int(v, 0)
                case "d_model":
                    inst.d_model = _as_int(v, 0)
                case "dim_multiplier":
                    inst.dim_multiplier = _as_int(v, 0)
                case "d_ff":
                    inst.d_ff = _as_int(v, 0)
                case "embed_dim":
                    inst.embed_dim = _as_int(v, 0)
                case "attn_mode":
                    inst.attn_mode = _normalize_attn_mode(v)
                case "attn_dim":
                    inst.attn_dim = _as_int(v, 0)
                case "head_dim":
                    inst.head_dim = _as_int(v, 0)
                case "sem_dim":
                    inst.sem_dim = _as_int(v, 0)
                case "geo_dim":
                    inst.geo_dim = _as_int(v, 0)
                case "decoupled_gate":
                    inst.decoupled_gate = _as_bool(v, True)
                case "decoupled_gate_dynamic":
                    inst.decoupled_gate_dynamic = _as_bool(v, True)
                case "rope":
                    inst.rope = _as_bool(v, True)
                case "rope_base":
                    inst.rope_base = _as_float(v, 10000.0)
                case "tie_qk":
                    inst.tie_qk = _as_bool(v, False)
                case "null_attn":
                    inst.null_attn = _as_bool(v, False)
                case "learned_temp":
                    inst.learned_temp = _as_bool(v, True)
                case "mlp":
                    mlp = str(v).strip().lower()
                    inst.mlp = "gelu" if mlp == "gelu" else "swiglu"
                case "dropout":
                    inst.dropout = _as_float(v, 0.0)
                case "train_long_seq_enabled":
                    inst.train_long_seq_enabled = _as_bool(v, True)
                case "train_long_seq_threshold":
                    inst.train_long_seq_threshold = None if v is None else _as_int(v, 0)
                case "train_long_seq_mem_block":
                    inst.train_long_seq_mem_block = None if v is None else _as_int(v, 0)
                case "train_long_seq_local_window":
                    inst.train_long_seq_local_window = None if v is None else _as_int(v, 0)
                case "train_long_seq_q_chunk":
                    inst.train_long_seq_q_chunk = None if v is None else _as_int(v, 0)
                case "train_long_seq_mem_summarizer":
                    s = str(v).strip().lower()
                    inst.train_long_seq_mem_summarizer = "conv" if s == "conv" else ("linear" if s == "linear" else "mean")
                case "diffusion_head":
                    inst.diffusion_head = _as_bool(v, False)
                case "diffusion_head_num_train_timesteps":
                    inst.diffusion_head_num_train_timesteps = _as_int(v, 1000)
                case "diffusion_head_num_infer_steps":
                    inst.diffusion_head_num_infer_steps = _as_int(v, 12)
                case "diffusion_head_time_embed_dim":
                    inst.diffusion_head_time_embed_dim = _as_int(v, 128)
                case "diffusion_head_mlp_mult":
                    inst.diffusion_head_mlp_mult = _as_int(v, 4)
                case "diffusion_head_cfg_dropout_p":
                    inst.diffusion_head_cfg_dropout_p = _as_float(v, 0.10)
                case "diffusion_head_cfg_guidance_scale":
                    inst.diffusion_head_cfg_guidance_scale = _as_float(v, 1.5)
                case "diffusion_head_scheduler":
                    inst.diffusion_head_scheduler = str(v).strip().lower()
                case "diffusion_head_loss_weight":
                    inst.diffusion_head_loss_weight = _as_float(v, 0.10)
                case _:
                    # Unknown/unsupported key (or a field we don't want to restore).
                    pass

        # Re-run post-init normalization (for fields we set after construction).
        inst.__post_init__()
        return inst

    def optimize(self, target_params: int) -> None:
        """
        Pure calculation: derive architecture from entropy + budget deterministically.
        No search, no scoring—just math.
        """
        # Step 1: Depth from capacity (deterministic formula), unless already specified.
        # Why: paper harness encodes depth in run_id/out_dir (e.g. `_l22_...`) and we
        # want that to remain a first-class intent without adding more CLI flags.
        if int(self.n_layer) > 0:
            n_layer: int = int(self.n_layer)
        else:
            log_params: float = math.log10(float(max(1, target_params)))
            # 1M→4, 5M→6, 10M→8, 50M→10, 100M→12, 500M→14, 1B→16
            n_layer = max(4, int(2 + 2 * log_params - 11.0))

        # Step 2: Entropy signals (data-driven ratios)
        v: float = math.log2(float(self.vocab_size) + 1.0)
        b: float = math.log2(float(self.block_size) + 1.0)
        denom: float = v + b
        if (not math.isfinite(denom)) or abs(denom) <= 1e-12:
            task = 0.5
            ctx = 0.5
        else:
            task = v / denom
            ctx = b / denom

        target_head_dim: int = int(round(64.0 + 64.0 * task))
        mlp_ratio: float = 2.5 + 3.5 * task
        attn_ratio: float = 0.55 + 0.45 * task

        # Step 3: Solve for d_model given budget
        # Prefer less aggressive embed compression for better quality
        mlp_mult: float = 3.0   # SwiGLU param multiplier
        attn_mult: float = 4.0  # q,k,v,o param multiplier

        solution = None
        for embed_ratio in [0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:  # Start higher
            # Try both tied and untied lm_head
            for tie_lm in [True, False]:
                result = self._solve_width(
                    target_params=target_params,
                    n_layer=n_layer,
                    embed_ratio=embed_ratio,
                    attn_ratio=attn_ratio,
                    mlp_ratio=mlp_ratio,
                    mlp_mult=mlp_mult,
                    attn_mult=attn_mult,
                    tie_lm=tie_lm,
                )
                if result is not None:
                    solution = result
                    break
            if solution is not None:
                break

        if solution is None:
            msg = (
                f"Auto-fit failed: no feasible solution for target_params={target_params}, "
                f"vocab_size={self.vocab_size}, block_size={self.block_size}"
            )
            raise ValueError(msg)

        d_model, embed_dim, attn_dim, d_ff = solution

        # Step 4: Heads from width (deterministic)
        n_head: int = self._pick_n_head(d_model, target_head_dim)
        head_dim: int = d_model // n_head

        # Fix attn_dim to be compatible with n_head (solver doesn't know n_head yet)
        # For decoupled+RoPE, need both splits divisible by n_head
        if _normalize_attn_mode(self.attn_mode) == "decoupled" and bool(self.rope):
            attn_dim = self._round_to_multiple(attn_dim, 2 * n_head)
        else:
            attn_dim = self._round_to_multiple(attn_dim, n_head)

        # Step 5: Mode-specific splits (structural)
        if _normalize_attn_mode(self.attn_mode) == "decoupled":
            geo_share: float = ctx * 0.45
            geo_dim: int = int(round(attn_dim * geo_share))
            geo_dim = self._round_to_multiple(geo_dim, n_head)
            if bool(self.rope):
                geo_dim = self._round_to_multiple(geo_dim, 2 * n_head)
            sem_dim: int = attn_dim - geo_dim
        else:
            geo_dim = 0
            sem_dim = attn_dim

        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.head_dim = head_dim
        self.attn_dim = attn_dim
        self.d_ff = d_ff
        self.embed_dim = embed_dim
        self.sem_dim = sem_dim
        self.geo_dim = geo_dim
        self.kv_head = self.kv_head_for_mode()

    def _solve_width(
        self,
        *,
        target_params: int,
        n_layer: int,
        embed_ratio: float,
        attn_ratio: float,
        mlp_ratio: float,
        mlp_mult: float,
        attn_mult: float,
        tie_lm: bool,
    ) -> tuple[int, int, int, int] | None:
        """
        Solve for d_model given budget and ratios.

        P_total ≈ vocab*embed_dim + embed_proj + n_layer*P_block + lm_head

        Where:
          embed_dim = embed_ratio * d_model
          attn_dim = attn_ratio * d_model
          d_ff = mlp_ratio * d_model
          P_block = attn_mult * d_model * attn_dim + mlp_mult * d_model * d_ff

        This is quadratic in d_model; we can solve or binary search.
        """
        vocab: int = self.vocab_size

        # Coefficients for quadratic: a*d² + b*d + c = 0, where we want P_total = target_params
        # P_embed = vocab * (embed_ratio * d) + (embed_ratio * d) * d = vocab*embed_ratio*d + embed_ratio*d²
        # P_block = attn_mult*d*(attn_ratio*d) + mlp_mult*d*(mlp_ratio*d) = (attn_mult*attn_ratio + mlp_mult*mlp_ratio)*d²
        # P_lm = vocab*d if not tied, else 0

        a: float = embed_ratio + n_layer * (attn_mult * attn_ratio + mlp_mult * mlp_ratio)
        b: float = vocab * embed_ratio + (0 if tie_lm else vocab)
        c: float = -float(target_params)

        # Quadratic formula
        discriminant: float = b * b - 4 * a * c
        if discriminant < 0:
            return None

        d_model_raw: float = (-b + math.sqrt(discriminant)) / (2 * a)
        if d_model_raw < 32:
            return None

        # Round to a device-appropriate multiple to ensure sane head divisors.
        # Why: if d_model is "prime-ish" (few divisors), `_pick_n_head` can only
        # choose extreme head counts, producing pathological head_dim (e.g. 8).
        d_model_multiple: int = 8
        try:
            if str(getattr(self.device, "type", "cpu")) == "cuda":
                d_model_multiple = 128
            elif str(getattr(self.device, "type", "cpu")) == "mps":
                d_model_multiple = 64
        except (AttributeError, TypeError, ValueError):
            d_model_multiple = 8

        d_model: int = self._round_to_multiple(int(d_model_raw), int(d_model_multiple))
        embed_dim: int = self._round_to_multiple(max(32, int(d_model * embed_ratio)), 8)
        attn_dim: int = self._round_to_multiple(int(d_model * attn_ratio), 4)  # divisible by heads later
        d_ff: int = self._round_to_multiple(int(d_model * mlp_ratio), 8)

        # Verify it fits budget
        p_check = self._estimate_embed_params(vocab, d_model, embed_dim, tie_lm) + n_layer * self._estimate_block_params(d_model, attn_dim, d_ff, mlp_mult, attn_mult)
        if p_check > target_params * 1.15:  # Allow 15% overshoot
            return None

        return (d_model, embed_dim, attn_dim, d_ff)

    def _round_to_multiple(self, x: int, m: int) -> int:
        return max(m, (x // m) * m)

    def _pick_n_head(self, d_model: int, target_head_dim: int) -> int:
        divs = [h for h in range(1, d_model + 1) if d_model % h == 0]
        if not divs:
            return 1

        if self.head_policy == "any":
            return min(divs, key=lambda h: abs((d_model // h) - target_head_dim))

        # "standard": prefer a conventional set when feasible; otherwise fall back to any divisor.
        preferred = [4, 6, 8, 12, 16, 24, 32, 40, 48, 64]
        preferred_divs = [h for h in preferred if d_model % h == 0]
        candidates = preferred_divs if preferred_divs else divs
        return min(candidates, key=lambda h: abs((d_model // h) - target_head_dim))

    def _estimate_embed_params(self, vocab: int, d_model: int, embed_dim: int, tie_lm_head: bool) -> int:
        p = vocab * embed_dim
        if embed_dim != d_model:
            p += embed_dim * d_model
        if not tie_lm_head:
            p += vocab * d_model
        return int(p)

    def _estimate_block_params(self, d_model: int, attn_dim: int, d_ff: int, mlp_mult: float, attn_mult: float) -> int:
        p_attn = attn_mult * d_model * attn_dim
        p_mlp = mlp_mult * d_model * d_ff
        return int(p_attn + p_mlp)

    def kv_head_for_mode(self) -> int | None:
        """GQA reduces KV bandwidth by sharing KV across query heads when it is safe."""
        if _normalize_attn_mode(self.attn_mode) != "gqa":
            return None
        # Heuristic: reduce KV heads when the context is long relative to depth (attention is dominated by bandwidth).
        try:
            denom = max(1, int(self.block_size).bit_length() // max(1, int(self.n_layer)))
        except (TypeError, ValueError, AttributeError):
            denom = 1
        kv = int(max(1, int(self.n_head) // int(denom)))
        # Ensure kv_head divides n_head.
        nh = int(max(1, int(self.n_head)))
        while kv > 1 and (nh % kv) != 0:
            kv -= 1
        return int(max(1, kv))


def _normalize_attn_mode(mode: object) -> str:
    """Best-effort normalization of attention mode inputs (strings, enums)."""
    v = getattr(mode, "value", mode)
    # Treat `None` as "unset"; preserve falsy-but-meaningful values (0/False) by not using `v or ""`.
    s = "" if v is None else str(v).strip().lower()
    if s == "":
        return "bottleneck"
    if s in ("baseline", "standard", "base"):
        return "standard"
    if s in ("gqa", "bottleneck", "decoupled"):
        return s
    raise ValueError(
        f'Unknown attn_mode={v!r} (normalized={s!r}). Accepted aliases: ("standard"/"baseline"/"base"), ("gqa"/"bottleneck"/"decoupled").'
    )
