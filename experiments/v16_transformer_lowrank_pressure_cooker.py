
import math
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS backend.")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    print("Using CPU backend.")
    return torch.device("cpu")


# ============================================================
# LOGGING
# ============================================================

class JsonlLogger:
    """Overwrites file on start; JSONL events for deep debugging."""
    def __init__(self, path: Optional[str]):
        self._fh = None
        if path:
            self._fh = open(path, "w", encoding="utf-8")

    def log(self, event: Dict[str, Any]):
        if not self._fh:
            return
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ============================================================
# DATA
# ============================================================

def load_token_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split()

def load_char_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return list(f.read())

def build_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(set(tokens))
    stoi = {t: i for i, t in enumerate(uniq)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos

def encode(tokens: List[str], stoi: Dict[str, int], device: torch.device) -> torch.Tensor:
    ids = [stoi[t] for t in tokens]
    return torch.tensor(ids, dtype=torch.long, device=device)


# ============================================================
# SPECTRAL RANK CONTROL
# ============================================================

class SpectralRankController:
    """
    Maintains an integer rank r. Given svals(G), pick the smallest r that captures
    energy_target of ||G||_F^2, then move toward it with step_max (aggressive when large).
    """
    def __init__(self, init_rank: int, min_rank: int, max_rank: int, energy_target: float, ema_decay: float, step_max: int):
        self.rank = int(init_rank)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)
        self.energy_target = float(energy_target)
        self.ema_decay = float(ema_decay)
        self.step_max = int(step_max)
        self._ema_rank = float(init_rank)

    def set_energy_target(self, x: float):
        self.energy_target = float(x)

    def set_step_max(self, x: int):
        self.step_max = int(x)

    def suggest_rank(self, svals: torch.Tensor) -> int:
        if svals.numel() == 0:
            return self.min_rank
        s2 = (svals.float() ** 2)
        total = float(s2.sum().item())
        if total <= 0:
            return self.min_rank
        frac = torch.cumsum(s2, dim=0) / total
        k = int(torch.searchsorted(frac, torch.tensor(self.energy_target)).item()) + 1
        return int(max(self.min_rank, min(self.max_rank, k)))

    def update(self, suggested: int) -> int:
        self._ema_rank = self.ema_decay * self._ema_rank + (1 - self.ema_decay) * float(suggested)
        target = int(round(self._ema_rank))
        target = max(self.min_rank, min(self.max_rank, target))

        delta = target - self.rank
        if delta == 0:
            return self.rank

        step = int(max(-self.step_max, min(self.step_max, delta)))
        self.rank = int(max(self.min_rank, min(self.max_rank, self.rank + step)))
        return self.rank


@torch.no_grad()
def svdvals_cpu(x: torch.Tensor) -> torch.Tensor:
    m = x.detach()
    if m.dtype not in (torch.float32, torch.float64):
        m = m.float()
    return torch.linalg.svdvals(m.cpu())


class VirtualLowRankLinear(nn.Module):
    """
    "Virtual" low-rank linear that keeps fixed-size params so AdamW momentum never resets:
      U_full: [out, max_rank]
      V_full: [max_rank, in]
    Active rank r just slices [:r].

    Rank updates are lazy: every svd_interval steps, we estimate singular values of a proxy
    for dL/dW and update r via a spectral controller.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_rank: int,
        max_rank: int,
        min_rank: int,
        energy_target: float,
        rank_step_max: int,
        svd_interval: int,
        svd_interval_min: int,
        svd_interval_max: int,
        svd_growth: float,
        svd_shrink: float,
        tail_energy_low: float,
        tail_energy_high: float,
        tag: str,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.max_rank = int(max_rank)
        self.min_rank = int(min_rank)
        self.tag = str(tag)

        init_rank = int(max(self.min_rank, min(self.max_rank, init_rank)))

        self.controller = SpectralRankController(
            init_rank=init_rank,
            min_rank=self.min_rank,
            max_rank=self.max_rank,
            energy_target=energy_target,
            ema_decay=0.2,
            step_max=rank_step_max,
        )

        self.U_full = nn.Parameter(torch.randn(self.out_features, self.max_rank) * 0.02)
        self.V_full = nn.Parameter(torch.randn(self.max_rank, self.in_features) * 0.02)

        self._counter = 0
        self.svd_interval = int(svd_interval)

        self.svd_interval_min = int(svd_interval_min)
        self.svd_interval_max = int(svd_interval_max)
        self.svd_growth = float(svd_growth)
        self.svd_shrink = float(svd_shrink)

        self.tail_energy_low = float(tail_energy_low)
        self.tail_energy_high = float(tail_energy_high)

    @property
    def rank(self) -> int:
        return int(self.controller.rank)

    def set_pressure(self, energy_target: float, rank_step_max: int, svd_interval_min: int):
        self.controller.set_energy_target(energy_target)
        self.controller.set_step_max(rank_step_max)
        self.svd_interval_min = int(svd_interval_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.rank
        U = self.U_full[:, :r]
        V = self.V_full[:r, :]
        return x @ V.t() @ U.t()

    def maybe_svd_rank_update(self, logger: Optional[JsonlLogger]):
        self._counter += 1
        if self._counter < self.svd_interval:
            return
        self._counter = 0

        r = self.rank
        if self.U_full.grad is None or self.V_full.grad is None:
            return

        dU = self.U_full.grad[:, :r]
        dV = self.V_full.grad[:r, :]
        U = self.U_full[:, :r]
        V = self.V_full[:r, :]

        # Dense proxy for dL/dW (out x in)
        G = dU @ V + U @ dV
        svals = svdvals_cpu(G)

        suggested = self.controller.suggest_rank(svals)
        old = self.rank
        new = self.controller.update(suggested)

        # tail energy beyond new
        s2 = (svals.float() ** 2)
        total = float(s2.sum().item()) if s2.numel() else 0.0
        tail = 0.0
        if total > 0 and new < s2.numel():
            tail = float(s2[new:].sum().item() / total)

        # adapt interval based on tail energy
        if tail < self.tail_energy_low:
            self.svd_interval = min(int(self.svd_interval * self.svd_growth), self.svd_interval_max)
        elif tail > self.tail_energy_high:
            self.svd_interval = max(int(self.svd_interval * self.svd_shrink), self.svd_interval_min)

        if logger:
            logger.log({
                "type": "rank_update",
                "tag": self.tag,
                "shape": [self.out_features, self.in_features],
                "old_rank": old,
                "suggested": int(suggested),
                "new_rank": int(new),
                "tail_energy": tail,
                "svd_interval_next": int(self.svd_interval),
            })


# ============================================================
# HEAD + BLOCK CONTROLLERS
# ============================================================

class HeadController(nn.Module):
    """
    Learnable head gates (sigmoid) plus optional hard-prune when gate < threshold.
    """
    def __init__(self, n_heads: int, prune_threshold: float):
        super().__init__()
        self.n_heads = int(n_heads)
        self.prune_threshold = float(prune_threshold)
        self.head_logit = nn.Parameter(torch.ones(self.n_heads) * 2.0)  # sigmoid ~0.88
        self.register_buffer("active", torch.ones(self.n_heads, dtype=torch.bool))

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.head_logit)

    def active_indices(self) -> torch.Tensor:
        return torch.nonzero(self.active, as_tuple=False).squeeze(-1)

    def reg(self) -> torch.Tensor:
        return self.gates().sum()

    @torch.no_grad()
    def maybe_prune(self, logger: Optional[JsonlLogger], tag_prefix: str):
        g = self.gates().detach()
        to_prune = (g < self.prune_threshold) & self.active
        if not to_prune.any():
            return
        pruned = torch.nonzero(to_prune, as_tuple=False).squeeze(-1).tolist()
        self.active[to_prune] = False
        if logger:
            logger.log({"type": "head_prune", "tag": tag_prefix, "pruned_heads": pruned})


class BlockController(nn.Module):
    """
    Learnable scalar gate for the whole block. While active, block residuals are scaled by gate.
    Can hard-prune when gate < threshold.
    """
    def __init__(self, prune_threshold: float):
        super().__init__()
        self.prune_threshold = float(prune_threshold)
        self.block_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid ~0.88
        self.register_buffer("active", torch.tensor(True))

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.block_logit)

    def reg(self) -> torch.Tensor:
        return self.gate()

    @torch.no_grad()
    def maybe_prune(self, logger: Optional[JsonlLogger], tag: str):
        if not bool(self.active.item()):
            return
        if float(self.gate().detach().item()) < self.prune_threshold:
            self.active.fill_(False)
            if logger:
                logger.log({"type": "block_prune", "tag": tag})


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        n_heads: int,
        init_rank: int,
        max_rank: int,
        min_rank: int,
        ff_mult: int,
        causal: bool,
        head_prune_threshold: float,
        block_prune_threshold: float,
        # low-rank / svd scheduling defaults
        energy_target: float,
        rank_step_max: int,
        svd_interval: int,
        svd_interval_min: int,
        svd_interval_max: int,
        svd_growth: float,
        svd_shrink: float,
        tail_energy_low: float,
        tail_energy_high: float,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.causal = bool(causal)

        tag_base = f"layer{layer_idx}"

        def lr(in_f, out_f, tag):
            return VirtualLowRankLinear(
                in_features=in_f,
                out_features=out_f,
                init_rank=init_rank,
                max_rank=max_rank,
                min_rank=min_rank,
                energy_target=energy_target,
                rank_step_max=rank_step_max,
                svd_interval=svd_interval,
                svd_interval_min=svd_interval_min,
                svd_interval_max=svd_interval_max,
                svd_growth=svd_growth,
                svd_shrink=svd_shrink,
                tail_energy_low=tail_energy_low,
                tail_energy_high=tail_energy_high,
                tag=tag,
            )

        # attention projections
        self.q = lr(d_model, d_model, f"{tag_base}.q")
        self.k = lr(d_model, d_model, f"{tag_base}.k")
        self.v = lr(d_model, d_model, f"{tag_base}.v")
        self.o = lr(d_model, d_model, f"{tag_base}.o")

        # ffn
        hidden = ff_mult * d_model
        self.ff1 = lr(d_model, hidden, f"{tag_base}.ff1")
        self.ff2 = lr(hidden, d_model, f"{tag_base}.ff2")

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.head_ctrl = HeadController(n_heads=n_heads, prune_threshold=head_prune_threshold)
        self.block_ctrl = BlockController(prune_threshold=block_prune_threshold)

    def set_pressure(self, energy_target: float, rank_step_max: int, svd_interval_min: int):
        for layer in (self.q, self.k, self.v, self.o, self.ff1, self.ff2):
            layer.set_pressure(energy_target, rank_step_max, svd_interval_min)

    def _causal_mask(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.causal:
            return scores
        T = scores.size(-1)
        mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
        return scores.masked_fill(mask.view(1, 1, T, T), -1e9)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.n_heads, self.d_head)
        k = self.k(x).view(B, T, self.n_heads, self.d_head)
        v = self.v(x).view(B, T, self.n_heads, self.d_head)

        active = self.head_ctrl.active_indices()
        if active.numel() == 0:
            return torch.zeros_like(x)

        q = q[:, :, active, :]
        k = k[:, :, active, :]
        v = v[:, :, active, :]

        scores = torch.einsum("bthd,bThd->bhtT", q, k) / math.sqrt(self.d_head)
        scores = self._causal_mask(scores)
        att = F.softmax(scores, dim=-1)
        out = torch.einsum("bhtT,bThd->bthd", att, v)

        gates = self.head_ctrl.gates()[active].view(1, 1, -1, 1)
        out = out * gates

        full = torch.zeros(B, T, self.n_heads, self.d_head, device=x.device, dtype=x.dtype)
        full[:, :, active, :] = out
        return full.reshape(B, T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.block_ctrl.active.item()):
            return x

        g = self.block_ctrl.gate()  # scalar tensor

        attn_resid = self.o(self.attention(self.ln1(x)))
        x = x + g * attn_resid

        ff_resid = self.ff2(F.gelu(self.ff1(self.ln2(x))))
        x = x + g * ff_resid
        return x

    def maybe_update_ranks(self, logger: Optional[JsonlLogger]):
        for layer in (self.q, self.k, self.v, self.o, self.ff1, self.ff2):
            layer.maybe_svd_rank_update(logger)

    @torch.no_grad()
    def maybe_prune(self, logger: Optional[JsonlLogger]):
        self.head_ctrl.maybe_prune(logger, tag_prefix=f"layer{self.layer_idx}")
        self.block_ctrl.maybe_prune(logger, tag=f"layer{self.layer_idx}")

    def reg_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head_ctrl.reg(), self.block_ctrl.reg()

    def stats(self) -> Dict[str, Any]:
        ranks = [self.q.rank, self.k.rank, self.v.rank, self.o.rank, self.ff1.rank, self.ff2.rank]
        return {
            "active_block": bool(self.block_ctrl.active.item()),
            "block_gate": float(self.block_ctrl.gate().detach().item()),
            "active_heads": int(self.head_ctrl.active.sum().item()),
            "ranks": ranks,
        }


# ============================================================
# MODEL
# ============================================================

class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        init_rank: int,
        max_rank: int,
        min_rank: int,
        ff_mult: int,
        causal: bool,
        head_prune_threshold: float,
        block_prune_threshold: float,
        # low-rank defaults
        energy_target: float,
        rank_step_max: int,
        svd_interval: int,
        svd_interval_min: int,
        svd_interval_max: int,
        svd_growth: float,
        svd_shrink: float,
        tail_energy_low: float,
        tail_energy_high: float,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.causal = bool(causal)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                layer_idx=i,
                d_model=d_model,
                n_heads=n_heads,
                init_rank=init_rank,
                max_rank=max_rank,
                min_rank=min_rank,
                ff_mult=ff_mult,
                causal=causal,
                head_prune_threshold=head_prune_threshold,
                block_prune_threshold=block_prune_threshold,
                energy_target=energy_target,
                rank_step_max=rank_step_max,
                svd_interval=svd_interval,
                svd_interval_min=svd_interval_min,
                svd_interval_max=svd_interval_max,
                svd_growth=svd_growth,
                svd_shrink=svd_shrink,
                tail_energy_low=tail_energy_low,
                tail_energy_high=tail_energy_high,
            ) for i in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        return self.fc_out(x)

    def set_pressure(self, energy_target: float, rank_step_max: int, svd_interval_min: int):
        for b in self.blocks:
            b.set_pressure(energy_target, rank_step_max, svd_interval_min)

    def maybe_update_ranks(self, logger: Optional[JsonlLogger]):
        for b in self.blocks:
            b.maybe_update_ranks(logger)

    @torch.no_grad()
    def maybe_prune(self, logger: Optional[JsonlLogger]):
        for b in self.blocks:
            b.maybe_prune(logger)

    def reg_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        head_reg = torch.zeros((), device=device)
        block_reg = torch.zeros((), device=device)
        for b in self.blocks:
            h, bl = b.reg_losses()
            head_reg = head_reg + h
            block_reg = block_reg + bl
        return head_reg, block_reg

    def rank_stats(self) -> Tuple[int, float, int]:
        ranks: List[int] = []
        for b in self.blocks:
            st = b.stats()
            if not st["active_block"]:
                continue
            ranks.extend(st["ranks"])
        if not ranks:
            return 0, 0.0, 0
        return int(min(ranks)), float(sum(ranks) / len(ranks)), int(max(ranks))

    def head_stats(self) -> Tuple[int, float, int]:
        hs: List[int] = []
        for b in self.blocks:
            st = b.stats()
            if not st["active_block"]:
                continue
            hs.append(int(st["active_heads"]))
        if not hs:
            return 0, 0.0, 0
        return int(min(hs)), float(sum(hs) / len(hs)), int(max(hs))

    def active_blocks(self) -> int:
        return sum(1 for b in self.blocks if bool(b.block_ctrl.active.item()))

    def snapshot_masks(self) -> Dict[str, Any]:
        return {
            "blocks": [
                {
                    "active": bool(b.block_ctrl.active.item()),
                    "block_gate": float(b.block_ctrl.gate().detach().item()),
                    "active_heads": b.head_ctrl.active.detach().cpu().tolist(),
                    "head_gates": b.head_ctrl.gates().detach().cpu().tolist(),
                } for b in self.blocks
            ]
        }


# ============================================================
# PRESSURE CONTROLLER (global compute budget)
# ============================================================

@dataclass
class PressureConfig:
    compute_target: float
    val_tolerance: float
    pressure_step: float
    pressure_min: float
    pressure_max: float
    warmup_epochs: int

    energy_target_hi: float
    energy_target_lo: float

    rank_step_hi: int
    rank_step_lo: int

    svd_interval_min_hi: int
    svd_interval_min_lo: int

    head_lambda_base: float
    block_lambda_base: float
    lambda_scale: float


class PressureController:
    """
    Closed-loop "autoscaler":
      - If quality is OK and compute is above target: increase pressure (compress more).
      - If quality degrades: decrease pressure.
    """
    def __init__(self, cfg: PressureConfig):
        self.cfg = cfg
        self.pressure = 0.0
        self.best_val = float("inf")

    def knobs(self) -> Tuple[float, int, int, float, float]:
        p = self.pressure
        energy = self.cfg.energy_target_hi + p * (self.cfg.energy_target_lo - self.cfg.energy_target_hi)
        rank_step = int(round(self.cfg.rank_step_hi + p * (self.cfg.rank_step_lo - self.cfg.rank_step_hi)))
        svd_min = int(round(self.cfg.svd_interval_min_hi + p * (self.cfg.svd_interval_min_lo - self.cfg.svd_interval_min_hi)))

        lam_head = self.cfg.head_lambda_base * (1.0 + self.cfg.lambda_scale * p)
        lam_block = self.cfg.block_lambda_base * (1.0 + self.cfg.lambda_scale * p)
        return float(energy), int(rank_step), int(svd_min), float(lam_head), float(lam_block)

    def update(self, epoch: int, val_loss: float, compute_ratio: float):
        if val_loss < self.best_val:
            self.best_val = val_loss

        if epoch <= self.cfg.warmup_epochs:
            return

        ok = (val_loss <= self.best_val * (1.0 + self.cfg.val_tolerance))
        need_more = (compute_ratio > self.cfg.compute_target)

        if ok and need_more:
            self.pressure = min(self.cfg.pressure_max, self.pressure + self.cfg.pressure_step)
        elif not ok:
            self.pressure = max(self.cfg.pressure_min, self.pressure - self.cfg.pressure_step)


# ============================================================
# FLOPs proxy (for controller feedback)
# ============================================================

def estimate_flops_units(model: TinyTransformer, T: int) -> float:
    lr_units = 0.0
    attn_units = 0.0
    for b in model.blocks:
        st = b.stats()
        if not st["active_block"]:
            continue

        # low-rank matmuls ~ rank*(in+out)*T
        for layer in (b.q, b.k, b.v, b.o, b.ff1, b.ff2):
            lr_units += float(layer.rank) * float(layer.in_features + layer.out_features)

        # attention matmul ~ active_heads*d_head*T^2
        attn_units += float(st["active_heads"]) * float(b.d_head)

    return lr_units * float(T) + attn_units * float(T) * float(T)

def estimate_flops_ratio(model: TinyTransformer, T: int, baseline_units: float) -> float:
    cur = estimate_flops_units(model, T)
    if baseline_units <= 0:
        return 1.0
    return float(cur / baseline_units)


# ============================================================
# TRAINING
# ============================================================

def train(args):
    device = get_device()
    logger = JsonlLogger(args.log_file)

    # data
    tokens = load_token_file(args.data_file) if args.data_file.endswith(".tokens") else load_char_file(args.data_file)
    stoi, itos = build_vocab(tokens)
    vocab_size = len(stoi)
    data = encode(tokens, stoi, device=device)

    # split
    n = data.numel()
    split = int(n * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    causal = not args.bidirectional

    print(f"Training on {device}")
    print(f"Vocab size: {vocab_size}, Train tokens: {train_data.numel()}, Val tokens: {val_data.numel()}")
    print(f"Model: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}, block={args.block_size}, causal={causal}")
    print(f"Ranks: init={args.init_rank}, max={args.max_rank}, min={args.min_rank}")
    print(f"Budget: target_FLOPs~{args.compute_target}, val_tol={args.val_tolerance}, warmup={args.warmup_epochs}")

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        init_rank=args.init_rank,
        max_rank=args.max_rank,
        min_rank=args.min_rank,
        ff_mult=args.ff_mult,
        causal=causal,
        head_prune_threshold=args.head_prune_threshold,
        block_prune_threshold=args.block_prune_threshold,
        energy_target=args.energy_target_hi,
        rank_step_max=args.rank_step_hi,
        svd_interval=args.svd_interval,
        svd_interval_min=args.svd_interval_min_hi,
        svd_interval_max=args.svd_interval_max,
        svd_growth=args.svd_growth,
        svd_shrink=args.svd_shrink,
        tail_energy_low=args.tail_energy_low,
        tail_energy_high=args.tail_energy_high,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    B = args.batch_size
    T = args.block_size

    baseline_units = estimate_flops_units(model, T)

    pcfg = PressureConfig(
        compute_target=args.compute_target,
        val_tolerance=args.val_tolerance,
        pressure_step=args.pressure_step,
        pressure_min=0.0,
        pressure_max=1.0,
        warmup_epochs=args.warmup_epochs,
        energy_target_hi=args.energy_target_hi,
        energy_target_lo=args.energy_target_lo,
        rank_step_hi=args.rank_step_hi,
        rank_step_lo=args.rank_step_lo,
        svd_interval_min_hi=args.svd_interval_min_hi,
        svd_interval_min_lo=args.svd_interval_min_lo,
        head_lambda_base=args.head_lambda,
        block_lambda_base=args.block_lambda,
        lambda_scale=args.lambda_scale,
    )
    controller = PressureController(pcfg)

    def get_batch(split_name: str):
        d = train_data if split_name == "train" else val_data
        if d.numel() <= T + 1:
            x = d[:-1].unsqueeze(0)
            y = d[1:].unsqueeze(0)
            return x, y
        ix = torch.randint(0, d.numel() - T - 1, (B,), device=device)
        x = torch.stack([d[i:i+T] for i in ix])
        y = torch.stack([d[i+1:i+T+1] for i in ix])
        return x, y

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        train_xent_sum = 0.0
        head_reg_sum = 0.0
        block_reg_sum = 0.0

        energy_target, rank_step_max, svd_min, lam_head, lam_block = controller.knobs()
        model.set_pressure(energy_target=energy_target, rank_step_max=rank_step_max, svd_interval_min=svd_min)

        for _ in range(args.steps_per_epoch):
            global_step += 1
            x, y = get_batch("train")

            logits = model(x)
            xent = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            head_reg, block_reg = model.reg_losses()
            loss = xent + lam_head * head_reg + lam_block * block_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()

            model.maybe_update_ranks(logger)

            opt.step()

            if global_step % args.prune_every == 0 and epoch > args.warmup_epochs:
                model.maybe_prune(logger)

            train_xent_sum += float(xent.item())
            head_reg_sum += float(head_reg.item())
            block_reg_sum += float(block_reg.item())

        train_xent = train_xent_sum / args.steps_per_epoch

        # eval
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for _ in range(args.val_batches):
                x, y = get_batch("val")
                logits = model(x)
                vloss += float(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
            val_loss = vloss / args.val_batches

        flops_ratio = estimate_flops_ratio(model, T, baseline_units)

        rmin, ravg, rmax = model.rank_stats()
        hmin, havg, hmax = model.head_stats()
        active_blocks = model.active_blocks()

        controller.update(epoch, val_loss, flops_ratio)

        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d} | Train {train_xent:.4f} | Val {val_loss:.4f} | "
            f"FLOPs~{flops_ratio:.3f} | R={rmin}/{ravg:.1f}/{rmax} | "
            f"H={hmin}/{havg:.1f}/{hmax} | Blocks {active_blocks}/{args.n_layers} | "
            f"P={controller.pressure:.2f} | {dt:.2f}s"
        )

        if logger:
            logger.log({
                "type": "epoch",
                "epoch": epoch,
                "train_xent": train_xent,
                "val_loss": val_loss,
                "flops_ratio": flops_ratio,
                "rank_min": rmin, "rank_avg": ravg, "rank_max": rmax,
                "head_min": hmin, "head_avg": havg, "head_max": hmax,
                "active_blocks": active_blocks,
                "pressure": controller.pressure,
                "knobs": {
                    "energy_target": energy_target,
                    "rank_step_max": rank_step_max,
                    "svd_interval_min": svd_min,
                    "lambda_head": lam_head,
                    "lambda_block": lam_block,
                },
                "regs_avg": {
                    "head": head_reg_sum / args.steps_per_epoch,
                    "block": block_reg_sum / args.steps_per_epoch,
                },
                "masks": model.snapshot_masks(),
                "time_sec": dt,
            })

    if args.save_model:
        payload = {
            "state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": vars(args),
            "final_masks": model.snapshot_masks(),
        }
        torch.save(payload, args.save_model)
        print(f"Saved model to {args.save_model}")

    logger.close()


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser()

    # data / run
    p.add_argument("--data-file", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--bidirectional", action="store_true", help="Disable causal mask (NOT recommended for next-token LM).")

    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)

    # low-rank
    p.add_argument("--init-rank", type=int, default=64)
    p.add_argument("--max-rank", type=int, default=64)
    p.add_argument("--min-rank", type=int, default=8)
    p.add_argument("--svd-interval", type=int, default=300)
    p.add_argument("--svd-interval-max", type=int, default=5000)
    p.add_argument("--svd-growth", type=float, default=1.5)
    p.add_argument("--svd-shrink", type=float, default=0.7)
    p.add_argument("--tail-energy-low", type=float, default=0.002)
    p.add_argument("--tail-energy-high", type=float, default=0.02)

    # pruning thresholds
    p.add_argument("--head-prune-threshold", type=float, default=0.15)
    p.add_argument("--block-prune-threshold", type=float, default=0.10)
    p.add_argument("--prune-every", type=int, default=500)

    # optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)

    # global budget controller
    p.add_argument("--compute-target", type=float, default=0.55)
    p.add_argument("--val-tolerance", type=float, default=0.02)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--pressure-step", type=float, default=0.10)

    p.add_argument("--energy-target-hi", type=float, default=0.99)
    p.add_argument("--energy-target-lo", type=float, default=0.88)

    p.add_argument("--rank-step-hi", type=int, default=32)
    p.add_argument("--rank-step-lo", type=int, default=8)

    p.add_argument("--svd-interval-min-hi", type=int, default=300)
    p.add_argument("--svd-interval-min-lo", type=int, default=50)

    p.add_argument("--head-lambda", type=float, default=1e-4)
    p.add_argument("--block-lambda", type=float, default=5e-5)
    p.add_argument("--lambda-scale", type=float, default=50.0)

    # outputs
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--save-model", type=str, default=None)

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
