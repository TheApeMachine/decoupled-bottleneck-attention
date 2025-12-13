#!/usr/bin/env python3
# v18_transformer_lowrank_alrt.py
#
# Adaptive Low-Rank Training (ALRT) — v18
# - Per-layer adaptive ranks driven by STABLE RANK estimates of W = U @ V
# - Rank controller: safety factor, EMA smoothing, threshold, max step (bidirectional)
# - True rank resizing (parameter shapes change) via truncated SVD "sqrt split"
# - Optimizer reinit after any resize (per ALRT report)
# - Clean console output; detailed JSONL file logging
# - Checkpointing + resume + (KV-cached) autoregressive generation
#
# TRAIN:
#   python3 v18_transformer_lowrank_alrt.py --mode train --data data.txt --out-dir runs/v18
#
# RESUME:
#   python3 v18_transformer_lowrank_alrt.py --mode train --data data.txt --out-dir runs/v18 --resume runs/v18/last.pt
#
# GENERATE:
#   python3 v18_transformer_lowrank_alrt.py --mode generate --ckpt runs/v18/best.pt --prompt "Once upon a time" --max-new-tokens 400
#
# SANITY (overfit one batch; loss should crash hard):
#   python3 v18_transformer_lowrank_alrt.py --mode train --data data.txt --out-dir runs/v18_overfit --overfit-one-batch --max-steps 2000

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, asdict
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Small utilities
# ============================================================

def pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    # MPS for Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class JsonlLogger:
    """
    Append-only JSONL logger. Flushes every write.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def log(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


# ============================================================
# Tokenizer (char-level, self-contained)
# ============================================================

class CharTokenizer:
    """
    Simple character-level tokenizer that can be serialized into checkpoints.
    """
    def __init__(self, stoi: Dict[str, int], itos: List[str], unk_id: int = 0):
        self.stoi = stoi
        self.itos = itos
        self.unk_id = int(unk_id)

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        # reserve 0 for <unk>
        chars = sorted(set(text))
        stoi = {"<unk>": 0}
        for i, ch in enumerate(chars, start=1):
            stoi[ch] = i
        itos = [""] * len(stoi)
        for ch, i in stoi.items():
            itos[i] = ch
        return cls(stoi=stoi, itos=itos, unk_id=0)

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in s]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            ii = int(i)
            if 0 <= ii < len(self.itos):
                out.append(self.itos[ii])
            else:
                out.append("?")
        return "".join(out)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": self.itos, "unk_id": self.unk_id}

    @classmethod
    def from_state_dict(cls, d: dict) -> "CharTokenizer":
        return cls(stoi=d["stoi"], itos=d["itos"], unk_id=d.get("unk_id", 0))


def load_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_dataset(
    data_path: Path,
    train_frac: float,
    device: str,
) -> Tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    text = load_text(data_path)
    tok = CharTokenizer.build(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(train_frac * len(ids))
    train_ids = ids[:n].to(device)
    val_ids = ids[n:].to(device)
    return tok, train_ids, val_ids


# ============================================================
# ALRT core: stable rank + controller + true resizing
# ============================================================

@dataclass
class RankControllerConfig:
    min_rank: int = 8
    max_rank: int = 64
    safety_factor: float = 1.5   # alpha in the report
    ema_decay: float = 0.8       # beta in the report
    change_threshold: float = 2.0
    max_step: int = 4
    power_iters: int = 5
    eps: float = 1e-8


class RankController:
    """
    Controller that maps stable-rank estimates to a smooth integer rank trajectory.

    It implements:
      - safety factor
      - EMA smoothing
      - thresholding
      - bounded step
      - bidirectional adjustment
    """
    def __init__(self, init_rank: int, cfg: RankControllerConfig, layer_max_rank: int):
        self.cfg = dataclasses.replace(cfg, max_rank=int(layer_max_rank))
        self.rank = int(max(self.cfg.min_rank, min(self.cfg.max_rank, init_rank)))
        self._ema_target = float(self.rank)

    def state_dict(self) -> dict:
        return {"rank": self.rank, "ema_target": self._ema_target, "cfg": asdict(self.cfg)}

    def load_state_dict(self, d: dict) -> None:
        self.rank = int(d["rank"])
        self._ema_target = float(d.get("ema_target", self.rank))
        if "cfg" in d:
            # keep runtime cfg but allow resume of controller hyperparams
            self.cfg = RankControllerConfig(**d["cfg"])

    def propose(self, stable_rank: float) -> Tuple[float, float]:
        """
        Returns (raw_target, ema_target) as floats.
        """
        raw = float(self.cfg.safety_factor * stable_rank)
        # clamp raw to legal range
        raw = max(float(self.cfg.min_rank), min(float(self.cfg.max_rank), raw))
        # EMA smooth
        self._ema_target = self.cfg.ema_decay * self._ema_target + (1.0 - self.cfg.ema_decay) * raw
        ema = max(float(self.cfg.min_rank), min(float(self.cfg.max_rank), self._ema_target))
        return raw, ema

    def update(self, stable_rank: float) -> Tuple[int, int, dict]:
        old = int(self.rank)
        raw, ema = self.propose(stable_rank)

        delta = ema - float(self.rank)
        if abs(delta) < self.cfg.change_threshold:
            return old, old, {"raw_target": raw, "ema_target": ema, "delta": delta, "step": 0}

        # bounded step, at least 1
        step_mag = min(float(self.cfg.max_step), abs(delta))
        step = int(math.copysign(max(1.0, step_mag), delta))
        new = int(self.rank + step)
        new = max(self.cfg.min_rank, min(self.cfg.max_rank, new))
        self.rank = int(new)

        return old, self.rank, {"raw_target": raw, "ema_target": ema, "delta": delta, "step": step}


class LowRankLinearAdaptiveSpectral(nn.Module):
    """
    Linear y = x @ W^T + b with W factorized as U @ V

    U: (out, r)
    V: (r, in)

    Rank r changes over training, guided by stable-rank(W).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_rank: int,
        rcfg: RankControllerConfig,
        bias: bool = True,
        name: str = "",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.name = str(name)

        layer_max = min(int(rcfg.max_rank), self.in_features, self.out_features)
        layer_min = min(int(rcfg.min_rank), layer_max)
        self.rcfg = dataclasses.replace(rcfg, min_rank=layer_min, max_rank=layer_max)

        r0 = int(max(layer_min, min(layer_max, init_rank)))
        self.controller = RankController(init_rank=r0, cfg=self.rcfg, layer_max_rank=layer_max)

        # Parameters (float32 by default; AMP handles compute)
        self.U = nn.Parameter(torch.randn(self.out_features, r0) * 0.02)
        self.V = nn.Parameter(torch.randn(r0, self.in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    @torch.no_grad()
    def stable_rank(self) -> Tuple[float, float, float]:
        """
        Returns (stable_rank, fro2, sigma_max^2).

        Computation is done in low-rank form:
          ||W||_F^2 = tr((U^T U)(V V^T))
          sigma_max via power iteration using matvecs with U,V
        """
        eps = self.rcfg.eps

        U = self.U.detach().float()
        V = self.V.detach().float()
        r = U.shape[1]
        if r == 0:
            return 0.0, 0.0, 0.0

        # Frobenius norm squared: tr((U^T U)(V V^T))
        UtU = U.t().matmul(U)        # (r,r)
        VVt = V.matmul(V.t())        # (r,r)
        fro2 = float((UtU * VVt).sum().item())

        # Power iteration for sigma_max^2
        # Start with random v in R^in
        v = torch.randn(self.in_features, device=U.device, dtype=torch.float32)
        v = v / (v.norm() + eps)

        sigma = 0.0
        for _ in range(int(self.rcfg.power_iters)):
            # u = W v = U (V v)
            u = U.matmul(V.matmul(v))  # (out,)
            u_norm = u.norm() + eps
            u = u / u_norm

            # v = W^T u = V^T (U^T u)
            v = V.t().matmul(U.t().matmul(u))  # (in,)
            v = v / (v.norm() + eps)

            sigma = float(u_norm.item())

        sigma2 = float(sigma * sigma)
        sr = float(fro2 / (sigma2 + eps))
        # stable rank is always <= true rank
        sr = max(0.0, min(float(r), sr))
        return sr, fro2, sigma2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in)
        # y = x @ (U V)^T = x @ V^T @ U^T
        h = x.matmul(self.V.t())     # (..., r)
        y = h.matmul(self.U.t())     # (..., out)
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def resize_rank(self, new_rank: int, method: str = "svd") -> Tuple[int, int]:
        """
        Resize factors to new_rank, attempting to preserve W when shrinking.

        method:
          - "svd": truncated SVD on W (sqrt split) when shrinking; random init when growing
          - "truncate": simple slice when shrinking; random init when growing (fast but lossy)
        """
        new_rank = int(new_rank)
        new_rank = max(self.rcfg.min_rank, min(self.rcfg.max_rank, new_rank))
        old_rank = self.rank
        if new_rank == old_rank:
            return old_rank, new_rank

        device = self.U.device
        dtype = self.U.dtype  # keep param dtype

        if new_rank < old_rank:
            if method == "truncate":
                U_new = self.U[:, :new_rank].detach().clone()
                V_new = self.V[:new_rank, :].detach().clone()
            else:
                # SVD on CPU for stability/memory
                W = (self.U.detach().float().matmul(self.V.detach().float())).cpu()
                try:
                    U_svd, S, Vh = torch.linalg.svd(W, full_matrices=False)
                except RuntimeError:
                    # fallback: truncate factors (safer than crashing training)
                    U_new = self.U[:, :new_rank].detach().clone()
                    V_new = self.V[:new_rank, :].detach().clone()
                else:
                    k = new_rank
                    U_k = U_svd[:, :k]
                    S_k = S[:k].clamp(min=0.0)
                    Vh_k = Vh[:k, :]
                    sqrtS = torch.sqrt(S_k + 1e-12)  # (k,)
                    U_new = (U_k * sqrtS.unsqueeze(0)).to(device=device, dtype=torch.float32)
                    V_new = (sqrtS.unsqueeze(1) * Vh_k).to(device=device, dtype=torch.float32)
        else:
            # Growing: keep old factors and append small random components
            U_old = self.U.detach().clone()
            V_old = self.V.detach().clone()
            add = new_rank - old_rank
            U_extra = torch.randn(self.out_features, add, device=device, dtype=torch.float32) * 0.02
            V_extra = torch.randn(add, self.in_features, device=device, dtype=torch.float32) * 0.02
            U_new = torch.cat([U_old.float(), U_extra], dim=1)
            V_new = torch.cat([V_old.float(), V_extra], dim=0)

        # Re-wrap as Parameters (this changes optimizer param set; caller should rebuild optimizer)
        self.U = nn.Parameter(U_new.to(device=device, dtype=dtype))
        self.V = nn.Parameter(V_new.to(device=device, dtype=dtype))
        self.controller.rank = int(new_rank)
        return old_rank, new_rank

    def rank_state_dict(self) -> dict:
        return {
            "name": self.name,
            "in": self.in_features,
            "out": self.out_features,
            "rank": self.rank,
            "controller": self.controller.state_dict(),
        }

    def load_rank_state_dict(self, d: dict) -> None:
        if "controller" in d:
            self.controller.load_state_dict(d["controller"])


# ============================================================
# Transformer (GPT-ish)
# ============================================================

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    init_rank: int
    min_rank: int
    max_rank: int
    # ALRT controller hyperparams
    safety_factor: float = 1.5
    ema_decay: float = 0.8
    change_threshold: float = 2.0
    max_step: int = 4
    power_iters: int = 5


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, rcfg: RankControllerConfig, name_prefix: str):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # low-rank projections
        self.q_proj = LowRankLinearAdaptiveSpectral(cfg.n_embd, cfg.n_embd, cfg.init_rank, rcfg, bias=False, name=f"{name_prefix}.q")
        self.k_proj = LowRankLinearAdaptiveSpectral(cfg.n_embd, cfg.n_embd, cfg.init_rank, rcfg, bias=False, name=f"{name_prefix}.k")
        self.v_proj = LowRankLinearAdaptiveSpectral(cfg.n_embd, cfg.n_embd, cfg.init_rank, rcfg, bias=False, name=f"{name_prefix}.v")
        self.o_proj = LowRankLinearAdaptiveSpectral(cfg.n_embd, cfg.n_embd, cfg.init_rank, rcfg, bias=False, name=f"{name_prefix}.o")

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if use_cache and cache is not None and ("k" in cache) and (cache["k"] is not None):
            past_k = cache["k"]
            past_v = cache["v"]
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Attention
        # Training: full sequence => causal
        # Generation incremental (T==1 with past): no future positions exist => non-causal is OK
        is_causal = (past_len == 0)

        # Use PyTorch SDPA (can use Flash/Math kernel depending on backend)
        try:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )  # (B, nh, T, hs)
        except Exception:
            # Fallback attention (more portable, slower):
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q * scale) @ k.transpose(-2, -1)  # (B, nh, T, Tk)
            if is_causal:
                # Only correct when Tk == T (our is_causal path is only used with no past cache).
                mask = torch.ones((T, T), device=att.device, dtype=torch.bool).tril()
                att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_drop(y)

        new_cache = None
        if use_cache:
            new_cache = {"k": k.detach(), "v": v.detach()}

        return y, new_cache


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, rcfg: RankControllerConfig, name_prefix: str):
        super().__init__()
        self.fc1 = LowRankLinearAdaptiveSpectral(cfg.n_embd, 4 * cfg.n_embd, cfg.init_rank, rcfg, bias=True, name=f"{name_prefix}.ff_up")
        self.fc2 = LowRankLinearAdaptiveSpectral(4 * cfg.n_embd, cfg.n_embd, cfg.init_rank, rcfg, bias=True, name=f"{name_prefix}.ff_down")
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, rcfg: RankControllerConfig, idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, rcfg, name_prefix=f"block{idx}.attn")
        self.mlp = MLP(cfg, rcfg, name_prefix=f"block{idx}.mlp")

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        a, new_cache = self.attn(self.ln1(x), use_cache=use_cache, cache=cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class LowRankTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        rcfg = RankControllerConfig(
            min_rank=cfg.min_rank,
            max_rank=cfg.max_rank,
            safety_factor=cfg.safety_factor,
            ema_decay=cfg.ema_decay,
            change_threshold=cfg.change_threshold,
            max_step=cfg.max_step,
            power_iters=cfg.power_iters,
        )
        self._rcfg = rcfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg, rcfg, i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[List[Optional[dict]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Optional[dict]]]]:
        """
        idx: (B,T)
        cache: list of per-layer dicts with keys 'k','v' or None
        Returns:
          logits: (B,T,V)
          loss: scalar tensor or None
          new_cache: cache list if use_cache else None
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size :]
            if targets is not None:
                targets = targets[:, -self.cfg.block_size :]
            T = idx.shape[1]

        # figure out past length from cache (for position embedding offset)
        past_len = 0
        if use_cache and cache is not None and len(cache) > 0 and cache[0] is not None and cache[0].get("k") is not None:
            past_len = int(cache[0]["k"].size(2))

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        pos = pos.clamp(max=self.cfg.block_size - 1)  # safety
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        new_cache = [None] * len(self.blocks) if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if (use_cache and cache is not None) else None
            x, nc = block(x, use_cache=use_cache, cache=layer_cache)
            if use_cache:
                new_cache[i] = nc

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_cache

    # -------------------------
    # Rank utilities
    # -------------------------

    def iter_lowrank_layers(self) -> Iterable[Tuple[str, LowRankLinearAdaptiveSpectral]]:
        for name, module in self.named_modules():
            if isinstance(module, LowRankLinearAdaptiveSpectral):
                yield name, module

    @torch.no_grad()
    def rank_stats(self) -> Tuple[int, float, int]:
        ranks = [m.rank for _, m in self.iter_lowrank_layers()]
        if not ranks:
            return 0, 0.0, 0
        t = torch.tensor(ranks, dtype=torch.float32)
        return int(t.min().item()), float(t.mean().item()), int(t.max().item())

    def get_rank_state(self) -> dict:
        return {"layers": {name: m.rank_state_dict() for name, m in self.iter_lowrank_layers()}}

    def load_rank_state(self, d: dict) -> None:
        layers = d.get("layers", {})
        for name, m in self.iter_lowrank_layers():
            if name in layers:
                m.load_rank_state_dict(layers[name])

    @torch.no_grad()
    def adapt_ranks(
        self,
        step: int,
        logger: Optional[JsonlLogger] = None,
        resize_method: str = "svd",
        log_full: bool = False,
    ) -> dict:
        """
        Compute stable rank per low-rank layer, update controller, resize if needed.

        Returns a summary dict.
        """
        start_t = time.time()
        changes = []
        full = []
        # first pass: compute stable ranks and proposed updates
        proposals: List[Tuple[str, LowRankLinearAdaptiveSpectral, int, int, float, float, float, dict]] = []
        for name, layer in self.iter_lowrank_layers():
            sr, fro2, sigma2 = layer.stable_rank()
            old, new, info = layer.controller.update(sr)
            proposals.append((name, layer, old, new, sr, fro2, sigma2, info))
            if log_full:
                full.append({
                    "layer": name,
                    "old_rank": int(old),
                    "new_rank": int(new),
                    "stable_rank": float(sr),
                    "fro2": float(fro2),
                    "sigma2": float(sigma2),
                    **{k: float(v) if isinstance(v, (int, float)) else v for k, v in info.items()},
                })

        # second pass: apply resizes
        for name, layer, old, new, sr, fro2, sigma2, info in proposals:
            if new != old:
                old2, new2 = layer.resize_rank(new, method=resize_method)
                changes.append({
                    "layer": name,
                    "old_rank": int(old2),
                    "new_rank": int(new2),
                    "stable_rank": float(sr),
                    "fro2": float(fro2),
                    "sigma2": float(sigma2),
                    **{k: float(v) if isinstance(v, (int, float)) else v for k, v in info.items()},
                })

        rmin, rmean, rmax = self.rank_stats()
        took = time.time() - start_t
        summary = {
            "type": "rank_update",
            "step": int(step),
            "changed_layers": int(len(changes)),
            "total_layers": int(len(proposals)),
            "r_min": int(rmin),
            "r_mean": float(rmean),
            "r_max": int(rmax),
            "resize_method": str(resize_method),
            "took_sec": float(took),
        }
        if logger is not None:
            logger.log(summary)
            for c in changes:
                logger.log({"type": "rank_change", "step": int(step), **c})
            if log_full:
                logger.log({"type": "rank_full", "step": int(step), "layers": full})
        return {"summary": summary, "changes": changes}


# ============================================================
# Optimizer / LR schedule
# ============================================================

def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> torch.optim.Optimizer:
    """
    AdamW with decoupled weight decay on weight-like tensors only (not bias/LN).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "ln" in name.lower() or "layernorm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=betas,
    )


def get_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * float(step) / float(max(1, warmup))
    # cosine decay
    t = (step - warmup) / float(max(1, max_steps - warmup))
    t = min(1.0, max(0.0, t))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))


# ============================================================
# Eval helper
# ============================================================

@torch.no_grad()
def estimate_loss(model: LowRankTransformerLM, get_batch, iters: int, amp: bool) -> float:
    model.eval()
    losses = []
    for _ in range(int(iters)):
        x, y = get_batch("val")
        with torch.autocast(device_type=("cuda" if amp else "cpu"), dtype=(torch.float16 if amp else torch.bfloat16), enabled=amp):
            _, loss, _ = model(x, y, use_cache=False, cache=None)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


# ============================================================
# Sampling (top-k / nucleus) + generation
# ============================================================

def top_k_top_p_filter(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    """
    logits: (B, V)
    Returns filtered logits (masked with -inf)
    """
    B, V = logits.shape
    out = logits

    if top_k is not None and top_k > 0 and top_k < V:
        v, ix = torch.topk(out, top_k, dim=-1)
        kth = v[:, -1].unsqueeze(-1)
        out = torch.where(out < kth, torch.full_like(out, float("-inf")), out)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # mask tokens with cumulative prob > top_p
        mask = cum > top_p
        # keep at least 1 token
        mask[:, 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        # scatter back
        out2 = torch.full_like(out, float("-inf"))
        out2.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        out = out2

    return out


@torch.no_grad()
def generate(
    model: LowRankTransformerLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    model.eval()
    device = idx.device
    cache: Optional[List[Optional[dict]]] = [None] * model.cfg.n_layer if use_cache else None

    # Prefill (build cache on the prompt)
    logits, _, cache = model(idx, targets=None, use_cache=use_cache, cache=cache)

    for _ in range(int(max_new_tokens)):
        logits_last = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(logits_last, dim=-1, keepdim=True)
        else:
            logits_last = logits_last / float(temperature)
            logits_last = top_k_top_p_filter(logits_last, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits_last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

        # Context window management (simple; crops cache too)
        if idx.size(1) > model.cfg.block_size:
            idx = idx[:, -model.cfg.block_size :]
            if use_cache and cache is not None:
                for i in range(len(cache)):
                    if cache[i] is None:
                        continue
                    k = cache[i]["k"]
                    v = cache[i]["v"]
                    if k is not None and k.size(2) > model.cfg.block_size:
                        cache[i]["k"] = k[:, :, -model.cfg.block_size :, :].contiguous()
                        cache[i]["v"] = v[:, :, -model.cfg.block_size :, :].contiguous()

        # Incremental step: feed only new token
        logits, _, cache = model(next_id, targets=None, use_cache=use_cache, cache=cache)

    return idx


# ============================================================
# Checkpointing
# ============================================================

def save_ckpt(
    out_dir: Path,
    name: str,
    model: LowRankTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    tokenizer: CharTokenizer,
    step: int,
    best_val: float,
    val_loss: float,
    extra: dict,
) -> Path:
    ckpt = {
        "step": int(step),
        "best_val": float(best_val),
        "val_loss": float(val_loss),
        "model_cfg": asdict(model.cfg),
        "model_state": model.state_dict(),
        "rank_state": model.get_rank_state(),
        "optim_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "tokenizer": tokenizer.state_dict(),
        "extra": extra,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "saved_at": time.time(),
    }
    path = out_dir / name
    _atomic_torch_save(ckpt, path)
    return path


def load_ckpt(path: Path, device: str) -> dict:
    return torch.load(Path(path), map_location=device)


# ============================================================
# Main
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "generate"])
    p.add_argument("--data", type=str, default=None, help="Path to training text file (train mode).")
    p.add_argument("--out-dir", type=str, default="runs/v18")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume (train).")
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to load (generate).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1337)

    # model
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # ranks
    p.add_argument("--init-rank", type=int, default=64)
    p.add_argument("--min-rank", type=int, default=8)
    p.add_argument("--max-rank", type=int, default=64)
    p.add_argument("--safety-factor", type=float, default=1.5)
    p.add_argument("--ema-decay", type=float, default=0.8)
    p.add_argument("--change-threshold", type=float, default=2.0)
    p.add_argument("--max-step", type=int, default=4)
    p.add_argument("--power-iters", type=int, default=5)
    p.add_argument("--resize-method", type=str, default="svd", choices=["svd", "truncate"])
    p.add_argument("--rank-update-every", type=int, default=0, help="Steps between rank updates (0 = once per epoch-ish).")
    p.add_argument("--log-ranks-full", action="store_true", help="Log full per-layer stable-rank table at each update (big).")

    # training
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=0, help="Generate a short sample every N steps (0=off).")
    p.add_argument("--sample-len", type=int, default=200)
    p.add_argument("--overfit-one-batch", action="store_true")

    # generation
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)

    args = p.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)

    amp_enabled = bool(args.amp and device.startswith("cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.jsonl"
    logger = JsonlLogger(log_path)

    if args.mode == "generate":
        if args.ckpt is None:
            raise SystemExit("--ckpt is required for --mode generate")
        ckpt = load_ckpt(Path(args.ckpt), device=device)

        cfg = ModelConfig(**ckpt["model_cfg"])
        model = LowRankTransformerLM(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        if "rank_state" in ckpt:
            model.load_rank_state(ckpt["rank_state"])

        tok = CharTokenizer.from_state_dict(ckpt["tokenizer"])

        prompt = args.prompt or ""
        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

        top_k = args.top_k if args.top_k and args.top_k > 0 else None
        top_p = args.top_p if args.top_p and args.top_p > 0 else None

        out = generate(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=True,
        )
        text = tok.decode(out[0].tolist())
        print(text)
        return

    # -------------------------
    # TRAIN
    # -------------------------
    if args.data is None:
        raise SystemExit("--data is required for --mode train")

    tok, train_ids, val_ids = build_dataset(Path(args.data), args.train_frac, device=device)
    vocab_size = len(tok.itos)

    # default steps_per_epoch heuristic
    # (contiguous data => approximate #blocks)
    steps_per_epoch = max(1, int(len(train_ids) // (args.batch_size * args.block_size)))
    rank_update_every = args.rank_update_every if args.rank_update_every and args.rank_update_every > 0 else steps_per_epoch

    # maybe resume
    ckpt = None
    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = load_ckpt(Path(args.resume), device=device)
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        # restore RNG for exact continuation when possible
        rng = ckpt.get("rng", {})
        try:
            random.setstate(rng.get("python"))
            torch.set_rng_state(rng.get("torch"))
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng.get("cuda"))
        except Exception:
            pass

    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        init_rank=args.init_rank,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        safety_factor=args.safety_factor,
        ema_decay=args.ema_decay,
        change_threshold=args.change_threshold,
        max_step=args.max_step,
        power_iters=args.power_iters,
    )

    model = LowRankTransformerLM(cfg).to(device)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if ckpt is not None:
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        if "rank_state" in ckpt:
            model.load_rank_state(ckpt["rank_state"])
        if "optim_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim_state"])
            except Exception:
                # shapes may differ after a resize; safest is to rebuild optimizer
                optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        if ckpt.get("scaler_state") and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass

    # fixed batch for overfit mode
    fixed_batch = None
    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal fixed_batch
        src = train_ids if split == "train" else val_ids
        if len(src) < args.block_size + 2:
            raise RuntimeError("Dataset too small for chosen --block-size.")
        if args.overfit_one_batch:
            if fixed_batch is None:
                ix = torch.randint(0, len(src) - args.block_size - 1, (args.batch_size,))
                x = torch.stack([src[i : i + args.block_size] for i in ix])
                y = torch.stack([src[i + 1 : i + args.block_size + 1] for i in ix])
                fixed_batch = (x.to(device), y.to(device))
            return fixed_batch
        ix = torch.randint(0, len(src) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([src[i : i + args.block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # initial log header
    rmin, rmean, rmax = model.rank_stats()
    baseline = math.log(vocab_size)
    logger.log({
        "type": "run_start",
        "time": time.time(),
        "device": device,
        "vocab_size": vocab_size,
        "baseline_nll_logV": baseline,
        "baseline_ppl": float(math.exp(baseline)),
        "steps_per_epoch": int(steps_per_epoch),
        "rank_update_every": int(rank_update_every),
        "args": vars(args),
        "model_cfg": asdict(cfg),
        "rank_stats": {"r_min": rmin, "r_mean": rmean, "r_max": rmax},
    })

    print(f"[v18] device={device} vocab={vocab_size} block={cfg.block_size} layers={cfg.n_layer} heads={cfg.n_head} embd={cfg.n_embd}")
    print(f"[v18] baseline loss log(V)={baseline:.3f} ppl={math.exp(baseline):.2f}")
    print(f"[v18] rank init/min/max = {cfg.init_rank}/{cfg.min_rank}/{cfg.max_rank} | update_every={rank_update_every} steps | resize={args.resize_method}")

    t0 = time.time()
    tokens_per_step = args.batch_size * args.block_size

    model.train()
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch("train")

        with torch.autocast(device_type=("cuda" if amp_enabled else "cpu"), dtype=(torch.float16 if amp_enabled else torch.bfloat16), enabled=amp_enabled):
            _, loss, _ = model(x, y, use_cache=False, cache=None)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # logging
        if (step + 1) % args.log_interval == 0 or step == start_step:
            rmin, rmean, rmax = model.rank_stats()
            ppl = float(math.exp(min(20.0, float(loss.item()))))
            elapsed = time.time() - t0
            tok_s = (tokens_per_step * args.log_interval) / max(1e-9, elapsed) if (step + 1) % args.log_interval == 0 else float("nan")
            t0 = time.time()

            # console (keep it readable)
            epoch_f = (step + 1) / float(steps_per_epoch)
            print(f"step {step+1:6d} | epoch {epoch_f:6.2f} | loss {loss.item():7.4f} | ppl {ppl:7.2f} | lr {lr:.2e} | r {rmin:2d}/{rmean:5.1f}/{rmax:2d} | tok/s {tok_s:8.0f}")

            logger.log({
                "type": "train_step",
                "time": time.time(),
                "step": int(step + 1),
                "epoch": float(epoch_f),
                "lr": float(lr),
                "loss": float(loss.item()),
                "ppl": float(ppl),
                "r_min": int(rmin),
                "r_mean": float(rmean),
                "r_max": int(rmax),
                "tokens_per_step": int(tokens_per_step),
                "tok_per_sec": float(tok_s) if not math.isnan(tok_s) else None,
            })

        # evaluation + ckpt
        if (step + 1) % args.eval_interval == 0 or (step + 1) == args.max_steps:
            val_loss = estimate_loss(model, get_batch, args.eval_iters, amp=amp_enabled)
            val_ppl = float(math.exp(min(20.0, val_loss)))
            rmin, rmean, rmax = model.rank_stats()

            print(f"eval @ {step+1:6d} | val_loss {val_loss:7.4f} | val_ppl {val_ppl:7.2f} | r {rmin:2d}/{rmean:5.1f}/{rmax:2d}")

            logger.log({
                "type": "eval",
                "time": time.time(),
                "step": int(step + 1),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "r_min": int(rmin),
                "r_mean": float(rmean),
                "r_max": int(rmax),
            })

            # checkpoint
            extra = {"notes": "v18 ALRT stable-rank + SVD resize", "rank_update_every": rank_update_every}
            save_ckpt(out_dir, "last.pt", model, optimizer, scaler, tok, step + 1, best_val, val_loss, extra)

            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(out_dir, "best.pt", model, optimizer, scaler, tok, step + 1, best_val, val_loss, extra)

        # optional sample
        if args.sample_every and args.sample_every > 0 and (step + 1) % args.sample_every == 0:
            prompt = "The "
            idx0 = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
            out_ids = generate(model, idx0, max_new_tokens=args.sample_len, temperature=0.8, top_k=50, top_p=0.0, use_cache=True)
            sample = tok.decode(out_ids[0].tolist())
            print("---- sample ----")
            print(sample)
            print("---------------")
            logger.log({"type": "sample", "time": time.time(), "step": int(step + 1), "text": sample})

        # rank update (end-of-epoch-ish by default)
        if (step + 1) % rank_update_every == 0 and (step + 1) < args.max_steps:
            before = model.rank_stats()
            res = model.adapt_ranks(
                step=step + 1,
                logger=logger,
                resize_method=args.resize_method,
                log_full=args.log_ranks_full,
            )
            after = model.rank_stats()

            if res["summary"]["changed_layers"] > 0:
                # Rebuild optimizer + scaler (controller hyperparams unchanged)
                optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
                scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

                print(
                    f"rank_update @ {step+1:6d} | changed {res['summary']['changed_layers']:2d}/{res['summary']['total_layers']:2d} "
                    f"| r {before[0]:2d}/{before[1]:5.1f}/{before[2]:2d} -> {after[0]:2d}/{after[1]:5.1f}/{after[2]:2d} "
                    f"| took {res['summary']['took_sec']:.2f}s"
                )

    logger.log({"type": "run_end", "time": time.time(), "best_val": float(best_val)})
    logger.close()
    print(f"[v18] done. best_val_loss={best_val:.4f} | logs={log_path} | ckpts={out_dir}")


if __name__ == "__main__":
    main()
