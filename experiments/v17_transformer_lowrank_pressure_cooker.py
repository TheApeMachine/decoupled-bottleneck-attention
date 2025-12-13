#!/usr/bin/env python3
# v17_transformer_lowrank_pressure_cooker.py
#
# TRAIN:
#   python3 v17_transformer_lowrank_pressure_cooker.py \
#     --mode train --data data.txt --out-dir runs/v17
#
# RESUME:
#   python3 v17_transformer_lowrank_pressure_cooker.py \
#     --mode train --data data.txt --out-dir runs/v17 \
#     --resume runs/v17/last.pt
#
# GENERATE:
#   python3 v17_transformer_lowrank_pressure_cooker.py \
#     --mode generate --ckpt runs/v17/best.pt \
#     --prompt "Once upon a time" --max-new-tokens 400
#
# SANITY (overfit one batch; loss should crash hard):
#   python3 v17_transformer_lowrank_pressure_cooker.py \
#     --mode train --data data.txt --out-dir runs/v17_overfit \
#     --overfit-one-batch --max-steps 2000

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Tokenizer (char-level, stable + simple)
# -------------------------

class CharTokenizer:
    def __init__(self, stoi: Dict[str, int], itos: list[str], unk_id: int = 0):
        self.stoi = stoi
        self.itos = itos
        self.unk_id = unk_id

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

    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in s]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append("")
        return "".join(out)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": self.itos, "unk_id": self.unk_id}

    @classmethod
    def from_state_dict(cls, d: dict) -> "CharTokenizer":
        return cls(stoi=d["stoi"], itos=d["itos"], unk_id=d.get("unk_id", 0))


# -------------------------
# Low-rank Linear (factorized weight: A @ B)
# Pressure cooker: increasing penalty on higher components to push info into early components
# -------------------------

class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int,
        bias: bool = True,
        pressure_profile: str = "quadratic",  # "linear" or "quadratic"
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Never exceed true matrix rank ceiling
        self.max_rank = int(min(max_rank, in_features, out_features))
        if self.max_rank < 1:
            raise ValueError(f"max_rank became < 1 (in={in_features}, out={out_features}, max_rank={max_rank})")

        self.current_rank = self.max_rank

        # A: (out, r), B: (r, in)
        # Initialization: small random so early training is stable.
        self.A = nn.Parameter(torch.randn(out_features, self.max_rank) * 0.02)
        self.B = nn.Parameter(torch.randn(self.max_rank, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Pressure weights: 0 on component 0, increasing to 1 on component max_rank-1.
        w = torch.linspace(0.0, 1.0, steps=self.max_rank)
        if pressure_profile == "quadratic":
            w = w * w
        elif pressure_profile == "linear":
            pass
        else:
            raise ValueError(f"Unknown pressure_profile: {pressure_profile}")

        self.register_buffer("pressure_w", w, persistent=False)

    def set_rank(self, new_rank: int) -> None:
        new_rank = int(new_rank)
        new_rank = max(1, min(new_rank, self.max_rank))
        self.current_rank = new_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.current_rank
        # x: (..., in)
        # h = x @ B^T -> (..., r)
        # y = h @ A^T -> (..., out)
        Br = self.B[:r, :]               # (r, in)
        Ar = self.A[:, :r]               # (out, r)
        h = x.matmul(Br.t())             # (..., r)
        y = h.matmul(Ar.t())             # (..., out)
        if self.bias is not None:
            y = y + self.bias
        return y

    def pressure_term(self) -> torch.Tensor:
        """
        Returns a scalar >= 0 that grows when higher-index components have large norms.
        You multiply this by a schedule-controlled alpha in the training loop.
        """
        # component energy: ||A[:,i]||^2 + ||B[i,:]||^2
        a2 = (self.A * self.A).sum(dim=0)     # (max_rank,)
        b2 = (self.B * self.B).sum(dim=1)     # (max_rank,)
        comp = a2 + b2                        # (max_rank,)
        return (self.pressure_w * comp).sum()


# -------------------------
# Transformer (GPT-ish), but all key linears are LowRankLinear
# -------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    max_rank: int
    pressure_profile: str


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn = LowRankLinear(
            cfg.n_embd,
            3 * cfg.n_embd,
            max_rank=cfg.max_rank,
            bias=True,
            pressure_profile=cfg.pressure_profile,
        )
        self.c_proj = LowRankLinear(
            cfg.n_embd,
            cfg.n_embd,
            max_rank=cfg.max_rank,
            bias=True,
            pressure_profile=cfg.pressure_profile,
        )

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # For non-flash fallback
        bias = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = LowRankLinear(
            cfg.n_embd,
            4 * cfg.n_embd,
            max_rank=cfg.max_rank,
            bias=True,
            pressure_profile=cfg.pressure_profile,
        )
        self.proj = LowRankLinear(
            4 * cfg.n_embd,
            cfg.n_embd,
            max_rank=cfg.max_rank,
            bias=True,
            pressure_profile=cfg.pressure_profile,
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LowRankTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Output head (tied weights) — keep full-rank; low-rank happens inside the network.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"T={T} > block_size={self.cfg.block_size}")

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def pressure_term(self) -> torch.Tensor:
        s = torch.zeros((), device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, LowRankLinear):
                s = s + m.pressure_term()
        return s

    def set_all_ranks(self, rank: int) -> None:
        for m in self.modules():
            if isinstance(m, LowRankLinear):
                m.set_rank(rank)

    def rank_stats(self) -> Tuple[int, int, float]:
        ranks = []
        for m in self.modules():
            if isinstance(m, LowRankLinear):
                ranks.append(m.current_rank)
        if not ranks:
            return 0, 0, 0.0
        return int(min(ranks)), int(max(ranks)), float(sum(ranks) / len(ranks))

    def get_rank_state(self) -> Dict[str, int]:
        d = {}
        for name, m in self.named_modules():
            if isinstance(m, LowRankLinear):
                d[name] = int(m.current_rank)
        return d

    def load_rank_state(self, state: Dict[str, int]) -> None:
        for name, m in self.named_modules():
            if isinstance(m, LowRankLinear) and name in state:
                m.set_rank(int(state[name]))


# -------------------------
# Rank schedule (the "pressure cooker" timeline)
# -------------------------

def rank_at_step(
    step: int,
    rank_init: int,
    rank_final: int,
    warmup_steps: int,
    anneal_steps: int,
) -> int:
    # Stay at rank_init during warmup, then linearly go to rank_final during anneal.
    if step < warmup_steps:
        return int(rank_init)
    if anneal_steps <= 0:
        return int(rank_final)
    t = (step - warmup_steps) / float(anneal_steps)
    t = max(0.0, min(1.0, t))
    r = rank_init + (rank_final - rank_init) * t
    return int(round(r))


def pressure_alpha_at_step(
    step: int,
    alpha_final: float,
    warmup_steps: int,
    anneal_steps: int,
) -> float:
    # 0 during warmup, ramp to alpha_final through anneal window
    if alpha_final <= 0:
        return 0.0
    if step < warmup_steps:
        return 0.0
    if anneal_steps <= 0:
        return float(alpha_final)
    t = (step - warmup_steps) / float(anneal_steps)
    t = max(0.0, min(1.0, t))
    return float(alpha_final) * t


# -------------------------
# Sampling (generate)
# -------------------------

@torch.no_grad()
def generate(
    model: LowRankTransformerLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, V)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            continue

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)

            mask = cum > top_p
            mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

            logits2 = torch.full_like(logits, float("-inf"))
            logits2.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
            logits = logits2

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx


# -------------------------
# Checkpointing
# -------------------------

def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

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

def load_ckpt(
    ckpt_path: Path,
    device: str,
) -> dict:
    return torch.load(ckpt_path, map_location=device)


# -------------------------
# Eval helper
# -------------------------

@torch.no_grad()
def estimate_loss(model: LowRankTransformerLM, get_batch, iters: int, amp: bool) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch()
        if amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "generate"], default="train")

    p.add_argument("--data", type=str, default=None, help="Path to a UTF-8-ish text file")
    p.add_argument("--out-dir", type=str, default="runs/v17")
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"

    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--seed", type=int, default=1337)

    # model
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)

    # low-rank schedule (pressure cooker)
    p.add_argument("--rank-init", type=int, default=128)
    p.add_argument("--rank-final", type=int, default=32)
    p.add_argument("--rank-warmup-steps", type=int, default=2000)
    p.add_argument("--rank-anneal-steps", type=int, default=12000)
    p.add_argument("--rank-update-every", type=int, default=50)

    p.add_argument("--pressure-alpha", type=float, default=1e-4, help="Final pressure strength (ramps from 0)")
    p.add_argument("--pressure-profile", choices=["linear", "quadratic"], default="quadratic")

    # training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)

    p.add_argument("--resume", type=str, default=None, help="Path to last.pt to resume training")
    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    p.add_argument("--compile", action="store_true", help="torch.compile(model) if available")
    p.add_argument("--overfit-one-batch", action="store_true")

    # generation
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--prompt", type=str, default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)

    args = p.parse_args()

    # perf knobs (safe defaults)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint first if generate or resume — so we can use its tokenizer/config reliably.
    ckpt = None
    if args.mode == "generate":
        if not args.ckpt:
            raise SystemExit("--ckpt is required for --mode generate")
        ckpt = load_ckpt(Path(args.ckpt), device=device)

    if args.mode == "train" and args.resume:
        ckpt = load_ckpt(Path(args.resume), device=device)

    # Load / build dataset + tokenizer
    if args.data is None:
        # fallback tiny text so the script always runs
        text = ("This is a demo dataset.\n" * 10000)
    else:
        text = Path(args.data).read_text(encoding="utf-8", errors="ignore")

    if ckpt is not None and "tokenizer" in ckpt:
        tokenizer = CharTokenizer.from_state_dict(ckpt["tokenizer"])
    else:
        tokenizer = CharTokenizer.build(text)

    vocab_size = tokenizer.vocab_size()
    uniform_loss = math.log(vocab_size)

    print(f"Vocab size: {vocab_size}")
    print(f"Uniform baseline loss log(V): {uniform_loss:.4f} (ppl ~ {math.exp(uniform_loss):.0f})")

    # Encode full text with tokenizer in use
    data_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data_ids))
    train_ids = data_ids[:n]
    val_ids = data_ids[n:]

    def get_batch(split: str):
        src = train_ids if split == "train" else val_ids
        if len(src) < args.block_size + 2:
            raise RuntimeError("Dataset too small for chosen --block-size.")
        ix = torch.randint(0, len(src) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([src[i : i + args.block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # Build model config
    if ckpt is not None and "model_cfg" in ckpt:
        cfg = ModelConfig(**ckpt["model_cfg"])
    else:
        cfg = ModelConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            max_rank=args.rank_init,
            pressure_profile=args.pressure_profile,
        )

    model = LowRankTransformerLM(cfg).to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.startswith("cuda")) else None

    start_step = 0
    best_val = float("inf")

    # Restore checkpoint into model/optimizer/scaler
    if ckpt is not None and args.mode == "train":
        try:
            model.load_state_dict(ckpt["model_state"], strict=True)
            if "rank_state" in ckpt:
                model.load_rank_state(ckpt["rank_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            if scaler is not None and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_step = int(ckpt.get("step", 0))
            best_val = float(ckpt.get("best_val", best_val))
            # RNG restore (best-effort)
            try:
                random.setstate(ckpt["rng"]["python"])
                torch.set_rng_state(ckpt["rng"]["torch"])
                if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
                    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
            except Exception:
                pass
            print(f"Resumed from {args.resume}: step={start_step}, best_val={best_val:.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to resume from {args.resume}: {e}")

    if args.mode == "generate":
        # Build model from ckpt, load weights, then sample
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "rank_state" in ckpt:
            model.load_rank_state(ckpt["rank_state"])

        prompt_ids = tokenizer.encode(args.prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = generate(
            model,
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(tokenizer.decode(out[0].tolist()))
        return

    # Learning rate schedule (cosine with warmup)
    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.learning_rate * step / max(1, args.warmup_steps)
        if step >= args.max_steps:
            return args.min_lr
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * t))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # Overfit-one-batch setup (debug correctness quickly)
    fixed_x, fixed_y = None, None
    if args.overfit_one_batch:
        fixed_x, fixed_y = get_batch("train")
        print("Overfit-one-batch enabled: training on a single fixed batch.")

    print("Starting training...")
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Rank schedule update (only every N steps to keep it cheap/clean)
        if (step % max(1, args.rank_update_every)) == 0:
            target_rank = rank_at_step(
                step=step,
                rank_init=args.rank_init,
                rank_final=args.rank_final,
                warmup_steps=args.rank_warmup_steps,
                anneal_steps=args.rank_anneal_steps,
            )
            model.set_all_ranks(target_rank)

        # Pressure alpha schedule (ramps from 0 after warmup)
        pressure_alpha = pressure_alpha_at_step(
            step=step,
            alpha_final=args.pressure_alpha,
            warmup_steps=args.rank_warmup_steps,
            anneal_steps=args.rank_anneal_steps,
        )

        if args.overfit_one_batch:
            x, y = fixed_x, fixed_y
        else:
            x, y = get_batch("train")

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                _, ce_loss = model(x, y)
                press = model.pressure_term()
                loss = ce_loss + pressure_alpha * press

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, ce_loss = model(x, y)
            press = model.pressure_term()
            loss = ce_loss + pressure_alpha * press

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % 50 == 0:
            rmin, rmax, rmean = model.rank_stats()
            ppl = math.exp(float(ce_loss.item()))
            print(
                f"step {step:6d} | lr {lr:.2e} | ce {ce_loss.item():.4f} | ppl {ppl:7.1f} | "
                f"rank(min/mean/max) {rmin}/{rmean:.1f}/{rmax} | pressα {pressure_alpha:.2e}"
            )

        if step > 0 and (step % args.eval_every == 0 or step == args.max_steps - 1):
            train_loss = estimate_loss(model, lambda: get_batch("train"), args.eval_iters, amp=(scaler is not None))
            val_loss = estimate_loss(model, lambda: get_batch("val"), args.eval_iters, amp=(scaler is not None))
            print(f"== eval @ step {step} | train {train_loss:.4f} | val {val_loss:.4f} | val_ppl {math.exp(val_loss):.1f}")

            # always save last
            save_ckpt(
                out_dir=out_dir,
                name="last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                tokenizer=tokenizer,
                step=step,
                best_val=best_val,
                val_loss=val_loss,
                extra={"args": vars(args)},
            )

            # save best
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(
                    out_dir=out_dir,
                    name="best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    tokenizer=tokenizer,
                    step=step,
                    best_val=best_val,
                    val_loss=val_loss,
                    extra={"args": vars(args)},
                )
                print(f"   (new best) best_val={best_val:.4f}")

    dt = time.time() - t0
    print(f"Done. Best val: {best_val:.4f}. Total seconds: {dt:.1f}")


if __name__ == "__main__":
    main()
