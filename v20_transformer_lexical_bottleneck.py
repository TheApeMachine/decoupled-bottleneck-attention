#!/usr/bin/env python3
# v20_transformer_lexical_bottleneck.py
#
# A "first-principles" architectural probe motivated by ALRT findings.
#
# v19 tested an ATTENTION bottleneck: decouple d_model from the attention
# subspace (attn_dim), motivated by ALRT's observation that Q/K projections
# compress extremely hard (mean effective ranks ~11 out of 512).
#
# v20 keeps that knob AND adds a second, more brute-force optimization target:
# the *lexical interface* (token embeddings + LM head).
#
# Why attack the lexical interface?
#   In small/medium models with a word-level vocab (~33k), the LM head matmul
#   often dominates FLOPs (it certainly does in v19's printed breakdown).
#   So we test whether logits can be computed through a narrower "embedding
#   space" without killing performance.
#
# Modification: Factorized input/output embeddings (ALBERT-style)
#   - Choose embed_dim <= d_model.
#   - Token embedding table is (V, embed_dim) instead of (V, d_model).
#   - A learned projection P maps embed_dim -> d_model for the residual stream.
#   - The output logits are computed with the tied factorization:
#       logits = (x @ P^T) @ E^T
#     where E is the token embedding table.
#
# Benefits (when embed_dim << d_model):
#   - big reduction in LM-head compute
#   - big reduction in embedding parameters
#   - still keeps a wide residual stream (d_model) for internal computation
#
# Hypothesis (tested here):
#   If attention's *pattern computation* truly lives in a small subspace, we can
#   hard-wire that by decoupling d_model from the attention subspace dimension:
#
#       Q, K, V : d_model -> d_attn   (with d_attn << d_model)
#       Attn out: d_attn  -> d_model
#
#   This should reduce:
#     - FLOPs in attention projections and score/application (scales with d_attn)
#     - KV-cache memory at inference (scales with d_attn)
#     - parameters in the attention module (scales with d_attn)
#
# Minimal extra twist (optional):
#   - --null-attn adds a learned "null" key/value token, allowing a query to put
#     probability mass on "attend nowhere" (a concrete version of that question).
#   - --tie-qk shares Q and K projection weights (tests whether separate Q/K is necessary).
#   - --learned-temp learns per-head temperature (relaxes fixed 1/sqrt(d)).
#
# The rest of the model is intentionally standard GPT-ish pre-norm for comparability.
#
# Runs on WikiText-2 (word-level tokenization) with the config in the ALRT report:
#   d_model=512, n_layer=6, n_head=8, d_ff=2048, block_size=256
#   AdamW lr=3e-4, wd=0.01, batch=32, steps_per_epoch=200, epochs=30
#
# USAGE (baseline-ish):
#   python3 v20_transformer_lexical_bottleneck.py --wikitext2-dir ./wikitext-2
#
# USAGE (attention bottleneck, e.g. d_attn=128):
#   python3 v20_transformer_lexical_bottleneck.py --wikitext2-dir ./wikitext-2 --attn-dim 128
#
# USAGE (lexical bottleneck, e.g. embed_dim=256):
#   python3 v20_transformer_lexical_bottleneck.py --wikitext2-dir ./wikitext-2 --embed-dim 256
#
# USAGE (stack them: smaller attention + smaller lexical interface):
#   python3 v20_transformer_lexical_bottleneck.py --wikitext2-dir ./wikitext-2 --attn-dim 128 --embed-dim 256
#
# Logs:
#   - console: readable headline metrics
#   - out_dir/train.jsonl: granular step/eval/ckpt metadata (append-only)
#
# Notes:
#   - This script is self-contained and does not require external tokenizers.
#   - If your WikiText-2 files are named differently, pass --train-file/--valid-file.
#
# The point is not "this will win"—it's a clean experiment that teaches us
# something about what attention really needs.
#
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


# ----------------------------
# Utilities
# ----------------------------

def pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class JsonlLogger:
    """Append-only JSONL logger (flushes every write)."""
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


def human_int(n: float) -> str:
    n = float(n)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


# ----------------------------
# WikiText-2 word tokenizer
# ----------------------------

class WordTokenizer:
    """
    Minimal word-level tokenizer:
      - splits on whitespace
      - inserts <eos> at end of each non-empty line (preserves boundaries)
      - uses <unk> for OOV
    """
    def __init__(self, stoi: Dict[str, int], itos: List[str], unk: str = "<unk>", eos: str = "<eos>"):
        self.stoi = stoi
        self.itos = itos
        self.unk = unk
        self.eos = eos
        self.unk_id = int(stoi[unk])
        self.eos_id = int(stoi[eos])

    @staticmethod
    def _line_to_tokens(line: str) -> List[str]:
        # WikiText-2 uses a lot of formatting; whitespace tokenization keeps it simple and reproducible.
        toks = line.strip().split()
        return toks

    @classmethod
    def build_from_train_text(cls, train_text: str, extra_specials: Optional[List[str]] = None) -> "WordTokenizer":
        extra_specials = extra_specials or []
        vocab = {}
        def add(tok: str):
            if tok not in vocab:
                vocab[tok] = 1
            else:
                vocab[tok] += 1

        # special tokens first
        specials = ["<unk>", "<eos>"] + [t for t in extra_specials if t not in ("<unk>", "<eos>")]
        for s in specials:
            add(s)

        for line in train_text.splitlines():
            toks = cls._line_to_tokens(line)
            if not toks:
                continue
            for t in toks:
                add(t)
            add("<eos>")

        # deterministic ordering: specials first, then alpha by token
        # (Frequency ordering is also common; but alphabetical is stable and easy to diff.)
        vocab_tokens = [t for t in vocab.keys() if t not in specials]
        vocab_tokens.sort()

        itos = specials + vocab_tokens
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode_text(self, text: str) -> List[int]:
        ids: List[int] = []
        for line in text.splitlines():
            toks = self._line_to_tokens(line)
            if not toks:
                continue
            for t in toks:
                ids.append(self.stoi.get(t, self.unk_id))
            ids.append(self.eos_id)
        return ids

    def decode_ids(self, ids: Iterable[int]) -> str:
        # Mostly for debugging/generation; preserves <eos> tokens as line breaks.
        out = []
        for i in ids:
            tok = self.itos[int(i)] if 0 <= int(i) < len(self.itos) else self.unk
            if tok == self.eos:
                out.append("\n")
            else:
                out.append(tok)
        return " ".join(out).replace(" \n ", "\n")

    def vocab_size(self) -> int:
        return len(self.itos)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": self.itos, "unk": self.unk, "eos": self.eos}

    @classmethod
    def from_state_dict(cls, d: dict) -> "WordTokenizer":
        return cls(stoi=d["stoi"], itos=d["itos"], unk=d.get("unk", "<unk>"), eos=d.get("eos", "<eos>"))


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def resolve_wikitext2_files(
    wikitext2_dir: Optional[Path],
    train_file: Optional[Path],
    valid_file: Optional[Path],
    test_file: Optional[Path],
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Tries common WikiText-2 file names.
    Accepts either:
      - --wikitext2-dir path containing train/valid/test
      - or explicit --train-file/--valid-file
    """
    if train_file and valid_file:
        return Path(train_file), Path(valid_file), Path(test_file) if test_file else None

    if not wikitext2_dir:
        raise SystemExit("Provide --wikitext2-dir or both --train-file and --valid-file")

    d = Path(wikitext2_dir)

    # common naming variants
    candidates = [
        ("wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"),
        ("train.txt", "valid.txt", "test.txt"),
        ("wikitext-2-v1/train.txt", "wikitext-2-v1/valid.txt", "wikitext-2-v1/test.txt"),
    ]
    for tr, va, te in candidates:
        trp = d / tr
        vap = d / va
        tep = d / te
        if trp.exists() and vap.exists():
            return trp, vap, tep if tep.exists() else None

    raise SystemExit(f"Could not find WikiText-2 files under {d}. Try --train-file/--valid-file explicitly.")


def build_wikitext2_dataset(
    train_path: Path,
    valid_path: Path,
    device: str,
) -> Tuple[WordTokenizer, torch.Tensor, torch.Tensor]:
    train_text = read_text(train_path)
    valid_text = read_text(valid_path)

    tok = WordTokenizer.build_from_train_text(train_text)
    train_ids = torch.tensor(tok.encode_text(train_text), dtype=torch.long, device=device)
    valid_ids = torch.tensor(tok.encode_text(valid_text), dtype=torch.long, device=device)

    return tok, train_ids, valid_ids


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random contiguous subsequences from flat token stream.
    """
    n = int(data.size(0))
    if n <= block_size + 1:
        raise RuntimeError(f"Dataset too small: n={n}, block_size={block_size}")
    ix = torch.randint(0, n - block_size - 1, (batch_size,), device=data.device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ----------------------------
# Model
# ----------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    embed_dim: int = 512     # factorized embedding dimension (<= n_embd for a lexical bottleneck)
    d_ff: int = 2048
    dropout: float = 0.1

    # Architectural modification knobs
    attn_dim: int = 512        # attention subspace dim (baseline: = n_embd)
    tie_qk: bool = False       # share Q and K projection weights
    null_attn: bool = False    # add a "null key/value" option to attend nowhere
    learned_temp: bool = True  # per-head learned attention temperature
    mlp: str = "gelu"          # "gelu" (baseline) or "swiglu"

    # keep biases off like many GPT-style configs
    bias: bool = False


class CausalSelfAttentionBottleneck(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.attn_dim % cfg.n_head == 0, "attn_dim must be divisible by n_head"
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.attn_dim // cfg.n_head
        self.attn_dim = cfg.attn_dim

        if cfg.tie_qk:
            self.qk_proj = nn.Linear(cfg.n_embd, cfg.attn_dim, bias=cfg.bias)
            self.q_proj = None
            self.k_proj = None
        else:
            self.q_proj = nn.Linear(cfg.n_embd, cfg.attn_dim, bias=cfg.bias)
            self.k_proj = nn.Linear(cfg.n_embd, cfg.attn_dim, bias=cfg.bias)
            self.qk_proj = None

        self.v_proj = nn.Linear(cfg.n_embd, cfg.attn_dim, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.attn_dim, cfg.n_embd, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # causal mask for max block_size
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

        if cfg.null_attn:
            # Shape: (1, n_head, 1, head_dim) so it broadcasts over batch and time.
            self.null_k = nn.Parameter(torch.zeros(1, cfg.n_head, 1, self.head_dim))
            self.null_v = nn.Parameter(torch.zeros(1, cfg.n_head, 1, self.head_dim))
            nn.init.normal_(self.null_k, mean=0.0, std=0.02)
            nn.init.zeros_(self.null_v)  # "attend nowhere" starts as "contribute nothing"
        else:
            self.null_k = None
            self.null_v = None

        if cfg.learned_temp:
            # Per-head log scale (initialized at 0 => scale=1).
            self.logit_scale = nn.Parameter(torch.zeros(cfg.n_head))
        else:
            self.logit_scale = None

    def _proj_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.tie_qk:
            assert self.qk_proj is not None
            q = self.qk_proj(x)
            k = q
        else:
            assert self.q_proj is not None and self.k_proj is not None
            q = self.q_proj(x)
            k = self.k_proj(x)
        v = self.v_proj(x)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        if T > self.cfg.block_size:
            raise ValueError(f"T={T} > block_size={self.cfg.block_size}")

        q, k, v = self._proj_qkv(x)  # (B,T,attn_dim)

        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Optionally append a learned "null" key/value that is always visible.
        if self.cfg.null_attn:
            assert self.null_k is not None and self.null_v is not None
            k = torch.cat([k, self.null_k.expand(B, -1, -1, -1)], dim=2)  # (B, nh, T+1, hd)
            v = torch.cat([v, self.null_v.expand(B, -1, -1, -1)], dim=2)  # (B, nh, T+1, hd)

        # Attention scores
        # scale: classic 1/sqrt(d) plus optional learned per-head multiplier
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.logit_scale is not None:
            # (1, nh, 1, 1)
            scale = scale * torch.exp(self.logit_scale).view(1, self.n_head, 1, 1)

        att = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T or T+1)

        # Causal mask
        causal = self.causal_mask[:, :, :T, :T]  # (1,1,T,T)
        if self.cfg.null_attn:
            # Always allow the null token as an extra column of True.
            null_col = torch.ones((1, 1, T, 1), device=att.device, dtype=torch.bool)
            causal = torch.cat([causal.to(att.device), null_col], dim=-1)  # (1,1,T,T+1)
        else:
            causal = causal.to(att.device)

        # Use dtype-min instead of -inf for better behavior in float16 on some backends.
        att = att.masked_fill(~causal, torch.finfo(att.dtype).min)

        # Softmax -> attention weights
        w = F.softmax(att, dim=-1)
        w = self.attn_drop(w)

        y = w @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, self.attn_dim)  # (B,T,attn_dim)

        y = self.out_proj(y)      # (B,T,n_embd)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.mlp == "swiglu":
            # SwiGLU: (xW1) * silu(xW2) with a single fused projection to 2*d_ff.
            self.fc = nn.Linear(cfg.n_embd, 2 * cfg.d_ff, bias=cfg.bias)
            self.proj = nn.Linear(cfg.d_ff, cfg.n_embd, bias=cfg.bias)
        else:
            # Baseline GELU FFN
            self.fc = nn.Linear(cfg.n_embd, cfg.d_ff, bias=cfg.bias)
            self.proj = nn.Linear(cfg.d_ff, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.cfg.mlp == "swiglu":
            a, b = x.chunk(2, dim=-1)
            x = a * F.silu(b)
        else:
            x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttentionBottleneck(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Factorized (lexical bottleneck) embeddings:
        #   token ids -> embed_dim -> (project) -> n_embd residual stream
        self.wte = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.wte_proj = None
        if cfg.embed_dim != cfg.n_embd:
            if cfg.embed_dim > cfg.n_embd:
                raise ValueError(f"embed_dim ({cfg.embed_dim}) must be <= n_embd ({cfg.n_embd}) for this probe")
            self.wte_proj = nn.Linear(cfg.embed_dim, cfg.n_embd, bias=False)

        # Positions live in the residual stream width.
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Output head:
        #   - If embed_dim == n_embd, we can use the usual tied linear head.
        #   - If embed_dim <  n_embd, we compute logits via the tied factorization:
        #       logits = (x @ P^T) @ E^T
        #     where P is wte_proj and E is wte.
        self.lm_head = None
        if cfg.embed_dim == cfg.n_embd:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"T={T} > block_size={self.cfg.block_size}")
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)

        tok = self.wte(idx)  # (B,T,embed_dim)
        if self.wte_proj is not None:
            tok = self.wte_proj(tok)  # (B,T,n_embd)
        x = tok + self.wpe(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)  # (B,T,V)
        else:
            # Lexical bottleneck / factorized head:
            #   x: (B,T,n_embd)
            #   wte_proj.weight: (n_embd, embed_dim)
            #   wte.weight: (V, embed_dim)
            assert self.wte_proj is not None
            x_e = x @ self.wte_proj.weight  # (B,T,embed_dim) == x @ P^T
            logits = x_e @ self.wte.weight.t()  # (B,T,V) == (x @ P^T) @ E^T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ----------------------------
# Metrics / accounting
# ----------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs_forward(cfg: ModelConfig) -> Dict[str, int]:
    """
    Very rough multiply-accumulate (MAC) counts for one forward pass of ONE sample
    with sequence length cfg.block_size.
    The relative contributions are what matter.
    """
    T = cfg.block_size
    d = cfg.n_embd
    E = cfg.embed_dim
    dh = cfg.attn_dim
    dff = cfg.d_ff
    V = cfg.vocab_size
    H = cfg.n_head

    # Attention projections: if tie_qk, QK uses one projection instead of 2.
    # Baseline uses 3 projections (Q,K,V).
    qk_mult = 1 if cfg.tie_qk else 2
    qkv = (qk_mult + 1) * T * d * dh
    out = T * dh * d

    # Attention score + apply scales with dh (total head dims).
    scores = T * T * dh
    apply = T * T * dh

    softmax = H * T * T  # not MACs, but "ops-ish"

    # FFN
    ffn_up = T * d * dff
    ffn_down = T * dff * d

    # Lexical interface (embedding projection + LM head)
    # Baseline (E == d): logits = x @ E_in^T  ->  T * d * V
    # Factorized (E <  d):
    #   input  projection: tok(E) -> d   ->  T * E * d
    #   output projection: x(d) -> E -> V
    #                    ->  T * (d * E + E * V)
    embed_proj = 0
    if E != d:
        embed_proj = T * E * d
    if E == d:
        lm = T * d * V
    else:
        lm = T * (d * E + E * V)

    # LayerNorm (tiny compared to matmuls)
    ln = 2 * T * d  # per block, crude

    per_block = {
        "attn_qkv_proj": qkv,
        "attn_scores": scores,
        "attn_softmax_ops": softmax,
        "attn_apply": apply,
        "attn_out_proj": out,
        "ffn_up": ffn_up,
        "ffn_down": ffn_down,
        "layernorm_ops": ln,
    }
    totals = {k: v * cfg.n_layer for k, v in per_block.items()}
    if embed_proj:
        totals["embed_proj"] = embed_proj
    totals["lm_head"] = lm
    return totals


def pretty_pct_table(items: Dict[str, int]) -> str:
    total = float(sum(items.values()))
    rows = []
    for k, v in sorted(items.items(), key=lambda kv: kv[1], reverse=True):
        rows.append((k, v, 100.0 * float(v) / max(1.0, total)))
    # make a clean aligned string (avoid markdown tables in console)
    out = []
    out.append("Approx forward compute breakdown (rough MACs/ops):")
    for k, v, pct in rows:
        out.append(f"  {k:>18s}: {human_int(v):>9s}  ({pct:5.1f}%)")
    return "\n".join(out)


# ----------------------------
# Eval / generation
# ----------------------------

@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    amp_dtype: Optional[torch.dtype] = None,
    device_type: str = "cpu",
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size=batch_size, block_size=block_size)
        if amp_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / len(losses))


@torch.no_grad()
def generate(
    model: GPTLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond, None)
        logits = logits[:, -1, :] / max(1e-8, float(temperature))

        if top_k and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -1e10), logits)

        if top_p < 1.0:
            # nucleus sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cdf = torch.cumsum(probs, dim=-1)
            mask = cdf > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, -1e10)
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return idx


# ----------------------------
# Training loop
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train", choices=["train", "generate"])

    # WikiText-2 inputs
    ap.add_argument("--wikitext2-dir", type=str, default=None, help="Directory containing wiki.train.tokens/wiki.valid.tokens")
    ap.add_argument("--train-file", type=str, default=None)
    ap.add_argument("--valid-file", type=str, default=None)
    ap.add_argument("--test-file", type=str, default=None)
    ap.add_argument("--data", type=str, default=None, help="Single text file to split 90/10 for train/val")

    # Output / checkpointing
    ap.add_argument("--out-dir", type=str, default="runs/v11")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint for --mode generate")

    # Model hyperparams (match report defaults)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--embed-dim", type=int, default=512, help="Token embedding / lexical bottleneck dimension (<= n_embd)")
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Architectural mod knobs
    ap.add_argument("--attn-dim", type=int, default=512, help="Attention subspace dimension (baseline = n_embd)")
    ap.add_argument("--tie-qk", action="store_true", help="Share Q and K projection weights")
    ap.add_argument("--null-attn", action="store_true", help="Add a learned null key/value option (attend nowhere)")
    ap.add_argument("--no-learned-temp", action="store_true", help="Disable learned per-head temperature")
    ap.add_argument("--mlp", type=str, default="gelu", choices=["gelu", "swiglu"], help="FFN nonlinearity (baseline=gelu)")

    # Training hyperparams (match report defaults)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)

    # Generation knobs
    ap.add_argument("--prompt", type=str, default="The")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)

    args = ap.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "train.jsonl")

    # Autocast dtype selection
    amp_dtype: Optional[torch.dtype] = None
    if args.dtype == "float16":
        amp_dtype = torch.float16
    elif args.dtype == "bfloat16":
        amp_dtype = torch.bfloat16

    # Load checkpoint if needed (resume or generate)
    ckpt = None
    if args.mode == "generate":
        if not args.ckpt:
            raise SystemExit("--ckpt is required for --mode generate")
        ckpt = torch.load(args.ckpt, map_location=device)
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device)

    # Dataset/tokenizer
    if ckpt is not None and "tokenizer" in ckpt:
        tokenizer = WordTokenizer.from_state_dict(ckpt["tokenizer"])
        # Still load data from disk for training; tokenizer ensures consistent ids.
        if args.data:
            text = read_text(Path(args.data))
            all_ids = torch.tensor(tokenizer.encode_text(text), dtype=torch.long, device=device)
            n = int(0.9 * len(all_ids))
            train_ids = all_ids[:n]
            valid_ids = all_ids[n:]
        else:
            train_path, valid_path, _ = resolve_wikitext2_files(
                Path(args.wikitext2_dir) if args.wikitext2_dir else None,
                Path(args.train_file) if args.train_file else None,
                Path(args.valid_file) if args.valid_file else None,
                Path(args.test_file) if args.test_file else None,
            )
            train_text = read_text(train_path)
            valid_text = read_text(valid_path)
            train_ids = torch.tensor(tokenizer.encode_text(train_text), dtype=torch.long, device=device)
            valid_ids = torch.tensor(tokenizer.encode_text(valid_text), dtype=torch.long, device=device)
    elif args.data:
        text = read_text(Path(args.data))
        tokenizer = WordTokenizer.build_from_train_text(text)
        all_ids = torch.tensor(tokenizer.encode_text(text), dtype=torch.long, device=device)
        n = int(0.9 * len(all_ids))
        train_ids = all_ids[:n]
        valid_ids = all_ids[n:]
    else:
        train_path, valid_path, _ = resolve_wikitext2_files(
            Path(args.wikitext2_dir) if args.wikitext2_dir else None,
            Path(args.train_file) if args.train_file else None,
            Path(args.valid_file) if args.valid_file else None,
            Path(args.test_file) if args.test_file else None,
        )
        tokenizer, train_ids, valid_ids = build_wikitext2_dataset(train_path, valid_path, device=device)

    vocab_size = tokenizer.vocab_size()
    uniform_loss = math.log(vocab_size)

    # Build model config (from ckpt if present, else from args)
    if ckpt is not None and "model_cfg" in ckpt:
        cfg = ModelConfig(**ckpt["model_cfg"])
    else:
        cfg = ModelConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            embed_dim=args.embed_dim,
            d_ff=args.d_ff,
            dropout=args.dropout,
            attn_dim=args.attn_dim,
            tie_qk=bool(args.tie_qk),
            null_attn=bool(args.null_attn),
            learned_temp=not bool(args.no_learned_temp),
            mlp=str(args.mlp),
            bias=False,
        )

    # Sanity checks
    if cfg.attn_dim % cfg.n_head != 0:
        raise SystemExit(f"--attn-dim {cfg.attn_dim} must be divisible by --n-head {cfg.n_head}")
    if cfg.block_size != args.block_size:
        # If resuming, cfg.block_size wins.
        args.block_size = cfg.block_size

    model = GPTLM(cfg).to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed (continuing): {e}")

    # AdamW: apply weight decay only to weight matrices (dim >= 2).
    # Do NOT decay LayerNorm scales, biases, or other 1D params (dim < 2).
    # This matches standard Transformer training practice and tends to improve stability.
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

    start_step = 0
    best_val = float("inf")

    # Restore from checkpoint
    if ckpt is not None and args.mode == "train":
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optim_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim_state"])
            except Exception:
                # optimizer shape mismatch can happen if you change config; keep going
                pass
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        # RNG restore
        if "rng" in ckpt:
            try:
                random.setstate(ckpt["rng"]["python"])
                torch.set_rng_state(ckpt["rng"]["torch"])
                if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
                    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
            except Exception:
                pass
        print(f"Resumed from {args.resume}: step={start_step}, best_val={best_val:.4f}")

    # Mode: generate
    if args.mode == "generate":
        model.load_state_dict(ckpt["model_state"], strict=True)  # type: ignore
        prompt_ids = tokenizer.encode_text(args.prompt)  # note: word-level; prompt treated as one line
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_id]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = generate(
            model=model,
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(tokenizer.decode_ids(out[0].tolist()))
        return

    # Print startup info (console)
    total_params = count_parameters(model)
    macs = estimate_macs_forward(cfg)
    print(f"Device: {device}")
    print(f"Train tokens: {train_ids.numel():,} | Valid tokens: {valid_ids.numel():,} | Vocab: {vocab_size:,}")
    print(f"Uniform baseline loss log(V): {uniform_loss:.4f} (ppl ~ {math.exp(uniform_loss):.0f})")
    print(
        f"Model: layers={cfg.n_layer}, d_model={cfg.n_embd}, embed_dim={cfg.embed_dim}, "
        f"heads={cfg.n_head}, d_ff={cfg.d_ff}, block={cfg.block_size}"
    )
    print(
        f"Attention mod: attn_dim={cfg.attn_dim} (baseline={cfg.n_embd}), "
        f"tie_qk={cfg.tie_qk}, null_attn={cfg.null_attn}, learned_temp={cfg.learned_temp}, mlp={cfg.mlp}"
    )
    if cfg.embed_dim != cfg.n_embd:
        print(f"Lexical mod: factorized embeddings enabled (embed_dim={cfg.embed_dim} < d_model={cfg.n_embd})")
    print(f"Parameters: {total_params:,} (~{human_int(total_params)})")
    print(pretty_pct_table(macs))

    # Also log a config event
    logger.log({
        "type": "run_config",
        "time": time.time(),
        "device": device,
        "dtype": args.dtype,
        "model_cfg": asdict(cfg),
        "train_tokens": int(train_ids.numel()),
        "valid_tokens": int(valid_ids.numel()),
        "vocab_size": int(vocab_size),
        "uniform_loss_logV": float(uniform_loss),
        "params": int(total_params),
        "approx_forward_macs": {k: int(v) for k, v in macs.items()},
        "args": vars(args),
    })

    max_steps = args.max_epochs * args.steps_per_epoch

    # Training
    model.train()
    t0 = time.time()
    last_log_t = time.time()

    for step in range(start_step, max_steps):
        # minibatch
        xb, yb = get_batch(train_ids, batch_size=args.batch_size, block_size=cfg.block_size)

        optimizer.zero_grad(set_to_none=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(xb, yb)
        else:
            _, loss = model(xb, yb)

        assert loss is not None
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging
        if step % args.log_every == 0:
            now = time.time()
            dt = now - last_log_t
            last_log_t = now
            toks = args.batch_size * cfg.block_size
            toks_per_s = toks / max(1e-9, dt)
            ppl = math.exp(float(loss.item()))
            lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"step {step:6d}/{max_steps} | "
                f"loss {loss.item():.4f} | ppl {ppl:7.2f} | "
                f"lr {lr:.2e} | tok/s {toks_per_s:8.0f}"
            )
            print(msg)

            logger.log({
                "type": "train_step",
                "time": now,
                "step": int(step),
                "loss": float(loss.item()),
                "ppl": float(ppl),
                "lr": float(lr),
                "tok_per_s": float(toks_per_s),
            })

        # Eval + checkpoint
        if (step > 0 and (step % args.eval_every == 0)) or (step == max_steps - 1):
            eval_t0 = time.time()
            train_loss = estimate_loss(
                model, train_ids, batch_size=args.batch_size, block_size=cfg.block_size,
                eval_iters=args.eval_iters, amp_dtype=amp_dtype, device_type=device
            )
            val_loss = estimate_loss(
                model, valid_ids, batch_size=args.batch_size, block_size=cfg.block_size,
                eval_iters=args.eval_iters, amp_dtype=amp_dtype, device_type=device
            )
            eval_dt = time.time() - eval_t0
            print(f"== eval @ step {step} | train {train_loss:.4f} | val {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f} | {eval_dt:.1f}s")

            logger.log({
                "type": "eval",
                "time": time.time(),
                "step": int(step),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_ppl": float(math.exp(val_loss)),
                "eval_seconds": float(eval_dt),
            })

            # Save last
            ckpt_obj = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "model_cfg": asdict(cfg),
                "tokenizer": tokenizer.state_dict(),
                "step": int(step),
                "best_val": float(best_val),
                "val_loss": float(val_loss),
                "rng": {
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
            atomic_torch_save(ckpt_obj, out_dir / "last.pt")

            # Save best
            if val_loss < best_val:
                best_val = float(val_loss)
                ckpt_obj["best_val"] = float(best_val)
                atomic_torch_save(ckpt_obj, out_dir / "best.pt")
                print(f"   (new best) best_val={best_val:.4f}")
                logger.log({
                    "type": "best",
                    "time": time.time(),
                    "step": int(step),
                    "best_val": float(best_val),
                })

    dt = time.time() - t0
    print(f"Done. Best val loss: {best_val:.4f}. Total time: {dt:.1f}s")
    logger.log({"type": "done", "time": time.time(), "best_val": float(best_val), "seconds": float(dt)})
    logger.close()


if __name__ == "__main__":
    main()
