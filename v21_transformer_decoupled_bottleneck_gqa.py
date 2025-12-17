#!/usr/bin/env python3
"""
v21_transformer_decoupled_bottleneck.py

One-file research Transformer that implements, in a runnable way:

1) RoPE (rotary positional embeddings).
2) KV-cache quantization (q8_0 / q4_0) for generation/inference.
3) Decoupled bottleneck attention:
      score = (Q_sem · K_sem^T) + (Q_geo · K_geo^T)
   with RoPE applied only on the geometric path.

Data format: whitespace-separated integer token IDs in a single file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Import instrumentation (optional, for deep analysis)
try:
    from instrumentation import InstrumentationConfig, Analyzer, register_hooks
    HAS_INSTRUMENTATION = True
except ImportError:
    HAS_INSTRUMENTATION = False
    print("Note: instrumentation.py not found. Run with --instrument off or create the file.")


# -----------------------------
# Utils
# -----------------------------

def pick_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Tokenizer (fallback for raw text)
# -----------------------------

class WordTokenizer:
    """
    Minimal word-level tokenizer from v20:
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


def read_tokens(path: str, tokenizer_mode: str = "word") -> Tuple[torch.Tensor, int]:
    """
    Returns (tokens_tensor, vocab_size).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    # 1) If user specifically asked for tiktoken
    if tokenizer_mode == "tiktoken":
        if tiktoken is None:
            raise ImportError("Please `pip install tiktoken` to use --tokenizer tiktoken")
        print(f"Loading {path} with tiktoken (gpt2)...")
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode_ordinary(text) 
        return torch.tensor(ids, dtype=torch.long), enc.n_vocab

    # 2) Try reading as space-separated integers (legacy/pre-processed format)
    # We do this manually to avoid numpy.fromfile
    try:
        # Check if file starts with a number? 
        # Actually, let's just try to read it as text and split into ints if it looks like numbers
        # Reading the whole file into memory as string might be heavy but consistent with v20
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        # heuristic: if the first few tokens are digits, assume integer format
        first_tokens = text.split(maxsplit=5)
        if first_tokens and all(t.isdigit() for t in first_tokens):
             ids = [int(t) for t in text.split()]
             if ids:
                 return torch.tensor(ids, dtype=torch.long), max(ids) + 1
    except ValueError:
        pass
        
    # 3) Fallback: WordTokenizer
    print(f"File {path} could not be read as space-separated integers. Tokenizing as words...")
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    tokenizer = WordTokenizer.build_from_train_text(text)
    ids = tokenizer.encode_text(text)
    print(f"Tokenized {len(ids)} tokens. Vocab size: {tokenizer.vocab_size()}")
    return torch.tensor(ids, dtype=torch.long), tokenizer.vocab_size()



def get_batch(tokens_cpu: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tokens_cpu: 1D CPU tensor
    Returns x,y on `device`.
    """
    if tokens_cpu.device.type != "cpu":
        tokens_cpu = tokens_cpu.cpu()
    n = tokens_cpu.size(0)
    if n <= block_size + 1:
        raise ValueError(f"Need > block_size+1 tokens, got {n} with block_size={block_size}")
    ix = torch.randint(0, n - block_size - 1, (batch_size,), device="cpu")
    x = torch.stack([tokens_cpu[i:i + block_size] for i in ix], dim=0).to(device)
    y = torch.stack([tokens_cpu[i + 1:i + block_size + 1] for i in ix], dim=0).to(device)
    return x, y


def neg_inf(dtype: torch.dtype) -> float:
    # Safer than -inf on some backends.
    return float(torch.finfo(dtype).min)


# -----------------------------
# RoPE
# -----------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE with cached cos/sin.
    """
    def __init__(self, rot_dim: int, base: float = 10000.0):
        super().__init__()
        if rot_dim % 2 != 0:
            raise ValueError(f"rot_dim must be even, got {rot_dim}")
        self.rot_dim = rot_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), str(dtype), int(seq_len))
        if key in self._cache:
            return self._cache[key]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        self._cache[key] = (cos, sin)
        return cos, sin

    def rotate(self, x: torch.Tensor, pos_offset: int) -> torch.Tensor:
        """
        x: (B,H,T,D)
        applies to first rot_dim of D
        """
        B, H, T, D = x.shape
        rot = self.rot_dim
        if rot > D:
            raise ValueError(f"rot_dim {rot} > head_dim {D}")
        cos, sin = self._cos_sin(pos_offset + T, x.device, x.dtype)
        cos = cos[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,rot/2)
        sin = sin[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)

        x_rot = x[..., :rot]
        x_pass = x[..., rot:]

        x1 = x_rot[..., :rot // 2]
        x2 = x_rot[..., rot // 2:rot]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2, x_pass], dim=-1)


# -----------------------------
# KV cache quantization
# -----------------------------

KVCacheKind = Literal["fp16", "fp32", "q8_0", "q4_0"]


@dataclass
class QuantSpec:
    kind: KVCacheKind
    dim: int
    qblock: int
    pad_dim: int
    n_blocks: int


def _qblock_eff(kind: KVCacheKind, dim: int, qblock: int) -> int:
    qb = min(qblock if qblock > 0 else 32, dim)
    if kind == "q4_0":
        if dim < 2:
            raise ValueError("q4_0 cache requires dim >= 2")
        # ensure even qb <= dim (or <= dim-1 if dim is odd)
        max_even = dim if (dim % 2 == 0) else (dim - 1)
        qb = min(qb, max_even)
        if qb < 2:
            qb = 2
        if qb % 2 != 0:
            qb -= 1
    return max(1, qb)


def make_quantspec(kind: KVCacheKind, dim: int, qblock: int) -> QuantSpec:
    qb = _qblock_eff(kind, dim, qblock)
    pad_dim = int(math.ceil(dim / qb) * qb)
    if kind == "q4_0" and (pad_dim % 2 != 0):
        pad_dim += qb
    n_blocks = pad_dim // qb
    return QuantSpec(kind=kind, dim=dim, qblock=qb, pad_dim=pad_dim, n_blocks=n_blocks)


def quantize_q8_0(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (..., dim) float
    returns (q int8 (..., pad_dim), scale fp16 (..., n_blocks))
    """
    if spec.kind != "q8_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)
    amax = x2.abs().amax(dim=-1)  # (N, nb)
    scale = (amax / 127.0).clamp(min=1e-8)
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    q = q.reshape(*orig, pad_dim)
    return q, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_q8_0(q: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "q8_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if q.size(-1) != pad_dim:
        raise ValueError(f"Expected q pad_dim {pad_dim}, got {q.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")
    orig = q.shape[:-1]
    q2 = q.reshape(-1, pad_dim).reshape(-1, nb, qb).to(torch.float32)
    s2 = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q2 * s2
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


def quantize_q4_0(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Q4_0-like: int4 packed into uint8, with fp16 scale per block.
    returns (packed uint8 (..., pad_dim//2), scale fp16 (..., n_blocks))
    """
    if spec.kind != "q4_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)
    amax = x2.abs().amax(dim=-1)
    scale = (amax / 7.0).clamp(min=1e-8)
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16)  # int16 for packing
    u = (q + 8).clamp(0, 15).to(torch.uint8)  # 0..15

    # pack two nibbles per byte via arithmetic (works on MPS too)
    u_even = u[..., 0::2]
    u_odd = u[..., 1::2]
    packed = (u_even * 16) + u_odd  # uint8

    packed = packed.reshape(*orig, pad_dim // 2)
    return packed, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_q4_0(packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "q4_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if packed.size(-1) != pad_dim // 2:
        raise ValueError(f"Expected packed last dim {pad_dim//2}, got {packed.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")

    orig = packed.shape[:-1]
    p2 = packed.reshape(-1, pad_dim // 2).to(torch.int16)
    hi = p2 // 16
    lo = p2 % 16
    u = torch.stack([hi, lo], dim=-1).reshape(-1, pad_dim)  # 0..15
    q = (u - 8).clamp(-8, 7).to(torch.float32)

    q = q.reshape(-1, nb, qb)
    s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q * s
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


class SeqCacheTensor:
    """
    A [B, max_seq_len, dim] sequence tensor stored in fp16/fp32/q8_0/q4_0.
    """
    def __init__(self, *, batch_size: int, max_seq_len: int, dim: int, kind: KVCacheKind, qblock: int, device: torch.device):
        self.kind = kind
        self.device = device
        self.spec = make_quantspec(kind, dim, qblock)
        self.pos = 0
        self.max_seq_len = max_seq_len

        if kind in ("fp16", "fp32"):
            dtype = torch.float16 if kind == "fp16" else torch.float32
            self.buf = torch.empty((batch_size, max_seq_len, dim), device=device, dtype=dtype)
            self.q = None
            self.s = None
        elif kind == "q8_0":
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim), device=device, dtype=torch.int8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
        elif kind == "q4_0":
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim // 2), device=device, dtype=torch.uint8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
        else:
            raise ValueError(kind)

    def append(self, x_new: torch.Tensor) -> int:
        """
        x_new: (B, T_new, dim) float
        returns old_pos (start index)
        """
        B, Tn, D = x_new.shape
        if D != self.spec.dim:
            raise ValueError(f"dim mismatch: expected {self.spec.dim}, got {D}")
        if self.pos + Tn > self.max_seq_len:
            raise ValueError(f"Cache overflow: pos {self.pos} + {Tn} > max {self.max_seq_len}")
        old = self.pos

        if self.kind in ("fp16", "fp32"):
            self.buf[:, old:old + Tn] = x_new.to(self.buf.dtype)
        elif self.kind == "q8_0":
            q, s = quantize_q8_0(x_new, self.spec)
            self.q[:, old:old + Tn] = q
            self.s[:, old:old + Tn] = s
        elif self.kind == "q4_0":
            q, s = quantize_q4_0(x_new, self.spec)
            self.q[:, old:old + Tn] = q
            self.s[:, old:old + Tn] = s
        else:
            raise ValueError(self.kind)

        self.pos += Tn
        return old

    def get(self, length: Optional[int] = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        returns (B, length, dim) float32 (or `dtype`)
        """
        L = self.pos if length is None else int(length)
        if L > self.pos:
            raise ValueError(f"Requested length {L} > cached length {self.pos}")
        if self.kind in ("fp16", "fp32"):
            return self.buf[:, :L].to(dtype)
        if self.kind == "q8_0":
            x = dequantize_q8_0(self.q[:, :L], self.s[:, :L], self.spec)
            return x.to(dtype)
        if self.kind == "q4_0":
            x = dequantize_q4_0(self.q[:, :L], self.s[:, :L], self.spec)
            return x.to(dtype)
        raise ValueError(self.kind)


class LayerKVCache:
    def __init__(self, *, batch_size: int, max_seq_len: int, k_dim: int, v_dim: int, kind: KVCacheKind, qblock: int, device: torch.device):
        self.k = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_dim, kind=kind, qblock=qblock, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, kind=kind, qblock=qblock, device=device)

    @property
    def pos(self) -> int:
        return self.k.pos

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        old = self.k.append(k_new)
        old2 = self.v.append(v_new)
        if old != old2:
            raise RuntimeError("K/V cache desync")
        return old

    def get(self, *, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k.get(dtype=dtype), self.v.get(dtype=dtype)


class DecoupledLayerKVCache:
    def __init__(self, *, batch_size: int, max_seq_len: int, k_sem_dim: int, k_geo_dim: int, v_dim: int, kind: KVCacheKind, qblock: int, device: torch.device):
        self.k_sem = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_sem_dim, kind=kind, qblock=qblock, device=device)
        self.k_geo = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_geo_dim, kind=kind, qblock=qblock, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, kind=kind, qblock=qblock, device=device)

    @property
    def pos(self) -> int:
        return self.k_sem.pos

    def append(self, k_sem_new: torch.Tensor, k_geo_new: torch.Tensor, v_new: torch.Tensor) -> int:
        old = self.k_sem.append(k_sem_new)
        old2 = self.k_geo.append(k_geo_new)
        old3 = self.v.append(v_new)
        if not (old == old2 == old3):
            raise RuntimeError("Decoupled cache desync")
        return old

    def get(self, *, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.k_sem.get(dtype=dtype), self.k_geo.get(dtype=dtype), self.v.get(dtype=dtype)


# -----------------------------
# Model
# -----------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int

    n_layer: int = 6
    n_head: int = 8
    kv_head: Optional[int] = None  # for GQA: number of KV heads (defaults to n_head)
    d_model: int = 512
    d_ff: int = 2048

    embed_dim: int = 512  # lexical bottleneck if < d_model

    attn_mode: Literal["standard", "bottleneck", "decoupled", "gqa"] = "bottleneck"
    attn_dim: int = 512    # total V dim (and Q/K dim for bottleneck)
    sem_dim: int = 32      # total semantic Q/K dim across heads (decoupled)
    geo_dim: int = 64      # total geometric Q/K dim across heads (decoupled)

    rope: bool = True
    rope_base: float = 10000.0

    tie_qk: bool = False
    null_attn: bool = False
    learned_temp: bool = True

    mlp: Literal["swiglu", "gelu"] = "swiglu"
    dropout: float = 0.0


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.mlp == "swiglu":
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        elif cfg.mlp == "gelu":
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        else:
            raise ValueError(cfg.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.mlp == "swiglu":
            x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        else:
            x = self.w2(F.gelu(self.w1(x)))
        return self.drop(x)


class DecoupledBottleneckAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_head
        self.H = H
        self.H_kv = H
        self.group_size = 1
        self.drop = nn.Dropout(cfg.dropout)

        def must_div(name: str, total: int) -> int:
            if total % H != 0:
                raise ValueError(f"{name} ({total}) must be divisible by n_head ({H})")
            return total // H

        if cfg.attn_mode == "standard":
            qk_dim = cfg.d_model
            v_dim = cfg.d_model
            self.qk_head_dim = must_div("d_model", qk_dim)
            self.v_head_dim = must_div("d_model", v_dim)

            self.q_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.k_proj = self.q_proj if cfg.tie_qk else nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, v_dim, bias=False)
            self.out_proj = nn.Linear(v_dim, cfg.d_model, bias=False)

            self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
            self.sem_head_dim = self.geo_head_dim = None
            self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_null = nn.Parameter(torch.zeros(1, 1, qk_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, v_dim)) if cfg.null_attn else None
            self.k_sem_null = self.k_geo_null = None

        elif cfg.attn_mode == "bottleneck":
            qk_dim = cfg.attn_dim
            v_dim = cfg.attn_dim
            self.qk_head_dim = must_div("attn_dim", qk_dim)
            self.v_head_dim = must_div("attn_dim", v_dim)
            if cfg.rope and (self.qk_head_dim % 2 != 0):
                raise ValueError("RoPE needs even head dim; pick attn_dim divisible by 2*n_head")

            self.q_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.k_proj = self.q_proj if cfg.tie_qk else nn.Linear(cfg.d_model, qk_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, v_dim, bias=False)
            self.out_proj = nn.Linear(v_dim, cfg.d_model, bias=False)

            self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
            self.sem_head_dim = self.geo_head_dim = None
            self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_null = nn.Parameter(torch.zeros(1, 1, qk_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, v_dim)) if cfg.null_attn else None
            self.k_sem_null = self.k_geo_null = None

        elif cfg.attn_mode == "gqa":
            # Grouped-Query Attention (GQA): Q has H heads, K/V has H_kv heads shared across groups.
            if cfg.attn_dim is None:
                raise ValueError("gqa requires --attn-dim")
            kv_head = cfg.kv_head if cfg.kv_head is not None else H
            if kv_head <= 0:
                raise ValueError("kv_head must be > 0")
            if H % kv_head != 0:
                raise ValueError(f"gqa requires n_head % kv_head == 0 (got n_head={H}, kv_head={kv_head})")
            self.H_kv = kv_head
            self.group_size = H // kv_head

            self.qk_head_dim = must_div("attn_dim", cfg.attn_dim)
            self.v_head_dim = self.qk_head_dim

            kv_dim = kv_head * self.qk_head_dim

            if cfg.rope and (self.qk_head_dim % 2 != 0):
                raise ValueError("RoPE requires an even head dim. Choose attn_dim divisible by 2*n_head.")

            if cfg.tie_qk:
                raise ValueError(
                    "tie_qk is not supported for gqa unless kv_head == n_head (use --attn-mode standard)."
                )

            self.q_proj = nn.Linear(cfg.d_model, cfg.attn_dim, bias=False)
            self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
            self.out_proj = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

            self.q_sem_proj = self.k_sem_proj = self.v_sem_proj = None
            self.q_geo_proj = self.k_geo_proj = self.v_geo_proj = None

            self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_null = nn.Parameter(torch.zeros(1, 1, cfg.attn_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, cfg.attn_dim)) if cfg.null_attn else None

            self.k_sem_null = self.v_sem_null = None
            self.k_geo_null = self.v_geo_null = None

        elif cfg.attn_mode == "decoupled":
            self.sem_head_dim = must_div("sem_dim", cfg.sem_dim)
            self.geo_head_dim = must_div("geo_dim", cfg.geo_dim)
            self.v_head_dim = must_div("attn_dim", cfg.attn_dim)
            if cfg.rope and (self.geo_head_dim % 2 != 0):
                raise ValueError("RoPE needs even geo_head_dim; pick geo_dim divisible by 2*n_head")

            self.q_sem = nn.Linear(cfg.d_model, cfg.sem_dim, bias=False)
            self.k_sem = self.q_sem if cfg.tie_qk else nn.Linear(cfg.d_model, cfg.sem_dim, bias=False)
            self.q_geo = nn.Linear(cfg.d_model, cfg.geo_dim, bias=False)
            self.k_geo = self.q_geo if cfg.tie_qk else nn.Linear(cfg.d_model, cfg.geo_dim, bias=False)

            self.v_proj = nn.Linear(cfg.d_model, cfg.attn_dim, bias=False)
            self.out_proj = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

            self.q_proj = self.k_proj = None
            self.qk_head_dim = None
            self.rotary = RotaryEmbedding(self.geo_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_sem_null = nn.Parameter(torch.zeros(1, 1, cfg.sem_dim)) if cfg.null_attn else None
            self.k_geo_null = nn.Parameter(torch.zeros(1, 1, cfg.geo_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, cfg.attn_dim)) if cfg.null_attn else None
            self.k_null = None
        else:
            raise ValueError(cfg.attn_mode)

        self.logit_scale = nn.Parameter(torch.zeros(H)) if cfg.learned_temp else None

    def _shape(self, x: torch.Tensor, head_dim: int, H: Optional[int] = None) -> torch.Tensor:
        # (B,T,H*hd)->(B,H,T,hd)
        B, T, D = x.shape
        H = self.H if H is None else H
        return x.view(B, T, H, head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,T,hd)->(B,T,H*hd)
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def _apply_logit_scale(self, scores: torch.Tensor) -> torch.Tensor:
        if self.logit_scale is None:
            return scores
        return scores * torch.exp(self.logit_scale.view(1, -1, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        cache: Optional[Any],
        pos_offset: int,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        x: (B,T,d_model)
        attn_mask only used when cache is None (training full attention)
        cache:
          - None (training)
          - LayerKVCache (standard/bottleneck)
          - DecoupledLayerKVCache (decoupled)
        pos_offset: absolute position for RoPE for x[:,0]
        """
        cfg = self.cfg
        B, T, _ = x.shape
        ninfty = neg_inf(x.dtype)

        if cfg.attn_mode in ("standard", "bottleneck"):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            qh = self._shape(q, self.qk_head_dim)
            kh = self._shape(k, self.qk_head_dim)
            vh = self._shape(v, self.v_head_dim)

            if self.rotary is not None:
                qh = self.rotary.rotate(qh, pos_offset)
                kh = self.rotary.rotate(kh, pos_offset)

            if cache is None:
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)  # (B,H,T,T)
                scores = self._apply_logit_scale(scores)

                if cfg.null_attn:
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                    s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                    s_null = self._apply_logit_scale(s_null)
                    scores = torch.cat([s_null, scores], dim=-1)  # (B,H,T,1+T)
                    if attn_mask is not None:
                        extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                        keep = torch.cat([extra, attn_mask], dim=-1)
                        scores = scores.masked_fill(~keep, ninfty)
                    attn = F.softmax(scores, dim=-1)
                    attn = self.drop(attn)
                    self.last_attn = attn  # Save for visualization
                    vals = torch.cat([v_null, vh], dim=-2)
                    out = torch.matmul(attn, vals)
                else:
                    if attn_mask is not None:
                        scores = scores.masked_fill(~attn_mask, ninfty)
                    attn = F.softmax(scores, dim=-1)
                    attn = self.drop(attn)
                    self.last_attn = attn  # Save for visualization
                    out = torch.matmul(attn, vh)

                y = self.out_proj(self._merge(out))
                return y, None

            old_len = cache.pos
            cache.append(self._merge(kh), self._merge(vh))
            k_all, v_all = cache.get(dtype=x.dtype)
            L = k_all.size(1)
            kh_all = self._shape(k_all, self.qk_head_dim)
            vh_all = self._shape(v_all, self.v_head_dim)

            scores = torch.matmul(qh, kh_all.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)  # (B,H,T,L)
            scores = self._apply_logit_scale(scores)

            if T > 1:
                key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                s_null = self._apply_logit_scale(s_null)
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                self.last_attn = attn  # Save for visualization
                vals = torch.cat([v_null, vh_all], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                self.last_attn = attn  # Save for visualization
                out = torch.matmul(attn, vh_all)

            y = self.out_proj(self._merge(out))
            return y, cache

        if cfg.attn_mode == "gqa":
            # Grouped-Query Attention forward pass. Q has H heads; K/V have H_kv heads shared across groups.
            q = self.q_proj(x)  # (B,T,attn_dim)
            k = self.k_proj(x)  # (B,T,kv_dim)
            v = self.v_proj(x)  # (B,T,kv_dim)

            qh = self._shape(q, self.qk_head_dim, H=self.H)
            kh = self._shape(k, self.qk_head_dim, H=self.H_kv)
            vh = self._shape(v, self.v_head_dim, H=self.H_kv)

            if self.rotary is not None:
                qh = self.rotary.rotate(qh, pos_offset)
                kh = self.rotary.rotate(kh, pos_offset)

            if cache is None:
                # Expand K/V to match Q-heads.
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)

                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = self._apply_logit_scale(scores)

                if cfg.null_attn:
                    # Null token is per-Q-head.
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H)
                    s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                    s_null = self._apply_logit_scale(s_null)

                    scores = torch.cat([s_null, scores], dim=-1)
                    if attn_mask is not None:
                        extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                        keep = torch.cat([extra, attn_mask], dim=-1)
                        scores = scores.masked_fill(~keep, ninfty)

                    attn = F.softmax(scores, dim=-1)
                    attn = self.drop(attn)
                    self.last_attn = attn  # Save for visualization

                    vals = torch.cat([v_null, vh_rep], dim=-2)
                    out = torch.matmul(attn, vals)
                else:
                    if attn_mask is not None:
                        scores = scores.masked_fill(~attn_mask, ninfty)
                    attn = F.softmax(scores, dim=-1)
                    attn = self.drop(attn)
                    self.last_attn = attn  # Save for visualization
                    out = torch.matmul(attn, vh_rep)

                y = self.out_proj(self._merge(out))
                return y, None

            # Cached path (incremental decoding)
            old_len = cache.pos
            cache.append(self._merge(kh), self._merge(vh))
            k_all, v_all = cache.get(dtype=x.dtype)
            L = k_all.size(1)

            kh_all = self._shape(k_all, self.qk_head_dim, H=self.H_kv)
            vh_all = self._shape(v_all, self.v_head_dim, H=self.H_kv)
            kh_rep = kh_all.repeat_interleave(self.group_size, dim=1)
            vh_rep = vh_all.repeat_interleave(self.group_size, dim=1)

            scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            scores = self._apply_logit_scale(scores)

            if T > 1:
                # Prefill: causal mask for the concatenated (old + new) cache.
                key_pos = torch.arange(L, device=x.device)[None, None, None, :]
                q_pos = (old_len + torch.arange(T, device=x.device))[None, None, :, None]
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                s_null = self._apply_logit_scale(s_null)

                scores = torch.cat([s_null, scores], dim=-1)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                self.last_attn = attn  # Save for visualization

                vals = torch.cat([v_null, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                self.last_attn = attn  # Save for visualization
                out = torch.matmul(attn, vh_rep)

            y = self.out_proj(self._merge(out))
            return y, cache

        # decoupled
        q_sem = self.q_sem(x)
        k_sem = self.k_sem(x)
        q_geo = self.q_geo(x)
        k_geo = self.k_geo(x)
        v = self.v_proj(x)

        qsh = self._shape(q_sem, self.sem_head_dim)
        ksh = self._shape(k_sem, self.sem_head_dim)
        qgh = self._shape(q_geo, self.geo_head_dim)
        kgh = self._shape(k_geo, self.geo_head_dim)
        vh = self._shape(v, self.v_head_dim)

        if self.rotary is not None:
            qgh = self.rotary.rotate(qgh, pos_offset)
            kgh = self.rotary.rotate(kgh, pos_offset)

        if cache is None:
            sem = torch.matmul(qsh, ksh.transpose(-2, -1)) / math.sqrt(self.sem_head_dim)
            geo = torch.matmul(qgh, kgh.transpose(-2, -1)) / math.sqrt(self.geo_head_dim)
            # Cache path scores for instrumentation
            self.last_sem_scores = sem
            self.last_geo_scores = geo
            scores = self._apply_logit_scale(sem + geo)

            if cfg.null_attn:
                ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
                kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
                vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)

                s_null = (
                    torch.matmul(qsh, ksn.transpose(-2, -1)) / math.sqrt(self.sem_head_dim)
                    + torch.matmul(qgh, kgn.transpose(-2, -1)) / math.sqrt(self.geo_head_dim)
                )
                s_null = self._apply_logit_scale(s_null)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                self.last_attn = attn  # Save for visualization
                vals = torch.cat([vn, vh], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                self.last_attn = attn  # Save for visualization
                out = torch.matmul(attn, vh)

            y = self.out_proj(self._merge(out))
            return y, None

        old_len = cache.pos
        cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
        k_sem_all, k_geo_all, v_all = cache.get(dtype=x.dtype)
        L = k_sem_all.size(1)

        ksh_all = self._shape(k_sem_all, self.sem_head_dim)
        kgh_all = self._shape(k_geo_all, self.geo_head_dim)
        vh_all = self._shape(v_all, self.v_head_dim)

        sem = torch.matmul(qsh, ksh_all.transpose(-2, -1)) / math.sqrt(self.sem_head_dim)
        geo = torch.matmul(qgh, kgh_all.transpose(-2, -1)) / math.sqrt(self.geo_head_dim)
        # Cache path scores for instrumentation
        self.last_sem_scores = sem
        self.last_geo_scores = geo
        scores = self._apply_logit_scale(sem + geo)

        if T > 1:
            key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
            q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
            keep = key_pos <= q_pos
            scores = scores.masked_fill(~keep, ninfty)

        if cfg.null_attn:
            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)

            s_null = (
                torch.matmul(qsh, ksn.transpose(-2, -1)) / math.sqrt(self.sem_head_dim)
                + torch.matmul(qgh, kgn.transpose(-2, -1)) / math.sqrt(self.geo_head_dim)
            )
            s_null = self._apply_logit_scale(s_null)
            scores = torch.cat([s_null, scores], dim=-1)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            self.last_attn = attn  # Save for visualization
            vals = torch.cat([vn, vh_all], dim=-2)
            out = torch.matmul(attn, vals)
        else:
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            self.last_attn = attn  # Save for visualization
            out = torch.matmul(attn, vh_all)

        y = self.out_proj(self._merge(out))
        return y, cache


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = DecoupledBottleneckAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor], cache: Optional[Any], pos_offset: int) -> Tuple[torch.Tensor, Optional[Any]]:
        a, cache = self.attn(self.ln1(x), attn_mask=attn_mask, cache=cache, pos_offset=pos_offset)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # lexical bottleneck
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.emb_in = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False) if cfg.embed_dim != cfg.d_model else None
        self.emb_out = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False) if cfg.embed_dim != cfg.d_model else None

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, *, caches: Optional[List[Any]] = None, pos_offset: int = 0) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        B, T = idx.shape
        if caches is None and T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}. Increase --block.")
        x = self.tok_emb(idx)
        if self.emb_in is not None:
            x = self.emb_in(x)
        x = self.drop(x)

        attn_mask = None
        if caches is None:
            attn_mask = self.causal_mask[:, :, :T, :T]

        new_caches: Optional[List[Any]] = [] if caches is not None else None
        for i, blk in enumerate(self.blocks):
            layer_cache = caches[i] if caches is not None else None
            x, layer_cache = blk(x, attn_mask=attn_mask, cache=layer_cache, pos_offset=pos_offset)
            if caches is not None:
                new_caches.append(layer_cache)

        x = self.ln_f(x)
        if self.emb_out is not None:
            x_small = self.emb_out(x)
        else:
            x_small = x
        logits = x_small @ self.tok_emb.weight.t()
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
    ) -> torch.Tensor:
        self.eval()
        device = prompt.device
        B, T0 = prompt.shape
        max_seq = T0 + max_new_tokens

        caches: List[Any] = []
        for _ in range(self.cfg.n_layer):
            if self.cfg.attn_mode == "decoupled":
                caches.append(
                    DecoupledLayerKVCache(
                        batch_size=B,
                        max_seq_len=max_seq,
                        k_sem_dim=self.cfg.sem_dim,
                        k_geo_dim=self.cfg.geo_dim,
                        v_dim=self.cfg.attn_dim,
                        kind=kv_cache,
                        qblock=kv_qblock,
                        device=device,
                    )
                )
            else:
                if self.cfg.attn_mode == "standard":
                    k_dim = v_dim = self.cfg.d_model
                elif self.cfg.attn_mode == "bottleneck":
                    k_dim = v_dim = self.cfg.attn_dim
                elif self.cfg.attn_mode == "gqa":
                    assert self.cfg.attn_dim is not None
                    # Q has n_head heads, K/V has kv_head heads -> cache stores only kv_head*head_dim.
                    head_dim = self.cfg.attn_dim // self.cfg.n_head
                    kv_head = self.cfg.kv_head if self.cfg.kv_head is not None else self.cfg.n_head
                    k_dim = v_dim = kv_head * head_dim
                else:
                    raise ValueError(f"Unknown attn_mode for KV cache: {self.cfg.attn_mode}")

                caches.append(
                    LayerKVCache(
                        batch_size=B,
                        max_seq_len=max_seq,
                        k_dim=k_dim,
                        v_dim=v_dim,
                        kind=kv_cache,
                        qblock=kv_qblock,
                        device=device,
                    )
                )

        logits, caches = self(prompt, caches=caches, pos_offset=0)

        out = prompt
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], neg_inf(next_logits.dtype))
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)
            logits, caches = self(next_id, caches=caches, pos_offset=out.size(1) - 1)

        return out


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    *,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split, tok in [("train", train_tokens), ("val", val_tokens)]:
        losses: List[float] = []
        for _ in range(eval_iters):
            x, y = get_batch(tok, batch_size, block_size, device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out["train"], out["val"]


def save_ckpt(out_dir: str, name: str, model: GPT, cfg: ModelConfig, step: int, best_val: float) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(
        {
            "config": asdict(cfg),
            "model": model.state_dict(),
            "step": step,
            "best_val": best_val,
        },
        path,
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=512)

    ap.add_argument("--attn-mode", type=str, default="bottleneck", choices=["standard", "bottleneck", "decoupled", "gqa"])
    ap.add_argument("--kv-head", type=int, default=None, help="For --attn-mode gqa: number of KV heads (must divide n_head). Default = n_head")
    ap.add_argument("--attn-dim", type=int, default=512)
    ap.add_argument("--sem-dim", type=int, default=32)
    ap.add_argument("--geo-dim", type=int, default=64)
    ap.add_argument("--no-rope", action="store_true")
    ap.add_argument("--rope-base", type=float, default=10000.0)
    ap.add_argument("--tie-qk", action="store_true")
    ap.add_argument("--null-attn", action="store_true")
    ap.add_argument("--no-learned-temp", action="store_true")

    ap.add_argument("--mlp", type=str, default="swiglu", choices=["swiglu", "gelu"])
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)

    ap.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    ap.add_argument("--ckpt", type=str, default=None)

    ap.add_argument("--prompt-tokens", type=str, default="0")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--kv-cache", type=str, default="fp16", choices=["fp16", "fp32", "q8_0", "q4_0"])
    ap.add_argument("--kv-qblock", type=int, default=32)
    ap.add_argument("--tokenizer", type=str, default="word", choices=["word", "tiktoken"])
    
    # Instrumentation
    ap.add_argument("--instrument", type=str, default="medium", 
                    choices=["off", "light", "medium", "heavy"],
                    help="Instrumentation level for deep analysis")
    ap.add_argument("--analysis-every", type=int, default=100,
                    help="Steps between deep analysis (for medium/heavy)")

    args = ap.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)

    # If user wants tiktoken but hasn't installed it, we might fail inside read_tokens.
    # However, if they didn't specify, we default to "word".
    # If they want to force "tiktoken", they can.
    
    if args.tokenizer == "tiktoken" and tiktoken is None:
         print("Warning: --tokenizer tiktoken requested but tiktoken not installed. Falling back to word tokenizer.")
         args.tokenizer = "word"
    
    if args.mode == "train":
        if not args.data or not args.out_dir:
             ap.error("the following arguments are required for training: --data, --out-dir")

        # returns torch tensor now
        tok_tensor, vocab = read_tokens(args.data, tokenizer_mode=args.tokenizer)
        
        n_total = tok_tensor.numel()
        n_val = int(n_total * 0.1)
        train_tokens = tok_tensor[:-n_val]
        val_tokens = tok_tensor[-n_val:]

        print(f"Device: {device}")
        print(f"Tokens: train {train_tokens.numel():,} | val {val_tokens.numel():,} | vocab {vocab:,}")
        print(f"Uniform baseline loss log(V): {math.log(vocab):.4f} (ppl ~ {vocab})")
    else:
        # Sampling mode doesn't strictly need data/out-dir if loading from ckpt,
        # but we need vocab size.
        # Ideally, vocab size should be stored in checkpoint config.
        # For now, let's load the checkpoint first to get config.
        pass

    if args.ckpt:
         print(f"Loading checkpoint from {args.ckpt}...")
         ckpt = torch.load(args.ckpt, map_location=device)
         # If config is in checkpoint, use it.
         if "config" in ckpt:
              cfg_dict = ckpt["config"]
              # Allow overriding some inference-time settings or if we want to change dropout etc (though not relevant for sample)
              # But really we should use the trained config.
              # Let's reconstruct ModelConfig from it.
              cfg = ModelConfig(**cfg_dict)
              vocab = cfg.vocab_size
         else:
              # Fallback if config not in ckpt (shouldn't happen with this script)
               if args.mode == "sample" and not args.data:
                    # If we can't find vocab size, we are stuck unless user provides it.
                    # But wait, we can infer it from the model weight shapes later?
                    # Simpler: require data for now if not in ckpt, or just trust the args if provided?
                    # The script previously loaded data even for sample to get vocab.
                    pass

    if args.mode == "train":
        cfg = ModelConfig(
            vocab_size=vocab,
            block_size=args.block,
            n_layer=args.layers,
            n_head=args.n_head,
            kv_head=args.kv_head,
            d_model=args.d_model,
            d_ff=args.d_ff,
            embed_dim=args.embed_dim,
            attn_mode=args.attn_mode,
            attn_dim=args.attn_dim,
            sem_dim=args.sem_dim,
            geo_dim=args.geo_dim,
            rope=not args.no_rope,
            rope_base=args.rope_base,
            tie_qk=args.tie_qk,
            null_attn=args.null_attn,
            learned_temp=not args.no_learned_temp,
            mlp=args.mlp,
            dropout=args.dropout,
        )

    model = GPT(cfg).to(device)

    if args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required for --mode sample")
        # We already loaded ckpt above if it exists
        model.load_state_dict(ckpt["model"], strict=True)
        
        # Tokenizer logic for prompt
        # If using word tokenizer, we need the vocab/stoi.
        # But this script is "stateless" about the tokenizer unless we saved it.
        # We didn't save tokenizer state in save_ckpt! We only saved config/model.
        # So we HAVE to load the data to rebuild the tokenizer to encode the prompt correctly?
        # OR we just assume the user provides raw IDs.
        
        # The script defaults prompt-tokens to "0", which implies raw IDs.
        # "1 2 3 4 5"
        
        try:
             prompt_ids = [int(t) for t in args.prompt_tokens.strip().split()]
        except ValueError:
             # If prompt is text, we need a tokenizer.
             # If we don't have data, we can't build the word tokenizer.
             # If using tiktoken, we can.
             if args.tokenizer == "tiktoken":
                  if tiktoken is None:
                        raise ImportError("tiktoken needed for text prompts")
                  enc = tiktoken.get_encoding("gpt2")
                  prompt_ids = enc.encode_ordinary(args.prompt_tokens)
             else:
                  raise ValueError("Cannot parse prompt as integers, and word tokenizer requires --data to build vocab.")

        prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        
        print(f"Generating {args.max_new_tokens} tokens...")
        t0 = time.time()
        out = model.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            kv_cache=args.kv_cache,
            kv_qblock=args.kv_qblock,
        )
        dt = time.time() - t0
        print(f"Time: {dt:.2f}s | Tok/s: {args.max_new_tokens/dt:.2f}")
        
        # Decode output
        out_ids = out[0].tolist()
        if args.tokenizer == "tiktoken":
             enc = tiktoken.get_encoding("gpt2")
             print(enc.decode(out_ids))
        else:
             print(out_ids)
             
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize deep instrumentation (if available)
    analyzer = None
    hooks = []
    if HAS_INSTRUMENTATION and args.instrument != "off":
        inst_config = InstrumentationConfig(
            level=args.instrument,
            analysis_every=args.analysis_every,
        )
        analyzer = Analyzer(inst_config, args.out_dir, asdict(cfg), args)
        hooks = register_hooks(model, analyzer)
        print(f"Instrumentation: {args.instrument} (analysis every {args.analysis_every} steps)")
        
        # Measure actual memory usage
        analyzer.measure_memory(model, args.batch_size, cfg.block_size)
    else:
        print("Instrumentation: off")
    
    best_val = float("inf")
    t_start = time.time()
    t_eval0 = time.time()
    tok_count = 0
    dt_acc = 0.0

    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            tr, va = estimate_loss(
                model,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                block_size=cfg.block_size,
                device=device,
            )
            ppl = math.exp(va) if va < 20 else float("inf")
            elapsed = time.time() - t_eval0
            is_best = va < best_val
            
            print(f"== eval step {step} | train {tr:.4f} | val {va:.4f} | val_ppl {ppl:.2f} | {elapsed:.1f}s")
            if analyzer:
                analyzer.log_eval(step, tr, va, ppl, elapsed, is_best)
            
            if is_best:
                best_val = va
                save_ckpt(args.out_dir, "best.pt", model, cfg, step, best_val)
                if analyzer:
                    analyzer.log_best(step, best_val)
                print(f"   (new best) {best_val:.4f}")
            t_eval0 = time.time()

        if step == args.steps:
            break

        x, y = get_batch(train_tokens, args.batch_size, cfg.block_size, device)
        t0 = time.time()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        
        # Run deep analysis (attention, gradients, representations)
        if analyzer:
            analyzer.analyze_step(step, model)
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        dt = time.time() - t0

        tok_count += x.numel()
        dt_acc += dt
        if step % 50 == 0:
            tok_s = tok_count / max(dt_acc, 1e-9)
            ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
            print(f"step {step:6d}/{args.steps} | loss {loss.item():.4f} | ppl {ppl:8.2f} | tok/s {tok_s:8.0f}")
            if analyzer:
                analyzer.log_train_step(step, loss.item(), ppl, tok_s, args.lr)
            tok_count = 0
            dt_acc = 0.0

    save_ckpt(args.out_dir, "last.pt", model, cfg, args.steps, best_val)
    
    # Finalize instrumentation with auto-visualization
    total_time = time.time() - t_start
    if analyzer:
        analyzer.finalize(best_val)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    print(f"Done. best_val={best_val:.4f} | total_time={total_time:.1f}s")


if __name__ == "__main__":
    main()
