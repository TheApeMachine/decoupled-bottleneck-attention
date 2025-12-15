#!/usr/bin/env python3
""" 
v24_transformer_decoupled_bottleneck_flash2pass.py

One-file research Transformer that implements, in a runnable way:

1) RoPE (rotary positional embeddings).
2) KV-cache quantization (q8_0 / q4_0 / nf4) for generation/inference.
3) Bottleneck and Decoupled Bottleneck attention:
      score = (Q_sem · K_sem^T) + (Q_geo · K_geo^T)
   with RoPE applied only on the geometric path.

"Survive scale" upgrades in this v24 variant:

  A) Streaming ("online softmax") decode that dequantizes only small sequence blocks.
  B) Optional fused GPU kernels (Triton, if installed).
     - v23: 1-pass fused decode update (dequant + online-softmax update per block).
     - v24: 2-pass "FlashAttention-style" split-K decode:
            Pass 1: parallel partitions compute (m_i, l_i, O_i) per partition.
            Pass 2: reduce partitions into a single (m, l, O) for the row.
            This gives sequence-length parallelism (useful when you have 1 query token, huge KV).

Data format: whitespace-separated integer token IDs in a single file.
"""

from __future__ import annotations

import argparse
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


# Optional: fused kernels via Triton.
# This file runs without Triton; if you install it and run on CUDA, decode can use fused dequant+attn updates.
TRITON_AVAILABLE = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    TRITON_AVAILABLE = True
except Exception:
    triton = None  # type: ignore
    tl = None  # type: ignore


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
# -----------------------------
# KV cache quantization
# -----------------------------

# NOTE: This research file keeps the quantization formats deliberately simple and kernel-friendly:
#   - q8_0: per-block symmetric int8 with fp16 scale (absmax / 127)
#   - q4_0: per-block symmetric int4 (packed into uint8) with fp16 scale (absmax / 7)
#   - nf4 : per-block "NormalFloat4" codebook (packed into uint8) with fp16 scale (absmax)
#
# For production-scale inference, you'd normally use fused kernels to avoid dequantizing the whole cache.
# This file adds a streaming ("online softmax") decode path that dequantizes in small sequence blocks.

KVCacheKind = Literal["fp16", "fp32", "q8_0", "q4_0", "nf4"]


@dataclass(frozen=True)
class QuantSpec:
    kind: KVCacheKind
    dim: int
    qblock: int
    pad_dim: int
    n_blocks: int


@dataclass(frozen=True)
class KVCacheTensorConfig:
    kind: KVCacheKind = "fp16"
    qblock: int = 32
    residual_len: int = 0   # keep a small fp16 "hot" window for the newest tokens (ring-buffer)


def _qblock_eff(kind: KVCacheKind, dim: int, qblock: int) -> int:
    qb = min(qblock if qblock > 0 else 32, dim)
    if kind in ("q4_0", "nf4"):
        if dim < 2:
            raise ValueError(f"{kind} cache requires dim >= 2")
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
    if kind in ("q4_0", "nf4") and (pad_dim % 2 != 0):
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


# NF4 codebook from QLoRA Appendix E / bitsandbytes (normalized to [-1, 1]).
# (We keep it explicit here so the file stays self-contained.)
NF4_LEVELS = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)


def quantize_nf4(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NF4: non-uniform 4-bit codebook quantization.

    x: (..., dim) float
    returns (packed uint8 (..., pad_dim//2), scale fp16 (..., n_blocks))

    NOTE: This is implemented in pure PyTorch for research (not fast vs kernel implementations).
    """
    if spec.kind != "nf4":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)  # (N, nb, qb)
    amax = x2.abs().amax(dim=-1)  # (N, nb)
    scale = amax.clamp(min=1e-8)  # map into [-1, 1] via absmax

    y = (x2 / scale.unsqueeze(-1)).clamp(-1.0, 1.0)  # (N, nb, qb)
    levels = NF4_LEVELS.to(device=y.device, dtype=torch.float32)

    # nearest-neighbor assignment into 16-level LUT
    # (N, nb, qb, 1) - (16,) -> (N, nb, qb, 16)
    diff = (y.to(torch.float32).unsqueeze(-1) - levels).abs()
    idx = diff.argmin(dim=-1).to(torch.uint8)  # 0..15

    # pack two 4-bit indices per byte (same packing layout as q4_0)
    idx_even = idx[..., 0::2]
    idx_odd = idx[..., 1::2]
    packed = (idx_even * 16) + idx_odd  # uint8

    packed = packed.reshape(*orig, pad_dim // 2)
    return packed, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_nf4(packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "nf4":
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
    idx = torch.stack([hi, lo], dim=-1).reshape(-1, pad_dim).to(torch.long)  # 0..15

    levels = NF4_LEVELS.to(device=packed.device, dtype=torch.float32)
    q = levels[idx]  # (N, pad_dim)

    q = q.reshape(-1, nb, qb)
    s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q * s
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


# -----------------------------
# Optional Triton fused kernels (decode only)
# -----------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _kv_decode_update_decoupled_q4q8q4(
        q_sem_ptr, q_geo_ptr,
        k_sem_q_ptr, k_sem_s_ptr,
        k_geo_q_ptr, k_geo_s_ptr,
        v_q_ptr, v_s_ptr,
        m_ptr, d_ptr, o_ptr,
        start: tl.int32,
        # runtime lengths
        L_prefix: tl.int32,
        # meta
        H: tl.constexpr,
        HD_SEM: tl.constexpr,
        HD_GEO: tl.constexpr,
        HD_V: tl.constexpr,
        QBLOCK_SEM: tl.constexpr,
        QBLOCK_GEO: tl.constexpr,
        QBLOCK_V: tl.constexpr,
        SEM_SCALE: tl.constexpr,
        GEO_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SUBBLOCKS: tl.constexpr,
        # strides
        stride_qsem_b: tl.constexpr, stride_qsem_h: tl.constexpr,
        stride_qgeo_b: tl.constexpr, stride_qgeo_h: tl.constexpr,
        stride_ksq_b: tl.constexpr, stride_ksq_t: tl.constexpr,
        stride_kss_b: tl.constexpr, stride_kss_t: tl.constexpr, stride_kss_c: tl.constexpr,
        stride_kgq_b: tl.constexpr, stride_kgq_t: tl.constexpr,
        stride_kgs_b: tl.constexpr, stride_kgs_t: tl.constexpr, stride_kgs_c: tl.constexpr,
        stride_vq_b: tl.constexpr, stride_vq_t: tl.constexpr,
        stride_vs_b: tl.constexpr, stride_vs_t: tl.constexpr, stride_vs_c: tl.constexpr,
        stride_o: tl.constexpr,
    ):
        """One streaming-update kernel: updates (m,d,o) for a block-range of tokens.

        This is intended for **decode (T==1)**, where we run online softmax.
        We fuse:
          - dequant (K_sem q4_0, K_geo q8_0, V q4_0)
          - logits computation
          - exp / sums
          - weighted value accumulation
        """
        pid = tl.program_id(0)  # 0 .. B*H-1
        b = pid // H
        h = pid - b * H

        # Load running state.
        m = tl.load(m_ptr + pid).to(tl.float32)
        d = tl.load(d_ptr + pid).to(tl.float32)
        dv = tl.arange(0, HD_V)
        o = tl.load(o_ptr + pid * stride_o + dv, mask=dv < HD_V, other=0.0).to(tl.float32)

        # Load query vectors.
        ds = tl.arange(0, HD_SEM)
        dg = tl.arange(0, HD_GEO)
        q_sem = tl.load(q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds, mask=ds < HD_SEM, other=0.0).to(tl.float32)
        q_geo = tl.load(q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg, mask=dg < HD_GEO, other=0.0).to(tl.float32)

        # Static loop: process NUM_SUBBLOCKS contiguous blocks of BLOCK_N tokens.
        for sb in tl.static_range(NUM_SUBBLOCKS):
            t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
            tm = t < L_prefix

            # ---- semantic keys: q4_0 ----
            ksd = h * HD_SEM + ds  # global dim indices in [0, sem_dim)
            ks_byte = ksd // 2
            ks_nib = ksd % 2
            ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
            ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.int32)
            ks_hi = ks_p >> 4
            ks_lo = ks_p & 0xF
            ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
            ks_q = ks_u.to(tl.int32) - 8
            ks_sb = ksd // QBLOCK_SEM
            ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
            ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
            k_sem = ks_q.to(tl.float32) * ks_s
            logit_sem = tl.sum(k_sem * q_sem[None, :], axis=1)

            # ---- geometric keys: q8_0 ----
            kgd = h * HD_GEO + dg  # global dim indices in [0, geo_dim)
            kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
            kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int32)
            kg_sb = kgd // QBLOCK_GEO
            kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
            kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
            k_geo = kg_q.to(tl.float32) * kg_s
            logit_geo = tl.sum(k_geo * q_geo[None, :], axis=1)

            logits = logit_sem * SEM_SCALE + logit_geo * GEO_SCALE
            logits = tl.where(tm, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m, block_max)
            exp_m = tl.exp(m - m_new)
            exp_logits = tl.exp(logits - m_new)

            # ---- values: q4_0 ----
            vd = h * HD_V + dv
            v_byte = vd // 2
            v_nib = vd % 2
            v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
            v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.int32)
            v_hi = v_p >> 4
            v_lo = v_p & 0xF
            v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
            v_q = v_u.to(tl.int32) - 8
            v_sb = vd // QBLOCK_V
            vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
            v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
            v_val = v_q.to(tl.float32) * v_s

            d = d * exp_m + tl.sum(exp_logits, axis=0)
            # weighted sum over tokens -> (HD_V,)
            wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
            o = o * exp_m + wv
            m = m_new

        tl.store(m_ptr + pid, m)
        tl.store(d_ptr + pid, d)
        tl.store(o_ptr + pid * stride_o + dv, o, mask=dv < HD_V)



    # -----------------------------
    # v24: 2-pass "FlashAttention-style" split-K decode kernels
    # -----------------------------
    #
    # Motivation:
    # - v23's fused kernel is "one program per (batch, head)" and loops over the sequence.
    #   Great for moderate context, but it can't parallelize over sequence length when T==1.
    # - This split-K design parallelizes across the KV sequence dimension by slicing it into partitions.
    #   Each partition computes local (m, d, o) using online-softmax, then we reduce partitions.
    #
    # This is decode-only (T==1), forward-only, and currently specialized for:
    #   K_sem: q4_0, K_geo: q8_0, V: q4_0  with qblock=32.

    @triton.jit
    def _kv_decode_partition_stats_decoupled_q4q8q4(
        q_sem_ptr, q_geo_ptr,
        k_sem_q_ptr, k_sem_s_ptr,
        k_geo_q_ptr, k_geo_s_ptr,
        v_q_ptr, v_s_ptr,
        m_part_ptr, d_part_ptr, o_part_ptr,
        L_prefix: tl.int32,
        P: tl.int32,
        PARTITION_SIZE: tl.constexpr,
        H: tl.constexpr,
        HD_SEM: tl.constexpr,
        HD_GEO: tl.constexpr,
        HD_V: tl.constexpr,
        QBLOCK_SEM: tl.constexpr,
        QBLOCK_GEO: tl.constexpr,
        QBLOCK_V: tl.constexpr,
        SEM_SCALE: tl.constexpr,
        GEO_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SUBBLOCKS: tl.constexpr,
        stride_qsem_b: tl.constexpr,
        stride_qsem_h: tl.constexpr,
        stride_qgeo_b: tl.constexpr,
        stride_qgeo_h: tl.constexpr,
        stride_ksq_b: tl.constexpr,
        stride_ksq_t: tl.constexpr,
        stride_kss_b: tl.constexpr,
        stride_kss_t: tl.constexpr,
        stride_kss_c: tl.constexpr,
        stride_kgq_b: tl.constexpr,
        stride_kgq_t: tl.constexpr,
        stride_kgs_b: tl.constexpr,
        stride_kgs_t: tl.constexpr,
        stride_kgs_c: tl.constexpr,
        stride_vq_b: tl.constexpr,
        stride_vq_t: tl.constexpr,
        stride_vs_b: tl.constexpr,
        stride_vs_t: tl.constexpr,
        stride_vs_c: tl.constexpr,
        stride_mp_row: tl.constexpr,
        stride_mp_part: tl.constexpr,
        stride_dp_row: tl.constexpr,
        stride_dp_part: tl.constexpr,
        stride_op_row: tl.constexpr,
        stride_op_part: tl.constexpr,
    ):
        pid_row = tl.program_id(0)  # 0 .. BH-1
        pid_part = tl.program_id(1) # 0 .. P-1 (grid)
        b = pid_row // H
        h = pid_row - b * H

        # Load query vectors (fp16) -> fp32.
        ds = tl.arange(0, HD_SEM)
        dg = tl.arange(0, HD_GEO)
        dv = tl.arange(0, HD_V)

        qsem = tl.load(q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds, mask=ds < HD_SEM, other=0.0).to(tl.float32)
        qgeo = tl.load(q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg, mask=dg < HD_GEO, other=0.0).to(tl.float32)

        # Local online-softmax state for this partition.
        m = -float("inf")
        d = 0.0
        o = tl.zeros([HD_V], dtype=tl.float32)

        start = pid_part * PARTITION_SIZE
        # Process PARTITION_SIZE tokens in NUM_SUBBLOCKS * BLOCK_N tiles.
        for sb in tl.static_range(0, NUM_SUBBLOCKS):
            t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
            tm = t < L_prefix

            # --- Semantic K (q4_0) dot ---
            # Global dim index within merged (H*HD_SEM)
            ksd = h * HD_SEM + ds                       # (HD_SEM,)
            ks_byte = ksd // 2
            ks_nib = ksd % 2
            ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
            ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.uint8)
            ks_hi = ks_p >> 4
            ks_lo = ks_p & 0xF
            ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
            ks_q = ks_u.to(tl.int32) - 8

            ks_sb = ksd // QBLOCK_SEM
            ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
            ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
            ksem = ks_q.to(tl.float32) * ks_s
            sem = tl.sum(ksem * qsem[None, :], axis=1)

            # --- Geometric K (q8_0) dot ---
            kgd = h * HD_GEO + dg
            kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
            kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int8).to(tl.float32)

            kg_sb = kgd // QBLOCK_GEO
            kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
            kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
            kgeo = kg_q * kg_s
            geo = tl.sum(kgeo * qgeo[None, :], axis=1)

            logits = sem * SEM_SCALE + geo * GEO_SCALE
            logits = tl.where(tm, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m, block_max)
            exp_m = tl.exp(m - m_new)

            exp_logits = tl.exp(logits - m_new)

            # --- V (q4_0) weighted sum ---
            vd = h * HD_V + dv
            v_byte = vd // 2
            v_nib = vd % 2
            v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
            v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.uint8)
            v_hi = v_p >> 4
            v_lo = v_p & 0xF
            v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
            v_q = v_u.to(tl.int32) - 8

            v_sb = vd // QBLOCK_V
            vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
            v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
            v_val = v_q.to(tl.float32) * v_s

            d = d * exp_m + tl.sum(exp_logits, axis=0)
            wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
            o = o * exp_m + wv
            m = m_new

        # Write partition stats
        tl.store(m_part_ptr + pid_row * stride_mp_row + pid_part * stride_mp_part, m)
        tl.store(d_part_ptr + pid_row * stride_dp_row + pid_part * stride_dp_part, d)
        tl.store(o_part_ptr + pid_row * stride_op_row + pid_part * stride_op_part + dv, o, mask=dv < HD_V)

    @triton.jit
    def _kv_decode_reduce_partitions(
        m_part_ptr, d_part_ptr, o_part_ptr,
        m_ptr, d_ptr, o_ptr,
        P: tl.int32,
        NUM_PARTS: tl.constexpr,
        HD_V: tl.constexpr,
        stride_mp_row: tl.constexpr,
        stride_mp_part: tl.constexpr,
        stride_dp_row: tl.constexpr,
        stride_dp_part: tl.constexpr,
        stride_op_row: tl.constexpr,
        stride_op_part: tl.constexpr,
        stride_o: tl.constexpr,
    ):
        pid_row = tl.program_id(0)  # 0..BH-1
        dv = tl.arange(0, HD_V)

        # First pass: global max over partitions.
        m = -float("inf")
        for p in tl.static_range(0, NUM_PARTS):
            p_i = tl.full([], p, tl.int32)
            pm = p_i < P
            mp = tl.load(m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part, mask=pm, other=-float("inf"))
            m = tl.maximum(m, mp)

        # Second pass: combine denominators and outputs.
        d = 0.0
        o = tl.zeros([HD_V], dtype=tl.float32)
        for p in tl.static_range(0, NUM_PARTS):
            p_i = tl.full([], p, tl.int32)
            pm = p_i < P
            mp = tl.load(m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part, mask=pm, other=-float("inf"))
            dp = tl.load(d_part_ptr + pid_row * stride_dp_row + p_i * stride_dp_part, mask=pm, other=0.0)
            op = tl.load(o_part_ptr + pid_row * stride_op_row + p_i * stride_op_part + dv, mask=pm & (dv < HD_V), other=0.0).to(tl.float32)

            scale = tl.where(pm, tl.exp(mp - m), 0.0)
            d += dp * scale
            o += op * scale

        tl.store(m_ptr + pid_row, m)
        tl.store(d_ptr + pid_row, d)
        tl.store(o_ptr + pid_row * stride_o + dv, o, mask=dv < HD_V)


def _triton_decoupled_q4q8q4_available() -> bool:
    return bool(TRITON_AVAILABLE)



class SeqCacheTensor:
    """
    A [B, max_seq_len, dim] sequence tensor stored in fp16/fp32/q8_0/q4_0/nf4.

    New in v22:
      - get_slice(start, end): dequantize only a slice (critical for long-context decode)
      - residual fp16 "hot" ring-buffer for the newest tokens (optional)
    """
    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        dim: int,
        cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.kind: KVCacheKind = cfg.kind
        self.device = device
        self.spec = make_quantspec(cfg.kind, dim, cfg.qblock)
        self.pos = 0
        self.max_seq_len = max_seq_len

        self.residual_len = int(max(0, cfg.residual_len))
        self._residual: Optional[torch.Tensor]
        self._residual_len_eff = min(self.residual_len, max_seq_len) if self.residual_len > 0 else 0

        if self.kind in ("fp16", "fp32"):
            dtype = torch.float16 if self.kind == "fp16" else torch.float32
            self.buf = torch.empty((batch_size, max_seq_len, dim), device=device, dtype=dtype)
            self.q = None
            self.s = None
            self._residual = None
        elif self.kind == "q8_0":
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim), device=device, dtype=torch.int8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
            self._residual = (
                torch.empty((batch_size, self._residual_len_eff, dim), device=device, dtype=torch.float16)
                if self._residual_len_eff > 0 else None
            )
        elif self.kind in ("q4_0", "nf4"):
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim // 2), device=device, dtype=torch.uint8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
            self._residual = (
                torch.empty((batch_size, self._residual_len_eff, dim), device=device, dtype=torch.float16)
                if self._residual_len_eff > 0 else None
            )
        else:
            raise ValueError(self.kind)

    @property
    def is_quantized(self) -> bool:
        return self.kind not in ("fp16", "fp32")

    def _residual_start(self) -> int:
        if self._residual is None:
            return self.pos  # empty range
        return max(0, self.pos - self._residual_len_eff)

    def _residual_gather(self, start: int, end: int) -> torch.Tensor:
        """
        Gather [start, end) from the fp16 residual ring buffer.
        Assumes the range is fully inside the residual window.
        """
        if self._residual is None:
            raise RuntimeError("No residual buffer allocated")
        if not (0 <= start <= end <= self.pos):
            raise ValueError(f"Invalid residual slice {start}:{end} for pos={self.pos}")
        rlen = self._residual_len_eff
        if rlen <= 0:
            raise RuntimeError("Residual length is 0")
        idx = (torch.arange(start, end, device=self.device, dtype=torch.long) % rlen)
        # (B, end-start, dim)
        return self._residual.index_select(1, idx)

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
        elif self.kind == "nf4":
            q, s = quantize_nf4(x_new, self.spec)
            self.q[:, old:old + Tn] = q
            self.s[:, old:old + Tn] = s
        else:
            raise ValueError(self.kind)

        # maintain residual fp16 ring for hot tokens (helps decode; negligible memory).
        if self._residual is not None:
            rlen = self._residual_len_eff
            if rlen > 0:
                if Tn >= rlen:
                    # Only the newest rlen tokens matter for the ring.
                    x_tail = x_new[:, -rlen:].to(torch.float16)
                    idx = (torch.arange(old + Tn - rlen, old + Tn, device=self.device, dtype=torch.long) % rlen)
                    self._residual[:, idx] = x_tail
                else:
                    x_fp16 = x_new.to(torch.float16)
                    idx = (torch.arange(old, old + Tn, device=self.device, dtype=torch.long) % rlen)
                    self._residual[:, idx] = x_fp16

        self.pos += Tn
        return old

    def get_slice(self, start: int, end: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        returns (B, end-start, dim) in `dtype`, dequantizing only the requested slice.
        """
        start = int(start); end = int(end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid slice {start}:{end}")
        if end > self.pos:
            raise ValueError(f"Requested end {end} > cached length {self.pos}")
        if start == end:
            # preserve shape semantics
            B = (self.buf.size(0) if self.buf is not None else self.q.size(0))  # type: ignore[union-attr]
            return torch.empty((B, 0, self.spec.dim), device=self.device, dtype=dtype)

        if self.kind in ("fp16", "fp32"):
            return self.buf[:, start:end].to(dtype)  # type: ignore[index]

        # residual fast-path (newest tokens)
        r_start = self._residual_start()
        if self._residual is not None and start >= r_start:
            return self._residual_gather(start, end).to(dtype)

        # mixed slice: older part dequant + tail from residual
        if self._residual is not None and end > r_start and start < r_start:
            a = self.get_slice(start, r_start, dtype=dtype)
            b = self._residual_gather(r_start, end).to(dtype)
            return torch.cat([a, b], dim=1)

        # fully in the quantized region
        if self.kind == "q8_0":
            x = dequantize_q8_0(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        if self.kind == "q4_0":
            x = dequantize_q4_0(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        if self.kind == "nf4":
            x = dequantize_nf4(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        raise ValueError(self.kind)

    def get(self, length: Optional[int] = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        returns (B, length, dim) in `dtype` (compatibility helper).
        """
        L = self.pos if length is None else int(length)
        return self.get_slice(0, L, dtype=dtype)


class LayerKVCache:
    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_dim: int,
        v_dim: int,
        k_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.k = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_dim, cfg=k_cfg, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg, device=device)

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
    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_sem_dim: int,
        k_geo_dim: int,
        v_dim: int,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.k_sem = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_sem_dim, cfg=k_sem_cfg, device=device)
        self.k_geo = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_geo_dim, cfg=k_geo_cfg, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg, device=device)

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

        def must_div(name: str, total: int, denom: int) -> int:
            if total % denom != 0:
                raise ValueError(f"{name} ({total}) must be divisible by {denom}")
            return total // denom

        if cfg.attn_mode == "standard":
            qk_dim = cfg.d_model
            v_dim = cfg.d_model
            self.qk_head_dim = must_div("d_model", qk_dim, H)
            self.v_head_dim = must_div("d_model", v_dim, H)

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
            self.qk_head_dim = must_div("attn_dim", qk_dim, H)
            self.v_head_dim = must_div("attn_dim", v_dim, H)
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
            kv_head = cfg.kv_head if cfg.kv_head is not None else H
            if kv_head <= 0:
                raise ValueError("kv_head must be > 0")
            if H % kv_head != 0:
                raise ValueError(f"gqa requires n_head % kv_head == 0 (got n_head={H}, kv_head={kv_head})")
            self.H_kv = kv_head
            self.group_size = H // kv_head

            self.qk_head_dim = must_div("attn_dim", cfg.attn_dim, H)
            self.v_head_dim = self.qk_head_dim
            kv_dim = kv_head * self.qk_head_dim

            if cfg.rope and (self.qk_head_dim % 2 != 0):
                raise ValueError("RoPE requires an even head dim. Choose attn_dim divisible by 2*n_head.")

            if cfg.tie_qk:
                raise ValueError("tie_qk is not supported for gqa unless kv_head == n_head (use --attn-mode standard).")

            self.q_proj = nn.Linear(cfg.d_model, cfg.attn_dim, bias=False)
            self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
            self.out_proj = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

            self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
            self.sem_head_dim = self.geo_head_dim = None
            self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

            self.k_null = nn.Parameter(torch.zeros(1, 1, kv_dim)) if cfg.null_attn else None
            self.v_null = nn.Parameter(torch.zeros(1, 1, kv_dim)) if cfg.null_attn else None

            self.k_sem_null = self.k_geo_null = None

        elif cfg.attn_mode == "decoupled":
            self.sem_head_dim = must_div("sem_dim", cfg.sem_dim, H)
            self.geo_head_dim = must_div("geo_dim", cfg.geo_dim, H)
            self.v_head_dim = must_div("attn_dim", cfg.attn_dim, H)
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

        # learned per-head temperature (applied as a multiplicative scale on logits)
        self.logit_scale = nn.Parameter(torch.zeros(H)) if cfg.learned_temp else None
        # Scratch buffers for Triton 2-pass (split-K) decode. Allocated lazily on CUDA.
        self._flash2_scratch = None  # type: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        self._flash2_scratch_cap = (0, 0, 0)  # (BH, parts_cap, hd_v)


    def _shape(self, x: torch.Tensor, head_dim: int, H: Optional[int] = None) -> torch.Tensor:
        # (B,T,H*hd)->(B,H,T,hd)
        B, T, D = x.shape
        H = self.H if H is None else H
        return x.view(B, T, H, head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,T,hd)->(B,T,H*hd)
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def _apply_logit_scale_to_q(self, q: torch.Tensor) -> torch.Tensor:
        # scores = (q @ k^T) * exp(logit_scale)
        if self.logit_scale is None:
            return q
        return q * torch.exp(self.logit_scale.view(1, -1, 1, 1))

    def _sdp(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Use PyTorch's fused SDPA when available (Flash / mem-efficient attention).
        # - If attn_mask is None, we can use is_causal=True and let the kernel handle causality.
        # - If attn_mask is provided (e.g., chunked prefill with KV-cache), we pass it explicitly.
        dropout_p = self.cfg.dropout if self.training else 0.0
        if attn_mask is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)

    def _streaming_decode_attn(
        self,
        *,
        q: torch.Tensor,          # (B,H,1,hd_qk)
        k_cache: SeqCacheTensor,  # stores (B,L,H*hd_qk) merged
        v_cache: SeqCacheTensor,  # stores (B,L,H*hd_v)  merged
        head_dim: int,
        decode_block: int,
        scale: float,
        v_head_dim: Optional[int] = None,
        k_null: Optional[torch.Tensor] = None,   # (B,H,1,hd_qk) or None
        v_null: Optional[torch.Tensor] = None,   # (B,H,1,hd_v)  or None
    ) -> torch.Tensor:
        """
        Streaming attention for decode (T==1): computes softmax(qK^T)V without materializing the full score vector.

        Returns: (B,H,1,hd_v)
        """
        B, H, Tq, hd = q.shape
        assert Tq == 1
        if v_head_dim is None:
            v_head_dim = head_dim

        L = k_cache.pos
        if L != v_cache.pos:
            raise RuntimeError("K/V cache desync in streaming decode")

        # We'll compute in fp32 for the running softmax state, but use fp16/bf16 matmuls where possible.
        compute_dtype = torch.float16 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

        m = torch.full((B, H, 1), -float("inf"), device=q.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, v_head_dim), device=q.device, dtype=torch.float32)

        qh = q.to(compute_dtype)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            # scores_f32: (B,H,1,Bl)
            block_max = scores_f32.amax(dim=-1)  # (B,H,1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)  # (B,H,1)

            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))  # (B,H,1,Bl)
            exp_scores_f16 = exp_scores.to(compute_dtype)

            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)  # (B,H,1)

            # (B,H,1,Bl) @ (B,H,Bl,hd) -> (B,H,1,hd)
            o = o * exp_m.unsqueeze(-1) + torch.matmul(exp_scores_f16, v_block_f16).to(torch.float32)
            m = m_new

        # Optional null token (one extra key/value)
        if k_null is not None and v_null is not None:
            # scores: (B,H,1,1)
            s = (qh * k_null.to(compute_dtype)).sum(dim=-1, keepdim=True).to(torch.float32) * scale
            update(s, v_null.to(compute_dtype))

        # Stream over cached sequence in blocks.
        blk = int(max(1, decode_block))
        for start in range(0, L, blk):
            end = min(L, start + blk)
            k_blk = k_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hd)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hdv)
            kbh = self._shape(k_blk, head_dim)                          # (B,H,Bl,hd)
            vbh = self._shape(v_blk, v_head_dim)                        # (B,H,Bl,hdv)

            # (B,H,1,hd) @ (B,H,hd,Bl) -> (B,H,1,Bl)
            scores = torch.matmul(qh, kbh.transpose(-2, -1)) * scale
            update(scores.to(torch.float32), vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q.dtype)

    def _streaming_decode_attn_decoupled(
        self,
        *,
        q_sem: torch.Tensor,          # (B,H,1,hd_sem)
        q_geo: torch.Tensor,          # (B,H,1,hd_geo)
        k_sem_cache: SeqCacheTensor,  # stores (B,L,H*hd_sem) merged
        k_geo_cache: SeqCacheTensor,  # stores (B,L,H*hd_geo) merged
        v_cache: SeqCacheTensor,      # stores (B,L,H*hd_v) merged
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
        k_sem_null: Optional[torch.Tensor] = None,  # (B,H,1,hd_sem)
        k_geo_null: Optional[torch.Tensor] = None,  # (B,H,1,hd_geo)
        v_null: Optional[torch.Tensor] = None,      # (B,H,1,hd_v)
    ) -> torch.Tensor:
        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = k_sem_cache.pos
        if not (L == k_geo_cache.pos == v_cache.pos):
            raise RuntimeError("Decoupled cache desync in streaming decode")

        compute_dtype = torch.float16 if q_sem.dtype in (torch.float16, torch.bfloat16) else q_sem.dtype

        m = torch.full((B, H, 1), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, v_head_dim), device=q_sem.device, dtype=torch.float32)

        qsh = q_sem.to(compute_dtype)
        qgh = q_geo.to(compute_dtype)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            block_max = scores_f32.amax(dim=-1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)

            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))
            exp_scores_f16 = exp_scores.to(compute_dtype)

            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)
            o = o * exp_m.unsqueeze(-1) + torch.matmul(exp_scores_f16, v_block_f16).to(torch.float32)
            m = m_new

        # Optional null token.
        if k_sem_null is not None and k_geo_null is not None and v_null is not None:
            s = (
                (qsh * k_sem_null.to(compute_dtype)).sum(dim=-1, keepdim=True).to(torch.float32) * sem_scale
                + (qgh * k_geo_null.to(compute_dtype)).sum(dim=-1, keepdim=True).to(torch.float32) * geo_scale
            )
            update(s, v_null.to(compute_dtype))

        blk = int(max(1, decode_block))
        for start in range(0, L, blk):
            end = min(L, start + blk)
            k_sem_blk = k_sem_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hd_sem)
            k_geo_blk = k_geo_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H*hd_geo)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)          # (B,Bl,H*hd_v)

            ksh = self._shape(k_sem_blk, sem_head_dim)  # (B,H,Bl,hd_sem)
            kgh = self._shape(k_geo_blk, geo_head_dim)  # (B,H,Bl,hd_geo)
            vbh = self._shape(v_blk, v_head_dim)        # (B,H,Bl,hd_v)

            s = (
                torch.matmul(qsh, ksh.transpose(-2, -1)).to(torch.float32) * sem_scale
                + torch.matmul(qgh, kgh.transpose(-2, -1)).to(torch.float32) * geo_scale
            )
            update(s, vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q_sem.dtype)

    def _fused_decode_attn_decoupled_q4q8q4(
        self,
        *,
        q_sem: torch.Tensor,          # (B,H,1,hd_sem)
        q_geo: torch.Tensor,          # (B,H,1,hd_geo)
        cache: "DecoupledLayerKVCache",
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
    ) -> torch.Tensor:
        """Decode-only fused path (T==1) for the common decoupled quant policy:

          K_sem: q4_0  (merged dim = sem_dim)
          K_geo: q8_0  (merged dim = geo_dim)
          V:     q4_0  (merged dim = attn_dim)

        Uses a Triton kernel (if installed) to fuse dequant + online-softmax update per block.
        Falls back to Python streaming if Triton is unavailable.
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if q_sem.device.type != "cuda":
            raise RuntimeError("Fused decode requires CUDA")
        if self.cfg.null_attn:
            raise RuntimeError("Fused decode currently assumes null_attn=False")

        # Enforce the expected cache layout.
        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = cache.pos

        # Split into a big quantized prefix and an fp16 residual tail (processed in PyTorch).
        rlen = cache.k_sem._residual_len_eff if cache.k_sem._residual is not None else 0
        r_start = max(0, L - rlen) if rlen > 0 else L
        L_prefix = int(r_start)

        # Triton expects contiguous (B,H,hd) query tensors.
        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        # Running state in fp32.
        BH = B * H
        m = torch.full((BH,), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((BH,), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

        if L_prefix > 0:
            # Choose a reasonable kernel tiling.
            block_n = 128
            num_sub = max(1, int(decode_block // block_n))
            step = block_n * num_sub

            # Shorthand tensors.
            ksq = cache.k_sem.q
            kss = cache.k_sem.s
            kgq = cache.k_geo.q
            kgs = cache.k_geo.s
            vq = cache.v.q
            vs = cache.v.s
            assert ksq is not None and kss is not None and kgq is not None and kgs is not None and vq is not None and vs is not None

            grid = (BH,)
            for start in range(0, L_prefix, step):
                _kv_decode_update_decoupled_q4q8q4[grid](
                    q_sem2,
                    q_geo2,
                    ksq,
                    kss,
                    kgq,
                    kgs,
                    vq,
                    vs,
                    m,
                    d,
                    o,
                    start,
                    L_prefix,
                    H=H,
                    HD_SEM=sem_head_dim,
                    HD_GEO=geo_head_dim,
                    HD_V=v_head_dim,
                    QBLOCK_SEM=32,
                    QBLOCK_GEO=32,
                    QBLOCK_V=32,
                    SEM_SCALE=sem_scale,
                    GEO_SCALE=geo_scale,
                    BLOCK_N=block_n,
                    NUM_SUBBLOCKS=num_sub,
                    stride_qsem_b=q_sem2.stride(0),
                    stride_qsem_h=q_sem2.stride(1),
                    stride_qgeo_b=q_geo2.stride(0),
                    stride_qgeo_h=q_geo2.stride(1),
                    stride_ksq_b=ksq.stride(0),
                    stride_ksq_t=ksq.stride(1),
                    stride_kss_b=kss.stride(0),
                    stride_kss_t=kss.stride(1),
                    stride_kss_c=kss.stride(2),
                    stride_kgq_b=kgq.stride(0),
                    stride_kgq_t=kgq.stride(1),
                    stride_kgs_b=kgs.stride(0),
                    stride_kgs_t=kgs.stride(1),
                    stride_kgs_c=kgs.stride(2),
                    stride_vq_b=vq.stride(0),
                    stride_vq_t=vq.stride(1),
                    stride_vs_b=vs.stride(0),
                    stride_vs_t=vs.stride(1),
                    stride_vs_c=vs.stride(2),
                    stride_o=o.stride(0),
                )

        # Process residual tail in PyTorch (small; uses fp16 residual ring via get_slice).
        if L_prefix < L:
            qsh = q_sem.to(torch.float16)
            qgh = q_geo.to(torch.float16)

            # reshape state to (B,H,1) / (B,H,1,hd)
            m_t = m.view(B, H, 1)
            d_t = d.view(B, H, 1)
            o_t = o.view(B, H, 1, v_head_dim)

            def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
                nonlocal m_t, d_t, o_t
                block_max = scores_f32.amax(dim=-1)
                m_new = torch.maximum(m_t, block_max)
                exp_m = torch.exp(m_t - m_new)
                exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1)).to(torch.float16)
                d_t = d_t * exp_m + exp_scores.sum(dim=-1).to(torch.float32)
                o_t = o_t * exp_m.unsqueeze(-1) + torch.matmul(exp_scores, v_block_f16).to(torch.float32)
                m_t = m_new

            # one or a few blocks only
            k_sem_blk = cache.k_sem.get_slice(L_prefix, L, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, L, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, L, dtype=torch.float16)
            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            s = (
                torch.matmul(qsh, ksh.transpose(-2, -1)).to(torch.float32) * sem_scale
                + torch.matmul(qgh, kgh.transpose(-2, -1)).to(torch.float32) * geo_scale
            )
            update(s, vbh)

            # flatten back
            m = m_t.view(BH)
            d = d_t.view(BH)
            o = o_t.view(BH, v_head_dim)

        out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
        return out.to(q_sem.dtype)


    def _fused_decode_attn_decoupled_q4q8q4_2pass(
        self,
        *,
        q_sem: torch.Tensor,          # (B,H,1,hd_sem)
        q_geo: torch.Tensor,          # (B,H,1,hd_geo)
        cache: "DecoupledLayerKVCache",
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
    ) -> torch.Tensor:
        """Decode-only fused path (T==1) using a 2-pass split-K ("FlashAttention-style") kernel.

        Pass 1 computes local (m, d, o) for each partition of the KV sequence in parallel.
        Pass 2 reduces partitions into a single (m, d, o) for the row.
        A tiny fp16 residual tail (hot window) is then folded in via the Python streaming updater.

        Specializes to the common decoupled heterogeneous quant policy:
          K_sem: q4_0  (merged dim = sem_dim)
          K_geo: q8_0  (merged dim = geo_dim)
          V:     q4_0  (merged dim = attn_dim)
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if q_sem.device.type != "cuda":
            raise RuntimeError("Fused decode requires CUDA")
        if self.cfg.null_attn:
            raise RuntimeError("Fused decode currently assumes null_attn=False")

        # Enforce the expected cache layout.
        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = cache.pos

        # Split into a big quantized prefix and an fp16 residual tail (processed in PyTorch).
        rlen = cache.k_sem._residual_len_eff if cache.k_sem._residual is not None else 0
        r_start = max(0, L - rlen) if rlen > 0 else L
        L_prefix = int(r_start)

        # Queries: contiguous (B,H,hd)
        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        BH = B * H

        # Running state in fp32 (flattened).
        m = torch.full((BH,), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((BH,), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

        if L_prefix > 0:
            # Partition sizing:
            # - decode_block is the user-facing knob. We round it up to a multiple of BLOCK_N.
            block_n = 128
            part_size = int(max(block_n, decode_block))
            if part_size % block_n != 0:
                part_size = ((part_size + block_n - 1) // block_n) * block_n
            num_sub = part_size // block_n

            P = int((L_prefix + part_size - 1) // part_size)

            # Allocate/reuse scratch buffers (BH, P_cap).
            # We grow capacity to the next power-of-two to reduce realloc+recompile churn.
            P_cap = 1 << (int(P - 1).bit_length())
            cap_BH, cap_P, cap_V = self._flash2_scratch_cap
            if (self._flash2_scratch is None) or (cap_BH < BH) or (cap_P < P_cap) or (cap_V != v_head_dim):
                m_part = torch.empty((BH, P_cap), device=q_sem.device, dtype=torch.float32)
                d_part = torch.empty((BH, P_cap), device=q_sem.device, dtype=torch.float32)
                o_part = torch.empty((BH, P_cap, v_head_dim), device=q_sem.device, dtype=torch.float32)
                self._flash2_scratch = (m_part, d_part, o_part)
                self._flash2_scratch_cap = (BH, P_cap, v_head_dim)
            else:
                m_part, d_part, o_part = self._flash2_scratch

            # Shorthand quant tensors.
            ksq = cache.k_sem.q
            kss = cache.k_sem.s
            kgq = cache.k_geo.q
            kgs = cache.k_geo.s
            vq = cache.v.q
            vs = cache.v.s
            assert ksq is not None and kss is not None and kgq is not None and kgs is not None and vq is not None and vs is not None

            # Pass 1: partition stats
            grid1 = (BH, P)
            _kv_decode_partition_stats_decoupled_q4q8q4[grid1](
                q_sem2,
                q_geo2,
                ksq,
                kss,
                kgq,
                kgs,
                vq,
                vs,
                m_part,
                d_part,
                o_part,
                L_prefix,
                P,
                PARTITION_SIZE=part_size,
                H=H,
                HD_SEM=sem_head_dim,
                HD_GEO=geo_head_dim,
                HD_V=v_head_dim,
                QBLOCK_SEM=32,
                QBLOCK_GEO=32,
                QBLOCK_V=32,
                SEM_SCALE=sem_scale,
                GEO_SCALE=geo_scale,
                BLOCK_N=block_n,
                NUM_SUBBLOCKS=num_sub,
                stride_qsem_b=q_sem2.stride(0),
                stride_qsem_h=q_sem2.stride(1),
                stride_qgeo_b=q_geo2.stride(0),
                stride_qgeo_h=q_geo2.stride(1),
                stride_ksq_b=ksq.stride(0),
                stride_ksq_t=ksq.stride(1),
                stride_kss_b=kss.stride(0),
                stride_kss_t=kss.stride(1),
                stride_kss_c=kss.stride(2),
                stride_kgq_b=kgq.stride(0),
                stride_kgq_t=kgq.stride(1),
                stride_kgs_b=kgs.stride(0),
                stride_kgs_t=kgs.stride(1),
                stride_kgs_c=kgs.stride(2),
                stride_vq_b=vq.stride(0),
                stride_vq_t=vq.stride(1),
                stride_vs_b=vs.stride(0),
                stride_vs_t=vs.stride(1),
                stride_vs_c=vs.stride(2),
                stride_mp_row=m_part.stride(0),
                stride_mp_part=m_part.stride(1),
                stride_dp_row=d_part.stride(0),
                stride_dp_part=d_part.stride(1),
                stride_op_row=o_part.stride(0),
                stride_op_part=o_part.stride(1),
                num_warps=4,
            )

            # Pass 2: reduce partitions -> (m, d, o) for the prefix.
            grid2 = (BH,)
            _kv_decode_reduce_partitions[grid2](
                m_part,
                d_part,
                o_part,
                m,
                d,
                o,
                P,
                NUM_PARTS=P_cap,
                HD_V=v_head_dim,
                stride_mp_row=m_part.stride(0),
                stride_mp_part=m_part.stride(1),
                stride_dp_row=d_part.stride(0),
                stride_dp_part=d_part.stride(1),
                stride_op_row=o_part.stride(0),
                stride_op_part=o_part.stride(1),
                stride_o=o.stride(0),
                num_warps=1,
            )

        # Process residual tail in PyTorch (small; uses fp16 residual ring via get_slice).
        if L_prefix < L:
            qsh = q_sem.to(torch.float16)
            qgh = q_geo.to(torch.float16)

            # reshape state to (B,H,1) / (B,H,1,hd)
            m_t = m.view(B, H, 1)
            d_t = d.view(B, H, 1)
            o_t = o.view(B, H, 1, v_head_dim)

            def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
                nonlocal m_t, d_t, o_t
                block_max = scores_f32.amax(dim=-1)
                m_new = torch.maximum(m_t, block_max)
                exp_m = torch.exp(m_t - m_new)
                exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1)).to(torch.float16)
                d_t = d_t * exp_m + exp_scores.sum(dim=-1).to(torch.float32)
                o_t = o_t * exp_m.unsqueeze(-1) + torch.matmul(exp_scores, v_block_f16).to(torch.float32)
                m_t = m_new

            k_sem_blk = cache.k_sem.get_slice(L_prefix, L, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, L, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, L, dtype=torch.float16)
            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            s = (
                torch.matmul(qsh, ksh.transpose(-2, -1)).to(torch.float32) * sem_scale
                + torch.matmul(qgh, kgh.transpose(-2, -1)).to(torch.float32) * geo_scale
            )
            update(s, vbh)

            m = m_t.view(BH)
            d = d_t.view(BH)
            o = o_t.view(BH, v_head_dim)

        out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
        return out.to(q_sem.dtype)


    def _streaming_decode_attn_gqa(
        self,
        *,
        q: torch.Tensor,          # (B,H,1,hd)
        k_cache: SeqCacheTensor,  # stores (B,L,H_kv*hd)
        v_cache: SeqCacheTensor,  # stores (B,L,H_kv*hd)
        head_dim: int,
        decode_block: int,
        scale: float,
        k_null: Optional[torch.Tensor] = None,  # (B,H_kv,1,hd) or None
        v_null: Optional[torch.Tensor] = None,  # (B,H_kv,1,hd) or None
    ) -> torch.Tensor:
        """
        Streaming decode for GQA without expanding KV heads to Q heads.
        Returns: (B,H,1,hd)
        """
        B, H, Tq, hd = q.shape
        assert Tq == 1
        H_kv = self.H_kv
        g = self.group_size
        if H != H_kv * g:
            raise RuntimeError("Invalid GQA head geometry")

        L = k_cache.pos
        if L != v_cache.pos:
            raise RuntimeError("K/V cache desync in streaming decode (gqa)")

        compute_dtype = torch.float16 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

        m = torch.full((B, H, 1), -float("inf"), device=q.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, hd), device=q.device, dtype=torch.float32)

        qh = q.to(compute_dtype)
        # reshape Q into KV groups: (B,H_kv,g,1,hd)
        qg = qh.view(B, H_kv, g, 1, hd)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            # scores_f32: (B,H,1,Bl)
            block_max = scores_f32.amax(dim=-1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)

            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))
            exp_scores_f16 = exp_scores.to(compute_dtype)

            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)

            # matmul in groups:
            # exp_scores: (B,H,1,Bl) -> (B,H_kv,g,1,Bl)
            es = exp_scores_f16.view(B, H_kv, g, 1, -1)
            # v_block_f16: (B,H_kv,Bl,hd) -> (B,H_kv,1,Bl,hd)
            vb = v_block_f16.unsqueeze(2)  # (B,H_kv,1,Bl,hd)
            # (B,H_kv,g,1,Bl) @ (B,H_kv,1,Bl,hd) -> (B,H_kv,g,1,hd)
            out_blk = torch.matmul(es, vb).to(torch.float32)
            o = o * exp_m.unsqueeze(-1) + out_blk.view(B, H, 1, hd)
            m = m_new

        # Optional null token (KV-head count).
        if k_null is not None and v_null is not None:
            # Expand to query heads logically by grouping.
            # scores: (B,H_kv,g,1,1) -> view (B,H,1,1)
            s = (qg * k_null.to(compute_dtype).unsqueeze(2)).sum(dim=-1, keepdim=True).to(torch.float32) * scale
            update(s.view(B, H, 1, 1), v_null.to(compute_dtype).unsqueeze(2).view(B, H, 1, hd))

        blk = int(max(1, decode_block))
        for start in range(0, L, blk):
            end = min(L, start + blk)
            k_blk = k_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H_kv*hd)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)  # (B,Bl,H_kv*hd)
            kbh = self._shape(k_blk, head_dim, H=H_kv)                   # (B,H_kv,Bl,hd)
            vbh = self._shape(v_blk, head_dim, H=H_kv)                   # (B,H_kv,Bl,hd)

            # scores per kv head/group: (B,H_kv,g,1,Bl)
            s = torch.matmul(qg, kbh.unsqueeze(2).transpose(-2, -1)) * scale
            update(s.view(B, H, 1, -1).to(torch.float32), vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q.dtype)

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
        attn_mask:
          - training: causal mask (B,1,T,T) bool, or None when using SDPA causal.
          - cached prefill: mask (B,1,T,L) bool to enforce causality inside the new chunk.
        cache:
          - None (training)
          - LayerKVCache (standard/bottleneck/gqa)
          - DecoupledLayerKVCache (decoupled)
        pos_offset: absolute position for RoPE for x[:,0]
        """
        cfg = self.cfg
        B, T, _ = x.shape
        ninfty = neg_inf(x.dtype)

        # -----------------------------
        # standard / bottleneck
        # -----------------------------
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

            qh = self._apply_logit_scale_to_q(qh)

            # ---- training / full-attn ----
            if cache is None:
                if not cfg.null_attn:
                    # Prefer SDPA/Flash when possible.
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    return y, None

                # Null-attn path (manual).
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                return y, None

            # ---- KV-cache path ----
            old_len = cache.pos

            # Fast prefill when the cache is empty: compute attention from local K/V, then append.
            if old_len == 0 and T > 1:
                if not cfg.null_attn:
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                # Null-attn prefill (manual).
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            # General cached path: append, then attend over the full cache.
            cache.append(self._merge(kh), self._merge(vh))
            L = cache.pos

            # Decode streaming for T==1 (critical for long context).
            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                if cfg.null_attn:
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                else:
                    k_null = v_null = None

                if cache.k.is_quantized or cache.v.is_quantized or cfg.null_attn:
                    out = self._streaming_decode_attn(
                        q=qh,
                        k_cache=cache.k,
                        v_cache=cache.v,
                        head_dim=self.qk_head_dim,
                        v_head_dim=self.v_head_dim,
                        decode_block=decode_block,
                        scale=(1.0 / math.sqrt(self.qk_head_dim)),
                        k_null=k_null,
                        v_null=v_null,
                    )
                else:
                    # fp16 cache on GPU -> SDPA is usually faster than Python streaming
                    k_all = self._shape(cache.k.get(dtype=qh.dtype), self.qk_head_dim)
                    v_all = self._shape(cache.v.get(dtype=qh.dtype), self.v_head_dim)
                    out = F.scaled_dot_product_attention(qh, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)

                y = self.out_proj(self._merge(out))
                return y, cache

            # Prefill/chunked attention (T>1): fallback (materializes K/V). This is still O(T*L).
            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, self.qk_head_dim)
            vh_all = self._shape(v_all, self.v_head_dim)

            scores = torch.matmul(qh, kh_all.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)  # (B,H,T,L)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh_all], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                out = torch.matmul(attn, vh_all)

            y = self.out_proj(self._merge(out))
            return y, cache

        # -----------------------------
        # gqa
        # -----------------------------
        if cfg.attn_mode == "gqa":
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            qh = self._shape(q, self.qk_head_dim, H=self.H)               # (B,H,T,hd)
            kh = self._shape(k, self.qk_head_dim, H=self.H_kv)            # (B,H_kv,T,hd)
            vh = self._shape(v, self.v_head_dim, H=self.H_kv)             # (B,H_kv,T,hd)

            if self.rotary is not None:
                qh = self.rotary.rotate(qh, pos_offset)
                kh = self.rotary.rotate(kh, pos_offset)

            qh = self._apply_logit_scale_to_q(qh)

            if cache is None:
                # For simplicity (and because this is a research file), we do a straightforward broadcast.
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)  # (B,H,T,hd)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)  # (B,H,T,hd)

                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    return y, None

                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                return y, None

            old_len = cache.pos

            if old_len == 0 and T > 1:
                # prefill without cache readback
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)
                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                # null-attn prefill (manual)
                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            cache.append(self._merge(kh), self._merge(vh))
            L = cache.pos

            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                if cfg.null_attn:
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                else:
                    k_null = v_null = None

                out = self._streaming_decode_attn_gqa(
                    q=qh,
                    k_cache=cache.k,
                    v_cache=cache.v,
                    head_dim=self.qk_head_dim,
                    decode_block=decode_block,
                    scale=(1.0 / math.sqrt(self.qk_head_dim)),
                    k_null=k_null,
                    v_null=v_null,
                )
                y = self.out_proj(self._merge(out))
                return y, cache

            # T>1 fallback
            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, self.qk_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)
            vh_all = self._shape(v_all, self.v_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)

            scores = torch.matmul(qh, kh_all.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv).repeat_interleave(self.group_size, dim=1)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh_all], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                out = torch.matmul(attn, vh_all)

            y = self.out_proj(self._merge(out))
            return y, cache

        # -----------------------------
        # decoupled
        # -----------------------------
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

        # Apply per-head temperature to both paths by scaling Q.
        qsh = self._apply_logit_scale_to_q(qsh)
        qgh = self._apply_logit_scale_to_q(qgh)

        sem_scale = 1.0 / math.sqrt(self.sem_head_dim)
        geo_scale = 1.0 / math.sqrt(self.geo_head_dim)

        if cache is None:
            if not cfg.null_attn:
                # Combine into a single SDPA call by concatenating along head_dim.
                q_cat = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)
                k_cat = torch.cat([ksh, kgh], dim=-1)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask)
                y = self.out_proj(self._merge(out))
                return y, None

            # Null-attn manual path
            sem = torch.matmul(qsh, ksh.transpose(-2, -1)) * sem_scale
            geo = torch.matmul(qgh, kgh.transpose(-2, -1)) * geo_scale
            scores = sem + geo

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)

            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)

            s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)

            if attn_mask is not None:
                extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                keep = torch.cat([extra, attn_mask], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh], dim=-2)
            out = torch.matmul(attn, vals)

            y = self.out_proj(self._merge(out))
            return y, None

        old_len = cache.pos

        # Empty-cache prefill: compute locally, then append (no dequant).
        if old_len == 0 and T > 1:
            if not cfg.null_attn:
                q_cat = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)
                k_cat = torch.cat([ksh, kgh], dim=-1)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask)
                y = self.out_proj(self._merge(out))
                cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
                return y, cache

            # Null-attn prefill manual
            sem = torch.matmul(qsh, ksh.transpose(-2, -1)) * sem_scale
            geo = torch.matmul(qgh, kgh.transpose(-2, -1)) * geo_scale
            scores = sem + geo

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)

            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)

            s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)

            if attn_mask is not None:
                extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                keep = torch.cat([extra, attn_mask], dim=-1)
                scores = scores.masked_fill(~keep, ninfty)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh], dim=-2)
            out = torch.matmul(attn, vals)

            y = self.out_proj(self._merge(out))
            cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
            return y, cache

        # General cached path.
        cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
        L = cache.pos

        # Streaming decode for T==1.
        if T == 1:
            decode_block = getattr(cache, "decode_block", 1024)
            if cfg.null_attn:
                ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
                kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
                vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
            else:
                ksn = kgn = vn = None

            if cache.k_sem.is_quantized or cache.k_geo.is_quantized or cache.v.is_quantized or cfg.null_attn:
                # Prefer fused kernels when available/allowed.
                use_fused = getattr(cache, "fused", "none")
                fused_ok = (
                    (not cfg.null_attn)
                    and use_fused in ("auto", "triton1pass", "triton2pass")
                    and _triton_decoupled_q4q8q4_available()
                    and cache.k_sem.kind == "q4_0"
                    and cache.k_geo.kind == "q8_0"
                    and cache.v.kind == "q4_0"
                )
                if fused_ok:
                    try:
                        if use_fused == "triton1pass":
                            out = self._fused_decode_attn_decoupled_q4q8q4(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                sem_head_dim=self.sem_head_dim,
                                geo_head_dim=self.geo_head_dim,
                                v_head_dim=self.v_head_dim,
                                decode_block=decode_block,
                                sem_scale=sem_scale,
                                geo_scale=geo_scale,
                            )
                        elif use_fused == "triton2pass":
                            out = self._fused_decode_attn_decoupled_q4q8q4_2pass(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                sem_head_dim=self.sem_head_dim,
                                geo_head_dim=self.geo_head_dim,
                                v_head_dim=self.v_head_dim,
                                decode_block=decode_block,
                                sem_scale=sem_scale,
                                geo_scale=geo_scale,
                            )
                        else:
                            # auto: 2-pass when the sequence is "long enough" that split-K parallelism helps.
                            if cache.pos >= 4 * int(decode_block):
                                out = self._fused_decode_attn_decoupled_q4q8q4_2pass(
                                    q_sem=qsh,
                                    q_geo=qgh,
                                    cache=cache,
                                    sem_head_dim=self.sem_head_dim,
                                    geo_head_dim=self.geo_head_dim,
                                    v_head_dim=self.v_head_dim,
                                    decode_block=decode_block,
                                    sem_scale=sem_scale,
                                    geo_scale=geo_scale,
                                )
                            else:
                                out = self._fused_decode_attn_decoupled_q4q8q4(
                                    q_sem=qsh,
                                    q_geo=qgh,
                                    cache=cache,
                                    sem_head_dim=self.sem_head_dim,
                                    geo_head_dim=self.geo_head_dim,
                                    v_head_dim=self.v_head_dim,
                                    decode_block=decode_block,
                                    sem_scale=sem_scale,
                                    geo_scale=geo_scale,
                                )
                    except Exception:
                        out = self._streaming_decode_attn_decoupled(
                            q_sem=qsh,
                            q_geo=qgh,
                            k_sem_cache=cache.k_sem,
                            k_geo_cache=cache.k_geo,
                            v_cache=cache.v,
                            sem_head_dim=self.sem_head_dim,
                            geo_head_dim=self.geo_head_dim,
                            v_head_dim=self.v_head_dim,
                            decode_block=decode_block,
                            sem_scale=sem_scale,
                            geo_scale=geo_scale,
                            k_sem_null=ksn,
                            k_geo_null=kgn,
                            v_null=vn,
                        )
                else:
                    out = self._streaming_decode_attn_decoupled(
                        q_sem=qsh,
                        q_geo=qgh,
                        k_sem_cache=cache.k_sem,
                        k_geo_cache=cache.k_geo,
                        v_cache=cache.v,
                        sem_head_dim=self.sem_head_dim,
                        geo_head_dim=self.geo_head_dim,
                        v_head_dim=self.v_head_dim,
                        decode_block=decode_block,
                        sem_scale=sem_scale,
                        geo_scale=geo_scale,
                        k_sem_null=ksn,
                        k_geo_null=kgn,
                        v_null=vn,
                    )
            else:
                # fp16 cache -> materialize and use SDPA
                k_sem_all = self._shape(cache.k_sem.get(dtype=qsh.dtype), self.sem_head_dim)
                k_geo_all = self._shape(cache.k_geo.get(dtype=qsh.dtype), self.geo_head_dim)
                v_all = self._shape(cache.v.get(dtype=qsh.dtype), self.v_head_dim)
                q_cat = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)
                k_cat = torch.cat([k_sem_all, k_geo_all], dim=-1)
                out = F.scaled_dot_product_attention(q_cat, k_cat, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)

            y = self.out_proj(self._merge(out))
            return y, cache

        # T>1 fallback (materialize).
        k_sem_all, k_geo_all, v_all = cache.get(dtype=x.dtype)
        ksh_all = self._shape(k_sem_all, self.sem_head_dim)
        kgh_all = self._shape(k_geo_all, self.geo_head_dim)
        vh_all = self._shape(v_all, self.v_head_dim)

        sem = torch.matmul(qsh, ksh_all.transpose(-2, -1)) * sem_scale
        geo = torch.matmul(qgh, kgh_all.transpose(-2, -1)) * geo_scale
        scores = sem + geo

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, ninfty)
        elif T > 1:
            key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
            q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
            keep = key_pos <= q_pos
            scores = scores.masked_fill(~keep, ninfty)

        if cfg.null_attn:
            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
            s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh_all], dim=-2)
            out = torch.matmul(attn, vals)
        else:
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
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

    def forward(
        self,
        idx: torch.Tensor,
        *,
        caches: Optional[List[Any]] = None,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        B, T = idx.shape
        if caches is None and T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}. Increase --block.")

        x = self.tok_emb(idx)
        if self.emb_in is not None:
            x = self.emb_in(x)
        x = self.drop(x)

        # Attention mask strategy:
        # - Training (no cache):
        #     * if null_attn is disabled, we pass attn_mask=None and let SDPA run with is_causal=True.
        #     * if null_attn is enabled (manual attention path), we provide a causal boolean mask.
        # - Cached prefill (T>1):
        #     * build a (1,1,T,L) boolean mask so the new chunk can attend to the prefix + causal within itself.
        #     * if cache is empty and null_attn is disabled, we again pass attn_mask=None so SDPA can use is_causal=True.
        # - Decode (T==1): no mask needed.
        attn_mask: Optional[torch.Tensor] = None
        if caches is None:
            if self.cfg.null_attn:
                attn_mask = self.causal_mask[:, :, :T, :T]
            else:
                attn_mask = None
        else:
            if T > 1:
                prev_len = caches[0].pos
                if prev_len == 0 and (not self.cfg.null_attn):
                    attn_mask = None
                else:
                    L = prev_len + T
                    key_pos = torch.arange(L, device=idx.device).view(1, 1, 1, L)
                    q_pos = (prev_len + torch.arange(T, device=idx.device)).view(1, 1, T, 1)
                    attn_mask = key_pos <= q_pos
            else:
                attn_mask = None

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
        # KV-cache controls (these matter *a lot* at long context)
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",  # {none, auto, triton1pass, triton2pass}
        # Optional heterogeneous overrides
        kv_cache_k: Optional[KVCacheKind] = None,
        kv_cache_v: Optional[KVCacheKind] = None,
        kv_cache_k_sem: Optional[KVCacheKind] = None,
        kv_cache_k_geo: Optional[KVCacheKind] = None,
        kv_qblock_k: Optional[int] = None,
        kv_qblock_v: Optional[int] = None,
        kv_qblock_k_sem: Optional[int] = None,
        kv_qblock_k_geo: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with optional heterogeneous KV-cache quantization.

        The cache API supports fp16/fp32/q8_0/q4_0/nf4, plus an fp16 residual window for the newest tokens.
        """
        self.eval()
        device = prompt.device
        B, T0 = prompt.shape
        max_seq = T0 + max_new_tokens

        if kv_fused not in ("none", "auto", "triton1pass", "triton2pass"):
            raise ValueError("kv_fused must be one of: none, auto, triton1pass, triton2pass")

        # Heterogeneous default for decoupled mode:
        # - Per the draft paper: keep the RoPE/geometric key path higher precision, compress the semantic path harder.
        #   (You can override this with --kv-cache-k-sem / --kv-cache-k-geo.)
        if self.cfg.attn_mode == "decoupled" and kv_cache == "q4_0":
            if kv_cache_k_geo is None:
                kv_cache_k_geo = "q8_0"
            if kv_cache_k_sem is None:
                kv_cache_k_sem = "q4_0"
            if kv_cache_v is None:
                kv_cache_v = "q4_0"

        def make_cfg(kind_override: Optional[KVCacheKind], qblock_override: Optional[int]) -> KVCacheTensorConfig:
            kind = kind_override if kind_override is not None else kv_cache
            qblock = qblock_override if qblock_override is not None else kv_qblock
            residual_len = kv_residual if kind not in ("fp16", "fp32") else 0
            return KVCacheTensorConfig(kind=kind, qblock=qblock, residual_len=residual_len)

        # default K/V configs (standard/bottleneck/gqa)
        k_cfg = make_cfg(kv_cache_k, kv_qblock_k)
        v_cfg = make_cfg(kv_cache_v, kv_qblock_v)

        # decoupled configs
        k_sem_cfg = make_cfg(kv_cache_k_sem, kv_qblock_k_sem)
        k_geo_cfg = make_cfg(kv_cache_k_geo, kv_qblock_k_geo)

        caches: List[Any] = []
        for _ in range(self.cfg.n_layer):
            if self.cfg.attn_mode == "decoupled":
                c = DecoupledLayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_sem_dim=self.cfg.sem_dim,
                    k_geo_dim=self.cfg.geo_dim,
                    v_dim=self.cfg.attn_dim,
                    k_sem_cfg=k_sem_cfg,
                    k_geo_cfg=k_geo_cfg,
                    v_cfg=make_cfg(kv_cache_v, kv_qblock_v),
                    device=device,
                )
                c.decode_block = kv_decode_block
                c.fused = kv_fused
                caches.append(c)
            else:
                if self.cfg.attn_mode == "standard":
                    k_dim = v_dim = self.cfg.d_model
                elif self.cfg.attn_mode == "bottleneck":
                    k_dim = v_dim = self.cfg.attn_dim
                elif self.cfg.attn_mode == "gqa":
                    head_dim = self.cfg.attn_dim // self.cfg.n_head
                    kv_head = self.cfg.kv_head if self.cfg.kv_head is not None else self.cfg.n_head
                    k_dim = v_dim = kv_head * head_dim
                else:
                    raise ValueError(f"Unknown attn_mode for KV cache: {self.cfg.attn_mode}")

                c = LayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_dim=k_dim,
                    v_dim=v_dim,
                    k_cfg=k_cfg,
                    v_cfg=v_cfg,
                    device=device,
                )
                c.decode_block = kv_decode_block
                c.fused = kv_fused
                caches.append(c)

        # Prefill (fills caches). Thanks to the attention module, the "empty-cache prefill" path avoids cache readback.
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

            # decode one token
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
    ap.add_argument("--compile", action="store_true", help="Use torch.compile(...) for speed (experimental).")
    ap.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode (if --compile).")
    ap.add_argument("--amp", action="store_true", help="Enable CUDA AMP (mixed precision) for training (experimental).")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                    help="AMP dtype for CUDA (bf16 recommended on Ampere+).")


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
    ap.add_argument("--kv-cache", type=str, default="fp16", choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Default KV-cache format (can be overridden per-tensor with the --kv-cache-* flags).")
    ap.add_argument("--kv-qblock", type=int, default=32, help="Quantization block size along the channel dimension.")
    ap.add_argument("--kv-residual", type=int, default=128,
                    help="Keep this many newest KV tokens in fp16 as a hot residual window (only for quantized caches).")
    ap.add_argument("--kv-decode-block", type=int, default=1024,
                    help="Sequence-block size for streaming decode attention (smaller = less memory, more Python overhead).")
    ap.add_argument("--kv-fused", type=str, default="auto", choices=["none", "auto", "triton1pass", "triton2pass"],
                    help="Use fused decode kernels when available. 'auto' picks a sensible kernel when Triton+CUDA are available.")
    ap.add_argument("--kv-cache-k", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override K cache kind (standard/bottleneck/gqa).")
    ap.add_argument("--kv-cache-v", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override V cache kind (standard/bottleneck/gqa and decoupled).")
    ap.add_argument("--kv-cache-k-sem", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override semantic K cache kind (decoupled only).")
    ap.add_argument("--kv-cache-k-geo", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
                    help="Override geometric K cache kind (decoupled only).")
    ap.add_argument("--kv-qblock-k", type=int, default=None, help="Override K qblock (standard/bottleneck/gqa).")
    ap.add_argument("--kv-qblock-v", type=int, default=None, help="Override V qblock.")
    ap.add_argument("--kv-qblock-k-sem", type=int, default=None, help="Override semantic K qblock (decoupled).")
    ap.add_argument("--kv-qblock-k-geo", type=int, default=None, help="Override geometric K qblock (decoupled).")
    ap.add_argument("--tokenizer", type=str, default="word", choices=["word", "tiktoken"])

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
        # Sampling mode requires --ckpt (we need vocab/model shape from checkpoint config).
        if not args.ckpt:
            ap.error("--ckpt is required for --mode sample")

    if args.ckpt:
         print(f"Loading checkpoint from {args.ckpt}...")
         ckpt = torch.load(args.ckpt, map_location=device)
         if args.mode == "sample" and "config" not in ckpt:
              ap.error("Checkpoint is missing 'config'; cannot infer model architecture for sampling.")
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
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode}).")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")


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
    kv_residual=args.kv_residual,
    kv_decode_block=args.kv_decode_block,
    kv_fused=args.kv_fused,
    kv_cache_k=args.kv_cache_k,
    kv_cache_v=args.kv_cache_v,
    kv_cache_k_sem=args.kv_cache_k_sem,
    kv_cache_k_geo=args.kv_cache_k_geo,
    kv_qblock_k=args.kv_qblock_k,
    kv_qblock_v=args.kv_qblock_v,
    kv_qblock_k_sem=args.kv_qblock_k_sem,
    kv_qblock_k_geo=args.kv_qblock_k_geo,
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
    use_amp = bool(args.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)


    best_val = float("inf")
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
            print(f"== eval step {step} | train {tr:.4f} | val {va:.4f} | val_ppl {ppl:.2f} | {(time.time()-t_eval0):.1f}s")
            if va < best_val:
                best_val = va
                save_ckpt(args.out_dir, "best.pt", model, cfg, step, best_val)
                print(f"   (new best) {best_val:.4f}")
            t_eval0 = time.time()

        if step == args.steps:
            break

        x, y = get_batch(train_tokens, args.batch_size, cfg.block_size, device)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
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
            tok_count = 0
            dt_acc = 0.0

    save_ckpt(args.out_dir, "last.pt", model, cfg, args.steps, best_val)
    print(f"Done. best_val={best_val:.4f}")


if __name__ == "__main__":
    main()