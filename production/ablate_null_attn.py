#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import tiktoken  # type: ignore

# Allow running as either:
#   - python -m production.ablate_null_attn  (preferred)
#   - python production/ablate_null_attn.py  (convenient)
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repo_root))

# Local imports must come after sys.path modification for direct script execution
from production.config import pick_device  # pylint: disable=wrong-import-position
from production.model import GPT, ModelConfig  # pylint: disable=wrong-import-position
from production.kvcache_backend import KVCacheTensorConfig  # pylint: disable=wrong-import-position
from production.runtime_tuning import load_token_ids_spec  # pylint: disable=wrong-import-position


def _load_tokens_auto(spec: str, *, want_len: int) -> list[int]:
    """Load tokens from either integer token IDs or raw text.

    - If `spec` points to a file that parses as whitespace-separated ints (or .npy), use it.
    - Otherwise, treat it as raw text and tokenize with tiktoken (GPT-2 encoding).
    """
    want_len = int(max(2, want_len))
    try:
        ids = load_token_ids_spec(str(spec))
        if len(ids) >= want_len:
            return [int(x) for x in ids]
    except (OSError, ValueError, TypeError):
        pass

    p = Path(str(spec))
    if not p.exists():
        raise ValueError(
            f"--tokens must be a path to token IDs (.txt/.npy) or raw text file; "
            f"not found: {spec}"
        )

    raw = p.read_text(encoding="utf-8", errors="ignore")
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(raw)

    if len(ids) < want_len:
        raise ValueError(f"Not enough tokenized IDs from text: need {want_len}, got {len(ids)}")

    return [int(x) for x in ids]


@torch.no_grad()
def eval_nll_chunked(
    model: GPT,
    tokens_1d: torch.Tensor,
    *,
    seq_len: int,
    chunk_size: int,
    device: torch.device,
) -> float:
    """Teacher-forced next-token NLL over `seq_len` tokens using cache-chunked forward passes."""
    if tokens_1d.dim() != 1:
        raise ValueError("tokens_1d must be 1D")
    total_len = int(seq_len)
    if total_len < 2:
        return float("nan")

    ids = (
        tokens_1d[: total_len + 1]
        .to(device=device, dtype=torch.long)
        .unsqueeze(0)
    )  # (1, total_len+1)
    # predict next token for each position 0..total_len-1 (target is 1..total_len)
    inp = ids[:, :total_len]
    tgt = ids[:, 1 : total_len + 1]

    # fp16 caches everywhere for evaluation (we're isolating the null_attn effect).
    fp16 = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
    caches: list[Any] = []
    for _ in range(int(model.cfg.n_layer)):
        caches.append(
            model.make_decoupled_layer_cache(
                batch_size=1,
                max_seq_len=total_len,
                k_sem_cfg=fp16,
                k_geo_cfg=fp16,
                v_cfg=fp16,
                device=device,
                decode_block=1024,
                fused="none",
            )
        )

    nll_sum = 0.0
    nll_count = 0
    for i in range(0, total_len, int(chunk_size)):
        end = min(total_len, i + int(chunk_size))
        x = inp[:, i:end]
        y = tgt[:, i:end]
        logits, caches = model(x, caches=caches, pos_offset=int(i))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        nll_sum += float(loss.item())
        nll_count += int(y.numel())

    return float(nll_sum / max(1, nll_count))


def _load_model(*, ckpt_path: str, device: torch.device, null_attn: bool, seq_len: int) -> GPT:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg_dict = ckpt.get("config", None)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")
    cfg = ModelConfig(**cfg_dict)
    if str(getattr(cfg, "attn_mode", "")) != "decoupled":
        raise ValueError("This ablation script targets decoupled attention checkpoints only.")

    cfg.null_attn = bool(null_attn)
    cfg.block_size = int(max(int(cfg.block_size), int(seq_len)))

    model = GPT(cfg).to(device)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[ablate] non-strict load (null_attn={null_attn}). "
            f"Missing={incompatible.missing_keys} "
            f"Unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def main(argv: Optional[list[str]] = None) -> int:
    """
    Ablate null_attn for decoupled checkpoints via teacher-forced NLL on a small token window.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Ablate null_attn for decoupled checkpoints via teacher-forced NLL "
            "on a small token window."
        )
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Token spec: path to .txt/.npy or whitespace-separated ints."
    )
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args(argv)

    dev = pick_device(args.device) if args.device is None else torch.device(str(args.device))
    ids = _load_tokens_auto(str(args.tokens), want_len=int(args.seq_len) + 1)
    tok = torch.tensor(ids, dtype=torch.long)

    m_off = _load_model(
        ckpt_path=str(args.ckpt), device=dev, null_attn=False, seq_len=int(args.seq_len)
    )
    m_on = _load_model(
        ckpt_path=str(args.ckpt), device=dev, null_attn=True, seq_len=int(args.seq_len)
    )

    nll_off = eval_nll_chunked(
        m_off,
        tok,
        seq_len=int(args.seq_len),
        chunk_size=int(args.chunk_size),
        device=dev
    )

    nll_on = eval_nll_chunked(
        m_on, tok,
        seq_len=int(args.seq_len),
        chunk_size=int(args.chunk_size),
        device=dev
    )

    dnll = float(nll_on - nll_off)
    ppl_ratio = float(math.exp(dnll)) if math.isfinite(dnll) else float("nan")

    print(json.dumps({
        "cfg_off": asdict(m_off.cfg),
        "cfg_on": asdict(m_on.cfg)
    }, indent=2, sort_keys=True))
    print(f"[ablate] seq_len={int(args.seq_len)} chunk={int(args.chunk_size)} device={dev}")
    print(f"[ablate] null_attn=False NLL={nll_off:.6f}")
    print(f"[ablate] null_attn=True  NLL={nll_on:.6f}")
    print(f"[ablate] ΔNLL(on-off)={dnll:.6f} (ppl_ratio={ppl_ratio:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
