#!/usr/bin/env python3
"""
test_rope_extrapolation_v29.py

RoPE extrapolation test for v29 checkpoints, using a .npy token stream.

This evaluates teacher-forced loss/perplexity at increasing context lengths by sampling
random contiguous windows from the token stream.

Example:
  python3.12 test_rope_extrapolation_v29.py \
    --ckpt runs/m4max_decoupled_48_96_seed1337/best.pt \
    --data-npy fineweb_100m.npy \
    --contexts 1024 2048 4096 8192 \
    --num-batches 50 \
    --batch-size 1 \
    --out assets/m4max_seed1337_rope_extrapolation.png
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple

import unittest
try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"numpy is required for this module but is not available: {e}")
try:
    import torch
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for this module but is not available: {e}")
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from v29_transformer_decoupled_bottleneck_instrumented import GPT, ModelConfig, pick_device
except Exception as e:  # pragma: no cover
    if __name__ == "__main__":
        raise
    raise unittest.SkipTest(f"v29_transformer_decoupled_bottleneck_instrumented import failed: {e}")


def _ensure_block_size(model: GPT, block_size: int, device: torch.device) -> None:
    if model.cfg.block_size >= block_size:
        return
    model.cfg.block_size = int(block_size)
    model.register_buffer(
        "causal_mask",
        torch.tril(torch.ones(model.cfg.block_size, model.cfg.block_size, dtype=torch.bool, device=device)).view(
            1, 1, model.cfg.block_size, model.cfg.block_size
        ),
        persistent=False,
    )


@torch.no_grad()
def eval_ppl(
    model: GPT,
    toks: np.ndarray,
    *,
    context_len: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    seed: int,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = int(toks.shape[0])
    total_loss = 0.0
    total_tokens = 0

    # We sample random windows. Each window is length context_len+1 (for next-token labels).
    for _ in range(int(num_batches)):
        # sample batch starts
        starts: List[int] = []
        for _b in range(int(batch_size)):
            s = rng.randint(0, max(0, n - (context_len + 2)))
            starts.append(s)
        batch = np.stack([toks[s : s + context_len + 1] for s in starts], axis=0).astype(np.int64)  # (B,T+1)
        batch_t = torch.tensor(batch, dtype=torch.long, device=device)
        x = batch_t[:, :-1]
        y = batch_t[:, 1:]
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(y.numel())

    avg = total_loss / max(1, total_tokens)
    ppl = math.exp(avg)
    return avg, ppl


def main() -> None:
    ap = argparse.ArgumentParser(description="RoPE extrapolation test (v29)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--data-npy", type=str, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--len", type=int, default=2_000_000, help="How many tokens to use from the stream")
    ap.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths to test. Default: [train_ctx, 2x, 4x] (capped to <=8192).",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="assets/rope_extrapolation_v29.png")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device={device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    train_ctx = int(cfg.block_size)
    if args.contexts is None or len(args.contexts) == 0:
        # Default: train_ctx, 2x, 4x (cap to keep runtime reasonable for dense attention).
        cands = [train_ctx, train_ctx * 2, train_ctx * 4]
        contexts = []
        for c in cands:
            if c <= 8192:
                contexts.append(int(c))
        contexts = sorted(set(contexts))
    else:
        contexts = sorted(set(int(x) for x in args.contexts))

    print(f"trained_ctx={train_ctx}")

    p = Path(args.data_npy)
    arr = np.load(p, mmap_mode="r")
    off = int(args.offset)
    n = int(args.len)
    if off < 0 or off + n > int(arr.shape[0]):
        raise SystemExit(f"Slice out of range: offset={off} len={n} total={int(arr.shape[0])}")
    toks = np.asarray(arr[off : off + n])

    results = []
    for ctx in contexts:
        _ensure_block_size(model, ctx, device)
        ratio = float(ctx) / float(train_ctx) if train_ctx > 0 else float("nan")
        tag = "within-train" if ratio <= 1.01 else f"{ratio:.1f}x"
        print(f"ctx={ctx} ({tag}): evaluating...", end=" ", flush=True)
        loss, ppl = eval_ppl(
            model,
            toks,
            context_len=ctx,
            batch_size=int(args.batch_size),
            num_batches=int(args.num_batches),
            device=device,
            seed=int(args.seed) + ctx,
        )
        print(f"loss={loss:.4f} ppl={ppl:.2f}")
        results.append((ctx, loss, ppl))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("context,loss,ppl\n")
        for ctx, loss, ppl in results:
            f.write(f"{ctx},{loss:.6f},{ppl:.6f}\n")
    print(f"✓ wrote {csv_path}")

    if HAS_MPL:
        ctxs = [r[0] for r in results]
        losses = [r[1] for r in results]
        ppls = [r[2] for r in results]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(ctxs, losses, "bo-", linewidth=2, markersize=7)
        ax1.set_xlabel("Context length")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss vs context")
        ax1.set_xscale("log", base=2)
        ax1.grid(True, alpha=0.25)

        ax2.plot(ctxs, ppls, "ro-", linewidth=2, markersize=7)
        ax2.set_xlabel("Context length")
        ax2.set_ylabel("Perplexity (log)")
        ax2.set_title("Perplexity vs context")
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.25)

        fig.suptitle("RoPE extrapolation (teacher-forced)", fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ wrote {out}")
    else:
        print("[warn] matplotlib not installed; skipping plot image.")


if __name__ == "__main__":
    main()


