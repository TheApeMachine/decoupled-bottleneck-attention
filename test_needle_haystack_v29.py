#!/usr/bin/env python3
"""
test_needle_haystack_v29.py

Long-context "needle-in-a-haystack" style sanity test for v29 checkpoints, using GPT-2 tokenization
so we can construct a natural-language key + query.

This is a lightweight retrieval probe:
  - We insert a single-token passkey in a long filler context at different depths.
  - We then query with the exact pattern "The passkey is" and check whether the next-token argmax
    matches the inserted passkey token.

It is not a full downstream evaluation, but it is a useful long-context regression/sanity check,
especially for RoPE-related changes.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import tiktoken  # type: ignore

    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

from v29_transformer_decoupled_bottleneck_instrumented import GPT, ModelConfig, pick_device


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


def _pick_single_token_passkeys(enc, n: int, seed: int) -> List[Tuple[str, int]]:
    # Candidate words; we filter to those that tokenize to exactly one token when preceded by a space.
    # (This is critical so the evaluation is a strict next-token check.)
    raw = [
        "apple", "banana", "orange", "grape", "pear", "peach", "mango", "lemon",
        "table", "window", "garden", "river", "mountain", "forest", "ocean", "planet",
        "silver", "gold", "purple", "yellow", "crimson", "coffee", "pencil", "paper",
        "camera", "engine", "signal", "kernel", "vector", "matrix", "cipher", "dragon",
        "castle", "rocket", "satellite", "library", "music", "winter", "summer", "autumn",
    ]
    rng = random.Random(seed)
    rng.shuffle(raw)
    out: List[Tuple[str, int]] = []
    for w in raw:
        tok = enc.encode(" " + w)
        if len(tok) == 1:
            out.append((w, tok[0]))
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError("Could not find enough single-token passkeys; extend candidate list.")
    return out


def _make_context(enc, *, ctx_len: int, depth: float, needle_tokens: List[int], query_tokens: List[int], filler_tokens: List[int]) -> torch.Tensor:
    """
    Build token ids with total length ~= ctx_len + len(query_tokens).
    The needle is inserted into the first ctx_len tokens at the specified depth.
    """
    needle_len = len(needle_tokens)
    if ctx_len < needle_len + 16:
        raise ValueError("ctx_len too small for needle")
    insert_pos = int((ctx_len - needle_len) * float(depth))
    insert_pos = max(8, min(insert_pos, ctx_len - needle_len - 8))

    # Build filler to desired length (ctx_len)
    if len(filler_tokens) < ctx_len:
        reps = (ctx_len // len(filler_tokens)) + 1
        filler_tokens = (filler_tokens * reps)[:ctx_len]
    else:
        filler_tokens = filler_tokens[:ctx_len]

    ctx = filler_tokens[:insert_pos] + needle_tokens + filler_tokens[insert_pos + needle_len :]
    full = ctx + query_tokens
    return torch.tensor(full, dtype=torch.long).unsqueeze(0)  # (1,T)


@torch.no_grad()
def _eval_one(model: GPT, x: torch.Tensor, expected_next: int, device: torch.device) -> Tuple[bool, float, float]:
    """
    Returns:
      success (argmax==expected),
      p(expected),
      delta_nll (nats) relative to self (i.e., NLL of expected token)
    """
    model.eval()
    logits, _ = model(x.to(device))
    last = logits[:, -1, :]  # (1,V)
    probs = torch.softmax(last, dim=-1)
    p = float(probs[0, expected_next].item())
    pred = int(probs.argmax(dim=-1).item())
    nll = float(F.cross_entropy(last, torch.tensor([expected_next], device=last.device), reduction="mean").item())
    return pred == expected_next, p, nll


def main() -> None:
    ap = argparse.ArgumentParser(description="Needle-in-a-haystack (passkey) probe for v29 checkpoints")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (best.pt)")
    ap.add_argument("--device", type=str, default=None, help="cpu|mps|cuda (default: auto)")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--trials", type=int, default=20, help="Trials per (context,depth)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="assets/needle_haystack_v29.png")
    args = ap.parse_args()

    if not HAS_TIKTOKEN:
        raise SystemExit("tiktoken is required: pip install tiktoken")

    device = pick_device(args.device)
    print(f"device={device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    # Needle + query pattern. We make the *next token* after the query equal the passkey token by construction.
    # Needle includes: "The passkey is <passkey>."
    # Query ends with:  "The passkey is"
    needle_prefix = enc.encode("The passkey is")
    query_tokens = enc.encode("\nQ: The passkey is")
    filler_tokens = enc.encode(
        "This is filler text used to construct long contexts. "
        "It does not contain the passkey. "
        "We repeat this to reach the desired token length.\n"
    )

    passkeys = _pick_single_token_passkeys(enc, n=max(4, args.trials), seed=int(args.seed))

    results: Dict[int, Dict[float, Dict[str, float]]] = {}
    for ctx_len in args.context_lengths:
        results[int(ctx_len)] = {}
        for depth in args.depths:
            ok = 0
            p_sum = 0.0
            nll_sum = 0.0
            for t in range(int(args.trials)):
                w, tok = passkeys[t % len(passkeys)]
                needle_tokens = needle_prefix + [tok] + enc.encode(".\n")
                x = _make_context(
                    enc,
                    ctx_len=int(ctx_len),
                    depth=float(depth),
                    needle_tokens=needle_tokens,
                    query_tokens=query_tokens,
                    filler_tokens=filler_tokens,
                )
                _ensure_block_size(model, int(x.shape[1]) + 1, device)
                success, p, nll = _eval_one(model, x, expected_next=tok, device=device)
                ok += int(success)
                p_sum += p
                nll_sum += nll
            acc = ok / float(args.trials)
            results[int(ctx_len)][float(depth)] = {
                "accuracy": acc,
                "p_expected": p_sum / float(args.trials),
                "nll_expected": nll_sum / float(args.trials),
            }
            print(f"ctx={ctx_len:5d} depth={depth:>5.0%}  acc={acc:>6.1%}  p={p_sum/args.trials:.3f}  nll={nll_sum/args.trials:.3f}")

    # Plot heatmap
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Also write CSV for reproducibility
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("context_len,depth,accuracy,p_expected,nll_expected\n")
        for ctx_len in args.context_lengths:
            for depth in args.depths:
                r = results[int(ctx_len)][float(depth)]
                f.write(f"{int(ctx_len)},{float(depth):.3f},{r['accuracy']:.4f},{r['p_expected']:.6f},{r['nll_expected']:.6f}\n")
    print(f"✓ wrote {csv_path}")

    if HAS_MPL:
        depths = list(map(float, args.depths))
        ctxs = list(map(int, args.context_lengths))
        mat = np.array([[results[c][d]["accuracy"] * 100.0 for d in depths] for c in ctxs], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d:.0%}" for d in depths])
        ax.set_yticks(range(len(ctxs)))
        ax.set_yticklabels([str(c) for c in ctxs])
        ax.set_xlabel("Needle depth (fraction of context)")
        ax.set_ylabel("Context length (tokens)")
        ax.set_title("Needle-in-a-Haystack (Passkey) Accuracy (%)", fontweight="bold")
        for i in range(len(ctxs)):
            for j in range(len(depths)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="black" if v > 50 else "white")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Accuracy (%)")
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ wrote {out}")
    else:
        print("[warn] matplotlib not installed; skipping plot image.")


if __name__ == "__main__":
    main()


