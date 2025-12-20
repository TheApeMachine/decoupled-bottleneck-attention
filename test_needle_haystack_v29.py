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
    import tiktoken  # type: ignore

    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

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


def _pick_single_token_passkeys(enc, n: int, seed: int) -> List[Tuple[str, int]]:
    """
    Pick N passkeys that are a *single* GPT-2 BPE token when they appear after "is"
    (i.e., they typically decode as " <word>" with a leading space).

    We avoid maintaining brittle hand-written word lists by scanning the vocabulary
    for "space-prefixed word-like" tokens.
    """
    n = int(n)
    if n <= 0:
        return []

    cands: List[Tuple[str, int]] = []
    vocab_n = int(getattr(enc, "n_vocab", 50257))
    for tid in range(vocab_n):
        s = enc.decode([tid])
        if not s or not s.startswith(" "):
            continue
        w = s[1:]
        # Keep only simple "word-like" tokens. (Avoid punctuation, multi-space, unicode artifacts.)
        if not w:
            continue
        if any(ch.isspace() for ch in w):
            continue
        if not w.isascii():
            continue
        w0 = w.replace("'", "").replace("-", "").replace("_", "")
        if not w0:
            continue
        if not w0.isalnum():
            continue
        if len(w) > 16:
            continue
        cands.append((w, int(tid)))

    if len(cands) < max(4, min(n, 8)):
        raise RuntimeError(f"Could not find enough word-like single-token passkeys in vocab (found {len(cands)}).")

    rng = random.Random(int(seed))
    rng.shuffle(cands)
    # If we still can't reach n, we will reuse (cycle) passkeys rather than hard-failing.
    if len(cands) >= n:
        return cands[:n]

    print(f"[warn] Requested {n} unique passkeys but only found {len(cands)}; will reuse passkeys.")
    out: List[Tuple[str, int]] = []
    for i in range(n):
        out.append(cands[i % len(cands)])
    return out


def _prompt_tokens(enc, prompt_style: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Returns (needle_prefix, query_tokens, needle_suffix).

    The evaluation is a strict next-token check: after feeding `x = (haystack-with-needle) + query_tokens`,
    we measure whether the next-token argmax equals the passkey token id.

    So `query_tokens` must end with the same prefix as `needle_prefix` such that the *next token* should be
    the passkey token.
    """
    s = str(prompt_style).strip().lower()
    if s in {"qa", "q", "default"}:
        needle_prefix = enc.encode("The passkey is")
        query_tokens = enc.encode("\nQ: The passkey is")
        needle_suffix = enc.encode(".\n")
        return needle_prefix, query_tokens, needle_suffix
    if s in {"key", "kv", "k:v"}:
        # More literal key/value phrasing; tends to be easier for base LMs than a question.
        needle_prefix = enc.encode("Passkey:")
        query_tokens = enc.encode("\nPasskey:")
        needle_suffix = enc.encode("\n")
        return needle_prefix, query_tokens, needle_suffix
    if s in {"repeat", "copy"}:
        # Imperative copy-style prompt.
        needle_prefix = enc.encode("Repeat:")
        query_tokens = enc.encode("\nRepeat:")
        needle_suffix = enc.encode("\n")
        return needle_prefix, query_tokens, needle_suffix
    raise ValueError(f"unknown --prompt-style: {prompt_style!r}")


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


def _make_context_no_needle(*, ctx_len: int, query_tokens: List[int], filler_tokens: List[int]) -> torch.Tensor:
    """Same total length as _make_context, but without inserting the needle."""
    if len(filler_tokens) < ctx_len:
        reps = (ctx_len // len(filler_tokens)) + 1
        filler_tokens = (filler_tokens * reps)[:ctx_len]
    else:
        filler_tokens = filler_tokens[:ctx_len]
    full = filler_tokens + query_tokens
    return torch.tensor(full, dtype=torch.long).unsqueeze(0)  # (1,T)


@torch.no_grad()
def _eval_one(model: GPT, x: torch.Tensor, expected_next: int, device: torch.device) -> Tuple[bool, bool, float, float, int]:
    """
    Returns:
      success (argmax==expected),
      top5 (expected in top-5),
      p(expected),
      nll_expected (nats),
      rank_expected (1=best)
    """
    model.eval()
    logits, _ = model(x.to(device))
    last = logits[:, -1, :]  # (1,V)
    probs = torch.softmax(last, dim=-1)
    p = float(probs[0, expected_next].item())
    pred = int(probs.argmax(dim=-1).item())
    top5 = torch.topk(probs, k=5, dim=-1).indices[0].detach().cpu().tolist()
    nll = float(F.cross_entropy(last, torch.tensor([expected_next], device=last.device), reduction="mean").item())
    exp_logit = float(last[0, expected_next].item())
    rank = int((last[0] > exp_logit).sum().detach().cpu().item()) + 1
    return pred == expected_next, (int(expected_next) in [int(x) for x in top5]), p, nll, rank


def main() -> None:
    ap = argparse.ArgumentParser(description="Needle-in-a-haystack (passkey) probe for v29 checkpoints")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (best.pt)")
    ap.add_argument("--device", type=str, default=None, help="cpu|mps|cuda (default: auto)")
    ap.add_argument(
        "--prompt-style",
        type=str,
        default="qa",
        choices=["qa", "key", "repeat"],
        help="Prompt template. 'qa' matches the original script; 'key' and 'repeat' are often easier for base LMs.",
    )
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths (tokens) for the haystack (excluding query). Default: [train_ctx/4, train_ctx/2, train_ctx].",
    )
    ap.add_argument("--trials", type=int, default=20, help="Trials per (context,depth)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="assets/needle_haystack_v29.png")
    ap.add_argument(
        "--also-plot",
        type=str,
        nargs="*",
        default=["p_expected", "top5", "delta_nll", "log10_p_ratio"],
        help="Also save extra heatmaps for these metrics (default: p_expected, top5). Choices: accuracy, top5, p_expected, nll_expected, rank_expected",
    )
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

    train_ctx = int(cfg.block_size)
    if args.context_lengths is None or len(args.context_lengths) == 0:
        cands = [max(16, train_ctx // 4), max(32, train_ctx // 2), train_ctx]
        context_lengths = sorted(set(int(c) for c in cands))
    else:
        context_lengths = sorted(set(int(x) for x in args.context_lengths))

    print(f"trained_ctx={train_ctx}")
    if any(c > train_ctx for c in context_lengths):
        print(f"[warn] requested context_lengths={context_lengths} exceed trained_ctx={train_ctx}.")
        print("[warn] This is extrapolation; strict next-token retrieval accuracy is often ~0% beyond train_ctx for base LMs.")

    enc = tiktoken.get_encoding("gpt2")

    print(f"prompt_style={args.prompt_style}")

    # Needle + query pattern. We make the *next token* after the query equal the passkey token by construction.
    needle_prefix, query_tokens, needle_suffix = _prompt_tokens(enc, args.prompt_style)
    filler_tokens = enc.encode(
        "This is filler text used to construct long contexts. "
        "It does not contain the passkey. "
        "We repeat this to reach the desired token length.\n"
    )

    passkeys = _pick_single_token_passkeys(enc, n=max(4, args.trials), seed=int(args.seed))

    results: Dict[int, Dict[float, Dict[str, float]]] = {}
    for ctx_len in context_lengths:
        results[int(ctx_len)] = {}
        for depth in args.depths:
            ok = 0
            ok5 = 0
            p_sum = 0.0
            nll_sum = 0.0
            rank_sum = 0.0
            # Diagnostics that remain meaningful even when strict accuracy is 0%.
            # delta_nll < 0 means the needle increased probability of the expected token.
            delta_nll_sum = 0.0
            log10_p_ratio_sum = 0.0
            for t in range(int(args.trials)):
                w, tok = passkeys[t % len(passkeys)]
                needle_tokens = needle_prefix + [tok] + needle_suffix
                x = _make_context(
                    enc,
                    ctx_len=int(ctx_len),
                    depth=float(depth),
                    needle_tokens=needle_tokens,
                    query_tokens=query_tokens,
                    filler_tokens=filler_tokens,
                )
                x0 = _make_context_no_needle(ctx_len=int(ctx_len), query_tokens=query_tokens, filler_tokens=filler_tokens)
                _ensure_block_size(model, int(max(x.shape[1], x0.shape[1])) + 1, device)

                success, top5, p, nll, rank = _eval_one(model, x, expected_next=tok, device=device)
                _, _, p0, nll0, _ = _eval_one(model, x0, expected_next=tok, device=device)
                ok += int(success)
                ok5 += int(top5)
                p_sum += p
                nll_sum += nll
                rank_sum += float(rank)
                delta_nll_sum += float(nll - nll0)
                log10_p_ratio_sum += float(np.log10((p + 1e-12) / (p0 + 1e-12)))
            acc = ok / float(args.trials)
            acc5 = ok5 / float(args.trials)
            results[int(ctx_len)][float(depth)] = {
                "accuracy": acc,
                "top5": acc5,
                "p_expected": p_sum / float(args.trials),
                "nll_expected": nll_sum / float(args.trials),
                "rank_expected": rank_sum / float(args.trials),
                "delta_nll": delta_nll_sum / float(args.trials),
                "log10_p_ratio": log10_p_ratio_sum / float(args.trials),
            }
            print(
                f"ctx={ctx_len:5d} depth={depth:>5.0%}  acc={acc:>6.1%}  top5={acc5:>6.1%}  "
                f"p={p_sum/args.trials:.6f}  nll={nll_sum/args.trials:.3f}  rank={rank_sum/args.trials:.1f}  "
                f"Δnll={delta_nll_sum/args.trials:+.3f}  log10(p/p0)={log10_p_ratio_sum/args.trials:+.2f}"
            )

    # Plot heatmap(s)
    out = Path(args.out)
    style = str(args.prompt_style).strip().lower()
    style_suffix = f"_{style}"
    # Auto-suffix outputs so multiple prompt styles don't overwrite each other.
    # IMPORTANT: keep the default ('qa') unsuffixed so paper.tex and wrapper scripts
    # can reference stable filenames like assets/m4max_baseline_needle_haystack.png.
    if style != "qa" and out.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg"} and not out.stem.endswith(style_suffix):
        out = out.with_name(f"{out.stem}{style_suffix}{out.suffix}")
        print(f"[note] auto-suffixed --out => {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Also write CSV for reproducibility
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("context_len,depth,accuracy,top5,p_expected,nll_expected,rank_expected,delta_nll,log10_p_ratio\n")
        for ctx_len in context_lengths:
            for depth in args.depths:
                r = results[int(ctx_len)][float(depth)]
                f.write(
                    f"{int(ctx_len)},{float(depth):.3f},{r['accuracy']:.4f},{r['top5']:.4f},"
                    f"{r['p_expected']:.6f},{r['nll_expected']:.6f},{r['rank_expected']:.3f},"
                    f"{r['delta_nll']:.6f},{r['log10_p_ratio']:.6f}\n"
                )
    print(f"✓ wrote {csv_path}")

    if HAS_MPL:
        depths = list(map(float, args.depths))
        ctxs = list(map(int, context_lengths))

        def _plot_metric(metric: str, path: Path) -> None:
            metric = str(metric)
            if metric not in {"accuracy", "top5", "p_expected", "nll_expected", "rank_expected", "delta_nll", "log10_p_ratio"}:
                raise ValueError(f"unknown metric: {metric}")

            raw = np.array([[float(results[c][d][metric]) for d in depths] for c in ctxs], dtype=np.float32)

            # Choose presentation
            if metric in {"accuracy", "top5"}:
                mat = raw * 100.0
                cmap = "RdYlGn"
                vmin, vmax = 0.0, 100.0
                title = f"Needle-in-a-Haystack: {metric} (%)"
                fmt = lambda v: f"{v:.0f}%"
                txt_color = lambda v: "black" if v > 50 else "white"
            elif metric == "p_expected":
                # log10 for readability; this remains meaningful even when strict accuracy is 0%.
                mat = np.log10(np.maximum(raw, 1e-12))
                cmap = "viridis"
                vmin, vmax = float(mat.min()), float(mat.max())
                title = "Needle-in-a-Haystack: log10 p(expected)"
                fmt = lambda v: f"{v:.1f}"
                txt_color = lambda _v: "white"
            elif metric == "nll_expected":
                mat = raw
                cmap = "magma"
                vmin, vmax = float(mat.min()), float(mat.max())
                title = "Needle-in-a-Haystack: NLL(expected) (nats)"
                fmt = lambda v: f"{v:.2f}"
                txt_color = lambda _v: "white"
            elif metric == "delta_nll":
                mat = raw
                cmap = "RdBu_r"
                # Center at 0 for interpretability.
                m = float(np.max(np.abs(mat)))
                vmin, vmax = -m, m
                title = "Needle effect: ΔNLL(expected) = NLL(with needle) - NLL(no needle)"
                fmt = lambda v: f"{v:+.2f}"
                txt_color = lambda _v: "black"
            elif metric == "log10_p_ratio":
                mat = raw
                cmap = "RdBu_r"
                m = float(np.max(np.abs(mat)))
                vmin, vmax = -m, m
                title = "Needle effect: log10(p_with / p_without)"
                fmt = lambda v: f"{v:+.2f}"
                txt_color = lambda _v: "black"
            else:  # rank_expected
                mat = raw
                cmap = "magma_r"
                vmin, vmax = float(mat.min()), float(mat.max())
                title = "Needle-in-a-Haystack: rank(expected) (1=best)"
                fmt = lambda v: f"{v:.0f}"
                txt_color = lambda _v: "white"

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(depths)))
            ax.set_xticklabels([f"{d:.0%}" for d in depths])
            ax.set_yticks(range(len(ctxs)))
            ax.set_yticklabels([str(c) for c in ctxs])
            ax.set_xlabel("Needle depth (fraction of context)")
            ax.set_ylabel("Context length (tokens)")
            ax.set_title(title, fontweight="bold")
            for i in range(len(ctxs)):
                for j in range(len(depths)):
                    v = float(mat[i, j])
                    ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9, color=txt_color(v))
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric)
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✓ wrote {path}")

        # Main plot = accuracy (for paper-friendly convention)
        _plot_metric("accuracy", out)

        # Optional extra plots (helpful when accuracy is all zeros)
        also = list(args.also_plot) if args.also_plot is not None else []
        for metric in also:
            if str(metric) == "accuracy":
                continue
            extra = out.with_name(f"{out.stem}_{metric}{out.suffix}")
            _plot_metric(str(metric), extra)
    else:
        print("[warn] matplotlib not installed; skipping plot image.")


if __name__ == "__main__":
    main()


