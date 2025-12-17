#!/usr/bin/env python3
"""
plot_rank_evidence.py

Generate a paper-ready figure that supports the "low intrinsic dimensionality" claim:
  - Singular value spectra of Q/K projection *activations* (not weights)
  - Entropy-based effective rank (eRank = exp(H(p))) per layer

By default this produces: assets/m4max_rank_evidence.png
which is referenced from paper.tex (Appendix).

Example:
  python3.12 plot_rank_evidence.py \
    --ckpt baseline=runs/m4max_baseline_seed1337/best.pt \
    --ckpt decoupled=runs/m4max_decoupled_48_96_seed1337/best.pt \
    --data-npy fineweb_100m.npy \
    --offset 0 --seq-len 1024 \
    --device mps \
    --out assets/m4max_rank_evidence.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


try:
    import matplotlib.pyplot as plt  # type: ignore

    HAS_MPL = True
except Exception:
    HAS_MPL = False


from v29_transformer_decoupled_bottleneck_instrumented import GPT, ModelConfig  # noqa: E402


@dataclass
class Captured:
    # proj_name -> list[layer] -> tensor (B,T,D)
    acts: Dict[str, List[Optional[torch.Tensor]]]


def _parse_kv(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"Expected label=path, got: {s}")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k or not v:
        raise ValueError(f"Bad label=path: {s}")
    return k, v


def _ensure_block_size(model: GPT, block_size: int, device: torch.device) -> None:
    if model.cfg.block_size >= block_size:
        return
    model.cfg.block_size = int(block_size)
    # Update causal mask buffer to avoid forward() rejecting long sequences when null_attn=True
    model.register_buffer(
        "causal_mask",
        torch.tril(torch.ones(model.cfg.block_size, model.cfg.block_size, dtype=torch.bool, device=device)).view(
            1, 1, model.cfg.block_size, model.cfg.block_size
        ),
        persistent=False,
    )


def _cov_singular_values(x: torch.Tensor) -> torch.Tensor:
    """
    Compute singular values of x via eigenvalues of covariance.
    x: (N, D)
    returns sv: (D,) sorted descending (sqrt(eigvals))
    """
    x = x.float()
    # center features (helps interpretability of spectra)
    x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    # (D,D)
    c = (x.T @ x) / max(1, n - 1)
    # eigvalsh returns ascending
    ev = torch.linalg.eigvalsh(c)
    ev = torch.clamp(ev, min=0.0)
    sv = torch.sqrt(ev).flip(0)
    return sv


def _entropy_effective_rank(sv: torch.Tensor) -> float:
    sv = sv.float()
    s = sv.sum()
    if not torch.isfinite(s) or s.item() <= 0:
        return float("nan")
    p = sv / s
    p = torch.clamp(p, min=1e-12)
    h = -(p * torch.log(p)).sum()
    return float(torch.exp(h).item())


def _collect_projections(model: GPT, x: torch.Tensor, device: torch.device) -> Captured:
    n_layer = int(model.cfg.n_layer)
    acts: Dict[str, List[Optional[torch.Tensor]]] = {
        "q": [None] * n_layer,
        "k": [None] * n_layer,
        "q_sem": [None] * n_layer,
        "k_sem": [None] * n_layer,
        "q_geo": [None] * n_layer,
        "k_geo": [None] * n_layer,
    }

    hooks = []

    def _save(name: str, layer: int):
        def _hook(_mod, _inp, out):
            # out: (B,T,D)
            if isinstance(out, (tuple, list)):
                out_t = out[0]
            else:
                out_t = out
            if not torch.is_tensor(out_t):
                return
            acts[name][layer] = out_t.detach().to(torch.float32).cpu()

        return _hook

    for li, blk in enumerate(model.blocks):
        attn = blk.attn

        # standard / bottleneck / gqa
        if getattr(attn, "q_proj", None) is not None:
            hooks.append(attn.q_proj.register_forward_hook(_save("q", li)))
        if getattr(attn, "k_proj", None) is not None:
            hooks.append(attn.k_proj.register_forward_hook(_save("k", li)))

        # decoupled
        if getattr(attn, "q_sem", None) is not None:
            hooks.append(attn.q_sem.register_forward_hook(_save("q_sem", li)))
        if getattr(attn, "k_sem", None) is not None:
            hooks.append(attn.k_sem.register_forward_hook(_save("k_sem", li)))
        if getattr(attn, "q_geo", None) is not None:
            hooks.append(attn.q_geo.register_forward_hook(_save("q_geo", li)))
        if getattr(attn, "k_geo", None) is not None:
            hooks.append(attn.k_geo.register_forward_hook(_save("k_geo", li)))

    with torch.no_grad():
        _ = model(x.to(device))

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    return Captured(acts=acts)


def _summarize_rank(c: Captured, max_sv: int, spectrum_layer: int) -> Dict[str, Dict[str, object]]:
    """
    Returns dict proj_name -> {'sv': Tensor[max_sv], 'erank': List[float]}
    """
    out: Dict[str, Dict[str, object]] = {}
    for proj_name, per_layer in c.acts.items():
        if all(v is None for v in per_layer):
            continue
        eranks: List[float] = []
        for v in per_layer:
            if v is None:
                eranks.append(float("nan"))
                continue
            x = v.reshape(-1, v.shape[-1])  # (B*T, D)
            sv = _cov_singular_values(x)
            eranks.append(_entropy_effective_rank(sv))

        # spectrum: use chosen layer if present, else first non-null
        li = spectrum_layer
        if li < 0:
            li = len(per_layer) + li
        li = max(0, min(li, len(per_layer) - 1))
        v_spec = per_layer[li]
        if v_spec is None:
            v_spec = next((t for t in per_layer if t is not None), None)
        if v_spec is None:
            continue
        x_spec = v_spec.reshape(-1, v_spec.shape[-1])
        sv_spec = _cov_singular_values(x_spec)
        sv_spec = sv_spec[:max_sv]
        sv_spec = sv_spec / (sv_spec.sum() + 1e-12)

        out[proj_name] = {
            "sv": sv_spec,
            "erank": eranks,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="label=path (repeatable)")
    ap.add_argument("--data-npy", type=str, default="fineweb_100m.npy", help="Calibration tokens (.npy)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|mps|cuda")
    ap.add_argument("--max-sv", type=int, default=128, help="How many singular values to plot")
    ap.add_argument("--spectrum-layer", type=int, default=-1, help="Which layer to use for spectrum (default: last)")
    ap.add_argument("--out", type=str, default="assets/m4max_rank_evidence.png")
    args = ap.parse_args()

    if not HAS_MPL:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    device = torch.device(args.device)

    # Load token slice
    p = Path(args.data_npy)
    if p.suffix != ".npy":
        raise SystemExit(f"--data-npy must be a .npy file (got {p})")
    arr = np.load(p, mmap_mode="r")
    off = int(args.offset)
    n = int(args.seq_len)
    if off < 0 or off + n >= int(arr.shape[0]):
        raise SystemExit(f"Slice out of range: offset={off} seq_len={n} len={int(arr.shape[0])}")
    toks = np.asarray(arr[off : off + n], dtype=np.int64)
    x = torch.tensor(toks, dtype=torch.long).unsqueeze(0)  # (1,T)

    # Collect + analyze for each checkpoint
    labels: List[str] = []
    summaries: Dict[str, Dict[str, Dict[str, object]]] = {}
    dims: Dict[str, Dict[str, int]] = {}
    n_layers_by_label: Dict[str, int] = {}

    for spec in args.ckpt:
        label, ckpt_path = _parse_kv(spec)
        labels.append(label)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ModelConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _ensure_block_size(model, int(args.seq_len), device)

        cap = _collect_projections(model, x, device)
        summaries[label] = _summarize_rank(cap, max_sv=int(args.max_sv), spectrum_layer=int(args.spectrum_layer))

        # record dims for legend friendliness
        d: Dict[str, int] = {}
        for k, v in cap.acts.items():
            t0 = next((t for t in v if t is not None), None)
            if t0 is not None:
                d[k] = int(t0.shape[-1])
        dims[label] = d
        n_layers_by_label[label] = int(cfg.n_layer)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_q_spec, ax_q_er = axes[0, 0], axes[0, 1]
    ax_k_spec, ax_k_er = axes[1, 0], axes[1, 1]

    def plot_spec(ax, proj_keys: List[str], title: str) -> None:
        for label in labels:
            for pk in proj_keys:
                if pk not in summaries[label]:
                    continue
                sv = summaries[label][pk]["sv"]
                assert torch.is_tensor(sv)
                d = dims[label].get(pk, sv.numel())
                ax.plot(
                    range(1, int(sv.numel()) + 1),
                    sv.numpy(),
                    linewidth=1.8,
                    label=f"{label}:{pk} (d={d})",
                )
        ax.set_title(title)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalized singular value")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")

    def plot_erank(ax, proj_keys: List[str], title: str) -> None:
        for label in labels:
            nL = n_layers_by_label[label]
            xs = list(range(nL))
            for pk in proj_keys:
                if pk not in summaries[label]:
                    continue
                er = summaries[label][pk]["erank"]
                d = dims[label].get(pk, None)
                ax.plot(xs, er, marker="o", markersize=3.5, linewidth=1.8, label=f"{label}:{pk}" + (f" (d={d})" if d else ""))
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy effective rank  (exp(H))")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")

    # Q: show standard q, and decoupled q_sem/q_geo if present
    plot_spec(ax_q_spec, ["q", "q_sem", "q_geo"], "Q projection activations: singular value spectrum (example layer)")
    plot_erank(ax_q_er, ["q", "q_sem", "q_geo"], "Q projection activations: entropy effective rank per layer")
    plot_spec(ax_k_spec, ["k", "k_sem", "k_geo"], "K projection activations: singular value spectrum (example layer)")
    plot_erank(ax_k_er, ["k", "k_sem", "k_geo"], "K projection activations: entropy effective rank per layer")

    fig.suptitle(
        f"Projection activation rank evidence (offset={int(args.offset)}, seq_len={int(args.seq_len)}; spectra use layer {int(args.spectrum_layer)})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ wrote {out}")


if __name__ == "__main__":
    main()


