from __future__ import annotations

import argparse
import inspect
import time
from typing import cast

import torch

from production.model import GPT, ModelConfig
from production.model.metrics import Metrics
from production.selfopt_cache import as_str_object_dict


def _torch_load_obj(path: str, *, device: torch.device) -> object:
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(str(path), map_location=device, weights_only=True)
    return torch.load(str(path), map_location=device)


def _as_state_dict(o: object) -> dict[str, torch.Tensor] | None:
    if not isinstance(o, dict):
        return None
    out: dict[str, torch.Tensor] = {}
    for k, v in o.items():
        if not isinstance(k, str):
            return None
        if not isinstance(v, torch.Tensor):
            return None
        out[k] = v
    return out


def _set_dropout_p(module: torch.nn.Module, *, p: float) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = float(p)


def _run_forward(
    model: GPT,
    idx: torch.Tensor,
    *,
    long_seq_enabled: bool,
    mem_block: int,
    local_window: int,
    q_chunk: int,
    summarizer: str,
) -> torch.Tensor:
    cfg = cast(ModelConfig, model.cfg)
    cfg.null_attn = False
    cfg.train_long_seq_enabled = bool(long_seq_enabled)
    cfg.train_long_seq_threshold = 0
    cfg.train_long_seq_mem_block = int(mem_block)
    cfg.train_long_seq_local_window = int(local_window)
    cfg.train_long_seq_q_chunk = int(q_chunk)
    cfg.train_long_seq_mem_summarizer = str(summarizer).strip().lower()
    logits, _ = model(idx)
    return logits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate long-seq approximation drift against exact SDPA (metrics-style).")
    _ = ap.add_argument("--ckpt", type=str, required=True)
    _ = ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    _ = ap.add_argument("--batch", type=int, default=1)
    _ = ap.add_argument("--ref-len", type=int, default=2048)
    _ = ap.add_argument("--approx-only-len", type=int, default=0)
    _ = ap.add_argument("--mem-block", type=int, default=128)
    _ = ap.add_argument("--local-window", type=int, default=1024)
    _ = ap.add_argument("--q-chunk", type=int, default=256)
    _ = ap.add_argument("--summarizer", type=str, default="conv")
    _ = ap.add_argument("--compute-kl", action="store_true")
    _ = ap.add_argument("--max-abs-logit-tol", type=float, default=0.1)
    _ = ap.add_argument("--delta-nll-tol", type=float, default=0.05)
    _ = ap.add_argument("--ppl-ratio-tol", type=float, default=1.05)
    _ = ap.add_argument("--kl-tol", type=float, default=None)
    args = ap.parse_args(argv)

    device = torch.device(str(args.device))
    ckpt_obj = _torch_load_obj(str(args.ckpt), device=device)
    ckpt = as_str_object_dict(ckpt_obj)
    if ckpt is None:
        raise ValueError("Checkpoint payload must be a dict-like object")
    cfg_dict = as_str_object_dict(ckpt.get("config", None))
    if cfg_dict is None:
        raise ValueError("Checkpoint missing 'config'.")

    cfg = ModelConfig.from_dict(cfg_dict, device=device)
    if str(getattr(cfg, "attn_mode", "")) != "decoupled":
        raise ValueError("Long-seq approximation drift validation is only supported for decoupled checkpoints")
    if bool(getattr(cfg, "null_attn", False)):
        cfg.null_attn = False

    model = GPT(cfg).to(device)
    sd = _as_state_dict(ckpt.get("model"))
    if sd is None:
        raise ValueError("Checkpoint missing 'model' state_dict")
    _ = model.load_state_dict(sd, strict=False)

    _set_dropout_p(model, p=0.0)
    model.train()
    cfg.dropout = 0.0

    B = int(args.batch)
    ref_len = int(args.ref_len)
    if ref_len <= 2:
        raise ValueError("--ref-len must be > 2")
    if int(ref_len) > int(cfg.block_size):
        raise ValueError(f"--ref-len {ref_len} exceeds checkpoint block_size {cfg.block_size}")

    torch.manual_seed(0)
    idx = torch.randint(0, int(cfg.vocab_size), (B, ref_len + 1), device=device, dtype=torch.long)
    idx_in = idx[:, :ref_len]
    tgt = idx[:, 1 : ref_len + 1]

    t0 = time.time()
    logits_ref = _run_forward(
        model,
        idx_in,
        long_seq_enabled=False,
        mem_block=int(args.mem_block),
        local_window=int(args.local_window),
        q_chunk=int(args.q_chunk),
        summarizer=str(args.summarizer),
    )
    t1 = time.time()

    logits_approx = _run_forward(
        model,
        idx_in,
        long_seq_enabled=True,
        mem_block=int(args.mem_block),
        local_window=int(args.local_window),
        q_chunk=int(args.q_chunk),
        summarizer=str(args.summarizer),
    )
    t2 = time.time()

    metrics = Metrics.compare(logits_ref, logits_approx, tgt, compute_kl=bool(args.compute_kl))
    mx = float(metrics.get("max_abs_logit", float("nan")))
    dnll = float(metrics.get("delta_nll", float("nan")))
    pr = float(metrics.get("ppl_ratio", float("nan")))
    klv = metrics.get("kl_base_cand", None)

    print(f"ref_len={ref_len} ref_s={t1 - t0:.3f} approx_s={t2 - t1:.3f} device={device}")
    print(f"metrics={metrics}")

    if not torch.isfinite(torch.tensor(mx)) or mx > float(args.max_abs_logit_tol):
        raise AssertionError(f"max_abs_logit {mx:.4g} > {float(args.max_abs_logit_tol):.4g}")
    if not torch.isfinite(torch.tensor(dnll)) or dnll > float(args.delta_nll_tol):
        raise AssertionError(f"delta_nll {dnll:.4g} > {float(args.delta_nll_tol):.4g}")
    if not torch.isfinite(torch.tensor(pr)) or pr > float(args.ppl_ratio_tol):
        raise AssertionError(f"ppl_ratio {pr:.4g} > {float(args.ppl_ratio_tol):.4g}")
    if args.kl_tol is not None and klv is not None:
        if not torch.isfinite(torch.tensor(float(klv))) or float(klv) > float(args.kl_tol):
            raise AssertionError(f"kl_base_cand {float(klv):.4g} > {float(args.kl_tol):.4g}")

    approx_len = int(args.approx_only_len)
    if approx_len > 0:
        if int(approx_len) > int(cfg.block_size):
            raise ValueError(f"--approx-only-len {approx_len} exceeds checkpoint block_size {cfg.block_size}")
        idx2 = torch.randint(0, int(cfg.vocab_size), (B, approx_len), device=device, dtype=torch.long)
        t3 = time.time()
        logits_long = _run_forward(
            model,
            idx2,
            long_seq_enabled=True,
            mem_block=int(args.mem_block),
            local_window=int(args.local_window),
            q_chunk=int(args.q_chunk),
            summarizer=str(args.summarizer),
        )
        t4 = time.time()
        if not bool(torch.isfinite(logits_long).all()):
            raise AssertionError("approx-only forward produced non-finite logits")
        print(f"approx_only_len={approx_len} approx_only_s={t4 - t3:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
