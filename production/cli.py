from __future__ import annotations

import argparse
import sys
from typing import Any, Optional


class _MinimalParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover
        # Provide a stronger hint for common migration failure mode.
        if "unrecognized arguments" in message:
            message = (
                f"{message}\n\n"
                f"Hint: this project now defaults to a minimal CLI. "
                f"Pass `--expert` to enable advanced/optimization flags."
            )
        super().error(message)


def _build_expert_arg_parser() -> argparse.ArgumentParser:
    """Build the full (expert) CLI argparser.

    This is the legacy v30-compatible surface area. It remains available behind `--expert`
    so the default UX can stay intent-driven and low-flag.
    """
    ap = argparse.ArgumentParser()

    # Global mode switches (minimal parser also exposes these).
    ap.add_argument(
        "--expert",
        action="store_true",
        help="Enable expert/legacy flags (optimization knobs, KV-cache overrides, etc.).",
    )
    ap.add_argument(
        "--no-selfopt",
        action="store_true",
        help="Disable all self-optimization/autotuning (debug/repro).",
    )

    # ---- Experiment suite controls (new in v27) ----
    # Import here so `--help` works even if other heavy deps are missing.
    from production.config import SIZE_PRESETS, EXP_PRESETS

    ap.add_argument(
        "--size",
        type=str,
        default=None,
        choices=list(SIZE_PRESETS.keys()),
        help="Preset model+train size (tiny/small/medium/large). Applies only when provided.",
    )
    ap.add_argument(
        "--exp",
        type=str,
        default=None,
        choices=sorted(list(EXP_PRESETS.keys()) + ["paper_all"]),
        help="Preset experiment configuration. Applies only when provided.",
    )
    ap.add_argument(
        "--run-root",
        type=str,
        default="runs",
        help="Root directory for auto run dirs when --out-dir is omitted.",
    )
    ap.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional suffix for auto run dirs, e.g. 'seed2' -> runs/small_decoupled_seed2",
    )
    ap.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config (after presets/overrides) and exit.",
    )

    # ---- Core I/O ----
    ap.add_argument("--data", type=str, default=None, help="Token dataset path. For train mode only.")
    ap.add_argument(
        "--data-format",
        type=str,
        default="auto",
        choices=["auto", "text", "npy", "bin", "pt"],
        help="Dataset format. 'text' expects whitespace-separated ints. 'npy' uses np.load(mmap). 'bin' uses np.memmap. 'pt' loads a torch tensor.",
    )
    ap.add_argument(
        "--data-dtype",
        type=str,
        default="uint16",
        help="For --data-format bin: numpy dtype (e.g. uint16, uint32, int32).",
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Explicit vocab size (recommended for binary/mmap datasets). If omitted and tokenizer=tiktoken, defaults to 50257.",
    )
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (tail split).")

    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)

    # ---- Compile / AMP ----
    ap.add_argument("--compile", action="store_true", help="Use torch.compile(...) for speed (experimental).")
    ap.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if a preset enables it.")
    ap.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (if --compile).",
    )
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Enable torch.amp autocast (mixed precision) for training on CUDA/MPS/CPU (experimental).",
    )
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP even if a preset enables it.")
    ap.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Autocast compute dtype. bf16 is usually safest; fp16 may require loss scaling.",
    )
    ap.add_argument(
        "--param-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Model parameter dtype. fp32 baseline; bf16/fp16 reduce memory ~2x (helpful for bigger models).",
    )
    ap.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="torch.set_float32_matmul_precision(...) hint for float32 matmuls (may improve speed).",
    )

    # ---- Model ----
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=512)

    ap.add_argument("--attn-mode", type=str, default="bottleneck", choices=["standard", "bottleneck", "decoupled", "gqa"])
    ap.add_argument(
        "--kv-head",
        type=int,
        default=None,
        help="For --attn-mode gqa: number of KV heads (must divide n_head). Default = n_head",
    )
    ap.add_argument("--attn-dim", type=int, default=512)
    ap.add_argument("--sem-dim", type=int, default=32)
    ap.add_argument("--geo-dim", type=int, default=64)

    # (Decoupled) Per-head sem/geo mixing gate. Enabled by default; disable with --no-decoupled-gate.
    ap.add_argument(
        "--no-decoupled-gate",
        action="store_true",
        help="Disable the per-head semantic vs geometric mixing gate (decoupled attention).",
    )

    ap.add_argument("--no-rope", action="store_true")
    ap.add_argument("--rope", action="store_true", help="Force-enable RoPE even if a preset would disable it.")
    ap.add_argument("--rope-base", type=float, default=10000.0)

    ap.add_argument("--tie-qk", action="store_true")
    ap.add_argument("--no-tie-qk", action="store_true", help="Force-disable tie_qk (useful for overrides with presets).")

    ap.add_argument("--null-attn", action="store_true")
    ap.add_argument("--no-null-attn", action="store_true", help="Force-disable null_attn (useful for overrides with presets).")

    ap.add_argument("--no-learned-temp", action="store_true")

    ap.add_argument("--mlp", type=str, default="swiglu", choices=["swiglu", "gelu"])
    ap.add_argument("--dropout", type=float, default=0.0)

    # ---- Training ----
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument(
        "--legacy-micro-steps",
        action="store_true",
        help="Legacy runner behavior: interpret --steps as micro-steps (optimizer steps happen every --grad-accum). "
             "Default is optimizer-step semantics (v29/v30-compatible).",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="Micro-batch size (per optimizer step if --grad-accum=1).")
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (global batch = batch_size * grad_accum).")
    ap.add_argument(
        "--batch-by-seq",
        type=str,
        default=None,
        help="Optional mapping from training seq_len -> (batch_size, grad_accum). "
             "Format: '512:64x1,1024:32x2,2048:8x8'. "
             "At runtime we pick the entry tuned for the smallest seq >= current seq (or the max entry if current seq is larger). "
             "Useful on MPS where (B*T*V) can hit INT_MAX at larger seq.",
    )
    ap.add_argument(
        "--batch-schedule",
        type=str,
        default=None,
        help="Optional batch/accum schedule in optimizer-step space. Format: '64x1@0,32x2@200,8x8@600'. "
             "If set, overrides --batch-by-seq and --batch-size/--grad-accum.",
    )
    ap.add_argument("--train-seq-len", type=int, default=0, help="Effective sequence length for training batches (0 = use --block).")
    ap.add_argument(
        "--seq-schedule",
        type=str,
        default=None,
        help="Optional seq-len curriculum: 'len@step,len@step,...' e.g. '256@0,512@1000,1024@3000'.",
    )
    ap.add_argument("--eval-seq-len", type=int, default=0, help="Eval sequence length (0 = match training seq-len).")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (recompute activations) to save memory.")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing even if a preset enables it.")
    # Training autotune (opt-in): short microbench sweep to recommend batch/accum/seq_len for this device.
    ap.add_argument(
        "--train-autotune",
        type=str,
        default="off",
        choices=["off", "quick"],
        help="Opt-in training autotune. Runs short probes, prints a recommendation, then exits (no training).",
    )
    ap.add_argument("--train-autotune-gbs", type=int, default=64, help="Target global batch size for autotune (gbs = batch_size * grad_accum).")
    # Defaults are intentionally small/fast for "quick" mode; override to widen the search.
    ap.add_argument("--train-autotune-seq-lens", type=str, default="512", help="Comma-separated seq_lens to probe (clamped to --block).")
    ap.add_argument("--train-autotune-batch-sizes", type=str, default="8,16,24,32,48,64", help="Comma-separated micro-batch sizes to try.")
    ap.add_argument("--train-autotune-warmup", type=int, default=0, help="Warmup optimizer steps per candidate during autotune.")
    ap.add_argument("--train-autotune-iters", type=int, default=1, help="Timed optimizer steps per candidate during autotune.")
    ap.add_argument(
        "--train-autotune-max-driver-gb",
        type=float,
        default=0.0,
        help="If >0, reject candidates whose peak device driver allocation exceeds this (GB). On MPS uses mps driver bytes.",
    )
    ap.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "lion"],
        help="Optimizer. lion uses ~1 momentum state (lower memory) and can be faster on big models.",
    )
    ap.add_argument("--adam-betas", type=str, default="0.9,0.95", help="AdamW betas as 'b1,b2'.")
    ap.add_argument("--adam-eps", type=float, default=1e-8, help="AdamW epsilon.")
    ap.add_argument("--lion-betas", type=str, default="0.9,0.99", help="Lion betas as 'b1,b2'.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"], help="Learning rate schedule.")
    ap.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps for lr schedule.")
    ap.add_argument("--min-lr", type=float, default=0.0, help="Minimum lr for cosine schedule.")
    ap.add_argument("--opt-foreach", action="store_true", help="Use foreach optimizer implementation when available (can be faster).")
    ap.add_argument("--opt-fused", action="store_true", help="Use fused optimizer implementation when available (CUDA only).")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--nan-policy",
        type=str,
        default="error",
        choices=["error", "skip", "reduce_lr"],
        help="What to do if loss/grad becomes non-finite during training. "
             "error: raise immediately. skip: skip the optimizer step. reduce_lr: skip + decay lr multiplier.",
    )
    ap.add_argument("--nan-lr-decay", type=float, default=0.5, help="When --nan-policy=reduce_lr, multiply lr by this factor on each non-finite event.")
    ap.add_argument("--save-every", type=int, default=0, help="Checkpoint interval (steps). 0 disables.")
    ap.add_argument("--eval-every", type=int, default=200, help="Eval interval (optimizer steps). 0 disables eval.")
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=100, help="Train-step logging interval (JSONL + console).")

    # ---- Instrumentation (new in v27) ----
    ap.add_argument(
        "--instrument",
        type=str,
        default="full",
        choices=["off", "basic", "medium", "full"],
        help="off: minimal. basic/medium: JSONL+summary+entropy. full: +HDF5 matrices/SVD + analysis.png",
    )
    ap.add_argument("--analysis-every", type=int, default=100, help="Deep analysis interval. 0 disables.")
    ap.add_argument("--analysis-max-tokens", type=int, default=256, help="Max tokens for attention matrix analysis.")
    ap.add_argument("--analysis-layers", type=str, default="0,-1", help="Comma-separated layer indices to analyze (supports negatives).")
    ap.add_argument("--analysis-heads", type=str, default="0", help="Comma-separated head indices to analyze for SVD/matrix dumps.")
    ap.add_argument("--analysis-topk", type=int, default=8, help="Top-k mass metric for sparsity.")
    ap.add_argument("--analysis-local-window", type=int, default=32, help="Locality window for 'local mass' metric.")
    ap.add_argument("--analysis-save-scores", action="store_true", help="(Decoupled) Save sem/geo score matrices into analysis.h5 (bigger).")
    ap.add_argument("--live", type=str, default="auto", choices=["auto", "off", "basic", "rich"], help="Console live dashboard. 'auto' uses rich if installed + TTY.")
    ap.add_argument("--live-update-every", type=int, default=1, help="Live dashboard refresh interval (optimizer steps).")
    ap.add_argument("--sync-timing", action="store_true", help="Synchronize device before timing/memory reads (more accurate, slightly slower).")
    ap.add_argument("--live-plot", action="store_true", help="Realtime matplotlib plots (dev only).")
    ap.add_argument("--tb", action="store_true", help="Write TensorBoard scalars (requires `tensorboard` package).")
    ap.add_argument("--wandb", action="store_true", help="Write Weights & Biases scalars (requires `wandb` package).")
    ap.add_argument("--wandb-project", type=str, default="experiments", help="W&B project name.")
    ap.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team (optional).")
    ap.add_argument("--wandb-name", type=str, default=None, help="W&B run name (optional). Defaults to out-dir basename.")
    ap.add_argument("--wandb-group", type=str, default=None, help="W&B group name (optional).")
    ap.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags (optional).")
    ap.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "online", "offline"],
                    help="W&B mode. 'offline' writes locally; 'online' syncs; 'disabled' no-op.")

    # ---- Mode ----
    ap.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    ap.add_argument("--ckpt", type=str, default=None)

    # ---- Sampling ----
    ap.add_argument("--prompt-tokens", type=str, default="0")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument(
        "--draft-ckpt",
        type=str,
        default=None,
        help="v31: Optional draft checkpoint for speculative decoding (faster decode by proposing k tokens and verifying with the main model).",
    )
    ap.add_argument("--spec-k", type=int, default=4, help="v31: Draft proposal length for speculative decoding.")
    ap.add_argument(
        "--spec-method",
        type=str,
        default="reject_sampling",
        choices=["reject_sampling", "greedy"],
        help="v31: Speculative decoding accept/reject method. 'reject_sampling' is the proper p/q gate; 'greedy' accepts only if argmax matches (debug).",
    )
    ap.add_argument(
        "--spec-extra-token",
        action="store_true",
        help="v31: If all k draft tokens are accepted, also sample one extra token from the verifier (more correct, slightly slower).",
    )
    ap.add_argument(
        "--spec-disable-below-accept",
        type=float,
        default=0.0,
        help="v31: Disable speculative decoding online if recent acceptance rate falls below this threshold (0 disables).",
    )

    # ---- KV cache / decode (generation) ----
    ap.add_argument(
        "--kv-cache",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
        help="Default KV-cache format (can be overridden per-tensor with the --kv-cache-* flags). "
             "For decoupled attention, note the geometric (RoPE) path is typically more sensitive to quantization error; "
             "defaults often keep k_geo at higher precision than k_sem to preserve positional fidelity.",
    )
    ap.add_argument("--kv-qblock", type=int, default=32, help="Quantization block size along the channel dimension.")
    ap.add_argument("--kv-residual", type=int, default=128, help="Keep this many newest KV tokens in fp16 as a hot residual window (only for quantized caches).")
    ap.add_argument(
        "--kv-policy",
        type=str,
        default=None,
        help="(Expert) Atomic KV-cache policy string for decoupled attention. "
        "Format: 'ksem=<kind>@<qblock>,kgeo=<kind>@<qblock>,v=<kind>@<qblock>,resid=<int>'. "
        "When set, overrides the per-tensor kv-cache flags; cache-policy self-tuning is disabled (decode tuning may still run).",
    )
    ap.add_argument(
        "--kv-decode-block",
        type=int,
        default=1024,
        help="Sequence-block size for streaming decode attention (smaller = less memory, more Python overhead).",
    )
    ap.add_argument(
        "--kv-fused",
        type=str,
        default="auto",
        choices=["none", "auto", "triton1pass", "triton2pass"],
        help="Use fused decode kernels when available. 'auto' picks a sensible kernel when Triton+CUDA are available.",
    )

    # v25 self-optimizer (decode performance)
    ap.add_argument(
        "--self-opt",
        type=str,
        default="none",
        choices=["none", "startup", "online"],
        help="Self-optimize decode-time knobs (decode_block, fused kernel choice, launch params).",
    )
    ap.add_argument("--self-opt-cache", type=str, default=None, help="JSON path to persist tuned plans across runs (optional).")
    ap.add_argument(
        "--self-opt-decode-blocks",
        type=str,
        default="256,512,1024,2048",
        help="Comma-separated candidate kv_decode_block values for tuning.",
    )
    ap.add_argument("--self-opt-block-n", type=str, default="128", help="Comma-separated BLOCK_N candidates for fused kernels (e.g. '64,128').")
    ap.add_argument("--self-opt-warps", type=str, default="4,8", help="Comma-separated num_warps candidates for fused kernels.")
    ap.add_argument("--self-opt-stages", type=str, default="2,3", help="Comma-separated num_stages candidates for fused kernels.")
    ap.add_argument(
        "--self-opt-kernel-profiles",
        type=str,
        default="auto",
        choices=["auto", "small", "off"],
        help="v32: Hierarchical decode tuning. 'auto' tries a small built-in set of per-GPU kernel profiles; 'small' tries fewer; 'off' disables profiles (use explicit block_n/warps/stages space).",
    )
    ap.add_argument(
        "--self-opt-expert-launch-space",
        action="store_true",
        help="v32: Use the legacy cross-product search over --self-opt-block-n/--self-opt-warps/--self-opt-stages (bigger but more exhaustive).",
    )
    ap.add_argument("--self-opt-warmup", type=int, default=1, help="Warmup iterations per candidate during tuning.")
    ap.add_argument("--self-opt-iters", type=int, default=3, help="Timed iterations per candidate during tuning.")
    ap.add_argument("--self-opt-interval", type=int, default=256, help="Online mode: tune at most once every N decode steps per bucket.")
    ap.add_argument("--self-opt-hysteresis", type=float, default=0.03, help="Online mode: require this relative improvement to switch plans.")
    ap.add_argument("--self-opt-verbose", action="store_true", help="Print tuning decisions + chosen plans.")
    ap.add_argument("--self-opt-verify", action="store_true", help="Verify candidate outputs vs baseline while tuning (slow; debugging).")
    ap.add_argument("--self-opt-verify-tol", type=float, default=5e-3, help="Max abs error allowed for --self-opt-verify (fp32).")

    # v26 cache-policy self-optimizer (kv_residual, quant kind, qblock) — decoupled only
    ap.add_argument(
        "--self-opt-scope",
        type=str,
        default="all",
        choices=["decode", "cache", "all"],
        help="Which knobs to self-optimize. 'cache' tunes kv_residual/quant/qblock at startup; 'decode' tunes decode kernels; 'all' does both.",
    )
    ap.add_argument("--self-opt-residuals", type=str, default="0,32,64,128", help="Comma-separated kv_residual candidates for cache-policy tuning.")
    ap.add_argument("--self-opt-qblocks", type=str, default="16,32,64", help="Comma-separated qblock candidates for cache-policy tuning (applied to all quantized decoupled tensors).")
    ap.add_argument("--self-opt-k-sem-kinds", type=str, default="q4_0,nf4,q8_0,fp16", help="Comma-separated semantic-K quantization kinds to consider (decoupled).")
    ap.add_argument(
        "--self-opt-k-geo-kinds",
        type=str,
        default="q8_0,q4_0,fp16",
        help="Comma-separated geometric-K quantization kinds to consider (decoupled). "
             "Geometric K (RoPE path) is usually kept at higher precision to preserve positional fidelity; "
             "include lower-precision kinds here only if you intend to empirically validate that assumption.",
    )
    ap.add_argument("--self-opt-v-kinds", type=str, default="q4_0,q8_0,fp16", help="Comma-separated V quantization kinds to consider (decoupled).")
    ap.add_argument(
        "--self-opt-mem-budget-mb",
        type=float,
        default=None,
        help="Absolute memory budget in MB for KV cache-policy tuning (decoupled). If unset, uses baseline*(1+--self-opt-mem-overhead-frac).",
    )
    ap.add_argument("--self-opt-mem-overhead-frac", type=float, default=0.10, help="If --self-opt-mem-budget-mb is unset, allow this fractional overhead over baseline (residual=0).")
    ap.add_argument(
        "--self-opt-policy-prefix-len",
        type=int,
        default=None,
        help="Prefix length to benchmark during cache-policy tuning. If unset, derives from prompt/max_seq.",
    )
    ap.add_argument("--self-opt-policy-warmup", type=int, default=1, help="Warmup iterations for cache-policy microbench.")
    ap.add_argument("--self-opt-policy-iters", type=int, default=3, help="Timed iterations for cache-policy microbench.")
    ap.add_argument("--self-opt-policy-hysteresis", type=float, default=0.02, help="Cache-policy hillclimb: require this relative improvement to accept a move.")
    ap.add_argument("--self-opt-prefer-low-mem-within", type=float, default=0.02, help="Cache-policy tie-break: if speed is within this fraction, prefer lower memory.")
    ap.add_argument(
        "--self-opt-policy-quality",
        action="store_true",
        help="(Slow) After choosing a cache policy, run a small teacher-forced logits check vs fp16-cache baseline.",
    )
    ap.add_argument(
        "--self-opt-calib-tokens",
        type=str,
        default=None,
        help="Calibration tokens for --self-opt-policy-quality (either a path to .txt/.npy or whitespace-separated ints). Defaults to --prompt-tokens.",
    )
    ap.add_argument("--self-opt-calib-prefill", type=int, default=128, help="Prefill length for policy quality check.")
    ap.add_argument("--self-opt-calib-decode", type=int, default=32, help="Number of teacher-forced decode steps for policy quality check.")
    ap.add_argument("--self-opt-quality-tol", type=float, default=0.5, help="Max abs logit error allowed for policy quality check.")
    ap.add_argument("--self-opt-quality-delta-nll-tol", type=float, default=0.02, help="Quality gate: max allowed ΔNLL (nats/token) vs fp16 baseline on calibration tokens. Default 0.02 (~2%% ppl hit).")
    ap.add_argument("--self-opt-quality-ppl-ratio-tol", type=float, default=1.02, help="Quality gate: max allowed ppl_cand/ppl_base on calibration tokens. Default 1.02 (~2%% ppl hit).")
    ap.add_argument("--self-opt-quality-kl-tol", type=float, default=None, help="Optional quality gate: max allowed KL(p_base||p_cand) in nats/token on calibration tokens.")
    ap.add_argument("--self-opt-quality-kl", action="store_true", help="Compute and print KL(p_base||p_cand) even if --self-opt-quality-kl-tol is unset (slow).")

    # Long-horizon quality gate (final accept check; much slower).
    ap.add_argument(
        "--self-opt-policy-quality-long",
        action="store_true",
        help="(Very slow) After the short policy quality check passes, run a longer-horizon teacher-forced check vs fp16-cache baseline (captures accumulated quantization error).",
    )
    ap.add_argument(
        "--self-opt-calib-long-tokens",
        type=str,
        default=None,
        help="Calibration tokens for --self-opt-policy-quality-long (path to .txt/.npy or whitespace-separated ints). Defaults to --self-opt-calib-tokens (or --prompt-tokens).",
    )
    ap.add_argument("--self-opt-calib-long-prefill", type=int, default=4096, help="Prefill length for long-horizon policy quality check.")
    ap.add_argument("--self-opt-calib-long-decode", type=int, default=128, help="Teacher-forced decode steps for long-horizon policy quality check.")
    ap.add_argument("--self-opt-quality-long-tol", type=float, default=None, help="Optional long-horizon max abs logit error tolerance (defaults to --self-opt-quality-tol if unset).")
    ap.add_argument("--self-opt-quality-long-delta-nll-tol", type=float, default=None, help="Optional long-horizon max allowed ΔNLL (nats/token) vs fp16 baseline.")
    ap.add_argument("--self-opt-quality-long-ppl-ratio-tol", type=float, default=None, help="Optional long-horizon max allowed ppl_cand/ppl_base vs fp16 baseline.")
    ap.add_argument("--self-opt-quality-long-kl-tol", type=float, default=None, help="Optional long-horizon max allowed KL(p_base||p_cand) in nats/token.")
    ap.add_argument("--self-opt-quality-long-kl", action="store_true", help="Compute and print long-horizon KL(p_base||p_cand) even if --self-opt-quality-long-kl-tol is unset (slow).")
    ap.add_argument(
        "--self-opt-layerwise-cache",
        action="store_true",
        help="v31: If a chosen global cache policy fails quality gates, try a layerwise fallback (promote early layers to fp16, keep later layers quantized).",
    )
    ap.add_argument("--kv-cache-k", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"], help="Override K cache kind (standard/bottleneck/gqa).")
    ap.add_argument("--kv-cache-v", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"], help="Override V cache kind (standard/bottleneck/gqa and decoupled).")
    ap.add_argument("--kv-cache-k-sem", type=str, default=None, choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"], help="Override semantic K cache kind (decoupled only).")
    ap.add_argument(
        "--kv-cache-k-geo",
        type=str,
        default=None,
        choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
        help="Override geometric K cache kind (decoupled only). "
             "Geometric K carries relative position via RoPE and is hypothesized to be highly sensitive to quantization; "
             "aggressively quantizing k_geo can distort the rotational signal and degrade long-range coherence.",
    )
    ap.add_argument("--kv-qblock-k", type=int, default=None, help="Override K qblock (standard/bottleneck/gqa).")
    ap.add_argument("--kv-qblock-v", type=int, default=None, help="Override V qblock.")
    ap.add_argument("--kv-qblock-k-sem", type=int, default=None, help="Override semantic K qblock.")
    ap.add_argument("--kv-qblock-k-geo", type=int, default=None, help="Override geometric K qblock.")

    # Tokenizer
    ap.add_argument("--tokenizer", type=str, default="word", choices=["word", "tiktoken"])

    return ap


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the minimal, intent-first CLI argparser (default UX).

    Advanced optimization knobs are intentionally hidden; use `--expert` to access them.
    """
    ap = _MinimalParser()

    from production.config import SIZE_PRESETS, EXP_PRESETS

    # Minimal surface area: intent + I/O + instrumentation.
    ap.add_argument("--expert", action="store_true", help="Enable expert/legacy flags.")
    ap.add_argument("--no-selfopt", action="store_true", help="Disable all self-optimization/autotuning (debug/repro).")

    ap.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Run mode.")
    ap.add_argument("--size", type=str, default=None, choices=list(SIZE_PRESETS.keys()), help="Model+train size preset.")
    ap.add_argument("--exp", type=str, default=None, choices=sorted(list(EXP_PRESETS.keys()) + ["paper_all"]), help="Experiment preset.")

    ap.add_argument("--data", type=str, default=None, help="Dataset path (train mode).")
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (sample mode).")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (optional if size+exp given).")
    ap.add_argument("--run-root", type=str, default="runs", help="Root for auto run dirs (when out-dir omitted).")
    ap.add_argument("--run-tag", type=str, default=None, help="Optional suffix for auto run dirs.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)

    # Sampling essentials (keep user intent; optimization stays automatic).
    ap.add_argument("--prompt-tokens", type=str, default="0", help="Prompt as whitespace-separated token IDs (or text if tokenizer=tiktoken).")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)

    # Training essentials.
    ap.add_argument("--steps", type=int, default=None, help="Training steps (defaults from size preset when set).")

    # Instrumentation: keep existing UX but minimal.
    ap.add_argument("--instrument", type=str, default="auto", choices=["auto", "off", "basic", "rich"], help="Console instrumentation.")
    ap.add_argument("--tb", action="store_true", help="Write TensorBoard scalars.")
    ap.add_argument("--wandb", action="store_true", help="Write Weights & Biases scalars.")
    ap.add_argument("--wandb-project", type=str, default="experiments")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument("--wandb-group", type=str, default=None)
    ap.add_argument("--wandb-tags", type=str, default=None)
    ap.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "online", "offline"])

    ap.add_argument("--print-config", action="store_true", help="Print resolved model config and exit.")

    return ap


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse argv using the minimal CLI by default, upgrading to expert when requested.

    For minimal runs, we still populate *all* attributes expected by the runtime by merging
    onto the expert parser's defaults. This keeps downstream code stable while the CLI stays small.
    """
    if argv is None:
        argv = sys.argv[1:]

    want_expert = "--expert" in argv
    if want_expert:
        ap = _build_expert_arg_parser()
        return ap.parse_args(argv)

    # Parse minimal args (will error on any legacy/optimization flags).
    mini = build_arg_parser().parse_args(argv)

    # Start from the expert defaults so downstream code sees every expected attribute.
    base = _build_expert_arg_parser().parse_args([])
    for k, v in vars(mini).items():
        setattr(base, k, v)
    return base


def run(args: argparse.Namespace) -> int:
    """Execute the CLI request with v30-compatible behavior."""
    import copy
    import math
    import torch

    from production.config import apply_exp_preset, apply_size_preset, default_out_dir
    from production.config import pick_device, set_seed

    # ---- CLI validations (fail fast before heavy init) ----
    sdb = getattr(args, "spec_disable_below_accept", None)
    if sdb is not None and (not math.isfinite(float(sdb)) or float(sdb) < 0.0 or float(sdb) > 1.0):
        raise ValueError("--spec-disable-below-accept must be between 0.0 and 1.0")

    # Apply paper suite presets (only when provided)
    apply_size_preset(args)
    apply_exp_preset(args)

    # Explicit "force disable" flags (useful because store_true args can't be negated otherwise)
    if getattr(args, "no_null_attn", False):
        args.null_attn = False
    if getattr(args, "no_tie_qk", False):
        args.tie_qk = False
    if getattr(args, "rope", False):
        args.no_rope = False
    if getattr(args, "no_compile", False):
        args.compile = False
    if getattr(args, "no_amp", False):
        args.amp = False
    if getattr(args, "no_grad_checkpoint", False):
        args.grad_checkpoint = False

    # Derive out_dir if omitted and size+exp provided
    inferred = default_out_dir(args)
    if getattr(args, "out_dir", None) is None and inferred is not None:
        args.out_dir = inferred

    device = pick_device(getattr(args, "device", None))
    set_seed(int(getattr(args, "seed", 1337)))

    # Matmul precision hint (mostly impacts float32 matmuls).
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(getattr(args, "matmul_precision", "high")))
    except Exception:
        pass

    # For paper_all, run each experiment sequentially (train mode only).
    if getattr(args, "mode", "train") == "train" and getattr(args, "exp", None) == "paper_all":
        for exp in ["paper_baseline", "paper_bottleneck", "paper_decoupled", "paper_gqa"]:
            a2 = copy.deepcopy(args)
            a2.exp = exp
            apply_exp_preset(a2)
            if getattr(a2, "no_null_attn", False):
                a2.null_attn = False
            if getattr(a2, "no_tie_qk", False):
                a2.tie_qk = False
            if getattr(a2, "rope", False):
                a2.no_rope = False
            inferred2 = default_out_dir(a2)
            if inferred2 is not None:
                a2.out_dir = inferred2
            from production.runner import run_single

            run_single(a2, device)
        return 0

    from production.runner import run_single

    run_single(args, device)
    return 0


def main() -> int:
    """Module entrypoint so `python -m production.cli ...` works."""
    import math

    args = parse_args()

    # Validate args immediately after parsing so invalid values fail fast (before any device/model init).
    sdb = getattr(args, "spec_disable_below_accept", None)
    if sdb is not None and (not math.isfinite(float(sdb)) or float(sdb) < 0.0 or float(sdb) > 1.0):
        raise ValueError("--spec-disable-below-accept must be between 0.0 and 1.0")

    return int(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

