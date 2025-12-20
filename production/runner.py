from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def run_single(args: argparse.Namespace, device: torch.device) -> None:
    # Local import so CLI --help doesn't require torch/tiktoken.
    from production.data import (
        determine_vocab_size,
        get_batch_any,
        infer_data_format,
        load_tokens_any,
        split_train_val,
    )
    from production.instrumentation import RunLogger
    from production.model import GPT, ModelConfig
    from production.runtime_tuning import KVSelfOptConfig

    try:
        import tiktoken  # type: ignore
    except Exception:
        tiktoken = None  # type: ignore

    # -------------------------
    # Sample mode
    # -------------------------
    if args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required for --mode sample")

        ckpt = torch.load(args.ckpt, map_location=device)
        cfg_dict = ckpt.get("config", None)
        if cfg_dict is None:
            raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")
        cfg = ModelConfig(**cfg_dict)
        model = GPT(cfg).to(device)

        incompatible = model.load_state_dict(ckpt["model"], strict=False)
        bad_missing = [k for k in incompatible.missing_keys if "decoupled_gate_logit" not in k]
        bad_unexpected = [k for k in incompatible.unexpected_keys if "decoupled_gate_logit" not in k]
        if bad_missing or bad_unexpected:
            model.load_state_dict(ckpt["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"[warn] Non-strict checkpoint load. Missing={incompatible.missing_keys} Unexpected={incompatible.unexpected_keys}")
        model.eval()

        # Prompt: either raw token IDs or text (tiktoken only)
        try:
            prompt_ids = [int(t) for t in args.prompt_tokens.strip().split()]
        except ValueError:
            if args.tokenizer != "tiktoken":
                raise ValueError("Text prompts require --tokenizer tiktoken")
            if tiktoken is None:
                raise ImportError("tiktoken needed for text prompts")
            enc = tiktoken.get_encoding("gpt2")
            prompt_ids = enc.encode_ordinary(args.prompt_tokens)

        prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

        def _csv_ints(s: Optional[str]) -> Tuple[int, ...]:
            if s is None:
                return ()
            parts: List[int] = []
            for x in str(s).split(","):
                x = x.strip()
                if not x:
                    continue
                try:
                    parts.append(int(x))
                except Exception:
                    pass
            return tuple(parts)

        def _csv_strs(s: Optional[str]) -> Tuple[str, ...]:
            if s is None:
                return ()
            parts: List[str] = []
            for x in str(s).split(","):
                x = x.strip()
                if x:
                    parts.append(x)
            return tuple(parts)

        self_opt_cfg = None
        if getattr(args, "self_opt", "none") != "none":
            self_opt_cfg = KVSelfOptConfig(
                mode=args.self_opt,
                scope=getattr(args, "self_opt_scope", "all"),
                decode_blocks=_csv_ints(getattr(args, "self_opt_decode_blocks", "")) or (256, 512, 1024, 2048),
                block_ns=_csv_ints(getattr(args, "self_opt_block_n", "")) or (128,),
                warps=_csv_ints(getattr(args, "self_opt_warps", "")) or (4, 8),
                stages=_csv_ints(getattr(args, "self_opt_stages", "")) or (2, 3),
                kernel_profiles=str(getattr(args, "self_opt_kernel_profiles", "auto")),
                expert_launch_space=bool(getattr(args, "self_opt_expert_launch_space", False)),
                warmup=int(getattr(args, "self_opt_warmup", 1)),
                iters=int(getattr(args, "self_opt_iters", 3)),
                interval=int(getattr(args, "self_opt_interval", 256)),
                hysteresis=float(getattr(args, "self_opt_hysteresis", 0.03)),
                cache_path=getattr(args, "self_opt_cache", None),
                verbose=bool(getattr(args, "self_opt_verbose", False)),
                verify=bool(getattr(args, "self_opt_verify", False)),
                verify_tol=float(getattr(args, "self_opt_verify_tol", 5e-3)),
                residuals=_csv_ints(getattr(args, "self_opt_residuals", "")) or (0, 32, 64, 128),
                qblocks=_csv_ints(getattr(args, "self_opt_qblocks", "")) or (16, 32, 64),
                k_sem_kinds=_csv_strs(getattr(args, "self_opt_k_sem_kinds", "")) or ("q4_0", "nf4", "q8_0", "fp16"),
                k_geo_kinds=_csv_strs(getattr(args, "self_opt_k_geo_kinds", "")) or ("q8_0", "q4_0", "fp16"),
                v_kinds=_csv_strs(getattr(args, "self_opt_v_kinds", "")) or ("q4_0", "q8_0", "fp16"),
                mem_budget_mb=getattr(args, "self_opt_mem_budget_mb", None),
                mem_overhead_frac=float(getattr(args, "self_opt_mem_overhead_frac", 0.10)),
                policy_prefix_len=getattr(args, "self_opt_policy_prefix_len", None),
                policy_warmup=int(getattr(args, "self_opt_policy_warmup", 1)),
                policy_iters=int(getattr(args, "self_opt_policy_iters", 3)),
                policy_hysteresis=float(getattr(args, "self_opt_policy_hysteresis", 0.02)),
                prefer_lower_mem_within=float(getattr(args, "self_opt_prefer_low_mem_within", 0.02)),
                policy_quality=bool(getattr(args, "self_opt_policy_quality", False)),
                calib_tokens=getattr(args, "self_opt_calib_tokens", None),
                calib_prefill=int(getattr(args, "self_opt_calib_prefill", 64)),
                calib_decode_steps=int(getattr(args, "self_opt_calib_decode", 8)),
                quality_tol=float(getattr(args, "self_opt_quality_tol", 0.5)),
                quality_delta_nll_tol=getattr(args, "self_opt_quality_delta_nll_tol", None),
                quality_ppl_ratio_tol=getattr(args, "self_opt_quality_ppl_ratio_tol", None),
                quality_kl_tol=getattr(args, "self_opt_quality_kl_tol", None),
                quality_compute_kl=bool(getattr(args, "self_opt_quality_kl", False)),
                policy_quality_long=bool(getattr(args, "self_opt_policy_quality_long", False)),
                calib_long_tokens=getattr(args, "self_opt_calib_long_tokens", None),
                calib_long_prefill=int(getattr(args, "self_opt_calib_long_prefill", 4096)),
                calib_long_decode_steps=int(getattr(args, "self_opt_calib_long_decode", 128)),
                quality_long_tol=getattr(args, "self_opt_quality_long_tol", None),
                quality_long_delta_nll_tol=getattr(args, "self_opt_quality_long_delta_nll_tol", None),
                quality_long_ppl_ratio_tol=getattr(args, "self_opt_quality_long_ppl_ratio_tol", None),
                quality_long_kl_tol=getattr(args, "self_opt_quality_long_kl_tol", None),
                quality_long_compute_kl=bool(getattr(args, "self_opt_quality_long_kl", False)),
                layerwise_cache=bool(getattr(args, "self_opt_layerwise_cache", False)),
            )

        logger = None
        if args.instrument != "off" or args.live_plot or args.tb or bool(getattr(args, "wandb", False)):
            logger = RunLogger(
                args.out_dir,
                instrument=args.instrument,
                cfg=cfg,
                args=args,
                device=device,
                live_plot=bool(args.live_plot),
                tb=bool(args.tb),
                wandb=bool(getattr(args, "wandb", False)),
            )

        print(f"Generating {args.max_new_tokens} tokens...")
        try:
            if getattr(args, "draft_ckpt", None):
                dckpt = torch.load(str(args.draft_ckpt), map_location=device)
                dcfg_dict = dckpt.get("config", None)
                if dcfg_dict is None:
                    raise ValueError("Draft checkpoint missing 'config'. Can't reconstruct draft model safely.")
                dcfg = ModelConfig(**dcfg_dict)
                draft = GPT(dcfg).to(device)
                incompatible_d = draft.load_state_dict(dckpt["model"], strict=False)
                bad_missing_d = [k for k in incompatible_d.missing_keys if "decoupled_gate_logit" not in k]
                bad_unexpected_d = [k for k in incompatible_d.unexpected_keys if "decoupled_gate_logit" not in k]
                if bad_missing_d or bad_unexpected_d:
                    draft.load_state_dict(dckpt["model"], strict=True)
                if incompatible_d.missing_keys or incompatible_d.unexpected_keys:
                    print(f"[warn] Non-strict draft checkpoint load. Missing={incompatible_d.missing_keys} Unexpected={incompatible_d.unexpected_keys}")

                # Basic safety: vocab size must match for token IDs to be meaningful.
                if int(dcfg.vocab_size) != int(cfg.vocab_size):
                    raise ValueError(f"Draft vocab_size {dcfg.vocab_size} != main vocab_size {cfg.vocab_size}")

                # Match main model's inference behavior (disable dropout, etc.)
                draft.eval()

                out = model.generate_speculative(
                    prompt,
                    draft_model=draft,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=(None if args.top_k is None else int(args.top_k)),
                    kv_cache=str(args.kv_cache),
                    kv_qblock=int(args.kv_qblock),
                    kv_residual=int(args.kv_residual),
                    kv_decode_block=int(args.kv_decode_block),
                    kv_fused=str(args.kv_fused),
                    self_opt=self_opt_cfg,
                    kv_cache_k=getattr(args, "kv_cache_k", None),
                    kv_cache_v=getattr(args, "kv_cache_v", None),
                    kv_cache_k_sem=getattr(args, "kv_cache_k_sem", None),
                    kv_cache_k_geo=getattr(args, "kv_cache_k_geo", None),
                    kv_qblock_k=getattr(args, "kv_qblock_k", None),
                    kv_qblock_v=getattr(args, "kv_qblock_v", None),
                    kv_qblock_k_sem=getattr(args, "kv_qblock_k_sem", None),
                    kv_qblock_k_geo=getattr(args, "kv_qblock_k_geo", None),
                    spec_k=int(getattr(args, "spec_k", 4)),
                    spec_method=str(getattr(args, "spec_method", "reject_sampling")),
                    spec_extra_token=bool(getattr(args, "spec_extra_token", False)),
                    spec_disable_below_accept=float(getattr(args, "spec_disable_below_accept", 0.0)),
                    log_callback=(logger.log if logger is not None else None),
                )
            else:
                out = model.generate(
                    prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=(None if args.top_k is None else int(args.top_k)),
                    kv_cache=str(args.kv_cache),
                    kv_qblock=int(args.kv_qblock),
                    kv_residual=int(args.kv_residual),
                    kv_decode_block=int(args.kv_decode_block),
                    kv_fused=str(args.kv_fused),
                    self_opt=self_opt_cfg,
                    kv_cache_k=getattr(args, "kv_cache_k", None),
                    kv_cache_v=getattr(args, "kv_cache_v", None),
                    kv_cache_k_sem=getattr(args, "kv_cache_k_sem", None),
                    kv_cache_k_geo=getattr(args, "kv_cache_k_geo", None),
                    kv_qblock_k=getattr(args, "kv_qblock_k", None),
                    kv_qblock_v=getattr(args, "kv_qblock_v", None),
                    kv_qblock_k_sem=getattr(args, "kv_qblock_k_sem", None),
                    kv_qblock_k_geo=getattr(args, "kv_qblock_k_geo", None),
                    log_callback=(logger.log if logger is not None else None),
                )
        finally:
            if logger is not None:
                logger.close()

        out_ids = out[0].detach().to("cpu").tolist()
        if args.tokenizer == "tiktoken":
            if tiktoken is None:
                raise ImportError("tiktoken not installed")
            enc = tiktoken.get_encoding("gpt2")
            print(enc.decode(out_ids))
        else:
            print(out_ids)
        return

    # -------------------------
    # Train mode
    # -------------------------
    if args.data is None:
        raise ValueError("--data is required for --mode train")
    if args.out_dir is None:
        raise ValueError("--out-dir is required for --mode train (or provide --size + --exp for auto dirs).")

    data_path = Path(args.data)
    fmt = infer_data_format(data_path, str(args.data_format))
    tokens_any = load_tokens_any(path=data_path, fmt=fmt, data_dtype=str(args.data_dtype))

    n_total = int(tokens_any.numel()) if isinstance(tokens_any, torch.Tensor) else int(len(tokens_any))
    if n_total < int(args.block) + 2:
        raise ValueError(f"Dataset too small: n_tokens={n_total} block={args.block}")

    train_view, val_view = split_train_val(tokens_any, val_frac=float(args.val_frac))
    vocab = determine_vocab_size(tokens_any=tokens_any, vocab_size=getattr(args, "vocab_size", None), tokenizer=str(args.tokenizer))

    cfg = ModelConfig(
        vocab_size=int(vocab),
        block_size=int(args.block),
        n_layer=int(args.layers),
        n_head=int(args.n_head),
        kv_head=args.kv_head,
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        embed_dim=int(args.embed_dim),
        attn_mode=str(args.attn_mode),
        attn_dim=int(args.attn_dim),
        sem_dim=int(args.sem_dim),
        geo_dim=int(args.geo_dim),
        decoupled_gate=(not args.no_decoupled_gate),
        rope=(not args.no_rope),
        rope_base=float(args.rope_base),
        tie_qk=bool(args.tie_qk),
        null_attn=bool(args.null_attn),
        learned_temp=(not args.no_learned_temp),
        mlp=str(args.mlp),
        dropout=float(args.dropout),
    )

    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
        return

    os.makedirs(str(args.out_dir), exist_ok=True)
    model = GPT(cfg).to(device)

    # Parameter dtype.
    def _supports_dtype(dev: torch.device, dt: torch.dtype) -> bool:
        try:
            x = torch.ones(8, device=dev, dtype=dt)
            y = (x * 1.0001).sum()
            _ = float(y.detach().to("cpu").item())
            return True
        except Exception:
            return False

    def resolve_dtype(dev: torch.device, spec: str, *, default: torch.dtype) -> torch.dtype:
        spec = str(spec).lower()
        if spec in ("fp32", "float32", "f32"):
            return torch.float32
        if spec in ("bf16", "bfloat16"):
            dt = torch.bfloat16
        elif spec in ("fp16", "float16", "f16"):
            dt = torch.float16
        else:
            dt = default
        if dev.type in ("cuda", "mps"):
            if dt in (torch.float16, torch.bfloat16) and not _supports_dtype(dev, dt):
                if dt == torch.float16 and _supports_dtype(dev, torch.bfloat16):
                    return torch.bfloat16
                return torch.float32
        if dev.type == "cpu" and dt == torch.float16:
            return torch.float32
        return dt

    param_dtype = resolve_dtype(device, str(args.param_dtype), default=torch.float32)
    if param_dtype != torch.float32:
        model = model.to(dtype=param_dtype)

    try:
        model.grad_checkpointing = bool(args.grad_checkpoint)
    except Exception:
        pass

    # torch.compile can be very expensive across many input shapes; for training autotune we
    # intentionally skip compile so the probe finishes quickly and doesn't look "hung" while compiling.
    compile_requested = bool(getattr(args, "compile", False))
    compile_explicit = "--compile" in sys.argv
    autotune_off = str(getattr(args, "train_autotune", "off")).lower() == "off"
    if compile_requested and hasattr(torch, "compile") and autotune_off:
        # MPS + torch.compile is still experimental and can introduce correctness issues for some models.
        # Only enable it on MPS when the user explicitly asked for it via --compile.
        if device.type == "mps" and not compile_explicit:
            print("[warn] skipping torch.compile on MPS (pass --compile explicitly to force-enable).")
        else:
            try:
                model = torch.compile(model, mode=str(args.compile_mode))
                print(f"torch.compile enabled (mode={args.compile_mode}).")
            except Exception as e:
                print(f"torch.compile failed, continuing without it: {e}")
    elif compile_requested and (not autotune_off):
        print("[autotune] skipping torch.compile during training autotune (too slow across many shapes).")

    # Ensure training-only flags survive compile wrappers.
    try:
        model.grad_checkpointing = bool(args.grad_checkpoint)
    except Exception:
        pass

    # AMP context
    amp_enabled = bool(getattr(args, "amp", False))
    amp_dtype = torch.bfloat16 if str(getattr(args, "amp_dtype", "bf16")) == "bf16" else torch.float16
    if device.type == "cpu":
        # CPU autocast is limited; allow but default to disabled.
        pass

    @contextlib.contextmanager
    def autocast_ctx():
        if not amp_enabled:
            yield
            return
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                yield
            return
        if device.type == "mps":
            with torch.autocast("mps", dtype=amp_dtype):
                yield
            return
        with torch.autocast("cpu", dtype=torch.bfloat16):
            yield

    # Optimizer
    def _parse_two_floats(s: str, default: Tuple[float, float]) -> Tuple[float, float]:
        try:
            a, b = str(s).split(",")
            return float(a), float(b)
        except Exception:
            return default

    class Lion(torch.optim.Optimizer):
        def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
            if lr <= 0.0:
                raise ValueError(f"Invalid lr: {lr}")
            b1, b2 = betas
            if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
                raise ValueError(f"Invalid betas: {betas}")
            defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):  # type: ignore
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = float(group["lr"])
                wd = float(group.get("weight_decay", 0.0))
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("Lion does not support sparse gradients.")
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(g, alpha=(1.0 - beta1))
                    p.add_(exp_avg.sign(), alpha=-lr)
                    exp_avg.mul_(beta2).add_(g, alpha=(1.0 - beta2))
            return loss

    opt_name = str(args.optimizer)
    if opt_name == "lion":
        lion_betas = _parse_two_floats(str(args.lion_betas), (0.9, 0.99))
        opt = Lion(model.parameters(), lr=float(args.lr), betas=lion_betas, weight_decay=float(args.weight_decay))
    else:
        adam_betas = _parse_two_floats(str(args.adam_betas), (0.9, 0.95))
        opt_kwargs: Dict[str, Any] = dict(
            lr=float(args.lr),
            betas=adam_betas,
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )
        # foreach/fused are best-effort; ignore if unsupported.
        if bool(getattr(args, "opt_foreach", False)):
            opt_kwargs["foreach"] = True
        if bool(getattr(args, "opt_fused", False)):
            opt_kwargs["fused"] = True
        try:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        except TypeError:
            opt_kwargs.pop("foreach", None)
            opt_kwargs.pop("fused", None)
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    def _first_nonfinite_grad(model: torch.nn.Module) -> Optional[str]:
        """Return a short description of the first parameter with a non-finite grad, if any."""
        try:
            for name, p in model.named_parameters():
                g = p.grad
                if g is None:
                    continue
                finite = torch.isfinite(g.detach())
                if bool(finite.all()):
                    continue
                # keep it cheap: just a few summary stats
                g_det = g.detach()
                num = int(g_det.numel())
                n_finite = int(finite.sum().to("cpu").item()) if num > 0 else 0
                # avoid nan in max by filtering if possible
                try:
                    max_abs = float(torch.nan_to_num(g_det.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().to("cpu").item())
                except Exception:
                    max_abs = float("nan")
                return f"{name} grad dtype={g_det.dtype} finite={n_finite}/{num} max|g|~{max_abs:.3g}"
        except Exception:
            return None
        return None

    def _clip_grad_norm_fp32(params, max_norm: float) -> torch.Tensor:
        """MPS-friendly grad clipping: accumulate norm in fp32 to avoid bf16 overflow in norm."""
        grads: List[torch.Tensor] = []
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            grads.append(g)
        if not grads:
            return torch.zeros([], device=device, dtype=torch.float32)
        total_sq = torch.zeros([], device=grads[0].device, dtype=torch.float32)
        for g in grads:
            gd = g.detach()
            if not torch.isfinite(gd).all():
                return torch.tensor(float("nan"), device=gd.device, dtype=torch.float32)
            total_sq = total_sq + (gd.float() * gd.float()).sum()
        total_norm = torch.sqrt(total_sq)
        # scale in-place
        denom = total_norm + 1e-6
        clip_coef = float(max_norm) / float(denom.to("cpu").item())
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef)
        return total_norm

    def lr_for_step(step: int, *, base_lr: float, total_steps: int, schedule: str, warmup_steps: int = 0, min_lr: float = 0.0) -> float:
        schedule = str(schedule).lower()
        total_steps = max(int(total_steps), 1)
        warmup_steps = max(int(warmup_steps), 0)
        if schedule == "constant":
            if warmup_steps > 0 and step < warmup_steps:
                return base_lr * (step + 1) / warmup_steps
            return base_lr
        if schedule == "cosine":
            if warmup_steps > 0 and step < warmup_steps:
                return base_lr * (step + 1) / warmup_steps
            denom = max(total_steps - warmup_steps, 1)
            t = (step - warmup_steps) / denom
            t = min(max(t, 0.0), 1.0)
            return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
        return base_lr

    # Logger
    want_logger = (
        str(args.instrument) != "off"
        or bool(getattr(args, "live_plot", False))
        or bool(getattr(args, "tb", False))
        or bool(getattr(args, "wandb", False))
    )
    logger = (
        RunLogger(
            str(args.out_dir),
            instrument=str(args.instrument),
            cfg=cfg,
            args=args,
            device=device,
            live_plot=bool(args.live_plot),
            tb=bool(args.tb),
            wandb=bool(getattr(args, "wandb", False)),
        )
        if want_logger
        else None
    )

    best_val = float("inf")
    last_step = 0
    t_start = time.time()
    tok_count = 0

    def device_synchronize(dev: torch.device) -> None:
        try:
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            elif dev.type == "mps":
                if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
        except Exception:
            pass

    def _empty_device_cache(dev: torch.device) -> None:
        try:
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def _is_oom_error(e: BaseException) -> bool:
        msg = str(e).lower()
        return (
            ("out of memory" in msg)
            or ("cuda error: out of memory" in msg)
            or ("cudnn error: out of memory" in msg)
            or ("mps backend out of memory" in msg)
            or ("resource exhausted" in msg)
        )

    def get_device_mem_stats(dev: torch.device) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            if dev.type == "cuda" and torch.cuda.is_available():
                out["cuda_mem_alloc_bytes"] = float(torch.cuda.memory_allocated(dev))
                out["cuda_mem_reserved_bytes"] = float(torch.cuda.memory_reserved(dev))
            elif dev.type == "mps":
                if hasattr(torch, "mps"):
                    if hasattr(torch.mps, "current_allocated_memory"):
                        out["mps_mem_alloc_bytes"] = float(torch.mps.current_allocated_memory())
                    if hasattr(torch.mps, "driver_allocated_memory"):
                        out["mps_mem_driver_bytes"] = float(torch.mps.driver_allocated_memory())
        except Exception:
            pass
        return out

    def _parse_seq_schedule(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
        if spec is None:
            return None
        spec = str(spec).strip()
        if not spec:
            return None
        pairs: List[Tuple[int, int]] = []
        for part in spec.split(","):
            part = part.strip()
            if not part or "@" not in part:
                continue
            a, b = part.split("@", 1)
            try:
                seq = int(a)
                st = int(b)
                pairs.append((st, seq))
            except Exception:
                continue
        pairs.sort(key=lambda x: x[0])
        return pairs if pairs else None

    def _seq_len_for_step(step_idx: int, *, default_seq_len: int, schedule: Optional[List[Tuple[int, int]]]) -> int:
        if not schedule:
            return int(default_seq_len)
        s = int(default_seq_len)
        for st, ln in schedule:
            if int(step_idx) >= int(st):
                s = int(ln)
            else:
                break
        return int(s)

    def _parse_bs_ga_pair(s: str) -> Optional[Tuple[int, int]]:
        s = str(s).strip().lower()
        if not s:
            return None
        # Accept separators: "64x1", "64*1"
        for sep in ("x", "*"):
            if sep in s:
                a, b = s.split(sep, 1)
                try:
                    bs = int(a.strip())
                    ga = int(b.strip())
                    if bs > 0 and ga > 0:
                        return bs, ga
                except Exception:
                    return None
        return None

    def _parse_batch_schedule(spec: Optional[str]) -> Optional[List[Tuple[int, int, int]]]:
        """'64x1@0,32x2@200' -> [(0,64,1),(200,32,2)] sorted by step."""
        if spec is None:
            return None
        spec = str(spec).strip()
        if not spec:
            return None
        out: List[Tuple[int, int, int]] = []
        for part in spec.split(","):
            part = part.strip()
            if not part or "@" not in part:
                continue
            lhs, rhs = part.split("@", 1)
            pair = _parse_bs_ga_pair(lhs)
            if pair is None:
                continue
            try:
                st = int(rhs.strip())
            except Exception:
                continue
            bs, ga = pair
            out.append((st, bs, ga))
        out.sort(key=lambda t: t[0])
        return out if out else None

    def _batch_for_step(step_idx: int, schedule: Optional[List[Tuple[int, int, int]]], *, default_bs: int, default_ga: int) -> Tuple[int, int]:
        if not schedule:
            return int(default_bs), int(default_ga)
        bs = int(default_bs)
        ga = int(default_ga)
        for st, b, g in schedule:
            if int(step_idx) >= int(st):
                bs = int(b)
                ga = int(g)
            else:
                break
        return int(max(1, bs)), int(max(1, ga))

    def _parse_batch_by_seq(spec: Optional[str]) -> Optional[Dict[int, Tuple[int, int]]]:
        """'512:64x1,1024:32x2' -> {512:(64,1), 1024:(32,2)}"""
        if spec is None:
            return None
        spec = str(spec).strip()
        if not spec:
            return None
        out: Dict[int, Tuple[int, int]] = {}
        for part in spec.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            a, b = part.split(":", 1)
            try:
                seq = int(a.strip())
            except Exception:
                continue
            pair = _parse_bs_ga_pair(b)
            if pair is None:
                continue
            bs, ga = pair
            out[int(seq)] = (int(bs), int(ga))
        return out if out else None

    def _batch_for_seq(seq_len: int, mapping: Optional[Dict[int, Tuple[int, int]]], *, default_bs: int, default_ga: int) -> Tuple[int, int]:
        """Conservative choice: pick mapping for the smallest seq_key >= current seq_len (or max key)."""
        if not mapping:
            return int(default_bs), int(default_ga)
        keys = sorted(int(k) for k in mapping.keys())
        chosen_key = keys[-1]
        for k in keys:
            if int(k) >= int(seq_len):
                chosen_key = int(k)
                break
        bs, ga = mapping[chosen_key]
        return int(max(1, bs)), int(max(1, ga))

    def estimate_loss(eval_iters: int) -> Tuple[float, float]:
        model.eval()
        losses_tr: List[float] = []
        losses_va: List[float] = []
        eval_seq = int(getattr(args, "eval_seq_len", 0) or 0)
        bs = int(getattr(args, "batch_size", 1))
        if eval_seq <= 0:
            eval_seq = int(getattr(args, "train_seq_len", 0) or 0)
        if eval_seq <= 0:
            eval_seq = int(args.block)
        eval_seq = int(min(max(2, eval_seq), int(cfg.block_size)))

        for _ in range(int(eval_iters)):
            xb, yb = get_batch_any(train_view, batch_size=bs, block_size=eval_seq, device=device)
            with autocast_ctx():
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses_tr.append(float(loss.detach().to("cpu").item()))
            xb, yb = get_batch_any(val_view, batch_size=bs, block_size=eval_seq, device=device)
            with autocast_ctx():
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses_va.append(float(loss.detach().to("cpu").item()))
        model.train()
        return float(sum(losses_tr) / max(1, len(losses_tr))), float(sum(losses_va) / max(1, len(losses_va)))

    # Training schedule: interpret schedule steps in optimizer-step space (v29/v30 semantics).
    seq_schedule = _parse_seq_schedule(getattr(args, "seq_schedule", None))
    batch_schedule = _parse_batch_schedule(getattr(args, "batch_schedule", None))
    batch_by_seq = _parse_batch_by_seq(getattr(args, "batch_by_seq", None))
    base_seq_len = int(getattr(args, "train_seq_len", 0) or 0)
    if base_seq_len <= 0:
        base_seq_len = int(cfg.block_size)
    base_seq_len = int(min(max(2, base_seq_len), int(cfg.block_size)))

    grad_accum_default = max(1, int(getattr(args, "grad_accum", 1)))
    micro_bs_default = max(1, int(getattr(args, "batch_size", 1)))

    # Opt-in training autotune: run short probes and exit (no training run performed).
    if str(getattr(args, "train_autotune", "off")).lower() != "off":
        mode = str(getattr(args, "train_autotune", "off")).lower()
        if mode == "quick":
            target_gbs = max(1, int(getattr(args, "train_autotune_gbs", 64)))
            try:
                bs_list = [int(x) for x in str(getattr(args, "train_autotune_batch_sizes", "")).split(",") if x.strip()]
            except Exception:
                bs_list = [1, 2, 4, 8, 16, 24, 32, 48, 64]
            # Try larger micro-batches first: they tend to be faster (less grad-accum overhead) and this
            # avoids spending forever benchmarking tiny microbatches (e.g. bs=1, ga=64).
            bs_list = sorted({max(1, int(x)) for x in bs_list}, reverse=True)
            try:
                seq_list = [int(x) for x in str(getattr(args, "train_autotune_seq_lens", "")).split(",") if x.strip()]
            except Exception:
                seq_list = [512, 1024, 2048]
            seq_list = [min(int(cfg.block_size), max(2, int(s))) for s in seq_list]
            seq_list = sorted({int(s) for s in seq_list})
            warm = max(0, int(getattr(args, "train_autotune_warmup", 1)))
            iters = max(1, int(getattr(args, "train_autotune_iters", 3)))
            max_driver_gb = float(getattr(args, "train_autotune_max_driver_gb", 0.0) or 0.0)

            def mem_driver_gb() -> float:
                m = get_device_mem_stats(device)
                if device.type == "mps":
                    b = float(m.get("mps_mem_driver_bytes", 0.0))
                elif device.type == "cuda":
                    b = float(m.get("cuda_mem_reserved_bytes", 0.0))
                else:
                    b = 0.0
                return b / (1024.0**3)

            print("[autotune] production training autotune enabled: probing candidates and then exiting.")
            best: Optional[Dict[str, Any]] = None
            best_by_seq: Dict[int, Dict[str, Any]] = {}
            for s in seq_list:
                print(f"[autotune] seq_len={s} (<= block={cfg.block_size})")
                best_s: Optional[Dict[str, Any]] = None
                for bs_try in bs_list:
                    # MPSGraph limitation: some ops (notably large logits tensors) fail when tensor numel exceeds INT_MAX.
                    # A practical proxy is B * T * vocab_size for logits in cross-entropy.
                    if device.type == "mps":
                        try:
                            max_elems = 2_147_483_647  # INT_MAX
                            logits_elems = int(bs_try) * int(s) * int(cfg.vocab_size)
                            if logits_elems > max_elems:
                                print(f"[autotune]   skipping bs={bs_try} (B*T*V={logits_elems} > INT_MAX) on MPS")
                                continue
                        except Exception:
                            pass
                    ga_try = max(1, target_gbs // bs_try)
                    gbs_eff = bs_try * ga_try
                    if gbs_eff < max(1, target_gbs // 4):
                        continue
                    print(f"[autotune]   trying bs={bs_try} ga={ga_try} (gbs={gbs_eff}) ...")
                    ok = True
                    peak = mem_driver_gb()
                    tok = 0
                    try:
                        model.train()
                        for _ in range(warm):
                            opt.zero_grad(set_to_none=True)
                            for _m in range(ga_try):
                                xb, yb = get_batch_any(train_view, batch_size=bs_try, block_size=s, device=device)
                                with autocast_ctx():
                                    logits, _ = model(xb)
                                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                                (loss / ga_try).backward()
                            if float(getattr(args, "grad_clip", 0.0)) > 0:
                                if device.type == "mps":
                                    _ = _clip_grad_norm_fp32(model.parameters(), float(args.grad_clip))
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                            opt.step()
                            peak = max(peak, mem_driver_gb())
                        device_synchronize(device)
                        t0 = time.perf_counter()
                        for _ in range(iters):
                            opt.zero_grad(set_to_none=True)
                            for _m in range(ga_try):
                                xb, yb = get_batch_any(train_view, batch_size=bs_try, block_size=s, device=device)
                                with autocast_ctx():
                                    logits, _ = model(xb)
                                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                                (loss / ga_try).backward()
                            if float(getattr(args, "grad_clip", 0.0)) > 0:
                                if device.type == "mps":
                                    _ = _clip_grad_norm_fp32(model.parameters(), float(args.grad_clip))
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                            opt.step()
                            tok += int(bs_try) * int(ga_try) * int(s)
                            peak = max(peak, mem_driver_gb())
                            if max_driver_gb > 0 and peak > max_driver_gb:
                                ok = False
                                break
                        device_synchronize(device)
                        dt = time.perf_counter() - t0
                        tok_s = float(tok / max(dt, 1e-9)) if ok else 0.0
                    except Exception as e:
                        ok = False
                        tok_s = 0.0
                        print(f"[autotune] bs={bs_try} ga={ga_try} (gbs={gbs_eff}) failed: {e}")
                    rec = {"seq_len": int(s), "batch_size": int(bs_try), "grad_accum": int(ga_try), "gbs": int(gbs_eff), "tok_s": float(tok_s), "peak_driver_gb": float(peak), "ok": bool(ok)}
                    if ok:
                        print(f"[autotune]     ok tok/s={tok_s:.0f} peak_driver={peak:.1f}GB")
                    if ok and (best_s is None or rec["tok_s"] > float(best_s["tok_s"])):
                        best_s = rec
                if best_s is not None:
                    print(f"[autotune] best@{s}: tok/s={best_s['tok_s']:.0f} bs={best_s['batch_size']} ga={best_s['grad_accum']} peak_driver={best_s['peak_driver_gb']:.1f}GB")
                    best = best_s  # keep last (largest seq in list) best
                    best_by_seq[int(s)] = dict(best_s)
            if best is not None:
                print("[autotune] recommendation (copy/paste): "
                      f"--batch-size {best['batch_size']} --grad-accum {best['grad_accum']} --train-seq-len {best['seq_len']}")
            if best_by_seq:
                parts = []
                for seq_k in sorted(best_by_seq.keys()):
                    r = best_by_seq[seq_k]
                    parts.append(f"{int(seq_k)}:{int(r['batch_size'])}x{int(r['grad_accum'])}")
                print('[autotune] batch-by-seq suggestion (copy/paste): --batch-by-seq "' + ",".join(parts) + '"')
            print("[autotune] done. Exiting now (no training run performed).")
            return

    # Timing accumulators for throughput measurement (optimizer steps)
    dt_acc = 0.0
    tok_acc = 0
    fwd_acc = 0.0
    bwd_acc = 0.0
    opt_acc = 0.0
    steps_in_acc = 0

    # Non-finite handling (opt-in via --nan-policy)
    nan_policy = str(getattr(args, "nan_policy", "error")).lower()
    nan_lr_decay = float(getattr(args, "nan_lr_decay", 0.5) or 0.5)
    lr_mult = 1.0

    model.train()
    opt.zero_grad(set_to_none=True)

    legacy_micro = bool(getattr(args, "legacy_micro_steps", False))
    if legacy_micro:
        print("[warn] --legacy-micro-steps enabled: interpreting --steps as micro-steps (legacy behavior).")
        if batch_schedule or batch_by_seq:
            raise ValueError("--batch-schedule/--batch-by-seq are defined in optimizer-step space and are incompatible with --legacy-micro-steps.")

    total_opt_steps = int(args.steps) if not legacy_micro else int(math.ceil(int(args.steps) / max(1, grad_accum_default)))

    # Self-optimizing batch caps (learned online via OOM events). Keys are seq_len.
    auto_bs_cap_by_seq: Dict[int, int] = {}
    warned_auto_caps: set[int] = set()
    warned_mps_logits: set[int] = set()

    for opt_step in range(1, total_opt_steps + 1):
        last_step = opt_step

        # LR schedule
        lr = lr_for_step(
            opt_step - 1,
            base_lr=float(args.lr),
            total_steps=int(args.steps),
            schedule=str(args.lr_schedule),
            warmup_steps=int(args.warmup_steps),
            min_lr=float(args.min_lr),
        )
        lr = float(lr) * float(lr_mult)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Determine training seq length for this optimizer step (schedule is in optimizer-step space).
        seq_len = _seq_len_for_step(opt_step - 1, default_seq_len=base_seq_len, schedule=seq_schedule)
        seq_len = int(min(max(2, seq_len), int(cfg.block_size)))

        # Determine batch/accum for this step.
        micro_bs, grad_accum = micro_bs_default, grad_accum_default
        if batch_by_seq is not None:
            micro_bs, grad_accum = _batch_for_seq(seq_len, batch_by_seq, default_bs=micro_bs, default_ga=grad_accum)
        if batch_schedule is not None:
            micro_bs, grad_accum = _batch_for_step(opt_step - 1, batch_schedule, default_bs=micro_bs, default_ga=grad_accum)

        # Desired global batch size for this step (best-effort constant even if we shrink micro_bs).
        gbs_target = int(micro_bs) * int(grad_accum)

        # Apply any learned caps from prior OOMs (caps at smaller seq_len also apply to larger seq_len).
        try:
            cap: Optional[int] = None
            for s, bs_cap in auto_bs_cap_by_seq.items():
                if int(s) <= int(seq_len):
                    cap = int(bs_cap) if cap is None else min(int(cap), int(bs_cap))
            if cap is not None and int(micro_bs) > int(cap):
                micro_bs = int(cap)
                grad_accum = int(math.ceil(float(gbs_target) / max(1.0, float(micro_bs))))
                if int(seq_len) not in warned_auto_caps:
                    print(f"[autobatch] applying learned cap @ seq={seq_len}: bs<={cap} (gbs_target={gbs_target})")
                    warned_auto_caps.add(int(seq_len))
        except Exception:
            pass

        # MPSGraph logits-size guard: automatically reduce micro-batch if (B*T*V) would exceed INT_MAX.
        if device.type == "mps":
            try:
                max_elems = 2_147_483_647  # INT_MAX
                denom = int(seq_len) * int(cfg.vocab_size)
                if denom <= 0:
                    denom = 1
                max_micro = int(max_elems // denom)
                if max_micro < 1:
                    raise ValueError(
                        f"MPSGraph limitation hit: T*vocab_size={denom} > INT_MAX. "
                        f"Reduce --train-seq-len/--block or use a smaller vocab."
                    )
                if int(micro_bs) > int(max_micro):
                    micro_bs = int(max_micro)
                    grad_accum = int(math.ceil(float(gbs_target) / max(1.0, float(micro_bs))))
                    if int(seq_len) not in warned_mps_logits:
                        print(
                            f"[autobatch] reducing batch_size on MPS to satisfy INT_MAX logits: "
                            f"bs<={max_micro} (seq={seq_len} vocab={cfg.vocab_size}, gbs_target={gbs_target})"
                        )
                        warned_mps_logits.add(int(seq_len))
            except Exception:
                pass

        # (batch_size, grad_accum) may be auto-adjusted on the fly on OOM.
        micro_bs_try = int(micro_bs)
        grad_accum_try = int(grad_accum)

        dt_step = 0.0
        fwd_step = 0.0
        bwd_step = 0.0
        opt_step_t = 0.0
        tok_step = 0
        loss_sum_t = torch.zeros([], device=device, dtype=torch.float32)

        while True:
            t_step0 = time.perf_counter()
            loss_sum_t = torch.zeros([], device=device, dtype=torch.float32)
            nonfinite = False
            nonfinite_detail: Optional[str] = None
            tok_step = 0
            fwd_step = 0.0
            bwd_step = 0.0
            opt_step_t = 0.0

            opt.zero_grad(set_to_none=True)

            try:
                # Backward over grad_accum microbatches.
                for _micro in range(grad_accum_try):
                    xb, yb = get_batch_any(train_view, batch_size=micro_bs_try, block_size=seq_len, device=device)
                    tok_step += int(xb.numel())

                    # Fail fast on corrupt token IDs. On GPU backends this can otherwise produce undefined behavior
                    # (instead of a clean "index out of range" error) and quickly lead to NaNs.
                    try:
                        x_oob = int(((xb < 0) | (xb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                        y_oob = int(((yb < 0) | (yb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                    except Exception:
                        x_oob = 0
                        y_oob = 0
                    if x_oob or y_oob:
                        try:
                            x_min = int(xb.detach().min().to("cpu").item())
                            x_max = int(xb.detach().max().to("cpu").item())
                        except Exception:
                            x_min = x_max = -1
                        try:
                            y_min = int(yb.detach().min().to("cpu").item())
                            y_max = int(yb.detach().max().to("cpu").item())
                        except Exception:
                            y_min = y_max = -1

                        dump_path = None
                        try:
                            os.makedirs(str(args.out_dir), exist_ok=True)
                            dump_path = os.path.join(str(args.out_dir), f"oob_tokens_step{opt_step}_micro{_micro}.pt")
                            torch.save(
                                {
                                    "opt_step": int(opt_step),
                                    "micro": int(_micro),
                                    "seq_len": int(seq_len),
                                    "batch_size": int(micro_bs_try),
                                    "grad_accum": int(grad_accum_try),
                                    "vocab_size": int(cfg.vocab_size),
                                    "x_oob": int(x_oob),
                                    "y_oob": int(y_oob),
                                    "xb": xb.detach().to("cpu"),
                                    "yb": yb.detach().to("cpu"),
                                },
                                dump_path,
                            )
                        except Exception:
                            dump_path = None

                        raise RuntimeError(
                            f"Out-of-range token ids detected at optimizer step {opt_step} micro={_micro+1}/{grad_accum_try}: "
                            f"xb_oob={x_oob} yb_oob={y_oob} xb[min,max]=[{x_min},{x_max}] yb[min,max]=[{y_min},{y_max}]"
                            + (f" dump={dump_path}" if dump_path else "")
                            + ". This usually means --vocab-size is wrong or the batch got corrupted (e.g. torch.compile on MPS). "
                              "Try --no-compile (and/or --no-amp) to isolate."
                        )

                    t1 = time.perf_counter()
                    with autocast_ctx():
                        logits, _ = model(xb)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                        loss_to_back = loss / grad_accum_try

                    # Accumulate loss without synchronizing to CPU (important for GPU throughput).
                    try:
                        loss_sum_t = loss_sum_t + loss.detach().float()
                    except Exception:
                        pass
                    fwd_step += time.perf_counter() - t1

                    if not torch.isfinite(loss.detach()).all():
                        nonfinite = True
                        # Ensure logs for this step reflect non-finite state even if we break early.
                        try:
                            loss_sum_t = torch.full_like(loss_sum_t, float("nan"))
                        except Exception:
                            pass

                        # Capture debug info for the failing microbatch (do NOT materially affect normal fast-path).
                        try:
                            loss_val = float(loss.detach().to("cpu").item())
                        except Exception:
                            loss_val = float("nan")
                        try:
                            x_min = int(xb.detach().min().to("cpu").item())
                            x_max = int(xb.detach().max().to("cpu").item())
                        except Exception:
                            x_min = x_max = -1
                        try:
                            y_min = int(yb.detach().min().to("cpu").item())
                            y_max = int(yb.detach().max().to("cpu").item())
                        except Exception:
                            y_min = y_max = -1
                        try:
                            x_oob = int(((xb < 0) | (xb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                            y_oob = int(((yb < 0) | (yb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                        except Exception:
                            x_oob = -1
                            y_oob = -1
                        try:
                            lg = logits.detach()
                            lg_any_nan = bool(torch.isnan(lg).any().detach().to("cpu").item())
                            lg_any_inf = bool(torch.isinf(lg).any().detach().to("cpu").item())
                        except Exception:
                            lg_any_nan = False
                            lg_any_inf = False
                        try:
                            lg_last = logits.detach()[0, -1, :].float()
                            lg_last_nan = bool(torch.isnan(lg_last).any().to("cpu").item())
                            lg_last_inf = bool(torch.isinf(lg_last).any().to("cpu").item())
                            lg_last_max_abs = float(
                                torch.nan_to_num(lg_last, nan=0.0, posinf=0.0, neginf=0.0).abs().max().to("cpu").item()
                            )
                        except Exception:
                            lg_last_nan = False
                            lg_last_inf = False
                            lg_last_max_abs = float("nan")

                        dump_path = None
                        try:
                            os.makedirs(str(args.out_dir), exist_ok=True)
                            dump_path = os.path.join(str(args.out_dir), f"nonfinite_loss_step{opt_step}_micro{_micro}.pt")
                            torch.save(
                                {
                                    "opt_step": int(opt_step),
                                    "micro": int(_micro),
                                    "seq_len": int(seq_len),
                                    "batch_size": int(micro_bs_try),
                                    "grad_accum": int(grad_accum_try),
                                    "vocab_size": int(cfg.vocab_size),
                                    "xb": xb.detach().to("cpu"),
                                    "yb": yb.detach().to("cpu"),
                                },
                                dump_path,
                            )
                        except Exception:
                            dump_path = None

                        nonfinite_detail = (
                            f"micro={_micro+1}/{grad_accum_try} loss={loss_val} "
                            f"xb[min,max]=[{x_min},{x_max}] yb[min,max]=[{y_min},{y_max}] xb_oob={x_oob} yb_oob={y_oob} "
                            f"logits_nan={lg_any_nan} logits_inf={lg_any_inf} "
                            f"logits_last_nan={lg_last_nan} logits_last_inf={lg_last_inf} logits_last_max|x|~{lg_last_max_abs:.3g}"
                            + (f" dump={dump_path}" if dump_path else "")
                        )
                        break

                    t2 = time.perf_counter()
                    loss_to_back.backward()
                    bwd_step += time.perf_counter() - t2

                t3 = time.perf_counter()
                if nonfinite:
                    # Clear grads to avoid contaminating future steps.
                    opt.zero_grad(set_to_none=True)
                    if nan_policy == "reduce_lr":
                        lr_mult = max(1e-6, float(lr_mult) * float(nan_lr_decay))
                        print(f"[warn] non-finite loss detected @ step {opt_step}; skipping step and reducing lr_mult -> {lr_mult:.3g}")
                        if nonfinite_detail:
                            print(f"[warn] non-finite loss detail: {nonfinite_detail}")
                    elif nan_policy == "skip":
                        print(f"[warn] non-finite loss detected @ step {opt_step}; skipping optimizer step")
                        if nonfinite_detail:
                            print(f"[warn] non-finite loss detail: {nonfinite_detail}")
                    else:
                        raise RuntimeError(
                            f"Non-finite loss detected at optimizer step {opt_step}. "
                            f"Try: --nan-policy reduce_lr, --no-learned-temp, --no-decoupled-gate, --no-compile, or --no-amp."
                            + (f" Detail: {nonfinite_detail}" if nonfinite_detail else "")
                        )
                else:
                    grad_clip = float(getattr(args, "grad_clip", 0.0) or 0.0)
                    if grad_clip > 0:
                        # On MPS + bf16 params, PyTorch's clip_grad_norm_ can produce a non-finite norm
                        # due to bf16 overflow during the norm reduction even when grads are finite.
                        if device.type == "mps":
                            gn = _clip_grad_norm_fp32(model.parameters(), grad_clip)
                        else:
                            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        if not torch.isfinite(gn.detach()).all():
                            nonfinite = True
                    if nonfinite:
                        # IMPORTANT: capture debug info BEFORE clearing grads.
                        extra = _first_nonfinite_grad(model)
                        opt.zero_grad(set_to_none=True)
                        if nan_policy == "reduce_lr":
                            lr_mult = max(1e-6, float(lr_mult) * float(nan_lr_decay))
                            print(f"[warn] non-finite grads detected @ step {opt_step}; skipping step and reducing lr_mult -> {lr_mult:.3g}")
                        elif nan_policy == "skip":
                            print(f"[warn] non-finite grads detected @ step {opt_step}; skipping optimizer step")
                        else:
                            raise RuntimeError(
                                f"Non-finite gradients detected at optimizer step {opt_step}. "
                                f"Try: --nan-policy reduce_lr, --no-learned-temp, --no-decoupled-gate, --grad-clip 0 (to isolate), "
                                f"--no-compile, or --no-amp."
                                + (f" First bad grad: {extra}" if extra else "")
                            )
                    else:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        opt_step_t = time.perf_counter() - t3

                if bool(getattr(args, "sync_timing", False)):
                    device_synchronize(device)
                dt_step = time.perf_counter() - t_step0
                break

            except RuntimeError as e:
                if _is_oom_error(e):
                    if micro_bs_try <= 1:
                        raise RuntimeError(
                            f"Out of memory at optimizer step {opt_step} even with batch_size=1 (seq_len={seq_len}). "
                            "Reduce model size, reduce --train-seq-len/--block, or enable --grad-checkpoint/--amp."
                        ) from e
                    new_bs = max(1, int(micro_bs_try // 2))
                    prev = int(auto_bs_cap_by_seq.get(int(seq_len), 1 << 30))
                    auto_bs_cap_by_seq[int(seq_len)] = min(prev, int(new_bs))
                    _empty_device_cache(device)
                    micro_bs_try = int(new_bs)
                    grad_accum_try = int(math.ceil(float(gbs_target) / max(1.0, float(micro_bs_try))))
                    print(
                        f"[autobatch] OOM @ step {opt_step} seq={seq_len}; retrying with "
                        f"batch_size={micro_bs_try} grad_accum={grad_accum_try} (gbs_target={gbs_target})"
                    )
                    continue
                raise

        # Update step-level accumulators using the *successful* attempt only.
        tok_count += int(tok_step)
        tok_acc += int(tok_step)
        fwd_acc += float(fwd_step)
        bwd_acc += float(bwd_step)
        opt_acc += float(opt_step_t)
        dt_acc += float(dt_step)
        steps_in_acc += 1

        # Use the actual micro-batch/accum used for this step when logging.
        micro_bs = int(micro_bs_try)
        grad_accum = int(grad_accum_try)

        # Logging (optimizer steps)
        if int(args.log_every) > 0 and (opt_step % int(args.log_every) == 0 or opt_step == 1):
            tok_s = float(tok_acc / max(dt_acc, 1e-9))
            try:
                loss_avg = float((loss_sum_t / max(1, grad_accum)).detach().to("cpu").item())
            except Exception:
                loss_avg = float("nan")
            ppl = float(math.exp(loss_avg)) if loss_avg < 20 else float("inf")
            ev = {
                "type": "train",
                "step": int(opt_step),
                "loss": float(loss_avg),
                "ppl": float(ppl),
                "lr": float(lr),
                "tok_s": float(tok_s),
                "seq_len": int(seq_len),
                "gbs": int(micro_bs * grad_accum),
                "ms_step": float(1000.0 * dt_acc / max(1, steps_in_acc)),
                "ms_fwd": float(1000.0 * fwd_acc / max(1, steps_in_acc)),
                "ms_bwd": float(1000.0 * bwd_acc / max(1, steps_in_acc)),
                "ms_opt": float(1000.0 * opt_acc / max(1, steps_in_acc)),
                **get_device_mem_stats(device),
            }
            if logger is not None:
                logger.log(ev)
            print(
                f"step {opt_step}/{total_opt_steps} loss={ev['loss']:.4f} ppl={ev['ppl']:.2f} "
                f"lr={lr:.3g} tok/s={tok_s:.0f} seq={seq_len} gbs={ev['gbs']} "
                f"ms/step={ev['ms_step']:.0f}"
            )
            # reset interval accumulators
            dt_acc = 0.0
            tok_acc = 0
            fwd_acc = 0.0
            bwd_acc = 0.0
            opt_acc = 0.0
            steps_in_acc = 0

        if int(args.eval_every) > 0 and (opt_step % int(args.eval_every) == 0 or opt_step == total_opt_steps):
            tr_loss, va_loss = estimate_loss(int(args.eval_iters))
            ev = {"type": "eval", "step": opt_step, "train_loss": tr_loss, "val_loss": va_loss}
            if logger is not None:
                logger.log(ev)
            print(f"[eval] step {opt_step}: train={tr_loss:.4f} val={va_loss:.4f}")
            if va_loss < best_val:
                best_val = va_loss
                torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(str(args.out_dir), "best.pt"))

        if int(args.save_every) > 0 and (opt_step % int(args.save_every) == 0):
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(str(args.out_dir), "last.pt"))

    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(str(args.out_dir), "last.pt"))
    if logger is not None:
        logger.finalize(best_val=best_val if best_val < float("inf") else float("nan"), last_step=last_step)
        logger.close()
