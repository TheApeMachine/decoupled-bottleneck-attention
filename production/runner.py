from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


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
                layerwise_cache=bool(getattr(args, "self_opt_layerwise_cache", False)),
            )

        logger = None
        if args.instrument != "off" or args.live_plot or args.tb:
            logger = RunLogger(
                args.out_dir,
                instrument=args.instrument,
                cfg=cfg,
                args=args,
                device=device,
                live_plot=bool(args.live_plot),
                tb=bool(args.tb),
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

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=str(args.compile_mode))
            print(f"torch.compile enabled (mode={args.compile_mode}).")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")

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
    logger = RunLogger(
        str(args.out_dir),
        instrument=str(args.instrument),
        cfg=cfg,
        args=args,
        device=device,
        live_plot=bool(args.live_plot),
        tb=bool(args.tb),
    ) if str(args.instrument) != "off" else None

    best_val = float("inf")
    last_step = 0
    t_start = time.time()
    tok_count = 0

    def estimate_loss(eval_iters: int) -> Tuple[float, float]:
        model.eval()
        losses_tr: List[float] = []
        losses_va: List[float] = []
        for _ in range(int(eval_iters)):
            xb, yb = get_batch_any(train_view, batch_size=int(args.batch_size), block_size=int(args.block), device=device)
            with autocast_ctx():
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses_tr.append(float(loss.detach().to("cpu").item()))
            xb, yb = get_batch_any(val_view, batch_size=int(args.batch_size), block_size=int(args.block), device=device)
            with autocast_ctx():
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses_va.append(float(loss.detach().to("cpu").item()))
        model.train()
        return float(sum(losses_tr) / max(1, len(losses_tr))), float(sum(losses_va) / max(1, len(losses_va)))

    model.train()
    opt.zero_grad(set_to_none=True)
    for step in range(int(args.steps)):
        last_step = step + 1

        # LR schedule
        lr = lr_for_step(
            step,
            base_lr=float(args.lr),
            total_steps=int(args.steps),
            schedule=str(args.lr_schedule),
            warmup_steps=int(args.warmup_steps),
            min_lr=float(args.min_lr),
        )
        for pg in opt.param_groups:
            pg["lr"] = lr

        xb, yb = get_batch_any(train_view, batch_size=int(args.batch_size), block_size=int(args.block), device=device)
        tok_count += int(xb.numel())

        with autocast_ctx():
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss = loss / max(1, int(args.grad_accum))

        loss.backward()

        if (step + 1) % int(args.grad_accum) == 0:
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            opt.zero_grad(set_to_none=True)

        if (step + 1) % int(args.log_every) == 0:
            dt = time.time() - t_start
            tok_s = (tok_count / dt) if dt > 0 else 0.0
            ppl = float(math.exp(float(loss.detach().to("cpu").item() * max(1, int(args.grad_accum))))) if float(loss.detach().to("cpu").item()) < 20 else float("inf")
            ev = {"type": "train", "step": step + 1, "loss": float(loss.detach().to("cpu").item() * max(1, int(args.grad_accum))), "ppl": ppl, "lr": lr, "tok_s": tok_s}
            if logger is not None:
                logger.log(ev)
            print(f"step {step+1}/{args.steps} loss={ev['loss']:.4f} ppl={ppl:.2f} lr={lr:.3g} tok/s={tok_s:.0f}")

        if int(args.eval_every) > 0 and (step + 1) % int(args.eval_every) == 0:
            tr_loss, va_loss = estimate_loss(int(args.eval_iters))
            ev = {"type": "eval", "step": step + 1, "train_loss": tr_loss, "val_loss": va_loss}
            if logger is not None:
                logger.log(ev)
            print(f"[eval] step {step+1}: train={tr_loss:.4f} val={va_loss:.4f}")
            if va_loss < best_val:
                best_val = va_loss
                torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(str(args.out_dir), "best.pt"))

        if int(args.save_every) > 0 and (step + 1) % int(args.save_every) == 0:
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(str(args.out_dir), "last.pt"))

    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(str(args.out_dir), "last.pt"))
    if logger is not None:
        logger.finalize(best_val=best_val if best_val < float("inf") else float("nan"), last_step=last_step)
        logger.close()


