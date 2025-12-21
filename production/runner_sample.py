from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from production.runtime_tuning import KVCachePolicy, KVSelfOptConfig
from production.run_config import SampleConfig
from production.selfopt_logging import SelfOptLogger


def run_sample(*, args: argparse.Namespace, device: torch.device, self_opt: Optional[KVSelfOptConfig]) -> None:
    """Sample/generate from a checkpoint."""
    # Local imports so CLI --help doesn't require torch/tiktoken.
    from production.instrumentation import RunLogger
    from production.model import GPT, ModelConfig

    try:
        import tiktoken  # type: ignore
    except Exception:
        tiktoken = None  # type: ignore

    cfg_run = SampleConfig.from_args(args)

    if not cfg_run.ckpt:
        raise ValueError("--ckpt is required for --mode sample")

    ckpt = torch.load(str(cfg_run.ckpt), map_location=device)
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
        prompt_ids = [int(t) for t in str(cfg_run.prompt_tokens).strip().split()]
    except ValueError:
        if str(cfg_run.tokenizer) != "tiktoken":
            raise ValueError("Text prompts require --tokenizer tiktoken")
        if tiktoken is None:
            raise ImportError("tiktoken needed for text prompts")
        enc = tiktoken.get_encoding("gpt2")
        prompt_ids = enc.encode_ordinary(str(cfg_run.prompt_tokens))

    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    # Expert override: force an atomic decoupled KV cache policy from a single string.
    if cfg_run.kv_policy:
        if str(getattr(cfg, "attn_mode", "")) != "decoupled":
            raise ValueError("--kv-policy is only supported for decoupled attention checkpoints")
        pol = KVCachePolicy.parse(str(cfg_run.kv_policy))
        # Apply as per-tensor overrides (so model.generate() stays unchanged).
        args.kv_cache_k_sem = pol.k_sem_kind
        args.kv_cache_k_geo = pol.k_geo_kind
        args.kv_cache_v = pol.v_kind
        args.kv_qblock_k_sem = int(pol.k_sem_qblock)
        args.kv_qblock_k_geo = int(pol.k_geo_qblock)
        args.kv_qblock_v = int(pol.v_qblock)
        args.kv_residual = int(pol.residual_len)

        # If selfopt is enabled, keep decode-plan tuning but disable cache-policy tuning (policy is forced).
        if self_opt is not None:
            try:
                self_opt.scope = "decode"
            except Exception:
                pass

    logger = None
    if cfg_run.instrument != "off" or bool(cfg_run.live_plot) or bool(cfg_run.tb) or bool(cfg_run.wandb):
        logger = RunLogger(
            cfg_run.out_dir,
            instrument=str(cfg_run.instrument),
            cfg=cfg,
            args=args,
            device=device,
            live_plot=bool(cfg_run.live_plot),
            tb=bool(cfg_run.tb),
            wandb=bool(cfg_run.wandb),
        )
    slog = SelfOptLogger(
        jsonl_path=(os.path.join(str(cfg_run.out_dir), "events.jsonl") if cfg_run.out_dir else None),
        run_logger=logger,
        echo=False,
    )

    print(f"Generating {int(cfg_run.max_new_tokens)} tokens...")
    try:
        if cfg_run.draft_ckpt:
            dckpt = torch.load(str(cfg_run.draft_ckpt), map_location=device)
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
                max_new_tokens=int(cfg_run.max_new_tokens),
                temperature=float(cfg_run.temperature),
                top_k=(None if cfg_run.top_k is None else int(cfg_run.top_k)),
                kv_cache=str(cfg_run.kv_cache),
                kv_qblock=int(cfg_run.kv_qblock),
                kv_residual=int(cfg_run.kv_residual),
                kv_decode_block=int(cfg_run.kv_decode_block),
                kv_fused=str(cfg_run.kv_fused),
                self_opt=self_opt,
                kv_cache_k=cfg_run.kv_cache_k,
                kv_cache_v=cfg_run.kv_cache_v,
                kv_cache_k_sem=cfg_run.kv_cache_k_sem,
                kv_cache_k_geo=cfg_run.kv_cache_k_geo,
                kv_qblock_k=cfg_run.kv_qblock_k,
                kv_qblock_v=cfg_run.kv_qblock_v,
                kv_qblock_k_sem=cfg_run.kv_qblock_k_sem,
                kv_qblock_k_geo=cfg_run.kv_qblock_k_geo,
                spec_k=int(cfg_run.spec_k),
                spec_method=str(cfg_run.spec_method),
                spec_extra_token=bool(cfg_run.spec_extra_token),
                spec_disable_below_accept=float(cfg_run.spec_disable_below_accept),
                log_callback=slog.log,
            )
        else:
            out = model.generate(
                prompt,
                max_new_tokens=int(cfg_run.max_new_tokens),
                temperature=float(cfg_run.temperature),
                top_k=(None if cfg_run.top_k is None else int(cfg_run.top_k)),
                kv_cache=str(cfg_run.kv_cache),
                kv_qblock=int(cfg_run.kv_qblock),
                kv_residual=int(cfg_run.kv_residual),
                kv_decode_block=int(cfg_run.kv_decode_block),
                kv_fused=str(cfg_run.kv_fused),
                self_opt=self_opt,
                kv_cache_k=cfg_run.kv_cache_k,
                kv_cache_v=cfg_run.kv_cache_v,
                kv_cache_k_sem=cfg_run.kv_cache_k_sem,
                kv_cache_k_geo=cfg_run.kv_cache_k_geo,
                kv_qblock_k=cfg_run.kv_qblock_k,
                kv_qblock_v=cfg_run.kv_qblock_v,
                kv_qblock_k_sem=cfg_run.kv_qblock_k_sem,
                kv_qblock_k_geo=cfg_run.kv_qblock_k_geo,
                log_callback=slog.log,
            )
    finally:
        slog.close()

    out_ids = out[0].detach().to("cpu").tolist()
    if str(cfg_run.tokenizer) == "tiktoken":
        if tiktoken is None:
            raise ImportError("tiktoken not installed")
        enc = tiktoken.get_encoding("gpt2")
        print(enc.decode(out_ids))
    else:
        print(out_ids)


