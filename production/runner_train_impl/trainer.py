"""
Training runner.

The training loop is inherently multi-stage (data → model → selfopt plan → optimize → eval/checkpoint).
A class keeps shared state explicit and enables carving the runner into small methods.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Protocol, cast

import torch
import torch.nn.functional as F

from production.data import TokenView
from production.run_config import TrainConfig
from production.selfopt_logging import SelfOptLogger

from production.runner_train_impl.amp import autocast_ctx, make_grad_scaler
from production.runner_train_impl.checkpoint import load_checkpoint, save_checkpoint
from production.runner_train_impl.data import load_dataset
from production.runner_train_impl.eval import estimate_loss
from production.runner_train_impl.mode import mode_from_str
from production.runner_train_impl.optim import build_optimizer
from production.runner_train_impl.schedule import lr_for_step
from production.runner_train_impl.summary import print_summary
from production.selfopt_controller import RuntimePlan, SelfOptController
import production.data as data_mod
from production.model import GPT, ModelConfig


class _TrainModel(Protocol):
    """Minimal callable model interface for typed training/eval loops."""

    def __call__(self, idx: torch.Tensor) -> tuple[torch.Tensor, object]: ...


class _BackwardTensor(Protocol):
    """Typed `backward()` surface so torch stubs don't leak Unknown into strict checks."""

    def backward(self) -> None: ...


class Trainer:
    """Orchestrates a single training run."""

    def __init__(self, *, args: argparse.Namespace, device: torch.device, self_opt: object | None) -> None:
        self.args: argparse.Namespace = args
        self.device: torch.device = device
        self.self_opt: object | None = self_opt
        self.run_cfg: TrainConfig = TrainConfig.from_args(args)

    def run(self) -> None:
        """Why: keep the public `run_train` function tiny while preserving behavior."""
        if not self.run_cfg.data:
            raise ValueError("--data is required for --mode train")
        if not self.run_cfg.out_dir:
            raise ValueError("--out-dir is required for --mode train (or provide --size + --exp for auto dirs).")

        os.makedirs(str(self.run_cfg.out_dir), exist_ok=True)

        slog = SelfOptLogger(
            jsonl_path=os.path.join(str(self.run_cfg.out_dir), "events.jsonl"),
            run_logger=None,
            echo=True,
        )

        # For GPT-2 BPE streams, vocab is known and doesn't require scanning.
        if self.run_cfg.vocab_size is None and str(self.run_cfg.tokenizer) == "tiktoken":
            try:
                self.args.vocab_size = 50257
            except (AttributeError, TypeError):
                pass
            self.run_cfg = TrainConfig.from_args(self.args)

        data_state = load_dataset(self.run_cfg)
        train_view = cast(TokenView, data_state.train_view)
        val_view = cast(TokenView, data_state.val_view)
        cfg = self._build_model_cfg(vocab_size=data_state.vocab_size)

        self._write_resolved_config(cfg)
        print_summary(args=self.args, device=self.device, cfg=cfg, n_total_tokens=int(data_state.n_total_tokens))

        # Config-only fast path (used by harness validation).
        if int(self.run_cfg.steps) == 0:
            return

        model = self._build_model(cfg)

        model, runtime_plan = self._plan_runtime(
            model=model,
            cfg=cfg,
            train_view=train_view,
            val_view=val_view,
            slog=slog,
        )

        bs0, ga0 = runtime_plan.batch_plan.by_seq.get(
            int(runtime_plan.train_seq_len), next(iter(runtime_plan.batch_plan.by_seq.values()))
        )

        # Why: if steps are AUTO (<0), train for ~1 epoch of the train split.
        total_steps = int(self.run_cfg.steps)
        if total_steps < 0:
            tok_per_step = int(bs0) * int(ga0) * int(runtime_plan.train_seq_len)
            tok_per_step = max(1, tok_per_step)
            total_steps = int(math.ceil(float(len(train_view)) / float(tok_per_step)))
            total_steps = max(1, total_steps)

        amp_enabled = bool(runtime_plan.amp_enabled)
        amp_dtype = runtime_plan.amp_dtype

        scaler = None
        if amp_enabled and self.device.type == "cuda" and amp_dtype == torch.float16:
            scaler = make_grad_scaler(enabled=True)

        opt = build_optimizer(
            name=str(self.run_cfg.optimizer),
            params=model.parameters(),
            lr=float(self.run_cfg.lr),
            weight_decay=float(self.run_cfg.weight_decay),
            adam_betas=str(self.run_cfg.adam_betas),
            adam_eps=float(self.run_cfg.adam_eps),
            lion_betas=str(self.run_cfg.lion_betas),
            foreach=bool(self.run_cfg.opt_foreach),
            fused=bool(self.run_cfg.opt_fused),
        )

        start_step = 0
        if bool(self.run_cfg.resume) and self.run_cfg.resume_path:
            ck = load_checkpoint(str(self.run_cfg.resume_path))
            if ck is not None:
                try:
                    _ = model.load_state_dict(ck.model_state, strict=False)
                    opt.load_state_dict(ck.optim_state)
                    start_step = int(max(0, ck.opt_step))
                except (RuntimeError, ValueError, TypeError):
                    start_step = 0

        def get_train(bs: int, sl: int) -> tuple[torch.Tensor, torch.Tensor]:
            return data_mod.get_batch_any(
                train_view, batch_size=int(bs), block_size=int(sl), device=self.device
            )

        def get_val(bs: int, sl: int) -> tuple[torch.Tensor, torch.Tensor]:
            return data_mod.get_batch_any(
                val_view, batch_size=int(bs), block_size=int(sl), device=self.device
            )

        # Main loop
        best_val = float("inf")
        last_eval = -1
        last_save = -1

        for step in range(int(start_step), int(total_steps)):
            lr = lr_for_step(
                step,
                base_lr=float(self.run_cfg.lr),
                total_steps=int(total_steps),
                schedule=str(self.run_cfg.lr_schedule),
                warmup_steps=int(self.run_cfg.warmup_steps),
                min_lr=float(self.run_cfg.min_lr),
            )
            for pg_any in opt.param_groups:
                pg = cast(dict[str, object], pg_any)
                pg["lr"] = float(lr)

            _ = model.train()
            opt.zero_grad(set_to_none=True)

            loss_sum = 0.0
            t0 = time.perf_counter()
            for _micro in range(int(ga0)):
                xb, yb = get_train(int(bs0), int(runtime_plan.train_seq_len))
                with autocast_ctx(self.device, enabled=amp_enabled, dtype=amp_dtype):
                    logits, _cache = cast(_TrainModel, model)(xb)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                    loss_to_back = loss / float(max(1, int(ga0)))
                loss_sum += float(loss.detach().float().cpu().item())
                if scaler is not None:
                    scaled = scaler.scale(loss_to_back)
                    cast(_BackwardTensor, scaled).backward()
                else:
                    cast(_BackwardTensor, loss_to_back).backward()

            # Grad clipping (best-effort; unscale if scaler supports it).
            if float(self.run_cfg.grad_clip) > 0.0:
                if scaler is not None:
                    unscale = getattr(cast(object, scaler), "unscale_", None)
                    if callable(unscale):
                        try:
                            _ = unscale(opt)
                        except (RuntimeError, ValueError, TypeError):
                            pass
                try:
                    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.run_cfg.grad_clip))
                except (RuntimeError, ValueError, TypeError):
                    pass

            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            dt = time.perf_counter() - t0

            if int(self.run_cfg.log_every) > 0 and (step % int(self.run_cfg.log_every) == 0):
                try:
                    print(f"[train] step={step} loss={loss_sum/float(max(1,int(ga0))):.4f} lr={lr:.3g} dt={dt:.3f}s", flush=True)
                except (OSError, UnicodeEncodeError):
                    pass

            do_eval = int(self.run_cfg.eval_every) > 0 and (step % int(self.run_cfg.eval_every) == 0) and step != last_eval
            do_save = int(self.run_cfg.save_every) > 0 and (step % int(self.run_cfg.save_every) == 0) and step != last_save
            is_last = step + 1 >= int(total_steps)

            if do_eval or is_last:
                last_eval = step
                tr_loss, va_loss = estimate_loss(
                    model=cast(_TrainModel, model),
                    get_batch_train=get_train,
                    get_batch_val=get_val,
                    eval_iters=int(self.run_cfg.eval_iters),
                    batch_size=int(bs0),
                    seq_len=int(runtime_plan.eval_seq_len),
                    autocast_ctx=(lambda: autocast_ctx(self.device, enabled=amp_enabled, dtype=amp_dtype)),
                )
                best_val = min(best_val, float(va_loss))
                try:
                    print(f"[eval] step={step} train={tr_loss:.4f} val={va_loss:.4f} best={best_val:.4f}", flush=True)
                except (OSError, UnicodeEncodeError):
                    pass

            if do_save or is_last:
                last_save = step
                _ = save_checkpoint(
                    out_dir=str(self.run_cfg.out_dir),
                    opt_step=int(step),
                    model=model,
                    optimizer=opt,
                    cfg=cfg,
                    extra={"best_val": float(best_val)},
                )

    def _build_model(self, cfg: ModelConfig) -> torch.nn.Module:
        """Why: isolate model construction so dtype/compile planning can wrap it cleanly."""
        return GPT(cfg).to(self.device)

    def _build_model_cfg(self, *, vocab_size: int) -> ModelConfig:
        """Why: translate TrainConfig fields into the model's config object."""

        cfg = ModelConfig(device=self.device)
        cfg.vocab_size = int(vocab_size)
        cfg.block_size = int(self.run_cfg.block)
        cfg.n_layer = int(self.run_cfg.layers)
        cfg.n_head = int(self.run_cfg.n_head)
        cfg.kv_head = self.run_cfg.kv_head
        cfg.d_model = int(self.run_cfg.d_model)
        cfg.d_ff = int(self.run_cfg.d_ff)
        cfg.embed_dim = int(self.run_cfg.embed_dim)

        cfg.attn_mode = mode_from_str(self.run_cfg.attn_mode)
        cfg.attn_dim = int(self.run_cfg.attn_dim) if int(self.run_cfg.attn_dim) > 0 else int(cfg.d_model)
        cfg.head_dim = int(cfg.attn_dim // max(1, cfg.n_head))
        cfg.sem_dim = int(self.run_cfg.sem_dim)
        cfg.geo_dim = int(self.run_cfg.geo_dim)

        cfg.decoupled_gate = (not bool(self.run_cfg.no_decoupled_gate))
        cfg.rope = (not bool(self.run_cfg.no_rope))
        cfg.rope_base = float(self.run_cfg.rope_base)
        cfg.tie_qk = bool(self.run_cfg.tie_qk)
        cfg.null_attn = bool(self.run_cfg.null_attn)
        cfg.learned_temp = (not bool(self.run_cfg.no_learned_temp))

        mlp = str(self.run_cfg.mlp or "swiglu").strip().lower()
        cfg.mlp = "gelu" if mlp == "gelu" else "swiglu"
        cfg.dropout = float(self.run_cfg.dropout)
        return cfg

    def _write_resolved_config(self, cfg: ModelConfig) -> None:
        """Why: emit a machine-readable record of what configuration was used."""
        try:
            path = os.path.join(str(self.run_cfg.out_dir), "resolved_config.json")
            raw_cfg_obj = cast(object, getattr(cfg, "__dict__", {}))
            cfg_dict: dict[str, object] = {}
            if isinstance(raw_cfg_obj, dict):
                raw_cfg_map = cast(dict[object, object], cast(object, raw_cfg_obj))
                for k_obj, v_obj in raw_cfg_map.items():
                    cfg_dict[str(k_obj)] = v_obj
            # Make enums JSON-friendly.
            if "attn_mode" in cfg_dict and hasattr(cfg_dict["attn_mode"], "value"):
                cfg_dict["attn_mode"] = getattr(cfg_dict["attn_mode"], "value")
            payload = {
                "train": dict(getattr(self.run_cfg, "__dict__", {})),
                "model": cfg_dict,
            }
            os.makedirs(str(self.run_cfg.out_dir), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                _ = f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except (OSError, ValueError, TypeError):
            pass

    def _plan_runtime(
        self,
        *,
        model: torch.nn.Module,
        cfg: ModelConfig,
        train_view: TokenView,
        val_view: TokenView,
        slog: SelfOptLogger,
    ) -> tuple[torch.nn.Module, RuntimePlan]:
        """
        Delegate runtime planning (dtype/AMP/batch/compile) to the always-on self-optimizer.
        """
        train_seq_cap = int(max(2, int(len(train_view)) - 2))
        eval_seq_cap = int(max(2, int(len(val_view)) - 2))

        log_path = None
        try:
            log_path = os.path.join(str(self.run_cfg.out_dir), "selfopt_decisions.jsonl")
        except (TypeError, AttributeError):
            log_path = None

        controller = SelfOptController(
            cache_path=getattr(self.self_opt, "cache_path", None) if self.self_opt is not None else None,
            log_path=log_path,
            device=self.device,
            cfg=cfg,
            logger=slog,
        )

        model2, runtime_plan = controller.plan_runtime(
            model=model,
            train_view=train_view,
            val_view=val_view,
            get_batch=lambda bs, sl: data_mod.get_batch_any(
                train_view,
                batch_size=int(bs),
                block_size=int(sl),
                device=self.device
            ),
            train_seq_len_cap=train_seq_cap,
            eval_seq_len_cap=eval_seq_cap,
        )
        return model2, runtime_plan


