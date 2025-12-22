"""
Training runner.

The training loop is inherently multi-stage
(data → model → selfopt plan → optimize → eval/checkpoint).
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
from production.console import get_console
from production.data import TokenView
from production.model import GPT, ModelConfig
from production.optimizer.counts import CountCodec
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


class _TrainModel(Protocol):
    """Minimal callable model interface for typed training/eval loops."""

    def __call__(self, idx: torch.Tensor) -> tuple[torch.Tensor, object]: ...


class _BackwardTensor(Protocol):
    """Typed `backward()` surface so torch stubs don't leak Unknown into strict checks."""

    def backward(self) -> None: ...


class _CudaGraphStepMarker(Protocol):
    """Typed `cudagraph_mark_step_begin()` callable surface (CUDA-only, optional)."""

    def __call__(self) -> None: ...




class Trainer:
    """Orchestrates a single training run."""

    def __init__(self, *, args: argparse.Namespace, device: torch.device, self_opt: object | None) -> None:
        self.args: argparse.Namespace = args
        self.device: torch.device = device
        self.self_opt: object | None = self_opt
        self.run_cfg: TrainConfig = TrainConfig.from_args(args)

    def run(self) -> None:
        """Why: keep the public `run_train` function tiny while preserving behavior."""
        console = get_console()
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

        with console.status("Loading dataset…"):
            data_state = load_dataset(self.run_cfg)

        train_view = cast(TokenView, data_state.train_view)
        val_view = cast(TokenView, data_state.val_view)
        cfg = self._build_model_cfg(vocab_size=data_state.vocab_size, n_total_tokens=data_state.n_total_tokens)

        self._write_resolved_config(cfg)

        # Config-only fast path (used by harness validation).
        if int(self.run_cfg.steps) == 0:
            slog.close()
            return

        # Initialize structured run logging for real training runs only (avoid
        # creating W&B runs during harness config-validation `--steps 0`).
        from production.instrumentation import RunLogger

        if (
            str(self.run_cfg.instrument) != "off"
            or bool(self.run_cfg.live_plot)
            or bool(self.run_cfg.tb)
            or bool(self.run_cfg.wandb)
        ):
            slog.run_logger = RunLogger(
                str(self.run_cfg.out_dir),
                instrument=str(self.run_cfg.instrument),
                cfg=cfg,
                args=self.args,
                device=self.device,
                live_plot=bool(self.run_cfg.live_plot),
                tb=bool(self.run_cfg.tb),
                wandb=bool(self.run_cfg.wandb),
            )

        with console.status("Building model…"):
            model = self._build_model(cfg)

        with console.status("Planning runtime (AMP/batch/compile)…"):
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
            try:
                self.args.steps = int(total_steps)
            except (AttributeError, TypeError):
                pass

        amp_enabled = bool(runtime_plan.amp_enabled)
        amp_dtype = runtime_plan.amp_dtype

        # Resolve derived cadence knobs (warmup/save) now that total_steps is known,
        # so the printed summary reflects real runtime behavior.
        warmup_steps = int(self.run_cfg.warmup_steps)
        save_every = int(self.run_cfg.save_every)
        eval_every = int(self.run_cfg.eval_every)

        def _auto_decade(x: int) -> int:
            if x <= 0:
                return 1
            exp = int(max(0, int(math.floor(math.log10(float(x)))) - 1))
            return int(math.pow(10.0, float(exp)))

        if warmup_steps <= 0 and str(self.run_cfg.lr_schedule).strip().lower() == "cosine":
            warmup_steps = int(min(max(1, _auto_decade(int(total_steps))), int(total_steps)))
            try:
                self.args.warmup_steps = int(warmup_steps)
            except (AttributeError, TypeError):
                pass

        eval_recommended = int(_auto_decade(int(total_steps)))
        if eval_every <= 0:
            eval_every = 0

        if save_every <= 0:
            save_every = int(min(max(1, eval_recommended * 10), int(total_steps)))
            try:
                self.args.save_every = int(save_every)
            except (AttributeError, TypeError):
                pass

        # Now that args are fully resolved, print the run summary.
        print_summary(args=self.args, device=self.device, cfg=cfg, n_total_tokens=int(data_state.n_total_tokens))

        try:
            console.print(
                f"[runtime] train_seq_len={int(runtime_plan.train_seq_len)} eval_seq_len={int(runtime_plan.eval_seq_len)} bs={int(bs0)} ga={int(ga0)} amp={int(amp_enabled)} amp_dtype={str(amp_dtype)}"
            )
        except (OSError, ValueError, TypeError):
            pass

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
        if bool(self.run_cfg.resume):
            resume_path = self.run_cfg.resume_path
            if not resume_path:
                try:
                    if self.run_cfg.out_dir:
                        cand = os.path.join(str(self.run_cfg.out_dir), "last.pt")
                        if os.path.isfile(cand):
                            resume_path = cand
                except (OSError, TypeError, ValueError):
                    resume_path = None

            if resume_path:
                ck = load_checkpoint(str(resume_path))
                if ck is not None:
                    try:
                        load_result = model.load_state_dict(ck.model_state, strict=False)
                        missing = cast(list[str], getattr(load_result, "missing_keys", []))
                        unexpected = cast(list[str], getattr(load_result, "unexpected_keys", []))
                        if missing or unexpected:
                            console.print(
                                f"[warn] Resume checkpoint key mismatch - missing: {len(missing)}, unexpected: {len(unexpected)}"
                            )
                        opt.load_state_dict(ck.optim_state)
                        # `opt_step` is "completed steps", so it is the next step index.
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

        # Hot-loop constants & locals (avoid repeated attribute lookups / conversions).
        model_call = cast(_TrainModel, model)
        ce = F.cross_entropy
        inv_ga = 1.0 / float(max(1, int(ga0)))
        bs_i = int(bs0)
        ga_i = int(ga0)
        train_sl = int(runtime_plan.train_seq_len)
        eval_sl = int(runtime_plan.eval_seq_len)
        tok_per_step = int(bs_i) * int(ga_i) * int(train_sl)
        vocab_size = int(cfg.vocab_size)
        grad_clip = float(self.run_cfg.grad_clip)
        log_every = int(self.run_cfg.log_every)
        # NOTE: warmup_steps/eval_every/save_every were resolved earlier (after runtime plan),
        # so the summary and start line match actual behavior.

        # If all periodic hooks are disabled, training can look like it's "hung" because
        # the loop is intentionally silent. Emit a one-time hint to reduce confusion.
        if log_every <= 0 and eval_every <= 0 and save_every <= 0:
            try:
                console.print(
                    "[train] No periodic hooks enabled (log/eval/save are 0). Training is still running; enable periodic logs via internal defaults if desired.",
                    flush=True,
                )
            except (OSError, ValueError, TypeError, UnicodeEncodeError):
                pass

        # Always emit a persistent start line (rich status spinners can be ephemeral).
        try:
            start_msg = (
                f"[train] starting: steps={int(total_steps)} (cfg.steps={int(self.run_cfg.steps)}) "
                + f"start_step={int(start_step)} warmup={int(warmup_steps)} "
                + f"log_every={int(log_every)} eval_every={int(eval_every)} save_every={int(save_every)}"
            )
            if eval_every <= 0:
                start_msg = start_msg + f" (eval recommended ~{int(eval_recommended)})"
            console.print(
                start_msg,
                flush=True,
            )
        except (OSError, ValueError, TypeError, UnicodeEncodeError):
            pass

        _ = model.train()

        # Runtime LR adaptation state
        loss_history: list[torch.Tensor] = []
        lr_multiplier: float = 1.0
        adapt_window: int = 50  # window for loss variance measurement
        adapt_every: int = 20   # check every N steps
        base_lr_config: float = float(self.run_cfg.lr)

        for step in range(int(start_step), int(total_steps)):
            lr = lr_for_step(
                step,
                base_lr=base_lr_config * lr_multiplier,  # Apply adaptive multiplier
                total_steps=int(total_steps),
                schedule=str(self.run_cfg.lr_schedule),
                warmup_steps=int(warmup_steps),
                min_lr=float(self.run_cfg.min_lr),
            )
            for pg_any in opt.param_groups:
                pg = cast(dict[str, object], pg_any)
                pg["lr"] = float(lr)

            opt.zero_grad(set_to_none=True)

            # Keep loss accounting on-device; one host sync per step (when/if we log).
            loss_sum_t = torch.zeros((), device=self.device, dtype=torch.float32)
            t0 = time.perf_counter()
            for _micro in range(int(ga_i)):
                xb, yb = get_train(bs_i, train_sl)
                with autocast_ctx(self.device, enabled=amp_enabled, dtype=amp_dtype):
                    # CUDA graphs + compile: mark step boundaries to avoid overwritten outputs.
                    if self.device.type == "cuda":
                        compiler_obj: object = getattr(torch, "compiler", object())
                        mark_obj: object = getattr(compiler_obj, "cudagraph_mark_step_begin", None)
                        if callable(mark_obj):
                            cast(_CudaGraphStepMarker, mark_obj)()
                    logits, _cache = model_call(xb)
                    # Avoid shape gymnastics in the hot path; logits last dim is vocab.
                    loss = ce(logits.view(-1, vocab_size), yb.view(-1))
                    loss_to_back = loss * float(inv_ga)
                loss_sum_t = loss_sum_t + loss.detach().to(dtype=torch.float32)
                if scaler is not None:
                    scaled = scaler.scale(loss_to_back)
                    cast(_BackwardTensor, scaled).backward()
                else:
                    cast(_BackwardTensor, loss_to_back).backward()

            # Grad clipping (best-effort; unscale if scaler supports it).
            if grad_clip > 0.0:
                if scaler is not None:
                    unscale = getattr(cast(object, scaler), "unscale_", None)
                    if callable(unscale):
                        try:
                            _ = unscale(opt)
                        except (RuntimeError, ValueError, TypeError):
                            pass
                try:
                    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                except (RuntimeError, ValueError, TypeError):
                    pass

            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            dt = time.perf_counter() - t0

            # Track loss for runtime adaptation (on-device, defer sync)
            # Store detached tensor; conversion to float happens only when needed
            loss_history.append((loss_sum_t * inv_ga).detach())
            # Bound memory growth: keep only the recent window plus slack
            loss_history = loss_history[-(adapt_window + 10):]

            # Runtime LR adaptation: adjust based on loss curvature/variance
            # NOTE: Use recorded loss count as the trigger (not `step`) so resume-from-checkpoint
            # doesn't immediately run adaptation with an empty history.
            if len(loss_history) > adapt_window and step % adapt_every == 0:
                # Convert to float only when adaptation logic runs
                recent_losses = [float(t.item()) for t in loss_history[-adapt_window:]]
                if len(recent_losses) >= adapt_window:
                    mean_loss = sum(recent_losses) / len(recent_losses)
                    variance = sum((l - mean_loss) ** 2 for l in recent_losses) / len(recent_losses)
                    rel_variance = variance / max(mean_loss ** 2, 1e-9)

                    # High variance → reduce LR (gradients too noisy)
                    # Low variance + plateau → increase LR slightly (can push harder)
                    # Loss spiking → reduce LR aggressively

                    # Only evaluate deeper multiplier logic once we've accumulated enough *new*
                    # samples post-resume, and ensure the slice is the intended length.
                    if len(loss_history) >= adapt_window + 10:
                        prev_window = loss_history[-(adapt_window + 10) :]
                        if len(prev_window) == adapt_window + 10:
                            prev_losses = [float(t.item()) for t in prev_window[:-10]]
                            prev_mean = sum(prev_losses) / len(prev_losses)
                            improvement = (prev_mean - mean_loss) / max(prev_mean, 1e-9)

                            # Loss plateau (< 0.5% improvement) and low variance → can push harder
                            if improvement < 0.005 and rel_variance < 0.01:
                                lr_multiplier = min(1.5, lr_multiplier * 1.05)
                            # High variance (noisy gradients) → smooth it out
                            elif rel_variance > 0.05:
                                lr_multiplier = max(0.5, lr_multiplier * 0.95)
                            # Loss spike (negative improvement) → back off
                            elif improvement < -0.01:
                                lr_multiplier = max(0.5, lr_multiplier * 0.85)

            if log_every > 0 and (step % log_every == 0):
                # Sync only when we actually need host-visible metrics (printing).
                # Convert tensor to float only when logging
                loss_avg = float(loss_history[-1].item()) if loss_history else 0.0
                tok_s = float(tok_per_step) / float(max(1e-9, dt))
                # Show adaptive LR multiplier when it's active
                lr_info = f"lr={lr:.3g}"
                if abs(lr_multiplier - 1.0) > 0.01:
                    lr_info += f" (×{lr_multiplier:.2f})"
                try:
                    console.print(
                        f"[train] step={step} loss={loss_avg:.4f} {lr_info} tok/s={tok_s:.0f} dt={dt:.3f}s",
                        flush=True,
                    )
                except (OSError, UnicodeEncodeError):
                    pass

                # Structured event (for JSONL/TB/W&B).
                try:
                    slog.log(
                        {
                            "type": "train",
                            "step": int(step),
                            "loss": float(loss_avg),
                            "lr": float(lr),
                            "tok_s": float(tok_s),
                            "seq_len": float(train_sl),
                            "gbs": float(tok_per_step),
                            "ms_step": float(dt * 1000.0),
                        }
                    )
                except (OSError, RuntimeError, ValueError, TypeError):
                    pass

            do_eval = (
                eval_every > 0
                and step > 0
                and (step % eval_every == 0)
                and step != last_eval
            )
            do_save = (
                save_every > 0
                and (step % save_every == 0)
                and step != last_save
            )
            is_last = step + 1 >= int(total_steps)

            if do_eval or is_last:
                last_eval = step
                was_training = bool(getattr(model, "training", False))
                try:
                    _ = model.eval()
                except Exception:
                    was_training = False

                tr_loss, va_loss = estimate_loss(
                    model=cast(_TrainModel, model),
                    get_batch_train=get_train,
                    get_batch_val=get_val,
                    eval_iters=int(self.run_cfg.eval_iters),
                    batch_size=bs_i,
                    seq_len=eval_sl,
                    autocast_ctx=(
                        lambda: autocast_ctx(
                            self.device, enabled=amp_enabled, dtype=amp_dtype
                        )
                    ),
                )

                if was_training:
                    try:
                        _ = model.train()
                    except Exception:
                        pass
                best_val = min(best_val, float(va_loss))
                try:
                    console.print(
                        f"[eval] step={step} train={tr_loss:.4f} val={va_loss:.4f} best={best_val:.4f}",
                        flush=True,
                    )
                except (OSError, UnicodeEncodeError):
                    pass

                # Structured eval event (for JSONL/TB/W&B).
                try:
                    slog.log(
                        {
                            "type": "eval",
                            "step": int(step),
                            "train_loss": float(tr_loss),
                            "val_loss": float(va_loss),
                        }
                    )
                except (OSError, RuntimeError, ValueError, TypeError):
                    pass

            if do_save or is_last:
                last_save = step
                _ = save_checkpoint(
                    out_dir=str(self.run_cfg.out_dir),
                    opt_step=int(step + 1),
                    model=model,
                    optimizer=opt,
                    cfg=cfg,
                    extra={"best_val": float(best_val)},
                )

        try:
            slog.finalize(best_val=float(best_val), last_step=int(total_steps))
        finally:
            slog.close()

    def _build_model(self, cfg: ModelConfig) -> torch.nn.Module:
        """Why: isolate model construction so dtype/compile planning can wrap it cleanly."""
        return GPT(cfg).to(self.device)

    def _compute_target_params(self, *, n_total_tokens: int) -> int:
        """
        Compute target model parameter count.

        Why: unify direct size overrides (--size) and dataset-derived sizing (Chinchilla-style)
        into a single computed value, rather than expecting it as a CLI arg.

        Logic mirrors production/optimizer/apply_impl/target_params.py:TargetParamsDeriver.apply
        """
        # First, check for explicit --size argument
        size_raw: object | None = getattr(self.args, "size", None)
        if size_raw is not None:
            # Parse size string (e.g., "100m", "1b", "1.5b")
            size = CountCodec.parse(size_raw)
            if size is None:
                raise ValueError(f"Unparseable --size {size_raw!r}. Use e.g. 100m, 1b, 1.5b, 2e9.")
            return int(size)

        # Otherwise, derive from dataset tokens using Chinchilla heuristic (tokens ≈ 20 × params)
        if n_total_tokens <= 0:
            raise ValueError(
                "Could not infer dataset token count. "
                + "Name dataset like `fineweb_20b.npy`, provide a sibling `.meta` with `tokens: ...`, "
                + "or pass `--size` explicitly."
            )

        tokens_per_param = 20.0
        target_params = int(max(1.0, float(n_total_tokens) / tokens_per_param))
        return target_params

    def _build_model_cfg(self, *, vocab_size: int, n_total_tokens: int) -> ModelConfig:
        """Why: translate TrainConfig fields into the model's config object."""

        cfg = ModelConfig(device=self.device)
        cfg.vocab_size = int(vocab_size)
        cfg.block_size = int(self.run_cfg.block)
        # Allow depth to be controlled via intent (e.g. encoded in out_dir like `_l22_...`)
        # without adding a CLI flag. If unset (0), ModelConfig.optimize() will choose it.
        try:
            cfg.n_layer = int(self.run_cfg.layers)
        except Exception:
            cfg.n_layer = 0

        # Set mode before optimize() so mode-specific logic can use it
        cfg.attn_mode = mode_from_str(self.run_cfg.attn_mode)
        cfg.rope = (not bool(self.run_cfg.no_rope))

        # Budget-aware auto-sizing: target_params derived from dataset tokens (Chinchilla-style)
        # or explicit --size argument. Compute it here rather than relying on self.args.target_params
        # which may not exist (and should be computed, not CLI-provided).
        target_params: int = self._compute_target_params(n_total_tokens=n_total_tokens)
        cfg.optimize(target_params)

        # Apply experiment-specific overrides (presets/CLI) after auto-sizing
        cfg.kv_head = self.run_cfg.kv_head
        cfg.decoupled_gate = (not bool(self.run_cfg.no_decoupled_gate))
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
        path = os.path.join(str(self.run_cfg.out_dir), "resolved_config.json")

        raw_cfg_obj = cast(object, getattr(cfg, "__dict__", {}))
        cfg_dict: dict[str, object] = {}
        if isinstance(raw_cfg_obj, dict):
            raw_cfg_map = cast(dict[object, object], cast(object, raw_cfg_obj))
            for k_obj, v_obj in raw_cfg_map.items():
                cfg_dict[str(k_obj)] = v_obj

        # Make values JSON-friendly (torch types, enums).
        if "device" in cfg_dict:
            try:
                cfg_dict["device"] = str(cfg_dict["device"])
            except Exception:
                cfg_dict["device"] = "cpu"
        if "attn_mode" in cfg_dict and hasattr(cfg_dict["attn_mode"], "value"):
            cfg_dict["attn_mode"] = getattr(cfg_dict["attn_mode"], "value")

        # Paper harness expects model keys at top-level (attn_mode/rope/tie_qk/null_attn).
        payload: dict[str, object] = dict(cfg_dict)
        payload["train"] = dict(getattr(self.run_cfg, "__dict__", {}))
        payload["model"] = dict(cfg_dict)

        os.makedirs(str(self.run_cfg.out_dir), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                _ = f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except (OSError, ValueError, TypeError) as e:
            # Do not fail training, but do not silently ignore reproducibility artifacts either.
            try:
                console = get_console()
                console.print(f"[warn] Failed to write resolved_config.json: {e}")
            except Exception:
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
