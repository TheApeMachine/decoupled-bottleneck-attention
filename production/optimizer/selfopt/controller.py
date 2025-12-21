"""Always-on runtime planning (no user-exposed knobs)."""

from __future__ import annotations

import time
from dataclasses import asdict
from collections.abc import Callable

import torch

from production.selfopt_cache import set_cache_entry
from production.selfopt_logging import SelfOptLogger, append_jsonl
from production.selfopt_utils import device_sig, hash_cfg
from production.train_tuning import tune_batch_by_seq, tune_torch_compile

from production.optimizer.selfopt.capabilities import choose_amp, choose_param_dtype
from production.optimizer.selfopt.types import RuntimePlan


class SelfOptController:
    """Always-on self-optimizer for runtime policy.

    This controller deliberately exposes *no* config surface: it derives decisions
    from device/model/data.
    """

    def __init__(
        self,
        *,
        cache_path: str | None,
        log_path: str | None,
        device: torch.device,
        cfg: object,
        logger: SelfOptLogger | None = None,
    ) -> None:
        self.cache_path: str | None = str(cache_path) if cache_path else None
        self.log_path: str | None = str(log_path) if log_path else None
        self.device: torch.device = device
        self.cfg: object = cfg
        self.logger: SelfOptLogger | None = logger

    def _log(self, ev: dict[str, object]) -> None:
        """Best-effort structured logging (must never break runtime)."""
        append_jsonl(self.log_path, ev)
        if self.logger is not None:
            try:
                self.logger.log(ev)
            except Exception:
                pass

    def plan_runtime(
        self,
        *,
        model: torch.nn.Module,
        train_view: object,
        val_view: object,
        get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
        train_seq_len_cap: int,
        eval_seq_len_cap: int,
    ) -> tuple[torch.nn.Module, RuntimePlan]:
        """Return (possibly wrapped) model + the chosen runtime plan.

        `train_seq_len_cap` / `eval_seq_len_cap` should already reflect dataset split feasibility.
        """
        _ = (train_view, val_view)

        # 1) Precision plan
        t0 = time.perf_counter()
        param_dtype = choose_param_dtype(self.device)
        amp_enabled, amp_dtype = choose_amp(self.device)

        if param_dtype != torch.float32:
            try:
                model = model.to(dtype=param_dtype)
            except Exception:
                param_dtype = torch.float32
        t_prec = time.perf_counter() - t0

        self._log(
            {
                "type": "selfopt_precision",
                "device": str(self.device),
                "param_dtype": str(param_dtype),
                "amp_enabled": bool(amp_enabled),
                "amp_dtype": str(amp_dtype),
                "dt_s": float(t_prec),
            }
        )

        # 2) Seq plan (runtime feasible; architectural max is cfg.block_size)
        block_size = int(getattr(self.cfg, "block_size", 0) or 0)
        train_seq_len = int(max(2, min(block_size or int(train_seq_len_cap), int(train_seq_len_cap))))
        eval_seq_len = int(
            max(2, min(block_size or int(eval_seq_len_cap), int(eval_seq_len_cap), int(train_seq_len)))
        )

        # 3) Batch plan by seq (throughput objective)
        seq_lens = sorted({int(x) for x in (train_seq_len, 1024, 2048) if int(x) > 0 and int(x) <= int(train_seq_len)})
        if not seq_lens:
            seq_lens = [int(train_seq_len)]

        t1 = time.perf_counter()
        batch_plan = tune_batch_by_seq(
            cache_path=self.cache_path,
            device=self.device,
            cfg=self.cfg,
            model=model,
            get_batch=get_batch,
            seq_lens=list(seq_lens),
            target_gbs=0,  # auto
            warmup=1,
            iters=2,
            verbose=False,
            amp_enabled=bool(amp_enabled),
            amp_dtype=amp_dtype,
        )
        t_batch = time.perf_counter() - t1

        self._log(
            {
                "type": "selfopt_batch_plan",
                "device": str(self.device),
                "train_seq_len": int(train_seq_len),
                "seq_lens": list(seq_lens),
                "plan": {int(k): [int(v[0]), int(v[1])] for k, v in batch_plan.by_seq.items()},
                "dt_s": float(t_batch),
            }
        )

        # 4) Compile plan (training) — only after we know base shape.
        bs0, ga0 = batch_plan.by_seq.get(int(train_seq_len), next(iter(batch_plan.by_seq.values())))
        t2 = time.perf_counter()
        model2, compile_plan = tune_torch_compile(
            cache_path=self.cache_path,
            device=self.device,
            cfg=self.cfg,
            model=model,
            get_batch=get_batch,
            batch_size=int(bs0),
            grad_accum=int(ga0),
            seq_len=int(train_seq_len),
            mode="reduce-overhead",
            warmup=1,
            iters=2,
            hysteresis=0.03,
            verbose=False,
            amp_enabled=bool(amp_enabled),
            amp_dtype=amp_dtype,
        )
        t_compile = time.perf_counter() - t2

        self._log(
            {
                "type": "selfopt_compile_plan",
                "device": str(self.device),
                "enabled": bool(getattr(compile_plan, "enabled", False)),
                "mode": str(getattr(compile_plan, "mode", "")),
                "train_seq_len": int(train_seq_len),
                "batch_size": int(bs0),
                "grad_accum": int(ga0),
                "dt_s": float(t_compile),
            }
        )

        metrics = {
            "t_precision_s": float(t_prec),
            "t_batch_tune_s": float(t_batch),
            "t_compile_tune_s": float(t_compile),
        }

        plan = RuntimePlan(
            param_dtype=param_dtype,
            amp_enabled=bool(amp_enabled),
            amp_dtype=amp_dtype,
            train_seq_len=int(train_seq_len),
            eval_seq_len=int(eval_seq_len),
            batch_plan=batch_plan,
            compile_plan=compile_plan,
            metrics=metrics,
        )

        # Persist plan for observability.
        if self.cache_path:
            try:
                set_cache_entry(
                    self.cache_path,
                    section="runtime_plan",
                    key=f"{device_sig(self.device)}|train_runtime|cfg={hash_cfg(self.cfg)}|seq={int(train_seq_len)}",
                    value={"plan": asdict(plan), "ts": float(time.time())},
                )
            except Exception:
                pass

        self._log(
            {
                "type": "selfopt_runtime_plan",
                "device": str(self.device),
                "train_seq_len": int(train_seq_len),
                "eval_seq_len": int(eval_seq_len),
                "plan": asdict(plan),
            }
        )

        return model2, plan


