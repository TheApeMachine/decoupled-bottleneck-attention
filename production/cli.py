"""
CLI for the production system.
"""
from __future__ import annotations
import argparse
import sys
from typing import NoReturn, cast

# Python 3.12+ has `typing.override`; older runtimes should use typing_extensions.
try:  # pragma: no cover
    from typing import override  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import override
import copy
import math
import torch
from production.config import pick_device, set_seed
from production.config import EXP_PRESETS
from production.optimizer import apply_dynamic_config
from production.runner import run_single

class _MinimalParser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:  # pragma: no cover
        # Provide a stronger hint for common migration failure mode.
        if "unrecognized arguments" in message:
            message = (
                f"{message}\n\n"
                f"Hint: this project uses an intent-first CLI and does not accept legacy optimization flags. "
                f"Specify only high-level intent (model size, dataset, result type); everything else is derived."
            )
        super().error(message)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the minimal, intent-first CLI argparser (default UX).

    Advanced optimization knobs are intentionally hidden; the system self-optimizes based on hardware.
    """
    ap = _MinimalParser()

    _ = ap.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Run mode.")
    _ = ap.add_argument(
        "--exp",
        type=str,
        default=None,
        choices=sorted(list(EXP_PRESETS.keys()) + ["paper_all"]),
        help="Experiment preset.",
    )
    _ = ap.add_argument(
        "--result",
        type=str,
        default=None,
        choices=["baseline", "bottleneck", "decoupled", "gqa"],
        help="Result/architecture intent (preferred over --exp).",
    )
    _ = ap.add_argument(
        "--size",
        type=str,
        default=None,
        help="Target model size in params (e.g. 100m, 1b). If omitted, inferred from dataset tokens.",
    )
    _ = ap.add_argument("--data", type=str, default=None, help="Dataset path (train mode).")
    _ = ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (sample mode).")
    _ = ap.add_argument("--out-dir", type=str, default=None, help="Output directory (optional if size+exp given).")
    _ = ap.add_argument("--resume", action="store_true", help="Resume training from out-dir/last.pt (train mode only).")
    _ = ap.add_argument(
        "--resume-path", type=str, default=None, help="Resume training from an explicit checkpoint path (train mode only)."
    )
    _ = ap.add_argument("--seed", type=int, default=1337)
    _ = ap.add_argument(
        "--steps",
        type=int,
        default=-1,
        help="Training steps: <0 auto, 0 validate-only, >0 explicit.",
    )
    _ = ap.add_argument(
        "--prompt-tokens",
        type=str,
        default="0",
        help="Prompt as whitespace-separated token IDs (or text if tokenizer=tiktoken).",
    )
    _ = ap.add_argument("--max-new-tokens", type=int, default=50)
    _ = ap.add_argument("--temperature", type=float, default=1.0)
    _ = ap.add_argument("--top-k", type=int, default=None)
    _ = ap.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write Weights & Biases scalars (default: enabled).",
    )

    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse argv using the minimal CLI.

    We still populate attributes expected by the runtime by merging onto internal defaults. This
    keeps downstream code stable while the public CLI stays small and footgun-free.
    """
    if argv is None:
        argv = sys.argv[1:]

    return build_arg_parser().parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """
    Execute the CLI request with v30-compatible behavior.
    """
    sdb = cast(object | None, getattr(args, "spec_disable_below_accept", None))
    if sdb is not None:
        try:
            sdb_f = float(str(sdb))
        except (TypeError, ValueError) as e:
            raise ValueError("--spec-disable-below-accept must be between 0.0 and 1.0") from e
        if not math.isfinite(sdb_f) or sdb_f < 0.0 or sdb_f > 1.0:
            raise ValueError("--spec-disable-below-accept must be between 0.0 and 1.0")

    device = pick_device(getattr(args, "device", None))
    set_seed(int(getattr(args, "seed", 1337)))

    # Matmul precision hint (mostly impacts float32 matmuls).
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(getattr(args, "matmul_precision", "high")))
    except (RuntimeError, ValueError, TypeError):
        pass

    # For paper_all, run each experiment sequentially (train mode only).
    if getattr(args, "mode", "train") == "train" and getattr(args, "exp", None) == "paper_all":
        for exp in ["paper_baseline", "paper_bottleneck", "paper_decoupled", "paper_gqa"]:
            a2 = copy.deepcopy(args)
            a2.exp = exp
            apply_dynamic_config(a2, device=device)
            run_single(a2, device)
        return 0

    apply_dynamic_config(args, device=device)
    run_single(args, device)
    return 0


def main() -> int:
    """
    Module entrypoint so `python -m production.cli ...` works.
    """
    args = parse_args()

    # Validate args immediately after parsing so invalid values fail fast (before any device/model init).
    sdb = cast(object | None, getattr(args, "spec_disable_below_accept", None))
    if sdb is not None:
        try:
            sdb_f = float(str(sdb))
        except (TypeError, ValueError) as e:
            raise ValueError("--spec-disable-below-accept must be between 0.0 and 1.0") from e
        if not math.isfinite(sdb_f) or sdb_f < 0.0 or sdb_f > 1.0:
            raise ValueError("--spec-disable-below-accept must be between 0.0 and 1.0")

    return int(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
