"""Heuristics for intent → concrete architecture/training defaults."""

from __future__ import annotations

import math


class HeuristicPlanner:
    """Pure heuristics for selecting architecture/training defaults."""

    @staticmethod
    def derive_lr_from_params(target_params: int) -> float:
        ref: float = 1_000_000_000.0
        base: float = 3e-4
        exp: float = 0.10
        tp: float = float(target_params)
        ratio: float = tp / ref
        scale: float = float(math.pow(ratio, -exp))
        return float(base * scale)

    @staticmethod
    def choose_layers(*, target_params: int, device_type: str) -> int:
        """Choose model depth that yields sane head_dim and fits common hardware profiles."""
        from production.config import derive_d_model, derive_n_head  # pylint: disable=import-outside-toplevel

        target_params = int(max(1, target_params))
        device_type = str(device_type or "cpu")

        if target_params < 50_000_000:
            candidates = (2, 4, 6, 8, 12)
            min_d_model = 64
            multiple = 64
        elif target_params < 500_000_000:
            candidates = (8, 12, 16, 20, 22, 24, 32)
            min_d_model = 256
            multiple = 128
        else:
            candidates = (12, 16, 20, 22, 24, 32, 48, 64, 96)
            min_d_model = 256
            multiple = 128

        depth_bias = 0.15 if device_type in ("cpu", "mps") else 0.0

        best_L = int(candidates[0])  # pylint: disable=invalid-name
        best_score = float("inf")

        for L in candidates:  # pylint: disable=invalid-name
            d_model = derive_d_model(
                layers=int(L),
                target_params=int(target_params),
                multiple=multiple,
                min_d_model=min_d_model,
            )
            n_head = derive_n_head(int(d_model))
            head_dim = int(d_model) / max(1, int(n_head))

            score = abs(float(head_dim) - 128.0) / 128.0
            score += depth_bias * (float(L) / 24.0)

            if score < best_score:
                best_score = float(score)
                best_L = int(L)  # pylint: disable=invalid-name

        return int(best_L)  # pylint: disable=invalid-name
