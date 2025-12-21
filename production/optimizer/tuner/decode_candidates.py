"""Decode-plan search space construction."""

from __future__ import annotations

import torch

from production.selfopt_utils import device_sig

from production.optimizer.tuner.config import KVSelfOptConfig
from production.optimizer.tuner.decode_plan import KVDecodePlan
from production.optimizer.tuner.profiles import get_triton_kernel_profiles
from production.optimizer.tuner.triton_availability import triton_decoupled_q4q8q4_available


def allowed_fused_modes(*, base_fused: str, cache: object) -> list[str]:
    """Allowed fused-kernel modes given base preference + cache layout."""
    base_fused = str(base_fused)
    if base_fused == "none":
        return ["none"]

    if not triton_decoupled_q4q8q4_available():
        return ["none"]

    ok = True
    try:
        ok = (
            getattr(getattr(cache, "k_sem"), "kind") == "q4_0"
            and getattr(getattr(cache, "k_geo"), "kind") == "q8_0"
            and getattr(getattr(cache, "v"), "kind") == "q4_0"
        )
    except (AttributeError, TypeError):
        ok = False
    if not ok:
        return ["none"]

    match base_fused:
        case "triton1pass" | "triton2pass":
            return [base_fused]
        case _:
            return ["none", "triton1pass", "triton2pass"]


def candidate_plans(
    cfg: KVSelfOptConfig,
    *,
    device: torch.device,
    base_decode_block: int,
    base_fused: str,
    cache: object,
) -> list[KVDecodePlan]:
    """Generate candidate decode plans for this cache/device."""
    fused_modes = allowed_fused_modes(base_fused=str(base_fused), cache=cache)

    decode_blocks = list(dict.fromkeys([int(base_decode_block), *cfg.decode_blocks]))
    decode_blocks = [int(x) for x in decode_blocks if int(x) > 0]
    decode_blocks.sort()

    use_profiles = (not bool(getattr(cfg, "expert_launch_space", False))) and (
        str(getattr(cfg, "kernel_profiles", "auto")) != "off"
    )

    block_ns = [int(x) for x in cfg.block_ns if int(x) > 0] or [128]
    warps = [int(x) for x in cfg.warps if int(x) > 0] or [4]
    stages = [int(x) for x in cfg.stages if int(x) > 0] or [2]

    plans: list[KVDecodePlan] = []
    for fused in fused_modes:
        for db in decode_blocks:
            if fused == "none":
                plans.append(KVDecodePlan(fused="none", decode_block=db))
                continue

            if use_profiles:
                profs = get_triton_kernel_profiles(
                    mode=str(getattr(cfg, "kernel_profiles", "auto")),
                    device_sig=device_sig(device),
                    fused=fused,
                    decode_block=int(db),
                )
                for pr in profs:
                    bn = int(pr.block_n)
                    if db < bn:
                        continue
                    if fused == "triton1pass":
                        plans.append(
                            KVDecodePlan(
                                fused=fused,
                                decode_block=db,
                                block_n=bn,
                                num_warps_1pass=int(pr.num_warps_1pass),
                                num_stages_1pass=int(pr.num_stages_1pass),
                            )
                        )
                    else:
                        plans.append(
                            KVDecodePlan(
                                fused=fused,
                                decode_block=db,
                                block_n=bn,
                                num_warps_part=int(pr.num_warps_part),
                                num_stages_part=int(pr.num_stages_part),
                                num_warps_reduce=int(pr.num_warps_reduce),
                                num_stages_reduce=int(pr.num_stages_reduce),
                            )
                        )
                continue

            for bn in block_ns:
                if db < bn:
                    continue
                for w in warps:
                    for st in stages:
                        if fused == "triton1pass":
                            plans.append(
                                KVDecodePlan(
                                    fused=fused,
                                    decode_block=db,
                                    block_n=bn,
                                    num_warps_1pass=w,
                                    num_stages_1pass=st,
                                )
                            )
                        else:
                            plans.append(
                                KVDecodePlan(
                                    fused=fused,
                                    decode_block=db,
                                    block_n=bn,
                                    num_warps_part=w,
                                    num_stages_part=st,
                                    num_warps_reduce=1,
                                    num_stages_reduce=1,
                                )
                            )
    return plans


