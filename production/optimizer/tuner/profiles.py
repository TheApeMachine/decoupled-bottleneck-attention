"""Kernel profile helpers for fused decode."""

from __future__ import annotations

__all__ = [
    "TritonKernelProfile",
    "parse_cc_from_device_sig",
    "get_triton_kernel_profiles",
]

from dataclasses import dataclass


@dataclass(frozen=True)
class TritonKernelProfile:
    """A small, named set of launch parameters for fused decode kernels."""

    name: str
    block_n: int
    # 1-pass params
    num_warps_1pass: int = 4
    num_stages_1pass: int = 2
    # 2-pass params
    num_warps_part: int = 4
    num_stages_part: int = 2
    num_warps_reduce: int = 1
    num_stages_reduce: int = 1


def parse_cc_from_device_sig(device_signature: str) -> int | None:
    """Parse compute capability from `_device_sig` string, returning e.g. 80 for cc80."""
    s = str(device_signature)
    if "cc" not in s:
        return None
    try:
        tail = s.split("cc", 1)[1]
        digs = ""
        for ch in tail:
            if ch.isdigit():
                digs += ch
            else:
                break
        if not digs:
            return None
        return int(digs)
    except (ValueError, TypeError):
        return None


def get_triton_kernel_profiles(
    *,
    mode: str,
    device_sig: str | None = None,
    device_signature: str | None = None,
    fused: str,
    decode_block: int,
) -> list[TritonKernelProfile]:
    """Return a small set of kernel profiles for fused decode."""
    mode = str(mode)
    fused = str(fused)
    db = int(decode_block)
    device_signature = str(device_sig if device_sig is not None else (device_signature or ""))

    if mode == "off":
        return []

    cc = parse_cc_from_device_sig(device_signature)
    is_modern = bool(cc is not None and cc >= 80)

    if db < 128:
        bn = 64
    elif db < 512:
        bn = 128
    else:
        bn = 128

    profs: list[TritonKernelProfile] = []
    match fused:
        case "triton1pass":
            profs.append(
                TritonKernelProfile(
                    name="latency", block_n=bn, num_warps_1pass=4, num_stages_1pass=2
                )
            )
            if mode == "auto":
                profs.append(
                    TritonKernelProfile(
                        name="throughput",
                        block_n=bn,
                        num_warps_1pass=(8 if is_modern else 4),
                        num_stages_1pass=(3 if is_modern else 2),
                    )
                )
        case "triton2pass":
            profs.append(
                TritonKernelProfile(
                    name="latency",
                    block_n=bn,
                    num_warps_part=4,
                    num_stages_part=2,
                    num_warps_reduce=1,
                    num_stages_reduce=1,
                )
            )
            if mode == "auto":
                profs.append(
                    TritonKernelProfile(
                        name="throughput",
                        block_n=bn,
                        num_warps_part=(8 if is_modern else 4),
                        num_stages_part=(3 if is_modern else 2),
                        num_warps_reduce=(2 if is_modern else 1),
                        num_stages_reduce=1,
                    )
                )
        case _:
            return []
    return profs


# Back-compat alias (avoid importing underscore names across modules).
_parse_cc_from_device_sig = parse_cc_from_device_sig


