"""Decode-plan representation for runtime tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVDecodePlan:
    """A plan for a fused decode kernel."""

    fused: str  # "none" | "triton1pass" | "triton2pass"
    decode_block: int

    # Fused-kernel tunables (ignored for fused="none")
    block_n: int = 128
    num_warps_1pass: int = 4
    num_stages_1pass: int = 2

    num_warps_part: int = 4
    num_stages_part: int = 2

    num_warps_reduce: int = 1
    num_stages_reduce: int = 1

    def apply_to_cache(self, cache: object) -> None:
        """Apply the plan to a cache (via setattr to avoid tight coupling)."""
        setattr(cache, "decode_block", int(self.decode_block))
        setattr(cache, "fused", str(self.fused))

        setattr(cache, "block_n", int(self.block_n))
        setattr(cache, "num_warps_1pass", int(self.num_warps_1pass))
        setattr(cache, "num_stages_1pass", int(self.num_stages_1pass))
        setattr(cache, "num_warps_part", int(self.num_warps_part))
        setattr(cache, "num_stages_part", int(self.num_stages_part))
        setattr(cache, "num_warps_reduce", int(self.num_warps_reduce))
        setattr(cache, "num_stages_reduce", int(self.num_stages_reduce))


