"""Optional Triton fused kernels for decoupled (DBA) decode.

Why this exists:
- Triton kernels are large and should not live in the core attention module.
- They are optional: the system must import and type-check without Triton installed.
- When Triton is available at runtime, these kernels accelerate quantized decoupled decode.

The kernels implement FlashAttention-style online softmax for the decoupled attention
score decomposition: score = (Q_sem · K_sem^T) * sem_scale + (Q_geo · K_geo^T) * geo_scale

Three kernels are provided:
1. kv_decode_update_decoupled_q4q8q4: Single-pass streaming update for short sequences
2. kv_decode_partition_stats_decoupled_q4q8q4: Partition kernel for split-K decode
3. kv_decode_reduce_partitions: Reduce kernel to combine partition stats
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.optimizer.triton_runtime import TRITON_AVAILABLE

__all__ = [
    "kv_decode_update_decoupled_q4q8q4",
    "kv_decode_partition_stats_decoupled_q4q8q4",
    "kv_decode_reduce_partitions",
]


# Placeholders so imports succeed even when Triton isn't available.
kv_decode_update_decoupled_q4q8q4: object | None = None
kv_decode_partition_stats_decoupled_q4q8q4: object | None = None
kv_decode_reduce_partitions: object | None = None


# Hide Triton code from type checker (TYPE_CHECKING=True) but load at runtime when available.
if not TYPE_CHECKING and TRITON_AVAILABLE:
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        @triton.jit
        def kv_decode_update_decoupled_q4q8q4(
            q_sem_ptr,
            q_geo_ptr,
            k_sem_null_ptr,
            k_geo_null_ptr,
            v_null_ptr,
            k_sem_q_ptr,
            k_sem_s_ptr,
            k_geo_q_ptr,
            k_geo_s_ptr,
            v_q_ptr,
            v_s_ptr,
            m_ptr,
            d_ptr,
            o_ptr,
            start: tl.int32,
            L_prefix: tl.int32,
            H: tl.constexpr,
            HD_SEM: tl.constexpr,
            HD_GEO: tl.constexpr,
            HD_V: tl.constexpr,
            QBLOCK_SEM: tl.constexpr,
            QBLOCK_GEO: tl.constexpr,
            QBLOCK_V: tl.constexpr,
            SEM_SCALE: tl.constexpr,
            GEO_SCALE: tl.constexpr,
            BLOCK_N: tl.constexpr,
            NUM_SUBBLOCKS: tl.constexpr,
            HAS_NULL: tl.constexpr,
            SEED_NULL: tl.constexpr,
            stride_qsem_b: tl.constexpr,
            stride_qsem_h: tl.constexpr,
            stride_qgeo_b: tl.constexpr,
            stride_qgeo_h: tl.constexpr,
            stride_ksn_b: tl.constexpr,
            stride_ksn_h: tl.constexpr,
            stride_kgn_b: tl.constexpr,
            stride_kgn_h: tl.constexpr,
            stride_vn_b: tl.constexpr,
            stride_vn_h: tl.constexpr,
            stride_ksq_b: tl.constexpr,
            stride_ksq_t: tl.constexpr,
            stride_kss_b: tl.constexpr,
            stride_kss_t: tl.constexpr,
            stride_kss_c: tl.constexpr,
            stride_kgq_b: tl.constexpr,
            stride_kgq_t: tl.constexpr,
            stride_kgs_b: tl.constexpr,
            stride_kgs_t: tl.constexpr,
            stride_kgs_c: tl.constexpr,
            stride_vq_b: tl.constexpr,
            stride_vq_t: tl.constexpr,
            stride_vs_b: tl.constexpr,
            stride_vs_t: tl.constexpr,
            stride_vs_c: tl.constexpr,
            stride_o: tl.constexpr,
        ):
            """One streaming-update kernel: updates (m,d,o) for a block-range of tokens.

            Fuses for decode (T==1) "online softmax" over potentially long KV prefixes:
            - dequant (K_sem q4_0, K_geo q8_0, V q4_0)
            - logits computation (decoupled sem + geo)
            - exp / denom update
            - weighted value accumulation
            """
            pid = tl.program_id(0)  # 0 .. B*H-1
            b = pid // H
            h = pid - b * H

            dv = tl.arange(0, HD_V)
            # Load running state unless we're explicitly seeding from the null token.
            if HAS_NULL and SEED_NULL:
                m = -float("inf")
                d = 0.0
                o = tl.zeros((HD_V,), dtype=tl.float32)
            else:
                m = tl.load(m_ptr + pid).to(tl.float32)
                d = tl.load(d_ptr + pid).to(tl.float32)
                o = tl.load(o_ptr + pid * stride_o + dv, mask=dv < HD_V, other=0.0).to(tl.float32)

            # Load query vectors.
            ds = tl.arange(0, HD_SEM)
            dg = tl.arange(0, HD_GEO)
            q_sem = tl.load(
                q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds,
                mask=ds < HD_SEM,
                other=0.0,
            ).to(tl.float32)
            q_geo = tl.load(
                q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg,
                mask=dg < HD_GEO,
                other=0.0,
            ).to(tl.float32)

            if HAS_NULL and SEED_NULL:
                ksn = tl.load(
                    k_sem_null_ptr + b * stride_ksn_b + h * stride_ksn_h + ds,
                    mask=ds < HD_SEM,
                    other=0.0,
                ).to(tl.float32)
                kgn = tl.load(
                    k_geo_null_ptr + b * stride_kgn_b + h * stride_kgn_h + dg,
                    mask=dg < HD_GEO,
                    other=0.0,
                ).to(tl.float32)
                s_null = tl.sum(q_sem * ksn, axis=0) * SEM_SCALE + tl.sum(q_geo * kgn, axis=0) * GEO_SCALE
                vn = tl.load(
                    v_null_ptr + b * stride_vn_b + h * stride_vn_h + dv,
                    mask=dv < HD_V,
                    other=0.0,
                ).to(tl.float32)
                m = s_null
                d = 1.0
                o = vn

            # Static loop: process NUM_SUBBLOCKS contiguous blocks of BLOCK_N tokens.
            for sb in tl.static_range(NUM_SUBBLOCKS):
                t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
                tm = t < L_prefix

                # ---- semantic keys: q4_0 ----
                ksd = h * HD_SEM + ds  # global dim indices in [0, sem_dim)
                ks_byte = ksd // 2
                ks_nib = ksd % 2
                ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
                ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.int32)
                ks_hi = ks_p >> 4
                ks_lo = ks_p & 0xF
                ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
                ks_q = ks_u.to(tl.int32) - 8
                ks_sb = ksd // QBLOCK_SEM
                ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
                ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
                k_sem = ks_q.to(tl.float32) * ks_s
                logit_sem = tl.sum(k_sem * q_sem[None, :], axis=1)

                # ---- geometric keys: q8_0 ----
                kgd = h * HD_GEO + dg  # global dim indices in [0, geo_dim)
                kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
                kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int32)
                kg_sb = kgd // QBLOCK_GEO
                kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
                kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
                k_geo = kg_q.to(tl.float32) * kg_s
                logit_geo = tl.sum(k_geo * q_geo[None, :], axis=1)

                logits = logit_sem * SEM_SCALE + logit_geo * GEO_SCALE
                logits = tl.where(tm, logits, -float("inf"))

                block_max = tl.max(logits, axis=0)
                m_new = tl.maximum(m, block_max)
                exp_m = tl.exp(m - m_new)
                exp_logits = tl.exp(logits - m_new)

                # ---- values: q4_0 ----
                vd = h * HD_V + dv
                v_byte = vd // 2
                v_nib = vd % 2
                v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
                v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.int32)
                v_hi = v_p >> 4
                v_lo = v_p & 0xF
                v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
                v_q = v_u.to(tl.int32) - 8
                v_sb = vd // QBLOCK_V
                vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
                v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
                v_val = v_q.to(tl.float32) * v_s

                d = d * exp_m + tl.sum(exp_logits, axis=0)
                wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
                o = o * exp_m + wv
                m = m_new

            tl.store(m_ptr + pid, m)
            tl.store(d_ptr + pid, d)
            tl.store(o_ptr + pid * stride_o + dv, o, mask=dv < HD_V)

        @triton.jit
        def kv_decode_partition_stats_decoupled_q4q8q4(
            q_sem_ptr,
            q_geo_ptr,
            k_sem_q_ptr,
            k_sem_s_ptr,
            k_geo_q_ptr,
            k_geo_s_ptr,
            v_q_ptr,
            v_s_ptr,
            m_part_ptr,
            d_part_ptr,
            o_part_ptr,
            L_prefix: tl.int32,
            _P: tl.int32,
            PARTITION_SIZE: tl.constexpr,
            H: tl.constexpr,
            HD_SEM: tl.constexpr,
            HD_GEO: tl.constexpr,
            HD_V: tl.constexpr,
            QBLOCK_SEM: tl.constexpr,
            QBLOCK_GEO: tl.constexpr,
            QBLOCK_V: tl.constexpr,
            SEM_SCALE: tl.constexpr,
            GEO_SCALE: tl.constexpr,
            BLOCK_N: tl.constexpr,
            NUM_SUBBLOCKS: tl.constexpr,
            stride_qsem_b: tl.constexpr,
            stride_qsem_h: tl.constexpr,
            stride_qgeo_b: tl.constexpr,
            stride_qgeo_h: tl.constexpr,
            stride_ksq_b: tl.constexpr,
            stride_ksq_t: tl.constexpr,
            stride_kss_b: tl.constexpr,
            stride_kss_t: tl.constexpr,
            stride_kss_c: tl.constexpr,
            stride_kgq_b: tl.constexpr,
            stride_kgq_t: tl.constexpr,
            stride_kgs_b: tl.constexpr,
            stride_kgs_t: tl.constexpr,
            stride_kgs_c: tl.constexpr,
            stride_vq_b: tl.constexpr,
            stride_vq_t: tl.constexpr,
            stride_vs_b: tl.constexpr,
            stride_vs_t: tl.constexpr,
            stride_vs_c: tl.constexpr,
            stride_mp_row: tl.constexpr,
            stride_mp_part: tl.constexpr,
            stride_dp_row: tl.constexpr,
            stride_dp_part: tl.constexpr,
            stride_op_row: tl.constexpr,
            stride_op_part: tl.constexpr,
        ):
            """Partition kernel for split-K (FlashAttention-style) decode.

            Parallelize decode across the KV sequence length by slicing the prefix into partitions.
            Each partition computes local (m, d, o) and we reduce partitions in a second kernel.
            """
            pid_row = tl.program_id(0)  # 0 .. BH-1
            pid_part = tl.program_id(1)  # 0 .. P-1
            b = pid_row // H
            h = pid_row - b * H

            ds = tl.arange(0, HD_SEM)
            dg = tl.arange(0, HD_GEO)
            dv = tl.arange(0, HD_V)

            qsem = tl.load(
                q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds,
                mask=ds < HD_SEM,
                other=0.0,
            ).to(tl.float32)
            qgeo = tl.load(
                q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg,
                mask=dg < HD_GEO,
                other=0.0,
            ).to(tl.float32)

            m = -float("inf")
            d = 0.0
            o = tl.zeros([HD_V], dtype=tl.float32)

            start = pid_part * PARTITION_SIZE
            for sb in tl.static_range(0, NUM_SUBBLOCKS):
                t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
                tm = t < L_prefix

                # --- Semantic K (q4_0) dot ---
                ksd = h * HD_SEM + ds
                ks_byte = ksd // 2
                ks_nib = ksd % 2
                ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
                ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.uint8)
                ks_hi = ks_p >> 4
                ks_lo = ks_p & 0xF
                ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
                ks_q = ks_u.to(tl.int32) - 8

                ks_sb = ksd // QBLOCK_SEM
                ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
                ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
                ksem = ks_q.to(tl.float32) * ks_s
                sem = tl.sum(ksem * qsem[None, :], axis=1)

                # --- Geometric K (q8_0) dot ---
                kgd = h * HD_GEO + dg
                kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
                kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int8).to(tl.float32)

                kg_sb = kgd // QBLOCK_GEO
                kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
                kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
                kgeo = kg_q * kg_s
                geo = tl.sum(kgeo * qgeo[None, :], axis=1)

                logits = sem * SEM_SCALE + geo * GEO_SCALE
                logits = tl.where(tm, logits, -float("inf"))

                block_max = tl.max(logits, axis=0)
                m_new = tl.maximum(m, block_max)
                exp_m = tl.exp(m - m_new)
                exp_logits = tl.exp(logits - m_new)

                # --- V (q4_0) weighted sum ---
                vd = h * HD_V + dv
                v_byte = vd // 2
                v_nib = vd % 2
                v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
                v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.uint8)
                v_hi = v_p >> 4
                v_lo = v_p & 0xF
                v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
                v_q = v_u.to(tl.int32) - 8

                v_sb = vd // QBLOCK_V
                vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
                v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
                v_val = v_q.to(tl.float32) * v_s

                d = d * exp_m + tl.sum(exp_logits, axis=0)
                wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
                o = o * exp_m + wv
                m = m_new

            tl.store(m_part_ptr + pid_row * stride_mp_row + pid_part * stride_mp_part, m)
            tl.store(d_part_ptr + pid_row * stride_dp_row + pid_part * stride_dp_part, d)
            tl.store(
                o_part_ptr + pid_row * stride_op_row + pid_part * stride_op_part + dv,
                o,
                mask=dv < HD_V,
            )

        @triton.jit
        def kv_decode_reduce_partitions(
            q_sem_ptr,
            q_geo_ptr,
            k_sem_null_ptr,
            k_geo_null_ptr,
            v_null_ptr,
            m_part_ptr,
            d_part_ptr,
            o_part_ptr,
            m_ptr,
            d_ptr,
            o_ptr,
            P: tl.int32,
            NUM_PARTS: tl.constexpr,
            HAS_NULL: tl.constexpr,
            H: tl.constexpr,
            HD_SEM: tl.constexpr,
            HD_GEO: tl.constexpr,
            HD_V: tl.constexpr,
            SEM_SCALE: tl.constexpr,
            GEO_SCALE: tl.constexpr,
            stride_qsem_b: tl.constexpr,
            stride_qsem_h: tl.constexpr,
            stride_qgeo_b: tl.constexpr,
            stride_qgeo_h: tl.constexpr,
            stride_ksn_b: tl.constexpr,
            stride_ksn_h: tl.constexpr,
            stride_kgn_b: tl.constexpr,
            stride_kgn_h: tl.constexpr,
            stride_vn_b: tl.constexpr,
            stride_vn_h: tl.constexpr,
            stride_mp_row: tl.constexpr,
            stride_mp_part: tl.constexpr,
            stride_dp_row: tl.constexpr,
            stride_dp_part: tl.constexpr,
            stride_op_row: tl.constexpr,
            stride_op_part: tl.constexpr,
            stride_o: tl.constexpr,
        ):
            """Reduce partitions for split-K decode.

            Combine local (m, d, o) from partitions into a single online-softmax state.
            """
            pid_row = tl.program_id(0)  # 0..BH-1
            b = pid_row // H
            h = pid_row - b * H
            dv = tl.arange(0, HD_V)

            if HAS_NULL:
                ds = tl.arange(0, HD_SEM)
                dg = tl.arange(0, HD_GEO)
                qsem = tl.load(
                    q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds,
                    mask=ds < HD_SEM,
                    other=0.0,
                ).to(tl.float32)
                qgeo = tl.load(
                    q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg,
                    mask=dg < HD_GEO,
                    other=0.0,
                ).to(tl.float32)
                ksn = tl.load(
                    k_sem_null_ptr + b * stride_ksn_b + h * stride_ksn_h + ds,
                    mask=ds < HD_SEM,
                    other=0.0,
                ).to(tl.float32)
                kgn = tl.load(
                    k_geo_null_ptr + b * stride_kgn_b + h * stride_kgn_h + dg,
                    mask=dg < HD_GEO,
                    other=0.0,
                ).to(tl.float32)
                s_null = tl.sum(qsem * ksn, axis=0) * SEM_SCALE + tl.sum(qgeo * kgn, axis=0) * GEO_SCALE
                vn = tl.load(
                    v_null_ptr + b * stride_vn_b + h * stride_vn_h + dv,
                    mask=dv < HD_V,
                    other=0.0,
                ).to(tl.float32)
            else:
                s_null = tl.full([], -float("inf"), tl.float32)
                vn = tl.zeros([HD_V], dtype=tl.float32)

            # First pass: global max over partitions.
            m = -float("inf")
            for p in tl.static_range(0, NUM_PARTS):
                p_i = tl.full([], p, tl.int32)
                pm = p_i < P
                mp = tl.load(
                    m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part,
                    mask=pm,
                    other=-float("inf"),
                )
                m = tl.maximum(m, mp)

            if HAS_NULL:
                m = tl.maximum(m, s_null)

            # Second pass: combine denominators and outputs.
            d = 0.0
            o = tl.zeros([HD_V], dtype=tl.float32)
            for p in tl.static_range(0, NUM_PARTS):
                p_i = tl.full([], p, tl.int32)
                pm = p_i < P
                mp = tl.load(
                    m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part,
                    mask=pm,
                    other=-float("inf"),
                )
                dp = tl.load(
                    d_part_ptr + pid_row * stride_dp_row + p_i * stride_dp_part,
                    mask=pm,
                    other=0.0,
                )
                op = tl.load(
                    o_part_ptr + pid_row * stride_op_row + p_i * stride_op_part + dv,
                    mask=pm & (dv < HD_V),
                    other=0.0,
                ).to(tl.float32)

                scale = tl.where(pm, tl.exp(mp - m), 0.0)
                d += dp * scale
                o += op * scale

            if HAS_NULL:
                s = tl.exp(s_null - m)
                d += s
                o += vn * s

            tl.store(m_ptr + pid_row, m)
            tl.store(d_ptr + pid_row, d)
            tl.store(o_ptr + pid_row * stride_o + dv, o, mask=dv < HD_V)
