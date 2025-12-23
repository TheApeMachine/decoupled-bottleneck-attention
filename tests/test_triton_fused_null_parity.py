import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.attention_impl.decoupled_attention_impl.triton_runtime import TRITON_AVAILABLE
from production.kvcache_backend import DecoupledLayerKVCache, KVCacheTensorConfig
from production.model import GPT, ModelConfig


class TestTritonFusedNullParity(unittest.TestCase):
    def _require_cuda_triton(self) -> None:
        if not TRITON_AVAILABLE:
            raise unittest.SkipTest("Triton is not available")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _cfg(self, *, null_attn: bool) -> ModelConfig:
        return ModelConfig(
            vocab_size=128,
            block_size=256,
            n_layer=2,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=32,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=bool(null_attn),
            tie_qk=True,
        )

    def _make_caches(self, cfg: ModelConfig, *, B: int, max_seq: int, residual_len: int) -> list[DecoupledLayerKVCache]:
        ks = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=residual_len)
        kg = KVCacheTensorConfig(kind="q8_0", qblock=32, residual_len=residual_len)
        vv = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=residual_len)
        caches: list[DecoupledLayerKVCache] = []
        for _ in range(int(cfg.n_layer)):
            caches.append(
                DecoupledLayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_sem_dim=int(cfg.sem_dim),
                    k_geo_dim=int(cfg.geo_dim),
                    v_dim=int(cfg.attn_dim),
                    k_sem_cfg=ks,
                    k_geo_cfg=kg,
                    v_cfg=vv,
                    device=torch.device("cuda"),
                )
            )
        return caches

    def _prefill(self, m: GPT, x: torch.Tensor, caches: list[DecoupledLayerKVCache]) -> None:
        pos = 0
        for t in range(int(x.size(1))):
            tok = x[:, t : t + 1]
            _lg, _caches = m(tok, caches=caches, pos_offset=pos)
            pos += 1

    def test_fused_matches_streaming_with_and_without_null(self) -> None:
        self._require_cuda_triton()
        torch.manual_seed(0)

        for null_attn in (False, True):
            cfg = self._cfg(null_attn=null_attn)
            m = GPT(cfg).eval().to(torch.device("cuda"))

            B = 2
            prefix_len = 96
            query_len = 1
            max_seq = prefix_len + query_len + 2
            x = torch.randint(0, cfg.vocab_size, (B, prefix_len + query_len), device=torch.device("cuda"), dtype=torch.long)

            for fused in ("triton1pass", "triton2pass"):
                for residual_len in (0, 256):
                    caches_stream = self._make_caches(cfg, B=B, max_seq=max_seq, residual_len=residual_len)
                    caches_fused = self._make_caches(cfg, B=B, max_seq=max_seq, residual_len=residual_len)

                    for c in caches_stream:
                        c.fused = "none"
                        c.decode_block = 64
                        c.block_n = 32
                    for c in caches_fused:
                        c.fused = "none"
                        c.decode_block = 64
                        c.block_n = 32

                    self._prefill(m, x[:, :prefix_len], caches_stream)
                    self._prefill(m, x[:, :prefix_len], caches_fused)

                    for c in caches_fused:
                        c.fused = fused

                    tok = x[:, prefix_len : prefix_len + 1]
                    logits_stream, _ = m(tok, caches=caches_stream, pos_offset=prefix_len)
                    logits_fused, _ = m(tok, caches=caches_fused, pos_offset=prefix_len)

                    diff = (logits_stream.float() - logits_fused.float()).abs().max().item()
                    self.assertLess(float(diff), 5e-2)


if __name__ == "__main__":
    unittest.main()

