"""
llama_upcycle_test provides tests for LlamaUpcycle.
"""
from __future__ import annotations

import unittest
from typing import TypeVar

import torch
from torch import nn

from caramba.config.embedder import EmbedderType, TokenEmbedderConfig
from caramba.config.layer import (
    AttentionLayerConfig,
    LayerType,
    LinearLayerConfig,
    RMSNormLayerConfig,
    SwiGLULayerConfig,
)
from caramba.config.model import ModelConfig, ModelType
from caramba.config.operation import (
    AttentionOperationConfig,
    MatmulOperationConfig,
    RMSNormOperationConfig,
    SwiGLUOperationConfig,
)
from caramba.config.topology import (
    NestedTopologyConfig,
    ResidualTopologyConfig,
    StackedTopologyConfig,
)
from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    DenseWeightConfig,
    RMSNormWeightConfig,
    SwiGLUWeightConfig,
)
from caramba.layer.attention import Attention
from caramba.layer.linear import Linear
from caramba.layer.rms_norm import RMSNorm
from caramba.layer.swiglu import SwiGLU
from caramba.load.llama_upcycle import LlamaUpcycle
from caramba.model.model import Model
from caramba.weight.attention_decoupled import DecoupledAttentionWeight

T = TypeVar("T", bound=nn.Module)


class LlamaUpcycleTest(unittest.TestCase):
    """
    LlamaUpcycleTest provides tests for loading Llama weights.
    """
    def test_loads_minimal_llama_block(self) -> None:
        """
        test loading a minimal one-layer Llama block into a model.
        """
        d_model = 8
        n_heads = 2
        n_kv_heads = 1
        head_dim = d_model // n_heads
        sem_dim = 2
        geo_dim = 6
        sem_head = sem_dim // n_heads
        geo_head = geo_dim // n_heads
        d_ff = 16
        vocab_size = 32

        norm_op = RMSNormOperationConfig(eps=1e-5)
        norm_weight = RMSNormWeightConfig(d_model=d_model)
        att_op = AttentionOperationConfig(is_causal=True, dropout_p=0.0)
        att_weight = DecoupledAttentionWeightConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            rope_base=10000.0,
            rope_dim=2,
            bias=False,
            gate=True,
        )
        mlp_op = SwiGLUOperationConfig()
        mlp_weight = SwiGLUWeightConfig(
            d_model=d_model,
            d_ff=d_ff,
            bias=False,
        )

        att_residual = ResidualTopologyConfig(
            layers=[
                RMSNormLayerConfig(
                    type=LayerType.RMS_NORM,
                    operation=norm_op,
                    weight=norm_weight,
                ),
                AttentionLayerConfig(
                    type=LayerType.ATTENTION,
                    operation=att_op,
                    weight=att_weight,
                ),
            ],
        )
        mlp_residual = ResidualTopologyConfig(
            layers=[
                RMSNormLayerConfig(
                    type=LayerType.RMS_NORM,
                    operation=norm_op,
                    weight=norm_weight,
                ),
                SwiGLULayerConfig(
                    type=LayerType.SWIGLU,
                    operation=mlp_op,
                    weight=mlp_weight,
                ),
            ],
        )
        nested = NestedTopologyConfig(
            repeat=1,
            layers=[att_residual, mlp_residual],
        )
        final_norm = RMSNormLayerConfig(
            type=LayerType.RMS_NORM,
            operation=norm_op,
            weight=norm_weight,
        )
        head = LinearLayerConfig(
            type=LayerType.LINEAR,
            operation=MatmulOperationConfig(),
            weight=DenseWeightConfig(
                d_in=d_model,
                d_out=vocab_size,
                bias=False,
            ),
        )
        topology = StackedTopologyConfig(layers=[nested, final_norm, head])

        model = Model(
            ModelConfig(
                type=ModelType.TRANSFORMER,
                embedder=TokenEmbedderConfig(
                    type=EmbedderType.TOKEN,
                    vocab_size=vocab_size,
                    d_model=d_model,
                ),
                topology=topology,
            )
        )

        state_dict: dict[str, torch.Tensor] = {}
        state_dict["model.embed_tokens.weight"] = torch.randn(
            vocab_size,
            d_model,
        )
        state_dict["model.layers.0.input_layernorm.weight"] = torch.randn(
            d_model,
        )
        state_dict[
            "model.layers.0.post_attention_layernorm.weight"
        ] = torch.randn(d_model)
        state_dict["model.layers.0.self_attn.q_proj.weight"] = torch.randn(
            d_model,
            d_model,
        )
        state_dict["model.layers.0.self_attn.k_proj.weight"] = torch.randn(
            n_kv_heads * head_dim,
            d_model,
        )
        state_dict["model.layers.0.self_attn.v_proj.weight"] = torch.randn(
            n_kv_heads * head_dim,
            d_model,
        )
        state_dict["model.layers.0.self_attn.o_proj.weight"] = torch.randn(
            d_model,
            d_model,
        )
        state_dict["model.layers.0.mlp.gate_proj.weight"] = torch.randn(
            d_ff,
            d_model,
        )
        state_dict["model.layers.0.mlp.up_proj.weight"] = torch.randn(
            d_ff,
            d_model,
        )
        state_dict["model.layers.0.mlp.down_proj.weight"] = torch.randn(
            d_model,
            d_ff,
        )
        state_dict["model.norm.weight"] = torch.randn(d_model)
        state_dict["lm_head.weight"] = torch.randn(vocab_size, d_model)

        LlamaUpcycle(model=model, state_dict=state_dict).apply()

        attn = self._first(model, Attention)
        self.assertIsInstance(attn, Attention)
        assert isinstance(attn, Attention)

        weight = attn.weight
        self.assertIsInstance(weight, DecoupledAttentionWeight)
        assert isinstance(weight, DecoupledAttentionWeight)

        rms = self._all(model, RMSNorm)
        self.assertEqual(len(rms), 3)
        torch.testing.assert_close(
            rms[0].weight.weight,
            state_dict["model.layers.0.input_layernorm.weight"],
        )
        torch.testing.assert_close(
            rms[1].weight.weight,
            state_dict["model.layers.0.post_attention_layernorm.weight"],
        )
        torch.testing.assert_close(
            rms[2].weight.weight,
            state_dict["model.norm.weight"],
        )

        mlp = self._first(model, SwiGLU)
        self.assertIsInstance(mlp, SwiGLU)
        assert isinstance(mlp, SwiGLU)
        torch.testing.assert_close(
            mlp.weight.w_gate.weight,
            state_dict["model.layers.0.mlp.gate_proj.weight"],
        )
        torch.testing.assert_close(
            mlp.weight.w_up.weight,
            state_dict["model.layers.0.mlp.up_proj.weight"],
        )
        torch.testing.assert_close(
            mlp.weight.w_down.weight,
            state_dict["model.layers.0.mlp.down_proj.weight"],
        )

        head = self._last(model, Linear)
        self.assertIsInstance(head, Linear)
        assert isinstance(head, Linear)
        torch.testing.assert_close(
            head.weight.weight,
            state_dict["lm_head.weight"],
        )

        embed = model.embedder.token_embedding
        self.assertIsNotNone(embed)
        assert embed is not None
        torch.testing.assert_close(
            embed.weight,
            state_dict["model.embed_tokens.weight"],
        )

        q_w = state_dict["model.layers.0.self_attn.q_proj.weight"]
        k_w = state_dict["model.layers.0.self_attn.k_proj.weight"]
        q_view = q_w.view(n_heads, head_dim, d_model)
        k_view = k_w.view(n_kv_heads, head_dim, d_model)
        q_sem = q_view[:, :sem_head, :].reshape_as(weight.q_sem.weight)
        q_geo = q_view[:, sem_head : sem_head + geo_head, :].reshape_as(
            weight.q_geo.weight
        )
        k_sem = k_view[:, :sem_head, :].reshape_as(weight.k_sem.weight)
        k_geo = k_view[:, sem_head : sem_head + geo_head, :].reshape_as(
            weight.k_geo.weight
        )
        torch.testing.assert_close(weight.q_sem.weight, q_sem)
        torch.testing.assert_close(weight.q_geo.weight, q_geo)
        torch.testing.assert_close(weight.k_sem.weight, k_sem)
        torch.testing.assert_close(weight.k_geo.weight, k_geo)
        torch.testing.assert_close(
            weight.v_proj.weight,
            state_dict["model.layers.0.self_attn.v_proj.weight"],
        )
        torch.testing.assert_close(
            weight.o_proj.weight,
            state_dict["model.layers.0.self_attn.o_proj.weight"],
        )
        gate_logit = weight.gate_logit
        assert gate_logit is not None
        torch.testing.assert_close(
            gate_logit,
            torch.zeros_like(gate_logit),
        )

    def _first(
        self,
        model: nn.Module,
        kind: type[T],
    ) -> T | None:
        """
        _first returns the first matching module.
        """
        for _name, module in model.named_modules():
            if isinstance(module, kind):
                return module
        return None

    def _last(
        self,
        model: nn.Module,
        kind: type[T],
    ) -> T | None:
        """
        _last returns the last matching module.
        """
        last: T | None = None
        for _name, module in model.named_modules():
            if isinstance(module, kind):
                last = module
        return last

    def _all(self, model: nn.Module, kind: type[T]) -> list[T]:
        """
        _all returns all matching modules in traversal order.
        """
        return [
            module
            for _name, module in model.named_modules()
            if isinstance(module, kind)
        ]
