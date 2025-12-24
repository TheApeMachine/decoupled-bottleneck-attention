"""
llama_upcycle provides Llama weight loading for upcycling into caramba models.
"""
from __future__ import annotations

from torch import Tensor, nn

from caramba.layer.attention import Attention
from caramba.layer.linear import Linear
from caramba.layer.rms_norm import RMSNorm
from caramba.layer.swiglu import SwiGLU
from caramba.load.llama_loader import init_decoupled_from_qkvo
from caramba.load.state_reader import StateReader
from caramba.model.embedder import Embedder
from caramba.weight.attention_decoupled import DecoupledAttentionWeight
from caramba.weight.attention_llama import LlamaAttentionWeight


class LlamaUpcycle:
    """
    LlamaUpcycle loads Llama-style checkpoints into caramba models.
    """
    def __init__(
        self,
        *,
        model: nn.Module,
        state_dict: dict[str, Tensor],
        prefix: str = "model",
        head_key: str = "lm_head.weight",
    ) -> None:
        """
        __init__ initializes a Llama upcycle loader.
        """
        self.model: nn.Module = model
        self.state: StateReader = StateReader(state_dict)
        self.prefix: str = prefix
        self.head_key: str = head_key

    def apply(self) -> None:
        """
        apply loads all supported weights into the model.
        """
        attn = self._collect(Attention)
        mlp = self._collect(SwiGLU)
        norms = self._collect(RMSNorm)

        if len(attn) != len(mlp):
            raise ValueError(
                "Expected attention/MLP counts to match, got "
                f"{len(attn)} and {len(mlp)}"
            )
        if len(norms) != (2 * len(attn) + 1):
            raise ValueError(
                "Expected 2 * n_layers + 1 RMSNorm layers, got "
                f"{len(norms)} for {len(attn)} layers"
            )

        self._load_embedder()
        self._load_blocks(attn=attn, mlp=mlp, norms=norms)
        self._load_final_norm(norms[-1])
        self._load_head()

    def _collect(self, kind: type[nn.Module]) -> list[nn.Module]:
        """
        _collect gathers modules by type in traversal order.
        """
        return [
            module
            for _name, module in self.model.named_modules()
            if isinstance(module, kind)
        ]

    def _load_embedder(self) -> None:
        """
        _load_embedder loads token embedding weights.
        """
        embed = self._find_embedder()
        if embed is None:
            raise ValueError("Model has no embedder.")
        if embed.token_embedding is None:
            raise ValueError("Embedder has no token embedding.")

        key = self.state.key(self.prefix, "embed_tokens", "weight")
        weight = self.state.get(key)
        if embed.token_embedding.weight.shape != weight.shape:
            raise ValueError(
                f"Embedder weight shape mismatch: "
                f"{embed.token_embedding.weight.shape} vs {weight.shape}"
            )
        embed.token_embedding.weight.data.copy_(weight)

    def _load_blocks(
        self,
        *,
        attn: list[nn.Module],
        mlp: list[nn.Module],
        norms: list[nn.Module],
    ) -> None:
        """
        _load_blocks loads per-layer attention, MLP, and norm weights.
        """
        for idx, (att, mlp_layer) in enumerate(zip(attn, mlp)):
            layer_prefix = self.state.key(self.prefix, "layers", str(idx))
            norm_in = norms[2 * idx]
            norm_post = norms[2 * idx + 1]

            self._load_rms_norm(
                norm_in,
                self.state.key(layer_prefix, "input_layernorm", "weight"),
            )
            self._load_rms_norm(
                norm_post,
                self.state.key(
                    layer_prefix,
                    "post_attention_layernorm",
                    "weight",
                ),
            )
            self._load_attention(
                att,
                layer_prefix=layer_prefix,
            )
            self._load_mlp(
                mlp_layer,
                layer_prefix=layer_prefix,
            )

    def _load_final_norm(self, norm: nn.Module) -> None:
        """
        _load_final_norm loads the final RMSNorm weight.
        """
        self._load_rms_norm(
            norm,
            self.state.key(self.prefix, "norm", "weight"),
        )

    def _load_head(self) -> None:
        """
        _load_head loads the LM head weight.
        """
        head = self._find_head()
        if head is None:
            raise ValueError("Model has no linear head.")

        weight = self.state.get_optional(self.head_key)
        if weight is None:
            if self.head_key != "lm_head.weight":
                raise ValueError(f"Missing state_dict key: {self.head_key}")
            embed_key = self.state.key(self.prefix, "embed_tokens", "weight")
            embed = self.state.get_optional(embed_key)
            if embed is None:
                raise ValueError(
                    "Missing state_dict key: lm_head.weight. "
                    f"Also missing {embed_key}, cannot infer tied head."
                )
            weight = embed
        if head.weight.weight.shape != weight.shape:
            raise ValueError(
                "Head weight shape mismatch: "
                f"{head.weight.weight.shape} vs {weight.shape}"
            )
        head.weight.weight.data.copy_(weight)

    def _load_rms_norm(self, layer: nn.Module, key: str) -> None:
        """
        _load_rms_norm loads RMSNorm weights into a layer.
        """
        if not isinstance(layer, RMSNorm):
            raise ValueError(f"Expected RMSNorm, got {type(layer)!r}")
        weight = self.state.get(key)
        if layer.weight.weight.shape != weight.shape:
            raise ValueError(
                f"RMSNorm weight shape mismatch: "
                f"{layer.weight.weight.shape} vs {weight.shape}"
            )
        layer.weight.weight.data.copy_(weight)

    def _load_attention(self, layer: nn.Module, *, layer_prefix: str) -> None:
        """
        _load_attention loads attention weights into a layer.
        """
        if not isinstance(layer, Attention):
            raise ValueError(f"Expected Attention, got {type(layer)!r}")

        q_w = self.state.get(
            self.state.key(layer_prefix, "self_attn", "q_proj", "weight")
        )
        k_w = self.state.get(
            self.state.key(layer_prefix, "self_attn", "k_proj", "weight")
        )
        v_w = self.state.get(
            self.state.key(layer_prefix, "self_attn", "v_proj", "weight")
        )
        o_w = self.state.get(
            self.state.key(layer_prefix, "self_attn", "o_proj", "weight")
        )
        q_b = self.state.get_optional(
            self.state.key(layer_prefix, "self_attn", "q_proj", "bias")
        )
        k_b = self.state.get_optional(
            self.state.key(layer_prefix, "self_attn", "k_proj", "bias")
        )
        v_b = self.state.get_optional(
            self.state.key(layer_prefix, "self_attn", "v_proj", "bias")
        )
        o_b = self.state.get_optional(
            self.state.key(layer_prefix, "self_attn", "o_proj", "bias")
        )

        weight = layer.weight
        if isinstance(weight, DecoupledAttentionWeight):
            init_decoupled_from_qkvo(
                student=weight,
                teacher_q=q_w,
                teacher_k=k_w,
                teacher_v=v_w,
                teacher_o=o_w,
                teacher_q_bias=q_b,
                teacher_k_bias=k_b,
                teacher_v_bias=v_b,
                teacher_o_bias=o_b,
            )
        elif isinstance(weight, LlamaAttentionWeight):
            self.state.copy_dense(weight.q_proj, weight=q_w, bias=q_b)
            self.state.copy_dense(weight.k_proj, weight=k_w, bias=k_b)
            self.state.copy_dense(weight.v_proj, weight=v_w, bias=v_b)
            self.state.copy_dense(weight.o_proj, weight=o_w, bias=o_b)
        else:
            raise ValueError(f"Unsupported attention weight: {type(weight)!r}")

    def _load_mlp(self, layer: nn.Module, *, layer_prefix: str) -> None:
        """
        _load_mlp loads SwiGLU weights into a layer.
        """
        if not isinstance(layer, SwiGLU):
            raise ValueError(f"Expected SwiGLU, got {type(layer)!r}")

        gate = self.state.get(
            self.state.key(layer_prefix, "mlp", "gate_proj", "weight")
        )
        up = self.state.get(
            self.state.key(layer_prefix, "mlp", "up_proj", "weight")
        )
        down = self.state.get(
            self.state.key(layer_prefix, "mlp", "down_proj", "weight")
        )

        self.state.copy_dense(layer.weight.w_gate, weight=gate, bias=None)
        self.state.copy_dense(layer.weight.w_up, weight=up, bias=None)
        self.state.copy_dense(layer.weight.w_down, weight=down, bias=None)

    def _find_embedder(self) -> Embedder | None:
        """
        _find_embedder returns the first embedder module.
        """
        embed = getattr(self.model, "embedder", None)
        if isinstance(embed, Embedder):
            return embed
        for _name, module in self.model.named_modules():
            if isinstance(module, Embedder):
                return module
        return None

    def _find_head(self) -> Linear | None:
        """
        _find_head returns the last linear layer in traversal order.
        """
        heads = [
            module
            for _name, module in self.model.named_modules()
            if isinstance(module, Linear)
        ]
        if not heads:
            return None
        return heads[-1]
