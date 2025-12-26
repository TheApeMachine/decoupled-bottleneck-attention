"""Llama checkpoint loading with DBA attention surgery.

When upcycling a Llama model to DBA, we can't just copy weights—the attention
architecture is different. Standard attention has Q/K/V projections; DBA has
separate semantic and geometric Q/K paths. This module handles the "surgery"
of initializing DBA projections from the original Llama weights.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from caramba.config.layer import AttentionMode
from caramba.layer.attention import AttentionLayer
from caramba.layer.linear import LinearLayer
from caramba.layer.rms_norm import RMSNormLayer
from caramba.layer.swiglu import SwiGLULayer
from caramba.loader.state_reader import StateReader
from caramba.model.embedder import Embedder


class LlamaUpcycle:
    """Loads Llama checkpoints into caramba models, handling architecture differences.

    For standard/GQA attention, weights copy directly. For DBA attention,
    we perform "attention surgery" using SVD to split the teacher's Q/K
    into semantic (content routing) and geometric (positional) components.
    """

    def __init__(
        self,
        model: nn.Module,
        state_dict: dict[str, Tensor],
        prefix: str = "model",
        head_key: str = "lm_head.weight",
    ) -> None:
        """Set up the loader with a target model and source weights.

        Args:
            model: The caramba model to load weights into
            state_dict: Llama checkpoint weights
            prefix: Key prefix in the state_dict (usually "model")
            head_key: Key for the LM head weight
        """
        self.model = model
        self.state = StateReader(state_dict)
        self.prefix = prefix
        self.head_key = head_key

    def apply(self) -> None:
        """Load all weights from the Llama checkpoint into the model.

        Handles embeddings, all transformer blocks (attention, MLP, norms),
        the final norm, and the LM head.
        """
        attn = self.collect(AttentionLayer)
        mlp = self.collect(SwiGLULayer)
        norms = self.collect(RMSNormLayer)

        if len(attn) != len(mlp):
            raise ValueError(
                f"Attention/MLP count mismatch: {len(attn)} vs {len(mlp)}"
            )
        if len(norms) != 2 * len(attn) + 1:
            raise ValueError(
                f"Expected {2 * len(attn) + 1} norms, got {len(norms)}"
            )

        self.load_embedder()
        self.load_blocks(attn=attn, mlp=mlp, norms=norms)
        self.load_final_norm(norms[-1])
        self.load_head()

    def collect(self, kind: type[nn.Module]) -> list[nn.Module]:
        """Find all modules of a given type in traversal order."""
        return [m for _, m in self.model.named_modules() if isinstance(m, kind)]

    def load_embedder(self) -> None:
        """Load the token embedding table."""
        embed = self.find_embedder()
        if embed is None or embed.token_embedding is None:
            raise ValueError("Model has no embedder with token_embedding")

        key = self.state.key(self.prefix, "embed_tokens", "weight")
        weight = self.state.get(key)
        embed.token_embedding.weight.data.copy_(weight)

    def load_blocks(
        self,
        attn: list[nn.Module],
        mlp: list[nn.Module],
        norms: list[nn.Module],
    ) -> None:
        """Load all transformer blocks: attention, MLP, and layer norms."""
        for idx, (att, mlp_layer) in enumerate(zip(attn, mlp)):
            layer_prefix = self.state.key(self.prefix, "layers", str(idx))

            self.load_rms_norm(norms[2 * idx], layer_prefix, "input_layernorm")
            self.load_rms_norm(
                norms[2 * idx + 1], layer_prefix, "post_attention_layernorm"
            )
            self.load_attention(att, layer_prefix)
            self.load_mlp(mlp_layer, layer_prefix)

    def load_final_norm(self, norm: nn.Module) -> None:
        """Load the final RMSNorm before the LM head."""
        key = self.state.key(self.prefix, "norm", "weight")
        if isinstance(norm, RMSNormLayer) and norm.weight is not None:
            norm.weight.data.copy_(self.state.get(key))

    def load_head(self) -> None:
        """Load the LM head (output projection to vocabulary).

        Falls back to using the embedding weights if lm_head.weight is missing
        (tied embeddings).
        """
        head = self.find_head()
        if head is None:
            raise ValueError("Model has no linear head")

        weight = self.state.get_optional(self.head_key)
        if weight is None:
            embed_key = self.state.key(self.prefix, "embed_tokens", "weight")
            weight = self.state.get(embed_key)

        head.linear.weight.data.copy_(weight)

    def load_rms_norm(self, layer: nn.Module, layer_prefix: str, name: str) -> None:
        """Load RMSNorm scale weights."""
        if not isinstance(layer, RMSNormLayer) or layer.weight is None:
            return
        key = self.state.key(layer_prefix, name, "weight")
        layer.weight.data.copy_(self.state.get(key))

    def load_attention(self, layer: nn.Module, layer_prefix: str) -> None:
        """Load attention weights, using DBA surgery for decoupled mode.

        For standard attention, weights copy directly. For DBA, we use SVD
        to initialize the semantic and geometric projections.
        """
        if not isinstance(layer, AttentionLayer):
            return

        attn_prefix = self.state.key(layer_prefix, "self_attn")

        q_weight = self.state.get(self.state.key(attn_prefix, "q_proj", "weight"))
        k_weight = self.state.get(self.state.key(attn_prefix, "k_proj", "weight"))
        v_weight = self.state.get(self.state.key(attn_prefix, "v_proj", "weight"))
        o_weight = self.state.get(self.state.key(attn_prefix, "o_proj", "weight"))

        if layer.mode == AttentionMode.DECOUPLED:
            self._load_attention_dba(layer, q_weight, k_weight, v_weight, o_weight)
        else:
            self._load_attention_standard(layer, q_weight, k_weight, v_weight, o_weight)

    def _load_attention_standard(
        self,
        layer: AttentionLayer,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
    ) -> None:
        """Copy weights directly for standard/GQA attention."""
        if layer.q_proj is None or layer.k_proj is None:
            raise ValueError("Standard attention layer missing Q/K projections")

        layer.q_proj.weight.data.copy_(q_weight)
        layer.k_proj.weight.data.copy_(k_weight)
        layer.v_proj.weight.data.copy_(v_weight)
        layer.out_proj.weight.data.copy_(o_weight)

    def _load_attention_dba(
        self,
        layer: AttentionLayer,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
    ) -> None:
        """Initialize DBA projections from teacher attention using SVD.

        The key insight is that semantic attention (content routing) is
        low-rank, while geometric attention (positional structure) captures
        remaining variance. We use SVD to separate these components.

        - Semantic Q/K: Top singular vectors (content/topic routing)
        - Geometric Q/K: Remaining singular vectors (position patterns)
        - Gate: Initialize to 0.5 (balanced weighting)
        - V and O: Copy directly (same dimension)
        """
        if layer.q_sem is None or layer.k_sem is None:
            raise ValueError("DBA layer missing semantic projections")
        if layer.q_geo is None or layer.k_geo is None:
            raise ValueError("DBA layer missing geometric projections")

        sem_dim = layer.q_sem.out_features
        geo_dim = layer.q_geo.out_features

        # V and O: copy directly (may truncate/pad for different v_dim)
        v_out_dim = layer.v_proj.out_features
        if v_weight.size(0) == v_out_dim:
            layer.v_proj.weight.data.copy_(v_weight)
        else:
            copy_dim = min(v_weight.size(0), v_out_dim)
            layer.v_proj.weight.data[:copy_dim, :].copy_(v_weight[:copy_dim, :])

        o_in_dim = layer.out_proj.in_features
        if o_weight.size(1) == o_in_dim:
            layer.out_proj.weight.data.copy_(o_weight)
        else:
            copy_dim = min(o_weight.size(1), o_in_dim)
            layer.out_proj.weight.data[:, :copy_dim].copy_(o_weight[:, :copy_dim])

        # SVD decomposition for Q
        self._init_dba_projection_from_svd(
            layer.q_sem.weight,
            layer.q_geo.weight,
            q_weight,
            sem_dim,
            geo_dim,
        )

        # SVD decomposition for K
        self._init_dba_projection_from_svd(
            layer.k_sem.weight,
            layer.k_geo.weight,
            k_weight,
            sem_dim,
            geo_dim,
        )

        # Initialize gate to balanced (sigmoid(0) = 0.5)
        if layer.decoupled_gate_logit is not None:
            layer.decoupled_gate_logit.data.zero_()

    def _init_dba_projection_from_svd(
        self,
        sem_weight: Tensor,
        geo_weight: Tensor,
        teacher_weight: Tensor,
        sem_dim: int,
        geo_dim: int,
    ) -> None:
        """Split teacher projection into semantic and geometric using SVD.

        The teacher's Q or K matrix is decomposed as U @ S @ Vh. The top
        singular vectors (semantic) capture content routing patterns; the
        remaining vectors (geometric) capture positional structure.
        """
        try:
            U, S, Vh = torch.linalg.svd(teacher_weight.float(), full_matrices=False)
        except Exception as e:
            # Fallback to simple truncation if SVD fails
            if "CUDA out of memory" in str(e) or "OutOfMemoryError" in type(e).__name__:
                torch.cuda.empty_cache()
            sem_weight.data.copy_(teacher_weight[:sem_dim, :].to(sem_weight.dtype))
            geo_weight.data.copy_(
                teacher_weight[sem_dim : sem_dim + geo_dim, :].to(geo_weight.dtype)
            )
            return

        rank = min(S.size(0), sem_dim + geo_dim)
        sem_rank = min(sem_dim, rank)
        geo_rank = min(geo_dim, max(0, rank - sem_dim))

        # Semantic: reconstruct from top singular vectors
        if sem_rank > 0:
            sem_proj = U[:, :sem_rank] @ torch.diag(S[:sem_rank]) @ Vh[:sem_rank, :]
            copy_rows = min(sem_dim, sem_proj.size(0))
            sem_weight.data[:copy_rows, :].copy_(
                sem_proj[:copy_rows, :].to(sem_weight.dtype)
            )

        # Geometric: reconstruct from next set of singular vectors
        if geo_rank > 0:
            geo_start = sem_rank
            geo_end = sem_rank + geo_rank
            geo_proj = (
                U[:, geo_start:geo_end]
                @ torch.diag(S[geo_start:geo_end])
                @ Vh[geo_start:geo_end, :]
            )
            copy_rows = min(geo_dim, geo_proj.size(0))
            geo_weight.data[:copy_rows, :].copy_(
                geo_proj[:copy_rows, :].to(geo_weight.dtype)
            )

    def load_mlp(self, layer: nn.Module, layer_prefix: str) -> None:
        """Load SwiGLU MLP weights (gate, up, down projections)."""
        if not isinstance(layer, SwiGLULayer):
            return

        mlp_prefix = self.state.key(layer_prefix, "mlp")
        layer.w_gate.weight.data.copy_(
            self.state.get(self.state.key(mlp_prefix, "gate_proj", "weight"))
        )
        layer.w_up.weight.data.copy_(
            self.state.get(self.state.key(mlp_prefix, "up_proj", "weight"))
        )
        layer.w_down.weight.data.copy_(
            self.state.get(self.state.key(mlp_prefix, "down_proj", "weight"))
        )

    def find_embedder(self) -> Embedder | None:
        """Find the model's embedder module."""
        for _, m in self.model.named_modules():
            if isinstance(m, Embedder):
                return m
        return None

    def find_head(self) -> LinearLayer | None:
        """Find the model's LM head (last LinearLayer)."""
        heads = [m for _, m in self.model.named_modules() if isinstance(m, LinearLayer)]
        return heads[-1] if heads else None
