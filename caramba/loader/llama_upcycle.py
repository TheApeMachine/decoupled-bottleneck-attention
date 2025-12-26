"""
llama_upcycle provides Llama weight loading for upcycling into caramba models.

Supports both standard attention loading and DBA surgery (splitting Q/K into
semantic and geometric paths).
"""
from __future__ import annotations

import math

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
    """Loads Llama-style checkpoints into caramba models.

    For standard/GQA attention layers, loads weights directly.
    For DBA (decoupled) attention layers, performs "attention surgery" -
    initializing semantic and geometric projections from the teacher Q/K.
    """

    def __init__(
        self,
        model: nn.Module,
        state_dict: dict[str, Tensor],
        prefix: str = "model",
        head_key: str = "lm_head.weight",
    ) -> None:
        self.model = model
        self.state = StateReader(state_dict)
        self.prefix = prefix
        self.head_key = head_key

    def apply(self) -> None:
        """Load all supported weights into the model."""
        attn = self.collect(AttentionLayer)
        mlp = self.collect(SwiGLULayer)
        norms = self.collect(RMSNormLayer)

        if len(attn) != len(mlp):
            raise ValueError(f"Attention/MLP count mismatch: {len(attn)} vs {len(mlp)}")
        if len(norms) != 2 * len(attn) + 1:
            raise ValueError(f"Expected {2 * len(attn) + 1} norms, got {len(norms)}")

        self.load_embedder()
        self.load_blocks(attn=attn, mlp=mlp, norms=norms)
        self.load_final_norm(norms[-1])
        self.load_head()

    def collect(self, kind: type[nn.Module]) -> list[nn.Module]:
        """Gather modules by type in traversal order."""
        return [m for _, m in self.model.named_modules() if isinstance(m, kind)]

    def load_embedder(self) -> None:
        """Load token embedding weights."""
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
        """Load per-layer attention, MLP, and norm weights."""
        for idx, (att, mlp_layer) in enumerate(zip(attn, mlp)):
            layer_prefix = self.state.key(self.prefix, "layers", str(idx))

            self.load_rms_norm(norms[2 * idx], layer_prefix, "input_layernorm")
            self.load_rms_norm(norms[2 * idx + 1], layer_prefix, "post_attention_layernorm")
            self.load_attention(att, layer_prefix)
            self.load_mlp(mlp_layer, layer_prefix)

    def load_final_norm(self, norm: nn.Module) -> None:
        """Load the final RMSNorm weight."""
        key = self.state.key(self.prefix, "norm", "weight")
        if isinstance(norm, RMSNormLayer) and norm.weight is not None:
            norm.weight.data.copy_(self.state.get(key))

    def load_head(self) -> None:
        """Load the LM head weight."""
        head = self.find_head()
        if head is None:
            raise ValueError("Model has no linear head")

        weight = self.state.get_optional(self.head_key)
        if weight is None:
            embed_key = self.state.key(self.prefix, "embed_tokens", "weight")
            weight = self.state.get(embed_key)

        head.linear.weight.data.copy_(weight)

    def load_rms_norm(self, layer: nn.Module, layer_prefix: str, name: str) -> None:
        """Load RMSNorm weights."""
        if not isinstance(layer, RMSNormLayer) or layer.weight is None:
            return
        key = self.state.key(layer_prefix, name, "weight")
        layer.weight.data.copy_(self.state.get(key))

    def load_attention(self, layer: nn.Module, layer_prefix: str) -> None:
        """Load attention weights, handling both standard and DBA modes."""
        if not isinstance(layer, AttentionLayer):
            return

        attn_prefix = self.state.key(layer_prefix, "self_attn")

        # Load teacher weights
        q_weight = self.state.get(self.state.key(attn_prefix, "q_proj", "weight"))
        k_weight = self.state.get(self.state.key(attn_prefix, "k_proj", "weight"))
        v_weight = self.state.get(self.state.key(attn_prefix, "v_proj", "weight"))
        o_weight = self.state.get(self.state.key(attn_prefix, "o_proj", "weight"))

        if layer.mode == AttentionMode.DECOUPLED:
            # DBA surgery - split Q/K into semantic and geometric paths
            self._load_attention_dba(layer, q_weight, k_weight, v_weight, o_weight)
        else:
            # Standard/GQA - direct copy
            self._load_attention_standard(layer, q_weight, k_weight, v_weight, o_weight)

    def _load_attention_standard(
        self,
        layer: AttentionLayer,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
    ) -> None:
        """Load weights directly into standard/GQA attention."""
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
        """Initialize DBA projections from teacher attention weights.

        Strategy (attention surgery):
        - Value and output: copy directly (same dimension)
        - Semantic Q/K: use top singular vectors (content routing is low-rank)
        - Geometric Q/K: use remaining singular vectors (positional info)
        - Gate: initialize to 0.5 (equal weighting of paths)
        """
        if layer.q_sem is None or layer.k_sem is None:
            raise ValueError("DBA layer missing semantic projections")
        if layer.q_geo is None or layer.k_geo is None:
            raise ValueError("DBA layer missing geometric projections")

        sem_dim = layer.q_sem.out_features
        geo_dim = layer.q_geo.out_features
        d_model = q_weight.size(1)

        # V and O: copy directly (may need truncation/expansion for different v_dim)
        v_out_dim = layer.v_proj.out_features
        if v_weight.size(0) == v_out_dim:
            layer.v_proj.weight.data.copy_(v_weight)
        else:
            # Truncate or pad
            copy_dim = min(v_weight.size(0), v_out_dim)
            layer.v_proj.weight.data[:copy_dim, :].copy_(v_weight[:copy_dim, :])

        o_in_dim = layer.out_proj.in_features
        if o_weight.size(1) == o_in_dim:
            layer.out_proj.weight.data.copy_(o_weight)
        else:
            copy_dim = min(o_weight.size(1), o_in_dim)
            layer.out_proj.weight.data[:, :copy_dim].copy_(o_weight[:, :copy_dim])

        # Use SVD to find the principal components for semantic (low-rank) path
        # and use remaining for geometric path
        # Q_weight is [out_features, d_model] = [n_heads * head_dim, d_model]

        # For Q: decompose and assign
        self._init_dba_projection_from_svd(
            layer.q_sem.weight,
            layer.q_geo.weight,
            q_weight,
            sem_dim,
            geo_dim,
        )

        # For K: decompose and assign
        self._init_dba_projection_from_svd(
            layer.k_sem.weight,
            layer.k_geo.weight,
            k_weight,
            sem_dim,
            geo_dim,
        )

        # Initialize gate to balanced (0.5)
        if layer.decoupled_gate_logit is not None:
            layer.decoupled_gate_logit.data.zero_()  # sigmoid(0) = 0.5

    def _init_dba_projection_from_svd(
        self,
        sem_weight: Tensor,  # Parameter to fill [sem_dim, d_model]
        geo_weight: Tensor,  # Parameter to fill [geo_dim, d_model]
        teacher_weight: Tensor,  # [teacher_out, d_model]
        sem_dim: int,
        geo_dim: int,
    ) -> None:
        """Initialize semantic/geometric projections using SVD of teacher weights.

        The semantic path captures the top singular vectors (content routing).
        The geometric path captures the next set (positional structure).
        """
        # SVD: teacher_weight = U @ S @ Vh
        # U: [out, rank], S: [rank], Vh: [rank, d_model]
        # The rows of Vh are the right singular vectors
        # We want to project into these subspaces

        # Use truncated SVD for efficiency
        try:
            U, S, Vh = torch.linalg.svd(teacher_weight.float(), full_matrices=False)
        except RuntimeError:
            # Fallback to simple truncation if SVD fails
            sem_weight.data.copy_(teacher_weight[:sem_dim, :].to(sem_weight.dtype))
            geo_weight.data.copy_(teacher_weight[sem_dim:sem_dim + geo_dim, :].to(geo_weight.dtype))
            return

        # Reconstruct projection matrices from top singular vectors
        # Semantic: top sem_dim singular vectors
        # Geometric: next geo_dim singular vectors

        rank = min(S.size(0), sem_dim + geo_dim)
        sem_rank = min(sem_dim, rank)
        geo_rank = min(geo_dim, max(0, rank - sem_dim))

        # W_sem = U_sem @ S_sem @ Vh_sem (reconstructed from top components)
        # For a linear projection, we can use U[:, :sem_rank] @ diag(S[:sem_rank]) @ Vh[:sem_rank, :]
        # But we need output shape [sem_dim, d_model]

        # Simpler approach: use the top rows of the teacher (which correlate with top SVD directions)
        # This is a reasonable approximation for initialization

        if sem_rank > 0:
            # Use top SVD directions projected back
            sem_proj = (U[:, :sem_rank] @ torch.diag(S[:sem_rank]) @ Vh[:sem_rank, :])
            # Take first sem_dim rows
            copy_rows = min(sem_dim, sem_proj.size(0))
            sem_weight.data[:copy_rows, :].copy_(sem_proj[:copy_rows, :].to(sem_weight.dtype))

        if geo_rank > 0:
            # Use next SVD directions for geometric
            geo_start = sem_rank
            geo_end = sem_rank + geo_rank
            geo_proj = (U[:, geo_start:geo_end] @ torch.diag(S[geo_start:geo_end]) @ Vh[geo_start:geo_end, :])
            copy_rows = min(geo_dim, geo_proj.size(0))
            geo_weight.data[:copy_rows, :].copy_(geo_proj[:copy_rows, :].to(geo_weight.dtype))

    def load_mlp(self, layer: nn.Module, layer_prefix: str) -> None:
        """Load SwiGLU gate/up/down weights."""
        if not isinstance(layer, SwiGLULayer):
            return

        mlp_prefix = self.state.key(layer_prefix, "mlp")
        layer.w_gate.weight.data.copy_(self.state.get(self.state.key(mlp_prefix, "gate_proj", "weight")))
        layer.w_up.weight.data.copy_(self.state.get(self.state.key(mlp_prefix, "up_proj", "weight")))
        layer.w_down.weight.data.copy_(self.state.get(self.state.key(mlp_prefix, "down_proj", "weight")))

    def find_embedder(self) -> Embedder | None:
        """Find the first Embedder module."""
        for _, m in self.model.named_modules():
            if isinstance(m, Embedder):
                return m
        return None

    def find_head(self) -> LinearLayer | None:
        """Find the last LinearLayer module."""
        heads = [m for _, m in self.model.named_modules() if isinstance(m, LinearLayer)]
        return heads[-1] if heads else None
