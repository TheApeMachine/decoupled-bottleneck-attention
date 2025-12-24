"""
layer provides layer modules.
"""
from __future__ import annotations

from torch import nn
from caramba.builder import Builder
from caramba.config.layer import (
    AttentionLayerConfig,
    DropoutLayerConfig,
    LayerConfig,
    LayerNormLayerConfig,
    LinearLayerConfig,
    MultiheadLayerConfig,
    RMSNormLayerConfig,
    SwiGLULayerConfig,
)
from caramba.config.topology import _TopologyConfigBase
from caramba.layer.attention import Attention
from caramba.layer.linear import Linear
from caramba.layer.normalize import Normalize
from caramba.layer.rms_norm import RMSNorm
from caramba.layer.swiglu import SwiGLU
from caramba.layer.multihead import Multihead
from caramba.layer.dropout import Dropout


class LayerBuilder(Builder[LayerConfig]):
    """
    LayerBuilder builds layer modules from config.
    """
    def build(self, config: LayerConfig) -> nn.Module:
        """
        build builds a layer module from config.
        """
        if isinstance(config, _TopologyConfigBase):
            raise ValueError(f"LayerBuilder only accepts LayerConfig, got {type(config)!r}")
        match config:
            case LinearLayerConfig() as c:
                out = Linear(c)
            case LayerNormLayerConfig() as c:
                out = Normalize(c)
            case RMSNormLayerConfig() as c:
                out = RMSNorm(c)
            case MultiheadLayerConfig() as c:
                out = Multihead(c)
            case DropoutLayerConfig() as c:
                out = Dropout(c)
            case AttentionLayerConfig() as c:
                out = Attention(c)
            case SwiGLULayerConfig() as c:
                out = SwiGLU(c)
            case _:
                layer_type = getattr(config, "type", type(config))
                raise ValueError(f"Unsupported layer type: {layer_type}")
        return out
