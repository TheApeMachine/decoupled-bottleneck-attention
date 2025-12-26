from __future__ import annotations

from caramba.config.embedder import TokenEmbedderConfig
from caramba.config.layer import AttentionLayerConfig, LayerType, SwiGLULayerConfig
from caramba.config.model import ModelConfig, ModelType
from caramba.config.topology import StackedTopologyConfig


def test_model_config_optimize_scales_common_transformer() -> None:
    cfg = ModelConfig(
        type=ModelType.GPT,
        embedder=TokenEmbedderConfig(vocab_size=32000, d_model=256),
        topology=StackedTopologyConfig(
            layers=[
                AttentionLayerConfig(type=LayerType.ATTENTION, d_model=256, n_heads=4),
                SwiGLULayerConfig(type=LayerType.SWIGLU, d_model=256, d_ff=1024),
            ],
            repeat=4,
        ),
        target_params=100_000_000,
    )
    opt = cfg.optimize()
    assert opt is not cfg
    assert isinstance(opt.embedder, TokenEmbedderConfig)
    assert opt.embedder.d_model % 64 == 0
    assert opt.topology.repeat >= 4
    # Verify the optimized config's total parameter count is close to target
    param_count_fn = getattr(opt, "parameter_count", None)
    if callable(param_count_fn) and cfg.target_params is not None:
        total_params: int = param_count_fn()  # type: ignore[assignment]
        target = cfg.target_params
        tolerance = 0.1  # 10% tolerance
        assert abs(total_params - target) / target < tolerance, (
            f"Optimized param count {total_params} not within {tolerance*100}% of target {target}"
        )