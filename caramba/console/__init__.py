"""Rich, structured console output for caramba.

Machine learning code produces a lot of output: training progress, metrics,
errors, and results. This module provides beautiful, consistent terminal
output using Rich, with semantic log levels and structured data display.

Usage:
    from caramba.console import logger

    logger.info("Starting training...")
    logger.success("Training complete!")
    logger.warning("Low memory detected")
    logger.error("Failed to load checkpoint")

    # Structured output
    logger.header("Training Phase", "blockwise distillation")
    logger.metric("loss", 0.0234)
    logger.key_value({"epochs": 10, "lr": 0.001})

    # Progress tracking
    for i in logger.progress(100, "Processing"):
        do_work(i)
"""
from caramba.console.logger import Logger, get_logger

# Module-level singleton for convenient import
logger = get_logger()

__all__ = ["Logger", "get_logger", "logger"]
