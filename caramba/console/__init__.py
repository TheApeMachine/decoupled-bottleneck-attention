"""
console provides rich, structured console output for caramba.

This module exposes a singleton Logger instance for consistent,
beautiful terminal output throughout the codebase.

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
