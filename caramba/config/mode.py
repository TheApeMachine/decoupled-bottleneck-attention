"""Run modes: what kind of execution to perform.

A run can train the model, generate samples for evaluation, or
run an interactive chat session.
"""
from __future__ import annotations

import enum


class Mode(enum.Enum):
    """What kind of execution to perform.

    TRAIN: Run the training loop
    SAMPLE: Generate text samples for evaluation
    CHAT: Interactive chat session
    """

    TRAIN = "train"
    SAMPLE = "sample"
    CHAT = "chat"
