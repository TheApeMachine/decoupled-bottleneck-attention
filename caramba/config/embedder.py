"""
embedder provides the embedder configuration.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from caramba.config import PositiveInt


class EmbedderType(str, enum.Enum):
    """
    EmbedderType provides the embedder type.
    """

    NONE = "none"
    TOKEN = "token"


class _EmbedderConfigBase(BaseModel):
    pass


class NoEmbedderConfig(_EmbedderConfigBase):
    """
    NoEmbedderConfig provides the no embedder configuration.
    """
    type: Literal[EmbedderType.NONE] = EmbedderType.NONE


class TokenEmbedderConfig(_EmbedderConfigBase):
    """
    TokenEmbedderConfig provides the token embedder configuration.
    """
    type: Literal[EmbedderType.TOKEN] = EmbedderType.TOKEN
    vocab_size: PositiveInt
    d_model: PositiveInt


EmbedderConfig: TypeAlias = Annotated[
    NoEmbedderConfig | TokenEmbedderConfig,
    Field(discriminator="type"),
]
