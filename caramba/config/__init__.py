"""config provides configuration classes."""
from __future__ import annotations

import enum
import re
from typing import Annotated, Protocol, TypeVar, cast

from pydantic import AfterValidator, BaseModel

from torch import nn
import importlib


T = TypeVar("T")


class ValidationType(enum.Enum):
    """ValidationType provides the validation type."""
    SHOULD_BE_TRUE = "should_be_true"
    SHOULD_BE_FALSE = "should_be_false"
    SHOULD_BE_EQUAL_TO = "should_be_equal_to"
    SHOULD_BE_NOT_EQUAL_TO = "should_be_not_equal_to"
    SHOULD_BE_POSITIVE = "should_be_positive"
    SHOULD_BE_NON_NEGATIVE = "should_be_non_negative"


class Config(BaseModel):
    """Config provides the configuration class."""
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

    def build(self) -> nn.Module:
        """Creates an instance of a Config type and passes in the config.

        This intentionally uses dynamic imports so adding a new layer/topology is
        "add module + class" rather than updating a central registry.
        """
        class _BuildType(Protocol):
            value: str  # Enum value, e.g., "StackedTopology"
            name: str   # Enum key, e.g., "STACKED"

            # The concrete enums implement this as a @classmethod, but calling it on
            # the enum member works fine (and keeps this simple for the type checker).
            def module_name(self) -> str: ...

        t = cast(_BuildType, getattr(self, "type"))
        class_name = t.value  # e.g., "StackedTopology"
        module_name = t.name.lower()  # e.g., "STACKED" -> "stacked"
        mod = importlib.import_module(f"{t.module_name()}.{module_name}")
        cls = getattr(mod, class_name)
        return cls(self)

    @staticmethod
    def check(left: T, validation_type: ValidationType, right: T | None = None) -> T:
        """Provides a centralized validation function to reduce boilerplate."""
        match validation_type:
            case ValidationType.SHOULD_BE_TRUE:
                if not left:
                    raise ValueError("value failed to validate")
                return left
            case ValidationType.SHOULD_BE_FALSE:
                if left:
                    raise ValueError("value failed to validate")
                return left
            case ValidationType.SHOULD_BE_EQUAL_TO:
                if left != right:
                    raise ValueError("value failed to validate")
                return left
            case ValidationType.SHOULD_BE_NOT_EQUAL_TO:
                if left == right:
                    raise ValueError("value failed to validate")
                return left
            case ValidationType.SHOULD_BE_POSITIVE:
                # For numeric types, enforce strictly > 0.
                if left <= 0:  # type: ignore[operator]
                    raise ValueError("value failed to validate")
                return left
            case ValidationType.SHOULD_BE_NON_NEGATIVE:
                # For numeric types, enforce >= 0.
                if left < 0:  # type: ignore[operator]
                    raise ValueError("value failed to validate")
                return left
            case _:
                raise ValueError("value failed to validate")

    @staticmethod
    def check_range(
        value: float,
        *,
        ge: float | None = None,
        gt: float | None = None,
        le: float | None = None,
        lt: float | None = None,
    ) -> float:
        """
        check_range validates numeric ranges in one place.

        Use this to build common aliases like Probability.
        """
        v = float(value)
        if ge is not None and v < ge:
            raise ValueError("value failed to validate")
        if gt is not None and v <= gt:
            raise ValueError("value failed to validate")
        if le is not None and v > le:
            raise ValueError("value failed to validate")
        if lt is not None and v >= lt:
            raise ValueError("value failed to validate")
        return value


# Centralized “validated primitive” aliases (use these in Config models).
PositiveInt = Annotated[
    int, AfterValidator(
        lambda v: Config.check(
            v, ValidationType.SHOULD_BE_POSITIVE
        )
    )
]
NonNegativeInt = Annotated[
    int, AfterValidator(
        lambda v: Config.check(
            v, ValidationType.SHOULD_BE_NON_NEGATIVE
        )
    )
]
PositiveFloat = Annotated[
    float, AfterValidator(
        lambda v: Config.check(
            v, ValidationType.SHOULD_BE_POSITIVE
        )
    )
]
NonNegativeFloat = Annotated[
    float, AfterValidator(
        lambda v: Config.check(
            v, ValidationType.SHOULD_BE_NON_NEGATIVE
        )
    )
]
Probability = Annotated[
    float, AfterValidator(
        lambda v: Config.check_range(v, ge=0.0, le=1.0)
    )
]