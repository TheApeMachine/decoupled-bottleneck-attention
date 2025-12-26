"""Configuration system: turning YAML into validated Python objects.

All model architectures, training settings, and benchmark configurations
are defined in YAML files and validated into Pydantic models. This keeps
configuration separate from code while ensuring type safety and clear
error messages when something is wrong.
"""
from __future__ import annotations

import enum
import importlib
from typing import Annotated, Protocol, TypeVar, cast

from pydantic import AfterValidator, BaseModel
from torch import nn


T = TypeVar("T")


class ValidationType(enum.Enum):
    """Types of value validation we support."""

    SHOULD_BE_TRUE = "should_be_true"
    SHOULD_BE_FALSE = "should_be_false"
    SHOULD_BE_EQUAL_TO = "should_be_equal_to"
    SHOULD_BE_NOT_EQUAL_TO = "should_be_not_equal_to"
    SHOULD_BE_POSITIVE = "should_be_positive"
    SHOULD_BE_NON_NEGATIVE = "should_be_non_negative"


class Config(BaseModel):
    """Base class for all configuration objects.

    Provides a `build()` method that dynamically constructs the nn.Module
    corresponding to this config, and validation helpers for enforcing
    constraints on config values.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

    def build(self) -> nn.Module:
        """Construct the nn.Module this config describes.

        Uses dynamic imports based on the config's `type` field, so adding
        a new layer type only requires adding the module—no central registry.
        """

        class _BuildType(Protocol):
            value: str
            name: str

            def module_name(self) -> str:
                ...

        t = cast(_BuildType, getattr(self, "type"))
        class_name = t.value
        module_name = t.name.lower()
        mod = importlib.import_module(f"{t.module_name()}.{module_name}")
        cls = getattr(mod, class_name)
        return cls(self)

    @staticmethod
    def check(left: T, validation_type: ValidationType, right: T | None = None) -> T:
        """Validate a value against a constraint, raising ValueError on failure."""
        match validation_type:
            case ValidationType.SHOULD_BE_TRUE:
                if not left:
                    raise ValueError(
                        f"Validation failed: {validation_type.name}: "
                        f"value={left!r} is not truthy"
                    )
                return left
            case ValidationType.SHOULD_BE_FALSE:
                if left:
                    raise ValueError(
                        f"Validation failed: {validation_type.name}: "
                        f"value={left!r} is not falsy"
                    )
                return left
            case ValidationType.SHOULD_BE_EQUAL_TO:
                if left != right:
                    raise ValueError(
                        f"Validation failed: {validation_type.name}: "
                        f"{left!r} != {right!r}"
                    )
                return left
            case ValidationType.SHOULD_BE_NOT_EQUAL_TO:
                if left == right:
                    raise ValueError(
                        f"Validation failed: {validation_type.name}: "
                        f"{left!r} == {right!r}"
                    )
                return left
            case ValidationType.SHOULD_BE_POSITIVE:
                if left <= 0:  # type: ignore[operator]
                    raise ValueError(
                        f"Validation failed: {validation_type.name}: {left!r} <= 0"
                    )
                return left
            case ValidationType.SHOULD_BE_NON_NEGATIVE:
                if left < 0:  # type: ignore[operator]
                    raise ValueError(
                        f"Validation failed: {validation_type.name}: {left!r} < 0"
                    )
                return left
            case _:
                raise ValueError(
                    f"Validation failed: unknown validation type {validation_type}"
                )

    @staticmethod
    def check_range(
        value: float,
        *,
        ge: float | None = None,
        gt: float | None = None,
        le: float | None = None,
        lt: float | None = None,
    ) -> float:
        """Validate a number is within a range."""
        v = float(value)
        if ge is not None and v < ge:
            raise ValueError(f"Validation failed: {v} < {ge} (expected >= {ge})")
        if gt is not None and v <= gt:
            raise ValueError(f"Validation failed: {v} <= {gt} (expected > {gt})")
        if le is not None and v > le:
            raise ValueError(f"Validation failed: {v} > {le} (expected <= {le})")
        if lt is not None and v >= lt:
            raise ValueError(f"Validation failed: {v} >= {lt} (expected < {lt})")
        return v


# Type aliases for validated primitives—use these in config models
PositiveInt = Annotated[
    int,
    AfterValidator(lambda v: Config.check(v, ValidationType.SHOULD_BE_POSITIVE)),
]
NonNegativeInt = Annotated[
    int,
    AfterValidator(lambda v: Config.check(v, ValidationType.SHOULD_BE_NON_NEGATIVE)),
]
PositiveFloat = Annotated[
    float,
    AfterValidator(lambda v: Config.check(v, ValidationType.SHOULD_BE_POSITIVE)),
]
NonNegativeFloat = Annotated[
    float,
    AfterValidator(lambda v: Config.check(v, ValidationType.SHOULD_BE_NON_NEGATIVE)),
]
Probability = Annotated[
    float,
    AfterValidator(lambda v: Config.check_range(v, ge=0.0, le=1.0)),
]
