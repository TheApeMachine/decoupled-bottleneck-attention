"""Module output tracing for distillation and analysis.

During distillation, we need to compare intermediate outputs between teacher
and student models—not just final logits. Trace hooks into PyTorch's forward
hooks to capture outputs from specific layers (identified by a predicate)
during a forward pass.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


class Trace:
    """Captures outputs from selected modules during forward passes.

    Used in blockwise distillation to compare teacher and student layer
    outputs. The predicate function identifies which modules to trace
    (typically attention layers).
    """

    def __init__(
        self,
        root: nn.Module,
        *,
        predicate: Callable[[str, nn.Module], bool],
    ) -> None:
        """Set up tracing for modules matching the predicate.

        Args:
            root: The model to trace
            predicate: Function (name, module) → bool that identifies
                      which modules should have their outputs captured
        """
        self._root = root
        self._predicate = predicate
        self._handles: list[RemovableHandle] = []
        self.outputs: list[Tensor] = []

    def attach(self) -> None:
        """Register forward hooks on all matching modules.

        After calling attach(), any forward pass through the model will
        capture outputs from the traced modules into self.outputs.
        """
        if self._handles:
            raise ValueError("Trace is already attached.")

        handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in self._iter_matches()
        ]
        if not handles:
            raise ValueError("Trace predicate matched no modules.")
        self._handles = handles

    def detach(self) -> None:
        """Remove all registered hooks.

        Call this when done tracing to avoid memory leaks and overhead.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        """Remove all captured outputs.

        Call this before each forward pass to start fresh.
        """
        self.outputs.clear()

    def _iter_matches(self) -> Iterable[tuple[str, nn.Module]]:
        """Yield all modules matching the predicate."""
        for name, module in self._root.named_modules():
            if self._predicate(name, module):
                yield name, module

    def _hook(
        self,
        name: str,
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        """Create a forward hook that captures module output.

        Handles both plain Tensor outputs and (Tensor, cache) tuples
        that attention layers return.
        """

        def _capture(_: nn.Module, __: tuple[object, ...], output: object) -> None:
            # Handle (output, cache) tuples from attention layers
            if isinstance(output, tuple) and len(output) >= 1:
                first = output[0]
                if isinstance(first, Tensor):
                    self.outputs.append(first)
                    return
                raise ValueError(
                    f"Trace expected Tensor as first tuple element from {name}, "
                    f"got {type(first)!r}"
                )

            if not isinstance(output, Tensor):
                raise ValueError(
                    f"Trace expected Tensor output from {name}, got {type(output)!r}"
                )
            self.outputs.append(output)

        return _capture

    def __enter__(self) -> "Trace":
        """Context manager entry: attach hooks."""
        self.attach()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        """Context manager exit: detach hooks."""
        self.detach()
