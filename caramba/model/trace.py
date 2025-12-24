"""
trace provides utilities for capturing module outputs.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


class Trace:
    """
    Trace captures outputs from matching modules during a forward pass.
    """
    def __init__(
        self,
        root: nn.Module,
        *,
        predicate: Callable[[str, nn.Module], bool],
    ) -> None:
        """
        __init__ initializes a trace for a module tree.
        """
        self._root: nn.Module = root
        self._predicate: Callable[[str, nn.Module], bool] = predicate
        self._handles: list[RemovableHandle] = []
        self.outputs: list[Tensor] = []

    def attach(self) -> None:
        """
        attach registers forward hooks on all matching modules.
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
        """
        detach removes all registered hooks.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        """
        clear removes all captured outputs.
        """
        self.outputs.clear()

    def _iter_matches(self) -> Iterable[tuple[str, nn.Module]]:
        """
        _iter_matches yields all named modules matching the predicate.
        """
        for name, module in self._root.named_modules():
            if self._predicate(name, module):
                yield name, module

    def _hook(
        self,
        name: str,
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        """
        _hook builds a forward hook that captures module outputs.
        """
        def _capture(_: nn.Module, __: tuple[object, ...], output: object) -> None:
            if not isinstance(output, Tensor):
                raise ValueError(
                    f"Trace expected Tensor output from {name}, got {type(output)!r}"
                )
            self.outputs.append(output)

        return _capture

    def __enter__(self) -> Trace:
        """
        __enter__ attaches hooks for a with-statement.
        """
        self.attach()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        """
        __exit__ detaches hooks for a with-statement.
        """
        self.detach()
