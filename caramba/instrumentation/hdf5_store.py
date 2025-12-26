"""Optional HDF5 storage for dense arrays.

JSONL is great for scalars, but dense arrays (activations, gradients, weights)
can be too large to reasonably embed in JSON. HDF5 provides a compact, indexed
format for storing these arrays alongside scalar logs.

This module is dependency-gated:
- If `h5py` is not installed, imports still succeed and the store disables itself.
- Failures are best-effort: training must never crash due to logging.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class _H5Group(Protocol):
    def create_dataset(self, name: str, *, data: object, **kwargs: object) -> object: ...

    def require_group(self, name: str) -> "_H5Group": ...


@runtime_checkable
class _H5File(Protocol):
    def require_group(self, name: str) -> _H5Group: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class _H5pyModule(Protocol):
    def File(self, filename: str, mode: str) -> _H5File: ...


def _try_import_h5py() -> _H5pyModule | None:
    try:
        if importlib.util.find_spec("h5py") is None:
            return None
        _h5py = importlib.import_module("h5py")
    except (ImportError, ModuleNotFoundError):
        return None
    return _h5py if isinstance(_h5py, _H5pyModule) else None


def _to_numpy(x: object) -> object:
    """Best-effort conversion of tensors to CPU numpy arrays without torch import."""

    # torch.Tensor: has detach/cpu/numpy
    try:
        detach = getattr(x, "detach", None)
        if callable(detach):
            # Call methods on the tensor in correct order: x.detach().cpu().numpy()
            x_detached = detach()
            cpu_fn = getattr(x_detached, "cpu", None)
            if callable(cpu_fn):
                x_cpu = cpu_fn()
                numpy_fn = getattr(x_cpu, "numpy", None)
                if callable(numpy_fn):
                    return numpy_fn()
    except Exception:
        pass

    # numpy arrays are already fine.
    return x


@dataclass
class H5Store:
    """Append-only HDF5 store with a simple hierarchical layout.

    Layout:
      /steps/<step>/<name> : dataset for each tensor/array
    """

    path: Path
    enabled: bool = True
    compression: str | None = "gzip"
    compression_opts: int | None = 4

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._h5py: _H5pyModule | None = None
        self._fh: _H5File | None = None

        if not self.enabled:
            return

        h5py_mod = _try_import_h5py()
        if h5py_mod is None:
            self.enabled = False
            return

        self._h5py = h5py_mod
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.enabled = False
            return

        try:
            self._fh = self._h5py.File(str(self.path), "a")
        except Exception:
            self.enabled = False
            self._fh = None

    def write_step(self, step: int, arrays: dict[str, object]) -> None:
        """Write named arrays under /steps/<step>/.

        Overwrites are avoided by default (if the dataset exists, we skip it).
        """

        if not self.enabled or self._fh is None:
            return

        try:
            root = self._fh.require_group("steps")
            g_step = root.require_group(str(int(step)))
            for name, value in arrays.items():
                key = str(name)
                # Skip if dataset already exists (best-effort, avoid churn).
                try:
                    existing = getattr(g_step, "get", None)
                    if callable(existing) and existing(key) is not None:
                        continue
                except Exception:
                    pass

                data = _to_numpy(value)
                kwargs: dict[str, Any] = {}
                if self.compression:
                    kwargs["compression"] = self.compression
                if self.compression_opts is not None:
                    kwargs["compression_opts"] = int(self.compression_opts)
                _ = g_step.create_dataset(key, data=data, **kwargs)

            try:
                self._fh.flush()
            except Exception:
                pass
        except Exception:
            # Disable after failure so training doesn't keep throwing.
            self.enabled = False
            try:
                self.close()
            except Exception:
                pass

    def close(self) -> None:
        """Close the underlying HDF5 file."""

        if self._fh is None:
            return
        try:
            self._fh.flush()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None

