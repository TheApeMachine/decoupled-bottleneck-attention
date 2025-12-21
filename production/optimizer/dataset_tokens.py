"""Dataset token-count inference helpers."""

from __future__ import annotations

import os
import re

from production.optimizer.counts import CountCodec


class DatasetTokenCounter:
    """Infer dataset token counts from filename, `.meta`, or config helpers."""

    _DATASET_COUNT_RE: re.Pattern[str] = re.compile(r"(\d+)([bm])", flags=re.IGNORECASE)

    @staticmethod
    def _read_tokens_from_meta(data_path: str) -> int | None:
        """Read `tokens: ...` from sibling `.meta` file if present."""
        try:
            p = str(data_path)
            meta_path = p + ".meta"
            if not os.path.exists(meta_path):
                return None
            with open(meta_path, "r", encoding="utf-8") as f:
                raw = f.read()
            for line in raw.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                if k.strip().lower() == "tokens":
                    return CountCodec.parse(v.strip())
        except (OSError, IOError, UnicodeDecodeError, ValueError, TypeError):
            return None
        return None

    @classmethod
    def infer(cls, data_path: str | None) -> int | None:
        if not data_path:
            return None
        try:
            from production.config import infer_dataset_tokens_from_path  # pylint: disable=import-outside-toplevel

            inferred = infer_dataset_tokens_from_path(str(data_path))
        except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError):
            inferred = None
        if inferred is not None:
            return int(inferred)
        return cls._read_tokens_from_meta(str(data_path))

    @classmethod
    def infer_with_source(cls, data_path: str | None) -> tuple[int | None, str]:
        if not data_path:
            return None, ""
        try:
            base = os.path.basename(str(data_path)).lower()
        except (OSError, TypeError):
            base = ""
        m = cls._DATASET_COUNT_RE.search(base)
        if m:
            try:
                k = int(m.group(1))
                suf = str(m.group(2)).lower()
                if suf == "m":
                    return int(k * 1_000_000), "filename"
                if suf == "b":
                    return int(k * 1_000_000_000), "filename"
            except (ValueError, TypeError, OverflowError):
                pass
        meta = cls._read_tokens_from_meta(str(data_path))
        if meta is not None:
            return int(meta), "meta"
        inferred = cls.infer(str(data_path))
        inferred_int = int(inferred) if inferred is not None else None
        source = "config" if inferred is not None else ""
        return inferred_int, source
