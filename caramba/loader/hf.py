"""provides Hugging Face download utilities

This allows a frictionless way to take the user's intent and
convert it into an automated data pipeline.
"""
from __future__ import annotations

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


_DEFAULT_FILES = [
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "pytorch_model.bin",
]


class HFLoader:
    """Handles loading models directly from Hugging Face Hub"""
    def __init__(
        self,
        *,
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Set the class level attributes"""
        if not repo_id:
            raise ValueError("repo_id must be non-empty")

        self.repo_id: str = repo_id
        self.revision: str | None = revision
        self.cache_dir: str | None = cache_dir

    def load(self) -> Path:
        """Download model and return the local path."""
        parts = [p for p in self.repo_id.split("/") if p]

        if len(parts) < 2:
            raise ValueError(f"repo_id must include org/model, got {self.repo_id!r}")

        if len(parts) == 2:
            repo_id = "/".join(parts)
            filename: str | None = None
        else:
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])

        resolved_filename: str
        if filename is None:
            api = HfApi()
            try:
                repo_files = set(api.list_repo_files(repo_id=repo_id, revision=self.revision))
            except Exception as e:
                raise ValueError(
                    f"Failed to list files for repo {repo_id!r} "
                    f"(revision={self.revision!r}): {e}"
                ) from e
            resolved_filename = next((f for f in _DEFAULT_FILES if f in repo_files), "")
            if not resolved_filename:
                raise ValueError(f"No default model file found in {repo_id}")
        else:
            resolved_filename = filename

        try:
            return Path(hf_hub_download(
                repo_id=repo_id,
                filename=resolved_filename,
                revision=self.revision,
                cache_dir=self.cache_dir,
            ))
        except Exception as e:
            raise ValueError(
                f"Failed to download {resolved_filename!r} from {repo_id!r} "
                f"(revision={self.revision!r}, cache_dir={self.cache_dir!r}): {e}"
            ) from e
