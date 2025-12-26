"""Hugging Face Hub integration for downloading model checkpoints.

Instead of manually downloading and managing checkpoint files, you can
reference models by their Hub ID (e.g., "meta-llama/Llama-3.2-1B") and
this module handles the download, caching, and path resolution.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Files we look for in order of preference
_DEFAULT_FILES = [
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "pytorch_model.bin",
]


class HFLoader:
    """Downloads model files from Hugging Face Hub.

    Handles both simple downloads (just the repo ID) and specific file
    paths within a repo. Files are cached locally so subsequent loads
    don't require re-downloading.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Configure the loader for a specific repo.

        Args:
            repo_id: Hub repo ID like "org/model" or "org/model/path/to/file"
            revision: Git ref (branch, tag, commit) to download from
            cache_dir: Local directory for caching downloads
        """
        if not repo_id:
            raise ValueError("repo_id must be non-empty")

        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir

    def load(self) -> Path:
        """Download the model and return the local path.

        If repo_id includes a file path (org/model/path/to/file.safetensors),
        downloads that specific file. Otherwise, looks for a default model
        file in the repo.
        """
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
            # No specific file requested—find a default model file
            api = HfApi()
            try:
                repo_files = set(
                    api.list_repo_files(repo_id=repo_id, revision=self.revision)
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to list files for repo {repo_id!r} "
                    f"(revision={self.revision!r}): {e}"
                ) from e

            resolved_filename = next(
                (f for f in _DEFAULT_FILES if f in repo_files), ""
            )
            if not resolved_filename:
                raise ValueError(f"No default model file found in {repo_id}")
        else:
            resolved_filename = filename

        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=resolved_filename,
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                )
            )
        except Exception as e:
            raise ValueError(
                f"Failed to download {resolved_filename!r} from {repo_id!r} "
                f"(revision={self.revision!r}, cache_dir={self.cache_dir!r}): {e}"
            ) from e
