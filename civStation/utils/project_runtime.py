"""Project-local runtime paths for ephemeral turn-runner artifacts."""

from __future__ import annotations

from pathlib import Path

_RUNTIME_DIRNAME = ".tmp"
_CACHE_DIRNAME = "civStation"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_runtime_root(base_dir: Path | str | None = None) -> Path:
    """Return the project-local root for ephemeral runtime artifacts."""
    if base_dir is not None:
        return Path(base_dir)
    return _PROJECT_ROOT / _RUNTIME_DIRNAME / _CACHE_DIRNAME
