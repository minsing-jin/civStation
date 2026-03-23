"""Utilities for saving a capped screenshot trajectory per run inside the project."""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

from computer_use_test.utils.project_runtime import get_project_runtime_root

_TRAJECTORY_DIRNAME = "screenshot_trajectories"
_DEFAULT_MAX_IMAGES = 20
_LABEL_SANITIZER_RE = re.compile(r"[^a-z0-9_-]+")
_active_session = None
_active_lock = threading.RLock()


def _sanitize_label(label: str) -> str:
    cleaned = _LABEL_SANITIZER_RE.sub("_", str(label).strip().lower()).strip("_")
    return cleaned or "capture"


def get_screenshot_trajectory_root(base_dir: Path | str | None = None) -> Path:
    root = get_project_runtime_root(base_dir=base_dir)
    return root / _TRAJECTORY_DIRNAME


class ScreenshotTrajectorySession:
    """Own one project-local directory containing the latest capped screenshot trajectory."""

    def __init__(self, path: Path, max_images: int = _DEFAULT_MAX_IMAGES):
        self.path = path
        self.max_images = max(1, int(max_images))
        self._lock = threading.RLock()
        self._closed = False
        self._capture_index = 0
        self._saved_paths: deque[Path] = deque()
        self.path.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def record(self, pil_image, label: str = "capture") -> Path | None:
        with self._lock:
            if self._closed:
                return None

            self._capture_index += 1
            filename = f"{self._capture_index:04d}_{_sanitize_label(label)}.png"
            output_path = self.path / filename
            image = pil_image.convert("RGB") if getattr(pil_image, "mode", "RGB") != "RGB" else pil_image
            image.save(output_path, format="PNG")
            self._saved_paths.append(output_path)

            while len(self._saved_paths) > self.max_images:
                oldest = self._saved_paths.popleft()
                oldest.unlink(missing_ok=True)

            self._write_manifest()
            return output_path

    def close(self) -> None:
        global _active_session  # noqa: PLW0603

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._write_manifest()

        with _active_lock:
            if _active_session is self:
                _active_session = None

    def _write_manifest(self) -> None:
        manifest = {
            "max_images": self.max_images,
            "captured_images": self._capture_index,
            "saved_images": len(self._saved_paths),
            "files": [path.name for path in self._saved_paths],
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def start_screenshot_trajectory_session(
    base_dir: Path | str | None = None,
    max_images: int = _DEFAULT_MAX_IMAGES,
) -> ScreenshotTrajectorySession:
    global _active_session  # noqa: PLW0603

    root = get_screenshot_trajectory_root(base_dir=base_dir)
    path = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = ScreenshotTrajectorySession(path=path, max_images=max_images)
    with _active_lock:
        _active_session = session
    return session


def record_screenshot_trajectory(pil_image, label: str = "capture") -> Path | None:
    with _active_lock:
        session = _active_session
    if session is None:
        return None
    return session.record(pil_image, label=label)
