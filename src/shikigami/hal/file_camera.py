"""Replay camera: a video file or a folder of images, looped, through the Pi stream contract.

For repeatable tests and for re-running footage recorded on the Pi:
``SHIKIGAMI_CAMERA_FILE=clip.mp4 uv run sim`` or ``--camera-file clip.mp4``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .camera_base import StreamSpec, ThreadedCamera

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FileCamera(ThreadedCamera):
    paced = False  # files read instantly; the base sleeps to hold the fps

    def __init__(
        self,
        path: str | Path,
        spec: StreamSpec | None = None,
        dashboard: Any | None = None,
        log: logging.Logger | None = None,
        loop: bool = True,
    ) -> None:
        super().__init__(
            spec or StreamSpec(),
            dashboard=dashboard,
            log=log or logging.getLogger("shikigami.file"),
        )
        self._path = Path(path)
        self._loop = loop
        self._images: list[Path] = []
        self._idx = 0
        self._cap: Any | None = None
        self.exhausted = False

    def _open(self) -> None:
        import cv2

        p = self._path
        if p.is_dir():
            self._images = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if not self._images:
                raise RuntimeError(f"no images in {p}")
        elif p.is_file():
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video {p}")
            self._cap = cap
        else:
            raise RuntimeError(f"camera file not found: {p}")
        self._log.info("camera: replaying %s (loop=%s)", p, self._loop)

    def _grab(self):
        import cv2

        if self.exhausted:
            return None
        if self._images:
            if self._idx >= len(self._images):
                if not self._loop:
                    self.exhausted = True
                    return None
                self._idx = 0
            img = cv2.imread(str(self._images[self._idx]))
            self._idx += 1
            return (img, None) if img is not None else None
        cap = self._cap
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok:
            if not self._loop:
                self.exhausted = True
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return (frame, None) if ok else None

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
