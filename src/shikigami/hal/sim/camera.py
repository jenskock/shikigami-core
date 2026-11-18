"""Webcam camera (Mac sim, or a USB cam on the Pi) via OpenCV.

Only produces frames. With ``camera.pi_profile`` (default) the frames are cropped
and resized to the Pi's streams and capped to the Pi's fps, so perception sees
exactly what it would see on the robot. Detection runs in ``perception``, not here.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from ..camera_base import StreamSpec, ThreadedCamera
from .dashboard import SimDashboard

CAPTURE_W, CAPTURE_H = 1280, 720  # webcams are 16:9; the 4:3 Pi view is cropped from this

_PERM_HINT = (
    "on macOS grant Camera permission to Cursor (or your terminal app) in "
    "System Settings > Privacy & Security > Camera, then restart"
)


class SimCamera(ThreadedCamera):
    paced = True  # cap.read() waits for the webcam

    def __init__(
        self,
        index: int = 0,
        preview: bool = False,
        dashboard: SimDashboard | None = None,
        log: logging.Logger | None = None,
        spec: StreamSpec | None = None,
    ) -> None:
        super().__init__(
            spec or StreamSpec(),
            dashboard=dashboard if preview else None,
            log=log or logging.getLogger("shikigami.sim.camera"),
        )
        self._index = index
        self._cap: Any | None = None

    def _open_capture(self):
        import cv2

        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        ok, _frame = cap.read()
        if not ok:
            cap.release()
            return None
        return cap

    def _open(self) -> None:
        # Open on the calling (main) thread: macOS shows the permission dialog there.
        self._cap = self._open_capture()
        if self._cap is None and sys.platform == "darwin":
            self._log.info("[sim] camera: open failed — waiting 1.5s for macOS permission dialog…")
            time.sleep(1.5)
            self._cap = self._open_capture()
        if self._cap is None:
            self._unavailable = f"no camera (index {self._index})"
            raise RuntimeError(f"cannot open webcam index {self._index} — {_PERM_HINT}")
        h, w = self._cap.read()[1].shape[:2]
        self._log.info(
            "camera: webcam %d at %dx%d -> main %s, lores %s, %s fps%s",
            self._index,
            w,
            h,
            self.spec.main_size or "native",
            self.spec.lores_size,
            self.spec.fps or "uncapped",
            "" if self.spec.main_size is None else " (Pi profile)",
        )

    def _grab(self):
        cap = self._cap
        if cap is None:
            return None
        ok, frame = cap.read()
        return (frame, None) if ok else None

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
