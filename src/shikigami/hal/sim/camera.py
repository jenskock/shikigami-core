"""Sim camera = Mac webcam via OpenCV.

Annotates a BGR buffer and pushes it to the Qt dashboard. Face detection stays
on the OpenCV path; the window chrome is Qt.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import numpy as np

from ...concurrency import run_detached
from ..interfaces import Camera
from .dashboard import CAM_H, CAM_W, SimDashboard

DETECT_EVERY = 3
CAPTURE_W, CAPTURE_H = 640, 360
PREVIEW_EVERY = 2  # paint cam ~10 Hz at 20 Hz reflex — webcam read is the slow part

_PERM_HINT = (
    "on macOS grant Camera permission to Cursor (or your terminal app) in "
    "System Settings > Privacy & Security > Camera, then restart"
)


class SimCamera(Camera):
    def __init__(
        self,
        index: int = 0,
        preview: bool = False,
        dashboard: SimDashboard | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._index = index
        self._preview = preview
        self._dash = dashboard
        self._log = log or logging.getLogger("shikigami.sim.camera")
        self._cap: Any | None = None
        self._cv2: Any | None = None
        self._detector: Any | None = None
        self.last_detections: list = []
        self._tick = 0
        self._unavailable: str | None = None
        self.last_frame: Any | None = None
        self._preview_buf = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    def _open_capture(self):
        cv2 = self._cv2
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

    async def start(self) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = self._open_capture()
        if self._cap is None and sys.platform == "darwin":
            self._log.info(
                "[sim] camera: open failed — waiting 1.5s for macOS permission dialog…"
            )
            await asyncio.sleep(1.5)
            self._cap = self._open_capture()

        if self._cap is None:
            self._unavailable = f"no camera (index {self._index})"
            raise RuntimeError(f"cannot open webcam index {self._index} — {_PERM_HINT}")

        self._unavailable = None
        if self._preview:
            try:
                from ...perception.face import FaceDetector

                self._detector = FaceDetector()
            except Exception as exc:
                self._detector = None
                self._log.warning("detection unavailable (%s) — camera without boxes", exc)

    async def read_frame(self) -> Any | None:
        if self._cap is None:
            return None
        # Webcam read blocks ~30ms+ on the calling thread — keep it off the asyncio/Qt pump.
        ok, frame = await run_detached(self._cap.read)
        if ok:
            self.last_frame = frame
            return frame
        return None

    async def preview_frame(self) -> None:
        if not self._preview or self._dash is None or not self._dash.active or self._cv2 is None:
            return
        self._tick += 1
        if self._tick % PREVIEW_EVERY != 0:
            return
        out = self._preview_buf
        out[:] = 0
        if self._cap is None:
            msg = self._unavailable or "no camera"
            self._cv2.putText(
                out, msg, (10, 30), self._cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            self._dash.set_camera(out)
            return

        frame = await self.read_frame()
        if frame is None:
            self._cv2.putText(
                out, "no camera", (10, 30), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            self._dash.set_camera(out)
            return

        boxes = self.last_detections
        if self._detector is not None and (self._tick % (DETECT_EVERY * PREVIEW_EVERY) == 0):

            def _detect():
                return self._detector.detect_all(frame)

            boxes = await run_detached(_detect)
            self.last_detections = boxes
        self._draw_frame(frame, out, boxes)
        self._dash.set_camera(out)

    def _draw_frame(self, frame, out, boxes) -> None:
        cv2 = self._cv2
        ph, pw = out.shape[:2]
        fh, fw = frame.shape[:2]
        scale = min(pw / fw, ph / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        out[:] = 0
        oy, ox = (ph - nh) // 2, (pw - nw) // 2
        out[oy : oy + nh, ox : ox + nw] = resized
        for label, x, y, w, h in boxes:
            bx, by = ox + int(x * scale), oy + int(y * scale)
            bw, bh = int(w * scale), int(h * scale)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(
                out, label, (bx, max(14, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    async def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
