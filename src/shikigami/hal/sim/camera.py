"""Sim camera = the Mac webcam, via OpenCV (lazy import).

With preview=True it draws the annotated feed (faces, cats) into the camera
panel of the shared dashboard window.
"""

from __future__ import annotations

import logging
from typing import Any

from ..interfaces import Camera
from .dashboard import SimDashboard

DETECT_EVERY = 4  # reflex ticks between detections (20 Hz loop -> 5 Hz, on the full-res frame)


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
        self._boxes: list | None = None
        self._tick = 0

    async def start(self) -> None:
        import cv2  # lazy: only needed when the camera is actually used

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"cannot open webcam index {self._index}")
        if self._preview:
            try:
                from ...perception.face import FaceDetector

                self._detector = FaceDetector()
            except Exception as exc:  # detection is optional; the raw feed still shows
                self._detector = None
                self._log.warning("detection unavailable (%s) — camera panel without boxes", exc)

    async def read_frame(self) -> Any | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    async def preview_frame(self) -> None:
        if (
            not self._preview
            or self._cap is None
            or self._dash is None
            or not self._dash.active
            or self._cv2 is None
        ):
            return
        frame = await self.read_frame()
        cam = self._dash.panel(SimDashboard.CAM_RECT)
        cam[:] = 0
        if frame is None:
            self._cv2.putText(
                cam, "no camera", (10, 30), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
        else:
            self._tick += 1
            if self._detector is not None and (
                self._boxes is None or self._tick % DETECT_EVERY == 0
            ):
                self._boxes = self._detector.detect_all(frame)  # full resolution
            self._draw_frame(frame, cam, self._boxes or [])
        self._dash.show()

    def _draw_frame(self, frame, out, boxes) -> None:
        """Aspect-preserving fit (letterbox) of the frame into the cam panel + scaled boxes."""
        cv2 = self._cv2
        ph, pw = out.shape[:2]
        fh, fw = frame.shape[:2]
        scale = min(pw / fw, ph / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = cv2.resize(frame, (nw, nh))
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
        # the window itself is owned by the dashboard (closed via expression.stop)
