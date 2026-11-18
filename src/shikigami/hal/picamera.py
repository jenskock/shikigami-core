"""Raspberry Pi camera via picamera2 (libcamera). Shared by the pidog and picrawler backends.

Two hardware streams from one sensor: ``main`` (RGB888, which picamera2 lays out as
BGR — OpenCV order) for snapshots and ``lores`` (YUV420, the only lores format on a
Pi 4) for detection. The frame rate is set in the sensor, so frames arrive at the Pi
rate the Mac emulation copies.

picamera2 is a Raspberry Pi OS system package (``sudo apt install python3-picamera2``);
the venv needs ``--system-site-packages`` to see it. Imported lazily — never on the Mac.
"""

from __future__ import annotations

import logging
from typing import Any

from .camera_base import StreamSpec, ThreadedCamera

INSTALL_HINT = (
    "picamera2 not available — on Raspberry Pi OS: sudo apt install python3-picamera2, "
    "and create the venv with --system-site-packages (or set camera.backend: webcam)"
)


def picamera2_available() -> bool:
    try:
        import picamera2  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


class PiCamera(ThreadedCamera):
    paced = True  # capture_request() waits for the sensor

    def __init__(self, spec: StreamSpec, log: logging.Logger | None = None) -> None:
        super().__init__(spec, dashboard=None, log=log or logging.getLogger("shikigami.picamera"))
        self._cam: Any | None = None

    def _open(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as exc:  # pragma: no cover - Pi only
            raise RuntimeError(f"{INSTALL_HINT} ({exc})") from exc
        main_size = self.spec.main_size or (1280, 960)
        cam = Picamera2()  # pragma: no cover - Pi only
        config = cam.create_video_configuration(
            main={"size": tuple(main_size), "format": "RGB888"},
            lores={"size": tuple(self.spec.lores_size), "format": "YUV420"},
            controls={"FrameRate": float(self.spec.fps or 30.0)},
            buffer_count=4,
        )
        cam.configure(config)
        cam.start()
        self._cam = cam
        self._log.info(
            "camera: picamera2 main %s, lores %s, %s fps",
            main_size,
            self.spec.lores_size,
            self.spec.fps,
        )

    def _grab(self):  # pragma: no cover - Pi only
        import cv2

        cam = self._cam
        if cam is None:
            return None
        req = cam.capture_request()
        try:
            main = req.make_array("main")
            yuv = req.make_array("lores")
        finally:
            req.release()
        lw, lh = self.spec.lores_size
        lores = cv2.cvtColor(yuv[: lh * 3 // 2, :lw], cv2.COLOR_YUV2BGR_I420)
        return main, lores

    def _close(self) -> None:  # pragma: no cover - Pi only
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            finally:
                self._cam = None
