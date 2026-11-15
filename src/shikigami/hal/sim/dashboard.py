"""Sim dashboard: ONE window composed of panels — face | stats | camera | spare.

Panels are just rects carved out of a shared canvas; add new parts by adding a
rect and drawing into it. Falls back to log-only when opencv/display is missing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

WINDOW_NAME = "shikigami"
WHITE = (255, 255, 255)


class SimDashboard:
    WIDTH = 1400
    HEIGHT = 560
    FACE_RECT = (0, 0, 480, 560)  # x, y, w, h — the animated face
    STATS_RECT = (490, 10, 300, 300)  # state / mood / motion / heard / latency / input
    SPARE_RECT = (490, 320, 300, 230)  # reserved for future panels
    CAM_RECT = (800, 10, 590, 540)  # webcam feed with detection boxes (big, letterboxed)

    def __init__(self, enabled: bool, log: logging.Logger | None = None) -> None:
        self._log = log or logging.getLogger("shikigami.sim.dashboard")
        self._enabled = enabled
        self._cv2: Any | None = None
        self.canvas: Any | None = None
        if enabled:
            try:
                import cv2
                import numpy as np

                self._cv2 = cv2
                self.canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
            except ImportError:
                self._enabled = False
                self._log.warning(
                    "opencv not installed — running sim without the dashboard window "
                    "(`uv sync --extra sim`)"
                )
        self._closed = False
        # in-window text input (the OpenCV window often has keyboard focus on macOS)
        self.input_buffer = ""
        self.on_submit: Callable[[str], None] | None = None

    @property
    def cv2(self) -> Any | None:
        return self._cv2

    @property
    def active(self) -> bool:
        """True when there is a real canvas + display to draw on (False = log-only)."""
        return self._enabled and self.canvas is not None

    def panel(self, rect: tuple[int, int, int, int]) -> Any:
        x, y, w, h = rect
        return self.canvas[y : y + h, x : x + w]

    async def start(self) -> None:
        if self._enabled and self._cv2 is not None:
            try:
                self._cv2.namedWindow(WINDOW_NAME, self._cv2.WINDOW_NORMAL)
            except Exception as exc:  # headless / no display
                self._enabled = False
                self._log.warning("cannot open dashboard window (%s) — log-only mode", exc)

    def show(self) -> None:
        if self._enabled and self._cv2 is not None:
            self._cv2.imshow(WINDOW_NAME, self.canvas)
            self._handle_key(self._cv2.waitKey(1))

    def _handle_key(self, key: int) -> None:
        """Capture typed lines from the window itself (Enter submits, Backspace edits)."""
        if key < 0:
            return
        if key in (13, 10):  # Enter
            line = self.input_buffer.strip()
            self.input_buffer = ""
            if line and self.on_submit is not None:
                self.on_submit(line)
        elif key in (8, 127):  # Backspace
            self.input_buffer = self.input_buffer[:-1]
        elif 32 <= key < 127:  # printable ASCII
            self.input_buffer += chr(key)

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._enabled and self._cv2 is not None:
            with contextlib.suppress(Exception):
                self._cv2.destroyWindow(WINDOW_NAME)
