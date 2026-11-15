"""PIL renderer → 240×240 RGB (and RGB565) for GC9A01 round SPI displays."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..state import FrameState
from .numpy_draw import draw_frame_rgb, to_rgb565


class PILRenderer:
    """Produces RGB images (and optional RGB565 bytes) with a circular mask."""

    def __init__(self, size: int = 240, *, circular: bool = True) -> None:
        self.size = size
        self.circular = circular
        self._buf = np.zeros((size, size, 3), dtype=np.uint8)
        self._Image: Any | None = None

    def _pil(self):
        if self._Image is None:
            from PIL import Image

            self._Image = Image
        return self._Image

    def draw(self, frame_state: FrameState) -> Any:
        rgb = draw_frame_rgb(
            frame_state,
            self.size,
            self.size,
            circular=self.circular,
            out=self._buf,
        )
        return self._pil().fromarray(rgb, mode="RGB")

    def draw_rgb565(self, frame_state: FrameState) -> bytes:
        rgb = draw_frame_rgb(
            frame_state,
            self.size,
            self.size,
            circular=self.circular,
            out=self._buf,
        )
        return to_rgb565(rgb)

    def draw_array(self, frame_state: FrameState) -> np.ndarray:
        return draw_frame_rgb(
            frame_state,
            self.size,
            self.size,
            circular=self.circular,
            out=self._buf,
        )
