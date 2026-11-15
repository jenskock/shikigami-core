"""OpenCV/BGR adapter for the sim dashboard face panel."""

from __future__ import annotations

import numpy as np

from ..state import FrameState
from .numpy_draw import draw_frame_rgb, to_bgr


class Cv2Renderer:
    """Draws into an existing BGR ndarray (e.g. dashboard face panel) or returns one."""

    def __init__(self, width: int = 480, height: int = 560) -> None:
        self.width = width
        self.height = height
        self._rgb = np.zeros((height, width, 3), dtype=np.uint8)

    def draw(self, frame_state: FrameState, out_bgr: np.ndarray | None = None) -> np.ndarray:
        rgb = draw_frame_rgb(
            frame_state,
            self.width if out_bgr is None else out_bgr.shape[1],
            self.height if out_bgr is None else out_bgr.shape[0],
            circular=False,
            out=self._rgb
            if out_bgr is None or self._rgb.shape[:2] == (out_bgr.shape[0], out_bgr.shape[1])
            else None,
        )
        if out_bgr is not None:
            # in-place channel swap — avoid to_bgr()'s full-frame .copy()
            out_bgr[:] = rgb[:, :, ::-1]
            return out_bgr
        return to_bgr(rgb)
