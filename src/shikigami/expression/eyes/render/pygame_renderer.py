"""Pygame renderer for the Mac eyes-demo preview window."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..state import FrameState
from .numpy_draw import draw_frame_rgb


class PygameRenderer:
    """Scalable preview; `draw` returns a pygame Surface."""

    def __init__(self, width: int = 480, height: int = 480, *, circular: bool = False) -> None:
        self.width = width
        self.height = height
        self.circular = circular
        self._buf = np.zeros((height, width, 3), dtype=np.uint8)
        self._pygame: Any | None = None
        self._surf: Any | None = None

    def _pg(self):
        if self._pygame is None:
            import pygame

            self._pygame = pygame
            if not pygame.get_init():
                pygame.init()
        return self._pygame

    def draw(self, frame_state: FrameState) -> Any:
        pg = self._pg()
        rgb = draw_frame_rgb(
            frame_state,
            self.width,
            self.height,
            circular=self.circular,
            out=self._buf,
        )
        # pygame wants a surface; make_surface expects (W,H,3) Fortran-ish — use surfarray
        if self._surf is None or self._surf.get_size() != (self.width, self.height):
            self._surf = pg.Surface((self.width, self.height))
        # surfarray.blit_array wants (w, h, 3)
        pg.surfarray.blit_array(self._surf, np.transpose(rgb, (1, 0, 2)))
        return self._surf
