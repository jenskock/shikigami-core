"""Renderer interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..state import FrameState


@runtime_checkable
class EyeRenderer(Protocol):
    def draw(self, frame_state: FrameState) -> Any:
        """Render a frame; return type is backend-specific (Surface, Image, ndarray…)."""
        ...
