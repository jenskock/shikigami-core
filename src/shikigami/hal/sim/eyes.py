"""Compatibility shim — procedural eyes now live in ``shikigami.expression.eyes``.

The sim dashboard still imports ``EyeAnimator`` / ``draw_eyes`` from here.
"""

from __future__ import annotations

from typing import Any

from shikigami.expression.eyes import EyesEngine
from shikigami.expression.eyes.render.cv2_renderer import Cv2Renderer
from shikigami.expression.eyes.state import EyesConfig, FrameState

# Back-compat names used by older tests / imports
EyePose = FrameState


class EyeAnimator:
    """Thin adapter matching the previous ``EyeAnimator.update(...)`` signature."""

    def __init__(self, cfg: EyesConfig | None = None) -> None:
        self._engine = EyesEngine(cfg or EyesConfig())
        self._renderer = Cv2Renderer()
        self.pose: FrameState = self._engine.frame

    def update(
        self,
        dt: float,
        emotion: str,
        gaze_x: float,
        gaze_y: float,
        speaking: bool,
    ) -> FrameState:
        self._engine.set_emotion(emotion)
        self._engine.look_at(gaze_x, gaze_y)
        self._engine.set_speaking(speaking)
        self.pose = self._engine.update(dt)
        return self.pose


def draw_eyes(cv2: Any, img: Any, pose: FrameState) -> None:
    """Draw current frame into a BGR panel (``cv2`` unused; kept for call-site compat)."""
    _ = cv2
    Cv2Renderer(img.shape[1], img.shape[0]).draw(pose, out_bgr=img)


def emotion_pose(
    emotion: str, gaze_x: float, gaze_y: float, speaking: bool, t: float
) -> FrameState:
    """Legacy helper: one-shot settle toward an emotion (for callers that peeked poses)."""
    eng = EyesEngine(EyesConfig(seed=0))
    eng.set_emotion(emotion)
    eng.look_at(gaze_x, gaze_y)
    eng.set_speaking(speaking)
    frame = eng.frame
    for _ in range(max(1, int(t * 60) + 20)):
        frame = eng.update(1 / 60)
    return frame
