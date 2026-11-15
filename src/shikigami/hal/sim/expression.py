"""Sim expression: draw device face buffer; Qt dashboard displays it 1:1.

Motion detail is local (``shikigami.expression.eyes``); the LLM only sends emote names.
Stats/chrome live in the Qt dashboard — not OpenCV putText.
"""

from __future__ import annotations

import logging

from shikigami.expression.eyes import EMOTION_NAMES, EyesEngine
from shikigami.expression.eyes.render.cv2_renderer import Cv2Renderer
from shikigami.expression.eyes.state import EyesConfig

from ..interfaces import Expression
from .dashboard import FACE_H, FACE_W, SimDashboard
from .state import SimState

EMOTIONS = tuple(EMOTION_NAMES) + ("talking",)


class SimExpression(Expression):
    def __init__(
        self,
        window: bool,
        state: SimState,
        dashboard: SimDashboard | None,
        log: logging.Logger | None = None,
        eyes_cfg: EyesConfig | None = None,
    ) -> None:
        self._window = window
        self._state = state
        self._dash = dashboard
        self._log = log or logging.getLogger("shikigami.sim.face")
        self._emotion = "neutral"
        self._speaking = False
        self._eyes = EyesEngine(eyes_cfg or EyesConfig())
        self._renderer = Cv2Renderer(FACE_W, FACE_H)

    async def start(self) -> None:
        if self._dash is not None:
            self._dash.state = self._state
            await self._dash.start()
            self._log.info("[sim] dashboard window opened")

    async def update(self, dt_s: float) -> None:
        self._eyes.set_speaking(self._speaking)
        self._eyes.look_at(self._state.gaze_x, self._state.gaze_y)
        self._eyes.update(dt_s)
        if self._dash is None:
            return
        self._dash.emotion = self._emotion
        if not self._window:
            return
        face = self._dash.face_buf
        face[:] = 0
        self._renderer.draw(self._eyes.frame, out_bgr=face)
        self._dash.mark_face_dirty()

    async def emote(self, name: str) -> None:
        key = name.lower().strip()
        if key == "talking":
            self._speaking = True
            self._emotion = "talking"
        elif key in EMOTION_NAMES:
            self._emotion = key
            self._eyes.set_emotion(key)
        else:
            self._emotion = "neutral"
            self._eyes.set_emotion("neutral")
        self._log.info("[sim] face: %s", self._emotion)

    async def set_mood(self, mood: str) -> None:
        self._state.mood = mood

    async def speak_visual(self, active: bool) -> None:
        self._speaking = active
        self._eyes.set_speaking(active)

    async def stop(self) -> None:
        if self._dash is not None:
            await self._dash.stop()
