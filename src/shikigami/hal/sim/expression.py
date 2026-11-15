"""Sim expression: the face panel of the dashboard window (strictly white on black).

Without a dashboard (headless/tests) it only logs.
"""

from __future__ import annotations

import logging
import time

from ..interfaces import Expression
from .dashboard import SimDashboard
from .state import SimState

EMOTIONS = ("neutral", "happy", "curious", "surprised", "sad", "sleepy", "talking")


class SimExpression(Expression):
    def __init__(
        self,
        window: bool,
        state: SimState,
        dashboard: SimDashboard | None,
        log: logging.Logger | None = None,
    ) -> None:
        self._window = window
        self._state = state
        self._dash = dashboard
        self._log = log or logging.getLogger("shikigami.sim.face")
        self._t = 0.0
        self._blink_remaining = 0.0
        self._next_blink = time.monotonic() + 2.5
        self._emotion = "neutral"
        self._speaking = False

    async def start(self) -> None:
        if self._dash is not None:
            await self._dash.start()
            self._log.info("[sim] dashboard window opened")

    async def update(self, dt_s: float) -> None:
        self._t += dt_s
        now = time.monotonic()
        if now >= self._next_blink:
            self._blink_remaining = 0.12
            self._next_blink = now + 2.0 + (now % 2.5)
        self._blink_remaining = max(0.0, self._blink_remaining - dt_s)
        if self._dash is not None and self._dash.active:
            self._render()
            self._dash.show()

    async def emote(self, name: str) -> None:
        self._emotion = name if name in EMOTIONS else "neutral"
        self._log.info("[sim] face: %s", self._emotion)

    async def set_mood(self, mood: str) -> None:
        self._state.mood = mood

    async def speak_visual(self, active: bool) -> None:
        self._speaking = active

    async def stop(self) -> None:
        if self._dash is not None:
            await self._dash.stop()

    # --- drawing -----------------------------------------------------------
    # Face design: white rounded head on black, two big black "lens" eyes with
    # glints, black mouth. White on black only.

    def _render(self) -> None:
        dash = self._dash
        cv2 = dash.cv2
        face = dash.panel(SimDashboard.FACE_RECT)
        face[:] = 0
        if self._window:
            self._draw_face(cv2, face)
        else:
            cv2.putText(
                face, "face off", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
        self._draw_stats(cv2, dash.panel(SimDashboard.STATS_RECT))

    def _draw_face(self, cv2, img) -> None:
        white = (255, 255, 255)
        black = (0, 0, 0)

        # head: white rounded rectangle
        x1, y1, x2, y2, r = 90, 90, 390, 300, 50
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), white, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), white, -1)
        for cx in (x1 + r, x2 - r):
            for cy in (y1 + r, y2 - r):
                cv2.circle(img, (cx, cy), r, white, -1)

        # eyes: two black lenses, each with a white glint — right one smaller (asymmetric look)
        eye_y, lx, rx = 185, 240 - 72, 240 + 72
        if self._emotion == "sleepy":
            eye_r = (24, 16)
        elif self._emotion == "curious":
            eye_r = (46, 32)
        else:
            eye_r = (40, 26)
        if self._blink_remaining > 0:  # blink = closed lids
            for ex, r in ((lx, eye_r[0]), (rx, eye_r[1])):
                cv2.rectangle(img, (ex - r, eye_y - 5), (ex + r, eye_y + 5), black, -1)
        else:
            for ex, r in ((lx, eye_r[0]), (rx, eye_r[1])):
                cv2.circle(img, (ex, eye_y), r, black, -1)
                cv2.circle(img, (ex - r // 3, eye_y - r // 3), 6, white, -1)  # glint

        # mouth: black on the white face
        my = 255
        if self._speaking or self._emotion == "talking":
            h = 6 + int(10 * abs(self._t % 0.6 - 0.3) / 0.3)  # oscillating mouth
            cv2.ellipse(img, (240, my), (20, h), 0, 0, 360, black, -1)
        elif self._emotion == "happy":
            cv2.ellipse(img, (240, my - 10), (34, 20), 0, 20, 160, black, 5)
        elif self._emotion == "sad":
            cv2.ellipse(img, (240, my + 24), (30, 16), 0, 200, 340, black, 5)
        elif self._emotion == "surprised":
            cv2.circle(img, (240, my), 11, black, -1)
        else:  # neutral / sleepy
            cv2.line(img, (240 - 28, my), (240 + 28, my), black, 5)

    def _draw_stats(self, cv2, panel) -> None:
        panel[:] = 0
        lines = [
            f"shikigami [{self._state.backend}]",
            f"mood:  {self._state.mood}",
            f"face:  {self._emotion}",
            f"motion: {self._state.last_motion or '-'}",
            f"status: {self._state.status}",
            f"heard: {self._state.last_heard[:34]}",
            f"latency: {self._state.last_latency[:38]}",
            f"you> {self._dash.input_buffer}{'_' if int(self._t * 2) % 2 == 0 else ' '}",
        ]
        y = 28
        for line in lines:
            cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 32
