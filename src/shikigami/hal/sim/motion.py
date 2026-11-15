"""Simulated motion: logs every command instead of driving motors."""

from __future__ import annotations

import asyncio
import logging

from ..interfaces import Motion
from .state import SimState

# Sim runs faster than real time so a turn stays snappy; capped for safety.
SIM_TIME_FACTOR = 0.2
MAX_SIM_SLEEP_S = 2.0


class SimMotion(Motion):
    def __init__(self, state: SimState, log: logging.Logger | None = None) -> None:
        self._state = state
        self._log = log or logging.getLogger("shikigami.sim.motion")
        self._moving = False

    @property
    def moving(self) -> bool:
        return self._moving

    async def _run(self, label: str, duration_s: float | None = None) -> None:
        self._state.last_motion = label
        self._log.info("[sim] motion: %s", label)
        if duration_s:
            self._moving = True
            await asyncio.sleep(min(duration_s * SIM_TIME_FACTOR, MAX_SIM_SLEEP_S))
            self._moving = False

    async def walk(
        self, direction: str = "forward", speed: float = 0.5, duration_s: float | None = None
    ) -> None:
        await self._run(f"walk {direction} speed={speed:.2f}", duration_s)

    async def turn(
        self, direction: str = "left", angle_deg: float | None = None, speed: float = 0.5
    ) -> None:
        angle = f" {angle_deg:.0f}deg" if angle_deg is not None else ""
        await self._run(f"turn {direction}{angle} speed={speed:.2f}", 1.0)

    async def sit(self) -> None:
        await self._run("sit")

    async def stand(self) -> None:
        await self._run("stand")

    async def lie(self) -> None:
        await self._run("lie down")

    async def wag(self, duration_s: float | None = None) -> None:
        await self._run("wag tail", duration_s or 2.0)

    async def look_at(self, x: float, y: float) -> None:
        self._state.gaze_x = max(-1.0, min(1.0, float(x)))
        self._state.gaze_y = max(-1.0, min(1.0, float(y)))
        await self._run(f"look_at ({x:+.2f}, {y:+.2f})")

    async def stop(self) -> None:
        self._moving = False
        await self._run("stop")

    async def play_action(self, name: str) -> None:
        await self._run(f"action: {name}", 1.5)
