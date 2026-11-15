"""HAL Motion adapter for Picrawler / SoftPicrawler (shared mapping)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from ..interfaces import Motion
from .soft import SERVO_LABELS


class _CrawlerLike(Protocol):
    stand_position: int
    current_coord: list

    def do_action(self, motion_name: str, step: int = 1, speed: int = 50) -> None: ...
    def do_step(self, _step: str | list, speed: int = 50, israise: bool = False) -> None: ...
    def current_step_all_leg_angle(self) -> list[float]: ...
    def current_step_all_leg_value(self) -> list[list[float]]: ...


# Flat lowered stance (lie) — tip coords close to the body, feet on the ground.
_LIE_STEP = [
    [45.0, 45.0, -20.0],
    [45.0, 0.0, -20.0],
    [45.0, 0.0, -20.0],
    [45.0, 45.0, -20.0],
]


def hal_speed_to_sdk(speed: float) -> int:
    return int(max(0.0, min(1.0, float(speed))) * 100)


def steps_for_duration(duration_s: float | None, *, default: int = 1) -> int:
    if duration_s is None or duration_s <= 0:
        return default
    # Rough: ~1 gait cycle ≈ 0.4 s on device; soft ignores wall time.
    return max(1, int(round(float(duration_s) / 0.4)))


class PicrawlerMotion(Motion):
    """Maps high-level HAL verbs onto Picrawler ``do_action`` / ``do_step``."""

    def __init__(
        self,
        crawler: Any,
        log: logging.Logger,
        state: Any | None = None,
    ) -> None:
        self._c: _CrawlerLike = crawler
        self._log = log
        self._state = state
        self._moving = False

    @property
    def moving(self) -> bool:
        return self._moving

    def _publish(self, label: str) -> None:
        if self._state is None:
            return
        self._state.last_motion = label
        angles = self._c.current_step_all_leg_angle()
        if hasattr(self._c, "current_step_all_leg_value"):
            coords = self._c.current_step_all_leg_value()
        else:
            coords = [list(c) for c in getattr(self._c, "current_coord", [])]
        self._state.servo_angles = [float(a) for a in angles]
        self._state.leg_coords = [[float(x) for x in leg] for leg in coords]
        self._state.servo_labels = list(SERVO_LABELS)

    async def _run(self, label: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        self._log.info("[picrawler] %s", label)
        self._moving = True
        try:
            await asyncio.to_thread(fn, *args, **kwargs)
            self._publish(label)
        finally:
            self._moving = False

    async def walk(
        self, direction: str = "forward", speed: float = 0.5, duration_s: float | None = None
    ) -> None:
        name = "backward" if direction in ("back", "backward") else "forward"
        sdk_speed = hal_speed_to_sdk(speed)
        steps = steps_for_duration(duration_s)
        await self._run(
            f"walk {direction} speed={speed:.2f}",
            self._c.do_action,
            name,
            steps,
            sdk_speed,
        )

    async def turn(
        self, direction: str = "left", angle_deg: float | None = None, speed: float = 0.5
    ) -> None:
        sdk_speed = hal_speed_to_sdk(speed)
        if angle_deg is not None:
            name = "turn left angle" if direction == "left" else "turn right angle"
        else:
            name = "turn left" if direction == "left" else "turn right"
        label = f"turn {direction}"
        if angle_deg is not None:
            label += f" {angle_deg:.0f}deg"
        label += f" speed={speed:.2f}"
        await self._run(label, self._c.do_action, name, 1, sdk_speed)

    async def sit(self) -> None:
        await self._run("sit", self._c.do_step, "sit", 40)

    async def stand(self) -> None:
        await self._run("stand", self._c.do_step, "stand", 40)

    async def lie(self) -> None:
        await self._run("lie", self._c.do_step, _LIE_STEP, 40)

    async def wag(self, duration_s: float | None = None) -> None:
        steps = steps_for_duration(duration_s, default=1)
        await self._run("wag/wave", self._c.do_action, "wave", steps, 50)

    async def look_at(self, x: float, y: float) -> None:
        if self._state is not None:
            self._state.gaze_x = max(-1.0, min(1.0, float(x)))
            self._state.gaze_y = max(-1.0, min(1.0, float(y)))
        # Prefer horizontal look; fall back to vertical when |y| dominates.
        if abs(x) >= abs(y):
            name = "look right" if x > 0.15 else "look left" if x < -0.15 else None
        else:
            name = "look up" if y > 0.15 else "look down" if y < -0.15 else None
        if name is None:
            self._publish(f"look_at ({x:+.2f}, {y:+.2f})")
            return
        await self._run(f"look_at ({x:+.2f}, {y:+.2f}) → {name}", self._c.do_action, name, 1, 50)

    async def stop(self) -> None:
        self._moving = False
        await self._run("stop", self._c.do_step, "stand", 40)

    async def play_action(self, name: str) -> None:
        # Accept both underscored and spaced SunFounder names.
        motion = name.replace("_", " ").strip()
        await self._run(f"action: {motion}", self._c.do_action, motion, 1, 50)
