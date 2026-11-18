"""HAL Motion adapter for Picrawler / SoftPicrawler (shared mapping).

Every crawler call runs one at a time behind a lock, in a worker thread.
Poses are published as one ``BodyPose`` snapshot, so the body view never mixes
the angles of one frame with the feet of another.

Following a face is mostly the eyes' job. The body turns only after the face
has stayed well off to one side for a while, and turns back once it is gone.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol

from ..interfaces import Motion
from ..sim.state import SimState
from .kinematics import (
    BodyPose,
    Leveler,
    feet_world,
    keyframe_secs,
    stance_odometry,
    support_plane,
)
from .soft import SERVO_LABELS, MoveList, SoftPicrawler


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


# The two standing shapes the gaits leave the robot in (they alternate sides).
_STAND_SHAPES = (
    [[45.0, 45.0, -50.0], [45.0, 0.0, -50.0], [45.0, 0.0, -50.0], [45.0, 45.0, -50.0]],
    [[45.0, 0.0, -50.0], [45.0, 45.0, -50.0], [45.0, 45.0, -50.0], [45.0, 0.0, -50.0]],
)

# Keyframes in one cycle of each gait (MoveList, without the stand-up prefix).
_GAIT_FRAMES = {"forward": 7, "backward": 7, "turn left": 6, "turn right": 6}
# Named actions that run faster than the requested speed. The dance has 871
# keyframes: at speed 50 it would take seven minutes on the robot.
_ACTION_SPEED = {"dance": 100}

# Face following: how far off-centre (gaze -1..1) the face must be, and for how
# long, before the body turns; and how long without it before turning back.
_FOLLOW_ENTER = 0.4
_FOLLOW_DWELL_S = 1.2
_FOLLOW_RELEASE_S = 2.0


def cycle_secs(gait: str, speed: float) -> float:
    """Seconds one cycle of a gait takes on the robot at a HAL speed (0..1)."""
    frames = _GAIT_FRAMES.get(gait, 7)
    # Gait keyframes move the joints well under the servo speed limit, so the
    # speed formula alone sets their length.
    return frames * keyframe_secs(1.0, hal_speed_to_sdk(speed))


def steps_for_duration(
    duration_s: float | None,
    *,
    default: int = 1,
    gait: str = "forward",
    speed: float = 0.5,
) -> int:
    if duration_s is None or duration_s <= 0:
        return default
    return max(1, int(round(float(duration_s) / cycle_secs(gait, speed))))


class PicrawlerMotion(Motion):
    """Maps high-level HAL verbs onto Picrawler ``do_action`` / ``do_step``."""

    def __init__(
        self,
        crawler: _CrawlerLike,
        log: logging.Logger,
        state: SimState | None = None,
    ) -> None:
        self._c = crawler
        self._log = log
        self._state = state
        self._moving = False
        self._lock = asyncio.Lock()
        # Odometry memory: last pose's feet in the level frame, and which touched.
        self._prev_level: list[tuple[float, float, float]] | None = None
        self._prev_contact: tuple[bool, ...] = ()
        # Face following.
        self._follow_side: str | None = None  # side the body is turned to
        self._pending_side: str | None = None
        self._pending_since = 0.0
        self._last_offcentre = 0.0
        self._follow_task: asyncio.Task | None = None

    @property
    def moving(self) -> bool:
        return self._moving

    def cycle_secs(self, speed: float, gait: str = "forward") -> float:
        return cycle_secs(gait, speed)

    def _publish(self, label: str | None = None) -> None:
        if self._state is None:
            return
        if label is not None:
            self._state.last_motion = label
        angles = [float(a) for a in self._c.current_step_all_leg_angle()]
        coords = [tuple(float(x) for x in leg[:3]) for leg in self._c.current_step_all_leg_value()]
        swing = tuple(getattr(self._c, "swing", [False] * 4))
        clamped = tuple(getattr(self._c, "clamped", [False] * 4))

        feet = feet_world(coords)
        support = support_plane(feet)
        level = Leveler.for_support(support)
        level_feet = [level.apply(f) for f in feet]
        x, y, yaw = self._state.body_x, self._state.body_y, self._state.body_yaw_deg
        if self._prev_level is not None and len(self._prev_contact) == len(feet):
            stance = [
                support.contact[i] and self._prev_contact[i] and not (i < len(swing) and swing[i])
                for i in range(len(feet))
            ]
            x, y, yaw = stance_odometry(self._prev_level, level_feet, stance, x, y, yaw)
        self._prev_level = level_feet
        self._prev_contact = support.contact

        st = self._state
        st.servo_angles = angles
        st.leg_coords = [list(c) for c in coords]
        st.servo_labels = list(SERVO_LABELS)
        st.body_x, st.body_y, st.body_yaw_deg = x, y, yaw
        st.body_pose = BodyPose(
            servo_angles=tuple(angles),
            leg_coords=tuple(coords),  # type: ignore[arg-type]
            body_x=x,
            body_y=y,
            body_yaw_deg=yaw,
            swing=swing,
            clamped=clamped,
        )

    async def _run(self, label: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        async with self._lock:
            self._log.info("[picrawler] %s", label)
            self._moving = True
            prev_cb = getattr(self._c, "on_pose", None)
            soft = isinstance(self._c, SoftPicrawler)
            if soft:
                self._c.on_pose = lambda: self._publish(label)  # type: ignore[union-attr]
            try:
                await asyncio.to_thread(fn, *args, **kwargs)
                self._publish(label)
            finally:
                if soft:
                    self._c.on_pose = prev_cb  # type: ignore[union-attr]
                self._moving = False
                if not label.startswith("look"):
                    self._follow_side = None

    async def walk(
        self, direction: str = "forward", speed: float = 0.5, duration_s: float | None = None
    ) -> None:
        name = "backward" if direction in ("back", "backward") else "forward"
        steps = steps_for_duration(duration_s, gait=name, speed=speed)
        await self._run(
            f"walk {direction} x{steps} speed={speed:.2f}",
            self._c.do_action,
            name,
            steps,
            hal_speed_to_sdk(speed),
        )

    async def turn(
        self,
        direction: str = "left",
        angle_deg: float | None = None,
        speed: float = 0.5,
        duration_s: float | None = None,
    ) -> None:
        if angle_deg is not None:
            name = "turn left angle" if direction == "left" else "turn right angle"
        else:
            name = "turn left" if direction == "left" else "turn right"
        steps = steps_for_duration(duration_s, gait=name.replace(" angle", ""), speed=speed)
        label = f"turn {direction} x{steps}"
        if angle_deg is not None:
            label += f" {angle_deg:.0f}deg"
        label += f" speed={speed:.2f}"
        await self._run(label, self._c.do_action, name, steps, hal_speed_to_sdk(speed))

    async def sit(self) -> None:
        await self._run("sit", self._c.do_step, "sit", 40)

    async def stand(self) -> None:
        await self._run("stand", self._c.do_step, "stand", 40)

    async def lie(self) -> None:
        await self._run("lie", self._c.do_step, _LIE_STEP, 40)

    async def wag(self, duration_s: float | None = None) -> None:
        frames = len(MoveList().wave)
        steps = 1
        if duration_s is not None and duration_s > 0:
            steps = max(1, int(round(duration_s / (frames * keyframe_secs(1.0, 50)))))
        await self._run("wag/wave", self._c.do_action, "wave", steps, 50)

    async def look_at(self, x: float, y: float) -> None:
        """Point the eyes at (x, y); turn the body only towards a face that stays aside.

        Never blocks: the face tracker calls this from the reflex loop.
        """
        x = max(-1.0, min(1.0, float(x)))
        y = max(-1.0, min(1.0, float(y)))
        if self._state is not None:
            self._state.gaze_x = x
            self._state.gaze_y = y
        now = time.monotonic()
        side = "right" if x > _FOLLOW_ENTER else "left" if x < -_FOLLOW_ENTER else None
        if side is None:
            self._pending_side = None
            return
        self._last_offcentre = now
        if side == self._follow_side:
            return
        if side != self._pending_side:
            self._pending_side, self._pending_since = side, now
            return
        busy = self._moving or self._lock.locked()
        running = self._follow_task is not None and not self._follow_task.done()
        if now - self._pending_since < _FOLLOW_DWELL_S or busy or running:
            return
        self._pending_side = None
        self._follow_task = asyncio.create_task(self._follow(side), name=f"look-{side}")

    async def _follow(self, side: str) -> None:
        try:
            await self._run(f"look {side} (face)", self._c.do_action, f"look {side}", 1, 50)
            self._follow_side = side
            while self._follow_side == side:
                await asyncio.sleep(0.25)
                idle = time.monotonic() - self._last_offcentre
                if idle >= _FOLLOW_RELEASE_S and not self._lock.locked():
                    await self.settle()
                    break
        except Exception as exc:  # a failed glance must not kill the tracker
            self._log.warning("face follow failed: %s", exc)
        finally:
            if self._follow_side == side:
                self._follow_side = None

    async def settle(self) -> None:
        """One keyframe back to the nearest standing shape.

        ``stand`` replays the library's stand-up sequence, which crouches first;
        after a glance or a gesture the robot only needs to straighten out.
        """
        current = self._c.current_step_all_leg_value()

        def gap(shape: list[list[float]]) -> float:
            return sum(
                abs(a - b)
                for leg, target in zip(current, shape, strict=False)
                for a, b in zip(leg, target, strict=False)
            )

        shape = min(_STAND_SHAPES, key=gap)
        await self._run("settle", self._c.do_step, [list(leg) for leg in shape], 50)

    async def stop(self) -> None:
        await self._run("stop", self._c.do_step, "stand", 40)

    async def play_action(self, name: str) -> None:
        # Accept both underscored and spaced SunFounder names.
        motion = name.replace("_", " ").strip()
        speed = _ACTION_SPEED.get(motion, 50)
        await self._run(f"action: {motion}", self._c.do_action, motion, 1, speed)
