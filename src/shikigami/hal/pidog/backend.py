"""PiDog backend (SunFounder PiDog quadruped) — runs on the Raspberry Pi only.

SunFounder libraries are imported ONLY in this module. On a Mac the import
fails and the constructor raises BackendUnavailable, so the core never needs
them. Install on the Pi with: pip install pidog robot_hat vilib
(see https://docs.sunfounder.com).
"""

from __future__ import annotations

import logging
from typing import Any

from ..cameras import make_camera
from ..errors import BackendUnavailable
from ..interfaces import HAL, Expression, Motion, SensorReading, Sensors
from ..sim.audio import SimAudio  # placeholder until a Pi audio HAT is wired

try:  # guarded: only importable on a Pi with SunFounder libs installed
    from pidog import PiDog  # type: ignore

    _HAS_PIDOG = True
except ImportError:  # pragma: no cover - depends on host
    PiDog = None  # type: ignore[assignment]
    _HAS_PIDOG = False

INSTALL_HINT = (
    "pidog backend requires SunFounder libs on Raspberry Pi OS: "
    "pip install pidog robot_hat vilib (https://docs.sunfounder.com)"
)


def _call_any(obj: Any, candidates: list[str], *args, **kwargs) -> Any:
    """Call the first method that exists; raise NotImplementedError otherwise."""
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise NotImplementedError(
        f"no known pidog API among {candidates!r} — check SunFounder pidog docs and map it here"
    )


class PiDogMotion(Motion):
    def __init__(self, dog: Any, log: logging.Logger) -> None:
        self._dog = dog
        self._log = log

    async def walk(
        self, direction: str = "forward", speed: float = 0.5, duration_s: float | None = None
    ) -> None:
        self._log.info("[pidog] walk %s speed=%.2f", direction, speed)
        raise NotImplementedError(
            "map walk() to pidog gait API (do_action('walk'/'back') or leg servo control)"
        )

    async def turn(
        self, direction: str = "left", angle_deg: float | None = None, speed: float = 0.5
    ) -> None:
        self._log.info("[pidog] turn %s %s", direction, angle_deg)
        raise NotImplementedError("map turn() to pidog API (do_action('turn_left'/...) or servo)")

    async def sit(self) -> None:
        _call_any(self._dog, ["sit", "do_action"], "sit")

    async def stand(self) -> None:
        _call_any(self._dog, ["stand", "do_action"], "stand")

    async def lie(self) -> None:
        _call_any(self._dog, ["lie_down", "liedown", "do_action"], "lie_down")

    async def wag(self, duration_s: float | None = None) -> None:
        _call_any(self._dog, ["wag_tail", "do_action"], "wag_tail")

    async def look_at(self, x: float, y: float) -> None:
        _call_any(self._dog, ["head_move", "look_at"], x, y)

    async def stop(self) -> None:
        _call_any(
            self._dog,
            ["stop", "reset"],
        )

    async def play_action(self, name: str) -> None:
        _call_any(self._dog, ["do_action"], name)


class PiDogExpression(Expression):
    """PiDog head/face: eyes are done via OLED/LED + servo head moves; sounds via speaker."""

    def __init__(self, dog: Any, log: logging.Logger) -> None:
        self._dog = dog
        self._log = log

    async def start(self) -> None: ...
    async def update(self, dt_s: float) -> None: ...

    async def emote(self, name: str) -> None:
        self._log.info("[pidog] emote %s", name)
        raise NotImplementedError("map emote() to pidog eye-expression/sound API")

    async def set_mood(self, mood: str) -> None:
        self._log.info("[pidog] mood=%s", mood)

    async def speak_visual(self, active: bool) -> None: ...
    async def stop(self) -> None: ...


class PiDogSensors(Sensors):
    def __init__(self, dog: Any) -> None:
        self._dog = dog

    async def read(self) -> SensorReading:
        # pidog exposes ultrasonic distance + touch via robot_hat; map here when wired
        raise NotImplementedError("map SensorReading to pidog ultrasonic/touch/IMU")


class PiDogHAL(HAL):
    def __init__(self, cfg) -> None:
        if not _HAS_PIDOG:
            raise BackendUnavailable(INSTALL_HINT)
        log = logging.getLogger("shikigami.pidog")
        dog = PiDog()  # uses default pin layout; adjust for your build
        super().__init__(
            backend="pidog",
            motion=PiDogMotion(dog, log),
            camera=make_camera(cfg, on_robot=True, log=log),
            audio=SimAudio.from_sim_config(cfg.sim, log=log),  # Pi audio HAT not wired yet
            sensors=PiDogSensors(dog),
            expression=PiDogExpression(dog, log),
        )
