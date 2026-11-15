"""PiCrawler backend (SunFounder PiCrawler spider) — runs on the Raspberry Pi only.

SunFounder libraries are imported ONLY in this module. On a Mac the import
fails and the constructor raises BackendUnavailable. Install on the Pi with:
pip install picrawler robot_hat vilib (see https://docs.sunfounder.com).
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import BackendUnavailable
from ..interfaces import HAL, Camera, Expression, Motion, SensorReading, Sensors
from ..sim.audio import SimAudio  # placeholder until a Pi audio HAT is wired

try:  # guarded: only importable on a Pi with SunFounder libs installed
    from picrawler import Picrawler  # type: ignore

    _HAS_PICRAWLER = True
except ImportError:  # pragma: no cover - depends on host
    Picrawler = None  # type: ignore[assignment]
    _HAS_PICRAWLER = False

INSTALL_HINT = (
    "picrawler backend requires SunFounder libs on Raspberry Pi OS: "
    "pip install picrawler robot_hat vilib (https://docs.sunfounder.com)"
)

# default leg pin layout for PiCrawler (BCM); adjust for your wiring
DEFAULT_PINS = [15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def _call_any(obj: Any, candidates: list[str], *args, **kwargs) -> Any:
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise NotImplementedError(
        f"no known picrawler API among {candidates!r} — map it to the SunFounder docs here"
    )


class PicrawlerMotion(Motion):
    def __init__(self, crawler: Any, log: logging.Logger) -> None:
        self._c = crawler
        self._log = log

    async def walk(
        self, direction: str = "forward", speed: float = 0.5, duration_s: float | None = None
    ) -> None:
        self._log.info("[picrawler] walk %s speed=%.2f", direction, speed)
        raise NotImplementedError(
            "map walk() to picrawler do_step/gait API (e.g. do_step([0,40,0]))"
        )

    async def turn(
        self, direction: str = "left", angle_deg: float | None = None, speed: float = 0.5
    ) -> None:
        self._log.info("[picrawler] turn %s %s", direction, angle_deg)
        raise NotImplementedError("map turn() to picrawler turn API")

    async def sit(self) -> None:
        raise NotImplementedError("spider has no sit; map to a lowered stance via do_step")

    async def stand(self) -> None:
        _call_any(
            self._c,
            ["stand", "reset"],
        )

    async def lie(self) -> None:
        raise NotImplementedError("map lie() to a flat stance via do_step")

    async def wag(self, duration_s: float | None = None) -> None:
        raise NotImplementedError("spider has no tail; map to a little dance step")

    async def look_at(self, x: float, y: float) -> None:
        raise NotImplementedError("map look_at() to head servo if equipped")

    async def stop(self) -> None:
        _call_any(
            self._c,
            ["stop", "reset"],
        )

    async def play_action(self, name: str) -> None:
        _call_any(self._c, ["do_action"], name)


class PicrawlerExpression(Expression):
    def __init__(self, log: logging.Logger) -> None:
        self._log = log

    async def start(self) -> None: ...
    async def update(self, dt_s: float) -> None: ...

    async def emote(self, name: str) -> None:
        self._log.info("[picrawler] emote %s", name)
        raise NotImplementedError("map emote() to LED/buzzer via robot_hat")

    async def set_mood(self, mood: str) -> None:
        self._log.info("[picrawler] mood=%s", mood)

    async def speak_visual(self, active: bool) -> None: ...
    async def stop(self) -> None: ...


class PicrawlerSensors(Sensors):
    async def read(self) -> SensorReading:
        raise NotImplementedError("map SensorReading to picrawler ultrasonic/IMU")


class PicrawlerCamera(Camera):
    async def start(self) -> None:
        raise NotImplementedError("use vilib camera on the Pi")

    async def read_frame(self) -> Any | None:
        raise NotImplementedError("use vilib camera on the Pi")

    async def stop(self) -> None: ...


class PicrawlerHAL(HAL):
    def __init__(self, cfg) -> None:
        if not _HAS_PICRAWLER:
            raise BackendUnavailable(INSTALL_HINT)
        log = logging.getLogger("shikigami.picrawler")
        crawler = Picrawler(DEFAULT_PINS)
        super().__init__(
            backend="picrawler",
            motion=PicrawlerMotion(crawler, log),
            camera=PicrawlerCamera(),
            audio=SimAudio(cfg.sim.sample_rate, cfg.sim.mic_device, log),  # TODO: Pi mic/speaker
            sensors=PicrawlerSensors(),
            expression=PicrawlerExpression(log),
        )
