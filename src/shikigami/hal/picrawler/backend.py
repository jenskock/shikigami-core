"""PiCrawler backend (SunFounder spider).

On a Raspberry Pi with SunFounder libs installed, uses the real ``picrawler``
package. On Mac / missing libs, falls back to ``SoftPicrawler`` plus the sim
camera/audio/expression/dashboard so development stays device-faithful.
"""

from __future__ import annotations

import logging
from typing import Any

from shikigami.expression.eyes.state import EyesConfig

from ..cameras import make_camera
from ..errors import BackendUnavailable
from ..interfaces import HAL, Camera, Expression, Motion, SensorReading, Sensors
from ..sim.audio import SimAudio
from ..sim.dashboard import SimDashboard
from ..sim.expression import SimExpression
from ..sim.sensors import SimSensors
from ..sim.state import SimState
from .motion import PicrawlerMotion
from .soft import SoftPicrawler

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
DEFAULT_PINS = SoftPicrawler.PIN_LIST


class PicrawlerExpression(Expression):
    def __init__(self, log: logging.Logger, inner: Expression | None = None) -> None:
        self._log = log
        self._inner = inner

    async def start(self) -> None:
        if self._inner is not None:
            await self._inner.start()

    async def update(self, dt_s: float) -> None:
        if self._inner is not None:
            await self._inner.update(dt_s)

    async def emote(self, name: str) -> None:
        self._log.info("[picrawler] emote %s", name)
        if self._inner is not None:
            await self._inner.emote(name)
            return
        raise NotImplementedError("map emote() to LED/buzzer via robot_hat")

    async def set_mood(self, mood: str) -> None:
        self._log.info("[picrawler] mood=%s", mood)
        if self._inner is not None:
            await self._inner.set_mood(mood)

    async def speak_visual(self, active: bool) -> None:
        if self._inner is not None:
            await self._inner.speak_visual(active)

    async def stop(self) -> None:
        if self._inner is not None:
            await self._inner.stop()


class PicrawlerSensors(Sensors):
    def __init__(self, inner: Sensors | None = None) -> None:
        self._inner = inner

    async def read(self) -> SensorReading:
        if self._inner is not None:
            return await self._inner.read()
        raise NotImplementedError("map SensorReading to picrawler ultrasonic/IMU")


class PicrawlerHAL(HAL):
    """Real Picrawler on Pi; SoftPicrawler + sim chrome when libs are missing."""

    def __init__(self, cfg) -> None:
        log = logging.getLogger("shikigami.picrawler")
        soft = not _HAS_PICRAWLER
        if soft:
            log.info("SunFounder picrawler not installed — using SoftPicrawler")
            # Keyframes play at the robot's own servo speed, sampled at 30 Hz.
            crawler: Any = SoftPicrawler(DEFAULT_PINS, realtime=True)
            crawler.display_hz = 30.0
            eyes_cfg = EyesConfig.from_mapping(cfg.eyes.model_dump()).apply_personality()
            self.state = SimState(backend="picrawler")
            dashboard = SimDashboard(enabled=cfg.sim.window or cfg.sim.camera_preview, log=log)
            dashboard.state = self.state
            self.dashboard = dashboard
            camera: Camera = make_camera(cfg, on_robot=False, dashboard=dashboard, log=log)
            audio = SimAudio.from_sim_config(
                cfg.sim,
                log=log,
                on_level=lambda n: setattr(self.state, "mic_level", n),
            )
            sensors: Sensors = SimSensors(
                touch_prob=cfg.sim.fake_touch_prob, sound_prob=cfg.sim.fake_sound_prob
            )
            expression: Expression = SimExpression(
                cfg.sim.window, self.state, dashboard, log, eyes_cfg=eyes_cfg
            )
            motion: Motion = PicrawlerMotion(crawler, log, state=self.state)
            crawler.get_playback_speed = lambda: float(self.state.playback_speed)
            crawler.realtime = False
            crawler.do_step("stand", 80)
            motion._publish("stand")
            crawler.realtime = True
            super().__init__(
                backend="picrawler",
                motion=motion,
                camera=camera,
                audio=audio,
                sensors=sensors,
                expression=expression,
            )
            return

        # Real device path
        assert Picrawler is not None
        try:
            crawler = Picrawler(DEFAULT_PINS)
        except Exception as exc:  # pragma: no cover - Pi hardware
            raise BackendUnavailable(f"{INSTALL_HINT} ({exc})") from exc

        self.state = None
        self.dashboard = None
        motion = PicrawlerMotion(crawler, log, state=None)
        super().__init__(
            backend="picrawler",
            motion=motion,
            camera=make_camera(cfg, on_robot=True, log=log),
            audio=SimAudio.from_sim_config(cfg.sim, log=log),
            sensors=PicrawlerSensors(),
            expression=PicrawlerExpression(log),
        )
