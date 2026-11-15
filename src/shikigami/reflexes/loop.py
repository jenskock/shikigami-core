"""Fast local reflex loop (~20 Hz, no network): obstacle stop, touch/sound reacts, idle face."""

from __future__ import annotations

import asyncio
import logging
import time

from ..app.bus import EventBus
from ..config import ReflexConfig
from ..hal.interfaces import HAL
from ..perception.tracker import FaceTracker

REACT_COOLDOWN_S = 2.0  # min time between touch/sound reactions, keeps the log readable


class ReflexLoop:
    def __init__(
        self,
        hal: HAL,
        bus: EventBus,
        cfg: ReflexConfig,
        tracker: FaceTracker | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._hal = hal
        self._bus = bus
        self._cfg = cfg
        self._tracker = tracker
        self._log = log or logging.getLogger("shikigami.reflex")
        self._running = False
        self._last_touch = 0.0
        self._last_sound = 0.0

    async def run(self) -> None:
        self._running = True
        dt = 1.0 / max(1.0, self._cfg.hz)
        self._log.info("reflex loop started @ %.0f Hz", self._cfg.hz)
        while self._running:
            try:
                now = time.monotonic()
                reading = await self._hal.sensors.read()
                if (
                    reading.distance_cm is not None
                    and reading.distance_cm < self._cfg.obstacle_cm
                    and self._hal.motion.moving
                ):
                    await self._hal.motion.stop()
                    await self._bus.publish(
                        "obstacle", {"distance_cm": round(reading.distance_cm, 1)}
                    )
                    self._log.warning("obstacle at %.0f cm — stopped", reading.distance_cm)
                if (
                    reading.touch
                    and self._cfg.touch_react
                    and now - self._last_touch >= REACT_COOLDOWN_S
                ):
                    self._last_touch = now
                    await self._bus.publish("touch", {})
                    await self._hal.expression.emote("surprised")
                    self._log.info("touch detected — reacting")
                if (
                    reading.sound_direction_deg is not None
                    and now - self._last_sound >= REACT_COOLDOWN_S
                ):
                    self._last_sound = now
                    await self._bus.publish(
                        "sound", {"direction_deg": round(reading.sound_direction_deg, 1)}
                    )
                    await self._hal.motion.look_at(reading.sound_direction_deg / 180.0, 0.0)
                dash = getattr(self._hal, "dashboard", None)
                if dash is not None:
                    dash.pump()  # flush keystrokes before/around heavy work
                await self._hal.expression.update(dt)
                if dash is not None:
                    dash.pump()
                await self._hal.camera.preview_frame()
                if self._tracker is not None:
                    await self._tracker.poll()
                if dash is not None:
                    dash.show()  # textures + stats; continuous pump is in dashboard
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("reflex tick failed")
            await asyncio.sleep(dt)

    def stop(self) -> None:
        self._running = False
