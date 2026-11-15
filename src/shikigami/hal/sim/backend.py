"""Sim backend: webcam + mic/speaker + face window + logged motion. Runs on macOS."""

from __future__ import annotations

import logging

from ..interfaces import HAL
from .audio import SimAudio
from .camera import SimCamera
from .dashboard import SimDashboard
from .expression import SimExpression
from .motion import SimMotion
from .sensors import SimSensors
from .state import SimState


class SimHAL(HAL):
    def __init__(self, cfg) -> None:
        self.state = SimState(backend="sim")
        log = logging.getLogger("shikigami.sim")
        dashboard = SimDashboard(enabled=cfg.sim.window or cfg.sim.camera_preview, log=log)
        self.dashboard = dashboard
        super().__init__(
            backend="sim",
            motion=SimMotion(self.state, log),
            camera=SimCamera(
                cfg.sim.camera_index,
                preview=cfg.sim.camera_preview,
                dashboard=dashboard,
                log=log,
            ),
            audio=SimAudio(cfg.sim.sample_rate, cfg.sim.mic_device, log),
            sensors=SimSensors(
                touch_prob=cfg.sim.fake_touch_prob, sound_prob=cfg.sim.fake_sound_prob
            ),
            expression=SimExpression(cfg.sim.window, self.state, dashboard, log),
        )
