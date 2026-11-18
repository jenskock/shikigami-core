"""Sim backend: webcam + mic/speaker + face window + logged motion. Runs on macOS."""

from __future__ import annotations

import logging

from shikigami.expression.eyes.state import EyesConfig

from ..cameras import make_camera
from ..interfaces import HAL
from .audio import SimAudio
from .dashboard import SimDashboard
from .expression import SimExpression
from .motion import SimMotion
from .sensors import SimSensors
from .state import SimState


def _eyes_cfg(cfg) -> EyesConfig:
    return EyesConfig.from_mapping(cfg.eyes.model_dump()).apply_personality()


class SimHAL(HAL):
    def __init__(self, cfg) -> None:
        self.state = SimState(backend="sim")
        log = logging.getLogger("shikigami.sim")
        dashboard = SimDashboard(enabled=cfg.sim.window or cfg.sim.camera_preview, log=log)
        dashboard.state = self.state
        self.dashboard = dashboard
        super().__init__(
            backend="sim",
            motion=SimMotion(self.state, log),
            camera=make_camera(cfg, on_robot=False, dashboard=dashboard, log=log),
            audio=SimAudio.from_sim_config(
                cfg.sim,
                log=log,
                on_level=lambda n: setattr(self.state, "mic_level", n),
            ),
            sensors=SimSensors(
                touch_prob=cfg.sim.fake_touch_prob, sound_prob=cfg.sim.fake_sound_prob
            ),
            expression=SimExpression(
                cfg.sim.window, self.state, dashboard, log, eyes_cfg=_eyes_cfg(cfg)
            ),
        )
