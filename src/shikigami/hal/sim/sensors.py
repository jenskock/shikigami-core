"""Simulated sensors: a random-walk distance sensor plus occasional touch/sound events."""

from __future__ import annotations

import random

from ..interfaces import SensorReading, Sensors


class SimSensors(Sensors):
    """Simulated sensors. Distance is a random walk; touch/sound events are random
    and OFF by default (fake_touch_prob / fake_sound_prob) — nothing here reads
    real hardware."""

    def __init__(self, touch_prob: float = 0.0, sound_prob: float = 0.0, seed: int = 0) -> None:
        self._touch_prob = touch_prob
        self._sound_prob = sound_prob
        self._rng = random.Random(seed)
        self._distance = 80.0

    async def read(self) -> SensorReading:
        # random walk around ~80 cm, clamped to plausible sensor range
        self._distance += self._rng.uniform(-6.0, 6.0)
        self._distance = min(300.0, max(5.0, self._distance))
        return SensorReading(
            distance_cm=self._distance,
            touch=self._rng.random() < self._touch_prob,
            imu={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            sound_direction_deg=(
                self._rng.uniform(-180.0, 180.0) if self._rng.random() < self._sound_prob else None
            ),
        )
