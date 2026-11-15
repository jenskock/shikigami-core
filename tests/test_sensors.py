"""SimSensors: touch/sound events follow the configured probabilities."""

import asyncio

from shikigami.hal.sim.sensors import SimSensors


def test_events_off_by_default():
    async def main():
        s = SimSensors()
        for _ in range(200):
            r = await s.read()
            assert r.touch is False
            assert r.sound_direction_deg is None
            assert 5.0 <= r.distance_cm <= 300.0

    asyncio.run(main())


def test_events_fire_when_probability_is_one():
    async def main():
        s = SimSensors(touch_prob=1.0, sound_prob=1.0)
        r = await s.read()
        assert r.touch is True
        assert r.sound_direction_deg is not None

    asyncio.run(main())
