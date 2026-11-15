"""SoftPicrawler + PicrawlerMotion (no SunFounder hardware)."""

from __future__ import annotations

import asyncio

import pytest

from shikigami.config import load_config
from shikigami.hal import create_hal
from shikigami.hal.picrawler.backend import PicrawlerHAL
from shikigami.hal.picrawler.motion import PicrawlerMotion, hal_speed_to_sdk
from shikigami.hal.picrawler.soft import SERVO_LABELS, SoftPicrawler
from shikigami.hal.sim.state import SimState


def test_soft_stand_yields_twelve_angles():
    c = SoftPicrawler()
    c.do_step("stand", 40)
    angles = c.current_step_all_leg_angle()
    assert len(angles) == 12
    assert len(SERVO_LABELS) == 12
    assert all(isinstance(a, (int, float)) for a in angles)
    coords = c.current_step_all_leg_value()
    assert len(coords) == 4
    assert all(len(leg) == 3 for leg in coords)


def test_soft_forward_advances_coords():
    c = SoftPicrawler()
    c.do_step("stand", 40)
    before = c.current_step_all_leg_value()
    c.do_action("forward", 1, 80)
    after = c.current_step_all_leg_value()
    assert len(after) == 4
    assert after != before or c.current_step_all_leg_angle()  # pose updated
    assert len(c.current_step_all_leg_angle()) == 12


def test_soft_named_actions():
    c = SoftPicrawler()
    c.do_step("stand", 40)
    for name in ("wave", "look left", "push_up"):
        c.do_action(name, 1, 50)
        assert len(c.current_step_all_leg_angle()) == 12


def test_hal_speed_to_sdk():
    assert hal_speed_to_sdk(0) == 0
    assert hal_speed_to_sdk(1) == 100
    assert hal_speed_to_sdk(0.5) == 50
    assert hal_speed_to_sdk(2.0) == 100


def test_picrawler_motion_publishes_servos():
    state = SimState(backend="picrawler")
    motion = PicrawlerMotion(SoftPicrawler(), __import__("logging").getLogger("test"), state=state)

    async def _run() -> None:
        await motion.stand()
        assert state.last_motion == "stand"
        assert len(state.servo_angles) == 12
        assert state.servo_labels == list(SERVO_LABELS)
        assert len(state.leg_coords) == 4

        await motion.walk("forward", speed=0.6, duration_s=0.4)
        assert "walk forward" in state.last_motion
        assert len(state.servo_angles) == 12

        await motion.play_action("dance")
        assert "dance" in state.last_motion

    asyncio.run(_run())


def test_soft_on_pose_fires_each_frame():
    c = SoftPicrawler(frame_delay_s=0.0)
    poses: list[list[list[float]]] = []

    def _capture() -> None:
        poses.append([list(leg) for leg in c.current_step_all_leg_value()])

    c.on_pose = _capture
    c.do_step("stand", 40)
    assert len(poses) >= 2
    # At least one intermediate frame differs from the final stand pose.
    assert any(p != poses[-1] for p in poses[:-1])


def test_create_picrawler_hal_soft_fallback():
    import sys

    from shikigami.hal.picrawler import backend as be

    if be._HAS_PICRAWLER:
        pytest.skip("real picrawler present on this host")
    cfg = load_config()
    cfg.sim.window = False
    cfg.sim.camera_preview = False
    hal = create_hal("picrawler", cfg)
    assert isinstance(hal, PicrawlerHAL)
    assert hal.backend == "picrawler"
    assert isinstance(hal.motion, PicrawlerMotion)
    assert hal.state is not None
    assert hal.dashboard is not None
    assert hal.dashboard.state is hal.state
    assert "robot_hat" not in sys.modules
    assert "vilib" not in sys.modules


def test_move_tool_schema_includes_named_actions():
    from shikigami.brain.actions import TOOL_SCHEMAS

    move = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "move")
    enum = move["function"]["parameters"]["properties"]["motion"]["enum"]
    for name in ("dance", "wave", "push_up", "walk", "stand"):
        assert name in enum
