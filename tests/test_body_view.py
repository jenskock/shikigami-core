"""PiCrawler kinematics and body-view math (no display required)."""

import math

import pytest

from shikigami.hal.picrawler.kinematics import (
    BODY_HALF,
    COXA,
    FEMUR,
    TIBIA,
    Leveler,
    feet_world,
    keyframe_secs,
    leg_chain_local,
    local_foot_to_body,
    rotate_yaw_xy,
    stance_odometry,
    support_plane,
)
from shikigami.hal.picrawler.soft import MoveList, SoftPicrawler
from shikigami.hal.sim.body_view import (
    body_to_world,
    center_on_body,
    ground_grid_lines,
    project_point,
    to_screen,
)

STAND = [[45, 45, -50], [45, 0, -50], [45, 0, -50], [45, 45, -50]]


def test_local_foot_to_body_rf_signs():
    x, y, z = local_foot_to_body(0, [10.0, 20.0, -30.0])
    assert x == BODY_HALF + 10.0
    assert y == BODY_HALF + 20.0
    assert z == -30.0


def test_local_foot_to_body_lf_flips_x():
    x, y, z = local_foot_to_body(1, [10.0, 20.0, -30.0])
    assert x == -BODY_HALF - 10.0
    assert y == BODY_HALF + 20.0
    assert z == -30.0


def test_feet_world_stand_shape():
    c = SoftPicrawler()
    c.do_step("stand", 40)
    coords = c.current_step_all_leg_value()
    assert len(coords) == 4
    feet = feet_world(coords)
    assert len(feet) == 4
    for foot in feet:
        assert len(foot) == 3


def test_project_point_origin_at_centerish():
    assert project_point((0.0, 0.0, 0.0), yaw_deg=0.0) == (0.0, 0.0)


def test_project_point_moves_with_yaw():
    p0 = project_point((50.0, 0.0, 0.0), yaw_deg=0.0)
    p90 = project_point((50.0, 0.0, 0.0), yaw_deg=90.0)
    assert abs(p0[0] - 50.0) < 1e-6
    assert abs(p90[0]) < 1e-6


def test_to_screen_centers_origin():
    sx, sy = to_screen((0.0, 0.0), scale=2.0, ox=100.0, oy=100.0)
    assert abs(sx - 100.0) < 1e-6
    assert abs(sy - 100.0) < 1e-6


def test_ground_grid_lines_on_plane():
    lines = ground_grid_lines(0.0, half=64.0, step=32.0)
    assert len(lines) >= 4
    assert all(a[2] == 0.0 and b[2] == 0.0 for a, b in lines)


def test_body_to_world_translates_and_yaws():
    p = body_to_world((10.0, 0.0, -50.0), body_x=100.0, body_y=200.0, body_yaw_deg=0.0)
    assert p == (110.0, 200.0, -50.0)
    p90 = body_to_world((10.0, 0.0, -50.0), body_x=0.0, body_y=0.0, body_yaw_deg=90.0)
    assert abs(p90[0] - 0.0) < 1e-6
    assert abs(p90[1] - 10.0) < 1e-6


def test_center_on_body_keeps_robot_framed():
    world = body_to_world((0.0, 0.0, -50.0), body_x=80.0, body_y=-40.0, body_yaw_deg=0.0)
    framed = center_on_body(world, 80.0, -40.0)
    assert abs(framed[0]) < 1e-6
    assert abs(framed[1]) < 1e-6
    assert framed[2] == -50.0


def test_ground_grid_follows_origin():
    lines = ground_grid_lines(0.0, half=64.0, step=32.0, origin_x=100.0, origin_y=50.0)
    xs = [a[0] for a, _ in lines] + [b[0] for _, b in lines]
    # Origin snaps to step grid (floor(100/32)*32 = 96); span stays ~2*half.
    assert min(xs) == 96.0 - 64.0
    assert max(xs) == 96.0 + 64.0
    mid = (min(xs) + max(xs)) / 2.0
    assert abs(mid - 96.0) < 1e-6


def test_rotate_yaw_xy_quarter_turn():
    x, y = rotate_yaw_xy(1.0, 0.0, 90.0)
    assert abs(x) < 1e-6
    assert abs(y - 1.0) < 1e-6


def test_leg_chain_matches_the_robots_own_kinematics():
    """Coxa, femur, tibia keep their lengths, and the foot lands where IK aimed it."""
    c = SoftPicrawler()
    for coord in ([45, 45, -50], [45, 0, -50], [70, 0, -30], [0, 120, 60], [20, 10, -70]):
        alpha, beta, gamma = c.coord2polar(list(coord))
        _, servo = c.limit_angle([beta, alpha, gamma])
        hip, coxa, knee, foot = leg_chain_local(servo)
        assert math.dist(hip, coxa) == pytest.approx(COXA)
        assert math.dist(coxa, knee) == pytest.approx(FEMUR)
        assert math.dist(knee, foot) == pytest.approx(TIBIA)
        assert foot == pytest.approx(tuple(c.polar2coord(servo)), abs=0.05)
        if coord[2] < 0:
            assert knee[2] > foot[2]  # standing legs: the knee rides above the foot


def test_standing_robot_is_level_on_four_feet():
    support = support_plane(feet_world(STAND))
    assert support.contact == (True, True, True, True)
    level = Leveler.for_support(support)
    assert level.height == pytest.approx(50.0)
    assert level.tilt_deg() == pytest.approx((0.0, 0.0), abs=1e-6)


def test_lifted_leg_hangs_in_the_air():
    lifted = [list(c) for c in STAND]
    lifted[1] = [70, 0, -30]
    support = support_plane(feet_world(lifted))
    assert support.contact == (True, False, True, True)
    level = Leveler.for_support(support)
    heights = [level.apply(f)[2] for f in feet_world(lifted)]
    assert heights[1] == pytest.approx(20.0, abs=0.5)
    assert [heights[i] for i in (0, 2, 3)] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_body_tilt_keeps_feet_down():
    """Shortening the rear legs pitches the nose up instead of floating the feet."""
    feet = feet_world(MoveList().rotate_body_absolute_y(15))
    support = support_plane(feet)
    assert all(support.contact)
    pitch, roll = Leveler.for_support(support).tilt_deg()
    assert abs(pitch) + abs(roll) > 10.0


def test_keyframe_time_follows_robot_hat():
    assert keyframe_secs(30, 50) == pytest.approx(0.50)  # 1000 - 9.9 * 50 ms
    assert keyframe_secs(30, 100) == pytest.approx(0.07)  # 428 deg/s cap
    assert keyframe_secs(0, 50) == pytest.approx(0.01)


def test_stance_odometry_planted_feet_push_body_forward():
    """Planted feet sliding back under the body carry the body forward."""
    prev = [(40.0, 40.0, 0.0), (-40.0, 40.0, 0.0), (-40.0, -40.0, 0.0), (40.0, -40.0, 0.0)]
    curr = [(40.0, 20.0, 0.0), (-40.0, 20.0, 0.0), (-40.0, -20.0, 0.0), (40.0, -20.0, 0.0)]
    stance = [True, True, False, False]
    x, y, yaw = stance_odometry(prev, curr, stance, 0.0, 0.0, 0.0)
    assert (x, y) == pytest.approx((0.0, 20.0))
    assert yaw == pytest.approx(0.0)


def test_stance_odometry_turns_against_the_planted_feet():
    prev = [(40.0, 0.0, 0.0), (-40.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, -40.0, 0.0)]
    curr = [(0.0, 40.0, 0.0), (0.0, -40.0, 0.0), (-40.0, 0.0, 0.0), (40.0, 0.0, 0.0)]
    _, _, yaw = stance_odometry(prev, curr, [True, True, False, False], 0.0, 0.0, 0.0)
    assert _angle_wrap(yaw) == pytest.approx(-90.0)


def test_swinging_feet_do_not_move_the_body():
    prev = [(40.0, 40.0, 0.0)] * 4
    curr = [(40.0, 90.0, 20.0)] * 4
    assert stance_odometry(prev, curr, [False] * 4, 1.0, 2.0, 3.0) == (1.0, 2.0, 3.0)


def test_rotate_yaw_xy_from_kinematics():
    assert rotate_yaw_xy(1.0, 0.0, 90.0) == pytest.approx((0.0, 1.0))


def _angle_wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0
