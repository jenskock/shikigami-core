"""Procedural body-view math (no display required)."""

from shikigami.hal.picrawler.soft import SoftPicrawler
from shikigami.hal.sim.body_view import (
    BODY_HALF,
    WORLD_GROUND_Z,
    feet_world,
    ground_grid_lines,
    local_foot_to_body,
    project_point,
    snap_to_ground,
    to_screen,
)


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
    lines = ground_grid_lines(-50.0, half=64.0, step=32.0)
    assert len(lines) >= 4
    assert all(a[2] == -50.0 and b[2] == -50.0 for a, b in lines)


def test_snap_keeps_lowest_feet_on_fixed_ground():
    feet = [(0.0, 0.0, -50.0), (10.0, 0.0, -30.0), (0.0, 10.0, -50.0), (10.0, 10.0, -50.0)]
    snapped, body_z = snap_to_ground(feet)
    assert min(f[2] for f in snapped) == WORLD_GROUND_Z
    assert snapped[1][2] == WORLD_GROUND_Z + 20.0  # lifted foot clears the floor
    assert body_z == WORLD_GROUND_Z + 50.0


def test_snap_all_legs_tucked_is_crouch_not_flight():
    """When every foot tucks up, body drops — robot stays on the floor."""
    stand = [(0.0, 0.0, -50.0)] * 4
    tuck = [(0.0, 0.0, -30.0)] * 4
    _, body_stand = snap_to_ground(stand)
    snapped_tuck, body_tuck = snap_to_ground(tuck)
    assert min(f[2] for f in snapped_tuck) == WORLD_GROUND_Z
    assert body_tuck < body_stand
    assert body_tuck == WORLD_GROUND_Z + 30.0
