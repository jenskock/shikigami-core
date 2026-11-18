"""PiCrawler geometry and timing, shared by the soft crawler, the motion adapter and the body view.

Pure math, no Qt and no hardware imports.

Frames
------
Leg-local: SunFounder's ``[x, y, z]`` per leg, origin at the coxa pivot, z up.
Body: x right, y forward, z up, origin at the centre of the hip square.
Level: the body frame rotated so the ground the robot stands on is z = 0.
World: level frame turned by the robot's yaw and moved by its floor position.

Timing follows ``robot_hat.Robot.servo_move``: every keyframe takes
``1000 - 9.9 * speed`` ms, stretched so no servo exceeds 428 deg/s, and the
servos move linearly in joint space in 10 ms steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

# Link lengths (mm): coxa, femur, tibia — SunFounder's C, A, B.
COXA = 33.0
FEMUR = 48.0
TIBIA = 78.0

BODY_SIDE = 77.0  # MoveList.LENGTH_SIDE: distance between neighbouring coxa pivots
BODY_HALF = BODY_SIDE / 2.0

# RF, LF, LR, RR — leg order of the SunFounder coordinate lists.
HIP_XY: tuple[tuple[float, float], ...] = (
    (BODY_HALF, BODY_HALF),
    (-BODY_HALF, BODY_HALF),
    (-BODY_HALF, -BODY_HALF),
    (BODY_HALF, -BODY_HALF),
)
# Leg-local x points sideways out of the body, y along it, away from the middle.
LOCAL_SIGN: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (1.0, -1.0),
)

SERVO_MAX_DPS = 428.0  # robot_hat Robot.max_dps
SERVO_STEP_S = 0.010  # robot_hat servo_move step time

# A foot this close to the ground plane carries weight.
CONTACT_EPS_MM = 3.0

Vec3 = tuple[float, float, float]


def keyframe_secs(max_delta_deg: float, speed: float) -> float:
    """How long ``servo_move`` takes for a keyframe whose largest joint change is given."""
    speed = max(0.0, min(100.0, float(speed)))
    max_delta = int(abs(max_delta_deg))  # robot_hat truncates the same way
    if max_delta == 0:
        return SERVO_STEP_S
    total_ms = -9.9 * speed + 1000.0
    if max_delta / total_ms * 1000.0 > SERVO_MAX_DPS:
        total_ms = max_delta / SERVO_MAX_DPS * 1000.0
    return max(1, int(total_ms / (SERVO_STEP_S * 1000.0))) * SERVO_STEP_S


def local_foot_to_body(leg_i: int, local: Sequence[float]) -> Vec3:
    """One leg-local point in the body frame."""
    hx, hy = HIP_XY[leg_i]
    sx, sy = LOCAL_SIGN[leg_i]
    return (hx + sx * float(local[0]), hy + sy * float(local[1]), float(local[2]))


def feet_world(leg_coords: Sequence[Sequence[float]]) -> list[Vec3]:
    """Body-frame foot tips for up to four legs; skips malformed entries."""
    return [local_foot_to_body(i, leg) for i, leg in enumerate(leg_coords[:4]) if len(leg) >= 3]


def leg_chain_local(servo: Sequence[float]) -> list[Vec3]:
    """Coxa pivot, femur root, knee and foot of one leg, from its servo angles.

    ``servo`` is one leg's slice of ``servo_positions``: ``[tibia, femur, coxa]``
    (SunFounder's beta, alpha, gamma). The foot matches ``polar2coord``.
    """
    beta, alpha, gamma = (float(v) for v in servo[:3])
    l1 = math.sqrt(FEMUR**2 + TIBIA**2 - 2 * FEMUR * TIBIA * math.cos(math.radians(90.0 + beta)))
    cos_root = (FEMUR**2 + l1**2 - TIBIA**2) / (2 * FEMUR * l1)
    root = math.degrees(math.acos(max(-1.0, min(1.0, cos_root))))
    femur_elev = math.radians(90.0 - alpha)
    foot_elev = math.radians(90.0 - alpha - root)
    ux = math.sin(math.radians(45.0 + gamma))
    uy = math.cos(math.radians(45.0 + gamma))
    knee_r = COXA + FEMUR * math.cos(femur_elev)
    foot_r = COXA + l1 * math.cos(foot_elev)
    return [
        (0.0, 0.0, 0.0),
        (COXA * ux, COXA * uy, 0.0),
        (knee_r * ux, knee_r * uy, FEMUR * math.sin(femur_elev)),
        (foot_r * ux, foot_r * uy, l1 * math.sin(foot_elev)),
    ]


def leg_chains_body(servo_positions: Sequence[float]) -> list[list[Vec3]]:
    """All four leg chains in the body frame, from the twelve servo angles."""
    chains: list[list[Vec3]] = []
    for leg in range(min(4, len(servo_positions) // 3)):
        local = leg_chain_local(servo_positions[leg * 3 : leg * 3 + 3])
        chains.append([local_foot_to_body(leg, p) for p in local])
    return chains


# --- vector helpers -------------------------------------------------------


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _seg_dist(p: Vec3, a: Vec3, b: Vec3) -> float:
    ab = _sub(b, a)
    t = max(0.0, min(1.0, _dot(_sub(p, a), ab) / max(1e-9, _dot(ab, ab))))
    q = (a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t)
    d = _sub(p, q)
    return math.sqrt(_dot(d, d))


def _triangle_dist(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> float:
    """Distance from ``p`` (already in the triangle's plane) to the triangle; 0 inside."""
    v0, v1, v2 = _sub(c, a), _sub(b, a), _sub(p, a)
    d00, d01, d02 = _dot(v0, v0), _dot(v0, v1), _dot(v0, v2)
    d11, d12 = _dot(v1, v1), _dot(v1, v2)
    den = d00 * d11 - d01 * d01
    if abs(den) > 1e-9:
        u = (d11 * d02 - d01 * d12) / den
        v = (d00 * d12 - d01 * d02) / den
        if u >= 0 and v >= 0 and u + v <= 1:
            return 0.0
    return min(_seg_dist(p, a, b), _seg_dist(p, b, c), _seg_dist(p, c, a))


# --- standing on the floor ------------------------------------------------


@dataclass(frozen=True)
class Support:
    """The ground plane under the robot, in the body frame: ``normal · p = offset``."""

    normal: Vec3 = (0.0, 0.0, 1.0)
    offset: float = 0.0
    contact: tuple[bool, ...] = field(default_factory=tuple)

    def height_of(self, p: Vec3) -> float:
        """How far ``p`` is above the ground."""
        return _dot(self.normal, p) - self.offset

    @property
    def body_height(self) -> float:
        return -self.offset


def support_plane(feet: Sequence[Vec3], *, eps: float = CONTACT_EPS_MM) -> Support:
    """The plane the robot rests on: three feet with no foot below it.

    Of those planes it picks the one that holds the body's centre over its
    triangle, so a lifted leg stays in the air while the other three carry the
    robot, and a body tilt keeps all four feet down.
    """
    if not feet:
        return Support()
    best: tuple[float, int, Vec3, float] | None = None
    for i, j, k in combinations(range(len(feet)), 3):
        a, b, c = feet[i], feet[j], feet[k]
        n = _cross(_sub(b, a), _sub(c, a))
        norm = math.sqrt(_dot(n, n))
        if norm < 1e-6:
            continue
        n = _scale(n, 1.0 / norm)
        if n[2] < 0:
            n = _scale(n, -1.0)
        if n[2] < 0.2:  # steeper than ~78 degrees: not something to stand on
            continue
        d = _dot(n, a)
        heights = [_dot(n, f) - d for f in feet]
        if min(heights) < -eps:
            continue
        com = _scale(n, d)  # the body centre dropped onto the plane
        off_balance = _triangle_dist(com, a, b, c)
        touching = sum(h <= eps for h in heights)
        key = (off_balance, -touching)
        if best is None or key < (best[0], best[1]):
            best = (off_balance, -touching, n, d)
    if best is None:
        low = min(f[2] for f in feet)
        return Support(
            normal=(0.0, 0.0, 1.0),
            offset=low,
            contact=tuple(f[2] - low <= eps for f in feet),
        )
    _, _, n, d = best
    return Support(
        normal=n,
        offset=d,
        contact=tuple(abs(_dot(n, f) - d) <= eps for f in feet),
    )


@dataclass(frozen=True)
class Leveler:
    """Turns body-frame points into the level frame for one support plane."""

    rot: tuple[Vec3, Vec3, Vec3]
    height: float

    @classmethod
    def for_support(cls, support: Support) -> Leveler:
        n = support.normal
        z = (0.0, 0.0, 1.0)
        v = _cross(n, z)
        s2 = _dot(v, v)
        c = _dot(n, z)
        if s2 < 1e-12:
            rot = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        else:
            k = (1.0 - c) / s2
            vx, vy, vz = v
            rot = (
                (1 - k * (vy * vy + vz * vz), -vz + k * vx * vy, vy + k * vx * vz),
                (vz + k * vx * vy, 1 - k * (vx * vx + vz * vz), -vx + k * vy * vz),
                (-vy + k * vx * vz, vx + k * vy * vz, 1 - k * (vx * vx + vy * vy)),
            )
        return cls(rot=rot, height=support.body_height)

    def apply(self, p: Vec3) -> Vec3:
        r = self.rot
        return (_dot(r[0], p), _dot(r[1], p), _dot(r[2], p) + self.height)

    def tilt_deg(self) -> tuple[float, float]:
        """(pitch, roll) of the body in degrees; nose up and right side up are positive."""
        fwd = self.apply((0.0, 1.0, 0.0))
        right = self.apply((1.0, 0.0, 0.0))
        h = self.height
        return (
            math.degrees(math.asin(max(-1.0, min(1.0, fwd[2] - h)))),
            math.degrees(math.asin(max(-1.0, min(1.0, -(right[2] - h))))),
        )


def rotate_yaw_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    """Rotate XY around Z by yaw_deg (degrees)."""
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (x * cy - y * sy, x * sy + y * cy)


def stance_odometry(
    prev_level: Sequence[Vec3],
    curr_level: Sequence[Vec3],
    stance: Sequence[bool],
    body_x: float,
    body_y: float,
    body_yaw_deg: float,
) -> tuple[float, float, float]:
    """Move the robot so the feet that carry it stay put on the floor.

    ``prev_level`` / ``curr_level`` are foot positions in the level frame of the
    previous and current pose. The rigid 2D motion that maps the current
    stance feet back onto their previous spots is the body's motion, reversed.
    """
    idx = [i for i, s in enumerate(stance) if s and i < len(prev_level) and i < len(curr_level)]
    if not idx:
        return body_x, body_y, body_yaw_deg
    pcx = sum(prev_level[i][0] for i in idx) / len(idx)
    pcy = sum(prev_level[i][1] for i in idx) / len(idx)
    ccx = sum(curr_level[i][0] for i in idx) / len(idx)
    ccy = sum(curr_level[i][1] for i in idx) / len(idx)
    theta = 0.0
    if len(idx) >= 2:
        num = den = 0.0
        for i in idx:
            ax, ay = curr_level[i][0] - ccx, curr_level[i][1] - ccy
            bx, by = prev_level[i][0] - pcx, prev_level[i][1] - pcy
            num += ax * by - ay * bx
            den += ax * bx + ay * by
        theta = math.atan2(num, den)
    rcx, rcy = rotate_yaw_xy(ccx, ccy, math.degrees(theta))
    tx, ty = pcx - rcx, pcy - rcy
    wx, wy = rotate_yaw_xy(tx, ty, body_yaw_deg)
    return body_x + wx, body_y + wy, (body_yaw_deg + math.degrees(theta)) % 360.0


@dataclass(frozen=True)
class BodyPose:
    """One consistent snapshot of the body for the view (published atomically)."""

    servo_angles: tuple[float, ...] = ()
    leg_coords: tuple[Vec3, ...] = ()
    body_x: float = 0.0
    body_y: float = 0.0
    body_yaw_deg: float = 0.0
    swing: tuple[bool, ...] = ()
    clamped: tuple[bool, ...] = ()
