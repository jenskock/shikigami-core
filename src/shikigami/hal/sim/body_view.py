"""Procedural 3D stick view of the PiCrawler quadruped for the sim dashboard.

Foot tips come from SoftPicrawler / Picrawler `leg_coords` (local per-leg XYZ).
Hips sit at the corners of a square of side `LENGTH_SIDE` (77 mm). Projection is
a simple yaw/pitch camera with fit-to-widget scaling — no Qt3D or mesh assets.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

# Match SoftPicrawler MoveList.LENGTH_SIDE (body square edge, mm).
BODY_SIDE = 77.0
BODY_HALF = BODY_SIDE / 2.0
BODY_THICKNESS = 12.0  # visual body height (mm), not from IK

# RF, LF, LR, RR — same order as SoftPicrawler LEG_NAMES / current_coord.
_HIP_XY: tuple[tuple[float, float], ...] = (
    (BODY_HALF, BODY_HALF),  # RF
    (-BODY_HALF, BODY_HALF),  # LF
    (-BODY_HALF, -BODY_HALF),  # LR
    (BODY_HALF, -BODY_HALF),  # RR
)
# Local X/Y sign into body frame (from move_body_absolute axis conventions).
_LOCAL_SIGN: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),  # RF
    (-1.0, 1.0),  # LF
    (-1.0, -1.0),  # LR
    (1.0, -1.0),  # RR
)

_ORBIT_DEG_PER_S = 12.0
_ELEVATION_DEG = 32.0
_PAD_FRAC = 0.12
# Fixed framing radius (mm) so auto-fit does not hide gait motion.
_REF_RADIUS = 150.0
# Fixed floor (SoftPicrawler Z_DEFAULT). Robot snaps down onto it — never flies.
WORLD_GROUND_Z = -50.0
_MIN_BODY_ABOVE = 20.0
_GROUND_HALF = 130.0
_GROUND_STEP = 32.0


def local_foot_to_body(leg_i: int, local: Sequence[float]) -> tuple[float, float, float]:
    """Map one leg's local [x, y, z] to body-frame XYZ (Z up; local Z is already up/down)."""
    lx, ly, lz = float(local[0]), float(local[1]), float(local[2])
    hx, hy = _HIP_XY[leg_i]
    sx, sy = _LOCAL_SIGN[leg_i]
    return (hx + sx * lx, hy + sy * ly, lz)


def feet_world(leg_coords: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    """Body-frame foot positions for up to 4 legs; skips malformed entries."""
    out: list[tuple[float, float, float]] = []
    for i, leg in enumerate(leg_coords[:4]):
        if len(leg) < 3:
            continue
        out.append(local_foot_to_body(i, leg))
    return out


def snap_to_ground(
    feet: Sequence[tuple[float, float, float]],
    *,
    ground_z: float = WORLD_GROUND_Z,
) -> tuple[list[tuple[float, float, float]], float]:
    """Snap lowest feet to `ground_z`; return (feet, body_z). Tucked legs crouch."""
    if not feet:
        return [], ground_z + _MIN_BODY_ABOVE
    min_z = min(f[2] for f in feet)
    dz = ground_z - min_z
    snapped = [(f[0], f[1], f[2] + dz) for f in feet]
    body_above = max(_MIN_BODY_ABOVE, -min_z)
    return snapped, ground_z + body_above


def ground_grid_lines(
    z: float,
    *,
    half: float = _GROUND_HALF,
    step: float = _GROUND_STEP,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Axis-aligned grid segments on the Z=z plane."""
    lines: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    n = int(half // step)
    for i in range(-n, n + 1):
        t = i * step
        lines.append(((-half, t, z), (half, t, z)))
        lines.append(((t, -half, z), (t, half, z)))
    return lines


def hip_world(leg_i: int, body_z: float = 0.0) -> tuple[float, float, float]:
    hx, hy = _HIP_XY[leg_i]
    return (hx, hy, body_z)


def body_corners(body_z: float = 0.0) -> list[tuple[float, float, float]]:
    """Top face corners RF→LF→LR→RR at body_z; bottom face offset by -BODY_THICKNESS."""
    top = [hip_world(i, body_z) for i in range(4)]
    bot = [hip_world(i, body_z - BODY_THICKNESS) for i in range(4)]
    return top + bot


def rotate_yaw_pitch(
    point: tuple[float, float, float], yaw_deg: float, pitch_deg: float
) -> tuple[float, float, float]:
    """Rotate body-frame point into camera space (yaw around Z, then pitch around X)."""
    x, y, z = point
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1 = x * cy - y * sy
    y1 = x * sy + y * cy
    z1 = z
    pitch = math.radians(pitch_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    y2 = y1 * cp - z1 * sp
    z2 = y1 * sp + z1 * cp
    return (x1, y2, z2)


def project_point(
    point: tuple[float, float, float],
    yaw_deg: float,
    pitch_deg: float = _ELEVATION_DEG,
) -> tuple[float, float]:
    """Orthographic project to 2D (x right, y up in view space before screen flip)."""
    # After yaw/pitch: camera looks down -Y; screen X ← world X', screen Y ← world Z'.
    x_cam, _, z_cam = rotate_yaw_pitch(point, yaw_deg, pitch_deg)
    return (x_cam, z_cam)


def to_screen(
    pt: tuple[float, float], scale: float, ox: float, oy: float
) -> tuple[float, float]:
    return (ox + pt[0] * scale, oy - pt[1] * scale)


class BodyView(QWidget):
    """Qt widget: stick quadruped driven by `leg_coords`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._feet: list[tuple[float, float, float]] = []
        self._raw: list[list[float]] = []
        self._yaw0 = 35.0
        self._t0 = time.monotonic()
        self.setMinimumSize(200, 280)
        self.setAutoFillBackground(False)
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 Hz orbit refresh
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def set_pose(self, leg_coords: Sequence[Sequence[float]]) -> None:
        raw = [[float(x) for x in leg[:3]] for leg in leg_coords if len(leg) >= 3]
        if raw == self._raw:
            return
        self._raw = raw
        self._feet = feet_world(raw)
        self.update()

    def _yaw(self) -> float:
        elapsed = time.monotonic() - self._t0
        return self._yaw0 + elapsed * _ORBIT_DEG_PER_S

    def paintEvent(self, event: object) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        painter.fillRect(self.rect(), QColor("#1a1d26"))

        if not self._feet:
            painter.setPen(QColor("#8b90a0"))
            painter.drawText(
                self.rect(),
                int(Qt.AlignmentFlag.AlignCenter),
                "no pose yet\n(move to publish)",
            )
            painter.end()
            return

        yaw = self._yaw()
        feet, body_z = snap_to_ground(self._feet)
        hips = [hip_world(i, body_z) for i in range(len(feet))]
        corners = body_corners(body_z)
        ground_z = WORLD_GROUND_Z

        usable = min(w, h) * (1.0 - 2.0 * _PAD_FRAC)
        scale = usable / (2.0 * _REF_RADIUS)
        ox, oy = w / 2.0, h / 2.0

        def scr(p3: tuple[float, float, float]) -> QPointF:
            x, y = to_screen(project_point(p3, yaw), scale, ox, oy)
            return QPointF(x, y)

        pen_ground = QPen(QColor("#2f3444"))
        pen_ground.setWidth(1)
        painter.setPen(pen_ground)
        for a, b in ground_grid_lines(ground_z):
            painter.drawLine(scr(a), scr(b))
        pen_ground_edge = QPen(QColor("#8b90a0"))
        pen_ground_edge.setWidth(1)
        painter.setPen(pen_ground_edge)
        gh = _GROUND_HALF
        edge = [
            (-gh, -gh, ground_z),
            (gh, -gh, ground_z),
            (gh, gh, ground_z),
            (-gh, gh, ground_z),
        ]
        for i in range(4):
            painter.drawLine(scr(edge[i]), scr(edge[(i + 1) % 4]))

        top, bot = corners[:4], corners[4:]
        pen_body = QPen(QColor("#3dd6c6"))
        pen_body.setWidth(2)
        painter.setPen(pen_body)
        for i in range(4):
            painter.drawLine(scr(top[i]), scr(top[(i + 1) % 4]))
            painter.drawLine(scr(bot[i]), scr(bot[(i + 1) % 4]))
            painter.drawLine(scr(top[i]), scr(bot[i]))

        pen_leg = QPen(QColor("#e8eaef"))
        pen_leg.setWidth(2)
        painter.setPen(pen_leg)
        for i, foot in enumerate(feet):
            painter.drawLine(scr(hips[i]), scr(foot))

        painter.setBrush(QColor("#3dd6c6"))
        painter.setPen(Qt.PenStyle.NoPen)
        for foot in feet:
            p = scr(foot)
            painter.drawEllipse(p, 4.0, 4.0)

        painter.end()
