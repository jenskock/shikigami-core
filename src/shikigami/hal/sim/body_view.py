"""Procedural 3D stick view of the PiCrawler quadruped for the sim dashboard.

Foot tips come from SoftPicrawler / Picrawler `leg_coords` (local per-leg XYZ).
Hips sit at the corners of a square of side `LENGTH_SIDE` (77 mm). Projection is
a simple yaw/pitch camera with fit-to-widget scaling — no Qt3D or mesh assets.

World pose (`body_x` / `body_y` / `body_yaw_deg`) places the robot on the floor;
the view stays centered on the body so the grid slides underneath while walking.
Camera yaw/zoom are manual (drag / scroll) — no auto-orbit.

Gait frames are body-local (legs only). World motion comes from stance-foot
odometry: planted feet stay fixed in world space, so the body/grid move opposite
their body-frame motion each frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
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

_ELEVATION_DEG = 32.0
_DEFAULT_YAW_DEG = 35.0
_PAD_FRAC = 0.12
# Framing radius (mm); zoom changes this via scroll.
_REF_RADIUS_DEFAULT = 150.0
_REF_RADIUS_MIN = 60.0
_REF_RADIUS_MAX = 400.0
# Fixed floor (SoftPicrawler Z_DEFAULT). Robot snaps down onto it — never flies.
WORLD_GROUND_Z = -50.0
_MIN_BODY_ABOVE = 20.0
_GROUND_HALF = 220.0
_GROUND_STEP = 32.0
_DRAG_YAW_PER_PX = 0.35
# Feet within this many mm of the lowest foot count as planted for odometry.
_CONTACT_Z_EPS_MM = 8.0


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


def rotate_yaw_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    """Rotate XY around Z by yaw_deg (degrees)."""
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (x * cy - y * sy, x * sy + y * cy)


def body_to_world(
    point: tuple[float, float, float],
    body_x: float,
    body_y: float,
    body_yaw_deg: float,
) -> tuple[float, float, float]:
    """Map a body-frame point into world XYZ using robot pose on the floor."""
    x, y, z = point
    wx, wy = rotate_yaw_xy(x, y, body_yaw_deg)
    return (wx + body_x, wy + body_y, z)


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
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Axis-aligned grid segments on the Z=z plane, snapped around (origin_x, origin_y)."""
    lines: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    # Snap grid to step boundaries so it stays world-aligned while the robot moves.
    ox = math.floor(origin_x / step) * step
    oy = math.floor(origin_y / step) * step
    n = int(half // step)
    for i in range(-n, n + 1):
        t = i * step
        lines.append(((ox - half, oy + t, z), (ox + half, oy + t, z)))
        lines.append(((ox + t, oy - half, z), (ox + t, oy + half, z)))
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


def to_screen(pt: tuple[float, float], scale: float, ox: float, oy: float) -> tuple[float, float]:
    return (ox + pt[0] * scale, oy - pt[1] * scale)


def center_on_body(
    point: tuple[float, float, float],
    body_x: float,
    body_y: float,
) -> tuple[float, float, float]:
    """Subtract body XY so the robot stays framed while the floor slides."""
    return (point[0] - body_x, point[1] - body_y, point[2])


def _angle_delta_deg(from_deg: float, to_deg: float) -> float:
    """Signed shortest delta from → to in degrees (−180, 180]."""
    return (to_deg - from_deg + 180.0) % 360.0 - 180.0


def planted_indices(
    feet: Sequence[tuple[float, float, float]],
    *,
    contact_z_eps: float = _CONTACT_Z_EPS_MM,
) -> list[int]:
    """Indices of feet near the lowest Z (treated as on the ground)."""
    if not feet:
        return []
    min_z = min(f[2] for f in feet)
    return [i for i, f in enumerate(feet) if f[2] <= min_z + contact_z_eps]


def integrate_stance_odometry(
    prev_feet: Sequence[tuple[float, float, float]] | None,
    curr_feet: Sequence[tuple[float, float, float]],
    body_x: float,
    body_y: float,
    body_yaw_deg: float,
    *,
    contact_z_eps: float = _CONTACT_Z_EPS_MM,
) -> tuple[float, float, float]:
    """Keep planted feet world-fixed; return updated (body_x, body_y, body_yaw_deg).

    SoftPicrawler publishes body-frame foot tips only. When a planted foot slides
    in that frame, the body must move the opposite way in the world so the foot
    sticks to the floor — classic stance-foot / contact odometry.
    """
    if prev_feet is None or len(prev_feet) != len(curr_feet) or len(curr_feet) < 1:
        return body_x, body_y, body_yaw_deg

    prev_plant = planted_indices(prev_feet, contact_z_eps=contact_z_eps)
    curr_plant = planted_indices(curr_feet, contact_z_eps=contact_z_eps)
    shared = [i for i in curr_plant if i in prev_plant]
    if not shared:
        return body_x, body_y, body_yaw_deg

    dyaw = 0.0
    if len(shared) >= 2:
        i0, i1 = shared[0], shared[1]
        a0 = math.degrees(
            math.atan2(
                prev_feet[i1][1] - prev_feet[i0][1],
                prev_feet[i1][0] - prev_feet[i0][0],
            )
        )
        a1 = math.degrees(
            math.atan2(
                curr_feet[i1][1] - curr_feet[i0][1],
                curr_feet[i1][0] - curr_feet[i0][0],
            )
        )
        # Chord rotated in body frame → body yaw moves opposite to keep world chord fixed.
        dyaw = -_angle_delta_deg(a0, a1)

    # Average planted-foot motion in body frame (translation of contact set).
    adx = sum(curr_feet[i][0] - prev_feet[i][0] for i in shared) / len(shared)
    ady = sum(curr_feet[i][1] - prev_feet[i][1] for i in shared) / len(shared)
    # Rotate into world using yaw before this frame's turn, then invert.
    wx, wy = rotate_yaw_xy(adx, ady, body_yaw_deg)
    return body_x - wx, body_y - wy, (body_yaw_deg + dyaw) % 360.0


class BodyView(QWidget):
    """Qt widget: stick quadruped driven by `leg_coords` + world pose."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._feet: list[tuple[float, float, float]] = []
        self._raw: list[list[float]] = []
        self._body_x = 0.0
        self._body_y = 0.0
        self._body_yaw_deg = 0.0
        self._cam_yaw = _DEFAULT_YAW_DEG
        self._ref_radius = _REF_RADIUS_DEFAULT
        self._drag_last: QPointF | None = None
        self.setMinimumSize(200, 280)
        self.setAutoFillBackground(False)
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def set_pose(
        self,
        leg_coords: Sequence[Sequence[float]],
        *,
        body_x: float = 0.0,
        body_y: float = 0.0,
        body_yaw_deg: float = 0.0,
    ) -> None:
        raw = [[float(x) for x in leg[:3]] for leg in leg_coords if len(leg) >= 3]
        pose_same = (
            raw == self._raw
            and body_x == self._body_x
            and body_y == self._body_y
            and body_yaw_deg == self._body_yaw_deg
        )
        if pose_same:
            return
        self._raw = raw
        self._feet = feet_world(raw)
        self._body_x = float(body_x)
        self._body_y = float(body_y)
        self._body_yaw_deg = float(body_yaw_deg)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_last is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        pos = event.position()
        dx = pos.x() - self._drag_last.x()
        self._cam_yaw = (self._cam_yaw + dx * _DRAG_YAW_PER_PX) % 360.0
        self._drag_last = pos
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        # Scroll up → zoom in (smaller framing radius).
        factor = 0.9 if delta > 0 else 1.1
        self._ref_radius = max(_REF_RADIUS_MIN, min(_REF_RADIUS_MAX, self._ref_radius * factor))
        self.update()
        event.accept()

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

        yaw = self._cam_yaw
        bx, by, byaw = self._body_x, self._body_y, self._body_yaw_deg
        feet_body, body_z = snap_to_ground(self._feet)
        hips_body = [hip_world(i, body_z) for i in range(len(feet_body))]
        corners_body = body_corners(body_z)
        ground_z = WORLD_GROUND_Z

        def to_view(p_body: tuple[float, float, float]) -> tuple[float, float, float]:
            return center_on_body(body_to_world(p_body, bx, by, byaw), bx, by)

        feet = [to_view(f) for f in feet_body]
        hips = [to_view(hpt) for hpt in hips_body]
        corners = [to_view(c) for c in corners_body]

        usable = min(w, h) * (1.0 - 2.0 * _PAD_FRAC)
        scale = usable / (2.0 * self._ref_radius)
        ox, oy = w / 2.0, h / 2.0

        def scr(p3: tuple[float, float, float]) -> QPointF:
            x, y = to_screen(project_point(p3, yaw), scale, ox, oy)
            return QPointF(x, y)

        pen_ground = QPen(QColor("#2f3444"))
        pen_ground.setWidth(1)
        painter.setPen(pen_ground)
        for a, b in ground_grid_lines(ground_z, origin_x=bx, origin_y=by):
            a_v = center_on_body(a, bx, by)
            b_v = center_on_body(b, bx, by)
            painter.drawLine(scr(a_v), scr(b_v))
        pen_ground_edge = QPen(QColor("#8b90a0"))
        pen_ground_edge.setWidth(1)
        painter.setPen(pen_ground_edge)
        gh = _GROUND_HALF
        edge = [
            (bx - gh, by - gh, ground_z),
            (bx + gh, by - gh, ground_z),
            (bx + gh, by + gh, ground_z),
            (bx - gh, by + gh, ground_z),
        ]
        for i in range(4):
            painter.drawLine(
                scr(center_on_body(edge[i], bx, by)),
                scr(center_on_body(edge[(i + 1) % 4], bx, by)),
            )

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
