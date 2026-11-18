"""3D stick view of the PiCrawler for the sim dashboard.

Each leg is drawn as the robot builds it — coxa, femur and tibia — from the
twelve servo angles, so lifts, swings and knee bends are visible. The body
rests on the plane its weight-bearing feet define (``kinematics.support_plane``):
it pitches and rolls when the legs ask for that, and a lifted foot hangs in
the air. Feet that carry weight are filled, swinging feet are hollow, and a
leg whose joints hit a limit turns red.

The camera follows the body; the floor grid slides underneath as it walks.
Drag sideways to orbit, up/down to change the elevation, scroll to zoom.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget

from ..picrawler.kinematics import (
    BODY_HALF,
    BodyPose,
    Leveler,
    Vec3,
    leg_chains_body,
    rotate_yaw_xy,
    support_plane,
)

BODY_THICKNESS = 12.0  # mm, drawn around the hip plane

_DEFAULT_ELEVATION_DEG = 28.0
_DEFAULT_YAW_DEG = 35.0
_PAD_FRAC = 0.1
_REF_RADIUS_DEFAULT = 150.0
_REF_RADIUS_MIN = 60.0
_REF_RADIUS_MAX = 400.0
_GROUND_HALF = 240.0
_GROUND_STEP = 30.0
_DRAG_YAW_PER_PX = 0.35
_DRAG_PITCH_PER_PX = 0.25

_C_BG = QColor("#1a1d26")
_C_GRID = QColor("#2f3444")
_C_GRID_EDGE = QColor("#4a5063")
_C_BODY = QColor("#3dd6c6")
_C_LEG = QColor("#e8eaef")
_C_LEG_FAR = QColor("#9aa0b0")
_C_CLAMP = QColor("#ff6b6b")
_C_SHADOW = QColor(0, 0, 0, 90)
_C_SUPPORT = QColor(61, 214, 198, 50)
_C_HUD = QColor("#8b90a0")


def body_to_world(
    point: Vec3,
    body_x: float,
    body_y: float,
    body_yaw_deg: float,
) -> Vec3:
    """Map a level-frame point into world XYZ using the robot's pose on the floor."""
    x, y, z = point
    wx, wy = rotate_yaw_xy(x, y, body_yaw_deg)
    return (wx + body_x, wy + body_y, z)


def center_on_body(point: Vec3, body_x: float, body_y: float) -> Vec3:
    """Subtract body XY so the robot stays framed while the floor slides."""
    return (point[0] - body_x, point[1] - body_y, point[2])


def ground_grid_lines(
    z: float = 0.0,
    *,
    half: float = _GROUND_HALF,
    step: float = _GROUND_STEP,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> list[tuple[Vec3, Vec3]]:
    """Axis-aligned grid segments on the Z=z plane, snapped around (origin_x, origin_y)."""
    lines: list[tuple[Vec3, Vec3]] = []
    ox = math.floor(origin_x / step) * step
    oy = math.floor(origin_y / step) * step
    n = int(half // step)
    for i in range(-n, n + 1):
        t = i * step
        lines.append(((ox - half, oy + t, z), (ox + half, oy + t, z)))
        lines.append(((ox + t, oy - half, z), (ox + t, oy + half, z)))
    return lines


def rotate_yaw_pitch(point: Vec3, yaw_deg: float, pitch_deg: float) -> Vec3:
    """Rotate a point into camera space (yaw around Z, then pitch around X)."""
    x, y, z = point
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1 = x * cy - y * sy
    y1 = x * sy + y * cy
    pitch = math.radians(pitch_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (x1, y1 * cp - z * sp, y1 * sp + z * cp)


def project_point(
    point: Vec3, yaw_deg: float, pitch_deg: float = _DEFAULT_ELEVATION_DEG
) -> tuple[float, float]:
    """Orthographic projection to 2D (x right, y up before the screen flip)."""
    x_cam, _, z_cam = rotate_yaw_pitch(point, yaw_deg, pitch_deg)
    return (x_cam, z_cam)


def depth(point: Vec3, yaw_deg: float, pitch_deg: float) -> float:
    """Distance into the screen; larger is further away."""
    return rotate_yaw_pitch(point, yaw_deg, pitch_deg)[1]


def to_screen(pt: tuple[float, float], scale: float, ox: float, oy: float) -> tuple[float, float]:
    return (ox + pt[0] * scale, oy - pt[1] * scale)


class Scene:
    """Everything one frame draws, in the view frame (world, centred on the body)."""

    def __init__(self, pose: BodyPose) -> None:
        self.pose = pose
        chains = leg_chains_body(pose.servo_angles)
        feet = [chain[3] for chain in chains]
        self.support = support_plane(feet)
        self.level = Leveler.for_support(self.support)
        self.chains = [[self._view(p) for p in chain] for chain in chains]
        top = [self._view((hx, hy, BODY_THICKNESS / 2)) for hx, hy in _corners()]
        bottom = [self._view((hx, hy, -BODY_THICKNESS / 2)) for hx, hy in _corners()]
        self.body_top, self.body_bottom = top, bottom
        self.nose = self._view((0.0, BODY_HALF + 14.0, BODY_THICKNESS / 2))
        self.centre = self._view((0.0, 0.0, 0.0))

    def _view(self, p: Vec3) -> Vec3:
        pose = self.pose
        world = body_to_world(self.level.apply(p), pose.body_x, pose.body_y, pose.body_yaw_deg)
        return center_on_body(world, pose.body_x, pose.body_y)

    @property
    def contact(self) -> tuple[bool, ...]:
        return self.support.contact


def _corners() -> list[tuple[float, float]]:
    return [
        (BODY_HALF, BODY_HALF),
        (-BODY_HALF, BODY_HALF),
        (-BODY_HALF, -BODY_HALF),
        (BODY_HALF, -BODY_HALF),
    ]


class BodyView(QWidget):
    """Qt widget: the crawler from its servo angles, standing on its support plane."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pose: BodyPose | None = None
        self._scene: Scene | None = None
        self._cam_yaw = _DEFAULT_YAW_DEG
        self._cam_pitch = _DEFAULT_ELEVATION_DEG
        self._ref_radius = _REF_RADIUS_DEFAULT
        self._drag_last: QPointF | None = None
        self.setMinimumSize(200, 280)
        self.setAutoFillBackground(False)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def set_body_pose(self, pose: BodyPose | None) -> None:
        if pose is None or pose is self._pose or len(pose.servo_angles) < 12:
            return
        self._pose = pose
        self._scene = Scene(pose)
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
        self._cam_yaw = (self._cam_yaw + (pos.x() - self._drag_last.x()) * _DRAG_YAW_PER_PX) % 360
        pitch = self._cam_pitch + (pos.y() - self._drag_last.y()) * _DRAG_PITCH_PER_PX
        self._cam_pitch = max(5.0, min(85.0, pitch))
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
        factor = 0.9 if delta > 0 else 1.1
        self._ref_radius = max(_REF_RADIUS_MIN, min(_REF_RADIUS_MAX, self._ref_radius * factor))
        self.update()
        event.accept()

    def paintEvent(self, event: object) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), _C_BG)
        scene = self._scene
        if scene is None:
            painter.setPen(_C_HUD)
            painter.drawText(
                self.rect(), int(Qt.AlignmentFlag.AlignCenter), "no pose yet\n(move to publish)"
            )
            painter.end()
            return
        self._paint_scene(painter, scene)
        painter.end()

    def _paint_scene(self, painter: QPainter, scene: Scene) -> None:
        w, h = float(self.width()), float(self.height())
        yaw, pitch = self._cam_yaw, self._cam_pitch
        scale = min(w, h) * (1.0 - 2.0 * _PAD_FRAC) / (2.0 * self._ref_radius)
        # Aim at a point a little above the floor so the robot sits mid-frame.
        aim = project_point((0.0, 0.0, scene.level.height * 0.5), yaw, pitch)
        ox, oy = w / 2.0 - aim[0] * scale, h / 2.0 + aim[1] * scale

        def scr(p: Vec3) -> QPointF:
            x, y = to_screen(project_point(p, yaw, pitch), scale, ox, oy)
            return QPointF(x, y)

        pose = scene.pose
        self._paint_floor(painter, scr, pose)

        # Shadow of the body and the triangle or square the robot stands on.
        shadow = QPolygonF([scr((p[0], p[1], 0.0)) for p in scene.body_bottom])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_C_SHADOW)
        painter.drawPolygon(shadow)
        planted = [
            (chain[3][0], chain[3][1], 0.0)
            for chain, down in zip(scene.chains, scene.contact, strict=False)
            if down
        ]
        if len(planted) >= 3:
            painter.setBrush(_C_SUPPORT)
            painter.drawPolygon(QPolygonF([scr(p) for p in _convex_order(planted)]))
        painter.setBrush(_C_HUD)
        painter.drawEllipse(scr((scene.centre[0], scene.centre[1], 0.0)), 2.5, 2.5)

        # Far legs first, then the body, then near legs, so overlaps read right.
        centre_depth = depth(scene.centre, yaw, pitch)
        order = sorted(
            range(len(scene.chains)),
            key=lambda i: -depth(scene.chains[i][1], yaw, pitch),
        )
        far = [i for i in order if depth(scene.chains[i][1], yaw, pitch) > centre_depth]
        near = [i for i in order if i not in far]
        for i in far:
            self._paint_leg(painter, scr, scene, i, far=True)
        self._paint_body(painter, scr, scene)
        for i in near:
            self._paint_leg(painter, scr, scene, i, far=False)

        tilt_pitch, tilt_roll = scene.level.tilt_deg()
        painter.setPen(_C_HUD)
        painter.drawText(
            8,
            int(h) - 8,
            f"pitch {tilt_pitch:+.0f}°  roll {tilt_roll:+.0f}°  height {scene.level.height:.0f} mm",
        )

    def _paint_floor(self, painter: QPainter, scr, pose: BodyPose) -> None:
        bx, by = pose.body_x, pose.body_y
        pen = QPen(_C_GRID)
        pen.setWidth(1)
        painter.setPen(pen)
        for a, b in ground_grid_lines(0.0, origin_x=bx, origin_y=by):
            painter.drawLine(scr(center_on_body(a, bx, by)), scr(center_on_body(b, bx, by)))
        pen = QPen(_C_GRID_EDGE)
        painter.setPen(pen)
        gh = _GROUND_HALF
        ox = math.floor(bx / _GROUND_STEP) * _GROUND_STEP
        oy = math.floor(by / _GROUND_STEP) * _GROUND_STEP
        edge = [(ox - gh, oy - gh), (ox + gh, oy - gh), (ox + gh, oy + gh), (ox - gh, oy + gh)]
        for i in range(4):
            a = center_on_body((*edge[i], 0.0), bx, by)
            b = center_on_body((*edge[(i + 1) % 4], 0.0), bx, by)
            painter.drawLine(scr(a), scr(b))

    def _paint_body(self, painter: QPainter, scr, scene: Scene) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(61, 214, 198, 40))
        painter.drawPolygon(QPolygonF([scr(p) for p in scene.body_top]))
        pen = QPen(_C_BODY)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        top, bot = scene.body_top, scene.body_bottom
        for i in range(4):
            painter.drawLine(scr(top[i]), scr(top[(i + 1) % 4]))
            painter.drawLine(scr(bot[i]), scr(bot[(i + 1) % 4]))
            painter.drawLine(scr(top[i]), scr(bot[i]))
        # Nose marker: which way is forward.
        front_mid = (
            (top[0][0] + top[1][0]) / 2,
            (top[0][1] + top[1][1]) / 2,
            (top[0][2] + top[1][2]) / 2,
        )
        painter.drawLine(scr(front_mid), scr(scene.nose))

    def _paint_leg(self, painter: QPainter, scr, scene: Scene, i: int, *, far: bool) -> None:
        chain = scene.chains[i]
        pose = scene.pose
        clamped = i < len(pose.clamped) and pose.clamped[i]
        colour = _C_CLAMP if clamped else (_C_LEG_FAR if far else _C_LEG)
        hip, coxa, knee, foot = (scr(p) for p in chain)
        for a, b, width in ((hip, coxa, 4.0), (coxa, knee, 3.0), (knee, foot, 2.2)):
            pen = QPen(colour)
            pen.setWidthF(width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(a, b)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(knee, 3.0, 3.0)
        # Swinging foot: a hollow ring and a drop line to the floor.
        down = i < len(scene.contact) and scene.contact[i]
        if down:
            painter.setBrush(_C_BODY)
            painter.drawEllipse(foot, 4.5, 4.5)
        else:
            pen = QPen(_C_BODY)
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(foot, 4.5, 4.5)
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            p = chain[3]
            painter.drawLine(foot, scr((p[0], p[1], 0.0)))


def _convex_order(points: Sequence[Vec3]) -> list[Vec3]:
    """Points sorted around their centroid, so they draw as a simple polygon."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
