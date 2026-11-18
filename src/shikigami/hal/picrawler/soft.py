"""SoftPicrawler — pure-Python twin of SunFounder's Picrawler API (no robot_hat).

Reimplements the public surface used by our HAL (`do_action`, `do_step`, `set_angle`,
IK, and named gaits) so Mac sim stays close to the Pi device. Geometry constants and
gait patterns follow SunFounder's PiCrawler docs / library behaviour
(https://github.com/sunfounder/picrawler); this module does not import that package.

With ``realtime=True`` every keyframe plays the way ``robot_hat.Robot.servo_move``
drives the servos: all joints move linearly in joint space, and the move takes as
long as it does on the robot (see ``kinematics.keyframe_secs``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from .kinematics import (
    feet_world,
    keyframe_secs,
    leg_chain_local,
    support_plane,
)

# Leg order: right-front, left-front, left-rear, right-rear
LEG_NAMES = ("RF", "LF", "LR", "RR")
# Per-leg joint order written by set_angle: [beta, alpha, gamma]
JOINT_NAMES = ("tibia", "femur", "coxa")
SERVO_LABELS: tuple[str, ...] = tuple(
    f"{leg} {joint}" for leg in LEG_NAMES for joint in JOINT_NAMES
)

GAIT_ACTIONS = (
    "forward",
    "backward",
    "turn left",
    "turn right",
    "turn left angle",
    "turn right angle",
)


def _check_stand(func: Callable[..., list]) -> Callable[..., list]:
    def wrapper(self: MoveList) -> list:
        action: list = []
        if not self.is_stand():
            action += self.stand
        action += func(self)
        return action

    return wrapper


def _normal_action(mode: int) -> Callable[[Callable[..., list]], Callable[..., list]]:
    def decorator(func: Callable[..., list]) -> Callable[..., list]:
        def wrapper(self: MoveList) -> list:
            if self.stand_position == 0:
                return func(self)
            temp = func(self)
            remapped: list = []
            for step in temp:
                if mode == 0:
                    remapped.append([step[1], step[0], step[3], step[2]])
                else:
                    remapped.append([step[2], step[3], step[0], step[1]])
            return remapped

        return wrapper

    return decorator


class MoveList(dict):
    """Named gait frame sequences — same keys as SunFounder Picrawler.MoveList."""

    LENGTH_SIDE = 77
    X_DEFAULT = 45
    X_TURN = 70
    X_START = 0
    Y_DEFAULT = 45
    Y_TURN = 130
    Y_WAVE = 120
    Y_START = 0
    Z_DEFAULT = -50
    Z_UP = -30
    Z_WAVE = 60
    Z_TURN = -40
    Z_PUSH = -76

    TEMP_A = math.sqrt(pow(2 * X_DEFAULT + LENGTH_SIDE, 2) + pow(Y_DEFAULT, 2))
    TEMP_B = 2 * (Y_START + Y_DEFAULT) + LENGTH_SIDE
    TEMP_C = math.sqrt(
        pow(2 * X_DEFAULT + LENGTH_SIDE, 2) + pow(2 * Y_START + Y_DEFAULT + LENGTH_SIDE, 2)
    )
    TEMP_ALPHA = math.acos((pow(TEMP_A, 2) + pow(TEMP_B, 2) - pow(TEMP_C, 2)) / 2 / TEMP_A / TEMP_B)
    TURN_X1 = (TEMP_A - LENGTH_SIDE) / 2
    TURN_Y1 = Y_START + Y_DEFAULT / 2
    TURN_X0 = TURN_X1 - TEMP_B * math.cos(TEMP_ALPHA)
    TURN_Y0 = TEMP_B * math.sin(TEMP_ALPHA) - TURN_Y1 - LENGTH_SIDE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.z_current = self.Z_UP
        self.stand_position = 0
        self.ready_state = 0
        self.angle = 30

    def __getitem__(self, item: str) -> Any:  # type: ignore[override]
        return getattr(self, item.replace(" ", "_"))

    def turn_angle_coord(self, angle: float) -> list[float]:
        a = math.atan(self.Y_DEFAULT / (self.X_DEFAULT + self.LENGTH_SIDE / 2))
        angle1 = a / math.pi * 180
        r1 = math.sqrt(pow(self.Y_DEFAULT, 2) + pow(self.X_DEFAULT + self.LENGTH_SIDE / 2, 2))
        x1 = r1 * math.cos((angle1 - angle) * math.pi / 180) - self.LENGTH_SIDE / 2
        y1 = r1 * math.sin((angle1 - angle) * math.pi / 180)
        x2 = (self.X_DEFAULT + self.LENGTH_SIDE / 2) * math.cos(angle * math.pi / 180) - (
            self.LENGTH_SIDE / 2
        )
        y2 = (self.X_DEFAULT + self.LENGTH_SIDE / 2) * math.sin(angle * math.pi / 180)
        b = math.atan((self.X_DEFAULT + self.LENGTH_SIDE / 2) / (self.Y_DEFAULT + self.LENGTH_SIDE))
        angle2 = b / math.pi * 180
        r2 = math.sqrt(
            pow(self.X_DEFAULT + self.LENGTH_SIDE / 2, 2)
            + pow(self.Y_DEFAULT + self.LENGTH_SIDE, 2)
        )
        x3 = r2 * math.sin((angle2 - angle) * math.pi / 180) - self.LENGTH_SIDE / 2
        y3 = r2 * math.cos((angle2 - angle) * math.pi / 180) - self.LENGTH_SIDE
        x3 += 10
        return [x1, y1, x2, y2, x3, y3]

    @property
    @_normal_action(0)
    def sit(self) -> list:
        self.z_current = self.Z_UP
        return [
            [
                [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
                [self.X_TURN, self.Y_START, self.z_current],
                [self.X_TURN, self.Y_START, self.z_current],
                [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
            ]
        ]

    @property
    def ready(self) -> list:
        self.ready_state = 1
        return [
            [
                [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
                [self.X_TURN, self.Y_START, self.z_current],
                [self.X_TURN, self.Y_START, self.z_current],
                [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
            ]
        ]

    @property
    @_normal_action(0)
    def stand(self) -> list:
        frames: list = []
        if self.ready_state == 0:
            frames += self.ready
        self.z_current = self.Z_DEFAULT
        z = self.z_current
        for scale in (0.35, 0.55, 0.75, 0.9, 1.0):
            frames.append(
                [
                    [self.X_DEFAULT, self.Y_DEFAULT, z * scale],
                    [self.X_DEFAULT, self.Y_START, z * scale],
                    [self.X_DEFAULT, self.Y_START, z * scale],
                    [self.X_DEFAULT, self.Y_DEFAULT, z * scale],
                ]
            )
        return frames

    def is_sit(self) -> bool:
        return self.z_current == self.Z_UP

    def is_stand(self) -> bool:
        return self.z_current == self.Z_DEFAULT

    @property
    @_check_stand
    @_normal_action(0)
    def forward(self) -> list:
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        return [
            [[xd, yd, z], [xt, xs, zu], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [xd, yd * 2, zu], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [xd, yd * 2, z], [xd, xs, z], [xd, yd, z]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xd, yd * 2, z]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xd, yd * 2, zu]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xt, xs, zu]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
        ]

    @property
    @_check_stand
    @_normal_action(0)
    def backward(self) -> list:
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        return [
            [[xd, yd, z], [xd, xs, z], [xt, xs, zu], [xd, yd, z]],
            [[xd, yd, z], [xd, xs, z], [xd, yd * 2, zu], [xd, yd, z]],
            [[xd, yd, z], [xd, xs, z], [xd, yd * 2, z], [xd, yd, z]],
            [[xd, yd * 2, z], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
            [[xd, yd * 2, zu], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
            [[xt, xs, zu], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
        ]

    @property
    @_check_stand
    @_normal_action(1)
    def turn_left(self) -> list:
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        tx0, ty0, tx1, ty1 = self.TURN_X0, self.TURN_Y0, self.TURN_X1, self.TURN_Y1
        return [
            [[xd, yd, z], [xd, xs, z], [xt, xs, zu], [xd, yd, z]],
            [[tx1, ty1, z], [tx1, ty1, z], [tx0, ty0, zu], [tx0, ty0, z]],
            [[tx1, ty1, z], [tx1, ty1, z], [tx0, ty0, z], [tx0, ty0, z]],
            [[tx1, ty1, z], [tx1, ty1, z], [tx0, ty0, z], [tx0, ty0, zu]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xt, xs, zu]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
        ]

    @property
    @_check_stand
    @_normal_action(1)
    def turn_right(self) -> list:
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        tx0, ty0, tx1, ty1 = self.TURN_X0, self.TURN_Y0, self.TURN_X1, self.TURN_Y1
        return [
            [[xd, yd, z], [xt, xs, zu], [xd, xs, z], [xd, yd, z]],
            [[tx0, ty0, z], [tx0, ty0, zu], [tx1, ty1, z], [tx1, tx1, z]],
            [[tx0, ty0, z], [tx0, ty0, z], [tx1, ty1, z], [tx1, tx1, z]],
            [[tx0, ty0, zu], [tx0, ty0, z], [tx1, ty1, z], [tx1, tx1, z]],
            [[xt, xs, zu], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
            [[xd, xs, z], [xd, yd, z], [xd, yd, z], [xd, xs, z]],
        ]

    @property
    def push_up(self) -> list:
        frames: list = []
        if not self.is_sit():
            frames += self.sit
        frames += [
            [
                [self.X_TURN, self.Y_START, self.Z_TURN],
                [self.X_TURN, self.Y_START, self.Z_TURN],
                [self.X_START, self.Y_TURN, self.Z_TURN],
                [self.X_START, self.Y_TURN, self.Z_TURN],
            ],
            [
                [self.X_TURN, self.Y_START, self.Z_PUSH],
                [self.X_TURN, self.Y_START, self.Z_PUSH],
                [self.X_START, self.Y_TURN, self.Z_TURN],
                [self.X_START, self.Y_TURN, self.Z_TURN],
            ],
            [
                [self.X_TURN, self.Y_START, self.Z_TURN],
                [self.X_TURN, self.Y_START, self.Z_TURN],
                [self.X_START, self.Y_TURN, self.Z_TURN],
                [self.X_START, self.Y_TURN, self.Z_TURN],
            ],
        ]
        if self.stand_position == 0:
            frames.append(
                [
                    [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
                    [self.X_TURN, self.Y_START, self.z_current],
                    [self.X_TURN, self.Y_START, self.z_current],
                    [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
                ]
            )
        else:
            frames.append(
                [
                    [self.X_TURN, self.Y_START, self.z_current],
                    [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
                    [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
                    [self.X_TURN, self.Y_START, self.z_current],
                ]
            )
        return frames

    @property
    @_check_stand
    @_normal_action(0)
    def wave(self) -> list:
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        return [
            [[xd, yd, z], [xt, xs, zu], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [self.X_START, self.Y_WAVE, self.Z_WAVE], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [self.X_START, self.Y_WAVE, zu], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [self.X_START, self.Y_WAVE, self.Z_WAVE], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [self.X_START, self.Y_WAVE, zu], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [xt, xs, zu], [xd, xs, z], [xd, yd, z]],
            [[xd, yd, z], [xd, xs, z], [xd, xs, z], [xd, yd, z]],
        ]

    @property
    @_check_stand
    @_normal_action(1)
    def look_left(self) -> list:
        li = self.turn_angle_coord(self.angle)
        t1 = [*li[0:2], self.z_current]
        t2 = [*li[2:4], self.z_current]
        t3 = [*li[4:6], self.z_current]
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        return [
            [[xd, yd, z], [xd, xs, z], [xt, xs, zu], [xd, yd, z]],
            [t1, t2, [xt, xs, zu], t3],
        ]

    @property
    @_check_stand
    @_normal_action(1)
    def look_right(self) -> list:
        li = self.turn_angle_coord(self.angle)
        t1 = [*li[0:2], self.z_current]
        t2 = [*li[2:4], self.z_current]
        t3 = [*li[4:6], self.z_current]
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        return [
            [[xd, yd, z], [xt, xs, zu], [xd, xs, z], [xd, yd, z]],
            [t3, [xt, xs, zu], t2, t1],
        ]

    @property
    @_check_stand
    @_normal_action(1)
    def turn_left_angle(self) -> list:
        li = self.turn_angle_coord(self.angle)
        x1, y1, x2, y2, x3, y3 = li
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        return [
            [[xd, yd, z], [xd, xs, z], [xt, xs, zu], [xd, yd, z]],
            [[x1, y1, z], [x2, y2, z], [xt, xs, zu], [x3, y3, z]],
            [[x1, y1, z], [x2, y2, z], [xd, yd, z], [x3, y3, z]],
            [[x1, y1, z], [x2, y2, z], [xd, yd, z], [x3, y3, zu]],
            [[x1, y1, z], [x2, y2, z], [xt, yd, z], [xt, xs, zu]],
            [[x1, y1, z], [x2, y2, z], [xt, yd, z], [xd, xs, z]],
        ]

    @property
    @_check_stand
    @_normal_action(1)
    def turn_right_angle(self) -> list:
        li = self.turn_angle_coord(self.angle)
        x1, y1, x2, y2, x3, y3 = li
        z, zu = self.z_current, self.Z_UP
        xd, yd, xs, xt = self.X_DEFAULT, self.Y_DEFAULT, self.Y_START, self.X_TURN
        return [
            [[xd, yd, z], [xt, xs, zu], [xd, xs, z], [xd, yd, z]],
            [[x3, y3, z], [xt, xs, zu], [x2, y2, z], [x1, y1, z]],
            [[x3, y3, z], [xd, yd, z], [x2, y2, z], [x1, y1, z]],
            [[x3, y3, zu], [xd, yd, z], [x2, y2, z], [x1, y1, z]],
            [[xt, xs, zu], [xd, yd, z], [x2, y2, z], [x1, y1, z]],
            [[xd, xs, z], [xd, yd, z], [x2, y2, z], [x1, y1, z]],
        ]

    @property
    @_check_stand
    @_normal_action(0)
    def look_up(self) -> list:
        return [
            [
                [self.X_DEFAULT, self.Y_DEFAULT, self.Z_DEFAULT],
                [self.X_DEFAULT, self.Y_START, self.Z_DEFAULT],
                [self.X_TURN, self.Y_START, self.Z_UP],
                [self.X_DEFAULT, self.Y_DEFAULT, self.Z_UP],
            ]
        ]

    @property
    @_check_stand
    @_normal_action(0)
    def look_down(self) -> list:
        z = self.z_current
        return [
            [
                [self.X_DEFAULT, self.Y_DEFAULT, self.Z_UP],
                [self.X_TURN, self.Y_START, self.Z_UP],
                [self.X_DEFAULT, self.Y_START, z],
                [self.X_DEFAULT, self.Y_DEFAULT, z],
            ]
        ]

    def rotate_body_absolute_x(self, degree_x: float) -> list:
        rad = degree_x * math.pi / 180
        dz = (self.LENGTH_SIDE / 2 + self.Y_DEFAULT) * math.sin(rad)
        dy = (self.LENGTH_SIDE / 2 + self.Y_DEFAULT) * (1 - math.cos(rad))
        return [
            [self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT + dz],
            [self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT - dz],
            [self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT - dz],
            [self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT + dz],
        ]

    def rotate_body_absolute_y(self, degree_y: float) -> list:
        rad = degree_y * math.pi / 180
        dz = (self.LENGTH_SIDE / 2 + self.X_DEFAULT) * math.sin(rad)
        dx = (self.LENGTH_SIDE / 2 + self.X_DEFAULT) * (1 - math.cos(rad))
        return [
            [self.X_DEFAULT - dx, self.Y_DEFAULT, self.Z_DEFAULT + dz],
            [self.X_DEFAULT - dx, self.Y_DEFAULT, self.Z_DEFAULT + dz],
            [self.X_DEFAULT - dx, self.Y_DEFAULT, self.Z_DEFAULT - dz],
            [self.X_DEFAULT - dx, self.Y_DEFAULT, self.Z_DEFAULT - dz],
        ]

    def move_body_absolute(self, x: float, y: float, z: float) -> list:
        return [
            [self.X_DEFAULT - x, self.Y_DEFAULT - y, self.Z_TURN - z],
            [self.X_DEFAULT + x, self.Y_DEFAULT - y, self.Z_TURN - z],
            [self.X_DEFAULT + x, self.Y_DEFAULT + y, self.Z_TURN - z],
            [self.X_DEFAULT - x, self.Y_DEFAULT + y, self.Z_TURN - z],
        ]

    @staticmethod
    def to_rad(deg: float) -> float:
        return deg * math.pi / 180

    def _circle(self, step_deg: int, *, reverse: bool = False, rise=None) -> list:
        """Body circles of radius 40 mm, one frame per ``step_deg``; ``rise(i)`` lifts z."""
        angles = range(360, 0, -step_deg) if reverse else range(0, 360, step_deg)
        return [
            self.move_body_absolute(
                40 * math.sin(self.to_rad(i)),
                40 * math.cos(self.to_rad(i)),
                rise(i) if rise is not None else 0,
            )
            for i in angles
        ]

    @property
    def dance(self) -> list:
        """The robot's full dance, frame for frame (sit, circles, tilts, rising spiral)."""
        frames: list = []
        if not self.is_sit():
            frames += self.sit
        z = self.z_current
        frames.append([[self.X_DEFAULT, self.Y_DEFAULT, z]] * 4)
        frames += self._circle(5)
        frames += self._circle(5, reverse=True)
        frames.append(self.rotate_body_absolute_x(-20))
        frames.append(self.rotate_body_absolute_x(20))
        frames.append(self.move_body_absolute(0, 0, 0))
        frames.append(self.rotate_body_absolute_y(-20))
        frames.append(self.rotate_body_absolute_y(20))
        for j in range(3):
            frames += self._circle(3, rise=lambda i, j=j: (i / 360.0 + j) * 15)
        for j in range(3, 0, -1):
            frames += self._circle(3, rise=lambda i, j=j: ((360 - i) / 360.0 + j - 1) * 15)
        frames.append(self.move_body_absolute(0, 0, 0))
        return frames


class SoftPicrawler:
    """Drop-in soft stand-in for ``picrawler.Picrawler`` (angles in memory only)."""

    A = 48
    B = 78
    C = 33
    PIN_LIST = [9, 10, 11, 3, 4, 5, 0, 1, 2, 6, 7, 8]

    def __init__(
        self,
        pin_list: list[int] | None = None,
        init_angles: list[float] | None = None,
        *,
        realtime: bool = False,
    ) -> None:
        self.pin_list = list(pin_list or self.PIN_LIST)
        self.move_list = MoveList()
        self.move_list_add: dict[str, list] = {"my action": None}  # type: ignore[dict-item]
        # Built once, like the real library: do_step("stand") replays these frames
        # and does not touch move_list.z_current, so the first gait after it
        # stands up again — the robot does that too.
        self.step_list = {
            "stand": self.move_list["stand"],
            "sit": self.move_list["sit"],
        }
        self.stand_position = 0
        self.current_coord = [[60.0, 0.0, -30.0] for _ in range(4)]
        self.coord_temp = [[60.0, 0.0, -30.0] for _ in range(4)]
        self.servo_positions = list(init_angles) if init_angles is not None else [0.0] * 12
        self.offset = [0.0] * 12
        self.realtime = bool(realtime)  # play keyframes at the robot's speed; False = snap
        self.on_pose: Callable[[], None] | None = None
        # Live multiplier from SimState.playback_speed (dashboard Speed control).
        self.get_playback_speed: Callable[[], float] | None = None
        # Body-view sample rate while interpolating between gait keyframes.
        self.display_hz = 30.0
        # Per leg, for the keyframe in progress: lifted at its start or end (swing),
        # and whether its joints hit a limit.
        self.swing: list[bool] = [False] * 4
        self.clamped: list[bool] = [False] * 4

    def coord2polar(self, coord: list[float]) -> list[float]:
        x, y, z = coord
        length = math.sqrt(x**2 + y**2 + z**2) or 0.1
        if length < self.C:
            scale = self.C / length
            x, y, z = scale * x, scale * y, scale * z
        elif length > (self.A + self.B + self.C):
            scale = (self.A + self.B + self.C) / length
            x, y, z = scale * x, scale * y, scale * z

        w = math.sqrt(x**2 + y**2)
        v = w - self.C
        u = math.sqrt(z**2 + v**2)
        u = max(30.0, min(91.58, u))
        cos_angle1 = (self.B**2 + self.A**2 - u**2) / (2 * self.B * self.A)
        cos_angle1 = max(-1.0, min(1.0, cos_angle1))
        beta = math.acos(cos_angle1)
        angle1 = math.atan2(z, v)
        cos_a2 = (self.A**2 + u**2 - self.B**2) / (2 * self.A * u)
        cos_a2 = max(-1.0, min(1.0, cos_a2))
        angle2 = math.acos(cos_a2)
        alpha = angle2 + angle1
        gamma = math.atan2(y, x)
        alpha = 90 - alpha / math.pi * 180
        beta = beta / math.pi * 180 - 90
        gamma = -(gamma / math.pi * 180 - 45)
        return [round(alpha, 4), round(beta, 4), round(gamma, 4)]

    def polar2coord(self, angles: list[float]) -> list[float]:
        alpha, beta, gamma = angles
        l1 = math.sqrt(
            self.A**2 + self.B**2 - 2 * self.A * self.B * math.cos((90 + alpha) / 180 * math.pi)
        )
        angle = math.acos((self.A**2 + l1**2 - self.B**2) / (2 * self.A * l1)) * 180 / math.pi
        angle = 90 - beta - angle
        length = l1 * math.cos(angle * math.pi / 180) + self.C
        x = length * math.sin((45 + gamma) * math.pi / 180)
        y = length * math.cos((45 + gamma) * math.pi / 180)
        z = l1 * math.sin(angle * math.pi / 180)
        return [round(x, 4), round(y, 4), round(z, 4)]

    @staticmethod
    def limit(lo: float, hi: float, x: float) -> float:
        return max(lo, min(hi, x))

    def limit_angle(self, angles: list[float]) -> tuple[bool, list[float]]:
        alpha, beta, gamma = angles
        limited = False
        a2 = self.limit(-90, 90, alpha)
        b2 = self.limit(-10, 90, beta)
        g2 = self.limit(-60, 60, gamma)
        if (a2, b2, g2) != (alpha, beta, gamma):
            limited = True
        return limited, [a2, b2, g2]

    def do_action(self, motion_name: str, step: int = 1, speed: int = 50) -> None:
        try:
            for _ in range(step):
                self.move_list.stand_position = self.stand_position
                if motion_name in GAIT_ACTIONS:
                    self.stand_position = (self.stand_position + 1) & 1
                action = self.move_list[motion_name]
                for frame in action:
                    self.do_step(frame, speed=speed)
        except AttributeError:
            action_add = self.move_list_add.get(motion_name)
            if action_add is None:
                raise KeyError(f"No such action: {motion_name!r}") from None
            for _ in range(step):
                for frame in action_add:
                    self.do_step(frame, speed=speed)

    def _playback_speed(self) -> float:
        if self.get_playback_speed is None:
            return 1.0
        return max(0.25, min(2.0, float(self.get_playback_speed())))

    def _solve(
        self, coords: list[list[float]], *, israise: bool = False
    ) -> tuple[list[float], list[list[float]], list[bool]]:
        """IK and joint limits for one keyframe: (servo angles, foot coords, clamped legs).

        Clamped legs end where their limited joints put them, as on the robot.
        """
        translate: list[float] = []
        clamped: list[bool] = []
        for coord in coords:
            alpha, beta, gamma = self.coord2polar(coord)
            limited, angles = self.limit_angle([beta, alpha, gamma])
            translate.extend(angles)
            clamped.append(limited)
        if any(clamped):
            if israise:
                raise ValueError("Coordinates out of controllable range.")
            feet = [self.polar2coord(translate[i * 3 : i * 3 + 3]) for i in range(4)]
        else:
            feet = [list(c) for c in coords]
        return translate, feet, clamped

    def _commit(self, angles: list[float], coords: list[list[float]]) -> None:
        self.servo_positions = list(angles)
        self.current_coord = [list(c) for c in coords]
        self.coord_temp = [list(c) for c in coords]
        if self.on_pose is not None:
            self.on_pose()

    def _commit_coords(self, coords: list[list[float]], *, israise: bool = False) -> list[float]:
        """IK + publish one pose (no sleep)."""
        angles, feet, clamped = self._solve(coords, israise=israise)
        self.clamped = clamped
        self._commit(angles, feet)
        return list(angles)

    @staticmethod
    def _lifted(coords: list[list[float]]) -> list[bool]:
        support = support_plane(feet_world(coords))
        return [not c for c in support.contact]

    def set_angle(
        self, angles_list: list[list[float]], speed: int = 50, israise: bool = False
    ) -> list[float]:
        """Apply joint angles; in realtime mode wait as long as the servos would take."""
        translate: list[float] = []
        limited_any = False
        for angles in angles_list:
            limited, clipped = self.limit_angle(angles)
            translate.extend(clipped)
            limited_any = limited_any or limited
        if limited_any:
            if israise:
                raise ValueError("Coordinates out of controllable range.")
            coords = []
            for i in range(4):
                coords.append(
                    self.polar2coord([translate[i * 3], translate[i * 3 + 1], translate[i * 3 + 2]])
                )
            self.current_coord = [list(c) for c in coords]
        else:
            self.current_coord = [list(c) for c in self.coord_temp]
        max_delta = max(
            (abs(a - b) for a, b in zip(translate, self.servo_positions, strict=False)),
            default=0.0,
        )
        self.servo_positions = list(translate)
        if self.on_pose is not None:
            self.on_pose()
        if self.realtime:
            time.sleep(keyframe_secs(max_delta, speed) / self._playback_speed())
        return list(translate)

    def do_step(self, _step: str | list, speed: int = 50, israise: bool = False) -> None:
        if isinstance(_step, str):
            if _step not in self.step_list:
                raise KeyError(f"Unknown step gait: {_step!r}")
            for one_step in self.step_list[_step]:
                self._apply_coords(one_step, speed, israise)
            return
        if isinstance(_step, list):
            self._apply_coords(_step, speed, israise)
            return
        raise TypeError(f'Invalid "_step" parameter: {type(_step)!r}')

    def _apply_coords(self, one_step: list, speed: int, israise: bool = False) -> None:
        """Move to one keyframe the way ``servo_move`` does on the robot.

        Every joint moves linearly from its current angle to its target, so the
        feet travel along the arcs the real legs make, and the move takes as long
        as on the robot. Poses are published at ``display_hz``.
        """
        target = [[float(v) for v in leg[:3]] for leg in one_step]
        if len(target) != 4:
            raise ValueError(f"expected 4 leg coords, got {len(target)}")
        angles, feet, clamped = self._solve(target, israise=israise)
        start_angles = list(self.servo_positions)
        start_feet = [list(c) for c in self.current_coord]
        self.clamped = clamped
        lifted_start = self._lifted(start_feet) if len(start_feet) == 4 else [False] * 4
        lifted_end = self._lifted(feet)
        self.swing = [a or b for a, b in zip(lifted_start, lifted_end, strict=True)]

        if not self.realtime or len(start_angles) != 12:
            self._commit(angles, feet)
            return

        max_delta = max(abs(a - b) for a, b in zip(angles, start_angles, strict=True))
        duration = keyframe_secs(max_delta, speed) / self._playback_speed()
        n = max(1, int(round(duration * max(1.0, float(self.display_hz)))))
        step_dt = duration / n
        for i in range(1, n + 1):
            if i == n:
                self._commit(angles, feet)
            else:
                t = i / n
                pose = [s0 + (s1 - s0) * t for s0, s1 in zip(start_angles, angles, strict=True)]
                tips = [list(leg_chain_local(pose[k * 3 : k * 3 + 3])[3]) for k in range(4)]
                self._commit(pose, tips)
            time.sleep(step_dt)

    def current_step_all_leg_angle(self) -> list[float]:
        return list(self.servo_positions)

    def current_step_all_leg_value(self) -> list[list[float]]:
        return [list(c) for c in self.current_coord]

    def add_action(self, action_name: str, action_list: list) -> None:
        self.move_list_add[action_name] = action_list
