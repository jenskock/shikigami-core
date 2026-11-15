"""Spring-driven eye pose interpolation with secondary motion."""

from __future__ import annotations

from dataclasses import dataclass, field

from .emotion import EmotionTarget
from .spring import Spring
from .state import EyesConfig, EyeState

_EYE_FIELDS = (
    "x",
    "y",
    "width",
    "height",
    "corner_radius",
    "top_lid_angle",
    "top_lid_coverage",
    "bottom_lid_coverage",
    "pupil_offset_x",
    "pupil_offset_y",
    "highlight_offset_x",
    "highlight_offset_y",
    "scale",
    "glow",
)
_COLOR_FIELDS = ("color_r", "color_g", "color_b")


def _springs_for_eye(eye: EyeState, omega: float) -> dict[str, Spring]:
    out: dict[str, Spring] = {}
    for name in _EYE_FIELDS:
        out[name] = Spring(value=getattr(eye, name), target=getattr(eye, name), omega=omega)
    for name in _COLOR_FIELDS:
        out[name] = Spring(
            value=float(getattr(eye, name)),
            target=float(getattr(eye, name)),
            omega=omega * 0.6,
        )
    return out


@dataclass
class EyeAnimator:
    """Interpolates left/right EyeState toward emotion targets via springs."""

    cfg: EyesConfig
    left: EyeState = field(default_factory=EyeState)
    right: EyeState = field(default_factory=EyeState)
    gap: float = 1.0
    _left_s: dict[str, Spring] = field(default_factory=dict)
    _right_s: dict[str, Spring] = field(default_factory=dict)
    _gap_s: Spring | None = None
    # secondary: squash during blink, overshoot on gaze saccades
    squash_y: float = 1.0
    _squash_s: Spring | None = None

    def __post_init__(self) -> None:
        base = self.cfg.base_eye()
        self.left = base.copy()
        self.right = base.copy()
        w = self.cfg.spring_omega
        self._left_s = _springs_for_eye(self.left, w)
        self._right_s = _springs_for_eye(self.right, w)
        self._gap_s = Spring(value=self.cfg.gap, target=self.cfg.gap, omega=w)
        self._squash_s = Spring(value=1.0, target=1.0, omega=self.cfg.blink_spring_omega)
        # gaze channels use snappier omega
        for side in (self._left_s, self._right_s):
            side["pupil_offset_x"].omega = self.cfg.gaze_spring_omega
            side["pupil_offset_y"].omega = self.cfg.gaze_spring_omega

    def set_targets(self, target: EmotionTarget) -> None:
        for name, spring in self._left_s.items():
            spring.set_target(float(getattr(target.left, name)))
        for name, spring in self._right_s.items():
            spring.set_target(float(getattr(target.right, name)))
        assert self._gap_s is not None
        self._gap_s.set_target(target.gap)

    def set_gaze(self, gx: float, gy: float) -> None:
        """Set pupil target offsets (−1..1). Applied on both eyes."""
        for side in (self._left_s, self._right_s):
            side["pupil_offset_x"].set_target(max(-1.0, min(1.0, gx)))
            side["pupil_offset_y"].set_target(max(-1.0, min(1.0, gy)))

    def nudge_gaze_overshoot(self, dx: float, dy: float) -> None:
        """Brief velocity kick so saccades overshoot organically."""
        for side in (self._left_s, self._right_s):
            side["pupil_offset_x"].velocity += dx * 8.0
            side["pupil_offset_y"].velocity += dy * 8.0

    def set_blink_squash(self, blink_amount: float) -> None:
        # squash vertically + slight horizontal stretch
        assert self._squash_s is not None
        self._squash_s.set_target(max(0.08, 1.0 - blink_amount * 0.92))

    def step(self, dt: float) -> tuple[EyeState, EyeState, float]:
        self._step_side(self._left_s, self.left, dt)
        self._step_side(self._right_s, self.right, dt)
        assert self._gap_s is not None and self._squash_s is not None
        self.gap = self._gap_s.step(dt)
        self.squash_y = self._squash_s.step(dt)
        stretch_x = 1.0 + (1.0 - self.squash_y) * 0.15
        left = self.left.copy()
        right = self.right.copy()
        left.height *= self.squash_y
        right.height *= self.squash_y
        left.width *= stretch_x
        right.width *= stretch_x
        return left, right, self.gap

    def _step_side(self, springs: dict[str, Spring], eye: EyeState, dt: float) -> None:
        for name, spring in springs.items():
            val = spring.step(dt)
            if name in _COLOR_FIELDS:
                setattr(eye, name, int(round(max(0.0, min(255.0, val)))))
            else:
                setattr(eye, name, val)
