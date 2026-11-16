"""Eye geometry / color state and YAML-backed config."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(slots=True)
class EyeState:
    """Per-eye visual parameters (display-agnostic, normalized where noted)."""

    x: float = 0.0  # center offset from default slot, normalized eye-widths
    y: float = 0.0
    width: float = 1.0  # relative to base eye width
    height: float = 1.0
    corner_radius: float = 0.85  # 0..1 of min(half-w, half-h)
    top_lid_angle: float = 0.0  # degrees; + = outer corner raised (angry brow)
    top_lid_coverage: float = 0.05  # 0 open .. 1 fully covering from top
    bottom_lid_coverage: float = 0.08
    bottom_lid_curve: float = 0.0  # 0 flat .. 1 arched up in the middle (smiling cheek)
    pupil_scale: float = 1.0  # 0 hides the pupil (smiling eyes), <1 pinpoint, >1 big
    pupil_offset_x: float = 0.0  # -1..1 inside open aperture
    pupil_offset_y: float = 0.0
    highlight_offset_x: float = -0.35
    highlight_offset_y: float = -0.35
    scale: float = 1.0
    color_r: int = 255
    color_g: int = 255
    color_b: int = 255
    glow: float = 0.35  # soft outer glow strength 0..1

    @property
    def color_rgb(self) -> tuple[int, int, int]:
        return (self.color_r, self.color_g, self.color_b)

    def copy(self) -> EyeState:
        return replace(self)


@dataclass(slots=True)
class FrameState:
    """Fully resolved pose for one render frame."""

    left: EyeState = field(default_factory=EyeState)
    right: EyeState = field(default_factory=EyeState)
    gap: float = 1.0  # inter-eye spacing scale
    time_s: float = 0.0
    # overlay / special FX flags consumed by renderers
    heart_eyes: float = 0.0  # 0..1
    dizzy: float = 0.0
    sleep_zzz: float = 0.0


@dataclass
class EyesConfig:
    """Tunable defaults loaded from YAML `eyes:` section."""

    color: tuple[int, int, int] = (255, 255, 255)
    glow: float = 0.35
    base_width: float = 1.0
    base_height: float = 1.0
    corner_radius: float = 0.85
    gap: float = 1.0
    blink_rate_per_min: float = 8.0  # mean blinks / minute (Poisson λ)
    double_blink_prob: float = 0.08
    gaze_wander: float = 0.35  # idle gaze wander amplitude
    saccade_amp: float = 0.04
    breath_amp: float = 0.025
    spring_omega: float = 14.0  # rad/s toward emotion targets
    gaze_spring_omega: float = 22.0
    blink_spring_omega: float = 40.0
    personality: str = "curious"  # curious | calm | mischievous
    seed: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> EyesConfig:
        if not data:
            return cls()
        color = data.get("color", (255, 255, 255))
        if isinstance(color, list | tuple) and len(color) == 3:
            color = (int(color[0]), int(color[1]), int(color[2]))
        else:
            color = (255, 255, 255)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {"color": color}
        for k, v in data.items():
            if k == "color":
                continue
            if k in known:
                kwargs[k] = v
        return cls(**kwargs)

    def apply_personality(self) -> EyesConfig:
        """Return a copy with personality-tuned blink/wander defaults."""
        cfg = EyesConfig(**{f: getattr(self, f) for f in self.__dataclass_fields__})
        p = self.personality.lower()
        if p == "curious":
            cfg.gaze_wander = max(cfg.gaze_wander, 0.45)
            cfg.saccade_amp = max(cfg.saccade_amp, 0.05)
        elif p == "calm":
            cfg.blink_rate_per_min = min(cfg.blink_rate_per_min, 6.0)
            cfg.gaze_wander = min(cfg.gaze_wander, 0.2)
            cfg.breath_amp = max(cfg.breath_amp, 0.03)
        elif p == "mischievous":
            cfg.blink_rate_per_min = min(cfg.blink_rate_per_min + 2.0, 12.0)
            cfg.double_blink_prob = max(cfg.double_blink_prob, 0.15)
            cfg.gaze_wander = max(cfg.gaze_wander, 0.5)
        return cfg

    def base_eye(self) -> EyeState:
        r, g, b = self.color
        return EyeState(
            width=self.base_width,
            height=self.base_height,
            corner_radius=self.corner_radius,
            color_r=r,
            color_g=g,
            color_b=b,
            glow=self.glow,
        )
