"""Emotion presets and continuous blends (weights or valence/arousal)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .state import EyeState


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"
    CURIOUS = "curious"
    LOVE = "love"
    CONFUSED = "confused"
    SCARED = "scared"
    FOCUSED = "focused"
    MISCHIEVOUS = "mischievous"


EMOTION_NAMES: tuple[str, ...] = tuple(e.value for e in Emotion)

# Valence (−1..1) / arousal (0..1) anchors for continuous mapping.
_VA: dict[Emotion, tuple[float, float]] = {
    Emotion.NEUTRAL: (0.0, 0.35),
    Emotion.HAPPY: (0.8, 0.55),
    Emotion.SAD: (-0.7, 0.25),
    Emotion.ANGRY: (-0.6, 0.85),
    Emotion.SURPRISED: (0.2, 0.95),
    Emotion.SLEEPY: (0.1, 0.1),
    Emotion.CURIOUS: (0.4, 0.65),
    Emotion.LOVE: (0.9, 0.5),
    Emotion.CONFUSED: (-0.1, 0.55),
    Emotion.SCARED: (-0.8, 0.9),
    Emotion.FOCUSED: (0.1, 0.7),
    Emotion.MISCHIEVOUS: (0.5, 0.75),
}


@dataclass(frozen=True, slots=True)
class EmotionTarget:
    """Target geometry for a named emotion (left/right may differ for brows)."""

    left: EyeState
    right: EyeState
    gap: float = 1.0


def _eye(
    *,
    width: float = 1.0,
    height: float = 1.0,
    corner: float = 0.85,
    top: float = 0.05,
    bottom: float = 0.08,
    angle: float = 0.0,
    scale: float = 1.0,
) -> EyeState:
    return EyeState(
        width=width,
        height=height,
        corner_radius=corner,
        top_lid_coverage=top,
        bottom_lid_coverage=bottom,
        top_lid_angle=angle,
        scale=scale,
    )


def _sym(eye: EyeState, gap: float = 1.0) -> EmotionTarget:
    return EmotionTarget(left=eye.copy(), right=eye.copy(), gap=gap)


def _presets() -> dict[Emotion, EmotionTarget]:
    return {
        Emotion.NEUTRAL: _sym(_eye()),
        Emotion.HAPPY: _sym(_eye(width=1.06, height=0.82, corner=0.9, top=0.08, bottom=0.42), 0.96),
        Emotion.SAD: EmotionTarget(
            left=_eye(width=1.0, height=0.9, top=0.38, bottom=0.06, angle=-12.0),
            right=_eye(width=1.0, height=0.9, top=0.38, bottom=0.06, angle=12.0),
            gap=0.98,
        ),
        Emotion.ANGRY: EmotionTarget(
            left=_eye(width=1.02, height=0.88, top=0.28, bottom=0.12, angle=18.0, corner=0.7),
            right=_eye(width=1.02, height=0.88, top=0.28, bottom=0.12, angle=-18.0, corner=0.7),
            gap=0.94,
        ),
        Emotion.SURPRISED: _sym(
            _eye(width=0.94, height=1.32, corner=0.75, top=0.0, bottom=0.0), 1.14
        ),
        Emotion.SLEEPY: _sym(
            _eye(width=1.12, height=0.38, corner=0.95, top=0.55, bottom=0.5), 0.94
        ),
        Emotion.CURIOUS: _sym(
            _eye(width=0.96, height=1.18, corner=0.8, top=0.0, bottom=0.05), 1.06
        ),
        Emotion.LOVE: _sym(_eye(width=1.08, height=0.95, corner=0.92, top=0.1, bottom=0.28), 0.98),
        Emotion.CONFUSED: EmotionTarget(
            left=_eye(width=0.98, height=1.05, top=0.12, bottom=0.1, angle=-8.0),
            right=_eye(width=1.05, height=1.15, top=0.02, bottom=0.05, angle=14.0),
            gap=1.04,
        ),
        Emotion.SCARED: _sym(
            _eye(width=0.9, height=1.25, corner=0.7, top=0.0, bottom=0.0, scale=0.95),
            1.18,
        ),
        Emotion.FOCUSED: _sym(
            _eye(width=0.92, height=0.85, corner=0.65, top=0.18, bottom=0.15), 0.9
        ),
        Emotion.MISCHIEVOUS: EmotionTarget(
            left=_eye(width=1.04, height=0.78, top=0.15, bottom=0.35, angle=10.0),
            right=_eye(width=1.04, height=0.78, top=0.22, bottom=0.3, angle=-6.0),
            gap=0.97,
        ),
    }


_PRESETS = _presets()


def emotion_targets(name: str | Emotion) -> EmotionTarget:
    key = Emotion(name) if not isinstance(name, Emotion) else name
    src = _PRESETS[key]
    return EmotionTarget(left=src.left.copy(), right=src.right.copy(), gap=src.gap)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_eye(a: EyeState, b: EyeState, t: float) -> EyeState:
    return EyeState(
        x=_lerp(a.x, b.x, t),
        y=_lerp(a.y, b.y, t),
        width=_lerp(a.width, b.width, t),
        height=_lerp(a.height, b.height, t),
        corner_radius=_lerp(a.corner_radius, b.corner_radius, t),
        top_lid_angle=_lerp(a.top_lid_angle, b.top_lid_angle, t),
        top_lid_coverage=_lerp(a.top_lid_coverage, b.top_lid_coverage, t),
        bottom_lid_coverage=_lerp(a.bottom_lid_coverage, b.bottom_lid_coverage, t),
        pupil_offset_x=_lerp(a.pupil_offset_x, b.pupil_offset_x, t),
        pupil_offset_y=_lerp(a.pupil_offset_y, b.pupil_offset_y, t),
        highlight_offset_x=_lerp(a.highlight_offset_x, b.highlight_offset_x, t),
        highlight_offset_y=_lerp(a.highlight_offset_y, b.highlight_offset_y, t),
        scale=_lerp(a.scale, b.scale, t),
        color_r=int(round(_lerp(a.color_r, b.color_r, t))),
        color_g=int(round(_lerp(a.color_g, b.color_g, t))),
        color_b=int(round(_lerp(a.color_b, b.color_b, t))),
        glow=_lerp(a.glow, b.glow, t),
    )


def blend_emotions(weights: dict[str, float]) -> EmotionTarget:
    """Weighted blend of emotion presets. Weights need not sum to 1 (renormalized)."""
    cleaned: list[tuple[EmotionTarget, float]] = []
    for name, w in weights.items():
        if w <= 0:
            continue
        cleaned.append((emotion_targets(name), float(w)))
    if not cleaned:
        return emotion_targets(Emotion.NEUTRAL)
    total = sum(w for _, w in cleaned)
    z = EyeState(
        width=0,
        height=0,
        corner_radius=0,
        top_lid_coverage=0,
        bottom_lid_coverage=0,
        scale=0,
        color_r=0,
        color_g=0,
        color_b=0,
        glow=0,
        highlight_offset_x=0,
        highlight_offset_y=0,
    )
    acc_l, acc_r = z.copy(), z.copy()
    acc_gap = 0.0
    for tgt, w in cleaned:
        t = w / total
        acc_l = _add_eye(acc_l, tgt.left, t)
        acc_r = _add_eye(acc_r, tgt.right, t)
        acc_gap += tgt.gap * t
    return EmotionTarget(left=acc_l, right=acc_r, gap=acc_gap)


def _add_eye(acc: EyeState, src: EyeState, w: float) -> EyeState:
    return EyeState(
        x=acc.x + src.x * w,
        y=acc.y + src.y * w,
        width=acc.width + src.width * w,
        height=acc.height + src.height * w,
        corner_radius=acc.corner_radius + src.corner_radius * w,
        top_lid_angle=acc.top_lid_angle + src.top_lid_angle * w,
        top_lid_coverage=acc.top_lid_coverage + src.top_lid_coverage * w,
        bottom_lid_coverage=acc.bottom_lid_coverage + src.bottom_lid_coverage * w,
        pupil_offset_x=acc.pupil_offset_x + src.pupil_offset_x * w,
        pupil_offset_y=acc.pupil_offset_y + src.pupil_offset_y * w,
        highlight_offset_x=acc.highlight_offset_x + src.highlight_offset_x * w,
        highlight_offset_y=acc.highlight_offset_y + src.highlight_offset_y * w,
        scale=acc.scale + src.scale * w,
        color_r=int(round(acc.color_r + src.color_r * w)),
        color_g=int(round(acc.color_g + src.color_g * w)),
        color_b=int(round(acc.color_b + src.color_b * w)),
        glow=acc.glow + src.glow * w,
    )


def from_valence_arousal(valence: float, arousal: float) -> EmotionTarget:
    """Map continuous valence (−1..1) / arousal (0..1) to a blend of nearest presets."""
    v = max(-1.0, min(1.0, valence))
    a = max(0.0, min(1.0, arousal))
    # Inverse-distance weights in VA space (arousal scaled to −1..1 for isotropic feel)
    weights: dict[str, float] = {}
    for emo, (ev, ea) in _VA.items():
        dv = v - ev
        da = (a - ea) * 2.0
        dist2 = dv * dv + da * da
        weights[emo.value] = 1.0 / (dist2 + 0.05)
    # Keep top-3 for a sharp blend
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return blend_emotions(dict(top))


def scale_intensity(base: EmotionTarget, intensity: float) -> EmotionTarget:
    """Lerp toward emotion from neutral by intensity (0..1.5; >1 overshoots)."""
    t = max(0.0, min(1.5, intensity))
    neu = emotion_targets(Emotion.NEUTRAL)
    return EmotionTarget(
        left=_lerp_eye(neu.left, base.left, t),
        right=_lerp_eye(neu.right, base.right, t),
        gap=_lerp(neu.gap, base.gap, t),
    )
