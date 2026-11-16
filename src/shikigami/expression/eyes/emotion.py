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
    """Target geometry for a named emotion (left/right may differ for brows).

    ``tint`` pulls the eye color towards an RGB value (love goes pink, angry red).
    """

    left: EyeState
    right: EyeState
    gap: float = 1.0
    tint: tuple[int, int, int] | None = None


def _eye(
    *,
    width: float = 1.0,
    height: float = 1.0,
    corner: float = 0.6,
    top: float = 0.0,
    bottom: float = 0.0,
    curve: float = 0.0,
    angle: float = 0.0,
    scale: float = 1.0,
    pupil: float = 1.0,
    y: float = 0.0,
) -> EyeState:
    return EyeState(
        y=y,
        width=width,
        height=height,
        corner_radius=corner,
        top_lid_coverage=top,
        bottom_lid_coverage=bottom,
        bottom_lid_curve=curve,
        top_lid_angle=angle,
        scale=scale,
        pupil_scale=pupil,
    )


def _sym(
    eye: EyeState, gap: float = 1.0, tint: tuple[int, int, int] | None = None
) -> EmotionTarget:
    return EmotionTarget(left=eye.copy(), right=eye.copy(), gap=gap, tint=tint)


def _brows(eye: EyeState, angle: float, gap: float = 1.0, tint=None) -> EmotionTarget:
    """Mirror a brow angle: the left eye gets ``angle``, the right eye its opposite."""
    left, right = eye.copy(), eye.copy()
    left.top_lid_angle = angle
    right.top_lid_angle = -angle
    return EmotionTarget(left=left, right=right, gap=gap, tint=tint)


def _presets() -> dict[Emotion, EmotionTarget]:
    # Each face has to read at a glance on a 240 px round display, so every
    # preset differs from the others in silhouette, not just by a few percent.
    return {
        Emotion.NEUTRAL: _sym(_eye(top=0.04, bottom=0.03)),
        # Smiling arcs, no pupil: the cheek pushes the lower lid up in the middle.
        Emotion.HAPPY: _sym(
            _eye(width=1.05, height=0.95, corner=0.8, bottom=0.45, curve=1.0, pupil=0.0), 0.98
        ),
        # Outer corners drop, eyes sink.
        Emotion.SAD: _brows(_eye(height=0.85, top=0.4, y=0.14, pupil=1.05), -17.0),
        # Brows slam down towards the nose.
        Emotion.ANGRY: _brows(
            _eye(height=0.85, corner=0.4, top=0.42, bottom=0.1, pupil=0.85),
            21.0,
            0.94,
            (255, 90, 70),
        ),
        Emotion.SURPRISED: _sym(_eye(height=1.25, corner=0.8, scale=1.08, pupil=0.6), 1.1),
        # Heavy upper lids, sagging outer corners.
        Emotion.SLEEPY: _brows(_eye(width=1.05, height=0.8, top=0.62, y=0.1), -5.0, 0.97),
        # One eye bigger than the other, like a tilted head.
        Emotion.CURIOUS: EmotionTarget(
            left=_eye(height=0.95, top=0.06, pupil=1.1),
            right=_eye(height=1.2, scale=1.08, pupil=1.15),
            gap=1.04,
        ),
        Emotion.LOVE: _sym(
            _eye(width=1.05, corner=0.85, bottom=0.3, curve=0.8, pupil=1.35),
            0.98,
            (255, 150, 200),
        ),
        # One brow down, the other eye wide.
        Emotion.CONFUSED: EmotionTarget(
            left=_eye(height=0.9, top=0.32, angle=-11.0, pupil=0.9),
            right=_eye(height=1.15, scale=1.05, pupil=0.9),
            gap=1.04,
        ),
        # Tall, narrow, pinpoint pupils, eyes apart.
        Emotion.SCARED: _sym(
            _eye(width=0.85, height=1.2, corner=0.75, scale=0.94, pupil=0.45), 1.16
        ),
        # A narrow, level slit.
        Emotion.FOCUSED: _sym(
            _eye(height=0.75, corner=0.35, top=0.26, bottom=0.2, pupil=0.9), 0.93
        ),
        # A smirk: one brow down, both cheeks up.
        Emotion.MISCHIEVOUS: EmotionTarget(
            left=_eye(height=0.9, top=0.36, bottom=0.24, curve=0.6, angle=15.0),
            right=_eye(height=0.95, top=0.1, bottom=0.3, curve=0.6, angle=-4.0),
            gap=0.97,
        ),
    }


_PRESETS = _presets()


def emotion_targets(name: str | Emotion) -> EmotionTarget:
    key = Emotion(name) if not isinstance(name, Emotion) else name
    src = _PRESETS[key]
    return EmotionTarget(left=src.left.copy(), right=src.right.copy(), gap=src.gap, tint=src.tint)


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
        bottom_lid_curve=_lerp(a.bottom_lid_curve, b.bottom_lid_curve, t),
        pupil_scale=_lerp(a.pupil_scale, b.pupil_scale, t),
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
        pupil_scale=0,
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
        bottom_lid_curve=acc.bottom_lid_curve + src.bottom_lid_curve * w,
        pupil_scale=acc.pupil_scale + src.pupil_scale * w,
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
        tint=base.tint,
    )
