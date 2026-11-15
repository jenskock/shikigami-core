"""High-level eyes API: emotion, gaze, blinks, reactions, autonomous life."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .animator import EyeAnimator
from .emotion import (
    Emotion,
    EmotionTarget,
    blend_emotions,
    emotion_targets,
    from_valence_arousal,
    scale_intensity,
)
from .life import LifeLayer
from .state import EyesConfig, FrameState

_REACT: dict[str, dict[str, float]] = {
    "startle": {"duration": 0.45, "surprised": 1.0, "scale_kick": 0.12},
    "laugh": {"duration": 0.7, "happy": 1.0, "bounce": 1.0},
    "heart": {"duration": 1.2, "love": 1.0, "heart_eyes": 1.0},
    "dizzy": {"duration": 1.4, "confused": 0.8, "dizzy": 1.0},
    "sleep": {"duration": 2.5, "sleepy": 1.0, "sleep_zzz": 1.0},
    "heart eyes": {"duration": 1.2, "love": 1.0, "heart_eyes": 1.0},
    "dizzy spiral": {"duration": 1.4, "confused": 0.8, "dizzy": 1.0},
    "zzz": {"duration": 2.5, "sleepy": 1.0, "sleep_zzz": 1.0},
}


@dataclass
class _EmotionHold:
    name: str
    intensity: float
    until: float | None  # monotonic deadline; None = hold forever


@dataclass
class _Reaction:
    kind: str
    until: float
    params: dict[str, float]


@dataclass
class EyesEngine:
    """Display-agnostic procedural eyes engine."""

    cfg: EyesConfig = field(default_factory=EyesConfig)
    animator: EyeAnimator = field(init=False)
    life: LifeLayer = field(init=False)
    _emotion: _EmotionHold = field(init=False)
    _blend: dict[str, float] | None = None
    _valence: float | None = None
    _arousal: float | None = None
    _gaze_x: float = 0.0
    _gaze_y: float = 0.0
    _speaking: bool = False
    _t: float = 0.0
    _reaction: _Reaction | None = None
    _last_gaze: tuple[float, float] = (0.0, 0.0)
    frame: FrameState = field(default_factory=FrameState)

    def __post_init__(self) -> None:
        self.cfg = self.cfg.apply_personality()
        self.animator = EyeAnimator(self.cfg)
        self.life = LifeLayer(self.cfg)
        self._emotion = _EmotionHold(name=Emotion.NEUTRAL.value, intensity=1.0, until=None)
        self.animator.set_targets(emotion_targets(Emotion.NEUTRAL))

    def look_at(self, x: float, y: float) -> None:
        """Gaze in normalized space (−1..1)."""
        self._gaze_x = max(-1.0, min(1.0, x))
        self._gaze_y = max(-1.0, min(1.0, y))
        self._kick_saccade_if_needed()

    def _kick_saccade_if_needed(self) -> None:
        dx = self._gaze_x - self._last_gaze[0]
        dy = self._gaze_y - self._last_gaze[1]
        if dx * dx + dy * dy > 0.04:
            self.animator.nudge_gaze_overshoot(dx, dy)
        self._last_gaze = (self._gaze_x, self._gaze_y)

    def set_emotion(
        self,
        name: str,
        intensity: float = 1.0,
        duration: float | None = None,
    ) -> None:
        key = name.lower().strip()
        if key not in {e.value for e in Emotion}:
            key = Emotion.NEUTRAL.value
        until = None if duration is None else time.monotonic() + max(0.0, duration)
        self._emotion = _EmotionHold(name=key, intensity=intensity, until=until)
        self._blend = None
        self._valence = None
        self._arousal = None

    def set_emotion_blend(self, weights: dict[str, float]) -> None:
        self._blend = dict(weights)
        self._valence = None
        self._arousal = None

    def set_valence_arousal(self, valence: float, arousal: float) -> None:
        self._valence = valence
        self._arousal = arousal
        self._blend = None

    def blink(self) -> None:
        self.life.request_blink()

    def wink(self, side: str) -> None:
        s = side.lower()
        if s not in ("left", "right"):
            s = "left"
        self.life.request_wink(s)

    def react(self, event: str) -> None:
        key = event.lower().strip()
        params = (
            _REACT.get(key)
            or _REACT.get(key.replace("_", " "))
            or _REACT.get(key.replace("-", " "))
        )
        if params is None:
            params = _REACT["startle"]
            key = "startle"
        dur = float(params["duration"])
        self._reaction = _Reaction(kind=key, until=time.monotonic() + dur, params=params)
        for emo in Emotion:
            if emo.value in params:
                self.set_emotion(emo.value, intensity=float(params[emo.value]), duration=dur)

    def set_speaking(self, active: bool) -> None:
        self._speaking = active

    def update(self, dt: float) -> FrameState:
        self._t += dt
        now = time.monotonic()
        if self._emotion.until is not None and now >= self._emotion.until:
            self._emotion = _EmotionHold(Emotion.NEUTRAL.value, 1.0, None)

        reaction_fx = self._tick_reaction(now)
        target = self._resolve_target()
        if self._speaking:
            target = _speaking_mod(target, self._t)

        self.life.update(dt)
        self.animator.set_targets(target)

        gx = self._gaze_x * 0.55 + self.life.wander_x + self.life.sacc_x
        gy = -self._gaze_y * 0.4 + self.life.wander_y + self.life.sacc_y
        if reaction_fx.get("dizzy", 0.0) > 0.0:
            d = reaction_fx["dizzy"]
            gx += math.sin(self._t * 9.0) * 0.35 * d
            gy += math.cos(self._t * 7.0) * 0.25 * d
        self.animator.set_gaze(gx, gy)

        blink = self.life.blink
        self.animator.set_blink_squash(blink.amount)
        left, right, gap = self.animator.step(dt)

        left, right = _apply_blink(left, right, blink.amount, blink.side)
        breath = (
            1.0
            + self.life.breath
            + reaction_fx.get("bounce", 0.0) * 0.08 * math.sin(self._t * 14.0)
        )
        left.scale *= breath
        right.scale *= breath
        left.top_lid_angle += self.life.brow_asym * 8.0
        right.top_lid_angle -= self.life.brow_asym * 8.0
        if kick := reaction_fx.get("scale_kick", 0.0):
            left.scale *= 1.0 + kick
            right.scale *= 1.0 + kick

        self.frame = FrameState(
            left=left,
            right=right,
            gap=gap,
            time_s=self._t,
            heart_eyes=reaction_fx.get("heart_eyes", 0.0),
            dizzy=reaction_fx.get("dizzy", 0.0),
            sleep_zzz=reaction_fx.get("sleep_zzz", 0.0),
        )
        return self.frame

    def _resolve_target(self) -> EmotionTarget:
        if self._blend:
            base = blend_emotions(self._blend)
        elif self._valence is not None and self._arousal is not None:
            base = from_valence_arousal(self._valence, self._arousal)
        else:
            base = scale_intensity(
                emotion_targets(self._emotion.name),
                self._emotion.intensity,
            )
        r, g, b = self.cfg.color
        base.left.color_r = base.right.color_r = r
        base.left.color_g = base.right.color_g = g
        base.left.color_b = base.right.color_b = b
        base.left.glow = base.right.glow = self.cfg.glow
        return base

    def _tick_reaction(self, now: float) -> dict[str, float]:
        if self._reaction is None:
            return {}
        if now >= self._reaction.until:
            self._reaction = None
            return {}
        dur = float(self._reaction.params["duration"])
        remaining = self._reaction.until - now
        env = max(0.0, min(1.0, remaining / max(1e-3, dur)))
        env = env * env
        out: dict[str, float] = {}
        for k, v in self._reaction.params.items():
            if k == "duration":
                continue
            if isinstance(v, (int, float)) and k not in {e.value for e in Emotion}:
                out[k] = float(v) * env
        return out


def _speaking_mod(target: EmotionTarget, t: float) -> EmotionTarget:
    breath = 0.03 * math.sin(t * 9.0)
    left = target.left.copy()
    right = target.right.copy()
    for eye in (left, right):
        eye.bottom_lid_coverage = min(0.65, eye.bottom_lid_coverage + 0.06 + breath)
        eye.height *= 1.0 + breath
    return EmotionTarget(left=left, right=right, gap=target.gap)


def _apply_blink(left, right, amount: float, side: str):
    left, right = left.copy(), right.copy()

    def lid(eye):
        eye.top_lid_coverage = min(0.72, eye.top_lid_coverage + amount * 0.45)
        eye.bottom_lid_coverage = min(0.72, eye.bottom_lid_coverage + amount * 0.45)
        return eye

    if side in ("both", "left"):
        left = lid(left)
    if side in ("both", "right"):
        right = lid(right)
    return left, right
