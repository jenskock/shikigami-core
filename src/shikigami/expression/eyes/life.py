"""Autonomous life: Poisson blinks, micro-saccades, idle gaze, breathing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .state import EyesConfig


@dataclass(slots=True)
class BlinkEvent:
    """One blink pulse; amount goes 0→peak→0 (peak < 1 keeps a visible slit)."""

    amount: float = 0.0
    peak: float = 0.78
    phase: str = "idle"  # idle | closing | opening | hold
    queue: int = 0  # remaining blinks in a double-blink chain
    side: str = "both"  # both | left | right


@dataclass
class LifeLayer:
    """Seedable autonomous motion layered on top of emotion targets."""

    cfg: EyesConfig
    rng: random.Random = field(default_factory=random.Random)
    blink: BlinkEvent = field(default_factory=BlinkEvent)
    _next_blink_s: float = 0.0
    _t: float = 0.0
    # idle gaze wander state
    wander_x: float = 0.0
    wander_y: float = 0.0
    _wander_tx: float = 0.0
    _wander_ty: float = 0.0
    _next_wander_s: float = 0.0
    # micro-saccade offsets
    sacc_x: float = 0.0
    sacc_y: float = 0.0
    _next_sacc_s: float = 0.0
    breath: float = 0.0
    # brow asymmetry drift
    brow_asym: float = 0.0
    _hold_s: float = 0.0
    _close_speed: float = 10.0
    _open_speed: float = 12.0

    def __post_init__(self) -> None:
        if self.cfg.seed is not None:
            self.rng.seed(self.cfg.seed)
        self._schedule_blink()
        self._schedule_wander()
        self._schedule_saccade()

    def _poisson_interval(self, rate_per_min: float) -> float:
        """Seconds until next event for Poisson process with given per-minute rate."""
        lam = max(1e-6, rate_per_min) / 60.0
        u = max(1e-12, self.rng.random())
        return -math.log(u) / lam

    def _schedule_blink(self) -> None:
        gap = self._poisson_interval(self.cfg.blink_rate_per_min)
        self._next_blink_s = max(2.8, gap)  # floor so Poisson can't fire back-to-back

    def _schedule_wander(self) -> None:
        self._next_wander_s = 0.8 + self.rng.random() * 2.4
        amp = self.cfg.gaze_wander
        self._wander_tx = (self.rng.random() * 2 - 1) * amp
        self._wander_ty = (self.rng.random() * 2 - 1) * amp * 0.55

    def _schedule_saccade(self) -> None:
        self._next_sacc_s = 0.12 + self.rng.random() * 0.55

    def request_blink(self, *, double: bool = False, side: str = "both") -> None:
        if self.blink.phase != "idle":
            self.blink.queue = max(self.blink.queue, 1 + int(double))
            return
        self.blink.side = side
        self.blink.phase = "closing"
        self.blink.queue = 1 if double else 0
        self.blink.peak = 0.72 + self.rng.random() * 0.14  # soft close, visible slit
        self._close_speed = 9.0 + self.rng.random() * 2.5
        self._open_speed = 11.0 + self.rng.random() * 3.0

    def request_wink(self, side: str) -> None:
        self.request_blink(side=side)
        self.blink.peak = 0.82

    def update(self, dt: float) -> None:
        self._t += dt
        self._update_blink(dt)
        self._update_wander(dt)
        self._update_saccade(dt)
        self.breath = math.sin(self._t * 1.7) * self.cfg.breath_amp
        self.brow_asym = 0.04 * math.sin(self._t * 0.37 + 0.6)

    def _update_blink(self, dt: float) -> None:
        b = self.blink
        if b.phase == "idle":
            self._next_blink_s -= dt
            if self._next_blink_s <= 0.0:
                double = self.rng.random() < self.cfg.double_blink_prob
                self.request_blink(double=double)
                self._schedule_blink()
            return
        peak = b.peak
        if b.phase == "closing":
            b.amount = min(peak, b.amount + self._close_speed * dt)
            if b.amount >= peak:
                b.amount = peak
                b.phase = "hold"
                self._hold_s = 0.015 + self.rng.random() * 0.025
        elif b.phase == "hold":
            self._hold_s -= dt
            if self._hold_s <= 0.0:
                b.phase = "opening"
        elif b.phase == "opening":
            b.amount = max(0.0, b.amount - self._open_speed * dt)
            if b.amount <= 0.0:
                b.amount = 0.0
                if b.queue > 0:
                    b.queue -= 1
                    b.peak = 0.70 + self.rng.random() * 0.12
                    self._close_speed = 9.0 + self.rng.random() * 2.5
                    self._open_speed = 11.0 + self.rng.random() * 3.0
                    b.phase = "closing"
                else:
                    b.phase = "idle"
                    b.side = "both"

    def _update_wander(self, dt: float) -> None:
        self._next_wander_s -= dt
        if self._next_wander_s <= 0.0:
            self._schedule_wander()
        k = 1.0 - math.exp(-2.2 * dt)
        self.wander_x += (self._wander_tx - self.wander_x) * k
        self.wander_y += (self._wander_ty - self.wander_y) * k

    def _update_saccade(self, dt: float) -> None:
        self._next_sacc_s -= dt
        amp = self.cfg.saccade_amp
        if self._next_sacc_s <= 0.0:
            self.sacc_x = (self.rng.random() * 2 - 1) * amp
            self.sacc_y = (self.rng.random() * 2 - 1) * amp * 0.7
            self._schedule_saccade()
        else:
            # decay between saccades
            decay = math.exp(-6.0 * dt)
            self.sacc_x *= decay
            self.sacc_y *= decay
