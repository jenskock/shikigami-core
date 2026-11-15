"""Autonomous life: Poisson blinks, micro-saccades, idle gaze, breathing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .state import EyesConfig


@dataclass(slots=True)
class BlinkEvent:
    """One blink pulse; amount goes 0→1→0 over the blink."""

    amount: float = 0.0
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
        self._next_blink_s = self._poisson_interval(self.cfg.blink_rate_per_min)

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

    def request_wink(self, side: str) -> None:
        self.request_blink(side=side)

    def update(self, dt: float, *, tracking: bool) -> None:
        self._t += dt
        self._update_blink(dt)
        self._update_wander(dt, tracking=tracking)
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
        # closing / opening with squash-friendly ease
        close_speed = 14.0 + self.rng.random() * 4.0
        open_speed = 18.0 + self.rng.random() * 6.0
        if b.phase == "closing":
            b.amount = min(1.0, b.amount + close_speed * dt)
            if b.amount >= 1.0:
                b.amount = 1.0
                b.phase = "hold"
                self._hold_s = 0.02 + self.rng.random() * 0.04
        elif b.phase == "hold":
            self._hold_s -= dt
            if self._hold_s <= 0.0:
                b.phase = "opening"
        elif b.phase == "opening":
            b.amount = max(0.0, b.amount - open_speed * dt)
            if b.amount <= 0.0:
                b.amount = 0.0
                if b.queue > 0:
                    b.queue -= 1
                    b.phase = "closing"
                else:
                    b.phase = "idle"
                    b.side = "both"

    def _update_wander(self, dt: float, *, tracking: bool) -> None:
        if tracking:
            # gently settle wander when actively tracking a face
            self.wander_x += (0.0 - self.wander_x) * min(1.0, 4.0 * dt)
            self.wander_y += (0.0 - self.wander_y) * min(1.0, 4.0 * dt)
            return
        self._next_wander_s -= dt
        if self._next_wander_s <= 0.0:
            self._schedule_wander()
        # soft approach to wander target (not a linear tween — exp decay)
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
