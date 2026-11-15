"""Critically-damped spring scalar (no linear tweens)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class Spring:
    """Critically-damped spring toward a moving target (ζ = 1).

    Analytic step from Game Programming Gems / Ryan Juckett style:
    x(t) = (x0 + (v0 + ω x0) t) e^{-ω t}   relative to target at 0.
    """

    value: float = 0.0
    velocity: float = 0.0
    target: float = 0.0
    omega: float = 14.0  # natural frequency (rad/s); higher = snappier

    def set_target(self, target: float) -> None:
        self.target = target

    def snap(self, value: float) -> None:
        self.value = value
        self.velocity = 0.0
        self.target = value

    def step(self, dt: float) -> float:
        if dt <= 0.0:
            return self.value
        # Cap dt so a hitch doesn't explode the analytic form
        dt = min(dt, 0.05)
        w = self.omega
        x0 = self.value - self.target
        v0 = self.velocity
        e = math.exp(-w * dt)
        # critically damped
        self.value = self.target + (x0 + (v0 + w * x0) * dt) * e
        self.velocity = (v0 - (v0 + w * x0) * w * dt) * e
        return self.value
