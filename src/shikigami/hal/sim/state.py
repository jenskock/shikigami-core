"""Shared mutable sim state — shown in the window and readable by the app."""

from __future__ import annotations


class SimState:
    def __init__(self, backend: str = "sim") -> None:
        self.backend = backend
        self.last_motion: str = ""
        self.mood: str = "neutral"
        self.last_heard: str = ""
        self.last_latency: str = ""
        self.status: str = "idle"
