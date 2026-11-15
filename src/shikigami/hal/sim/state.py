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
        self.mic: str = ""
        self.mic_level: float = 0.0
        self.gaze_x: float = 0.0  # -1..1, drives eye look (from look_at / face tracker)
        self.gaze_y: float = 0.0
        # Body / servo telemetry (picrawler soft or real, when published by Motion).
        self.servo_angles: list[float] = []
        self.servo_labels: list[str] = []
        self.leg_coords: list[list[float]] = []
