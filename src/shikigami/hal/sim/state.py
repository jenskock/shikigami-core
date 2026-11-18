"""Shared mutable sim state — shown in the window and readable by the app."""

from __future__ import annotations

_TOOL_LOG_MAX = 200


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
        self.mic_muted: bool = False
        self.gaze_x: float = 0.0  # -1..1, drives eye look (from look_at / face tracker)
        self.gaze_y: float = 0.0
        # Body / servo telemetry (picrawler soft or real, when published by Motion).
        self.servo_angles: list[float] = []
        self.servo_labels: list[str] = []
        self.leg_coords: list[list[float]] = []
        # World pose on the floor (mm / deg) — BODY view translates the stick figure.
        self.body_x: float = 0.0
        self.body_y: float = 0.0
        self.body_yaw_deg: float = 0.0
        # Everything the BODY view draws, replaced as one object per pose so the
        # view never reads half of one frame and half of the next.
        self.body_pose: object | None = None
        # SoftPicrawler animation pace multiplier (1.0 = the robot's own speed).
        self.playback_speed: float = 1.0
        self.tool_log: list[str] = []  # dashboard TOOLS panel (newest last)

    def append_tool(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        self.tool_log.append(text)
        if len(self.tool_log) > _TOOL_LOG_MAX:
            del self.tool_log[: len(self.tool_log) - _TOOL_LOG_MAX]
