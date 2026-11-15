"""Abstract hardware interfaces (HAL).

The core only ever talks to these interfaces. Vendor libraries (SunFounder
`pidog` / `picrawler` / `robot_hat` / `vilib`) may only be imported inside the
matching backend module — never here, never in the core.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SensorReading:
    distance_cm: float | None = None
    touch: bool = False
    imu: dict[str, float] = field(default_factory=dict)
    sound_direction_deg: float | None = None


class Motion(abc.ABC):
    @property
    def moving(self) -> bool:
        return False

    @abc.abstractmethod
    async def walk(
        self, direction: str = "forward", speed: float = 0.5, duration_s: float | None = None
    ) -> None: ...

    @abc.abstractmethod
    async def turn(
        self, direction: str = "left", angle_deg: float | None = None, speed: float = 0.5
    ) -> None: ...

    @abc.abstractmethod
    async def sit(self) -> None: ...

    @abc.abstractmethod
    async def stand(self) -> None: ...

    @abc.abstractmethod
    async def lie(self) -> None: ...

    @abc.abstractmethod
    async def wag(self, duration_s: float | None = None) -> None: ...

    @abc.abstractmethod
    async def look_at(self, x: float, y: float) -> None: ...  # normalized -1.0 .. 1.0

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def play_action(self, name: str) -> None: ...


class Camera(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def read_frame(self) -> Any | None: ...  # backend-specific frame (e.g. ndarray)

    async def preview_frame(self) -> None:
        """Show the latest frame in a preview window, annotated if supported. No-op default."""
        return None

    @abc.abstractmethod
    async def stop(self) -> None: ...


class Audio(abc.ABC):
    @abc.abstractmethod
    async def record(self, max_s: float = 8.0, silence_s: float = 1.2) -> bytes:
        """Record mic audio with VAD; returns WAV bytes."""
        ...

    @abc.abstractmethod
    async def play(self, data: bytes, fmt: str = "wav") -> None: ...

    def set_muted(self, muted: bool) -> None:
        """Optional: abort in-flight record when muted. Default no-op."""
        _ = muted


class Sensors(abc.ABC):
    @abc.abstractmethod
    async def read(self) -> SensorReading: ...


class Expression(abc.ABC):
    """Face / LEDs / expressive sounds."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def update(self, dt_s: float) -> None: ...  # idle animation tick, called ~20-50 Hz

    @abc.abstractmethod
    async def emote(self, name: str) -> None: ...

    @abc.abstractmethod
    async def set_mood(self, mood: str) -> None: ...

    @abc.abstractmethod
    async def speak_visual(self, active: bool) -> None: ...  # mouth/LEDs while talking

    @abc.abstractmethod
    async def stop(self) -> None: ...


@dataclass
class HAL:
    """A complete hardware abstraction: one instance per backend."""

    backend: str
    motion: Motion
    camera: Camera
    audio: Audio
    sensors: Sensors
    expression: Expression

    async def start(self) -> None:
        await self.expression.start()
        try:
            await self.camera.start()
        except Exception as exc:  # camera is optional at runtime
            logging.getLogger("shikigami.hal").warning("camera unavailable: %s", exc)

    async def stop(self) -> None:
        await self.camera.stop()
        await self.expression.stop()
