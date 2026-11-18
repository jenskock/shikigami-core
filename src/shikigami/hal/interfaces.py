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


@dataclass(frozen=True)
class Frame:
    """One camera frame in both streams. Same shapes on Pi and Mac (see CameraConfig).

    ``main`` and ``lores`` are BGR uint8 ndarrays: ``main`` is the full frame for
    snapshots, ``lores`` the small frame detectors run on. ``seq`` increases by one
    per delivered frame; ``t`` is ``time.monotonic()`` at capture.
    """

    main: Any
    lores: Any
    seq: int
    t: float


class Camera(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def read_frame(self) -> Any | None: ...  # latest main frame (BGR ndarray)

    def latest(self) -> Frame | None:
        """Newest frame without blocking, or None. Cameras without streams return None."""
        return None

    async def snapshot_jpeg(self, max_side: int = 768, quality: int = 80) -> bytes | None:
        """JPEG of the newest main frame, longest side <= max_side. None without a frame."""
        return None

    async def preview_frame(self, detections: list | None = None) -> None:
        """Show the latest frame with detection boxes, if there is a preview. No-op default."""
        return None

    @abc.abstractmethod
    async def stop(self) -> None: ...


class Audio(abc.ABC):
    @abc.abstractmethod
    async def record(self, max_s: float = 8.0, silence_s: float = 1.2) -> bytes:
        """Legacy capture hook. The Pipecat transport owns the mic and this returns empty WAV."""
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
