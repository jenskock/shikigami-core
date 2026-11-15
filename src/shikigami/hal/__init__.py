"""HAL backend selection.

Backend is chosen via SHIKIGAMI_BACKEND env var or config (``sim|pidog|picrawler``).
Vendor imports happen lazily inside each backend module, so importing this
package never pulls in SunFounder libraries.
"""

from __future__ import annotations

import os

from .errors import BackendUnavailable
from .interfaces import (
    HAL,
    Audio,
    Camera,
    Expression,
    Motion,
    SensorReading,
    Sensors,
)

__all__ = [
    "HAL",
    "Audio",
    "BackendUnavailable",
    "Camera",
    "Expression",
    "Motion",
    "SensorReading",
    "Sensors",
    "create_hal",
    "resolve_backend",
]

VALID_BACKENDS = ("sim", "pidog", "picrawler")


def resolve_backend(configured: str | None = None) -> str:
    """Env SHIKIGAMI_BACKEND wins over the config value; default 'sim'."""
    backend = (os.environ.get("SHIKIGAMI_BACKEND") or configured or "sim").strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {VALID_BACKENDS}")
    return backend


def create_hal(backend: str, cfg) -> HAL:
    """Instantiate a backend. `cfg` is the root shikigami Config."""
    backend = backend.strip().lower()
    if backend == "sim":
        from .sim.backend import SimHAL

        return SimHAL(cfg)
    if backend == "pidog":
        from .pidog.backend import PiDogHAL

        return PiDogHAL(cfg)
    if backend == "picrawler":
        from .picrawler.backend import PicrawlerHAL

        return PicrawlerHAL(cfg)
    raise ValueError(f"unknown backend {backend!r}; expected one of {VALID_BACKENDS}")
