"""PiCrawler backend.

Exports load on first use. The sim dashboard imports ``picrawler.kinematics``
while the sim package is still loading, and ``picrawler.backend`` imports the
sim camera, so loading the backend eagerly here would be a circular import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PicrawlerHAL": ".backend",
    "PicrawlerMotion": ".motion",
    "SERVO_LABELS": ".soft",
    "SoftPicrawler": ".soft",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)
