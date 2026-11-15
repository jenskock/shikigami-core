"""Per-stage latency measurement for each turn (stt, llm, tts, exec...)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timing:
    name: str
    ms: float


class LatencyTracker:
    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logging.getLogger("shikigami.latency")
        self._t0 = time.perf_counter()
        self.timings: list[Timing] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t) * 1000
            self.timings.append(Timing(name, ms))
            self._log.info("stage %-12s %7.0f ms", name, ms)

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def summary(self) -> str:
        parts = " ".join(f"{t.name}={t.ms:.0f}ms" for t in self.timings)
        return f"{parts} total={self.total_ms:.0f}ms".strip()
