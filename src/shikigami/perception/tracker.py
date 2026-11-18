"""Face tracking: newest perception result -> look_at + face_seen / face_lost events."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from ..config import PerceptionConfig
from ..hal.interfaces import Expression, Motion
from .pipeline import STALE_S, PerceptionResult

__all__ = ["STALE_S", "FaceTracker"]  # STALE_S: no fresh result this long = "no face"


class Publisher(Protocol):
    async def publish(self, topic: str, payload: dict | None = None) -> Any: ...


class ResultSource(Protocol):
    def latest(self) -> PerceptionResult | None: ...


class FaceTracker:
    def __init__(
        self,
        perception: ResultSource,
        motion: Motion,
        expression: Expression,
        publish: Publisher,
        cfg: PerceptionConfig,
        log: logging.Logger | None = None,
    ) -> None:
        self.perception = perception
        self._motion = motion
        self._expression = expression
        self._publish = publish
        self._cfg = cfg
        self._log = log or logging.getLogger("shikigami.tracker")
        self._last_act = 0.0
        self._last_seq = -1
        self._face_present = False

    async def poll(self) -> None:
        drain = getattr(self.perception, "drain_events", None)
        if drain is not None:  # person_arrived / person_left from the perception thread
            for topic, payload in drain():
                self._log.info("%s %s", topic, payload)
                await self._publish(topic, payload)
        if not self._cfg.track_faces:
            return
        res = self.perception.latest()
        now = time.monotonic()
        stale = res is None or now - res.t > STALE_S
        if not stale and res.seq == self._last_seq:
            return  # nothing new since the last tick
        faces = [] if stale else res.of("face")
        if res is not None:
            self._last_seq = res.seq
        if not faces:
            if self._face_present:
                self._face_present = False
                await self._publish("face_lost", {})
            return
        face = max(faces, key=lambda d: d.area)
        fx, fy = face.center
        cx = fx * 2 - 1
        cy = -(fy * 2 - 1)
        await self._motion.look_at(cx, cy)
        if now - self._last_act < self._cfg.face_cooldown_s:
            return
        self._last_act = now
        if not self._face_present:
            self._face_present = True
            await self._publish(
                "face_seen", {"x": round(cx, 2), "y": round(cy, 2), "faces": len(faces)}
            )
            await self._expression.emote("curious")
            self._log.info("face seen at (%.2f, %.2f) — looking", cx, cy)
