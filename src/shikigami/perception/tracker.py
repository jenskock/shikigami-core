"""Face tracking: watch the camera feed, look at faces, publish events (no network)."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from ..config import PerceptionConfig

POLL_INTERVAL_S = 0.2  # detection cadence (5 Hz); the reflex loop calls poll() at ~20 Hz


class Publisher(Protocol):
    async def publish(self, topic: str, payload: dict | None = None) -> Any: ...


class FaceTracker:
    def __init__(
        self,
        camera: Any,
        motion: Any,
        expression: Any,
        publish: Publisher,
        cfg: PerceptionConfig,
        detector: Any | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._camera = camera
        self._motion = motion
        self._expression = expression
        self._publish = publish
        self._cfg = cfg
        self._detector = detector
        self._log = log or logging.getLogger("shikigami.tracker")
        self._last_poll = 0.0
        self._last_act = 0.0
        self._face_present = False

    async def poll(self) -> None:
        if not self._cfg.track_faces:
            return
        now = time.monotonic()
        if now - self._last_poll < POLL_INTERVAL_S:
            return
        self._last_poll = now
        frame = await self._camera.read_frame()
        if frame is None:
            return
        if self._detector is None:
            try:
                from .face import FaceDetector

                self._detector = FaceDetector()
            except Exception as exc:
                self._log.warning("face detection unavailable (%s) — tracking disabled", exc)
                self._cfg.track_faces = False
                return
        faces = [d for d in self._detector.detect_all(frame) if d[0] == "face"]
        if not faces:
            if self._face_present:
                self._face_present = False
                await self._publish("face_lost", {})
            return
        if now - self._last_act < self._cfg.face_cooldown_s:
            return
        self._last_act = now
        _, x, y, w, h = max(faces, key=lambda d: d[3] * d[4])
        fh, fw = frame.shape[:2]
        cx = (x + w / 2) / fw * 2 - 1
        cy = -((y + h / 2) / fh * 2 - 1)  # image y grows downward; look_at y grows upward
        await self._motion.look_at(cx, cy)
        if not self._face_present:
            self._face_present = True
            await self._publish(
                "face_seen", {"x": round(cx, 2), "y": round(cy, 2), "faces": len(faces)}
            )
            await self._expression.emote("curious")
            self._log.info("face seen at (%.2f, %.2f) — looking", cx, cy)
