"""Face tracking: reuse camera preview detections when available, else detect once."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from ..config import PerceptionConfig


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
        self._last_act = 0.0
        self._face_present = False

    async def poll(self) -> None:
        if not self._cfg.track_faces:
            return
        frame = getattr(self._camera, "last_frame", None)
        if frame is None:
            frame = await self._camera.read_frame()
        boxes = list(getattr(self._camera, "last_detections", None) or [])
        faces = [b for b in boxes if b[0] == "face"]
        if not faces and frame is not None and self._detector is not None:
            faces = [d for d in self._detector.detect_all(frame) if d[0] == "face"]
        if not faces:
            if self._face_present:
                self._face_present = False
                await self._publish("face_lost", {})
            return
        now = time.monotonic()
        _, x, y, w, h = max(faces, key=lambda d: d[3] * d[4])
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        cx = (x + w / 2) / fw * 2 - 1
        cy = -((y + h / 2) / fh * 2 - 1)
        if now - self._last_act < self._cfg.face_cooldown_s:
            await self._motion.look_at(cx, cy)
            return
        self._last_act = now
        await self._motion.look_at(cx, cy)
        if not self._face_present:
            self._face_present = True
            await self._publish(
                "face_seen", {"x": round(cx, 2), "y": round(cy, 2), "faces": len(faces)}
            )
            await self._expression.emote("curious")
            self._log.info("face seen at (%.2f, %.2f) — looking", cx, cy)
