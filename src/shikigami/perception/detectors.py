"""Detectors on the low-res stream. Output is resolution-independent (normalized boxes).

- ``YuNetDetector``: OpenCV ``FaceDetectorYN`` (YuNet ONNX) — robust faces + 5 landmarks
  (the landmarks feed face recognition later). Needs ``make setup-vision``.
- ``HaarDetector``: the old OpenCV Haar cascades — fallback for faces, and cat faces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

YUNET_FILE = "face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class Detection:
    """A box in normalized frame coords (0..1, origin top-left)."""

    label: str
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    landmarks: tuple[tuple[float, float], ...] | None = None  # normalized (x, y) points

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


class Detector(Protocol):
    name: str

    def detect(self, bgr: Any) -> list[Detection]: ...


class YuNetDetector:
    name = "yunet"

    def __init__(self, model_path: str | Path, score: float = 0.7) -> None:
        import cv2

        self._cv2 = cv2
        self._model = str(model_path)
        self._score = score
        self._size: tuple[int, int] | None = None
        self._net = cv2.FaceDetectorYN.create(self._model, "", (320, 240), score, 0.3, 50)

    def detect(self, bgr: Any) -> list[Detection]:
        h, w = bgr.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(bgr)
        if faces is None:
            return []
        out: list[Detection] = []
        for f in faces:
            x, y, bw, bh = (float(v) for v in f[:4])
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            lms = tuple((float(f[4 + 2 * i]) / w, float(f[5 + 2 * i]) / h) for i in range(5))
            out.append(
                Detection(
                    "face",
                    x0 / w,
                    y0 / h,
                    (x1 - x0) / w,
                    (y1 - y0) / h,
                    score=float(f[14]),
                    landmarks=lms,
                )
            )
        return out


class HaarDetector:
    name = "haar"

    def __init__(self, labels: tuple[str, ...] = ("face", "cat")) -> None:
        from .face import FaceDetector

        self._inner = FaceDetector(labels)
        self.labels = labels

    def detect(self, bgr: Any) -> list[Detection]:
        h, w = bgr.shape[:2]
        return [
            Detection(label, x / w, y / h, bw / w, bh / h)
            for label, x, y, bw, bh in self._inner.detect_all(bgr)
        ]


class CompositeDetector:
    def __init__(self, parts: list[Detector]) -> None:
        self.parts = parts
        self.name = "+".join(p.name for p in parts) or "none"

    def detect(self, bgr: Any) -> list[Detection]:
        out: list[Detection] = []
        for p in self.parts:
            out.extend(p.detect(bgr))
        return out


def build_detector(cfg, models_root: Path, log: logging.Logger | None = None) -> CompositeDetector:
    """From ``PerceptionConfig``. YuNet if requested and present, else Haar faces."""
    log = log or logging.getLogger("shikigami.perception")
    parts: list[Detector] = []
    haar_labels: list[str] = []
    if cfg.face_detector == "yunet":
        model = models_root / cfg.models_dir / YUNET_FILE
        if model.is_file():
            try:
                parts.append(YuNetDetector(model, score=cfg.face_score))
            except Exception as exc:
                log.warning("YuNet failed to load (%s) — falling back to Haar faces", exc)
                haar_labels.append("face")
        else:
            log.warning("YuNet model missing (%s) — run `make setup-vision`; using Haar", model)
            haar_labels.append("face")
    else:
        haar_labels.append("face")
    if cfg.cats:
        haar_labels.append("cat")
    if haar_labels:
        parts.append(HaarDetector(tuple(haar_labels)))
    return CompositeDetector(parts)
