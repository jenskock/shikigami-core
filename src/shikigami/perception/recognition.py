"""Face recognition: SFace embeddings (OpenCV Zoo) + an in-memory gallery of known people.

Embeddings are taken from the ``main`` frame (same size on Mac and Pi), aligned with
the YuNet landmarks, so an enrollment on the Mac matches on the Pi and vice versa.
The gallery lives in memory for the perception thread; ``MemoryStore`` persists it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .detectors import Detection

SFACE_FILE = "face_recognition_sface_2021dec.onnx"
EMBED_DIM = 128
MAX_PER_NAME = 30  # newest embeddings kept per person


class Embedder(Protocol):
    def embed(self, bgr: Any, face: Detection) -> np.ndarray | None: ...


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def yunet_row(face: Detection, w: int, h: int) -> np.ndarray:
    """Rebuild YuNet's 15-float face row (pixel coords) for an image of size w x h."""
    row = [face.x * w, face.y * h, face.w * w, face.h * h]
    for lx, ly in face.landmarks or ():
        row += [lx * w, ly * h]
    row.append(face.score)
    return np.asarray(row, dtype=np.float32).reshape(1, -1)


class SFaceRecognizer:
    name = "sface"

    def __init__(self, model_path: str | Path) -> None:
        import cv2

        self._cv2 = cv2
        self._net = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, bgr: Any, face: Detection) -> np.ndarray | None:
        """L2-normalized 128-d embedding, or None when the face can't be aligned."""
        if not face.landmarks:
            return None  # Haar faces have no landmarks; recognition needs YuNet
        h, w = bgr.shape[:2]
        crop = self._net.alignCrop(bgr, yunet_row(face, w, h))
        return normalize(self._net.feature(crop))


class FaceGallery:
    """Known people -> embeddings. Thread-safe; shared by perception and the app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_name: dict[str, np.ndarray] = {}

    @classmethod
    def from_rows(cls, rows: list[tuple[str, bytes]]) -> FaceGallery:
        g = cls()
        for name, blob in rows:
            g.add(name, [np.frombuffer(blob, dtype=np.float32)])
        return g

    def add(self, name: str, embeddings: list[np.ndarray]) -> None:
        if not embeddings:
            return
        new = np.stack([normalize(e) for e in embeddings])
        with self._lock:
            old = self._by_name.get(name)
            mat = new if old is None else np.vstack([old, new])
            self._by_name[name] = mat[-MAX_PER_NAME:]

    def forget(self, name: str) -> bool:
        with self._lock:
            for key in list(self._by_name):
                if key.lower() == name.lower():
                    del self._by_name[key]
                    return True
        return False

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._by_name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_name)

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Best (name, cosine score) over all people; (None, 0.0) for an empty gallery."""
        e = normalize(embedding)
        best_name, best = None, 0.0
        with self._lock:
            for name, mat in self._by_name.items():
                score = float(np.max(mat @ e))
                if score > best:
                    best_name, best = name, score
        return best_name, best


def build_recognizer(cfg, models_root: Path, log: logging.Logger | None = None):
    """SFace if enabled and downloaded, else None (faces then stay 'unknown')."""
    log = log or logging.getLogger("shikigami.perception")
    if not cfg.recognize:
        return None
    if cfg.face_detector != "yunet":
        log.warning("face recognition needs face_detector: yunet — recognition off")
        return None
    model = models_root / cfg.models_dir / SFACE_FILE
    if not model.is_file():
        log.warning("SFace model missing (%s) — run `make setup-vision`; recognition off", model)
        return None
    try:
        return SFaceRecognizer(model)
    except Exception as exc:
        log.warning("SFace failed to load (%s) — recognition off", exc)
        return None
