"""Detection with OpenCV Haar cascades (lazy import; optional at runtime).

Faces plus cat faces — cheap on CPU, and a robot buddy should notice cats.
"""

from __future__ import annotations

from typing import Any

# label -> (cascade file, minSize) — minSize filters small false positives (glasses etc.)
CASCADES = {
    "face": ("haarcascade_frontalface_default.xml", (80, 80)),
    "cat": ("haarcascade_frontalcatface.xml", (48, 48)),
}


class FaceDetector:
    def __init__(self, labels: tuple[str, ...] = ("face", "cat")) -> None:
        import cv2  # lazy

        self._cv2 = cv2
        self._cascades = {
            label: (
                cv2.CascadeClassifier(cv2.data.haarcascades + CASCADES[label][0]),
                CASCADES[label][1],
            )
            for label in labels
        }

    def detect(self, frame: Any) -> list[tuple[int, int, int, int]]:
        """Return (x, y, w, h) boxes for detected human faces in a BGR frame."""
        return [box for label, *box in self.detect_all(frame) if label == "face"]

    def detect_all(self, frame: Any) -> list[tuple[str, int, int, int, int]]:
        """Return (label, x, y, w, h) boxes for everything the cascades find (full-res frame)."""
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        found: list[tuple[str, int, int, int, int]] = []
        for label, (cascade, min_size) in self._cascades.items():
            for x, y, w, h in cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
            ):
                found.append((label, int(x), int(y), int(w), int(h)))
        return found
