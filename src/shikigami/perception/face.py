"""Detection with OpenCV Haar cascades (lazy import; optional at runtime).

Faces plus cat faces — cheap on CPU. Detection runs on a downscaled gray
frame; boxes are scaled back to the full-res coordinate system.
"""

from __future__ import annotations

from typing import Any

# label -> (cascade file, minSize at ~640px width)
CASCADES = {
    "face": ("haarcascade_frontalface_default.xml", (48, 48)),
    "cat": ("haarcascade_frontalcatface.xml", (32, 32)),
}

DETECT_WIDTH = 320  # detect at this width; scale boxes back to full frame


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
        """Return (label, x, y, w, h) in full-frame coords (detects on a small gray)."""
        cv2 = self._cv2
        fh, fw = frame.shape[:2]
        if fw > DETECT_WIDTH:
            scale = DETECT_WIDTH / fw
            small = cv2.resize(frame, (DETECT_WIDTH, max(1, int(fh * scale))))
        else:
            scale = 1.0
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        inv = 1.0 / scale
        found: list[tuple[str, int, int, int, int]] = []
        for label, (cascade, min_size) in self._cascades.items():
            for x, y, w, h in cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=min_size
            ):
                found.append(
                    (
                        label,
                        int(x * inv),
                        int(y * inv),
                        int(w * inv),
                        int(h * inv),
                    )
                )
        return found
