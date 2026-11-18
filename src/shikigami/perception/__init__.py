"""Perception: detection on the camera's low-res stream, face tracking. Same code on Mac and Pi.

Turn-taking lives in the voice pipeline.
"""

from .detectors import Detection, build_detector
from .pipeline import Perception, PerceptionResult
from .tracker import FaceTracker

__all__ = ["Detection", "FaceTracker", "Perception", "PerceptionResult", "build_detector"]
