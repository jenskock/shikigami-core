"""Procedural expressive eyes — pure animation core + pluggable renderers."""

from .controller import EyesEngine
from .emotion import EMOTION_NAMES, Emotion, blend_emotions, emotion_targets
from .state import EyesConfig, EyeState, FrameState

__all__ = [
    "EyesEngine",
    "Emotion",
    "EMOTION_NAMES",
    "EyeState",
    "FrameState",
    "EyesConfig",
    "blend_emotions",
    "emotion_targets",
]
