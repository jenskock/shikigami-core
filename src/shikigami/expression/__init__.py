"""Procedural expression: eyes, emotes, gaze — display-agnostic core + renderers."""

from .eyes import Emotion, EyesEngine, EyeState, FrameState

__all__ = ["EyesEngine", "Emotion", "EyeState", "FrameState"]
