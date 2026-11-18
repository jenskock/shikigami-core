"""Structured actions the LLM can request via tool/function calling, and their parsing."""

from __future__ import annotations

import json
from enum import StrEnum
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field, TypeAdapter


class _ToolFunction(Protocol):
    name: str
    arguments: str | dict[str, Any]


class _ToolCall(Protocol):
    function: _ToolFunction


class ChatMessage(Protocol):
    tool_calls: list[_ToolCall] | None
    content: str | None


class ActionType(StrEnum):
    MOVE = "move"
    SAY = "say"
    EMOTE = "emote"
    REMEMBER = "remember"
    IDLE = "idle"
    QUIET = "quiet"


class MoveAction(BaseModel):
    type: Literal[ActionType.MOVE] = ActionType.MOVE
    motion: str = "walk"  # walk|turn|sit|stand|lie|wag|look_at|stop|dance|wave|…
    direction: str = "forward"  # forward|back|left|right
    speed: float = 0.5
    duration_s: float | None = None
    steps: int | None = None  # gait cycles for walk/turn (e.g. "3 Schritte" → 3)
    angle_deg: float | None = None


class SayAction(BaseModel):
    type: Literal[ActionType.SAY] = ActionType.SAY
    text: str


class EmoteAction(BaseModel):
    type: Literal[ActionType.EMOTE] = ActionType.EMOTE
    name: str = "neutral"  # models sometimes omit name; neutral is the honest default


class RememberAction(BaseModel):
    type: Literal[ActionType.REMEMBER] = ActionType.REMEMBER
    key: str
    value: str


class IdleAction(BaseModel):
    type: Literal[ActionType.IDLE] = ActionType.IDLE
    seconds: float = 5.0


class QuietAction(BaseModel):
    type: Literal[ActionType.QUIET] = ActionType.QUIET
    minutes: float | None = None  # None = default; 0 = may speak up again


Action = Annotated[
    MoveAction | SayAction | EmoteAction | RememberAction | IdleAction | QuietAction,
    Field(discriminator="type"),
]
ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

TOOL_NAMES = {a.value for a in ActionType}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": (
                "Actually move the robot body. Required whenever the human asks "
                "for a physical action (walk/laufen, turn/drehen, dance/tanzen, "
                "sit, stand, wave, …). Speech alone does nothing — call this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion": {
                        "type": "string",
                        "enum": [
                            "walk",
                            "turn",
                            "sit",
                            "stand",
                            "lie",
                            "wag",
                            "look_at",
                            "stop",
                            # Picrawler / SunFounder named actions (via play_action)
                            "dance",
                            "wave",
                            "push_up",
                            "forward",
                            "backward",
                            "turn_left",
                            "turn_right",
                        ],
                        "description": (
                            "Body verb. Use dance for tanzen/dance (never walk). "
                            "Use walk/forward/backward for walking. "
                            "Use turn_left/turn_right or turn+direction for turning."
                        ),
                    },
                    "direction": {"type": "string", "enum": ["forward", "back", "left", "right"]},
                    "speed": {"type": "number", "minimum": 0, "maximum": 1},
                    "duration_s": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Optional seconds. Prefer steps for counted strides.",
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Gait cycles for walk/turn. '3 Schritte' / 'three steps' → steps=3."
                        ),
                    },
                    "angle_deg": {"type": "number"},
                },
                "required": ["motion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emote",
            "description": (
                "Set the face. Pick the one that fits this exact moment, not a favourite. "
                "neutral: plain facts, listening, nothing special. happy: really pleased, "
                "praised, greeted. love: affection, being petted, thanked. curious: a "
                "question, something new, being asked to look. surprised: unexpected news. "
                "scared: danger, being startled or asked to be scared. sad: bad news, "
                "sorry, left alone. angry: told off, annoyed, playfully grumpy. sleepy: "
                "tired, bedtime, lying down. confused: did not understand. focused: "
                "concentrating on a task, walking, counting. mischievous: joking, teasing, "
                "tricks, dancing. Defaults to neutral."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "neutral",
                            "happy",
                            "sad",
                            "angry",
                            "surprised",
                            "sleepy",
                            "curious",
                            "love",
                            "confused",
                            "scared",
                            "focused",
                            "mischievous",
                        ],
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a fact about the human in long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "idle",
            "description": "Do nothing for a while (breathe/idle animation).",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "number", "minimum": 0}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quiet",
            "description": (
                "Stop speaking up on your own (greetings, questions) for a while, when "
                'the human asks you to be quiet ("sei mal still", "lass mich in Ruhe"). '
                "minutes=0 when they say you may talk again. You still answer when spoken to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "number",
                        "minimum": 0,
                        "description": "How long. Omit for the default; 0 = talk again.",
                    }
                },
            },
        },
    },
]


def parse_tool_call(name: str, arguments: str | dict[str, Any]) -> Action:
    """Parse one tool call (name + JSON arguments) into a validated Action."""
    if name not in TOOL_NAMES:
        raise ValueError(f"unknown tool {name!r}; expected one of {sorted(TOOL_NAMES)}")
    args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    if not isinstance(args, dict):
        raise ValueError(f"tool arguments for {name!r} must be an object, got {type(args)}")
    return ACTION_ADAPTER.validate_python({"type": name, **args})


def format_tool_call(action: Action, *, max_text: int = 80) -> str:
    """Human-readable `name(arg=…)` line for logs / dashboard transparency."""
    data = action.model_dump(exclude={"type"}, exclude_none=True)
    parts: list[str] = []
    for key, value in data.items():
        if isinstance(value, str):
            text = value if len(value) <= max_text else value[: max_text - 1] + "…"
            parts.append(f"{key}={text!r}")
        else:
            parts.append(f"{key}={value!r}")
    return f"{action.type.value}({', '.join(parts)})"


def parse_message(message: ChatMessage) -> list[Action]:
    """Parse a chat-completion message (object with .tool_calls / .content) into actions.

    Plain text content with no tool calls becomes a single SayAction.
    """
    actions: list[Action] = []
    for tc in message.tool_calls or []:
        actions.append(parse_tool_call(tc.function.name, tc.function.arguments))
    if not actions:
        content = (message.content or "").strip()
        if content:
            actions.append(SayAction(text=content))
    return actions


def fake_message(tool_calls: list[tuple[str, dict[str, Any]]], content: str = "") -> ChatMessage:
    """Build a SimpleNamespace shaped like a chat completion message (tests/mock)."""
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name=n, arguments=json.dumps(a)))
            for n, a in tool_calls
        ],
        content=content,
    )
