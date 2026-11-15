"""Structured actions the LLM can request via tool/function calling, and their parsing."""

from __future__ import annotations

import json
from enum import StrEnum
from types import SimpleNamespace
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ActionType(StrEnum):
    MOVE = "move"
    SAY = "say"
    EMOTE = "emote"
    REMEMBER = "remember"
    IDLE = "idle"


class MoveAction(BaseModel):
    type: Literal[ActionType.MOVE] = ActionType.MOVE
    motion: str = "walk"  # walk|turn|sit|stand|lie|wag|look_at|stop|<named action>
    direction: str = "forward"  # forward|back|left|right
    speed: float = 0.5
    duration_s: float | None = None
    angle_deg: float | None = None


class SayAction(BaseModel):
    type: Literal[ActionType.SAY] = ActionType.SAY
    text: str


class EmoteAction(BaseModel):
    type: Literal[ActionType.EMOTE] = ActionType.EMOTE
    name: str


class RememberAction(BaseModel):
    type: Literal[ActionType.REMEMBER] = ActionType.REMEMBER
    key: str
    value: str


class IdleAction(BaseModel):
    type: Literal[ActionType.IDLE] = ActionType.IDLE
    seconds: float = 5.0


Action = Annotated[
    MoveAction | SayAction | EmoteAction | RememberAction | IdleAction,
    Field(discriminator="type"),
]
ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

TOOL_NAMES = {a.value for a in ActionType}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the body: walk, turn, sit, stand, lie, wag, look_at, stop, ...",
            "parameters": {
                "type": "object",
                "properties": {
                    "motion": {
                        "type": "string",
                        "enum": ["walk", "turn", "sit", "stand", "lie", "wag", "look_at", "stop"],
                    },
                    "direction": {"type": "string", "enum": ["forward", "back", "left", "right"]},
                    "speed": {"type": "number", "minimum": 0, "maximum": 1},
                    "duration_s": {"type": "number", "minimum": 0},
                    "angle_deg": {"type": "number"},
                },
                "required": ["motion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Speak a short, in-character reply out loud.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emote",
            "description": "Show a facial/expression animation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["neutral", "happy", "curious", "surprised", "sad", "sleepy"],
                    }
                },
                "required": ["name"],
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
]


def parse_tool_call(name: str, arguments: str | dict[str, Any]) -> Action:
    """Parse one tool call (name + JSON arguments) into a validated Action."""
    if name not in TOOL_NAMES:
        raise ValueError(f"unknown tool {name!r}; expected one of {sorted(TOOL_NAMES)}")
    args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    if not isinstance(args, dict):
        raise ValueError(f"tool arguments for {name!r} must be an object, got {type(args)}")
    return ACTION_ADAPTER.validate_python({"type": name, **args})


def parse_message(message: Any) -> list[Action]:
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


def fake_message(tool_calls: list[tuple[str, dict[str, Any]]], content: str = "") -> Any:
    """Build a SimpleNamespace shaped like a chat completion message (tests/mock)."""
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name=n, arguments=json.dumps(a)))
            for n, a in tool_calls
        ],
        content=content,
    )
