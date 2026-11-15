"""Action parsing: tool calls -> validated Action models."""

import json
from types import SimpleNamespace

import pytest

from shikigami.brain.actions import (
    ACTION_ADAPTER,
    ActionType,
    EmoteAction,
    IdleAction,
    MoveAction,
    RememberAction,
    SayAction,
    fake_message,
    parse_message,
    parse_tool_call,
)


def test_parse_move():
    args = json.dumps({"motion": "walk", "direction": "forward", "speed": 0.8, "duration_s": 2.0})
    a = parse_tool_call("move", args)
    assert isinstance(a, MoveAction)
    assert a.type == ActionType.MOVE
    assert a.motion == "walk" and a.speed == 0.8 and a.duration_s == 2.0


def test_parse_move_defaults():
    a = parse_tool_call("move", {"motion": "sit"})
    assert isinstance(a, MoveAction)
    assert a.direction == "forward" and a.speed == 0.5 and a.duration_s is None


def test_parse_say_emote_remember_idle():
    assert isinstance(parse_tool_call("say", {"text": "hi"}), SayAction)
    assert isinstance(parse_tool_call("emote", {"name": "happy"}), EmoteAction)
    r = parse_tool_call("remember", {"key": "favorite_color", "value": "blue"})
    assert isinstance(r, RememberAction) and r.value == "blue"
    i = parse_tool_call("idle", {})
    assert isinstance(i, IdleAction) and i.seconds == 5.0


def test_parse_unknown_tool_raises():
    with pytest.raises(ValueError, match="unknown tool"):
        parse_tool_call("fly", {})


def test_parse_bad_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_call("say", "{not json")


def test_parse_invalid_args_raise():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_tool_call("say", {})  # missing required "text"


def test_parse_message_tool_calls():
    msg = fake_message([("say", {"text": "hello"}), ("emote", {"name": "curious"})])
    actions = parse_message(msg)
    assert [type(a) for a in actions] == [SayAction, EmoteAction]


def test_parse_message_content_fallback():
    msg = SimpleNamespace(tool_calls=None, content="  just words  ")
    actions = parse_message(msg)
    assert len(actions) == 1
    assert isinstance(actions[0], SayAction) and actions[0].text == "just words"


def test_parse_message_empty():
    assert parse_message(SimpleNamespace(tool_calls=[], content="")) == []


def test_action_roundtrip():
    a = parse_tool_call("move", {"motion": "turn", "direction": "right", "angle_deg": 90})
    dumped = ACTION_ADAPTER.dump_python(a)
    assert dumped["type"] == "move" and dumped["angle_deg"] == 90
    again = ACTION_ADAPTER.validate_python(dumped)
    assert again == a
