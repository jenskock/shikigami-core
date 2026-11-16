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
    format_tool_call,
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
    assert a.steps is None


def test_parse_move_steps():
    a = parse_tool_call("move", {"motion": "walk", "steps": 3, "direction": "forward"})
    assert isinstance(a, MoveAction)
    assert a.steps == 3


def test_duration_for_move_prefers_steps():
    from shikigami.app.loop import _SECS_PER_STEP, _duration_for_move

    stepped = MoveAction(motion="walk", steps=3, duration_s=9.0)
    assert _duration_for_move(stepped) == pytest.approx(3 * _SECS_PER_STEP)
    timed = MoveAction(motion="walk", duration_s=1.2)
    assert _duration_for_move(timed) == pytest.approx(1.2)
    assert _duration_for_move(MoveAction(motion="sit")) is None


def test_move_schema_requires_real_body_action():
    from shikigami.brain.actions import TOOL_SCHEMAS

    move = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "move")
    desc = move["function"]["description"]
    assert "Speech alone does nothing" in desc
    props = move["function"]["parameters"]["properties"]
    assert "steps" in props
    assert "dance" in props["motion"]["enum"]
    assert "tanzen" in props["motion"]["description"].lower()


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
    msg = fake_message([("emote", {"name": "curious"}), ("move", {"motion": "wag"})])
    actions = parse_message(msg)
    assert [type(a) for a in actions] == [EmoteAction, MoveAction]


def test_tool_schemas_have_no_say():
    from shikigami.brain.actions import TOOL_SCHEMAS

    names = [spec["function"]["name"] for spec in TOOL_SCHEMAS]
    assert "say" not in names
    assert "move" in names and "emote" in names


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


def test_format_tool_call():
    assert format_tool_call(EmoteAction(name="happy")) == "emote(name='happy')"
    assert "say(text=" in format_tool_call(SayAction(text="hi"))
    long = format_tool_call(SayAction(text="x" * 100), max_text=20)
    assert "…" in long and len(long) < 60
