"""Brain with the mock LLM: full turn in -> structured actions out, no network."""

import asyncio

from shikigami.brain.actions import EmoteAction, MoveAction, SayAction
from shikigami.brain.agent import Brain
from shikigami.config import REPO_ROOT, LLMConfig
from shikigami.memory.store import MemoryStore
from shikigami.personality.loader import load_personality


def _brain(tmp_path) -> Brain:
    cfg = LLMConfig(mock=True)
    personality = load_personality(REPO_ROOT / "config" / "personality.yaml")
    return Brain(cfg, personality, MemoryStore(tmp_path / "m.sqlite3"))


def test_mock_respond_returns_actions(tmp_path):
    brain = _brain(tmp_path)
    result = asyncio.run(brain.respond("hello there", [], {"mood": "neutral"}))
    types = [type(a) for a in result.actions]
    assert SayAction in types and EmoteAction in types and MoveAction in types
    say = next(a for a in result.actions if isinstance(a, SayAction))
    assert "hello there" in say.text
    assert result.model == "mock"
    assert result.total_ms >= 0


def test_system_prompt_contains_personality(tmp_path):
    brain = _brain(tmp_path)
    brain._memory.remember("name", "Jens")
    msgs = brain.build_messages("hi", [], {"mood": "curious"})
    assert msgs[0]["role"] == "system"
    assert "Kogane" in msgs[0]["content"]
    assert "name: Jens" in msgs[0]["content"]
    assert "always reply in German" in msgs[0]["content"]
    assert "say" in msgs[0]["content"]
    assert msgs[1]["content"].endswith("human says: hi")


def test_system_content_includes_situation_without_the_user_line(tmp_path):
    brain = _brain(tmp_path)
    text = brain.system_content({"mood": "curious", "moving": False}, [])
    assert "Kogane" in text
    assert "state:" in text
    assert "human says:" not in text
