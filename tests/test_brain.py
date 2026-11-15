"""Brain with the mock LLM: full turn in -> structured actions out, no network."""

import asyncio
from types import SimpleNamespace

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
    assert msgs[1]["content"].endswith("human says: hi")


def _tc(index, name=None, arguments=None):
    return SimpleNamespace(index=index, function=SimpleNamespace(name=name, arguments=arguments))


def _chunk(tool_calls=None, content=None, finish=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, delta=delta)])


class _FakeClient:
    def __init__(self, chunks):
        completions = SimpleNamespace(create=lambda **kw: iter(chunks))
        self.chat = SimpleNamespace(completions=completions)


def test_streaming_emits_actions_and_sentences_incrementally(tmp_path):
    chunks = [
        _chunk(tool_calls=[_tc(0, "emote", '{"name": "happy"}')]),
        _chunk(tool_calls=[_tc(1, "say", '{"text": "Hi there! How are')]),
        _chunk(tool_calls=[_tc(1, None, ' you today?"}')], finish="tool_calls"),
    ]
    brain = Brain(
        LLMConfig(mock=False),
        load_personality(REPO_ROOT / "config/personality.yaml"),
        MemoryStore(tmp_path / "m.db"),
    )
    brain._client = _FakeClient(chunks)
    actions_seen, sentences = [], []

    async def on_action(a):
        actions_seen.append(a)

    async def on_sentence(s):
        sentences.append(s)

    result = asyncio.run(
        brain.respond("hi", [], {"mood": "neutral"}, on_action=on_action, on_sentence=on_sentence)
    )
    # emote finalized when the next tool call starts; say at finish_reason
    assert [type(a) for a in actions_seen] == [EmoteAction, SayAction]
    # "Hi there!" streamed as soon as its sentence boundary arrived; tail flushed at end
    assert sentences == ["Hi there!", "How are you today?"]
    assert result.reply == "Hi there! How are you today?"
    assert result.ttft_ms is not None
    assert [type(a) for a in result.actions] == [EmoteAction, SayAction]
