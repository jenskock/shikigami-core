"""Phase 4: speaking up on its own — greetings, unknown faces, cooldowns, quiet mode."""

import asyncio
from types import SimpleNamespace

from shikigami.app.bus import NOT_PERSISTED
from shikigami.brain.actions import QuietAction, parse_tool_call
from shikigami.brain.attention import AttentionPolicy
from shikigami.config import AttentionConfig


class KV(dict):
    def get_value(self, key, default=None):
        return self.get(key, default)

    def set_value(self, key, value):
        self[key] = value


class World:
    def __init__(self, **cfg):
        self.wall = 1_000_000.0
        self.mono = 5_000.0
        self.can = True
        self.last_turn = 0.0
        self.notices, self.emotes = [], []
        self.kv = KV()
        self.policy = AttentionPolicy(
            AttentionConfig(**cfg),
            self.kv,
            notice=self.notice,
            can_speak=lambda: self.can,
            last_turn=lambda: self.last_turn,
            emote=self.emote,
            clock=lambda: self.wall,
            mono=lambda: self.mono,
        )

    async def notice(self, note):
        if not self.can:
            return False
        self.notices.append(note)
        return True

    async def emote(self, name):
        self.emotes.append(name)

    def arrive(self, name="Jens", advance_s=0.0):
        self.wall += advance_s
        self.mono += advance_s
        ev = SimpleNamespace(kind="person_arrived", payload={"name": name})
        asyncio.run(self.policy.on_event(ev))


def test_known_person_greeted_once_per_cooldown():
    w = World()
    w.arrive("Jens")
    assert len(w.notices) == 1 and "Greet Jens" in w.notices[0]
    w.arrive("Jens", advance_s=600)  # 10 min later: within 3 h
    assert len(w.notices) == 1 and w.emotes[-1] == "happy"
    w.arrive("Jens", advance_s=3 * 3600)
    assert len(w.notices) == 2


def test_cooldown_survives_restart():
    w = World()
    w.arrive("Jens")
    w2 = World()
    w2.kv = w.kv
    w2.policy._kv = w.kv
    w2.arrive("Jens", advance_s=60)
    assert w2.notices == []


def test_no_greeting_right_after_a_conversation_turn_but_counts_as_greeted():
    w = World()
    w.last_turn = w.mono - 5  # we were just talking
    w.arrive("Jens")
    assert w.notices == [] and w.emotes == ["happy"]
    w.last_turn = 0.0
    w.arrive("Jens", advance_s=120)  # later arrival within cooldown: still no greeting
    assert w.notices == []


def test_pipeline_busy_or_not_ready_does_not_mark_greeted():
    w = World()
    w.can = False  # e.g. right at startup, or a turn in flight
    w.arrive("Jens")
    assert w.notices == [] and "greeted:jens" not in w.kv
    w.can = True
    w.arrive("Jens", advance_s=30)
    assert len(w.notices) == 1


def test_quiet_mode_blocks_speech_but_not_eyes():
    w = World()
    w.policy.set_quiet(30)
    w.arrive("Jens")
    assert w.notices == [] and w.emotes == ["happy"]
    w.policy.set_quiet(0)
    w.arrive("Jens", advance_s=1)
    assert len(w.notices) == 1


def test_quiet_mode_expires():
    w = World()
    w.policy.set_quiet(None)  # default 30 min
    w.arrive("Jens", advance_s=31 * 60)
    assert len(w.notices) == 1


def test_min_gap_between_proactive_lines():
    w = World()
    w.arrive("Jens")
    w.arrive("Anna", advance_s=10)  # someone else 10 s later: too soon to talk again
    assert len(w.notices) == 1 and w.emotes[-1] == "happy"
    w.arrive("Anna", advance_s=120)
    assert len(w.notices) == 2


def test_unknown_face_asked_once_per_cooldown_and_always_curious():
    w = World()
    w.arrive("unknown")
    assert len(w.notices) == 1 and "who they are" in w.notices[0]
    w.arrive("unknown", advance_s=5 * 60)
    assert len(w.notices) == 1 and w.emotes == ["curious", "curious"]
    w2 = World(ask_unknown=False)
    w2.arrive("unknown")
    assert w2.notices == [] and w2.emotes == ["curious"]


def test_proactive_off_only_reacts_with_eyes():
    w = World(proactive=False)
    w.arrive("Jens")
    w.arrive("unknown", advance_s=120)
    assert w.notices == [] and w.emotes == ["happy", "curious"]


def test_quiet_tool_parses():
    a = parse_tool_call("quiet", {"minutes": 10})
    assert isinstance(a, QuietAction) and a.minutes == 10
    assert parse_tool_call("quiet", {}).minutes is None


def test_reflex_events_not_persisted():
    assert {"face_seen", "face_lost"} <= NOT_PERSISTED
    assert "person_arrived" not in NOT_PERSISTED


def test_voice_notice_never_queues_and_marks_the_message(tmp_path):
    from shikigami.brain.agent import Brain
    from shikigami.config import REPO_ROOT, Config, LLMConfig
    from shikigami.memory.store import MemoryStore
    from shikigami.personality.loader import load_personality
    from shikigami.voice.pipeline import VoiceSession

    cfg = Config()
    memory = MemoryStore(tmp_path / "m.sqlite3")
    brain = Brain(
        LLMConfig(mock=False), load_personality(REPO_ROOT / "config/personality.yaml"), memory
    )

    async def none(*_a):
        return None

    session = VoiceSession(
        cfg,
        brain,
        memory,
        execute=none,
        expression=SimpleNamespace(speak_visual=none),
        moving=lambda: False,
        on_level=None,
        state=lambda: {},
        events=lambda: [],
        note_tool_line=lambda _a: None,
    )
    queued = []

    class Worker:
        async def queue_frame(self, frame):
            queued.append(frame)

    async def main():
        assert await session.notice("x") is False  # pipeline not running
        session._worker = Worker()
        session._user_speaking = True
        assert await session.notice("x") is False  # someone is talking
        session._user_speaking = False
        assert await session.notice("Jens came into view.") is True
        assert not session.idle  # the proactive turn owns the pipeline now
        assert await session.notice("again") is False  # never queues up
        session.finish_turn()

    asyncio.run(main())
    (frame,) = queued
    msg = frame.messages[0]
    assert msg["role"] == "user" and msg["content"].startswith("(Nobody said this.")
    memory.close()


def test_tool_only_response_asks_for_words_once():
    from pipecat.services.llm_service import FunctionCallParams

    from shikigami.voice.tools import build_tool_schemas

    class Hooks:
        def __init__(self):
            self.nudges = 0

        def note_tool(self, a): ...
        async def execute(self, a): ...
        def tool_enter(self):
            return 0

        def tool_exit(self, turn=None): ...

        def nudge_for_words(self):
            self.nudges += 1
            return self.nudges == 1  # VoiceSession allows one per user message

    hooks = Hooks()
    emote = next(s for s in build_tool_schemas(hooks) if s.name == "emote")
    runs = []

    async def cb(result, *, properties=None):
        runs.append(properties.run_llm)

    def call(spoken):
        llm = SimpleNamespace(response_text=spoken)
        asyncio.run(
            emote.handler(
                FunctionCallParams(
                    function_name="emote",
                    tool_call_id="e",
                    arguments={"name": "curious"},
                    llm=llm,  # type: ignore[arg-type]
                    pipeline_worker=None,  # type: ignore[arg-type]
                    context=None,  # type: ignore[arg-type]
                    result_callback=cb,
                )
            )
        )

    call("Hallo du!")  # words came with the emote: no follow-up
    call("")  # emote only: run the LLM once more so it speaks
    call("")  # still nothing: no loop
    assert runs == [False, True, False]


def test_voice_session_nudges_once_per_user_message(tmp_path):
    from shikigami.voice.pipeline import VoiceSession

    s = VoiceSession.__new__(VoiceSession)
    s._nudged_at, s._user_msgs = -1, 3
    s._log = __import__("logging").getLogger("t")
    assert s.nudge_for_words() is True
    assert s.nudge_for_words() is False  # same message: the follow-up stayed silent too
    s._user_msgs = 4  # the human said something new
    assert s.nudge_for_words() is True
