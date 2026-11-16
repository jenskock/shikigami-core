"""Pipecat wiring: tool handlers, barge-in gate, and a fake-audio speak path."""

import asyncio
from types import SimpleNamespace

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame, VADUserStartedSpeakingFrame
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from shikigami.brain.actions import TOOL_SCHEMAS, MoveAction, SayAction
from shikigami.brain.agent import Brain
from shikigami.config import REPO_ROOT, LLMConfig, load_config
from shikigami.memory.store import MemoryStore
from shikigami.personality.loader import load_personality
from shikigami.voice.mute import MicMute, MovingGateTurnStart, interruptions_enabled
from shikigami.voice.pipeline import (
    CompatibleLLMService,
    VoiceSession,
    _user_params,
    context_has_dialogue,
    empty_dialogue_rejection,
)
from shikigami.voice.tools import build_tool_schemas
from shikigami.voice.transport import FakeAudioTransport


class _Hooks:
    def __init__(self) -> None:
        self.tts_enabled = True
        self.actions = []
        self.speech = 0
        self.depth = 0

    def note_tool(self, action) -> None:
        self.actions.append(action)

    async def execute(self, action) -> None:
        self.actions.append(("ran", action))

    def expect_speech(self) -> None:
        self.speech += 1

    def tool_enter(self) -> None:
        self.depth += 1

    def tool_exit(self) -> None:
        self.depth -= 1


class _LLM:
    def __init__(self) -> None:
        self.frames = []

    async def push_frame(self, frame) -> None:
        self.frames.append(frame)


def test_tool_schemas_match_and_say_speaks_without_a_second_generation():
    hooks = _Hooks()
    schemas = build_tool_schemas(hooks)
    names = [schema.name for schema in schemas]
    assert names == [spec["function"]["name"] for spec in TOOL_SCHEMAS]
    say = next(schema for schema in schemas if schema.name == "say")
    move = next(schema for schema in schemas if schema.name == "move")
    assert say.properties["text"]["type"] == "string"
    assert "wag" in move.properties["motion"]["enum"]

    llm = _LLM()
    results = []

    async def result_callback(result, *, properties=None):
        results.append((result, properties))

    async def run() -> None:
        await say.handler(
            FunctionCallParams(
                function_name="say",
                tool_call_id="s",
                arguments={"text": "Hallo"},
                llm=llm,  # type: ignore[arg-type]
                pipeline_worker=None,  # type: ignore[arg-type]
                context=None,  # type: ignore[arg-type]
                result_callback=result_callback,
            )
        )
        await move.handler(
            FunctionCallParams(
                function_name="move",
                tool_call_id="m",
                arguments={"motion": "wag"},
                llm=llm,  # type: ignore[arg-type]
                pipeline_worker=None,  # type: ignore[arg-type]
                context=None,  # type: ignore[arg-type]
                result_callback=result_callback,
            )
        )

    asyncio.run(run())
    assert isinstance(hooks.actions[0], SayAction)
    assert hooks.actions[0].text == "Hallo"
    assert hooks.speech == 1
    assert isinstance(llm.frames[0], TTSSpeakFrame)
    assert llm.frames[0].text == "Hallo"
    ran = [item for item in hooks.actions if isinstance(item, tuple)]
    assert len(ran) == 1 and isinstance(ran[0][1], MoveAction)
    assert len(llm.frames) == 1
    assert hooks.depth == 0
    assert all(props is not None and props.run_llm is False for _result, props in results)


def test_interruptions_follow_motion():
    assert interruptions_enabled(barge_in=True, interrupt_while_moving=False, moving=False)
    assert not interruptions_enabled(barge_in=True, interrupt_while_moving=False, moving=True)
    assert interruptions_enabled(barge_in=True, interrupt_while_moving=True, moving=True)
    assert not interruptions_enabled(barge_in=False, interrupt_while_moving=True, moving=False)

    moving = {"value": True}
    gate = MovingGateTurnStart(
        barge_in=True,
        interrupt_while_moving=False,
        moving=lambda: moving["value"],
    )
    seen: list[bool] = []

    async def capture(**_kwargs):
        seen.append(gate._enable_interruptions)

    gate.report_user_turn_started = capture  # type: ignore[method-assign]

    async def run() -> None:
        await gate.process_frame(VADUserStartedSpeakingFrame())
        moving["value"] = False
        await gate.process_frame(VADUserStartedSpeakingFrame())

    asyncio.run(run())
    assert seen == [False, True]


def test_mic_pipeline_uses_silero_and_smart_turn():
    cfg = load_config()
    cfg.voice.smart_turn = True
    params = _user_params(cfg, audio_in=True, mute=MicMute(), moving=lambda: False)
    assert isinstance(params.vad_analyzer, SileroVADAnalyzer)
    stop = params.user_turn_strategies.stop[0]
    assert isinstance(stop, TurnAnalyzerUserTurnStopStrategy)
    assert isinstance(stop._turn_analyzer, LocalSmartTurnAnalyzerV3)
    assert stop._turn_analyzer._params.stop_secs == cfg.sim.vad_silence_s
    assert params.empty_user_turn is None
    assert params.user_turn_stop_timeout == cfg.stt.timeout_s
    assert CompatibleLLMService.supports_developer_role is False


def test_text_pipeline_does_not_load_smart_turn():
    cfg = load_config()
    params = _user_params(cfg, audio_in=False, mute=MicMute(), moving=lambda: False)
    assert params.vad_analyzer is None
    assert isinstance(params.user_turn_strategies.stop[0], SpeechTimeoutUserTurnStopStrategy)
    assert params.empty_user_turn is None


def test_system_or_developer_only_context_is_not_sent():
    system = {"role": "system", "content": "You are Kogane"}
    developer = {
        "role": "developer",
        "content": "The user may have said something, but it was not recognized.",
    }
    assert not context_has_dialogue([system])
    assert not context_has_dialogue([system, developer])
    assert not context_has_dialogue([system, {"role": "user", "content": "   "}])
    assert context_has_dialogue([system, {"role": "user", "content": "Hallo"}])
    assert context_has_dialogue(
        [system, {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]
    )
    rejection = Exception(
        "Error code: 400 - messages must contain at least one non-system message with content"
    )
    assert empty_dialogue_rejection(rejection)
    assert not empty_dialogue_rejection(Exception("Error code: 400 - unknown model"))


class _BytesTTS(TTSService):
    def __init__(self, pcm: bytes) -> None:
        super().__init__(
            settings=TTSSettings(model="bytes", voice=None, language=None),
            sample_rate=16000,
            push_start_frame=True,
            push_stop_frames=True,
        )
        self._pcm = pcm

    async def run_tts(self, text: str, context_id: str):
        del text
        yield TTSAudioRawFrame(self._pcm, 16000, 1, context_id=context_id)


def test_fake_transport_plays_tts_and_marks_speaking(tmp_path):
    cfg = load_config()
    cfg.memory.path = str(tmp_path / "m.db")
    cfg.sim.input = "text"
    cfg.stt.enabled = False
    cfg.tts.enabled = True
    cfg.tts.provider = "none"  # the injected service is what speaks
    cfg.voice.transport = "fake"
    cfg.voice.smart_turn = False
    memory = MemoryStore(cfg.memory.path)
    brain = Brain(
        LLMConfig(mock=False),
        load_personality(REPO_ROOT / "config" / "personality.yaml"),
        memory,
    )
    transport = FakeAudioTransport()
    spoken: list[bool] = []

    async def speak_visual(on: bool) -> None:
        spoken.append(on)

    pcm = b"\x00\x10" * 8000  # 0.5 s at 16 kHz
    session = VoiceSession(
        cfg,
        brain,
        memory,
        execute=lambda _action: _async_none(),
        expression=SimpleNamespace(speak_visual=speak_visual),
        moving=lambda: False,
        on_level=None,
        state=lambda: {"mood": "neutral", "moving": False, "backend": "sim"},
        events=lambda: [],
        note_tool_line=lambda _action: None,
        transport=transport,
        tts_service=_BytesTTS(pcm),
    )

    async def run() -> None:
        await session.start()
        assert session._worker is not None
        await session._worker.queue_frame(TTSSpeakFrame("Hallo"))
        for _ in range(50):
            if transport.played and any(spoken):
                break
            await asyncio.sleep(0.1)
        await session.stop()

    asyncio.run(run())
    assert b"".join(transport.played)
    assert True in spoken
    memory.close()


async def _async_none():
    return None
