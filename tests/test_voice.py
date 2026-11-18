"""Pipecat wiring: body tools, the echo guard, turn starts, and a fake-audio speak path."""

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_start.transcription_user_turn_start_strategy import (
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from shikigami.brain.actions import TOOL_SCHEMAS, EmoteAction, MoveAction
from shikigami.brain.agent import Brain
from shikigami.config import REPO_ROOT, LLMConfig, load_config
from shikigami.memory.store import MemoryStore
from shikigami.personality.loader import load_personality
from shikigami.voice.barge_in import (
    MotionAwareVADUserTurnStartStrategy,
    interruptions_enabled,
)
from shikigami.voice.mute import MicMute
from shikigami.voice.pipeline import (
    CompatibleLLMService,
    LlmTurnProcessor,
    TurnGate,
    VoiceSession,
    _http_client,
    _user_params,
    context_has_dialogue,
    empty_dialogue_rejection,
    portable_messages,
)
from shikigami.voice.piper_http import wav_pcm
from shikigami.voice.tools import STOP_TOOL_NAME, build_tool_schemas, speak_frame
from shikigami.voice.transport import (
    FakeAudioTransport,
    Playout,
    SpeakerActivity,
    _SoundInput,
    _SoundOutput,
)


class _Hooks:
    def __init__(self) -> None:
        self.actions = []
        self.depth = 0

    def note_tool(self, action) -> None:
        self.actions.append(action)

    async def execute(self, action) -> None:
        self.actions.append(("ran", action))

    def tool_enter(self) -> int:
        self.depth += 1
        return 7

    def tool_exit(self, turn: int | None = None) -> None:
        assert turn == 7
        self.depth -= 1


class _LLM:
    def __init__(self) -> None:
        self.frames = []

    async def push_frame(self, frame) -> None:
        self.frames.append(frame)


def test_body_tool_schemas_run_without_speak_frames():
    hooks = _Hooks()
    schemas = build_tool_schemas(hooks)
    names = [schema.name for schema in schemas]
    assert names[0] == STOP_TOOL_NAME
    assert names[1:] == [spec["function"]["name"] for spec in TOOL_SCHEMAS]
    assert "say" not in names
    emote = next(schema for schema in schemas if schema.name == "emote")
    move = next(schema for schema in schemas if schema.name == "move")
    assert "wag" in move.properties["motion"]["enum"]

    llm = _LLM()
    results = []

    async def result_callback(result, *, properties=None):
        results.append((result, properties))

    async def run() -> None:
        await emote.handler(
            FunctionCallParams(
                function_name="emote",
                tool_call_id="e",
                arguments={"name": "happy"},
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
    assert isinstance(hooks.actions[0], EmoteAction)
    ran = [item for item in hooks.actions if isinstance(item, tuple)]
    assert len(ran) == 2
    assert isinstance(ran[0][1], EmoteAction)
    assert isinstance(ran[1][1], MoveAction)
    assert llm.frames == []
    assert hooks.depth == 0
    assert all(props is not None and props.run_llm is False for _result, props in results)
    apology = speak_frame("Sorry.")
    assert isinstance(apology, TTSSpeakFrame)
    assert apology.text == "Sorry."
    assert apology.append_to_context is False


def test_interruptions_follow_motion():
    assert interruptions_enabled(barge_in=True, interrupt_while_moving=False, moving=False)
    assert not interruptions_enabled(barge_in=True, interrupt_while_moving=False, moving=True)
    assert interruptions_enabled(barge_in=True, interrupt_while_moving=True, moving=True)
    assert not interruptions_enabled(barge_in=False, interrupt_while_moving=True, moving=False)


def _sound_input(speaker: SpeakerActivity | None, *, gain: float = 1.0) -> _SoundInput:
    return _SoundInput(
        TransportParams(audio_in_enabled=True, audio_in_sample_rate=16000),
        device=0,
        native_rate=16000,
        gain=gain,
        mute=MicMute(),
        on_level=None,
        log=SimpleNamespace(debug=lambda *_a, **_k: None),  # type: ignore[arg-type]
        speaker=speaker,
    )


def _sound_output(speaker: SpeakerActivity, playout: Playout | None) -> _SoundOutput:
    output = _SoundOutput(
        TransportParams(audio_out_enabled=True, audio_out_sample_rate=16000),
        device=0,
        log=SimpleNamespace(debug=lambda *_a, **_k: None),  # type: ignore[arg-type]
        speaker=speaker,
    )
    output._playout = playout
    return output


def test_echo_guard_closes_mic_while_speaker_is_audible():
    """The robot's own voice never reaches VAD or STT."""
    speaker = SpeakerActivity(tail_secs=0.05)
    speaker.add_latency(0.03)
    assert speaker.hold_secs == 0.08
    mic = _sound_input(speaker)
    voice = (np.sin(np.arange(320) / 5.0) * 6000).astype(np.int16)

    pcm, level = mic.capture(voice, 16000)
    assert np.array_equal(pcm, voice)
    assert level > 0

    playout = Playout(64000)
    output = _sound_output(speaker, playout)
    playout.put(voice.tobytes())
    assert output.render(640) == voice.tobytes()
    assert speaker.audible

    pcm, level = mic.capture(voice, 16000)
    assert pcm.shape == voice.shape  # silence, not a gap: the VAD must see speech end
    assert not pcm.any()
    assert level == 0.0

    # Silence from the device does not keep the mic closed.
    time.sleep(0.05)
    assert output.render(640) == bytes(640)
    time.sleep(0.05)
    assert not speaker.audible
    pcm, _level = mic.capture(voice, 16000)
    assert np.array_equal(pcm, voice)


def test_mic_stays_open_without_echo_guard():
    speaker = SpeakerActivity(tail_secs=1.0)
    speaker.mark()
    mic = _sound_input(None, gain=2.0)
    voice = np.full(320, 1000, dtype=np.int16)
    pcm, _level = mic.capture(voice, 16000)
    assert np.array_equal(pcm, voice * 2)


def test_playout_pads_with_silence_and_paces_the_writer():
    playout = Playout(8)
    assert playout.take(4) == bytes(4)
    assert playout.put(b"\x01\x02\x03\x04")
    assert playout.take(6) == b"\x01\x02\x03\x04\x00\x00"

    done: list[bool] = []
    writer = threading.Thread(target=lambda: done.append(playout.put(bytes(range(1, 13)))))
    writer.start()
    time.sleep(0.05)
    assert done == []  # 12 bytes do not fit in 8: the writer waits for the device
    assert playout.take(8) == bytes(range(1, 9))
    writer.join(timeout=2)
    assert done == [True]
    assert playout.take(4) == bytes(range(9, 13))


def test_playout_clear_drops_speech_on_interruption():
    playout = Playout(8)
    done: list[bool] = []
    writer = threading.Thread(target=lambda: done.append(playout.put(bytes(range(1, 21)))))
    writer.start()
    time.sleep(0.05)
    playout.clear()
    writer.join(timeout=2)
    assert done == [False]  # the rest of the interrupted chunk is not played
    assert playout.take(4) == bytes(4)
    assert playout.put(b"\x09\x09")  # the next utterance plays normally
    assert playout.take(2) == b"\x09\x09"


def test_vad_turn_start_interrupts_unless_moving():
    moving = {"value": False}
    strategy = MotionAwareVADUserTurnStartStrategy(
        barge_in=True,
        interrupt_while_moving=False,
        moving=lambda: moving["value"],
    )
    started: list[bool] = []

    async def capture(_strategy, params) -> None:
        started.append(params.enable_interruptions)

    strategy.add_event_handler("on_user_turn_started", capture)

    async def run() -> None:
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        moving["value"] = True
        await strategy.process_frame(VADUserStartedSpeakingFrame())

    asyncio.run(run())
    assert started == [True, False]


def test_mic_pipeline_starts_turns_on_vad():
    """The echo guard keeps the robot out of the mic, so plain VAD starts a turn."""
    cfg = load_config()
    cfg.voice.smart_turn = False
    params = _user_params(cfg, audio_in=True, mute=MicMute(), moving=lambda: False)
    assert isinstance(params.vad_analyzer, SileroVADAnalyzer)
    start = params.user_turn_strategies.start
    assert isinstance(start[0], MotionAwareVADUserTurnStartStrategy)
    assert isinstance(start[1], TranscriptionUserTurnStartStrategy)
    assert start[1]._enable_interruptions is False
    stop = params.user_turn_strategies.stop[0]
    assert isinstance(stop, SpeechTimeoutUserTurnStopStrategy)
    assert params.empty_user_turn is None
    assert params.user_turn_stop_timeout == cfg.stt.timeout_s
    assert CompatibleLLMService.supports_developer_role is False


def test_mic_pipeline_optional_smart_turn():
    cfg = load_config()
    cfg.voice.smart_turn = True
    params = _user_params(cfg, audio_in=True, mute=MicMute(), moving=lambda: False)
    stop = params.user_turn_strategies.stop[0]
    assert isinstance(stop, TurnAnalyzerUserTurnStopStrategy)
    assert isinstance(stop._turn_analyzer, LocalSmartTurnAnalyzerV3)


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


def test_portable_messages_rewrites_mid_conversation_system_for_claude():
    messages = [
        {"role": "system", "content": "You are Kogane"},
        {"role": "user", "content": "Hallo"},
        {"role": "developer", "content": '{"ok": true}'},
        {"role": "system", "content": "updated situation"},
    ]
    out = portable_messages(messages)
    assert out[0]["role"] == "system"
    assert out[2]["role"] == "user"
    assert out[2]["content"].startswith("[Background update, not said by the user]")
    assert out[3]["role"] == "user"
    assert "updated situation" in out[3]["content"]


class _BytesTTS(TTSService):
    def __init__(self, pcm: bytes, *, chunk_size: int | None = None) -> None:
        super().__init__(
            settings=TTSSettings(model="bytes", voice=None, language=None),
            sample_rate=16000,
            push_start_frame=True,
            push_stop_frames=True,
        )
        self._pcm = pcm
        self._chunk_size = chunk_size

    async def run_tts(self, text: str, context_id: str):
        del text
        if not self._chunk_size:
            yield TTSAudioRawFrame(self._pcm, 16000, 1, context_id=context_id)
            return
        for offset in range(0, len(self._pcm), self._chunk_size):
            piece = self._pcm[offset : offset + self._chunk_size]
            yield TTSAudioRawFrame(piece, 16000, 1, context_id=context_id)


def _voice_session(tmp_path, transport, tts_service, spoken: list[bool]):
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

    async def speak_visual(on: bool) -> None:
        spoken.append(on)

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
        tts_service=tts_service,
    )
    return session, memory


def test_fake_transport_plays_tts_and_marks_speaking(tmp_path):
    transport = FakeAudioTransport()
    spoken: list[bool] = []
    pcm = b"\x00\x10" * 8000  # 0.5 s at 16 kHz
    session, memory = _voice_session(tmp_path, transport, _BytesTTS(pcm), spoken)

    async def run() -> None:
        await session.start()
        assert session._worker is not None
        await session._worker.queue_frame(speak_frame("Hallo"))
        for _ in range(50):
            if transport.played and any(spoken):
                break
            await asyncio.sleep(0.1)
        await session.stop()

    asyncio.run(run())
    assert b"".join(transport.played)
    assert True in spoken
    memory.close()


def test_multi_sentence_tts_speak_frame_plays_whole_pcm(tmp_path):
    """Apology TTSSpeakFrame still reaches the transport as the full PCM."""
    transport = FakeAudioTransport()
    spoken: list[bool] = []
    pcm = b"\x01\x02" * 16000
    session, memory = _voice_session(tmp_path, transport, _BytesTTS(pcm, chunk_size=640), spoken)

    async def run() -> None:
        await session.start()
        assert session._worker is not None
        await session._worker.queue_frame(
            speak_frame("Hallo! Ich bin bereit. Wie kann ich dir helfen?")
        )
        for _ in range(100):
            played = b"".join(transport.played)
            if len(played) >= len(pcm):
                break
            await asyncio.sleep(0.05)
        await session.stop()

    asyncio.run(run())
    played = b"".join(transport.played)
    assert played == pcm
    assert True in spoken and False in spoken
    memory.close()


def _bare_session(*, ui: list | None = None) -> VoiceSession:
    session = VoiceSession.__new__(VoiceSession)
    session._cfg = SimpleNamespace(tts=SimpleNamespace(enabled=True, provider="piper-http"))
    session._tools = 0
    session._turn = 0
    session._bot_speaking = False
    session._user_speaking = False
    session._llm_done = False
    session._followup_pending = False
    session._user_msgs = 0
    session._nudged_at = -1
    session.last_turn_t = 0.0
    session._mute = MicMute()
    session._idle = asyncio.Event()
    session._idle.clear()
    session._timeout_task = None
    session._input = ""
    session._reply = ""
    session._action_types = []
    session._ttfa_t0 = None
    session._ttfa_ms = None
    session._llm = None
    session._memory = SimpleNamespace(log_event=lambda *_a, **_k: None)
    session._log = SimpleNamespace(
        info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
        exception=lambda *_a, **_k: None,
    )
    session._note_tool_line = lambda _action: None
    session._on_heard = None
    session.last_summary = ""
    updates = ui if ui is not None else []

    def on_ui(**fields):
        updates.append(fields)

    session._on_ui = on_ui
    return session


def test_turn_goes_idle_without_bot_stopped():
    """Playback is not a lock — LLM end + tools exited is enough."""
    session = _bare_session()
    session._reply = "Hallo"
    session.tool_enter()
    session.tool_exit()
    assert not session.idle
    session.note_llm_finished()
    assert session.idle


def test_emote_alone_does_not_finish_before_llm_end_frame():
    session = _bare_session()
    session._reply = "Hi"
    session.tool_enter()
    session.tool_exit()
    assert not session.idle
    session.note_llm_finished()
    assert session.idle


def test_turn_without_words_stays_silent():
    """No canned line is spoken in the model's place."""
    session = _bare_session()
    session._input = "Hallo, hörst du mich?"
    pushed: list = []

    class _LLM:
        async def push_frame(self, frame) -> None:
            pushed.append(frame)

    session._llm = _LLM()  # type: ignore[assignment]

    async def run() -> None:
        session.note_llm_finished()
        await asyncio.sleep(0.02)

    asyncio.run(run())
    assert session.idle
    assert session._reply == ""
    assert pushed == []


def test_streamed_reply_keeps_its_spaces():
    session = _bare_session()
    for piece in ("Einverstanden,", " ich", " drehe mich."):
        session.append_reply_text(piece)
    assert session._reply == "Einverstanden, ich drehe mich."


def test_interruption_ends_the_turn_at_once():
    """Cut off before the first word: the turn ends and the gate opens."""
    session = _bare_session()
    session._input = "Lauf mal los"

    async def run() -> None:
        session.mark_busy()
        turn = session.tool_enter()  # a body tool is still running
        await run_test(
            LlmTurnProcessor(session),
            frames_to_send=[InterruptionFrame()],
            expected_down_frames=[InterruptionFrame],
        )
        assert session.idle
        assert session._reply == ""

        # The old turn's tool finishing does not close the next turn.
        session.mark_busy()
        session.tool_enter()
        session.tool_exit(turn)
        assert session._tools == 1
        assert not session.idle
        session.finish_turn()

    asyncio.run(run())


def test_turn_gate_drops_waiting_contexts_on_interruption():
    session = _bare_session()
    gate = TurnGate(session)
    waiting: list[int] = []

    async def run() -> None:
        # The pump is not running, so both contexts wait in the gate.
        await run_test(
            gate,
            frames_to_send=[
                LLMContextFrame(LLMContext()),
                LLMContextFrame(LLMContext()),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
            ],
            expected_down_frames=[InterruptionFrame],
        )

    queued = gate._queue.put_nowait

    def counting_put(frame) -> None:
        queued(frame)
        waiting.append(gate._queue.qsize())

    gate._queue.put_nowait = counting_put  # type: ignore[method-assign]
    asyncio.run(run())
    assert waiting == [1, 2]
    assert gate._queue.empty()


def test_voice_ui_status_tracks_thinking_speaking_idle():
    ui: list[dict] = []
    session = _bare_session(ui=ui)

    async def run() -> None:
        session.mark_busy()
        assert ui[-1] == {"status": "thinking"}
        session._reply = "Hi"
        session.note_llm_finished()
        assert session.idle
        assert ui[-1]["status"] == "idle"
        session.note_bot_started()
        assert ui[-1] == {"status": "speaking"}
        session.note_bot_stopped()
        # turn already idle; speaking frames are visual only
        assert session.idle

    asyncio.run(run())


def test_wav_pcm_keeps_full_payload():
    import io
    import wave

    buf = io.BytesIO()
    pcm = b"\x10\x20" * 5000
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(pcm)
    out, rate = wav_pcm(buf.getvalue())
    assert rate == 22050
    assert out == pcm


def test_http_client_uses_openai_timeout_not_httpx():
    """httpx.Timeout nested under openai 3.x makes STT die with float+Timeout."""
    import httpx
    from openai import AsyncOpenAI
    from openai import Timeout as OpenAITimeout

    client = _http_client(30.0)
    assert isinstance(client.timeout, OpenAITimeout)
    assert not isinstance(client.timeout, httpx.Timeout)
    api = AsyncOpenAI(api_key="x", base_url="http://127.0.0.1:9/v1", http_client=client)
    assert isinstance(api.timeout, OpenAITimeout)


async def _async_none():
    return None


def test_turn_waits_for_the_followup_a_tool_asked_for():
    """A tool result with run_llm=True runs the LLM past the gate: the turn stays open."""
    session = _bare_session()
    session.expect_followup()
    session.note_llm_finished()  # first response (tools only) ended
    assert not session.idle
    session.note_llm_started()  # the follow-up arrives
    session.append_reply_text("Hallo du!")
    session.note_llm_finished()
    assert session.idle and session._reply == "Hallo du!"
