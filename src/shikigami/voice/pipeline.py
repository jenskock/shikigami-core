"""Pipecat voice session: mic or text in, body tools and speech out.

Speech is the LLM text stream. Body tools (move, emote, remember, idle) run
beside it. The robot's HAL, eyes, reflexes, and dashboard stay in ``App``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

import aiohttp
from openai import BadRequestError, DefaultAsyncHttpxClient
from openai import Timeout as OpenAITimeout
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start.transcription_user_turn_start_strategy import (
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.runner import WorkerRunner

from ..brain.actions import Action, SayAction
from ..brain.agent import Brain
from ..config import Config
from ..memory.store import MemoryStore
from .barge_in import MotionAwareVADUserTurnStartStrategy
from .mute import ManualMuteStrategy, MicMute
from .openai_compat_tts import FreeTextVoiceOpenAITTSService
from .piper_http import PiperHttpTTSService
from .tools import ToolHooks, build_tool_schemas
from .transport import FakeAudioTransport, SounddeviceTransport

_LANG_ALIASES = {
    "german": "de",
    "deutsch": "de",
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "japanese": "ja",
}


_DIALOGUE_ROLES = frozenset({"user", "assistant", "tool"})


def context_has_dialogue(messages: list) -> bool:
    """True when the completion has a user, assistant, or tool message with content.

    OpenAI-compatible servers (llama.cpp, vLLM, Ollama, some routers) reject a
    request whose only messages are system or developer, and Pipecat then
    marks the LLM service unusable for the rest of the process.
    """
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in _DIALOGUE_ROLES:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if message.get("tool_calls"):
            return True
    return False


def empty_dialogue_rejection(exception: BaseException) -> bool:
    """The 400 from a completion that had no user, assistant, or tool content."""
    text = str(exception)
    return "non-system message" in text and "content" in text


def stt_language(name: str) -> Language:
    """Map a personality or config language onto a Pipecat ``Language``."""
    key = name.strip().lower().replace("_", "-")
    key = _LANG_ALIASES.get(key, key)
    if not key:
        return Language.EN
    try:
        return Language(key)
    except ValueError:
        short = key.split("-", 1)[0]
        try:
            return Language(short)
        except ValueError:
            return Language.EN


class SituationProcessor(FrameProcessor):
    """Refresh the system message on the context the LLM is about to see.

    ``on_user_turn_stopped`` fires after that context has already been pushed,
    so the refresh has to happen on the frame itself.
    """

    def __init__(self, session: VoiceSession, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self._session.prepare_context(frame.context)
            if not context_has_dialogue(frame.context.messages):
                roles = [
                    message.get("role", "?")
                    for message in frame.context.messages
                    if isinstance(message, dict)
                ]
                self._session.drop_undialogued_turn(roles)
                return
        await self.push_frame(frame, direction)


class TurnGate(FrameProcessor):
    """Run one LLM context at a time so a typed turn cannot overlap a spoken one.

    Idle means LLM done and body tools exited — not that playback finished.
    An interruption drops the contexts still waiting: the utterance that cut
    in sends the same conversation again, with its own words added.
    """

    def __init__(self, session: VoiceSession, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._queue: asyncio.Queue[LLMContextFrame] = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._queue.put_nowait(frame)
            return
        if isinstance(frame, InterruptionFrame):
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
        await self.push_frame(frame, direction)

    async def pump(self) -> None:
        while True:
            frame = await self._queue.get()
            try:
                self._session.mark_busy()
                await self.push_frame(frame)
                await self._session.wait_idle()
            finally:
                self._queue.task_done()


class SpeakVisualProcessor(FrameProcessor):
    """Drive ``expression.speak_visual`` from the output transport's speaking frames."""

    def __init__(self, session: VoiceSession, expression, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._expression = expression
        self._on = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._session.note_bot_started()
            if not self._on:
                self._on = True
                await self._expression.speak_visual(True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._session.note_bot_stopped()
            if self._on:
                self._on = False
                await self._expression.speak_visual(False)
        await self.push_frame(frame, direction)


class LlmTurnProcessor(FrameProcessor):
    """Follow one LLM response: its text, its end, and the user cutting it off."""

    def __init__(self, session: VoiceSession, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            # The interruption flushes this processor's queue, and with it the
            # LLMFullResponseEndFrame of the response it cancelled.
            self._session.note_interrupted()
        elif direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._session.note_llm_started()
            elif isinstance(frame, LLMTextFrame):
                self._session.append_reply_text(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._session.note_llm_finished()
        await self.push_frame(frame, direction)


def _http_client(timeout_s: float) -> DefaultAsyncHttpxClient:
    """HTTP client for OpenAI-compatible STT/TTS.

    OpenAI 3.x ships its own ``Timeout`` (httpx2). Passing std ``httpx.Timeout``
    nests and blows up mid-request as ``float + Timeout``, which kills STT so
    the mic hears nothing. Use the SDK's Timeout (or a bare float).
    """
    return DefaultAsyncHttpxClient(timeout=OpenAITimeout(timeout_s))


def build_tts(
    cfg: Config,
    *,
    session: aiohttp.ClientSession | None,
) -> TTSService | None:
    """The configured speech service, or None when speech is off."""
    if not cfg.tts.enabled:
        return None
    provider = cfg.tts.provider
    match provider:
        case "none":
            return None
        case "piper":
            raise RuntimeError(
                "tts.provider=piper would import in-process Piper (GPL-3.0). "
                "Use piper-http and start the server with `make setup-piper`."
            )
        case "piper-http":
            if session is None:
                raise RuntimeError("piper-http needs an aiohttp session")
            return PiperHttpTTSService(
                base_url=cfg.tts.base_url,
                aiohttp_session=session,
                timeout_s=cfg.tts.timeout_s,
            )
        case "openai":
            # Free-text voice: stock OpenAI ids, or VoiceDesign descriptions via Requesty.
            return FreeTextVoiceOpenAITTSService(
                api_key=cfg.tts.api_key or "not-needed",
                base_url=cfg.tts.base_url or None,
                http_client=_http_client(cfg.tts.timeout_s),
                settings=FreeTextVoiceOpenAITTSService.Settings(
                    model=cfg.tts.model,
                    voice=cfg.tts.voice,
                ),
            )
        case _:
            raise RuntimeError(f"unknown tts.provider {provider!r}")


def _user_params(
    cfg: Config,
    *,
    audio_in: bool,
    mute: MicMute,
    moving: Callable[[], bool],
) -> LLMUserAggregatorParams:
    # An empty start or stop list is filled with the defaults, and those
    # defaults construct LocalSmartTurnAnalyzerV3. Text-only turns pass a
    # cheap non-empty list so the ONNX model stays unloaded.
    if audio_in:
        vad = SileroVADAnalyzer(
            params=VADParams(
                confidence=cfg.voice.vad_confidence,
                stop_secs=cfg.voice.vad_stop_secs,
            )
        )
        # The transport's echo guard keeps the robot's own voice out of the
        # mic, so whatever the VAD hears is the user. A transcript that
        # arrives after its turn was closed opens a new one, without
        # interrupting anything.
        start = [
            MotionAwareVADUserTurnStartStrategy(
                barge_in=cfg.voice.barge_in,
                interrupt_while_moving=cfg.voice.interrupt_while_moving,
                moving=moving,
            ),
            TranscriptionUserTurnStartStrategy(enable_interruptions=False),
        ]
        if cfg.voice.smart_turn:
            stop = [
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(
                        params=SmartTurnParams(stop_secs=cfg.sim.vad_silence_s)
                    )
                )
            ]
        else:
            # Wait for the transcript before closing the turn.
            stop = [
                SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=cfg.sim.vad_silence_s,
                    wait_for_transcript=True,
                )
            ]
    else:
        vad = None
        start = [TranscriptionUserTurnStartStrategy()]
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=cfg.sim.vad_silence_s)]
    # No transcript: stay quiet. llama.cpp, vLLM, and Ollama treat a
    # developer-role "please repeat" as a system message, reject the request,
    # and that 400 takes the LLM offline. The stop timeout is the STT budget,
    # so a slow Whisper reply still joins this turn.
    return LLMUserAggregatorParams(
        vad_analyzer=vad,
        user_mute_strategies=[ManualMuteStrategy(mute)],
        user_turn_strategies=UserTurnStrategies(start=start, stop=stop),
        user_idle_timeout=0,
        user_turn_stop_timeout=cfg.stt.timeout_s,
        empty_user_turn=None,
    )


_BACKGROUND_PREFIX = "[Background update, not said by the user] "


def portable_messages(messages: list) -> list:
    """Rewrite mid-conversation system/developer messages as user messages.

    Claude (and other non-OpenAI models behind Requesty) reject system-like
    messages after the first turn. Pipecat can deliver tool results as
    ``developer``; keep the first system prompt, rewrite the rest.
    """
    out: list = []
    for index, message in enumerate(messages):
        if (
            index > 0
            and isinstance(message, dict)
            and message.get("role") in {"developer", "system"}
        ):
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            message = {"role": "user", "content": f"{_BACKGROUND_PREFIX}{content or ''}"}
        out.append(message)
    return out


class CompatibleLLMService(OpenAILLMService):
    """Chat completions for servers that are not OpenAI's own API.

    ``developer`` is sent as ``user``. Mid-conversation system/developer
    messages are rewritten for Claude on Requesty. A rejection that means
    "this turn had no spoken text" does not take the service offline.
    Spoken words come from the model text stream (and rare apology frames).
    """

    supports_developer_role = False

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        messages = params.get("messages")
        if isinstance(messages, list):
            params["messages"] = portable_messages(messages)
        return params

    def _classify_error(self, exception: Exception) -> ErrorCategory | None:
        if isinstance(exception, BadRequestError) and empty_dialogue_rejection(exception):
            return ErrorCategory.APPLICATION
        return super()._classify_error(exception)


class VoiceSession(ToolHooks):
    """One Pipecat pipeline for the life of the process."""

    def __init__(
        self,
        cfg: Config,
        brain: Brain,
        memory: MemoryStore,
        *,
        execute: Callable[[Action], Awaitable[None]],
        expression,
        moving: Callable[[], bool],
        on_level: Callable[[float], None] | None,
        state: Callable[[], dict],
        events: Callable[[], list],
        note_tool_line: Callable[[Action], None],
        on_ui: Callable[..., None] | None = None,
        on_heard: Callable[[str], Awaitable[None]] | None = None,
        log: logging.Logger | None = None,
        transport: BaseTransport | None = None,
        tts_service: TTSService | None = None,
    ) -> None:
        self._cfg = cfg
        self._brain = brain
        self._memory = memory
        self._execute_action = execute
        self._expression = expression
        self._moving = moving
        self._on_level = on_level
        self._state = state
        self._events = events
        self._note_tool_line = note_tool_line
        self._on_ui = on_ui
        self._on_heard = on_heard
        self._log = log or logging.getLogger("shikigami.voice")
        self._transport_override = transport
        self._tts_override = tts_service
        self._mute = MicMute()
        self._audio_in = cfg.sim.input == "mic" and cfg.stt.enabled
        self.listening = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._tools = 0
        # Counts turns, so a body tool that outlives its turn cannot close the next.
        self._turn = 0
        self._bot_speaking = False
        # True only after LLMFullResponseEndFrame (tools already ran).
        self._llm_done = True
        self._reply = ""
        self._input = ""
        self._action_types: list[str] = []
        self._ttfa_t0: float | None = None
        self._ttfa_ms: float | None = None
        self._timeout_task: asyncio.Task | None = None
        self._pump_task: asyncio.Task | None = None
        self._runner_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._worker: PipelineWorker | None = None
        self._runner: WorkerRunner | None = None
        self._http: aiohttp.ClientSession | None = None
        self._gate: TurnGate | None = None
        self._llm: CompatibleLLMService | None = None
        self._transport: BaseTransport | None = transport
        self.last_summary = ""

    @property
    def idle(self) -> bool:
        return self._idle.is_set()

    @property
    def tts_enabled(self) -> bool:
        return self._cfg.tts.enabled and self._cfg.tts.provider != "none"

    @property
    def transport(self) -> BaseTransport | None:
        return self._transport

    def set_muted(self, muted: bool) -> None:
        self._mute.set_muted(muted)

    def note_tool(self, action: Action) -> None:
        self._action_types.append(action.type.value)
        self._note_tool_line(action)

    async def execute(self, action: Action) -> None:
        await self._execute_action(action)

    def tool_enter(self) -> int:
        """A body tool started. Returns the turn it belongs to, for ``tool_exit``."""
        self._tools += 1
        return self._turn

    def tool_exit(self, turn: int | None = None) -> None:
        if turn is not None and turn != self._turn:
            return
        if self._tools <= 0:
            return
        self._tools -= 1
        self._check_done()

    def note_bot_started(self) -> None:
        self._bot_speaking = True
        self._note_ttfa()
        self._ui(status="speaking")

    def note_bot_stopped(self) -> None:
        self._bot_speaking = False
        if not self.idle:
            self._ui(status="thinking")

    def on_assistant_text(self, text: str) -> None:
        """Capture the spoken reply from the assistant aggregator."""
        if text.strip():
            self._reply = text.strip()

    def append_reply_text(self, text: str) -> None:
        """Accumulate streamed LLM tokens so a silent tool-only turn is detectable."""
        if text:
            self._reply += text

    def note_llm_started(self) -> None:
        self._llm_done = False
        self._ui(status="thinking")

    def note_llm_finished(self) -> None:
        self._llm_done = True
        self._check_done()

    def note_interrupted(self) -> None:
        """The user cut in. The turn is over now.

        Body tools keep running (they are not cancelled by an interruption),
        but they no longer hold the next turn back.
        """
        if self.idle:
            return
        self._llm_done = True
        self.finish_turn()

    def prepare_context(self, context: LLMContext) -> None:
        messages = context.messages
        content = self._brain.system_content(self._state(), self._events())
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            messages[0]["content"] = content
        else:
            messages.insert(0, {"role": "system", "content": content})
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                text = message.get("content")
                if isinstance(text, str):
                    self._input = text
                break

    def drop_undialogued_turn(self, roles: list[str]) -> None:
        self._log.warning(
            "not sending this turn to the model (no user text; roles=%s)",
            ",".join(roles) or "none",
        )
        self.finish_turn()

    def mark_busy(self) -> None:
        self._turn += 1
        self._tools = 0
        self._bot_speaking = False
        self._llm_done = False
        self._reply = ""
        self._action_types = []
        self._ttfa_t0 = time.perf_counter()
        self._ttfa_ms = None
        self._idle.clear()
        self._ui(status="thinking")
        if self._timeout_task is not None:
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._turn_timeout())

    async def wait_idle(self) -> None:
        await self._idle.wait()

    def finish_turn(self) -> None:
        if self._idle.is_set():
            return
        if self._timeout_task is not None:
            self._timeout_task.cancel()
            self._timeout_task = None
        self._bot_speaking = False
        if self._reply.strip():
            # Log spoken text for the dashboard; speech itself came from LLM TTS.
            self._note_tool_line(SayAction(text=self._reply))
            self._log.info("say: %s", self._reply)
        self._memory.log_event(
            "turn",
            {
                "input": self._input,
                "reply": self._reply,
                "actions": list(self._action_types),
            },
        )
        summary = self._latency_summary()
        self.last_summary = summary
        self._log.info("turn done — %s", summary)
        self._ui(status="idle", last_latency=summary)
        self._idle.set()

    async def start(self) -> None:
        if self._cfg.llm.mock:
            self._log.info("voice: mock LLM, Pipecat pipeline not started")
            return
        try:
            await self._start_pipeline()
        except Exception:
            self.listening = False
            self._log.exception("voice pipeline failed to start; typed text will not reach the LLM")

    async def stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
        if self._runner is not None:
            await self._runner.cancel()
        if self._runner_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
        if self._http is not None:
            await self._http.close()
            self._http = None
        self.finish_turn()

    async def submit_text(self, text: str) -> str:
        """One typed turn. Spoken turns enter through the transport instead."""
        self._input = text
        if self._cfg.llm.mock or self._worker is None:
            if self._worker is None and not self._cfg.llm.mock:
                self._log.error("voice pipeline is not running")
                return ""
            return await self._mock_turn(text)
        # Wait out a spoken turn already in flight, then own the idle flag
        # so this typed turn cannot be released by the previous one finishing.
        await self.wait_idle()
        self.mark_busy()
        await self._worker.queue_frame(
            LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True)
        )
        await self.wait_idle()
        return self._reply

    async def _mock_turn(self, text: str) -> str:
        self.mark_busy()
        result = await self._brain.respond(text, self._events(), self._state())
        for action in result.actions:
            self.note_tool(action)
            await self.execute(action)
        self._reply = result.reply
        self.finish_turn()
        return result.reply

    async def _start_pipeline(self) -> None:
        self._http = aiohttp.ClientSession()
        tts = (
            self._tts_override
            if self._tts_override is not None
            else build_tts(self._cfg, session=self._http)
        )
        audio_out = tts is not None
        transport, output_rate = self._open_transport(audio_out)
        self._transport = transport
        stt = self._build_stt() if self._audio_in else None
        llm = CompatibleLLMService(
            api_key=self._cfg.llm.api_key or "not-needed",
            base_url=self._cfg.llm.base_url,
            settings=CompatibleLLMService.Settings(
                model=self._cfg.llm.model,
                temperature=self._cfg.llm.temperature,
                max_tokens=self._cfg.llm.max_tokens,
            ),
        )
        self._llm = llm

        @llm.event_handler("on_error")
        async def on_llm_error(service, error) -> None:
            if error.category is ErrorCategory.INVALID_REQUEST:
                self._log.warning("voice LLM request rejected, continuing: %s", error.error)
                await service.set_usable(True)
            self.finish_turn()

        # 1.16 still accepts a system message and warns that 2.0 wants
        # system_instruction. The situation processor rewrites this message
        # on every turn, which that setting does not.
        context = LLMContext(
            messages=[
                {
                    "role": "system",
                    "content": self._brain.system_content(self._state(), self._events()),
                }
            ],
            tools=build_tool_schemas(self),
        )
        pair = LLMContextAggregatorPair(
            context,
            user_params=_user_params(
                self._cfg,
                audio_in=self._audio_in,
                mute=self._mute,
                moving=self._moving,
            ),
        )

        @pair.user().event_handler("on_user_turn_started")
        async def _listening(_aggregator, _strategy) -> None:
            if self.idle:
                self._ui(status="listening")

        @pair.user().event_handler("on_user_turn_stopped")
        async def _heard(_aggregator, _strategy, message) -> None:
            text = (message.content or "").strip()
            if text:
                self._log.info("heard: %s", text)
                self._ui(last_heard=text, status="thinking")
                if self._on_heard is not None:
                    await self._on_heard(text)
            else:
                self._log.warning("user turn ended with no transcript")

        @pair.assistant().event_handler("on_assistant_turn_stopped")
        async def _on_assistant(_aggregator, message) -> None:
            self.on_assistant_text(message.content or "")

        situation = SituationProcessor(self)
        gate = TurnGate(self)
        self._gate = gate
        llm_turn = LlmTurnProcessor(self)
        speak = SpeakVisualProcessor(self, self._expression)
        processors: list[FrameProcessor] = [transport.input()]
        if stt is not None:
            processors.append(stt)
        processors.append(pair.user())
        processors.append(gate)
        processors.append(situation)
        processors.append(llm)
        processors.append(llm_turn)
        if tts is not None:
            processors.append(tts)
        processors.append(transport.output())
        processors.append(speak)
        processors.append(pair.assistant())

        worker = PipelineWorker(
            Pipeline(processors),
            params=PipelineParams(
                audio_in_sample_rate=self._cfg.sim.sample_rate,
                audio_out_sample_rate=output_rate,
            ),
            enable_rtvi=False,
            idle_timeout_secs=None,
        )
        self._worker = worker

        @worker.event_handler("on_pipeline_started")
        async def _started(_worker, _frame) -> None:
            self._ready.set()

        @worker.event_handler("on_pipeline_error")
        async def _error(_worker, frame) -> None:
            self._log.warning("pipeline error: %s", getattr(frame, "error", frame))
            self.finish_turn()

        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        self._runner = runner
        await runner.add_workers(worker)
        self._pump_task = asyncio.create_task(gate.pump(), name="shikigami-turn-gate")
        self._runner_task = asyncio.create_task(runner.run(auto_end=True), name="shikigami-voice")
        await asyncio.wait_for(self._ready.wait(), timeout=self._cfg.llm.timeout_s)
        self.listening = self._audio_in
        self._log.info(
            "voice pipeline up (transport=%s, stt=%s, tts=%s, barge_in=%s, echo_guard=%s)",
            type(transport).__name__,
            "on" if stt is not None else "off",
            self._cfg.tts.provider if tts is not None else "off",
            self._cfg.voice.barge_in,
            getattr(transport, "echo_guard", False),
        )
        if self._audio_in and audio_out and not getattr(transport, "echo_guard", False):
            self._log.warning(
                "echo guard is off: the mic stays open while the speaker plays. "
                "Without headphones the robot will hear and answer itself."
            )

    def _open_transport(self, audio_out: bool) -> tuple[BaseTransport, int]:
        if self._transport_override is not None:
            rate = getattr(self._transport_override, "output_rate", self._cfg.sim.sample_rate)
            return self._transport_override, int(rate)
        if self._cfg.voice.transport == "fake":
            fake = FakeAudioTransport(output_rate=self._cfg.sim.sample_rate)
            return fake, fake.output_rate
        try:
            live = SounddeviceTransport(
                mic_device=self._cfg.sim.mic_device,
                speaker_device=self._cfg.sim.speaker_device,
                pipeline_rate=self._cfg.sim.sample_rate,
                gain=self._cfg.sim.mic_gain,
                mute=self._mute,
                on_level=self._on_level,
                audio_in=self._audio_in,
                audio_out=audio_out,
                log=self._log,
                echo_guard=self._cfg.voice.echo_guard,
                echo_tail_secs=self._cfg.voice.echo_tail_secs,
            )
            return live, live.output_rate
        except Exception as exc:
            self._log.warning(
                "sounddevice unavailable (%s) — continuing without a mic or speaker",
                exc,
            )
            fake = FakeAudioTransport(output_rate=self._cfg.sim.sample_rate)
            self._audio_in = False
            return fake, fake.output_rate

    def _build_stt(self) -> OpenAISTTService:
        return OpenAISTTService(
            api_key=self._cfg.stt.api_key or "not-needed",
            base_url=self._cfg.stt.base_url or None,
            http_client=_http_client(self._cfg.stt.timeout_s),
            ttfs_p99_latency=self._cfg.stt.timeout_s,
            settings=OpenAISTTService.Settings(
                model=self._cfg.stt.model,
                language=stt_language(self._cfg.stt.language),
            ),
        )

    def _check_done(self) -> None:
        # Wait for LLMFullResponseEndFrame and every body tool handler exited.
        # Playback is not a lock — Status speaking is visual only.
        if not self._llm_done or self._tools > 0:
            return
        if self._input.strip() and not self._reply.strip() and not self._action_types:
            # Nothing is said in the model's place. Reasoning tokens count
            # against llm.max_tokens, so a cap that is too low leaves no room
            # for the answer itself.
            self._log.warning(
                "the model said nothing for %r — if this repeats, raise llm.max_tokens",
                self._input,
            )
        self.finish_turn()

    def _ui(self, **fields: object) -> None:
        if self._on_ui is not None:
            self._on_ui(**fields)

    def _note_ttfa(self) -> None:
        if self._ttfa_t0 is None or self._ttfa_ms is not None:
            return
        self._ttfa_ms = (time.perf_counter() - self._ttfa_t0) * 1000
        self._log.info("ttfa %.0f ms", self._ttfa_ms)
        self._ttfa_t0 = None

    def _latency_summary(self) -> str:
        if self._ttfa_ms is None:
            return "ttfa=n/a"
        return f"ttfa={self._ttfa_ms:.0f}ms"

    async def _turn_timeout(self) -> None:
        try:
            await asyncio.sleep(self._cfg.llm.timeout_s + self._cfg.tts.timeout_s)
        except asyncio.CancelledError:
            return
        self._log.warning(
            "voice turn timed out after %.0fs",
            self._cfg.llm.timeout_s + self._cfg.tts.timeout_s,
        )
        self.finish_turn()
