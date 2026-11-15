"""Asyncio main loop: wires HAL + reflexes + perception + brain + memory + voice."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from typing import Any

from ..brain.actions import (
    Action,
    EmoteAction,
    IdleAction,
    MoveAction,
    RememberAction,
    SayAction,
    format_tool_call,
)
from ..brain.agent import Brain
from ..concurrency import run_detached
from ..config import Config
from ..hal.interfaces import HAL
from ..hal.sim.state import SimState
from ..memory.store import MemoryStore
from ..perception.tracker import FaceTracker
from ..reflexes.loop import ReflexLoop
from ..voice.stt import STTClient
from ..voice.tts import TTSClient
from .bus import EventBus
from .latency import LatencyTracker
from .stdin_lines import StdinLines


class App:
    def __init__(
        self,
        cfg: Config,
        hal: HAL,
        brain: Brain,
        memory: MemoryStore,
        bus: EventBus,
        stt: STTClient | None,
        tts: TTSClient | None,
        log: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.hal = hal
        self.brain = brain
        self.memory = memory
        self.bus = bus
        self.stt = stt
        self.tts = tts
        self.log = log or logging.getLogger("shikigami.app")
        # SimHAL and SoftPicrawler both expose `.state` + `.dashboard`.
        self._ui_state: SimState | None = getattr(hal, "state", None)
        self._dashboard = getattr(hal, "dashboard", None)
        self._reflex: ReflexLoop | None = None
        self._reflex_task: asyncio.Task | None = None
        self._play_queue: asyncio.Queue | None = None
        self._player_task: asyncio.Task | None = None
        self._tts_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()  # serialize mic / stdin / dashboard turns
        self._streamed_say = False
        self._stop_event = asyncio.Event()
        # silence-end → first PCM (mic turns only); see _mark_ttfa_start / _note_first_audio
        self._ttfa_t0: float | None = None
        self._ttfa_ms: float | None = None

    async def start(self) -> None:
        await self.hal.start()
        if self._dashboard is not None:
            dash = self._dashboard
            dash.on_submit = lambda line: asyncio.create_task(self._submit_text(line))
            dash.on_close = self.request_stop  # quit button / window X
            dash.on_mute_toggle = self.toggle_mic_mute
        self._play_queue = asyncio.Queue()
        self._player_task = asyncio.create_task(self._player())
        tracker = None
        if self.cfg.perception.track_faces:
            tracker = FaceTracker(
                self.hal.camera,
                self.hal.motion,
                self.hal.expression,
                self.bus.publish,
                self.cfg.perception,
            )
        self._reflex = ReflexLoop(self.hal, self.bus, self.cfg.reflexes, tracker=tracker)
        self._reflex_task = asyncio.create_task(self._reflex.run())
        self.log.info(
            "app started (backend=%s, input=%s, stt=%s)",
            self.hal.backend,
            self.cfg.sim.input,
            "on" if self.stt else "off",
        )
        await self.brain.warmup()  # keep-alive + cold-start removal for turn #1

    def request_stop(self) -> None:
        """Ask the input loops to exit (dashboard quit button, window X, q key)."""
        self._stop_event.set()

    def toggle_mic_mute(self) -> None:
        """Dashboard Mute / Ctrl+M — pause capture; text input still works."""
        if self._ui_state is None:
            return
        muted = not self._ui_state.mic_muted
        self._ui_state.mic_muted = muted
        self.hal.audio.set_muted(muted)
        self._set_sim_state(mic="muted" if muted else "idle", mic_level=0.0)
        self.log.info("mic %s", "muted" if muted else "unmuted")

    def _mic_is_muted(self) -> bool:
        return self._ui_state is not None and self._ui_state.mic_muted

    async def _wait_while_muted(self) -> bool:
        """Block until unmuted. Returns False if stop was requested."""
        self._set_sim_state(mic="muted", mic_level=0.0)
        while self._mic_is_muted():
            if self._stop_event.is_set():
                return False
            await asyncio.sleep(0.1)
        return True

    async def _race_stop(self, coro):
        """Await coro, but return None as soon as the stop event fires (cancelling coro)."""
        task = asyncio.ensure_future(coro)
        stop = asyncio.ensure_future(self._stop_event.wait())
        done, _ = await asyncio.wait({task, stop}, return_when=asyncio.FIRST_COMPLETED)
        stop.cancel()
        if task in done:
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return None

    async def stop(self) -> None:
        if self._reflex:
            self._reflex.stop()
        if self._reflex_task:
            self._reflex_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reflex_task
        if self._play_queue is not None:
            self._play_queue.put_nowait(None)
        if self._player_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._player_task
        await self.hal.stop()
        self.memory.close()

    def _mark_ttfa_start(self) -> None:
        """Start TTFA clock at VAD silence-end (before STT)."""
        self._ttfa_t0 = time.perf_counter()
        self._ttfa_ms = None

    def _note_first_audio(self) -> None:
        """Log silence-end → first PCM once per mic turn."""
        if self._ttfa_t0 is None:
            return
        self._ttfa_ms = (time.perf_counter() - self._ttfa_t0) * 1000
        self.log.info("ttfa %.0f ms", self._ttfa_ms)
        self._ttfa_t0 = None

    def _latency_summary(self, tracker: LatencyTracker) -> str:
        summary = tracker.summary()
        if self._ttfa_ms is not None:
            return f"ttfa={self._ttfa_ms:.0f}ms {summary}"
        return summary

    async def _player(self) -> None:
        while True:
            item = await self._play_queue.get()
            try:
                if item is None:
                    return
                data, fmt = item
                self._note_first_audio()
                await self.hal.expression.speak_visual(True)
                try:
                    await self.hal.audio.play(data, fmt)
                except Exception as exc:  # e.g. no audio device — don't kill the queue
                    self.log.warning("audio play failed: %s", exc)
                await self.hal.expression.speak_visual(False)
            finally:
                self._play_queue.task_done()

    async def _synth_and_queue(self, text: str) -> None:
        """TTS one sentence and enqueue it for in-order playback. Serialized by a lock
        so queue order == sentence order, but runs in a worker thread (overlaps LLM)."""
        assert self.tts is not None
        async with self._tts_lock:
            data, fmt = await run_detached(lambda: self.tts.synth(text))
        if data:
            await self._play_queue.put((data, fmt))

    def _state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "backend": self.hal.backend,
            "mood": self.memory.get_mood(),
            "moving": self.hal.motion.moving,
        }
        if self._ui_state is not None:
            state["last_motion"] = self._ui_state.last_motion
        return state

    def _set_sim_state(self, **kw: Any) -> None:
        if self._ui_state is not None:
            for k, v in kw.items():
                setattr(self._ui_state, k, v)

    def _note_tool(self, action: Action) -> None:
        """Push every finalized tool call into logs + dashboard tool panel."""
        line = format_tool_call(action)
        self.log.info("tool %s", line)
        self._note_tool_marker(line)

    def _note_tool_marker(self, line: str) -> None:
        if self._ui_state is not None:
            self._ui_state.append_tool(line)

    async def _submit_text(self, text: str) -> None:
        """Dashboard fire-and-forget entry: serialize via handle_text, log failures."""
        try:
            await self.handle_text(text)
        except Exception as exc:
            self.log.exception("dashboard turn failed: %s", exc)

    async def handle_text(self, text: str) -> None:
        """One full turn: context -> streaming LLM -> actions -> HAL. Latency per stage.

        Serialized by `_turn_lock` so mic / stdin / dashboard cannot overlap turns
        (memory, TTS queue, SimState). move/emote/remember execute as soon as their
        tool call finalizes (on_action); say-text is synthesized sentence-by-sentence
        while the LLM streams (on_sentence) and played back in order from a queue.
        """
        async with self._turn_lock:
            await self._handle_text_locked(text)

    async def _handle_text_locked(self, text: str) -> None:
        tracker = LatencyTracker(self.log)
        self._streamed_say = False
        # text-input turns have no VAD clock; don't reuse a previous mic ttfa
        if self._ttfa_t0 is None:
            self._ttfa_ms = None
        self._set_sim_state(last_heard=text, status="thinking")
        heard = text if len(text) <= 48 else text[:47] + "…"
        self._note_tool_marker(f"── turn · {heard} ──")
        await self.bus.publish("heard", {"text": text})
        with tracker.stage("context"):
            events = self.memory.recent_events(8)
            state = self._state_dict()

        noted = 0

        async def on_action(action) -> None:
            nonlocal noted
            self._note_tool(action)
            noted += 1
            if not isinstance(action, SayAction):  # say is spoken via streamed sentences
                await self.execute(action, tracker)

        pending_tts: list[asyncio.Task] = []

        async def on_sentence(sentence: str) -> None:
            if self.tts is not None:
                # set now, not in the task — avoids a skip race at turn end
                self._streamed_say = True
                pending_tts.append(asyncio.create_task(self._synth_and_queue(sentence)))

        with tracker.stage("llm"):
            result = await self.brain.respond(
                text, events, state, on_action=on_action, on_sentence=on_sentence
            )
        # Plain-content fallback SayActions never went through on_action.
        for action in result.actions[noted:]:
            self._note_tool(action)
        await asyncio.gather(*pending_tts)  # all sentences synthesized+queued before measuring play
        for action in result.actions:
            if isinstance(action, SayAction) and self._streamed_say:
                continue  # already spoken via streamed sentences
            await self.execute(action, tracker)
        self.memory.log_event(
            "turn",
            {
                "input": text,
                "reply": result.reply,
                "actions": [a.type.value for a in result.actions],
            },
        )
        with tracker.stage("play"):
            await self._play_queue.join()  # wait until queued speech finished playing
        summary = self._latency_summary(tracker)
        self._set_sim_state(status="idle", last_latency=summary)
        self.log.info("turn done — latency %s", summary)

    async def execute(self, action, tracker: LatencyTracker) -> None:
        if isinstance(action, MoveAction):
            with tracker.stage("exec:move"):
                await self._exec_move(action)
        elif isinstance(action, SayAction):
            await self._exec_say(action, tracker)
        elif isinstance(action, EmoteAction):
            with tracker.stage("exec:emote"):
                await self.hal.expression.emote(action.name)
        elif isinstance(action, RememberAction):
            with tracker.stage("exec:remember"):
                self.memory.remember(action.key, action.value)
                self.log.info("remembered %s=%s", action.key, action.value)
        elif isinstance(action, IdleAction):
            with tracker.stage("exec:idle"):
                await asyncio.sleep(min(action.seconds, 30.0))

    async def _exec_move(self, a: MoveAction) -> None:
        m = self.hal.motion
        if a.motion == "walk":
            await m.walk(a.direction, a.speed, a.duration_s)
        elif a.motion == "turn":
            await m.turn(a.direction, a.angle_deg, a.speed)
        elif a.motion == "sit":
            await m.sit()
        elif a.motion == "stand":
            await m.stand()
        elif a.motion == "lie":
            await m.lie()
        elif a.motion == "wag":
            await m.wag(a.duration_s)
        elif a.motion == "look_at":
            await m.look_at(0.0, 0.0)
        elif a.motion == "stop":
            await m.stop()
        else:
            await m.play_action(a.motion)

    async def _exec_say(self, a: SayAction, tracker: LatencyTracker) -> None:
        self.log.info("say: %s", a.text)
        data, fmt = b"", "none"
        if self.tts is not None:
            with tracker.stage("tts"):
                data, fmt = await run_detached(lambda: self.tts.synth(a.text))
        if data:
            self._note_first_audio()
            await self.hal.expression.speak_visual(True)
            with tracker.stage("play"):
                await self.hal.audio.play(data, fmt)
            await self.hal.expression.speak_visual(False)

    async def run(self) -> None:
        if self.cfg.sim.input == "mic" and self.stt is None:
            self.log.warning(
                "sim.input=mic but stt is disabled — falling back to text input (use --mic)"
            )
        if self.cfg.sim.input == "mic" and self.stt is not None:
            await self._run_mic()
        else:
            await self._run_text()

    async def _run_text(self) -> None:
        print("shikigami sim — type a line + Enter (Ctrl-D to quit, Quit/Esc in the dashboard).")
        print("note: you can also type in the dashboard chat field (Enter submits).")
        print("you> ", end="", flush=True)
        lines = StdinLines(sys.stdin.fileno())
        try:
            while True:
                try:
                    line = await self._race_stop(lines.__anext__())
                except StopAsyncIteration:  # Ctrl-D / EOF
                    break
                if line is None:  # stop requested (window closed / quit button)
                    break
                if line.strip():
                    await self.handle_text(line.strip())
                    print("you> ", end="", flush=True)
        finally:
            lines.close()

    async def _listen_once(self) -> str:
        """Record one utterance (VAD) and transcribe it; '' on silence. Stages timed."""
        tracker = LatencyTracker(self.log)
        self._set_sim_state(mic="listening…")
        with tracker.stage("record"):
            wav = await self.hal.audio.record()
        if not wav:
            # no speech (or silent mic) — skip the STT round-trip
            self._set_sim_state(mic="idle", mic_level=0.0)
            self.log.info("listen — latency %s (no speech)", tracker.summary())
            return ""
        # VAD just closed the utterance — TTFA runs until first PCM out
        self._mark_ttfa_start()
        self._set_sim_state(mic="transcribing…", mic_level=0.0)
        with tracker.stage("stt"):
            text = await run_detached(lambda: self.stt.transcribe(wav))
        self._set_sim_state(mic="idle", last_heard=text or "(stt empty)")
        self.log.info("listen — latency %s", tracker.summary())
        return text

    async def _run_mic(self) -> None:
        assert self.stt is not None
        print("shikigami sim — listening on the mic (Ctrl-C to quit, Quit/Esc in the dashboard).")
        print("Mute mic via the dashboard button or Ctrl+M; type in the chat field anytime.")
        print("watch Mic + Level in the dashboard side panel.")
        while True:
            if self._mic_is_muted():
                if not await self._race_stop(self._wait_while_muted()):
                    return
                continue
            try:
                text = await self._race_stop(self._listen_once())
            except Exception as exc:
                self.log.warning(
                    "mic/STT failed (%s) — falling back to text input "
                    "(on macOS: `brew install portaudio`, then `uv sync --extra sim`)",
                    exc,
                )
                await self._run_text()
                return
            if text is None:  # stop requested (window closed / quit button)
                return
            if self._mic_is_muted():
                # Mute aborted mid-listen — don't treat as a failed utterance.
                continue
            if text:
                await self.handle_text(text)
            else:
                self.log.info(
                    "heard nothing usable — speak longer/clearer, or check data/last-mic.wav"
                )
