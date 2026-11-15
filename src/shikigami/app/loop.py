"""Asyncio main loop: wires HAL + reflexes + perception + brain + memory + voice."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import Any

from ..brain.actions import EmoteAction, IdleAction, MoveAction, RememberAction, SayAction
from ..brain.agent import Brain
from ..config import Config
from ..hal.interfaces import HAL
from ..hal.sim.backend import SimHAL
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
        self._sim = hal if isinstance(hal, SimHAL) else None
        self._reflex: ReflexLoop | None = None
        self._reflex_task: asyncio.Task | None = None
        self._play_queue: asyncio.Queue | None = None
        self._player_task: asyncio.Task | None = None
        self._tts_lock = asyncio.Lock()
        self._streamed_say = False

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        await self.hal.start()
        if self._sim is not None and self._sim.dashboard is not None:
            self._sim.dashboard.on_submit = lambda line: asyncio.create_task(self.handle_text(line))
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
        self.log.info("app started (backend=%s, input=%s)", self.hal.backend, self.cfg.sim.input)
        await self.brain.warmup()  # keep-alive + cold-start removal for turn #1

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

    # -- speech playback (queued, overlaps with the next LLM turn) -----------
    async def _player(self) -> None:
        while True:
            item = await self._play_queue.get()
            try:
                if item is None:
                    return
                data, fmt = item
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
            data, fmt = await asyncio.to_thread(self.tts.synth, text)
        if data:
            await self._play_queue.put((data, fmt))

    # -- one turn ----------------------------------------------------------
    def _state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "backend": self.hal.backend,
            "mood": self.memory.get_mood(),
            "moving": self.hal.motion.moving,
        }
        if self._sim is not None:
            state["last_motion"] = self._sim.state.last_motion
        return state

    def _set_sim_state(self, **kw: str) -> None:
        if self._sim is not None:
            for k, v in kw.items():
                setattr(self._sim.state, k, v)

    async def handle_text(self, text: str) -> None:
        """One full turn: context -> streaming LLM -> actions -> HAL. Latency per stage.

        move/emote/remember execute as soon as their tool call finalizes (on_action);
        say-text is synthesized sentence-by-sentence while the LLM streams (on_sentence)
        and played back in order from a queue, so TTS overlaps generation.
        """
        tracker = LatencyTracker(self.log)
        self._streamed_say = False
        self._set_sim_state(last_heard=text, status="thinking")
        await self.bus.publish("heard", {"text": text})
        with tracker.stage("context"):
            events = self.memory.recent_events(8)
            state = self._state_dict()

        async def on_action(action) -> None:
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
        self._set_sim_state(status="idle", last_latency=tracker.summary())
        self.log.info("turn done — latency %s", tracker.summary())

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
                data, fmt = await asyncio.to_thread(self.tts.synth, a.text)
        if data:
            await self.hal.expression.speak_visual(True)
            with tracker.stage("play"):
                await self.hal.audio.play(data, fmt)
            await self.hal.expression.speak_visual(False)

    # -- input loop --------------------------------------------------------
    async def run(self) -> None:
        if self.cfg.sim.input == "mic" and self.stt is not None:
            await self._run_mic()
        else:
            await self._run_text()

    async def _run_text(self) -> None:
        print("shikigami sim — type a line + Enter (Ctrl-D to quit).")
        print("note: you can also type directly into the shikigami window (it often has focus).")
        print("you> ", end="", flush=True)
        lines = StdinLines(sys.stdin.fileno())
        try:
            async for line in lines:
                if line.strip():
                    await self.handle_text(line.strip())
                    print("you> ", end="", flush=True)
        finally:
            lines.close()

    async def _run_mic(self) -> None:
        assert self.stt is not None
        print("shikigami sim — listening on the mic (Ctrl-C to quit):")
        while True:
            tracker = LatencyTracker(self.log)
            wav = await self.hal.audio.record()
            with tracker.stage("stt"):
                text = await asyncio.to_thread(self.stt.transcribe, wav)
            if text:
                await self.handle_text(text)
            self.log.info("listen loop — latency %s", tracker.summary())
