"""Asyncio main loop: wires HAL + reflexes + perception + brain + memory + voice."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import Any, assert_never

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
from ..config import Config
from ..hal.interfaces import HAL
from ..hal.sim.state import SimState
from ..memory.store import MemoryStore
from ..perception.tracker import FaceTracker
from ..reflexes.loop import ReflexLoop
from ..voice.pipeline import VoiceSession
from .bus import EventBus
from .latency import LatencyTracker
from .stdin_lines import StdinLines

# SoftPicrawler / device gait: ~1 cycle ≈ 0.4 s (matches PicrawlerMotion.steps_for_duration).
_SECS_PER_STEP = 0.4


def _duration_for_move(action: MoveAction) -> float | None:
    """Prefer explicit steps (counted strides) over duration_s."""
    if action.steps is not None and action.steps > 0:
        return float(action.steps) * _SECS_PER_STEP
    return action.duration_s


# One small body gesture per face, from the crawler's named actions. None of
# them walks or turns the robot; each ends standing where it started.
EMOTE_GESTURES: dict[str, tuple[str, ...]] = {
    "happy": ("wave",),
    "love": ("look up", "stand"),
    "curious": ("look left", "look right", "stand"),
    "surprised": ("look up", "stand"),
    "confused": ("look right", "stand"),
    "scared": ("look down", "stand"),
    "sad": ("look down", "stand"),
    "sleepy": ("look down", "stand"),
    "angry": ("push up", "stand"),
    "mischievous": ("look right", "look left", "stand"),
}


class App:
    def __init__(
        self,
        cfg: Config,
        hal: HAL,
        brain: Brain,
        memory: MemoryStore,
        bus: EventBus,
        log: logging.Logger | None = None,
        *,
        use_voice: bool = True,
    ) -> None:
        self.cfg = cfg
        self.hal = hal
        self.brain = brain
        self.memory = memory
        self.bus = bus
        self.log = log or logging.getLogger("shikigami.app")
        # SimHAL and SoftPicrawler both expose `.state` + `.dashboard`.
        self._ui_state: SimState | None = getattr(hal, "state", None)
        self._dashboard = getattr(hal, "dashboard", None)
        self._reflex: ReflexLoop | None = None
        self._reflex_task: asyncio.Task | None = None
        self._turn_lock = asyncio.Lock()  # serialize stdin / dashboard turns
        self._body_lock = asyncio.Lock()  # one body motion at a time: moves and gestures
        self._posture = "stand"  # stand | sit | lie, as last asked for
        self._gestures: set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()
        self.voice: VoiceSession | None = None
        self._use_voice = use_voice

    async def start(self) -> None:
        await self.hal.start()
        if self._dashboard is not None:
            dash = self._dashboard
            dash.on_submit = lambda line: asyncio.create_task(self._submit_text(line))
            dash.on_close = self.request_stop  # quit button / window X
            dash.on_mute_toggle = self.toggle_mic_mute
        if self._use_voice and self.brain is not None:
            self.voice = VoiceSession(
                self.cfg,
                self.brain,
                self.memory,
                execute=self.execute,
                expression=self.hal.expression,
                moving=lambda: self.hal.motion.moving,
                on_level=self._on_mic_level,
                state=self._state_dict,
                events=lambda: self.memory.recent_events(8),
                note_tool_line=self._note_tool,
                on_ui=self._set_sim_state,
                on_heard=self._on_voice_heard,
                log=logging.getLogger("shikigami.voice"),
            )
            await self.voice.start()
            if self.voice.listening:
                self._set_sim_state(status="idle", mic="idle")
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
        listening = self.voice.listening if self.voice is not None else False
        self.log.info(
            "app started (backend=%s, input=%s, voice=%s)",
            self.hal.backend,
            self.cfg.sim.input,
            "listening" if listening else "text",
        )
        if self.brain is not None:
            await self.brain.warmup()

    def request_stop(self) -> None:
        """Ask the input loops to exit (dashboard quit button, window X, q key)."""
        self._stop_event.set()

    def toggle_mic_mute(self) -> None:
        """Dashboard Mute / Ctrl+M — drop mic audio; typed text still works."""
        if self._ui_state is None:
            return
        muted = not self._ui_state.mic_muted
        self._ui_state.mic_muted = muted
        self.hal.audio.set_muted(muted)
        if self.voice is not None:
            self.voice.set_muted(muted)
        self._set_sim_state(mic="muted" if muted else "idle", mic_level=0.0)
        self.log.info("mic %s", "muted" if muted else "unmuted")

    def _on_mic_level(self, level: float) -> None:
        self._set_sim_state(mic_level=level)

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
        if self.voice is not None:
            await self.voice.stop()
        if self._reflex:
            self._reflex.stop()
        if self._reflex_task:
            self._reflex_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reflex_task
        await self.hal.stop()
        self.memory.close()

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
            for key, value in kw.items():
                setattr(self._ui_state, key, value)

    def _note_tool(self, action: Action) -> None:
        """Push every finalized tool call into logs + dashboard tool panel."""
        line = format_tool_call(action)
        self.log.info("tool %s", line)
        if self._ui_state is not None:
            self._ui_state.append_tool(line)

    def _note_tool_marker(self, line: str) -> None:
        if self._ui_state is not None:
            self._ui_state.append_tool(line)

    async def _on_voice_heard(self, text: str) -> None:
        """Mic turn transcript: same dashboard breadcrumbs as a typed turn."""
        heard = text if len(text) <= 48 else text[:47] + "…"
        self._note_tool_marker(f"── turn · {heard} ──")
        await self.bus.publish("heard", {"text": text})

    async def _submit_text(self, text: str) -> None:
        """Dashboard fire-and-forget entry: serialize via handle_text, log failures."""
        try:
            await self.handle_text(text)
        except Exception as exc:
            self.log.exception("dashboard turn failed: %s", exc)

    async def handle_text(self, text: str) -> None:
        """One typed turn. Mic turns enter the Pipecat pipeline on their own."""
        async with self._turn_lock:
            await self._handle_text_locked(text)

    async def _handle_text_locked(self, text: str) -> None:
        self._set_sim_state(last_heard=text, status="thinking")
        heard = text if len(text) <= 48 else text[:47] + "…"
        self._note_tool_marker(f"── turn · {heard} ──")
        await self.bus.publish("heard", {"text": text})
        if self.voice is None:
            self._set_sim_state(status="idle")
            return
        await self.voice.submit_text(text)
        self._set_sim_state(status="idle", last_latency=self.voice.last_summary)

    async def execute(self, action, tracker: LatencyTracker | None = None) -> None:
        tracker = tracker or LatencyTracker(self.log)
        if isinstance(action, MoveAction):
            with tracker.stage("exec:move"):
                await self._exec_move(action)
        elif isinstance(action, SayAction):
            self.log.info("say: %s", action.text)
        elif isinstance(action, EmoteAction):
            with tracker.stage("exec:emote"):
                await self.hal.expression.emote(action.name)
            self._start_gesture(action.name)
        elif isinstance(action, RememberAction):
            with tracker.stage("exec:remember"):
                self.memory.remember(action.key, action.value)
                self.log.info("remembered %s=%s", action.key, action.value)
        elif isinstance(action, IdleAction):
            with tracker.stage("exec:idle"):
                await asyncio.sleep(min(action.seconds, 30.0))
        else:
            assert_never(action)

    def _start_gesture(self, emote: str) -> None:
        """Play the face's body gesture in the background, if the body is free.

        Skipped while the robot moves, sits, or lies, and while another gesture
        runs: a gesture must never fight a move the human asked for.
        """
        steps = EMOTE_GESTURES.get(emote.lower().strip())
        if (
            not steps
            or not self.cfg.reflexes.emote_gestures
            or self._posture != "stand"
            or self.hal.motion.moving
            or self._body_lock.locked()
        ):
            return
        task = asyncio.create_task(self._run_gesture(steps), name=f"gesture-{emote}")
        self._gestures.add(task)
        task.add_done_callback(self._gestures.discard)

    async def _run_gesture(self, steps: tuple[str, ...]) -> None:
        motion = self.hal.motion
        try:
            async with self._body_lock:
                for step in steps:
                    if step == "stand":
                        await motion.stand()
                    else:
                        await motion.play_action(step)
        except Exception as exc:
            self.log.warning("gesture %s failed: %s", "/".join(steps), exc)

    async def _exec_move(self, action: MoveAction) -> None:
        # A gesture in flight finishes first; it is short and ends standing.
        async with self._body_lock:
            await self._exec_move_locked(action)
        if action.motion in ("sit", "lie"):
            self._posture = action.motion
        elif action.motion != "look_at":
            self._posture = "stand"

    async def _exec_move_locked(self, action: MoveAction) -> None:
        motion = self.hal.motion
        duration_s = _duration_for_move(action)
        match action.motion:
            case "walk" | "forward" | "backward":
                direction = (
                    "backward"
                    if action.motion == "backward" or action.direction in ("back", "backward")
                    else "forward"
                )
                await motion.walk(direction, action.speed, duration_s)
            case "turn" | "turn_left" | "turn_right":
                direction = (
                    "left"
                    if action.motion == "turn_left"
                    else "right"
                    if action.motion == "turn_right"
                    else action.direction
                )
                await motion.turn(direction, action.angle_deg, action.speed)
            case "sit":
                await motion.sit()
            case "stand":
                await motion.stand()
            case "lie":
                await motion.lie()
            case "wag":
                await motion.wag(duration_s)
            case "look_at":
                await motion.look_at(0.0, 0.0)
            case "stop":
                await motion.stop()
            case other:
                await motion.play_action(other)

    async def run(self) -> None:
        listening = self.voice is not None and self.voice.listening
        if self.cfg.sim.input == "mic" and not listening:
            self.log.warning("sim.input=mic but the voice pipeline is not listening — text input")
            print("shikigami — type a line + Enter (Ctrl-D to quit, Quit/Esc in the dashboard).")
        elif listening:
            print(
                "shikigami — listening (Ctrl-C to quit). "
                "Mute with the dashboard button or Ctrl+M. Type in the chat field or here."
            )
        else:
            print("shikigami — type a line + Enter (Ctrl-D to quit, Quit/Esc in the dashboard).")
            print("note: you can also type in the dashboard chat field (Enter submits).")
        await self._run_text()

    async def _run_text(self) -> None:
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
