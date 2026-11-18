"""Attention: when Kogane speaks up on its own because of what the camera saw.

Listens to ``person_arrived`` / ``person_left`` and decides between a spoken line
(the LLM writes it, so it stays in character) and a silent reaction (eyes/body).
Speaking needs every gate to pass:

- proactive talk is on and Kogane is not told to be quiet;
- the voice pipeline can speak up now (idle, nobody talking);
- no conversation turn just happened (``busy_window_s``) — then they are already
  talking, and a greeting would butt in;
- at least ``min_gap_s`` since the last proactive line;
- per person: no greeting within ``greet_cooldown_h`` (stored, survives restarts);
  unknown faces are asked who they are at most every ``unknown_ask_cooldown_min``.

A greeting skipped because we are already talking counts as done: they are here
and interacting, so there is nothing to make up for later.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ..config import AttentionConfig

UNKNOWN = "unknown"


class KV(Protocol):
    def get_value(self, key: str, default: str | None = None) -> str | None: ...
    def set_value(self, key: str, value: str) -> None: ...


class AttentionPolicy:
    def __init__(
        self,
        cfg: AttentionConfig,
        kv: KV,
        *,
        notice: Callable[[str], Awaitable[bool]],  # start a proactive turn; False = not now
        can_speak: Callable[[], bool],  # pipeline idle and nobody talking
        last_turn: Callable[[], float],  # monotonic end of the last conversation turn
        emote: Callable[[str], Awaitable[None]],
        clock: Callable[[], float] = time.time,  # wall clock: cooldowns survive restarts
        mono: Callable[[], float] = time.monotonic,
        log: logging.Logger | None = None,
    ) -> None:
        self._cfg = cfg
        self._kv = kv
        self._notice = notice
        self._can_speak = can_speak
        self._last_turn = last_turn
        self._emote = emote
        self._clock = clock
        self._mono = mono
        self._log = log or logging.getLogger("shikigami.attention")
        self.quiet_until = 0.0  # wall clock
        self._last_line = -1e9  # monotonic time of our last proactive line

    # -- quiet mode -------------------------------------------------------------------

    def set_quiet(self, minutes: float | None) -> None:
        """Be quiet for ``minutes`` (None = default); 0 = talk again."""
        if minutes is None:
            minutes = self._cfg.quiet_default_min
        self.quiet_until = self._clock() + max(0.0, minutes) * 60 if minutes > 0 else 0.0
        self._log.info("proactive speech %s", f"off for {minutes:g} min" if minutes else "on")

    @property
    def quiet(self) -> bool:
        return self._clock() < self.quiet_until

    # -- events -----------------------------------------------------------------------

    async def on_event(self, event: Any) -> None:
        """Bus handler for person_arrived / person_left."""
        name = str((event.payload or {}).get("name") or UNKNOWN)
        if event.kind == "person_arrived":
            if name == UNKNOWN:
                await self._unknown_arrived()
            else:
                await self._known_arrived(name)

    async def _known_arrived(self, name: str) -> None:
        key = f"greeted:{name.lower()}"
        last = float(self._kv.get_value(key) or 0)
        now = self._clock()
        if now - last < self._cfg.greet_cooldown_h * 3600:
            await self._emote("happy")  # seen recently: a happy look, no speech
            return
        why = self._why_not_speak()
        if why == "talking":
            self._kv.set_value(key, str(now))  # already interacting: counts as greeted
            await self._emote("happy")
            return
        if why is None and await self._say(
            f"{name} just came into view. Greet {name} warmly by name, in one short sentence."
        ):
            self._kv.set_value(key, str(now))
            self._log.info("greeting %s", name)
            return
        await self._emote("happy")  # quiet / not a good moment: greet with the eyes

    async def _unknown_arrived(self) -> None:
        await self._emote("curious")
        if not self._cfg.ask_unknown:
            return
        last = float(self._kv.get_value("asked:unknown") or 0)
        now = self._clock()
        if now - last < self._cfg.unknown_ask_cooldown_min * 60:
            return
        if self._why_not_speak() is None and await self._say(
            "Someone you don't recognize came into view. Say hello and ask, once and "
            "briefly, who they are."
        ):
            self._kv.set_value("asked:unknown", str(now))
            self._log.info("asking an unknown face who they are")

    # -- gates ------------------------------------------------------------------------

    def _why_not_speak(self) -> str | None:
        """None = may speak now; else the reason (for logs and the 'talking' case)."""
        if not self._cfg.proactive:
            return "proactive off"
        if self.quiet:
            return "quiet"
        mono = self._mono()
        last_turn = self._last_turn()
        if last_turn and mono - last_turn < self._cfg.busy_window_s:
            return "talking"
        if not self._can_speak():
            # A turn in flight, someone speaking, or the pipeline not up yet. Not
            # "talking": that would mark them greeted without a word at startup.
            return "busy"
        if mono - self._last_line < self._cfg.min_gap_s:
            return "too soon"
        return None

    async def _say(self, note: str) -> bool:
        if not await self._notice(note):
            return False
        self._last_line = self._mono()
        return True
