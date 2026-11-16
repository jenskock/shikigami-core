"""LLM agent: personality and memory in, structured actions out.

Live turns run through the Pipecat pipeline (``voice.pipeline``). This class
still builds the system prompt, warms the OpenAI-compatible client, and
answers ``--mock`` with a canned action list.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from openai import OpenAI

from ..concurrency import run_detached
from ..config import LLMConfig
from ..memory.store import MemoryStore
from ..personality.loader import Personality
from .actions import Action, fake_message, parse_message
from .prompts import build_situation, build_system_prompt, build_user_message

OnAction = Callable[[Action], Awaitable[None]]


@dataclass
class BrainResult:
    actions: list[Action] = field(default_factory=list)
    reply: str = ""
    ttft_ms: float | None = None
    total_ms: float = 0.0
    model: str = ""


class Brain:
    def __init__(
        self,
        cfg: LLMConfig,
        personality: Personality,
        memory: MemoryStore,
        log: logging.Logger | None = None,
    ) -> None:
        self._cfg = cfg
        self._personality = personality
        self._memory = memory
        self._log = log or logging.getLogger("shikigami.brain")
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self._cfg.base_url,
                api_key=self._cfg.api_key or "not-needed",  # Ollama ignores the key
                timeout=self._cfg.timeout_s,
            )
        return self._client

    async def warmup(self) -> None:
        """One tiny completion at startup, so turn #1 doesn't pay the cold-start cost.

        The client is kept alive afterwards (httpx keep-alive), so later turns skip
        TLS/routing setup. The Pipecat service has its own client; this ping still
        warms the router.
        """
        if self._cfg.mock:
            return
        try:
            client = self._get_client()
            t0 = time.perf_counter()
            await run_detached(
                lambda: client.chat.completions.create(
                    model=self._cfg.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=16,  # requesty rejects max_tokens < 16
                )
            )
            self._log.info("llm warmup ok in %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            self._log.warning("llm warmup failed: %s", exc)

    def system_content(self, state: dict, events: list) -> str:
        """System message for a live turn: personality, mood, facts, and the situation."""
        facts = [f"{f.key}: {f.value}" for f in self._memory.facts()]
        mood = str(state.get("mood", "neutral"))
        return (
            build_system_prompt(self._personality, mood, facts)
            + "\n\n"
            + build_situation(events, state)
        )

    def build_messages(self, user_text: str, events: list, state: dict) -> list[dict[str, str]]:
        facts = [f"{f.key}: {f.value}" for f in self._memory.facts()]
        mood = state["mood"]
        return [
            {"role": "system", "content": build_system_prompt(self._personality, mood, facts)},
            {"role": "user", "content": build_user_message(user_text, events, state)},
        ]

    async def respond(
        self,
        user_text: str,
        events: list,
        state: dict,
        on_action: OnAction | None = None,
    ) -> BrainResult:
        """Mock reply. Live speech goes through the Pipecat pipeline, not this method."""
        t0 = time.perf_counter()
        if not self._cfg.mock:
            raise RuntimeError(
                "Brain.respond is only the --mock path. Live turns run in the Pipecat pipeline."
            )
        return await self._mock_respond(user_text, t0, on_action)

    async def _mock_respond(
        self,
        user_text: str,
        t0: float,
        on_action: OnAction | None,
    ) -> BrainResult:
        """No-network canned reply for dev/tests (`llm.mock: true`)."""
        reply = f"(mock Shikigami) You said: {user_text!r}. Hi!"
        message = fake_message(
            [
                ("emote", {"name": "happy"}),
                ("move", {"motion": "wag", "duration_s": 1.0}),
            ],
            content=reply,
        )
        actions = parse_message(message)
        if on_action:
            for action in actions:
                await on_action(action)
        return BrainResult(
            actions=actions,
            reply=reply,
            ttft_ms=None,
            total_ms=(time.perf_counter() - t0) * 1000,
            model="mock",
        )
