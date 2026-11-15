"""LLM agent: events + compact state in, structured actions out (streaming tool calls).

The stream is consumed incrementally: each action is emitted via `on_action` as soon
as its tool call finalizes, and say-text is emitted sentence-by-sentence via
`on_sentence` while the model is still generating — so motion and speech start
before the LLM turn finishes (perceived latency << total LLM time).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import LLMConfig
from ..personality.loader import Personality
from .actions import TOOL_SCHEMAS, Action, SayAction, fake_message, parse_message, parse_tool_call

OnAction = Callable[[Action], Awaitable[None]]
OnSentence = Callable[[str], Awaitable[None]]

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_TEXT_ARG = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)')


@dataclass
class BrainResult:
    actions: list[Action] = field(default_factory=list)
    reply: str = ""
    ttft_ms: float | None = None  # time to first token
    total_ms: float = 0.0
    model: str = ""


def _partial_say_text(args: str) -> str:
    """Best-effort extraction of the `text` field from a partially streamed say tool call."""
    m = _TEXT_ARG.search(args)
    if not m:
        return ""
    try:
        return json.loads(f'"{m.group(1)}"')
    except json.JSONDecodeError:
        return m.group(1)


class Brain:
    def __init__(
        self,
        cfg: LLMConfig,
        personality: Personality,
        memory: Any,
        log: logging.Logger | None = None,
    ) -> None:
        self._cfg = cfg
        self._personality = personality
        self._memory = memory
        self._log = log or logging.getLogger("shikigami.brain")
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._cfg.base_url,
                api_key=self._cfg.api_key or "not-needed",  # Ollama ignores the key
                timeout=self._cfg.timeout_s,
            )
        return self._client

    async def warmup(self) -> None:
        """One tiny completion at startup, so turn #1 doesn't pay the cold-start cost.

        The client is kept alive afterwards (httpx keep-alive), so later turns skip
        TLS/routing setup.
        """
        if self._cfg.mock:
            return
        try:
            client = self._get_client()
            t0 = time.perf_counter()
            await asyncio.to_thread(
                client.chat.completions.create,
                model=self._cfg.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=16,  # requesty rejects max_tokens < 16
            )
            self._log.info("llm warmup ok in %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            self._log.warning("llm warmup failed: %s", exc)

    def build_messages(self, user_text: str, events: list, state: dict) -> list[dict[str, str]]:
        facts = [f"{f.key}: {f.value}" for f in self._memory.facts()]
        mood = state.get("mood", "neutral")
        from .prompts import build_system_prompt, build_user_message

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
        on_sentence: OnSentence | None = None,
    ) -> BrainResult:
        t0 = time.perf_counter()
        if self._cfg.mock:
            return await self._mock_respond(user_text, t0, on_action, on_sentence)
        client = self._get_client()
        messages = self.build_messages(user_text, events, state)
        stream = await asyncio.to_thread(
            client.chat.completions.create,
            model=self._cfg.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            stream=True,
        )
        ttft, actions, content = await self._consume(stream, on_action, on_sentence)
        total_ms = (time.perf_counter() - t0) * 1000
        if not actions and content.strip():
            actions = [SayAction(text=content.strip())]
        reply = next((a.text for a in actions if a.type == "say"), "")
        self._log.info(
            "llm: model=%s ttft=%s total=%.0fms actions=%s",
            self._cfg.model,
            f"{ttft:.0f}ms" if ttft is not None else "n/a",
            total_ms,
            [a.type.value for a in actions],
        )
        return BrainResult(
            actions=actions, reply=reply, ttft_ms=ttft, total_ms=total_ms, model=self._cfg.model
        )

    async def _consume(
        self,
        stream: Any,
        on_action: OnAction | None,
        on_sentence: OnSentence | None,
    ) -> tuple[float | None, list[Action], str]:
        """Consume a streaming completion chunk-by-chunk (one `next` per worker-thread hop,
        so the event loop stays responsive between chunks)."""
        t0 = time.perf_counter()
        ttft: float | None = None
        tools: dict[int, dict[str, str]] = {}
        open_idx: int | None = None
        actions: list[Action] = []
        content_parts: list[str] = []
        sent = 0  # chars of streamed text already emitted as sentences

        async def emit_sentences(text: str) -> None:
            nonlocal sent
            if not on_sentence:
                return
            for m in _SENTENCE_BREAK.finditer(text):
                if m.end() > sent:
                    seg = text[sent : m.end()].strip()
                    if seg:
                        await on_sentence(seg)
                    sent = m.end()

        async def flush_text(text: str) -> None:
            nonlocal sent
            if on_sentence and sent < len(text):
                seg = text[sent:].strip()
                if seg:
                    await on_sentence(seg)
                sent = len(text)

        async def finalize(idx: int) -> None:
            nonlocal open_idx
            v = tools.pop(idx, None)
            open_idx = None
            if not v or not v["name"]:
                return
            try:
                action = parse_tool_call(v["name"], v["arguments"] or "{}")
            except Exception as exc:
                self._log.warning("unparseable tool call %r: %s", v["name"], exc)
                return
            actions.append(action)
            if on_action:
                await on_action(action)
            if action.type == "say":
                await flush_text(action.text)  # emit what streaming didn't cover

        it = iter(stream)
        while True:
            chunk = await asyncio.to_thread(next, it, None)
            if chunk is None:
                break
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                content_parts.append(delta.content)
                await emit_sentences("".join(content_parts))
            for tc in delta.tool_calls or []:
                if open_idx is not None and tc.index != open_idx:
                    await finalize(open_idx)  # a new tool call starts -> previous is complete
                open_idx = tc.index
                slot = tools.setdefault(tc.index, {"name": "", "arguments": ""})
                if tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
                    if slot["name"] == "say":
                        await emit_sentences(_partial_say_text(slot["arguments"]))
            if choice.finish_reason and open_idx is not None:
                await finalize(open_idx)
        if open_idx is not None:
            await finalize(open_idx)
        content = "".join(content_parts)
        await flush_text(content)
        return ttft, actions, content

    async def _mock_respond(
        self,
        user_text: str,
        t0: float,
        on_action: OnAction | None,
        on_sentence: OnSentence | None,
    ) -> BrainResult:
        """No-network canned reply for dev/tests (`llm.mock: true`)."""
        message = fake_message(
            [
                ("emote", {"name": "happy"}),
                ("say", {"text": f"(mock Shikigami) You said: {user_text!r}. Hi!"}),
                ("move", {"motion": "wag", "duration_s": 1.0}),
            ]
        )
        actions = parse_message(message)
        if on_action:
            for a in actions:
                await on_action(a)
        reply = next((a.text for a in actions if a.type == "say"), "")
        if on_sentence and reply:
            await on_sentence(reply)
        return BrainResult(
            actions=actions,
            reply=reply,
            ttft_ms=None,
            total_ms=(time.perf_counter() - t0) * 1000,
            model="mock",
        )
