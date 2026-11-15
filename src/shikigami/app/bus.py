"""Minimal async event bus between layers."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    ts: float
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = {}

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    async def publish(self, topic: str, payload: dict[str, Any] | None = None) -> Event:
        event = Event(ts=time.time(), kind=topic, payload=payload or {})
        for handler in self._subs.get(topic, []):
            await handler(event)
        for handler in self._subs.get("*", []):
            await handler(event)
        return event
