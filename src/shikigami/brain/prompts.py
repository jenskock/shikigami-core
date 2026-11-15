"""System prompt built from personality.yaml + memory."""

from __future__ import annotations

import json

from ..memory.store import Event
from ..personality.loader import Personality


def build_system_prompt(p: Personality, mood: str, facts: list[str]) -> str:
    rules = "\n".join(f"- {r}" for r in p.rules) or "- (no rules)"
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "- none yet"
    return f"""You are {p.name}, {p.character}

Voice: {p.voice}

Rules:
{rules}

Current mood: {mood}

Facts you remember about your human:
{facts_block}

Each turn you receive recent events and your current state, then decide what to
do using the tools: move, say, emote, remember, idle. Use 1-3 actions per turn.
Always include at least one `say` or `emote` so your human can follow along."""


def build_user_message(user_text: str, events: list[Event], state: dict) -> str:
    lines = [f"state: {json.dumps(state, default=str)}"]
    for e in events:
        lines.append(f"event: {e.kind} {json.dumps(e.payload, default=str)}")
    lines.append(f"human says: {user_text}")
    return "\n".join(lines)
