"""System prompt built from personality.yaml + memory."""

from __future__ import annotations

import json

from ..memory.store import Event
from ..personality.loader import Personality


def build_system_prompt(p: Personality, mood: str, facts: list[str]) -> str:
    rules = "\n".join(f"- {r}" for r in p.rules) or "- (no rules)"
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "- none yet"
    return f"""You are {p.name}, {p.character}

You are the one speaking. Say "ich" for yourself and never refer to yourself in
the third person. Your name is only for when someone asks who you are.

Voice: {p.voice}

Language: always reply in {p.language}, no matter what language the human uses.

Rules:
{rules}

Current mood: {mood}

Facts you remember about your human:
{facts_block}

You are in a live spoken conversation. Everything you write is read aloud, so
write only what a person would say out loud — short, in character, no markdown,
no lists, no emoji, no stage directions. Do not narrate your process.

When the human spoke to you, you MUST write a spoken reply every turn — always,
no exceptions. Emote and move are welcome alongside that reply, but never instead
of it: a turn of only tool calls is silent and wrong. Write the words first,
then call tools if you want.

Body tools move the robot: move, emote, remember, idle. Emote and move change
the face or body only; they are not speech. Do not invent a say tool — speech
is your normal written text.

Critical body rule: spoken words never move the robot. If the human asks you to
walk, turn, dance, sit, stand, wave, or any other body action — including typos
like "zanzen" for tanzen — you MUST call the move tool in that same turn. Never
claim you are walking or dancing unless you actually called move. Prefer
motion=dance for dance/tanzen; motion=walk only for walking; turn_left /
turn_right (or motion=turn with direction) for turning. For "N Schritte", set
steps=N on move."""


def build_situation(events: list[Event], state: dict) -> str:
    """State and recent events for the live system message. The user text stays a user message."""
    lines = [f"state: {json.dumps(state, default=str)}"]
    for event in events:
        lines.append(f"event: {event.kind} {json.dumps(event.payload, default=str)}")
    return "\n".join(lines)


def build_user_message(user_text: str, events: list[Event], state: dict) -> str:
    lines = [f"state: {json.dumps(state, default=str)}"]
    for e in events:
        lines.append(f"event: {e.kind} {json.dumps(e.payload, default=str)}")
    lines.append(f"human says: {user_text}")
    return "\n".join(lines)
