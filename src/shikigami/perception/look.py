"""look(): one camera snapshot -> a cloud vision model -> the robot's spoken answer.

The voice LLM calls ``look`` when it is asked what it sees. The vision model answers
the question directly in the robot's voice, and the tool handler speaks that answer
itself. There is no second voice-LLM completion to retell it: in tests that second
completion often emitted only an emote and stayed silent, and it cost another round
trip. The answer also goes into the tool result, so the conversation knows what was
seen. Same code on Mac and Pi: the snapshot is the camera's ``main`` frame.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from typing import Any

VISION_PROMPT = """You are the eyes and voice of a small robot. You get one camera \
snapshot and answer the human out loud about it.

Seeing rules:
- Only what is clearly visible. If something is unclear or too small, say so simply \
instead of guessing.
- Do not guess anyone's identity, age, mood, health or private details.
- {names}

Speaking rules:
- {persona}
- Talk to the human directly ("du"); the person talking to you is usually the \
closest face.
- One or two short, simple sentences, the most important thing first. Plain spoken \
words only: no lists, no markdown, no emoji, no stage directions.
"""

NAMES_KNOWN = (
    "Face recognition says: {view}. Use exactly these names for those people (the "
    "largest/closest face first). Never name anyone else. If you can't tell which "
    "person is who, just don't use names."
)
NAMES_NONE = "Never name anyone; you do not know who people are."
DEFAULT_PERSONA = "Speak in English, friendly and simple."

_SENTENCE_END = re.compile(r"(?s)^(.*[.!?…])")


def has_names(view: str) -> bool:
    """True if in_view names someone ("who: Jens", "who: Jens + 1 unknown")."""
    if "who:" not in view:
        return False
    who = view.split("who:", 1)[1]
    return any(not part.strip().endswith("unknown") for part in who.split("+"))


def persona_from(personality: Any) -> str:
    """The robot's voice and language, from ``Personality``, for the vision prompt."""
    if personality is None:
        return DEFAULT_PERSONA
    voice = getattr(personality, "voice", "") or ""
    return (
        f"You are {personality.name}. Always answer in {personality.language}. "
        f'Say "ich" about yourself. Voice: {voice}'
    ).strip()


def build_messages(
    jpeg: bytes, question: str, view: str, persona: str = DEFAULT_PERSONA
) -> list[dict]:
    names = NAMES_KNOWN.format(view=view) if has_names(view) else NAMES_NONE
    system = VISION_PROMPT.format(names=names, persona=persona)
    question = question.strip()
    ask = f"The human asked: {question!r}" if question else "What do you see?"
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ask},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]


def trim_to_sentence(text: str) -> str:
    """A reply cut off by the token limit: keep the complete sentences only."""
    m = _SENTENCE_END.match(text.strip())
    return m.group(1).strip() if m else ""


class Looker:
    """Takes snapshots and asks the vision model. One instance per app."""

    def __init__(
        self,
        camera: Any,
        llm_cfg,
        perception_cfg,
        camera_cfg,
        view: Any = None,
        persona: str = DEFAULT_PERSONA,
        save_to: Path | None = None,
        client: Any = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._camera = camera
        self._llm = llm_cfg
        self._pcfg = perception_cfg
        self._ccfg = camera_cfg
        self._view = view or (lambda: "unknown")  # () -> in_view string
        self._persona = persona
        self._save_to = save_to
        self._client = client
        self._log = log or logging.getLogger("shikigami.look")

    @property
    def model(self) -> str:
        return self._pcfg.vision_model or self._llm.model

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self._llm.base_url,
                api_key=self._llm.api_key or "not-needed",
                timeout=self._pcfg.look_timeout_s,
            )
        return self._client

    async def _ask(self, messages: list[dict]) -> str:
        resp = await self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            # Hidden reasoning counts against this; 400 cut answers off mid-word.
            max_tokens=self._pcfg.look_max_tokens,
            temperature=0.2,
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if getattr(choice, "finish_reason", None) == "length":
            text = trim_to_sentence(text)
        return text

    async def look(self, question: str = "") -> dict[str, Any]:
        """Tool result: {"ok": True, "saw": "..."} or {"ok": False, "error": "..."}.

        ``saw`` is already the spoken answer, in the robot's voice.
        """
        jpeg = await self._camera.snapshot_jpeg(max_side=self._ccfg.snapshot_max_side)
        if not jpeg:
            return {"ok": False, "error": "the camera has no picture right now"}
        view = self._view()
        if self._save_to is not None:
            try:
                self._save_to.parent.mkdir(parents=True, exist_ok=True)
                self._save_to.write_bytes(jpeg)  # last snapshot, for debugging
            except OSError:
                pass
        messages = build_messages(jpeg, question, view, self._persona)
        t0 = time.perf_counter()
        text = ""
        try:
            for _attempt in range(2):  # one retry for an empty / fully cut-off answer
                text = await self._ask(messages)
                if text:
                    break
        except Exception as exc:
            self._log.warning("look failed: %s", exc)
            return {"ok": False, "error": f"looking failed ({type(exc).__name__})"}
        ms = (time.perf_counter() - t0) * 1000
        self._log.info("look (%s, %.0f ms, %d KB): %s", self.model, ms, len(jpeg) // 1024, text)
        if not text:
            return {"ok": False, "error": "the picture could not be described"}
        return {"ok": True, "saw": text}
