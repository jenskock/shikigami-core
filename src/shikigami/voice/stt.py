"""Speech-to-text client: any OpenAI-compatible /audio/transcriptions endpoint
(cloud Whisper API, or a Whisper server on the jesktop inference box)."""

from __future__ import annotations

import logging
from typing import Any

from ..config import STTConfig


class STTClient:
    def __init__(self, cfg: STTConfig, log: logging.Logger | None = None) -> None:
        self._cfg = cfg
        self._log = log or logging.getLogger("shikigami.stt")
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._cfg.base_url,
                api_key=self._cfg.api_key or "not-needed",
                timeout=self._cfg.timeout_s,
            )
        return self._client

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV bytes; returns text (may be empty)."""
        client = self._get_client()
        resp = client.audio.transcriptions.create(
            model=self._cfg.model,
            file=("audio.wav", wav_bytes, "audio/wav"),
        )
        text = (resp.text or "").strip()
        self._log.info("stt: %d bytes -> %r", len(wav_bytes), text)
        return text
