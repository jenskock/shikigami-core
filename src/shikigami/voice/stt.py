"""Speech-to-text client: any OpenAI-compatible /audio/transcriptions endpoint
(cloud Whisper API, or a Whisper server on the jesktop inference box)."""

from __future__ import annotations

import logging
from typing import Any

from ..config import REPO_ROOT, STTConfig
from .wavutil import prepare_wav_for_stt, wav_stats

# personality language names -> ISO-639-1 for the transcriptions API
_LANGUAGE_CODES = {
    "german": "de",
    "deutsch": "de",
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "dutch": "nl",
}


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

    def _language_code(self) -> str | None:
        raw = (self._cfg.language or "").strip()
        if not raw:
            return None
        if len(raw) == 2:
            return raw.lower()
        return _LANGUAGE_CODES.get(raw.lower())

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV bytes (AGC + trim); returns text (may be empty)."""
        if not wav_bytes:
            return ""
        before = wav_stats(wav_bytes)
        prepared = prepare_wav_for_stt(wav_bytes)
        after = wav_stats(prepared)
        self._log.info(
            "stt audio: %.1fs peak=%.0f→%.0f (%d bytes)",
            before["duration_s"],
            before["peak"],
            after["peak"],
            len(prepared),
        )

        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "file": ("audio.wav", prepared, "audio/wav"),
        }
        if lang := self._language_code():
            kwargs["language"] = lang
        resp = client.audio.transcriptions.create(**kwargs)
        text = (resp.text or "").strip()
        self._log.info("stt: %d bytes -> %r", len(prepared), text)
        if not text:
            dump = REPO_ROOT / "data" / "last-mic.wav"
            try:
                dump.parent.mkdir(parents=True, exist_ok=True)
                dump.write_bytes(prepared)
                self._log.warning(
                    "stt returned empty — wrote %s (play it to hear what the mic sent)",
                    dump,
                )
            except OSError as exc:
                self._log.warning("stt empty and could not dump wav: %s", exc)
        return text
