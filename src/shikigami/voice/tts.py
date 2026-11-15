"""Text-to-speech behind one interface.

provider=openai:     OpenAI-compatible /audio/speech (cloud or local server).
provider=piper:      local `piper` binary via subprocess (fast, runs on the Pi too).
provider=piper-http: remote Piper HTTP server, e.g. on jesktop:
                     `python3 -m piper.http_server --model de_DE-thorsten-medium.onnx --port 5000`
                     (install `piper-tts[http]`; see scripts/setup-piper.sh).
                     Client auto-detects the endpoint: /api/tts (piper 1.3.x) or
                     /synthesize (piper >= 1.8), POST {"text": ...} -> WAV bytes.
provider=none:       disabled (text only).
Always returns WAV bytes so sounddevice can play them directly.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..config import TTSConfig


class TTSClient:
    def __init__(self, cfg: TTSConfig, log: logging.Logger | None = None) -> None:
        self._cfg = cfg
        self._log = log or logging.getLogger("shikigami.tts")
        self._client: Any | None = None
        self._piper_path: str | None = (
            None  # discovered once: /api/tts (1.3.x) vs /synthesize (>=1.8)
        )
        self._piper_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._cfg.base_url,
                api_key=self._cfg.api_key or "not-needed",
                timeout=self._cfg.timeout_s,
            )
        return self._client

    def synth(self, text: str) -> tuple[bytes, str]:
        """Synthesize text; returns (audio_bytes, format). Empty bytes = disabled."""
        if self._cfg.provider == "none" or not text.strip():
            return b"", "none"
        if self._cfg.provider == "piper":
            return self._synth_piper(text), "wav"
        if self._cfg.provider == "piper-http":
            return self._synth_piper_http(text), "wav"
        return self._synth_openai(text), "wav"

    def _synth_piper_http(self, text: str) -> bytes:
        import httpx

        base = self._cfg.base_url.rstrip("/")
        payload = {"text": text}
        with self._piper_lock:
            if self._piper_path is None:
                # probe once; reuse the successful response so we don't synth twice
                for path in ("/api/tts", "/synthesize"):
                    resp = httpx.post(
                        base + path, json=payload, timeout=self._cfg.timeout_s
                    )
                    if resp.status_code != 404:
                        self._piper_path = path
                        resp.raise_for_status()
                        self._log.info(
                            "tts(piper-http %s): %d chars -> %d bytes",
                            path,
                            len(text),
                            len(resp.content),
                        )
                        return resp.content
                raise RuntimeError(f"no TTS endpoint found on piper server at {base}")
            path = self._piper_path
        resp = httpx.post(base + path, json=payload, timeout=self._cfg.timeout_s)
        resp.raise_for_status()
        self._log.info(
            "tts(piper-http %s): %d chars -> %d bytes",
            path,
            len(text),
            len(resp.content),
        )
        return resp.content

    def _synth_openai(self, text: str) -> bytes:
        client = self._get_client()
        # always request WAV — config is Literal["wav"] and sim playback needs it
        resp = client.audio.speech.create(
            model=self._cfg.model,
            voice=self._cfg.voice,
            input=text,
            response_format="wav",
        )
        data = resp.content
        self._log.info("tts(openai): %d chars -> %d bytes", len(text), len(data))
        return data

    def _synth_piper(self, text: str) -> bytes:
        if not self._cfg.piper_voice:
            raise RuntimeError("tts.provider=piper needs tts.piper_voice (path to .onnx model)")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            subprocess.run(
                ["piper", "--model", self._cfg.piper_voice, "--output_file", str(out)],
                input=text.encode(),
                check=True,
                timeout=self._cfg.timeout_s,
                capture_output=True,
            )
            data = out.read_bytes()
        finally:
            out.unlink(missing_ok=True)
        self._log.info("tts(piper): %d chars -> %d bytes", len(text), len(data))
        return data
