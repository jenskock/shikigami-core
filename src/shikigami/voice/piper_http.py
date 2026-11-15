"""HTTP client for an out-of-process Piper server.

Pipecat's ``PiperHttpTTSService`` speaks this same protocol, but in 1.16 the
class lives in ``pipecat.services.piper.tts``, which imports the GPL-3.0
``piper`` package at import time and refuses to load without it. Importing it
would pull in-process Piper into this MIT codebase. This service posts JSON to
the server ``scripts/setup-piper.sh`` starts and never imports ``piper``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSStoppedFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

_SYNTH_PATHS = ("/synthesize", "/api/tts")


async def _one_chunk(body: bytes) -> AsyncGenerator[bytes, None]:
    yield body


def candidate_urls(base_url: str) -> list[str]:
    """Synthesis URLs to try, piper >= 1.8 first, then the 1.3 ``/api/tts`` path.

    A base that already ends in one of those paths is used as-is.
    """
    base = base_url.strip().rstrip("/")
    path = urlparse(base).path
    if path in _SYNTH_PATHS:
        return [base]
    return [base + path for path in _SYNTH_PATHS]


class PiperHttpTTSService(TTSService):
    """POST ``{"text": ...}`` to a Piper HTTP server and stream the WAV back.

    The server process (see ``scripts/setup-piper.sh``) owns the voice model.
    This client does not send a voice id: ``tts.voice`` is the OpenAI voice
    name and would not match a Piper model.
    """

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        timeout_s: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(
            settings=TTSSettings(model="piper-http", voice=None, language=None),
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )
        self._candidates = candidate_urls(base_url)
        self._session = aiohttp_session
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._url: str | None = None
        self._lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    async def _resolve_url(self, text: str) -> tuple[str, bytes | None]:
        """Return the working endpoint and, on the probe, the first WAV body."""
        if self._url is not None:
            return self._url, None
        async with self._lock:
            if self._url is not None:
                return self._url, None
            last_error = "no response"
            for url in self._candidates:
                try:
                    async with self._session.post(
                        url,
                        json={"text": text},
                        headers={"Content-Type": "application/json"},
                        timeout=self._timeout,
                    ) as response:
                        if response.status == 404:
                            continue
                        body = await response.read()
                        if response.status != 200:
                            last_error = f"{response.status} {body[:200]!r}"
                            continue
                        self._url = url
                        logger.info("piper-http: using {}", url)
                        return url, body
                except aiohttp.ClientError as exc:
                    last_error = str(exc)
            raise RuntimeError(
                f"no Piper HTTP endpoint at {self._candidates!r} ({last_error}). "
                "Start one with `make setup-piper`."
            )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            url, probed = await self._resolve_url(text)
        except Exception as exc:
            yield ErrorFrame(error=f"piper-http: {exc}")
            return
        if probed is not None:
            async for frame in self._stream_audio_frames_from_iterator(
                _one_chunk(probed), strip_wav_header=True, context_id=context_id
            ):
                yield frame
            return
        try:
            async with self._session.post(
                url,
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            ) as response:
                if response.status != 200:
                    err = await response.text()
                    yield ErrorFrame(error=f"piper-http {response.status}: {err}")
                    yield TTSStoppedFrame(context_id=context_id)
                    return
                chunks = response.content.iter_chunked(self.chunk_size or 4096)
                async for frame in self._stream_audio_frames_from_iterator(
                    chunks, strip_wav_header=True, context_id=context_id
                ):
                    yield frame
        except Exception as exc:
            yield ErrorFrame(error=f"piper-http: {exc}")
