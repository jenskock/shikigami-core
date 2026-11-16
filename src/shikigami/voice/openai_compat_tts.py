"""OpenAI-compatible TTS that allows free-text ``voice`` values.

Pipecat's ``OpenAITTSService`` only accepts the stock OpenAI voice enum.
Requesty Deepinfra models such as ``Qwen/Qwen3-TTS-VoiceDesign`` take a
natural-language voice description in the same ``voice`` field, so this
service skips that enum check.

OpenAI-compatible speech PCM is always 24 kHz. Frames must be labeled with
that source rate so Pipecat can resample to the device rate. Using the
pipeline/device rate here (often 48 kHz) plays the audio at 2x — chipmunk.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from loguru import logger
from openai import BadRequestError
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.openai.tts import OPENAI_SAMPLE_RATE, OpenAITTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class FreeTextVoiceOpenAITTSService(OpenAITTSService):
    """Same as OpenAI TTS, but ``voice`` may be any non-empty string."""

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        voice = self._settings.voice
        if not voice:
            yield ErrorFrame(error="OpenAI-compatible TTS voice must be specified")
            return
        try:
            create_params = {
                "input": text,
                "model": self._settings.model,
                "voice": voice,
                "response_format": "pcm",
            }
            if self._settings.instructions:
                create_params["instructions"] = self._settings.instructions
            if self._settings.speed:
                create_params["speed"] = self._settings.speed

            async with self._client.audio.speech.with_streaming_response.create(
                **create_params
            ) as response:
                if response.status_code != 200:
                    error = await response.text()
                    logger.error(
                        f"{self} error getting audio (status: {response.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        error=(
                            f"Error getting audio (status: {response.status_code}, error: {error})"
                        )
                    )
                    return

                await self.start_tts_usage_metrics(text)
                async for chunk in response.iter_bytes(self.chunk_size):
                    if chunk:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            chunk, OPENAI_SAMPLE_RATE, 1, context_id=context_id
                        )
        except BadRequestError as exc:
            yield ErrorFrame(error=f"Unknown error occurred: {exc}")
