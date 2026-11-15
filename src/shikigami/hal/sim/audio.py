"""Sim audio = Mac mic/speaker via sounddevice (lazy import).

record() returns 16-bit mono WAV bytes ready for an STT endpoint.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave

from ..interfaces import Audio


def _require_audio():
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "audio needs the sim extra: `uv sync --extra sim` "
            "(on macOS also `brew install portaudio`)"
        ) from exc
    return sd, np


def _to_wav(pcm, sample_rate: int) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


class SimAudio(Audio):
    def __init__(
        self,
        sample_rate: int = 16000,
        mic_device: int | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._sr = sample_rate
        self._device = mic_device
        self._log = log or logging.getLogger("shikigami.sim.audio")

    async def record(self, max_s: float = 8.0, silence_s: float = 1.2) -> bytes:
        from ...perception.vad import EnergyVAD

        sd, np = _require_audio()
        chunk = int(self._sr * 0.1)  # 100 ms blocks
        vad = EnergyVAD(silence_s=silence_s)
        frames: list = []
        started = False
        t0 = time.monotonic()

        def _rec() -> bytes:
            block = sd.rec(
                chunk, samplerate=self._sr, channels=1, dtype="int16", device=self._device
            )
            sd.wait()
            return block

        while time.monotonic() - t0 < max_s:
            block = await asyncio.to_thread(_rec)
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
            state = vad.feed(rms, dt_s=0.1)
            if state == "speech_start":
                started = True
                self._log.info("[sim] mic: speech started")
            frames.append(block)
            if started and state == "speech_end":
                self._log.info("[sim] mic: speech ended (silence)")
                break

        pcm = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
        return _to_wav(pcm, self._sr)

    async def play(self, data: bytes, fmt: str = "wav") -> None:
        sd, np = _require_audio()
        if fmt != "wav" or not data:
            self._log.warning("[sim] audio: cannot play format %r (%d bytes)", fmt, len(data))
            return
        with wave.open(io.BytesIO(data), "rb") as w:
            sr = w.getframerate()
            channels = w.getnchannels()
            raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels)

        def _play() -> None:
            sd.play(samples, samplerate=sr)
            sd.wait()

        self._log.info("[sim] speaker: playing %d samples @ %d Hz", len(samples), sr)
        await asyncio.to_thread(_play)
