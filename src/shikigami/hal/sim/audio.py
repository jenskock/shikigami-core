"""Sim audio: Mac mic/speaker via sounddevice (lazy import)."""

from __future__ import annotations

import io
import logging
import threading
import time
import wave

from ...concurrency import run_detached
from ..interfaces import Audio

_LEVEL_REF_RMS = 250.0
_PA_SETTLE_S = 0.12


def mic_level(rms: float) -> float:
    """Dashboard meter, 0..1, from an int16 RMS. The voice transport uses the same scale."""
    return min(1.0, max(0.0, rms) / _LEVEL_REF_RMS)


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


def resolve_input_device(sd, index: int | None) -> int:
    """sounddevice input index. ``None`` is the host default."""
    if index is not None:
        return int(index)
    info = sd.query_devices(kind="input")
    return int(info["index"])


def resolve_output_device(sd, index: int | None) -> int:
    """sounddevice output index. ``None`` prefers a MacBook speaker over an external DAC."""
    if index is not None:
        return int(index)
    for i, device in enumerate(sd.query_devices()):
        if device["max_output_channels"] <= 0:
            continue
        name = device["name"].lower()
        if "macbook" in name and ("lautsprecher" in name or "speaker" in name):
            return i
    info = sd.query_devices(kind="output")
    return int(info["index"])


def _resample_int16(pcm, src_sr: int, dst_sr: int, np):
    """Linear resample mono int16 PCM. No-op when rates match."""
    if src_sr == dst_sr:
        return np.asarray(pcm, dtype=np.int16).ravel()
    x = np.asarray(pcm, dtype=np.float32).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.int16)
    n_dst = int(round(x.size * dst_sr / src_sr))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.int16)
    t_src = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    y = np.interp(t_dst, t_src, x)
    return np.clip(y, -32768, 32767).astype(np.int16)


# Piper often hits int16 full-scale; leave headroom so CoreAudio resampling
# / speaker DAC don't clip into crackle.
_PLAY_PEAK_TARGET = 28000.0


def _prepare_play_pcm(samples, channels: int, src_sr: int, dst_sr: int, np):
    """Mono-ize if needed, resample to device rate, soft-limit near-clip peaks."""
    pcm = np.asarray(samples, dtype=np.int16)
    pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16) if channels > 1 else pcm.ravel()
    pcm = _resample_int16(pcm, src_sr, dst_sr, np)
    if pcm.size == 0:
        return pcm
    peak = float(np.max(np.abs(pcm)))
    if peak > _PLAY_PEAK_TARGET:
        pcm = np.clip(pcm.astype(np.float32) * (_PLAY_PEAK_TARGET / peak), -32768, 32767).astype(
            np.int16
        )
    return pcm


class SimAudio(Audio):
    def __init__(
        self,
        sample_rate: int = 16000,
        mic_device: int | None = None,
        speaker_device: int | None = None,
        vad_threshold: float = 40.0,
        vad_silence_s: float = 1.0,
        vad_listen_s: float = 3.0,
        mic_gain: float = 8.0,
        on_level=None,
        log: logging.Logger | None = None,
    ) -> None:
        self._sr = sample_rate
        self._device = mic_device
        self._speaker_device = speaker_device
        self._vad_threshold = vad_threshold
        self._vad_silence_s = vad_silence_s
        self._vad_listen_s = vad_listen_s
        self._mic_gain = max(1.0, float(mic_gain))
        self._on_level = on_level
        self._log = log or logging.getLogger("shikigami.sim.audio")
        self._device_logged = False
        self._speaker_logged = False
        self._pa_lock = threading.Lock()
        self._muted = False

    def set_muted(self, muted: bool) -> None:
        """When True, abort the in-flight record loop ASAP (returns empty WAV)."""
        self._muted = bool(muted)

    @classmethod
    def from_sim_config(cls, sim, log: logging.Logger | None = None, on_level=None) -> SimAudio:
        """Build from Config.sim — keeps sim / pidog / picrawler backends in sync."""
        return cls(
            sample_rate=sim.sample_rate,
            mic_device=sim.mic_device,
            speaker_device=sim.speaker_device,
            vad_threshold=sim.vad_threshold,
            vad_silence_s=sim.vad_silence_s,
            vad_listen_s=sim.vad_listen_s,
            mic_gain=sim.mic_gain,
            on_level=on_level,
            log=log,
        )

    def _resolve_device(self, sd) -> int:
        return resolve_input_device(sd, self._device)

    def _resolve_speaker(self, sd) -> int:
        """Prefer built-in Mac speakers over external DACs (often silent when idle)."""
        return resolve_output_device(sd, self._speaker_device)

    def _log_device_once(self, sd, device: int, capture_sr: int) -> None:
        if self._device_logged:
            return
        self._device_logged = True
        info = sd.query_devices(device)
        self._log.info(
            "[sim] mic: input device #%d %r (native %d Hz → wav %d Hz)",
            device,
            info["name"],
            capture_sr,
            self._sr,
        )

    def _log_speaker_once(self, sd, device: int) -> None:
        if self._speaker_logged:
            return
        self._speaker_logged = True
        info = sd.query_devices(device)
        self._log.info("[sim] speaker: output device #%d %r", device, info["name"])

    async def record(self, max_s: float = 8.0, silence_s: float | None = None) -> bytes:
        """The Pipecat transport owns the microphone. This stays on the HAL so backends compile."""
        del max_s, silence_s
        self._log.info("mic capture moved to the Pipecat transport; record() returns empty")
        return b""

    async def play(self, data: bytes, fmt: str = "wav") -> None:
        sd, np = _require_audio()
        if fmt != "wav" or not data:
            self._log.warning("[sim] audio: cannot play format %r (%d bytes)", fmt, len(data))
            return
        with wave.open(io.BytesIO(data), "rb") as w:
            src_sr = w.getframerate()
            channels = w.getnchannels()
            raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        out = self._resolve_speaker(sd)
        self._log_speaker_once(sd, out)
        info = sd.query_devices(out)
        play_sr = int(info.get("default_samplerate") or src_sr)
        src_peak = int(np.max(np.abs(samples))) if samples.size else 0
        pcm = _prepare_play_pcm(samples, channels, src_sr, play_sr, np)
        peak = int(np.max(np.abs(pcm))) if pcm.size else 0

        def _play() -> None:
            with self._pa_lock:
                sd.play(pcm, samplerate=play_sr, device=out, blocking=True)
                time.sleep(_PA_SETTLE_S)

        self._log.info(
            "[sim] speaker: playing %d samples @ %d Hz (from %d Hz peak %d→%d) on #%d",
            len(pcm),
            play_sr,
            src_sr,
            src_peak,
            peak,
            out,
        )
        await run_detached(_play)
