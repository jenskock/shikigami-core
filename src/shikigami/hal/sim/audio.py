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
_CALIBRATE_CHUNKS = 5
_PA_SETTLE_S = 0.12


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
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    else:
        pcm = pcm.ravel()
    pcm = _resample_int16(pcm, src_sr, dst_sr, np)
    if pcm.size == 0:
        return pcm
    peak = float(np.max(np.abs(pcm)))
    if peak > _PLAY_PEAK_TARGET:
        pcm = np.clip(
            pcm.astype(np.float32) * (_PLAY_PEAK_TARGET / peak), -32768, 32767
        ).astype(np.int16)
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
        if self._device is not None:
            return int(self._device)
        info = sd.query_devices(kind="input")
        return int(info["index"])

    def _resolve_speaker(self, sd) -> int:
        """Prefer built-in Mac speakers over external DACs (often silent when idle)."""
        if self._speaker_device is not None:
            return int(self._speaker_device)
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] <= 0:
                continue
            name = d["name"].lower()
            if "macbook" in name and ("lautsprecher" in name or "speaker" in name):
                return i
        info = sd.query_devices(kind="output")
        return int(info["index"])

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

    def _record_sync(self, max_s: float, silence_s: float) -> bytes:
        """Blocking capture: one InputStream for the whole utterance (no per-chunk open)."""
        sd, np = _require_audio()
        with self._pa_lock:
            time.sleep(_PA_SETTLE_S)
            try:
                return self._record_locked(sd, np, max_s, silence_s)
            finally:
                time.sleep(_PA_SETTLE_S)

    def _record_locked(self, sd, np, max_s: float, silence_s: float) -> bytes:
        from ...perception.vad import EnergyVAD

        device = self._resolve_device(sd)
        info = sd.query_devices(device)
        capture_sr = int(info["default_samplerate"])
        self._log_device_once(sd, device, capture_sr)

        chunk = max(1, int(capture_sr * 0.1))  # ~100 ms blocks at native rate
        threshold = self._vad_threshold
        vad = EnergyVAD(threshold=threshold, silence_s=silence_s)
        frames: list = []
        noise_rms: list[float] = []
        started = False
        calibrated = False
        peak = 0.0
        aborted_mute = False
        t0 = time.monotonic()
        listen_s = min(max_s, self._vad_listen_s)

        with sd.InputStream(
            samplerate=capture_sr,
            channels=1,
            dtype="int16",
            device=device,
            blocksize=chunk,
        ) as stream:
            while time.monotonic() - t0 < max_s:
                if self._muted:
                    self._log.info("[sim] mic: muted — aborting capture")
                    aborted_mute = True
                    started = False
                    break
                block, _overflowed = stream.read(chunk)
                if self._mic_gain != 1.0:
                    block = np.clip(
                        block.astype(np.float32) * self._mic_gain, -32768, 32767
                    ).astype(np.int16)
                rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                peak = max(peak, rms)
                if self._on_level is not None:
                    self._on_level(min(1.0, rms / _LEVEL_REF_RMS))
                frames.append(block)

                if not calibrated:
                    noise_rms.append(rms)
                    if len(noise_rms) >= _CALIBRATE_CHUNKS:
                        floor = float(np.percentile(noise_rms, 25))
                        threshold = max(self._vad_threshold, floor * 4.0, floor + 20.0)
                        vad.threshold = threshold
                        calibrated = True
                        self._log.info(
                            "[sim] mic: noise floor %.0f → threshold %.0f", floor, threshold
                        )
                    continue

                state = vad.feed(rms, dt_s=0.1)
                if state == "speech_start":
                    started = True
                    self._log.info("[sim] mic: speech started (threshold %.0f)", threshold)
                elif state == "silence":
                    started = False
                if started and state == "speech_end":
                    self._log.info("[sim] mic: speech ended — peak rms %.0f", peak)
                    break
                if not started and (time.monotonic() - t0) >= listen_s:
                    break

        if not started:
            if not aborted_mute:
                hint = (
                    " — mic looks dead (near-zero): grant Microphone permission to your "
                    "terminal in macOS System Settings > Privacy & Security, or set sim.mic_device"
                    if peak < 5
                    else " — speak louder, or lower sim.vad_threshold"
                )
                self._log.info(
                    "[sim] mic: no speech detected, peak rms %.0f (threshold %.0f)%s",
                    peak,
                    threshold,
                    hint,
                )
            if self._on_level is not None:
                self._on_level(0.0)
            return b""

        pcm = np.concatenate(frames).ravel()
        pcm = _resample_int16(pcm, capture_sr, self._sr, np)
        return _to_wav(pcm, self._sr)

    async def record(self, max_s: float = 8.0, silence_s: float | None = None) -> bytes:
        sil = self._vad_silence_s if silence_s is None else silence_s
        return await run_detached(lambda: self._record_sync(max_s, sil))

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
