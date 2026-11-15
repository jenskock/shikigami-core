"""Small WAV helpers for the STT path (normalize / trim / stats)."""

from __future__ import annotations

import io
import wave
from typing import Any


def _require_np():
    import numpy as np

    return np


def wav_stats(wav_bytes: bytes) -> dict[str, Any]:
    """Return duration_s / rms / peak for a mono-or-stereo 16-bit WAV."""
    np = _require_np()
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if pcm.size == 0:
        return {"duration_s": 0.0, "rms": 0.0, "peak": 0.0, "sr": sr}
    return {
        "duration_s": float(pcm.size / sr),
        "rms": float(np.sqrt(np.mean(pcm**2))),
        "peak": float(np.max(np.abs(pcm))),
        "sr": sr,
    }


def prepare_wav_for_stt(
    wav_bytes: bytes,
    *,
    target_peak: float = 12000.0,
    trim_rms: float = 25.0,
    pad_s: float = 0.15,
) -> bytes:
    """Trim near-silence and peak-normalize so soft MacBook captures reach STT.

    Real speech at peak ~200 is often returned as '' by cloud STT; the same clip
    attenuated from a loud TTS source still works — AGC closes that gap.
    """
    np = _require_np()
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sw != 2 or pcm.size == 0:
        return wav_bytes
    pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16) if ch > 1 else pcm.ravel()

    # per-100ms RMS mask → keep from first to last "active" window, with pad
    win = max(1, sr // 10)
    active = []
    for i in range(0, pcm.size, win):
        chunk = pcm[i : i + win].astype(np.float32)
        if chunk.size and float(np.sqrt(np.mean(chunk**2))) >= trim_rms:
            active.append(i)
    if active:
        pad = int(pad_s * sr)
        start = max(0, active[0] - pad)
        end = min(pcm.size, active[-1] + win + pad)
        pcm = pcm[start:end]

    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 1.0 and peak < target_peak:
        gain = target_peak / peak
        pcm = np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
