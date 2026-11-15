"""WAV prepare/normalize helpers for the STT path."""

import io
import wave

import pytest

from shikigami.voice.wavutil import prepare_wav_for_stt, wav_stats

np = pytest.importorskip("numpy")


def _wav(pcm, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


def test_prepare_raises_soft_peak_to_target():
    # 0.5 s silence + 0.5 s soft speech (peak 200) + silence
    sr = 16000
    silence = np.zeros(sr // 2, dtype=np.int16)
    t = np.arange(sr // 2) / sr
    speech = (np.sin(2 * np.pi * 220 * t) * 200).astype(np.int16)
    raw = _wav(np.concatenate([silence, speech, silence]), sr)
    before = wav_stats(raw)
    assert before["peak"] < 250
    out = prepare_wav_for_stt(raw, target_peak=12000)
    after = wav_stats(out)
    assert after["peak"] >= 10000
    assert after["duration_s"] < before["duration_s"]  # trimmed
