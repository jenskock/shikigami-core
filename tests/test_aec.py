"""NLMS acoustic echo cancellation: reference ring + energy drop on delayed echo."""

import numpy as np

from shikigami.voice.aec import NlmsEchoCanceller, ReferenceRing


def test_reference_ring_read_latest_pads_and_delays():
    ring = ReferenceRing(100, capacity_secs=1.0)
    ring.write(np.arange(10, dtype=np.int16))
    latest = ring.read_latest(5)
    assert latest.tolist() == [5, 6, 7, 8, 9]
    delayed = ring.read_latest(5, delay_samples=2)
    assert delayed.tolist() == [3, 4, 5, 6, 7]
    padded = ring.read_latest(20)
    assert padded[:10].tolist() == [0] * 10
    assert padded[10:].tolist() == list(range(10))


def test_aec_passthrough_when_disabled_or_cold():
    aec = NlmsEchoCanceller(16000, enabled=False)
    near = np.arange(100, dtype=np.int16)
    assert np.array_equal(aec.process(near), near)

    cold = NlmsEchoCanceller(16000, enabled=True)
    # No reference yet → subtracts ~0, returns near unchanged in practice.
    out = cold.process(near)
    assert out.dtype == np.int16
    assert out.shape == near.shape
    # With empty reference the filter learns zeros; output ≈ input.
    assert np.mean(np.abs(out.astype(np.float32) - near.astype(np.float32))) < 5.0


def test_nlms_reduces_delayed_echo_energy():
    rate = 16000
    aec = NlmsEchoCanceller(rate, filter_ms=32.0, delay_ms=0.0, enabled=True, mu=0.4)
    rng = np.random.default_rng(0)
    n = rate * 2
    speaker = (rng.standard_normal(n) * 9000).astype(np.float32)
    delay = 48
    echo = np.zeros(n, dtype=np.float32)
    echo[delay:] = 0.45 * speaker[:-delay]
    near = echo + rng.standard_normal(n).astype(np.float32) * 40.0
    near_i = np.clip(near, -32768, 32767).astype(np.int16)
    speaker_i = np.clip(speaker, -32768, 32767).astype(np.int16)

    block = 320
    cleaned: list[np.ndarray] = []
    for offset in range(0, n, block):
        end = min(n, offset + block)
        aec.push_reference(speaker_i[offset:end])
        cleaned.append(aec.process(near_i[offset:end]))
    out = np.concatenate(cleaned)

    half = n // 2
    rms_in = float(np.sqrt(np.mean(near_i[half:].astype(np.float32) ** 2)))
    rms_out = float(np.sqrt(np.mean(out[half:].astype(np.float32) ** 2)))
    assert rms_in > 500
    assert rms_out < rms_in * 0.55


def test_push_reference_accepts_bytes():
    aec = NlmsEchoCanceller(8000, filter_ms=16.0)
    pcm = (np.arange(100, dtype=np.int16) * 10).tobytes()
    aec.push_reference(pcm)
    out = aec.process(np.zeros(100, dtype=np.int16))
    assert out.shape == (100,)
