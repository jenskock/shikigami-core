"""Adaptive acoustic echo cancellation for the local sounddevice path.

Speaker PCM is the reference; mic PCM is cleaned before VAD/STT. Uses NLMS
(no native WebRTC dependency). Thread-safe for PortAudio callbacks.
"""

from __future__ import annotations

import threading

import numpy as np

_RING_SECS = 0.5
_MU = 0.25
_EPS = 1e-8
# Skip filter updates when near-end looks like talk-over (near ≫ ref).
_DOUBLE_TALK_RATIO = 8.0
_REF_ACTIVE_RMS = 80.0


class ReferenceRing:
    """Thread-safe int16 mono ring of recent speaker samples at pipeline rate."""

    def __init__(self, sample_rate: int, *, capacity_secs: float = _RING_SECS) -> None:
        self._rate = max(1, int(sample_rate))
        self._capacity = max(self._rate // 10, int(self._rate * capacity_secs))
        self._buf = np.zeros(self._capacity, dtype=np.int16)
        self._size = 0
        self._write = 0
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._rate

    def write(self, pcm: np.ndarray | bytes) -> None:
        if isinstance(pcm, (bytes, bytearray)):
            samples = np.frombuffer(pcm, dtype=np.int16).copy()
        else:
            samples = np.asarray(pcm, dtype=np.int16).ravel()
        if samples.size == 0:
            return
        with self._lock:
            n = samples.size
            if n >= self._capacity:
                samples = samples[-self._capacity :]
                n = samples.size
                self._buf[:] = samples
                self._write = 0
                self._size = self._capacity
                return
            first = min(n, self._capacity - self._write)
            self._buf[self._write : self._write + first] = samples[:first]
            second = n - first
            if second:
                self._buf[:second] = samples[first:]
            self._write = (self._write + n) % self._capacity
            self._size = min(self._capacity, self._size + n)

    def read_latest(self, n: int, *, delay_samples: int = 0) -> np.ndarray:
        """Oldest→newest block of ``n`` samples ending ``delay_samples`` before now.

        Pads with zeros on the left when the ring does not yet hold enough history.
        """
        n = max(0, int(n))
        delay = max(0, int(delay_samples))
        out = np.zeros(n, dtype=np.int16)
        if n == 0:
            return out
        with self._lock:
            available = max(0, self._size - delay)
            take = min(n, available)
            if take == 0:
                return out
            # First index of the take-window in the ring (oldest of the block).
            start = (self._write - delay - take) % self._capacity
            start_out = n - take
            first = min(take, self._capacity - start)
            out[start_out : start_out + first] = self._buf[start : start + first]
            second = take - first
            if second:
                out[start_out + first : start_out + take] = self._buf[:second]
        return out

    def clear(self) -> None:
        with self._lock:
            self._buf.fill(0)
            self._size = 0
            self._write = 0


class NlmsEchoCanceller:
    """Normalized LMS echo canceller. Passthrough when disabled or cold."""

    def __init__(
        self,
        sample_rate: int,
        *,
        filter_ms: float = 64.0,
        delay_ms: float = 0.0,
        enabled: bool = True,
        mu: float = _MU,
    ) -> None:
        self._rate = max(1, int(sample_rate))
        taps = max(32, int(self._rate * max(1.0, filter_ms) / 1000.0))
        # Cap for Pi CPU; 64 ms @ 16 kHz ≈ 1024 taps.
        self._taps = min(taps, 1024)
        self._delay_samples = max(0, int(self._rate * max(0.0, delay_ms) / 1000.0))
        self.enabled = bool(enabled)
        self._mu = float(mu)
        self._w = np.zeros(self._taps, dtype=np.float32)
        self._x = np.zeros(self._taps, dtype=np.float32)
        self._ring = ReferenceRing(self._rate)
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def ring(self) -> ReferenceRing:
        return self._ring

    def push_reference(self, pcm: np.ndarray | bytes) -> None:
        """Speaker PCM at the canceller sample rate."""
        if not self.enabled:
            return
        self._ring.write(pcm)

    def process(self, near: np.ndarray) -> np.ndarray:
        """Return cleaned near-end int16. No-op when disabled."""
        samples = np.asarray(near, dtype=np.int16).ravel()
        if not self.enabled or samples.size == 0:
            return samples
        ref = self._ring.read_latest(samples.size, delay_samples=self._delay_samples)
        return self._nlms(samples, ref)

    def _nlms(self, near: np.ndarray, ref: np.ndarray) -> np.ndarray:
        near_f = near.astype(np.float32)
        ref_f = ref.astype(np.float32)
        out = np.empty_like(near_f)
        with self._lock:
            w = self._w
            x = self._x
            mu = self._mu
            for i in range(near_f.size):
                x[1:] = x[:-1]
                x[0] = ref_f[i]
                y = float(np.dot(w, x))
                e = near_f[i] - y
                out[i] = e
                ref_power = float(np.dot(x, x))
                if ref_power < _REF_ACTIVE_RMS * _REF_ACTIVE_RMS:
                    continue
                # Double-talk: near much louder than the reference filter input.
                if abs(near_f[i]) > _DOUBLE_TALK_RATIO * (abs(y) + 1.0):
                    continue
                w += (mu / (ref_power + _EPS)) * e * x
        return np.clip(out, -32768, 32767).astype(np.int16)
