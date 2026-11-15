"""Energy-based voice activity detection on PCM RMS levels. Pure python, unit-testable."""

from __future__ import annotations


class EnergyVAD:
    def __init__(
        self,
        threshold: float = 300.0,
        silence_s: float = 1.0,
        min_speech_s: float = 0.3,
    ) -> None:
        self.threshold = threshold
        self.silence_s = silence_s
        self.min_speech_s = min_speech_s
        self.speaking = False
        self._speech_t = 0.0
        self._silence_t = 0.0

    def reset(self) -> None:
        self.speaking = False
        self._speech_t = 0.0
        self._silence_t = 0.0

    def feed(self, rms: float, dt_s: float = 0.1) -> str:
        """Feed one chunk's RMS level. Returns speech_start|speech|speech_end|silence."""
        if rms >= self.threshold:
            self._silence_t = 0.0
            if not self.speaking:
                self.speaking = True
                self._speech_t = 0.0
                return "speech_start"
            self._speech_t += dt_s
            return "speech"
        if self.speaking:
            self._silence_t += dt_s
            if self._silence_t >= self.silence_s:
                self.speaking = False
                # a too-short burst is treated as noise, not speech
                return "speech_end" if self._speech_t >= self.min_speech_s else "silence"
            return "speech"
        return "silence"
