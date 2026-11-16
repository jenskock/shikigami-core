"""Local audio for the Pipecat pipeline.

``LocalAudioTransport`` needs the ``pipecat-ai[local]`` extra (PyAudio) and
has no mute, mic-gain, or level callback. The dashboard already uses
sounddevice, so this transport does too: device choice, the MacBook-speaker
preference, mute, and the mic meter stay in one place.

``FakeAudioTransport`` plays into a buffer. Tests and machines without
PortAudio use it.

The mic sits next to the speaker, so the robot hears its own voice. The echo
guard closes the mic for as long as the speaker is audible: the output device
callback notes every block that carries sound, and the input callback hands
the pipeline silence until that sound has died away. VAD and STT never see
the robot's own speech, so it cannot interrupt or answer itself.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from ..hal.sim.audio import (
    _resample_int16,
    mic_level,
    resolve_input_device,
    resolve_output_device,
)
from .mute import MicMute

# Samples at or below this are silence padding, not sound that can echo.
_AUDIBLE_PEAK = 32
# Speech waiting between the pipeline and the device. Small, so an
# interruption is heard at once; large enough to ride out a busy event loop.
_PLAYOUT_SECS = 0.12


class SpeakerActivity:
    """Whether the speaker is audible at the mic right now.

    ``mark`` is called from the output device callback for each block that
    carries sound. The speaker counts as audible until ``hold_secs`` after the
    last such block: the configured room tail plus both device latencies.
    """

    def __init__(self, tail_secs: float) -> None:
        self._tail = max(0.0, float(tail_secs))
        self._latency = 0.0
        self._last = float("-inf")

    @property
    def hold_secs(self) -> float:
        return self._tail + self._latency

    def add_latency(self, secs: float) -> None:
        """Count a device's buffering towards how long the echo lingers."""
        self._latency += max(0.0, float(secs))

    def mark(self) -> None:
        self._last = time.monotonic()

    @property
    def audible(self) -> bool:
        return time.monotonic() - self._last < self.hold_secs


class Playout:
    """Bounded PCM queue between the pipeline and the output device callback.

    ``put`` blocks while the queue is full, which paces the pipeline at the
    speed of the device. ``take`` never blocks and pads with silence, so the
    device stream runs continuously and reports exactly what it played.
    """

    def __init__(self, capacity_bytes: int) -> None:
        self._capacity = max(2, int(capacity_bytes))
        self._buf = bytearray()
        self._cond = threading.Condition()
        self._generation = 0
        self._closed = False

    def put(self, audio: bytes) -> bool:
        """Queue ``audio``. False when ``clear`` or ``close`` dropped it."""
        offset = 0
        with self._cond:
            generation = self._generation
            while offset < len(audio):
                while (
                    len(self._buf) >= self._capacity
                    and not self._closed
                    and generation == self._generation
                ):
                    self._cond.wait(0.1)
                if self._closed or generation != self._generation:
                    return False
                room = self._capacity - len(self._buf)
                self._buf += audio[offset : offset + room]
                offset += room
        return True

    def take(self, size: int) -> bytes:
        """The next ``size`` bytes for the device, padded with silence."""
        with self._cond:
            out = bytes(self._buf[:size])
            del self._buf[:size]
            self._cond.notify_all()
        if len(out) < size:
            out += bytes(size - len(out))
        return out

    def clear(self) -> None:
        """Drop queued speech, and the rest of a ``put`` that is still waiting."""
        with self._cond:
            self._buf.clear()
            self._generation += 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._buf.clear()
            self._cond.notify_all()


class _FakeInput(BaseInputTransport):
    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)


class _FakeOutput(BaseOutputTransport):
    def __init__(self, params: TransportParams, played: list[bytes]) -> None:
        super().__init__(params)
        self._played = played

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if frame.audio:
            self._played.append(bytes(frame.audio))
        return True


class FakeAudioTransport(BaseTransport):
    """No devices. Output PCM is appended to ``played``."""

    def __init__(self, *, input_rate: int = 16000, output_rate: int = 16000) -> None:
        super().__init__()
        self.played: list[bytes] = []
        self.output_rate = output_rate
        self._params = TransportParams(
            audio_in_enabled=False,
            audio_in_sample_rate=input_rate,
            audio_out_enabled=True,
            audio_out_sample_rate=output_rate,
            audio_out_end_silence_secs=0,
        )
        self._input: _FakeInput | None = None
        self._output: _FakeOutput | None = None

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = _FakeInput(self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = _FakeOutput(self._params, self.played)
        return self._output


def _require_sounddevice():
    """Import sounddevice on first use. PortAudio is absent on some machines."""
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(
            "sounddevice needs PortAudio (`brew install portaudio` on macOS). "
            "The voice pipeline will keep running without a mic or speaker."
        ) from exc
    return sd


class _SoundInput(BaseInputTransport):
    def __init__(
        self,
        params: TransportParams,
        *,
        device: int,
        native_rate: int,
        gain: float,
        mute: MicMute,
        on_level,
        log: logging.Logger,
        speaker: SpeakerActivity | None = None,
    ) -> None:
        super().__init__(params)
        self._device = device
        self._native_rate = native_rate
        self._gain = gain
        self._mute = mute
        self._on_level = on_level
        self._log = log
        # Set when the echo guard is on: the mic is closed while this is audible.
        self._speaker = speaker
        self._stream: Any = None

    def capture(self, samples: np.ndarray, pipeline_rate: int) -> tuple[np.ndarray, float]:
        """One device block as (pipeline PCM, meter level).

        While the speaker is audible the PCM is silence. Silence, not a gap:
        the VAD has to see an utterance end when the robot starts talking over it.
        """
        if self._gain != 1.0 and samples.size:
            samples = np.clip(samples.astype(np.float32) * self._gain, -32768, 32767).astype(
                np.int16
            )
        pcm = _resample_int16(samples, self._native_rate, pipeline_rate, np)
        if pcm.size == 0:
            return pcm, 0.0
        if self._speaker is not None and self._speaker.audible:
            return np.zeros_like(pcm), 0.0
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        return pcm, mic_level(rms)

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        sd = _require_sounddevice()
        block = max(1, int(self._native_rate * 0.02))
        loop = self.get_event_loop()

        def _callback(indata, _frames, _time, status) -> None:
            if status:
                self._log.debug("mic status: %s", status)
            if self._mute.active:
                if self._on_level is not None:
                    self._on_level(0.0)
                return
            samples = np.asarray(indata, dtype=np.int16).reshape(-1)
            pcm, level = self.capture(samples, self.sample_rate)
            if pcm.size == 0:
                return
            if self._on_level is not None:
                self._on_level(level)
            audio = InputAudioRawFrame(
                audio=pcm.tobytes(),
                sample_rate=self.sample_rate,
                num_channels=1,
            )
            asyncio.run_coroutine_threadsafe(self.push_audio_frame(audio), loop)

        try:
            self._stream = sd.InputStream(
                samplerate=self._native_rate,
                channels=1,
                dtype="int16",
                device=self._device,
                blocksize=block,
                callback=_callback,
            )
            self._stream.start()
            if self._speaker is not None:
                self._speaker.add_latency(float(self._stream.latency))
            self._log.info(
                "mic open: device #%d native %d Hz → pipeline %d Hz (echo guard %s)",
                self._device,
                self._native_rate,
                self.sample_rate,
                "on" if self._speaker is not None else "off",
            )
        except Exception as exc:
            self._log.warning("mic unavailable (%s) — text input still works", exc)
            self._stream = None
        await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()


class _SoundOutput(BaseOutputTransport):
    def __init__(
        self,
        params: TransportParams,
        *,
        device: int,
        log: logging.Logger,
        speaker: SpeakerActivity,
    ) -> None:
        super().__init__(params)
        self._device = device
        self._log = log
        self._speaker = speaker
        self._playout: Playout | None = None
        self._stream: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shikigami-speaker")

    def render(self, size: int) -> bytes:
        """The next ``size`` bytes for the device. Notes whether they carry sound."""
        playout = self._playout
        audio = playout.take(size) if playout is not None else bytes(size)
        samples = np.frombuffer(audio, dtype=np.int16)
        if samples.size and int(np.abs(samples.astype(np.int32)).max()) > _AUDIBLE_PEAK:
            self._speaker.mark()
        return audio

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        sd = _require_sounddevice()
        self._playout = Playout(int(self.sample_rate * _PLAYOUT_SECS) * 2)

        def _callback(outdata, _frames, _time, status) -> None:
            if status:
                self._log.debug("speaker status: %s", status)
            outdata[:] = self.render(len(outdata))

        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self._device,
                callback=_callback,
            )
            self._stream.start()
            self._speaker.add_latency(float(self._stream.latency))
            self._log.info(
                "speaker open: device #%d @ %d Hz (echo hold %.0f ms)",
                self._device,
                self.sample_rate,
                self._speaker.hold_secs * 1000,
            )
        except Exception as exc:
            self._log.warning("speaker unavailable (%s)", exc)
            self._stream = None
        await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()
        if self._playout is not None:
            self._playout.close()
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        self._executor.shutdown(wait=False)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Speech already queued for the device would otherwise play on after
        # the user has cut in.
        if isinstance(frame, InterruptionFrame) and self._playout is not None:
            self._playout.clear()
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._stream is None or self._playout is None or not frame.audio:
            return False
        return await self.get_event_loop().run_in_executor(
            self._executor, self._playout.put, frame.audio
        )


class SounddeviceTransport(BaseTransport):
    """Mic and speaker via sounddevice, with mute, a level callback, and the echo guard.

    ``echo_guard=False`` leaves the mic open while the robot speaks. Only
    headphones, or a speakerphone that cancels its own echo, make that safe.
    """

    def __init__(
        self,
        *,
        mic_device: int | None,
        speaker_device: int | None,
        pipeline_rate: int,
        gain: float,
        mute: MicMute,
        on_level,
        audio_in: bool,
        audio_out: bool,
        log: logging.Logger | None = None,
        echo_guard: bool = True,
        echo_tail_secs: float = 0.3,
    ) -> None:
        super().__init__()
        self._log = log or logging.getLogger("shikigami.voice.audio")
        sd = _require_sounddevice()
        self._in_index = resolve_input_device(sd, mic_device) if audio_in else 0
        self._out_index = resolve_output_device(sd, speaker_device) if audio_out else 0
        in_info = sd.query_devices(self._in_index) if audio_in else {"default_samplerate": 16000}
        out_info = sd.query_devices(self._out_index) if audio_out else {"default_samplerate": 24000}
        self._native_in = int(in_info["default_samplerate"])
        self.output_rate = int(out_info["default_samplerate"])
        self._mute = mute
        self._on_level = on_level
        self._gain = gain
        self._audio_in = audio_in
        self._audio_out = audio_out
        self.speaker = SpeakerActivity(echo_tail_secs)
        self.echo_guard = bool(echo_guard and audio_in and audio_out)
        self._params = TransportParams(
            audio_in_enabled=audio_in,
            audio_in_sample_rate=pipeline_rate,
            audio_out_enabled=audio_out,
            audio_out_sample_rate=self.output_rate,
            audio_out_end_silence_secs=0,
        )
        self._input: _SoundInput | None = None
        self._output: _SoundOutput | None = None

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = _SoundInput(
                self._params,
                device=self._in_index,
                native_rate=self._native_in,
                gain=self._gain,
                mute=self._mute,
                on_level=self._on_level,
                log=self._log,
                speaker=self.speaker if self.echo_guard else None,
            )
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = _SoundOutput(
                self._params,
                device=self._out_index,
                log=self._log,
                speaker=self.speaker,
            )
        return self._output
