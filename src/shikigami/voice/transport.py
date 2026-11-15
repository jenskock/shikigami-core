"""Local audio for the Pipecat pipeline.

``LocalAudioTransport`` needs the ``pipecat-ai[local]`` extra (PyAudio) and
has no mute, mic-gain, or level callback. The dashboard already uses
sounddevice, so this transport does too: device choice, the MacBook-speaker
preference, mute, and the mic meter stay in one place.

``FakeAudioTransport`` plays into a buffer. Tests and machines without
PortAudio use it.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
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
    ) -> None:
        super().__init__(params)
        self._device = device
        self._native_rate = native_rate
        self._gain = gain
        self._mute = mute
        self._on_level = on_level
        self._log = log
        self._stream: Any = None

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        sd = _require_sounddevice()
        block = max(1, int(self._native_rate * 0.02))
        loop = self.get_event_loop()

        def _callback(indata, _frames, _time, status) -> None:
            if status:
                self._log.debug("mic status: %s", status)
            if self._mute.manual:
                if self._on_level is not None:
                    self._on_level(0.0)
                return
            samples = np.asarray(indata, dtype=np.int16).reshape(-1)
            if self._gain != 1.0 and samples.size:
                samples = np.clip(samples.astype(np.float32) * self._gain, -32768, 32767).astype(
                    np.int16
                )
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
            if self._on_level is not None:
                self._on_level(mic_level(rms))
            pcm = _resample_int16(samples, self._native_rate, self.sample_rate, np)
            if pcm.size == 0:
                return
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
            self._log.info(
                "mic open: device #%d native %d Hz → pipeline %d Hz",
                self._device,
                self._native_rate,
                self.sample_rate,
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
    ) -> None:
        super().__init__(params)
        self._device = device
        self._log = log
        self._stream: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shikigami-speaker")

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        sd = _require_sounddevice()
        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self._device,
            )
            self._stream.start()
            self._log.info("speaker open: device #%d @ %d Hz", self._device, self.sample_rate)
        except Exception as exc:
            self._log.warning("speaker unavailable (%s)", exc)
            self._stream = None
        await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        self._executor.shutdown(wait=False)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._stream is None or not frame.audio:
            return False
        await self.get_event_loop().run_in_executor(self._executor, self._stream.write, frame.audio)
        return True


class SounddeviceTransport(BaseTransport):
    """Mic and speaker via sounddevice, with mute and a level callback."""

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
            )
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = _SoundOutput(self._params, device=self._out_index, log=self._log)
        return self._output
