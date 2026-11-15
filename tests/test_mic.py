"""Mic input path: _listen_once with fakes, run() fallback, VAD level reporting."""

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from shikigami.app.bus import EventBus
from shikigami.app.latency import LatencyTracker
from shikigami.app.loop import App
from shikigami.config import load_config
from shikigami.hal.sim import audio as audio_mod
from shikigami.hal.sim.audio import SimAudio, _prepare_play_pcm, _resample_int16
from shikigami.memory.store import MemoryStore

np = pytest.importorskip("numpy")


def _app(tmp_path, stt):
    cfg = load_config()
    cfg.memory.path = str(tmp_path / "m.db")
    hal = SimpleNamespace(audio=SimpleNamespace(record=lambda *a, **k: _wav()))
    return App(
        cfg,
        hal,
        brain=None,
        memory=MemoryStore(cfg.memory.path),
        bus=EventBus(),
        stt=stt,
        tts=None,
        log=logging.getLogger("test"),
    )


async def _wav():
    return b"RIFF...."  # content irrelevant for the fake


async def _empty():
    return b""


def test_listen_once_returns_transcribed_text(tmp_path):
    app = _app(tmp_path, stt=SimpleNamespace(transcribe=lambda wav: "hallo welt"))
    text = asyncio.run(app._listen_once())
    assert text == "hallo welt"
    assert app._ttfa_t0 is not None  # silence-end starts the TTFA clock
    assert app._ttfa_ms is None


def test_listen_once_empty_on_silence(tmp_path):
    app = _app(tmp_path, stt=SimpleNamespace(transcribe=lambda wav: ""))
    assert asyncio.run(app._listen_once()) == ""


def test_listen_once_skips_stt_when_record_returns_empty(tmp_path):
    calls = []
    app = _app(tmp_path, stt=SimpleNamespace(transcribe=lambda wav: calls.append(wav) or "x"))
    app.hal.audio.record = _empty
    assert asyncio.run(app._listen_once()) == ""
    assert calls == []
    assert app._ttfa_t0 is None


def test_note_first_audio_logs_ttfa(tmp_path, caplog):
    app = _app(tmp_path, stt=None)
    app._mark_ttfa_start()
    time.sleep(0.02)
    with caplog.at_level(logging.INFO, logger="test"):
        app._note_first_audio()
    assert app._ttfa_ms is not None and app._ttfa_ms >= 15
    assert app._ttfa_t0 is None
    assert "ttfa" in caplog.text
    prev = app._ttfa_ms
    app._note_first_audio()  # only first audio counts
    assert app._ttfa_ms == prev
    summary = app._latency_summary(LatencyTracker())
    assert summary.startswith(f"ttfa={prev:.0f}ms")


def test_default_vad_silence_is_half_second():
    cfg = load_config()
    assert cfg.sim.vad_silence_s == 0.5


def test_run_falls_back_to_text_when_stt_disabled(tmp_path):
    app = _app(tmp_path, stt=None)
    app.cfg.sim.input = "mic"
    called = []

    async def fake_text():
        called.append(True)

    app._run_text = fake_text
    asyncio.run(app.run())
    assert called == [True]


def test_handle_text_serializes_concurrent_turns(tmp_path):
    """Dashboard + mic must not overlap — one turn at a time."""
    app = _app(tmp_path, stt=None)
    active = 0
    max_active = 0

    async def slow_turn(_text: str) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1

    app._handle_text_locked = slow_turn

    async def main():
        await asyncio.gather(app.handle_text("a"), app.handle_text("b"), app.handle_text("c"))

    asyncio.run(main())
    assert max_active == 1


def test_toggle_mic_mute_updates_state_and_audio(tmp_path):
    app = _app(tmp_path, stt=None)
    muted_flags = []
    app._ui_state = SimpleNamespace(mic_muted=False, mic="", mic_level=1.0)
    app.hal.audio = SimpleNamespace(set_muted=muted_flags.append)

    app.toggle_mic_mute()
    assert app._ui_state.mic_muted is True
    assert app._ui_state.mic == "muted"
    assert app._ui_state.mic_level == 0.0
    assert muted_flags == [True]

    app.toggle_mic_mute()
    assert app._ui_state.mic_muted is False
    assert app._ui_state.mic == "idle"
    assert muted_flags == [True, False]


def test_run_mic_skips_listen_while_muted(tmp_path):
    app = _app(tmp_path, stt=SimpleNamespace(transcribe=lambda wav: "hi"))
    app._ui_state = SimpleNamespace(mic_muted=True, mic="", mic_level=0.0)
    listened = []

    async def fake_listen():
        listened.append(True)
        return "should-not-run"

    async def stop_soon():
        await asyncio.sleep(0.15)
        app.request_stop()

    app._listen_once = fake_listen

    async def main():
        await asyncio.gather(app._run_mic(), stop_soon())

    asyncio.run(main())
    assert listened == []


def test_sim_audio_from_sim_config_copies_vad():
    cfg = load_config()
    cfg.sim.vad_threshold = 123.0
    cfg.sim.vad_silence_s = 0.7
    cfg.sim.vad_listen_s = 2.5
    cfg.sim.sample_rate = 22050
    cfg.sim.mic_device = 4
    cfg.sim.speaker_device = 3
    a = SimAudio.from_sim_config(cfg.sim)
    assert a._vad_threshold == 123.0
    assert a._vad_silence_s == 0.7
    assert a._vad_listen_s == 2.5
    assert a._sr == 22050
    assert a._device == 4
    assert a._speaker_device == 3


def test_resolve_speaker_prefers_macbook(monkeypatch):
    class FakeSD:
        def query_devices(self, idx=None, kind=None):
            # indices match PortAudio list positions
            devs = [
                {"name": "D10s", "max_output_channels": 2},
                {"name": "MacBook Air-Lautsprecher", "max_output_channels": 2},
            ]
            if kind == "output":
                return {**devs[0], "index": 0}
            if idx is None:
                return devs
            return devs[idx]

    a = SimAudio(speaker_device=None)
    assert a._resolve_speaker(FakeSD()) == 1  # MacBook over D10s
    a2 = SimAudio(speaker_device=0)
    assert a2._resolve_speaker(FakeSD()) == 0


def test_resample_int16_halves_rate():
    # 8 samples @ 16 kHz → 4 samples @ 8 kHz
    pcm = np.arange(8, dtype=np.int16)
    out = _resample_int16(pcm, 16000, 8000, np)
    assert out.dtype == np.int16
    assert len(out) == 4


def test_prepare_play_pcm_resamples_and_soft_limits_peak():
    # Piper-like: full-scale tone @ 22050 → device 48000 with headroom
    n = 2205  # 0.1 s
    t = np.arange(n, dtype=np.float32) / 22050.0
    pcm = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    out = _prepare_play_pcm(pcm, channels=1, src_sr=22050, dst_sr=48000, np=np)
    assert out.dtype == np.int16
    assert abs(len(out) - int(round(n * 48000 / 22050))) <= 1
    assert int(np.max(np.abs(out))) <= 28000


def test_play_uses_device_native_rate(monkeypatch):
    import io
    import wave

    played = {}

    class FakeSD:
        def query_devices(self, idx=None, kind=None):
            del kind
            return {
                "name": "MacBook Air-Lautsprecher",
                "index": 2,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            }

        def play(self, data, samplerate=None, device=None, blocking=None):
            played["n"] = len(data)
            played["sr"] = samplerate
            played["device"] = device
            played["blocking"] = blocking
            played["peak"] = int(np.max(np.abs(data)))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        tone = (np.sin(np.linspace(0, 20, 2205)) * 32767).astype(np.int16)
        w.writeframes(tone.tobytes())

    monkeypatch.setattr(audio_mod, "_require_audio", lambda: (FakeSD(), np))
    monkeypatch.setattr(audio_mod.time, "sleep", lambda *_a, **_k: None)
    a = SimAudio(speaker_device=2)
    asyncio.run(a.play(buf.getvalue(), "wav"))
    assert played["sr"] == 48000
    assert played["device"] == 2
    assert played["blocking"] is True
    assert played["peak"] <= 28000
    assert played["n"] > 2205  # upsampled


class _FakeStream:
    def __init__(self, parent):
        self._parent = parent

    def read(self, n):
        return self._parent.rec(n), False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSD:
    def __init__(self, make_block, native_sr=48000, device=1):
        self.calls = 0
        self.default = SimpleNamespace(device=(device, 2))
        self._make_block = make_block
        self._native_sr = native_sr
        self._device = device

    def query_devices(self, idx=None, kind=None):
        if kind == "input":
            return {
                "name": "fake mic",
                "index": self._device,
                "default_samplerate": float(self._native_sr),
            }
        return {"name": "fake mic", "default_samplerate": float(self._native_sr)}

    def InputStream(self, **kwargs):
        assert kwargs["samplerate"] == self._native_sr
        assert kwargs["device"] == self._device
        return _FakeStream(self)

    def rec(self, n):
        self.calls += 1
        return self._make_block(self.calls, n)


def test_record_vad_and_level_callback(monkeypatch):
    native = 48000
    chunk = int(native * 0.1)
    rng = np.random.default_rng(0)

    def make_block(call, n):
        t0 = call * 0.1
        t = np.arange(n) / native + t0
        # quiet ambient for calibration (5 chunks), then speech, then silence
        if 0.5 < t0 < 1.2:  # after noise calibration; > min_speech_s
            return (np.sin(2 * np.pi * 440 * t) * 200).astype(np.int16).reshape(-1, 1)
        return (rng.standard_normal(n) * 8).astype(np.int16).reshape(-1, 1)

    fake = _FakeSD(make_block, native_sr=native)
    monkeypatch.setattr(audio_mod, "_require_audio", lambda: (fake, np))
    levels = []
    a = SimAudio(
        sample_rate=16000,
        vad_threshold=40.0,
        vad_silence_s=0.2,
        vad_listen_s=3.0,
        mic_gain=1.0,  # keep fake amplitudes as-authored
        on_level=levels.append,
    )
    wav = asyncio.run(a.record(max_s=3.0))
    assert wav[:4] == b"RIFF"
    assert fake.calls < 30  # stopped early at speech end, not after max_s
    assert chunk == 4800
    assert levels and max(levels) > 0.3  # ~140/250 shows on the bar


def test_record_silence_aborts_at_listen_s_and_returns_empty(monkeypatch, caplog):
    native = 16000

    def make_block(_call, n):
        return np.zeros((n, 1), dtype=np.int16)

    monkeypatch.setattr(
        audio_mod, "_require_audio", lambda: (_FakeSD(make_block, native_sr=native), np)
    )
    levels = []
    a = SimAudio(
        sample_rate=16000,
        vad_threshold=40.0,
        vad_listen_s=0.5,
        mic_gain=1.0,
        on_level=levels.append,
    )
    with caplog.at_level(logging.INFO, logger="shikigami.sim.audio"):
        wav = asyncio.run(a.record(max_s=8.0))
    assert wav == b""
    assert levels and max(levels) == 0.0
    assert "no speech detected" in caplog.text
    # ~0.5 s of 100 ms chunks, not a full 8 s
    assert a._vad_listen_s == 0.5


def test_record_aborts_when_muted(monkeypatch, caplog):
    native = 16000
    a = SimAudio(sample_rate=16000, vad_threshold=40.0, vad_listen_s=3.0, mic_gain=1.0)

    def mute_after_a_few_chunks(call, n):
        if call >= 3:
            a.set_muted(True)
        return (np.ones(n, dtype=np.int16) * 3).reshape(-1, 1)

    fake = _FakeSD(mute_after_a_few_chunks, native_sr=native)
    monkeypatch.setattr(audio_mod, "_require_audio", lambda: (fake, np))
    with caplog.at_level(logging.INFO, logger="shikigami.sim.audio"):
        wav = asyncio.run(a.record(max_s=8.0))
    assert wav == b""
    assert fake.calls < 20
    assert "muted" in caplog.text
