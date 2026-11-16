"""Mic path that stays in the HAL: device choice, playback, mute, turn serialization."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from shikigami.app.bus import EventBus
from shikigami.app.loop import App
from shikigami.config import load_config
from shikigami.hal.sim import audio as audio_mod
from shikigami.hal.sim.audio import SimAudio, _prepare_play_pcm, _resample_int16
from shikigami.memory.store import MemoryStore

np = pytest.importorskip("numpy")


def _app(tmp_path):
    cfg = load_config()
    cfg.memory.path = str(tmp_path / "m.db")
    hal = SimpleNamespace(
        audio=SimpleNamespace(set_muted=lambda *_a, **_k: None),
        expression=SimpleNamespace(),
        motion=SimpleNamespace(moving=False),
        camera=None,
        backend="sim",
    )
    return App(
        cfg,
        hal,
        brain=None,
        memory=MemoryStore(cfg.memory.path),
        bus=EventBus(),
        log=logging.getLogger("test"),
        use_voice=False,
    )


def test_default_vad_silence_is_half_second():
    cfg = load_config()
    assert cfg.sim.vad_silence_s == 0.5


def test_run_falls_back_to_text_when_voice_is_off(tmp_path):
    app = _app(tmp_path)
    app.cfg.sim.input = "mic"
    called = []

    async def fake_text():
        called.append(True)

    app._run_text = fake_text
    asyncio.run(app.run())
    assert called == [True]


def test_handle_text_serializes_concurrent_turns(tmp_path):
    """Dashboard + stdin must not overlap — one turn at a time."""
    app = _app(tmp_path)
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
    app = _app(tmp_path)
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


def test_sim_audio_from_sim_config_copies_vad():
    cfg = load_config()
    cfg.sim.vad_threshold = 123.0
    cfg.sim.vad_silence_s = 0.7
    cfg.sim.vad_listen_s = 2.5
    cfg.sim.sample_rate = 22050
    cfg.sim.mic_device = 4
    cfg.sim.speaker_device = 3
    audio = SimAudio.from_sim_config(cfg.sim)
    assert audio._vad_threshold == 123.0
    assert audio._vad_silence_s == 0.7
    assert audio._vad_listen_s == 2.5
    assert audio._sr == 22050
    assert audio._device == 4
    assert audio._speaker_device == 3


def test_resolve_speaker_prefers_macbook(monkeypatch):
    del monkeypatch

    class FakeSD:
        def query_devices(self, idx=None, kind=None):
            devs = [
                {"name": "D10s", "max_output_channels": 2},
                {"name": "MacBook Air-Lautsprecher", "max_output_channels": 2},
            ]
            if kind == "output":
                return {**devs[0], "index": 0}
            if idx is None:
                return devs
            return devs[idx]

    audio = SimAudio(speaker_device=None)
    assert audio._resolve_speaker(FakeSD()) == 1
    pinned = SimAudio(speaker_device=0)
    assert pinned._resolve_speaker(FakeSD()) == 0


def test_resample_int16_halves_rate():
    pcm = np.arange(8, dtype=np.int16)
    out = _resample_int16(pcm, 16000, 8000, np)
    assert out.dtype == np.int16
    assert len(out) == 4


def test_prepare_play_pcm_resamples_and_soft_limits_peak():
    n = 2205
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
            del idx, kind
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
    audio = SimAudio(speaker_device=2)
    asyncio.run(audio.play(buf.getvalue(), "wav"))
    assert played["sr"] == 48000
    assert played["device"] == 2
    assert played["blocking"] is True
    assert played["peak"] <= 28000
    assert played["n"] > 2205


def test_record_is_a_stub(caplog):
    audio = SimAudio()
    with caplog.at_level(logging.INFO, logger="shikigami.sim.audio"):
        wav = asyncio.run(audio.record())
    assert wav == b""
    assert "Pipecat" in caplog.text
