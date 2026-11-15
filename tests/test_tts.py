"""TTS providers: piper-http posts to the remote Piper server; none disables."""

import httpx
import pytest
from pydantic import ValidationError

from shikigami.config import TTSConfig
from shikigami.voice.tts import TTSClient


class _Resp:
    def __init__(self, content=b"wav-bytes", status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        pass


def _patch_post(monkeypatch, routes):
    """routes: path -> status code; returns list of requested urls."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        path = url.split("/", 3)[3]  # strip scheme+host
        return _Resp(status_code=routes.get("/" + path, 404))

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def test_piper_http_autodetects_synthesize_1_8(monkeypatch):
    calls = _patch_post(monkeypatch, {"/api/tts": 404, "/synthesize": 200})
    tts = TTSClient(TTSConfig(provider="piper-http", base_url="http://jesktop:5000/", timeout_s=7))
    data, fmt = tts.synth("hello there")
    assert data == b"wav-bytes" and fmt == "wav"
    assert calls[-1][0] == "http://jesktop:5000/synthesize"
    assert calls[-1][1] == {"text": "hello there"}
    assert calls[-1][2] == 7
    # second call reuses the discovered path (no more probing)
    tts.synth("again")
    assert calls[-1][0].endswith("/synthesize")


def test_piper_http_legacy_api_tts_1_3(monkeypatch):
    calls = _patch_post(monkeypatch, {"/api/tts": 200})
    tts = TTSClient(TTSConfig(provider="piper-http", base_url="http://localhost:5000"))
    data, _ = tts.synth("hallo")
    assert data == b"wav-bytes"
    assert calls[-1][0] == "http://localhost:5000/api/tts"


def test_none_provider_returns_empty():
    tts = TTSClient(TTSConfig(provider="none"))
    assert tts.synth("hi") == (b"", "none")


def test_response_format_rejects_non_wav():
    with pytest.raises(ValidationError):
        TTSConfig(response_format="mp3")


def test_openai_synth_always_requests_wav(monkeypatch):
    seen = {}

    class _Speech:
        def create(self, **kw):
            seen.update(kw)
            return type("R", (), {"content": b"RIFF"})()

    class _Client:
        audio = type("A", (), {"speech": _Speech()})()

    tts = TTSClient(TTSConfig(provider="openai", model="tts-1", voice="alloy"))
    monkeypatch.setattr(tts, "_get_client", lambda: _Client())
    data, fmt = tts.synth("hi")
    assert data == b"RIFF" and fmt == "wav"
    assert seen["response_format"] == "wav"
