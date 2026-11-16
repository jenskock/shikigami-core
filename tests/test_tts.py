"""piper-http URL probe. The GPL piper package is not imported."""

import asyncio
import io
import wave

import pytest
from pydantic import ValidationError

from shikigami.config import TTSConfig, load_config
from shikigami.voice.pipeline import build_tts
from shikigami.voice.piper_http import PiperHttpTTSService, candidate_urls


def _wav(sample_rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x01" * 20)
    return buf.getvalue()


class _Response:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8", "replace")

    @property
    def content(self):
        body = self._body

        class _Content:
            def iter_chunked(self, _size):
                async def _gen():
                    yield body

                return _gen()

        return _Content()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _Session:
    def __init__(self, routes: dict[str, int]):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json=None, headers=None, timeout=None):
        del headers, timeout
        self.calls.append((url, json))
        path = "/" + url.split("/", 3)[3]
        status = self.routes.get(path, 404)
        body = _wav() if status == 200 else b"missing"
        return _Response(status, body)


def test_candidate_urls_probe_both_paths():
    assert candidate_urls("http://jesktop:5000/") == [
        "http://jesktop:5000/synthesize",
        "http://jesktop:5000/api/tts",
    ]
    assert candidate_urls("http://jesktop:5000/synthesize") == ["http://jesktop:5000/synthesize"]


def test_piper_http_autodetects_synthesize_and_reuses_it():
    session = _Session({"/api/tts": 404, "/synthesize": 200})
    service = PiperHttpTTSService(
        base_url="http://jesktop:5000/",
        aiohttp_session=session,  # type: ignore[arg-type]
        timeout_s=7,
    )
    service._sample_rate = 22050

    async def run() -> bytes:
        frames = [frame async for frame in service.run_tts("hello there", "ctx")]
        await service.run_tts("again", "ctx2").__anext__()
        return b"".join(frame.audio for frame in frames)

    audio = asyncio.run(run())
    assert audio
    assert session.calls[0][0].endswith("/synthesize") or session.calls[0][0].endswith("/api/tts")
    assert session.calls[0][1] == {"text": "hello there"}
    assert session.calls[-1][0].endswith("/synthesize")
    assert session.calls[-1][1] == {"text": "again"}
    assert sum(url.endswith("/api/tts") for url, _payload in session.calls) == 0


def test_piper_http_legacy_api_tts():
    session = _Session({"/api/tts": 200, "/synthesize": 404})
    service = PiperHttpTTSService(
        base_url="http://localhost:5000",
        aiohttp_session=session,  # type: ignore[arg-type]
    )
    service._sample_rate = 16000
    frames = asyncio.run(_collect(service))
    assert frames
    assert session.calls[-1][0] == "http://localhost:5000/api/tts"


async def _collect(service: PiperHttpTTSService):
    return [frame async for frame in service.run_tts("hallo", "ctx")]


def test_response_format_rejects_non_wav():
    with pytest.raises(ValidationError):
        TTSConfig(response_format="mp3")


def test_in_process_piper_is_refused():
    cfg = load_config()
    cfg.tts.provider = "piper"
    cfg.tts.enabled = True
    with pytest.raises(RuntimeError, match="GPL"):
        build_tts(cfg, session=None)


def test_openai_tts_accepts_free_text_voice_design():
    from shikigami.voice.openai_compat_tts import FreeTextVoiceOpenAITTSService

    cfg = load_config()
    cfg.tts.provider = "openai"
    cfg.tts.enabled = True
    cfg.tts.base_url = "https://router.eu.requesty.ai/v1"
    cfg.tts.model = "deepinfra/Qwen/Qwen3-TTS-VoiceDesign"
    cfg.tts.voice = "A clear young female German voice, soft cheerful and playful"
    cfg.tts.api_key = "test-key"
    service = build_tts(cfg, session=None)
    assert isinstance(service, FreeTextVoiceOpenAITTSService)
    assert service._settings.voice == cfg.tts.voice
    assert service._settings.model == cfg.tts.model


def test_openai_compat_tts_labels_pcm_as_24khz():
    """Device output is often 48 kHz; mislabeling 24 kHz PCM as 48 kHz chipmunks."""
    from pipecat.frames.frames import TTSAudioRawFrame
    from pipecat.services.openai.tts import OPENAI_SAMPLE_RATE

    from shikigami.voice.openai_compat_tts import FreeTextVoiceOpenAITTSService

    class _Resp:
        status_code = 200

        async def text(self):
            return ""

        def iter_bytes(self, _size):
            async def _gen():
                yield b"\x00\x01" * 32

            return _gen()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _Stream:
        def create(self, **_kwargs):
            return _Resp()

    class _Speech:
        with_streaming_response = _Stream()

    class _Audio:
        speech = _Speech()

    class _Client:
        audio = _Audio()

    service = FreeTextVoiceOpenAITTSService(
        api_key="x",
        settings=FreeTextVoiceOpenAITTSService.Settings(
            model="deepinfra/Qwen/Qwen3-TTS-VoiceDesign",
            voice="A clear young female German voice",
        ),
    )
    service._client = _Client()
    service._sample_rate = 48000  # what a Mac speaker transport would set

    async def _usage(_text):
        return None

    async def _ttfb():
        return None

    service.start_tts_usage_metrics = _usage  # type: ignore[method-assign]
    service.stop_ttfb_metrics = _ttfb  # type: ignore[method-assign]

    frames = asyncio.run(_collect_openai(service))
    audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert audio
    assert all(f.sample_rate == OPENAI_SAMPLE_RATE == 24000 for f in audio)


async def _collect_openai(service):
    return [frame async for frame in service.run_tts("hallo", "ctx")]
