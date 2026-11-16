"""Config loading + personality loading."""

from shikigami.config import REPO_ROOT, load_config
from shikigami.personality.loader import load_personality


def test_load_default_config(monkeypatch):
    monkeypatch.delenv("SHIKIGAMI_BACKEND", raising=False)
    cfg = load_config()
    assert cfg.backend == "picrawler"
    assert cfg.llm.model
    assert cfg.llm.max_tokens >= 1024  # hidden reasoning tokens count against the cap
    assert cfg.sim.input == "mic"
    assert cfg.reflexes.hz >= 20
    assert cfg.tts.provider == "piper-http"
    assert cfg.tts.base_url == "http://jesktop:5000"
    assert cfg.voice.barge_in is True
    assert cfg.voice.interrupt_while_moving is False
    assert cfg.voice.smart_turn is False
    assert cfg.voice.echo_guard is True
    assert cfg.voice.echo_tail_secs == 0.3
    assert cfg.voice.transport == "sounddevice"


def test_env_overrides_backend(monkeypatch):
    monkeypatch.setenv("SHIKIGAMI_BACKEND", "sim")
    assert load_config().backend == "sim"


def test_env_overrides_llm(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://jesktop:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("SHIKIGAMI_API_KEY", "sk-test")
    cfg = load_config()
    assert cfg.llm.base_url == "http://jesktop:11434/v1"
    assert cfg.llm.model == "qwen2.5:7b"
    assert cfg.llm.api_key == "sk-test"
    # stt falls back to the llm base url; tts has its own piper-http default
    assert cfg.stt.base_url == "http://jesktop:11434/v1"
    assert cfg.tts.base_url == "http://jesktop:5000"


def test_voice_env_overrides(monkeypatch):
    monkeypatch.setenv("VOICE_BARGE_IN", "false")
    monkeypatch.setenv("VOICE_INTERRUPT_WHILE_MOVING", "yes")
    monkeypatch.setenv("VOICE_SMART_TURN", "0")
    monkeypatch.setenv("VOICE_ECHO_GUARD", "false")
    monkeypatch.setenv("VOICE_ECHO_TAIL_SECS", "0.5")
    monkeypatch.setenv("SHIKIGAMI_VOICE_TRANSPORT", "fake")
    cfg = load_config()
    assert cfg.voice.barge_in is False
    assert cfg.voice.interrupt_while_moving is True
    assert cfg.voice.smart_turn is False
    assert cfg.voice.echo_guard is False
    assert cfg.voice.echo_tail_secs == 0.5
    assert cfg.voice.transport == "fake"


def test_personality_loads():
    p = load_personality(REPO_ROOT / "config" / "personality.yaml")
    assert p.name == "Kogane"
    assert p.character
    assert p.language == "German"  # keep in sync with the TTS voice
    assert len(p.rules) >= 3
