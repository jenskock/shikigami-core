"""Config loading + personality loading."""

from shikigami.config import REPO_ROOT, load_config
from shikigami.personality.loader import load_personality


def test_load_default_config(monkeypatch):
    monkeypatch.delenv("SHIKIGAMI_BACKEND", raising=False)
    cfg = load_config()
    assert cfg.backend == "picrawler"
    assert cfg.llm.model
    assert cfg.llm.max_tokens == 96  # hard cap: llm time ~= tokens generated
    assert cfg.sim.input == "mic"
    assert cfg.reflexes.hz >= 20


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
    # stt/tts fall back to the llm base url
    assert cfg.stt.base_url == "http://jesktop:11434/v1"


def test_personality_loads():
    p = load_personality(REPO_ROOT / "config" / "personality.yaml")
    assert p.name == "Kogane"
    assert p.character
    assert p.language == "German"  # matches the de_DE piper voice
    assert len(p.rules) >= 3
