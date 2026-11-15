"""Configuration loading: config/default.yaml + personality path + .env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class LLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout_s: float = 30.0
    mock: bool = False  # canned reply, no network (dev/tests)


class STTConfig(BaseModel):
    base_url: str = ""  # falls back to llm.base_url
    api_key: str = ""
    model: str = "whisper-1"
    timeout_s: float = 30.0
    enabled: bool = False


class TTSConfig(BaseModel):
    provider: str = "openai"  # openai | piper | none
    base_url: str = ""  # falls back to llm.base_url
    api_key: str = ""
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    piper_voice: str = ""
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "wav"
    timeout_s: float = 30.0
    enabled: bool = True


class SimConfig(BaseModel):
    window: bool = True
    camera_preview: bool = True  # second window: webcam feed + detections
    camera_index: int = 0
    fake_touch_prob: float = 0.0  # per-tick probability of a simulated touch event
    fake_sound_prob: float = 0.0  # per-tick probability of a simulated sound event
    mic_device: int | None = None
    sample_rate: int = 16000
    input: str = "text"  # text | mic


class PerceptionConfig(BaseModel):
    track_faces: bool = True  # look at detected faces via the camera
    face_cooldown_s: float = 2.0  # min time between look_at/emote reactions to faces


class ReflexConfig(BaseModel):
    hz: float = 20.0
    obstacle_cm: float = 25.0
    touch_react: bool = True


class MemoryConfig(BaseModel):
    path: str = "data/memory.sqlite3"


class PersonalityRef(BaseModel):
    path: str = "config/personality.yaml"


class Config(BaseModel):
    backend: str = Field(default="sim")  # sim | pidog | picrawler
    log_level: str = "INFO"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    reflexes: ReflexConfig = Field(default_factory=ReflexConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    personality: PersonalityRef = Field(default_factory=PersonalityRef)


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def load_config(path: str | Path | None = None) -> Config:
    """Load YAML config, then apply environment overrides (.env is loaded by __main__)."""
    path = Path(path or os.environ.get("SHIKIGAMI_CONFIG") or DEFAULT_CONFIG_PATH)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(**data)

    if v := _env("SHIKIGAMI_BACKEND"):
        cfg.backend = v
    if v := _env("SHIKIGAMI_LOG_LEVEL"):
        cfg.log_level = v

    if v := _env("SHIKIGAMI_API_KEY", "OPENAI_API_KEY"):
        cfg.llm.api_key = v
    if v := _env("LLM_BASE_URL"):
        cfg.llm.base_url = v
    if v := _env("LLM_MODEL"):
        cfg.llm.model = v

    if v := _env("STT_BASE_URL"):
        cfg.stt.base_url = v
    if v := _env("STT_MODEL"):
        cfg.stt.model = v
    if v := _env("STT_API_KEY", "SHIKIGAMI_API_KEY", "OPENAI_API_KEY"):
        cfg.stt.api_key = v
    if not cfg.stt.base_url:
        cfg.stt.base_url = cfg.llm.base_url

    if v := _env("TTS_PROVIDER"):
        cfg.tts.provider = v
    if v := _env("TTS_BASE_URL"):
        cfg.tts.base_url = v
    if v := _env("TTS_MODEL"):
        cfg.tts.model = v
    if v := _env("TTS_VOICE"):
        cfg.tts.voice = v
    if v := _env("TTS_PIPER_VOICE"):
        cfg.tts.piper_voice = v
    tts_key = _env("TTS_API_KEY", "SHIKIGAMI_API_KEY", "OPENAI_API_KEY")
    if tts_key:
        cfg.tts.api_key = tts_key
    if not cfg.tts.base_url:
        cfg.tts.base_url = cfg.llm.base_url

    return cfg
