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
    max_tokens: int = 1536  # hidden reasoning tokens count against this too
    timeout_s: float = 30.0
    mock: bool = False  # canned reply, no network (dev/tests)


class STTConfig(BaseModel):
    base_url: str = ""  # falls back to llm.base_url
    api_key: str = ""
    model: str = "whisper-1"
    language: str = ""  # optional ISO-639-1 or name (German/de); empty = auto-detect
    timeout_s: float = 30.0
    enabled: bool = True


TTSProvider = Literal["openai", "piper", "piper-http", "none"]
VoiceTransport = Literal["sounddevice", "fake"]


class VoiceConfig(BaseModel):
    """Pipecat turn-taking. Silero replaces the old energy VAD."""

    barge_in: bool = True
    interrupt_while_moving: bool = False
    smart_turn: bool = False  # ket-chat voice uses speech-timeout + wait_for_transcript
    vad_confidence: float = 0.7
    vad_stop_secs: float = 0.2  # Silero stop; keep this short when Smart Turn is on
    echo_guard: bool = True  # close the mic while the speaker is audible
    echo_tail_secs: float = 0.3  # room echo after the last sound; device latency is added
    transport: VoiceTransport = "sounddevice"


class TTSConfig(BaseModel):
    provider: TTSProvider = "piper-http"  # openai | piper | piper-http | none
    base_url: str = ""  # falls back to llm.base_url
    api_key: str = ""
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    piper_voice: str = ""
    # WAV only — SimAudio.play (sounddevice) cannot decode mp3/opus/…
    response_format: Literal["wav"] = "wav"
    timeout_s: float = 30.0
    enabled: bool = True


class SimConfig(BaseModel):
    window: bool = True
    camera_preview: bool = True  # second window: webcam feed + detections
    camera_index: int = 0
    fake_touch_prob: float = 0.0  # per-tick probability of a simulated touch event
    fake_sound_prob: float = 0.0  # per-tick probability of a simulated sound event
    vad_threshold: float = (
        40.0  # minimum speech RMS (int16); auto-raised above noise floor while listening
    )
    vad_silence_s: float = 0.5  # silence after speech before the utterance ends
    vad_listen_s: float = 3.0  # give up waiting for speech after this many seconds
    mic_device: int | None = None
    speaker_device: int | None = None  # null = prefer MacBook speakers over external DACs
    mic_gain: float = 8.0  # software gain on capture (MacBook mics are soft ~peak 200)
    sample_rate: int = 16000  # WAV/STT rate; capture uses the device's native rate
    input: str = "mic"  # text | mic


class PerceptionConfig(BaseModel):
    track_faces: bool = True  # look at detected faces via the camera
    face_cooldown_s: float = 2.0  # min time between look_at/emote reactions to faces


class ReflexConfig(BaseModel):
    hz: float = 20.0
    obstacle_cm: float = 25.0
    touch_react: bool = True
    emote_gestures: bool = True  # a small body gesture goes with each face


class MemoryConfig(BaseModel):
    path: str = "data/memory.sqlite3"


class PersonalityRef(BaseModel):
    path: str = "config/personality.yaml"


class EyesConfigModel(BaseModel):
    """Procedural eyes (expression.eyes) — color, blink rate, personality motion."""

    color: tuple[int, int, int] = (255, 255, 255)
    glow: float = 0.35
    base_width: float = 1.0
    base_height: float = 1.0
    corner_radius: float = 0.85
    gap: float = 1.0
    blink_rate_per_min: float = 8.0
    double_blink_prob: float = 0.08
    gaze_wander: float = 0.35
    saccade_amp: float = 0.04
    breath_amp: float = 0.025
    spring_omega: float = 14.0
    gaze_spring_omega: float = 22.0
    blink_spring_omega: float = 40.0
    personality: str = "curious"  # curious | calm | mischievous
    seed: int | None = None


class Config(BaseModel):
    backend: str = Field(default="sim")  # sim | pidog | picrawler
    log_level: str = "INFO"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    reflexes: ReflexConfig = Field(default_factory=ReflexConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    personality: PersonalityRef = Field(default_factory=PersonalityRef)
    eyes: EyesConfigModel = Field(default_factory=EyesConfigModel)


_TTS_PROVIDERS = ("openai", "piper", "piper-http", "none")
_VOICE_TRANSPORTS = ("sounddevice", "fake")
_TRUE = {"1", "true", "yes", "on"}


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in _TRUE


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
        if v not in _TTS_PROVIDERS:
            raise ValueError(f"TTS_PROVIDER must be one of {_TTS_PROVIDERS}, got {v!r}")
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

    if (flag := _env_bool("VOICE_BARGE_IN")) is not None:
        cfg.voice.barge_in = flag
    if (flag := _env_bool("VOICE_INTERRUPT_WHILE_MOVING")) is not None:
        cfg.voice.interrupt_while_moving = flag
    if (flag := _env_bool("VOICE_SMART_TURN")) is not None:
        cfg.voice.smart_turn = flag
    if (flag := _env_bool("VOICE_ECHO_GUARD")) is not None:
        cfg.voice.echo_guard = flag
    if v := _env("VOICE_ECHO_TAIL_SECS"):
        cfg.voice.echo_tail_secs = float(v)
    if v := _env("SHIKIGAMI_VOICE_TRANSPORT"):
        if v not in _VOICE_TRANSPORTS:
            raise ValueError(
                f"SHIKIGAMI_VOICE_TRANSPORT must be one of {_VOICE_TRANSPORTS}, got {v!r}"
            )
        cfg.voice.transport = v

    return cfg
