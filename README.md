# shikigami-core

A tiny autonomous robot companion ("buddy") for the living room — its own personality,
expressive behavior, and LLM-driven reactions at low latency.

**Same code on Mac and Raspberry Pi.** Only the lowest layer (the HAL) differs:

- `sim` backend on the Mac: one **Qt (PySide6)** dashboard — device-resolution face
  buffer (OLED-style procedural eyes, 1:1 with the Pi path) | stats + chat field |
  webcam feed with face/cat boxes (faces drive gaze/`look_at` and emit
  `face_seen`/`face_lost`). Mac mic/speaker; motion commands are logged.
  Type in the terminal or in the dashboard chat field (Enter submits). Quit via the
  **Quit** button, Esc, or the window close button. Mute the mic with **Mute mic**
  or Ctrl+M.
  Ctrl-C in the terminal works too — blocking audio/network calls run on daemon
  threads, so even a stuck PortAudio call can't hang shutdown.
- `pidog` / `picrawler` backends: thin wrappers over SunFounder's Python libs on the Pi.
  Those libs are imported **only inside their backend module**. On Mac, `picrawler` falls
  back to SoftPicrawler (same `do_action`/`do_step` surface + IK/gaits, live stick body in
  the dashboard BODY panel) so you can develop without GPIO.

Backend is chosen via `SHIKIGAMI_BACKEND=sim|pidog|picrawler` (or `backend:` in
`config/default.yaml`).

## Architecture

```
src/shikigami/
├── hal/            # abstract interfaces + backends
│   ├── interfaces.py   # Motion, Camera, Audio, Sensors, Expression, HAL
│   ├── sim/            # macOS: webcam/mic/speaker, face window, logged motion
│   ├── pidog/          # Pi stub (import-guarded SunFounder pidog)
│   └── picrawler/      # Pi stub (import-guarded SunFounder picrawler)
├── expression/     # procedural eyes (display-agnostic) + pygame/PIL/cv2 renderers
├── reflexes/       # fast local loop (~20 Hz, no network): obstacle stop, touch/sound reacts, idle face
├── perception/     # face detection (OpenCV, lazy)
├── brain/          # personality, tools, mock LLM. Live turns run in the voice pipeline
├── memory/         # SQLite store: facts, recent events, mood
├── personality/    # personality.yaml -> validated model -> system prompt
├── voice/          # Pipecat: Silero VAD, Smart Turn, STT, tool handlers, TTS
└── app/            # asyncio main loop, event bus, per-stage latency logging
```

## Procedural eyes

Eyes are computed every frame (no GIF loops): spring-damped lids/pupils, Poisson
blinks, micro-saccades, idle gaze wander, and emotion blends. The LLM only sends
high-level `emote` / `look_at` intents — motion detail stays local and instant.

```bash
uv run eyes-demo          # pygame preview: keys 1–9 emotions, mouse = gaze, space = blink
uv run eyes-bench         # PIL 240×240 RGB565 throughput (target ≥30 fps)
uv run python scripts/capture_eyes_preview.py   # writes docs/eyes_preview.png
```

Config knobs live under `eyes:` in `config/default.yaml` (color, blink rate,
`personality: curious|calm|mischievous`). Optional ESP32 offload protocol:
`shikigami.expression.eyes.protocol` + stub firmware under
`expression/eyes/firmware/`. Emotion contact sheet: `docs/eyes_preview.png`.

Speech is a [Pipecat](https://github.com/pipecat-ai/pipecat) pipeline. Everything else
(HAL, eyes, reflexes, face tracking, memory, the dashboard) stays in this repo.

```
mic (sounddevice) ─┐
                   ├─ Silero VAD + Smart Turn ─ OpenAI-compatible STT
typed text ────────┘                                    │
                                                         ▼
                              system prompt (mood, facts, state) + tools
                                                         │
                         move / emote / remember / idle ─┼─ say → TTS → speaker
                                                         │
                                              eyes speak_visual while audio plays
```

`say` is spoken by pushing text at the TTS service. The other tools call the HAL.
Tool results do not start a second LLM generation, so the model speaks through `say`.
Barge-in is on by default and is held off while `motion.moving` is true
(`voice.interrupt_while_moving`). Dashboard mute drops mic audio immediately; the
chat field still submits.

`llm.max_tokens` stays capped (96). A warmup call at startup removes cold-start cost.
Each spoken turn logs `ttfa=…ms` from the moment the turn is accepted until the first
output audio. That number was not measured on a real mic in this environment.

TTS default is `piper-http` at `http://jesktop:5000`. `make setup-piper` creates a
**separate** venv, installs `piper-tts` (GPL-3.0), downloads `de_DE-thorsten-medium`,
and serves `POST /synthesize` (see `scripts/setup-piper.sh`). The app never imports
`piper`. `tts.provider: piper` is refused for that reason. On a Mac, port 5000 is
often AirPlay Receiver — pass another port to the setup script and set `TTS_BASE_URL`.
Optional `tts.provider: openai` uses Pipecat's OpenAI TTS client, which asks for PCM
at 24 kHz. Some OpenAI-compatible routers only return WAV; if speech is silent, use
piper-http.

STT is Pipecat's OpenAI-compatible client (`/audio/transcriptions`). For low latency,
`make setup-whisper` on jesktop (faster-whisper-server on :8000, default
`Systran/faster-whisper-medium`). Cloud fallback: Requesty voxtral, via `STT_BASE_URL`.

## Quickstart (macOS)

```bash
brew install portaudio   # mic/speaker
uv sync                  # venv + deps
cp .env.example .env     # fill in API key / endpoints
uv run sim               # dashboard + mic — that's it
```

Same thing: `make run-sim`. Type instead of speak: `uv run sim --text-input`.

**Permissions (esp. inside Cursor's terminal):** enable **Camera** + **Microphone** for
**Cursor** (or Terminal) under System Settings > Privacy & Security. `--no-camera` skips
the webcam.

One-shot mock turn (no API key, no windows):

```bash
uv run sim --mock --no-window --no-camera --no-tts --text-input --text "hello!"
```

`.env` (Requesty by default; local Ollama on jesktop is a one-line change):

```bash
LLM_BASE_URL=https://router.requesty.ai/v1
LLM_MODEL=openai/gpt-4o-mini
SHIKIGAMI_API_KEY=rqsty-sk-...
```

STT is OpenAI-compatible `/audio/transcriptions` inside the Pipecat pipeline. For low
latency, run Whisper on `jesktop` (`make setup-whisper`, default
`Systran/faster-whisper-medium` on :8000) and point `STT_BASE_URL=http://jesktop:8000/v1`.
Cloud fallback: Requesty voxtral. Dashboard shows `mic:` + level bar + `heard:`.
Mute with the **Mute mic** button or **Ctrl+M** (text chat still works).

The mic stays open. Silero (`voice.vad_confidence`, `voice.vad_stop_secs`) detects
speech; Smart Turn (`voice.smart_turn`, fed by `sim.vad_silence_s`) decides the turn
is finished. `sim.vad_threshold` and `sim.vad_listen_s` are unused. Capture runs at
the device's native rate and is resampled to `sim.sample_rate` (16 kHz). If PortAudio
is missing, the pipeline logs that and keeps typed text and tools working
(`SHIKIGAMI_VOICE_TRANSPORT=fake` forces that path).

Troubleshooting a silent mic:

- Check the level bar: if it stays at 0 while you talk, grant **Microphone**
  permission to the terminal app, or set `sim.mic_device`.
- If the bar moves but turns never start, lower `voice.vad_confidence`.

## Configuration

- `config/default.yaml` — everything: backend, llm/stt/tts endpoints+models, sim
  (face window, camera preview window, camera index, mic, input mode, simulated
  touch/sound event probabilities), perception, reflexes, memory path, personality
  path. CLI flags `--no-window` / `--no-camera` override.
- Sim sensors are simulated (distance is a random walk; touch/sound events are random
  and off by default — set `sim.fake_touch_prob` / `sim.fake_sound_prob` to demo them).
  Only the webcam/mic/speaker are real hardware in the sim backend.
- `config/personality.yaml` — name, character, voice, rules, greeting, and
  `language:` (the LLM always replies in it — keep it in sync with your TTS voice,
  currently `German` for the de_DE piper voice). Becomes the system prompt.
- `eyes:` in `config/default.yaml` — procedural eye color/shape, blink rate, and
  motion personality (`curious` blinks/wanders more; `calm` slower; `mischievous`
  more double-blinks).
- `.env` (gitignored, see `.env.example`) — secrets + overrides. Env wins over YAML:
  `SHIKIGAMI_BACKEND`, `SHIKIGAMI_LANGUAGE`, `LLM_BASE_URL`, `LLM_MODEL`,
  `SHIKIGAMI_API_KEY` / `OPENAI_API_KEY`, `STT_*`, `TTS_*`, `VOICE_BARGE_IN`,
  `VOICE_INTERRUPT_WHILE_MOVING`, `VOICE_SMART_TURN`, `SHIKIGAMI_VOICE_TRANSPORT`,
  `DEPLOY_HOST`.

Cloud ↔ local is a config change: point `LLM_BASE_URL` / `STT_BASE_URL` / `TTS_BASE_URL`
at your Ollama / Whisper (`make setup-whisper`) / Piper (`make setup-piper`) servers on
`jesktop` and no code changes.

## Raspberry Pi

Target: Pi 3 Model B+ (Bookworm 64-bit) — no LLM on device; it only runs the HAL,
reflexes, STT/TTS clients and talks to the inference box. PiDog preferred (head /
expression), PiCrawler as alternative. The `picrawler` motion path maps HAL verbs to
SunFounder `do_action`/`do_step` (SoftPicrawler on Mac when libs are missing). Install
on the Pi: `pip install pidog robot_hat vilib` (or `picrawler ...`).

```bash
uv run sim                 # SoftPicrawler + BODY dashboard on Mac (config backend: picrawler)
uv run sim --backend sim   # label-only sim HAL (no gait / body pose)
```

```bash
make deploy            # rsync to DEPLOY_HOST (placeholder)
DEPLOY_RUN=1 make deploy  # rsync + start on the Pi
```

## Development

```bash
make test     # pytest (no camera/mic/network needed)
make lint     # ruff check + format --check
make fmt
make run-sim
```

Python 3.11 via `uv` (`.python-version`), locked with `uv.lock`. Tests cover action
parsing, HAL backend selection (incl. the SunFounder import guards), config/env
overrides, memory, the mock brain, and the Pipecat pipeline (tool mapping, mute,
and a fake-audio speak path). `pipecat-ai` is pinned to 1.16.x. Silero and Smart
Turn ship as ONNX in that package; the `local-smart-turn` extra (PyTorch) is not
installed. Real mic, speaker, and latency were not measured here.

## Design principles

- `cv2`/`sounddevice` stay lazy. `pipecat-ai` (and its ONNX runtime) is a core
  dependency because the voice pipeline runs on the Pi as well as the Mac. Smart
  Turn's ONNX model is about 8 MB; measure RSS on the Pi 3B+ before treating that
  as free.
- Reflexes cover LLM latency: the fast local loop reacts to obstacles/touch/sound
  even while the brain is "thinking".
- Secrets only in `.env`, never committed.
- No ROS 2 (for now).

## Roadmap

Wake word, richer PiDog/PiCrawler mapping, mood model, continuous eval of prompts,
optional on-device small model on the jesktop box.
