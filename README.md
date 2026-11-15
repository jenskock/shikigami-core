# shikigami-core

A tiny autonomous robot companion ("buddy") for the living room — its own personality,
expressive behavior, and LLM-driven reactions at low latency.

**Same code on Mac and Raspberry Pi.** Only the lowest layer (the HAL) differs:

- `sim` backend on the Mac: one **Qt (PySide6)** dashboard — device-resolution face
  buffer (OLED-style procedural eyes, 1:1 with the Pi path) | stats + chat field |
  webcam feed with face/cat boxes (faces drive gaze/`look_at` and emit
  `face_seen`/`face_lost`). Mac mic/speaker; motion commands are logged.
  Type in the terminal or in the dashboard chat field (Enter submits). Quit via the
  **Quit** button, Esc, or the window close button.
  Ctrl-C in the terminal works too — blocking audio/network calls run on daemon
  threads, so even a stuck PortAudio call can't hang shutdown.
- `pidog` / `picrawler` backends: thin wrappers over SunFounder's Python libs on the Pi.
  Those libs are imported **only inside their backend module**. On Mac, `picrawler` falls
  back to SoftPicrawler (same `do_action`/`do_step` surface + IK/gaits, angles in the
  dashboard SERVOS panel) so you can develop without GPIO.

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
├── perception/     # VAD, face detection (OpenCV, lazy)
├── brain/          # LLM agent: events+state in, structured actions out (tool calls, streaming)
├── memory/         # SQLite store: facts, recent events, mood
├── personality/    # personality.yaml -> validated model -> system prompt
├── voice/          # STT (Whisper / OpenAI-compatible) + TTS (OpenAI-compatible / Piper)
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

Data flow per turn: `mic/text -> [stt] -> brain (LLM, streaming tool calls) -> actions
(move/say/emote/remember/idle) -> HAL (motion/expression/voice)` with every stage timed
and logged (`latency context=.. llm=.. play=.. total=..`).

Latency measures: `llm.max_tokens` is hard-capped (96) since generation time ≈ tokens;
a warmup call at startup removes cold-start cost (client keep-alive afterwards); the LLM
stream is consumed incrementally — move/emote actions execute the moment their tool call
finalizes, and say-text is synthesized sentence-by-sentence (TTS runs in the background,
played in order) so speech overlaps generation instead of adding to it. For the fastest
TTS use Piper on `jesktop` (`tts.provider: piper-http`, German voice): run
`make setup-piper` there (creates a venv, installs piper-tts, downloads
`de_DE-thorsten-medium`, starts the server on :5000 — see `scripts/setup-piper.sh`).
For the fastest STT run `make setup-whisper` on jesktop (faster-whisper-server on
:8000, default `Systran/faster-whisper-medium` — see `scripts/setup-whisper.sh`).

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

STT is OpenAI-compatible `/audio/transcriptions`. For low latency, run Whisper on
`jesktop` (`make setup-whisper`, default `Systran/faster-whisper-medium` on :8000) and
point `STT_BASE_URL=http://jesktop:8000/v1`. Cloud fallback: Requesty voxtral.
Dashboard shows `mic:` + level bar + `heard:`. Tune with `sim.vad_threshold` /
`sim.vad_silence_s` (default 0.5 s after speech). Each spoken turn logs `ttfa=…ms`
(silence-end → first PCM). Mic errors fall back to text input.

Mic capture uses a continuous stream at the device's native rate (often 48 kHz),
then resamples to `sim.sample_rate` (16 kHz) for STT. If no speech starts within
`sim.vad_listen_s` (default 3 s), the cycle ends without calling STT.

Troubleshooting "listening… but nothing happens" (logs `no speech detected, peak rms …`):

- Check the level bar in the dashboard: if it stays at 0 while you talk, the mic
  delivers silence — on macOS grant **Microphone** permission to your terminal app
  (System Settings > Privacy & Security > Microphone), or pick another device with
  `sim.mic_device` (the app logs `input device #N … native … Hz`).
- If the bar moves but never triggers, lower `sim.vad_threshold`; each quiet cycle
  logs `peak rms … (threshold …)` so you can see how close you got.

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
  `SHIKIGAMI_API_KEY` / `OPENAI_API_KEY`, `STT_*`, `TTS_*`, `DEPLOY_HOST`.

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
uv run sim                 # SoftPicrawler + servo dashboard on Mac (config backend: picrawler)
uv run sim --backend sim   # label-only sim HAL (no gait/servos)
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
overrides, memory, VAD and the mock brain.

## Design principles

- Core never imports vendor or heavy libs; `cv2`/`sounddevice`/`numpy` are lazy and
  only in the `sim` extra — the Pi 3B+ stays light (no GPU, limited RAM).
- Reflexes cover LLM latency: the fast local loop reacts to obstacles/touch/sound
  even while the brain is "thinking".
- Secrets only in `.env`, never committed.
- No ROS 2 (for now).

## Roadmap

Wake word, richer PiDog/PiCrawler mapping, mood model, continuous eval of prompts,
optional on-device small model on the jesktop box.
