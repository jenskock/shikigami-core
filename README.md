# shikigami-core

A tiny autonomous robot companion ("buddy") for the living room — its own personality,
expressive behavior, and LLM-driven reactions at low latency.

**Same code on Mac and Raspberry Pi.** Only the lowest layer (the HAL) differs:

- `sim` backend on the Mac: one dashboard window (panels: animated face | stats | webcam
  feed with face/cat detection boxes — detected faces also drive `look_at` and emit
  `face_seen`/`face_lost` events), Mac mic/speaker, and motion commands are logged.
  Text input works both in the terminal and directly in the dashboard window (the
  `you>` line at the bottom of the stats panel) — the OpenCV window often has keyboard
  focus on macOS. The camera panel letterboxes the feed to keep its aspect ratio.
- `pidog` / `picrawler` backends on the Pi: thin wrappers over SunFounder's Python libs.
  Those libs are imported **only inside their backend module**, so the Mac never needs them.

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
├── reflexes/       # fast local loop (~20 Hz, no network): obstacle stop, touch/sound reacts, idle face
├── perception/     # VAD, face detection (OpenCV, lazy)
├── brain/          # LLM agent: events+state in, structured actions out (tool calls, streaming)
├── memory/         # SQLite store: facts, recent events, mood
├── personality/    # personality.yaml -> validated model -> system prompt
├── voice/          # STT (Whisper / OpenAI-compatible) + TTS (OpenAI-compatible / Piper)
└── app/            # asyncio main loop, event bus, per-stage latency logging
```

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

## Quickstart (macOS)

```bash
brew install portaudio        # needed by sounddevice for mic/speaker
uv sync --extra sim           # venv + lockfile + sim deps (opencv, sounddevice, numpy)
cp .env.example .env          # fill in LLM endpoint/key; see below
make run-sim                  # interactive: type a line, Shikigami answers
```

One-shot turn without any API key (mock LLM, no windows, no TTS):

```bash
uv run shikigami --backend sim --mock --no-window --no-camera --no-tts --text "hello!"
# or pipe lines into the interactive loop (exits cleanly at EOF):
printf 'hello!\n' | uv run shikigami --backend sim --mock --no-window --no-camera --no-tts
```

With a real LLM (any OpenAI-compatible endpoint — the default is the Requesty router,
local Ollama on `jesktop` is a one-line change):

```bash
# .env — default cloud (Requesty, provider-prefixed model names)
LLM_BASE_URL=https://router.requesty.ai/v1
LLM_MODEL=openai/gpt-4o-mini
SHIKIGAMI_API_KEY=rqsty-sk-...
# STT (Requesty voxtral) and TTS (openai/tts-1) default to the same router; see .env.example
# local alternative: LLM_BASE_URL=http://jesktop:11434/v1, LLM_MODEL=qwen2.5:7b (no key needed)
uv run --extra sim shikigami --backend sim --text "what's up?"
```

Mic input instead of typing: set `sim.input: mic` and `stt.enabled: true` in
`config/default.yaml` (STT = any OpenAI-compatible `/audio/transcriptions` — Requesty
`mistral/voxtral-mini-latest` by default, or a Whisper server on `jesktop`).

## Configuration

- `config/default.yaml` — everything: backend, llm/stt/tts endpoints+models, sim
  (face window, camera preview window, camera index, mic, input mode, simulated
  touch/sound event probabilities), perception, reflexes, memory path, personality
  path. CLI flags `--no-window` / `--no-camera` override.
- Sim sensors are simulated (distance is a random walk; touch/sound events are random
  and off by default — set `sim.fake_touch_prob` / `sim.fake_sound_prob` to demo them).
  Only the webcam/mic/speaker are real hardware in the sim backend.
- `config/personality.yaml` — name, character, voice, rules, greeting. Becomes the
  system prompt.
- `.env` (gitignored, see `.env.example`) — secrets + overrides. Env wins over YAML:
  `SHIKIGAMI_BACKEND`, `LLM_BASE_URL`, `LLM_MODEL`, `SHIKIGAMI_API_KEY` / `OPENAI_API_KEY`,
  `STT_*`, `TTS_*`, `DEPLOY_HOST`.

Cloud ↔ local is a config change: point `LLM_BASE_URL` at your Ollama/Whisper/Piper
server on `jesktop` and no code changes.

## Raspberry Pi

Target: Pi 3 Model B+ (Bookworm 64-bit) — no LLM on device; it only runs the HAL,
reflexes, STT/TTS clients and talks to the inference box. PiDog preferred (head /
expression), PiCrawler as alternative. The `pidog`/`picrawler` backends are stubs:
import-guarded, mapped where the SunFounder API is known, `NotImplementedError` where
it still needs mapping. Install on the Pi: `pip install pidog robot_hat vilib`
(or `picrawler ...`).

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
