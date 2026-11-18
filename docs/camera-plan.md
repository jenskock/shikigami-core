# Camera integration plan

Status: phases 1–3 implemented (2026-11-18) — camera contract, Mac emulation, YuNet perception,
face tracks + SFace recognition, person_arrived/left, remember_face/forget_face, enroll script,
look() tool (snapshot -> vision model). Phases 4–5 not started.

## Goals

1. **Real Pi camera** — face tracking/gaze works on the robot, not just the Mac webcam.
2. **The LLM can see** — via local text events, plus an on-demand `look()` snapshot.
3. **Recognize people** (and the cat) — and tie identities to memory.
4. **Proactive but polite** — greets known people, comments occasionally, rate-limited.

Non-goals for now: vision-based navigation/obstacle avoidance, following people.
Privacy is not a design constraint (snapshots may go to the cloud).

## Where we start

| Piece | Today |
|---|---|
| `hal/interfaces.py` `Camera` | `start` / `read_frame` / `preview_frame` / `stop` |
| `hal/sim/camera.py` | Mac webcam, runs detection for the preview, exposes `last_frame` / `last_detections` |
| `hal/picrawler/backend.py` `PicrawlerCamera`, `hal/pidog/backend.py` `PiDogCamera` | stubs: `NotImplementedError("use vilib camera on the Pi")` |
| `perception/face.py` | OpenCV **Haar cascades** (face + cat face) at 320 px |
| `perception/tracker.py` | largest face → `motion.look_at`, publishes `face_seen` / `face_lost` |
| Voice LLM | `vertex/claude-haiku-5-5@eu` via Requesty (OpenAI-compatible) — multimodal, so `look()` can reuse the same endpoint |
| Voice context | already receives `memory.recent_events(8)` — the natural channel for vision events |

## Architecture

```
 picamera2 / webcam
        │  lores 320×240 (detect)        main frame (snapshots)
        ▼                                     │
 ┌──────────────── perception (Pi-local, own thread) ───────────────┐
 │ T0  YuNet face detect  ~15 fps ──► tracker ──► look_at (reflex) │
 │ T1  SFace embed on new track ──► match faces DB ──► identity    │
 │ T2  cat/object detector ~1 Hz (YOLO-nano, ONNX)                 │
 └───────────────┬──────────────────────────────────────────────────┘
                 │ bus events: person_arrived(name|unknown), person_left,
                 │             cat_seen, cat_left
        ┌────────┴─────────┐
        ▼                  ▼
  memory.log_event    attention policy ──► proactive speech / emote
        │                                  (cooldowns, never mid-turn)
        ▼
  voice LLM context (text)   ◄── look(question?) tool ──► snapshot JPEG
                                                            ──► cloud VLM ──► text
```

### 1. Camera HAL

- New `hal/pi/camera.py` (shared by `picrawler` and `pidog`): `PiCamera` on **picamera2**
  (not vilib — vilib brings its own web server/threads/detection that fight our asyncio loop).
  - Two streams from one sensor: `lores` 320×240 for detection, `main` ~1296×972 for snapshots.
  - Capture in a daemon thread; keep only the latest frame (no queue growth).
  - Import-guarded like the SunFounder libs.
- Extend `Camera` with a default-implemented
  `async def snapshot_jpeg(max_side: int = 768, quality: int = 80) -> bytes | None`.
  `SimCamera` implements it from the webcam frame so Mac and Pi share `look()`.
- Formalize the duck-typed `last_frame` / `last_detections` into a small `Perception`
  object owned by the app, rather than living inside `SimCamera`.

### 1b. Mac emulation parity (hard requirement)

The Mac webcam path must behave **exactly** like the Pi so everything can be developed and tested
without the robot. Rules:

- **One perception code path.** Detection, tracking, recognition, events, `look()` and the
  attention policy live in `perception/` / `brain/` and never know which camera they run on.
  `SimCamera` stops doing its own detection; it only produces frames. The dashboard preview draws
  boxes from the shared `Perception` results (what you see is what the Pi would compute).
- **Same camera contract.** `SimCamera` emulates picamera2's two streams: it delivers a `lores`
  frame resized to the configured `detect_size` (320×240) and a `main` frame cropped/resized to the
  Pi sensor aspect (4:3) for snapshots. Same color order (BGR at the boundary), same orientation
  handling (flip/rotate from config).
- **Pi timing profile.** `camera.pi_profile: true` (default on Mac) caps the capture/detect rate to
  the Pi's measured fps and runs T1/T2 at their Pi rates, so debouncing, cooldowns and "how fast does
  it react" feel identical. Turn it off only to debug at full speed.
- **Same models and thresholds.** Identical ONNX files from `make setup-vision`, identical
  `perception:` config; no Mac-only accelerators (CoreML etc.) that would change results.
- **Shared faces DB.** Enrollment on the Mac writes the same `faces` table, so people enrolled
  while developing are recognized on the Pi (and vice versa, by copying the DB).
- **Replay source.** `camera.backend: file` plays a recorded clip/frame folder through the same
  contract — for repeatable tests and for re-running something recorded on the Pi.
- **Parity check.** A test runs the same recorded frames through the pipeline with the `webcam`
  emulation and the `picamera2` frame shapes and asserts identical events.

The existing PiCrawler-on-Mac fallback (SoftPicrawler + `SimCamera`) keeps working unchanged.

### 2. Perception tiers

| Tier | Rate | Model | Cost on Pi 4 (est.) | Output |
|---|---|---|---|---|
| T0 | ~15 fps | OpenCV `FaceDetectorYN` (YuNet, 320×240) | ~10–20 ms/frame | boxes + landmarks → tracker |
| T1 | on new/uncertain track | OpenCV `FaceRecognizerSF` (SFace, 128-d) | ~30–50 ms/face | identity + confidence |
| T2 | ~1 Hz | YOLO-nano (ONNX Runtime or NCNN), cat/person/few objects | ~200–400 ms/frame | `cat_seen`, object labels |

- YuNet replaces the Haar face cascade (more robust to angle/light, gives landmarks for SFace alignment).
  Keep the Haar cat-face cascade until T2 lands, then drop it.
- Simple IoU track IDs so recognition runs once per track, re-checked every ~5 s or when confidence is low.
- Matching: cosine similarity vs. stored embeddings (several per person); threshold in config;
  below threshold → `unknown`.
- Model files (`.onnx`) downloaded by a `make setup-vision` target into `data/models/`, not committed.
- Same pipeline on Mac (webcam); see 1b.

### 3. Events → LLM (text)

New bus topics, debounced (a person must be present/absent for N s before an event fires):

- `person_arrived {name | "unknown", track}` / `person_left {name}`
- `cat_seen` / `cat_left`
- (`face_seen` / `face_lost` stay for reflexes/back-compat)

Logged via `memory.log_event`, so they already reach the voice prompt through `recent_events`.
Also expose a compact "currently in view" line in the voice state (e.g. `in view: Jens, cat`).

### 4. `look(question?)` tool

- Voice tool: grab `snapshot_jpeg()`, send to the same OpenAI-compatible endpoint as an
  `image_url` data URI with a short prompt (+ optional question and who's in view), return a
  1–3 sentence description as the tool result.
- Separate call (not injecting the image into the voice conversation) — keeps the Pipecat
  context text-only and lets us pick a different vision model via `vision.model`.
- Latency ~1–2 s: play a "thinking"/peering emote while waiting; timeout → graceful "can't see well".

### 5. Recognition enrollment

- **Conversational:** `remember_face(name)` tool — the LLM calls it when someone introduces
  themselves; stores embeddings of the largest face in view (collect ~5 over 2 s for robustness).
  `forget_face(name)` for removal.
- **Script:** `scripts/enroll_face.py --name Anna` — captures N frames, stores embeddings;
  `--list`, `--delete`.
- Storage: new `faces` table in the SQLite `MemoryStore` (`name`, `embedding BLOB`, `created_at`),
  name links to existing facts so "Anna likes tea" and Anna's face are the same person.

### 6. Attention policy (proactive behavior)

A small component subscribed to perception events, deciding *whether* to act:

- Greet a known person on `person_arrived` at most once per `greet_cooldown_h` (default 3 h).
- Unknown face: curious emote; optionally ask for a name at most once per session.
- Cat: occasional comment (probability + `cat_comment_cooldown_min`).
- **Never** speaks while the user is talking, a turn is in flight, or TTS is playing; then
  it falls back to non-verbal reactions (emote, look, small gesture).
- Proactive speech goes through the voice session as a synthetic turn (LLM writes the line, so
  personality stays consistent); needs a small `VoiceSession.inject_event(text)` entry point.

## Config

Phase 1 config lives in `config/default.yaml` under `camera:` and `perception:`.
Later phases add, under `perception:`, `recognize`, `match_threshold`, `objects`,
`objects_hz` and `vision_model`. They also add an `attention:` block with `proactive`,
`greet_cooldown_h` and `cat_comment_cooldown_min`.

## Phases

1. **Pi camera + YuNet tracking** — `PiCamera` (picamera2), `snapshot_jpeg`, YuNet in
   `perception/`, tracker unchanged in behavior; move detection out of `SimCamera`, add the
   Pi timing profile and the `file` replay backend. *Done when:* eyes follow a face on the Mac
   and on the Pi with the same code; measure Pi fps, CPU and temperature with the voice pipeline
   running and feed those numbers into `pi_profile`.
2. **Recognition + enrollment** — track IDs, SFace, `faces` table, `remember_face` tool,
   enroll script, `person_arrived/left` events. *Done when:* it says your name after you walk in.
3. **`look()` tool** — snapshot → VLM → text. *Done when:* "what am I holding?" works end to end.
4. **Attention policy** — proactive greetings/comments with cooldowns and turn-safety.
5. **Object/cat detection (T2)** — YOLO-nano; retire Haar cat cascade.

## Risks / open questions

- **Pi 4 thermals/CPU** with T0 + voice pipeline: fallback is T0 at 8 fps while speaking/listening,
  and a heatsink/fan. Measure in phase 1.
- **picamera2 on the robot OS**: needs libcamera (Bookworm); check the SunFounder image ships it,
  and that camera orientation/mounting matches (flip in config).
- **Recognition accuracy** in living-room lighting at 320×240 — may need to crop the face from
  the `main` stream for SFace rather than the lores frame.
- **Proactive annoyance** — tune cooldowns in practice; add a "be quiet" voice command that
  disables proactive speech for a while.
- **Testing** — unit tests with recorded frames/fixtures for tracker, matching, debouncing and
  the attention policy; no camera needed in CI.
