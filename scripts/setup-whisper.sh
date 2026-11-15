#!/usr/bin/env bash
# Install faster-whisper-server, download a model, start the OpenAI-compatible
# STT API, and self-test it.
# Run this on jesktop (as planned) or locally — then point shikigami at it:
#   STT_BASE_URL=http://jesktop:8000/v1
#   STT_MODEL=Systran/faster-whisper-medium
#   STT_API_KEY=not-needed
#
# Usage: scripts/setup-whisper.sh [model] [port]
#   model: HuggingFace / faster-whisper id (default: Systran/faster-whisper-medium)
#   port:  8000 (default; piper uses 5000)
# Env:   WHISPER_CUDA=0 to force CPU (CUDA_VISIBLE_DEVICES=). WHISPER_VENV /
#        WHISPER_DATA_DIR to override paths. Optional HF_TOKEN for Hub rate limits.
set -euo pipefail

MODEL="${1:-Systran/faster-whisper-medium}"
PORT="${2:-8000}"
VENV="${WHISPER_VENV:-$HOME/whisper-venv}"
BASE_URL="http://localhost:${PORT}"
DATA_DIR="${WHISPER_DATA_DIR:-$HOME/whisper-data}"
SERVER_PID=""

step() { printf '\n==> %s\n' "$*"; }

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap 'ec=$?; if [ "$ec" -ne 0 ]; then cleanup; fi' EXIT

step "creating venv at $VENV"
python3 -m venv "$VENV"

step "installing faster-whisper-server (pulls faster-whisper / ctranslate2 — can take minutes)"
"$VENV/bin/pip" install --disable-pip-version-check --upgrade "faster-whisper-server>=0.0.3"

mkdir -p "$DATA_DIR"
LOG="$VENV/server.log"

# faster-whisper-server 0.0.3 crashes on first request if the model is not already
# in the HF cache (buggy logger.info in _log_download_if_needed). Pre-download
# (and warm-load) so the self-test never hits that path.
step "pre-downloading model $MODEL into the HuggingFace cache (first run can take a while)"
MODEL="$MODEL" "$VENV/bin/python" - <<'PY'
import os
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

model_id = os.environ["MODEL"]
print("downloading", model_id)
path = download_model(model_id)
print("cached at", path)
print("warm-loading WhisperModel (device=auto)…")
WhisperModel(model_id)  # device=auto → CUDA when available
print("model ready")
PY

# Single-model mode ignores the request "model" form field and always uses this path.
if [ "${WHISPER_CUDA:-1}" = "0" ]; then
  export CUDA_VISIBLE_DEVICES=""
  step "starting whisper server on port $PORT (cpu forced, model=$MODEL, log: $LOG)"
else
  step "starting whisper server on port $PORT (cuda=${WHISPER_CUDA:-auto}, model=$MODEL, log: $LOG)"
fi

"$VENV/bin/faster-whisper-server" "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --log-level info >"$LOG" 2>&1 &
SERVER_PID=$!

step "waiting for the server to accept requests on $BASE_URL"
up=0
for _ in $(seq 1 180); do
  if curl -sf -o /dev/null "$BASE_URL/health"; then up=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server process died — last log lines:" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  sleep 1
done
if [ "$up" != "1" ]; then
  echo "server did not come up within 180s — last log lines:" >&2
  tail -40 "$LOG" >&2
  exit 1
fi
echo "server is up (pid $SERVER_PID)"

step "self-test: generate a short WAV and POST /v1/audio/transcriptions"
OUT="$DATA_DIR/selftest.wav"
"$VENV/bin/python" - <<'PY' "$OUT"
import sys, wave, struct, math
path = sys.argv[1]
sr, dur, f0 = 16000, 0.6, 440.0
n = int(sr * dur)
with wave.open(path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    for i in range(n):
        sample = int(12000 * math.sin(2 * math.pi * f0 * i / sr))
        w.writeframes(struct.pack("<h", sample))
print(path)
PY

RESP="$DATA_DIR/selftest.json"
code=$(curl -s -o "$RESP" -w "%{http_code}" -X POST "$BASE_URL/v1/audio/transcriptions" \
  -F "file=@${OUT}" \
  -F "model=${MODEL}" \
  -F "language=de" \
  -F "response_format=json")
echo "HTTP $code -> $RESP"
if [ "$code" != "200" ]; then
  echo "transcription failed — body / last log lines:" >&2
  cat "$RESP" >&2 || true
  tail -40 "$LOG" >&2
  exit 1
fi
echo "response: $(head -c 200 "$RESP")"

# success — keep the server running; clear the failure-only trap cleanup
trap - EXIT

echo
echo "OK — whisper is running on $BASE_URL (pid $SERVER_PID)"
echo "shikigami config:"
echo "  STT_BASE_URL=http://<this-host>:${PORT}/v1"
echo "  STT_MODEL=${MODEL}"
echo "  STT_API_KEY=not-needed"
echo "logs: tail -f $LOG    stop: kill $SERVER_PID    (Ctrl-C also stops it)"
wait "$SERVER_PID"
