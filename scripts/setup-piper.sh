#!/usr/bin/env bash
# Install piper-tts, download a (German) voice, start the Piper HTTP server, and self-test it.
# This venv is separate from the app. shikigami posts JSON to the server and never
# imports piper-tts, so the GPL-3.0 package stays in this process.
# Run this on jesktop (as planned) or locally on the Mac — then point shikigami at it:
# TTS_PROVIDER=piper-http, TTS_BASE_URL=http://jesktop:5000 (or http://127.0.0.1:5000).
# macOS AirPlay Receiver already binds port 5000; pass another port if `uv run sim` is local.
#
# Usage: scripts/setup-piper.sh [voice] [port]
#   voice: e.g. de_DE-thorsten-medium (default), de_DE-ramona-medium, de_DE-karlsson-low (small/fast)
#   port:  5000 (default)
# Env:   PIPER_CUDA=1 to use the GPU (--cuda), PIPER_VENV / PIPER_VOICES_DIR to override paths.
set -euo pipefail

VOICE="${1:-de_DE-thorsten-medium}"
PORT="${2:-5000}"
VENV="${PIPER_VENV:-$HOME/piper-venv}"
VOICES_DIR="${PIPER_VOICES_DIR:-$HOME/piper-voices}"
BASE_URL="http://localhost:${PORT}"

step() { printf '\n==> %s\n' "$*"; }

step "creating venv at $VENV"
python3 -m venv "$VENV"

step "installing piper-tts[http] (pulls onnxruntime — big download, progress shown; can take minutes)"
# [http] extra = flask, needed for `piper.http_server`
"$VENV/bin/pip" install --disable-pip-version-check --upgrade "piper-tts[http]>=1.3"

# voice files on HF: <lang>/<locale>/<speaker>/<quality>/<voice>.onnx[.json]
# e.g. de_DE-thorsten-medium -> de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx
quality="${VOICE##*-}"   # medium
base="${VOICE%-*}"       # de_DE-thorsten
locale="${base%%-*}"     # de_DE
speaker="${base#*-}"     # thorsten
lang="${locale%%_*}"     # de
prefix="https://huggingface.co/rhasspy/piper-voices/resolve/main/$lang/$locale/$speaker/$quality"

step "voice $VOICE -> $VOICES_DIR"
mkdir -p "$VOICES_DIR"
for f in "$VOICE.onnx" "$VOICE.onnx.json"; do
  if [ ! -s "$VOICES_DIR/$f" ]; then
    echo "downloading $prefix/$f"
    curl -L --fail --progress-bar -o "$VOICES_DIR/$f" "$prefix/$f"
  else
    echo "already have $VOICES_DIR/$f"
  fi
done

CUDA_FLAG=""
if [ "${PIPER_CUDA:-0}" = "1" ]; then
  CUDA_FLAG="--cuda"
fi

LOG="$VENV/server.log"
step "starting piper http server on port $PORT (cuda=${PIPER_CUDA:-0}, log: $LOG)"
# endpoint: POST /synthesize {"text": ...} (piper >= 1.8) or /api/tts (1.3.x)
"$VENV/bin/python" -m piper.http_server \
  --model "$VOICES_DIR/$VOICE.onnx" --data-dir "$VOICES_DIR" --port "$PORT" $CUDA_FLAG >"$LOG" 2>&1 &
SERVER_PID=$!

step "waiting for the server to accept requests on $BASE_URL"
up=0
for _ in $(seq 1 60); do
  if curl -sf -o /dev/null "$BASE_URL/"; then up=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server process died — last log lines:" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 1
done
if [ "$up" != "1" ]; then
  echo "server did not come up within 60s — last log lines:" >&2
  tail -20 "$LOG" >&2
  exit 1
fi
echo "server is up (pid $SERVER_PID)"

step "self-test: POST a German sentence, expect WAV bytes"
OUT="$VOICES_DIR/selftest.wav"
code=$(curl -s -o "$OUT" -w "%{http_code}" -X POST "$BASE_URL/synthesize" \
  -H 'Content-Type: application/json' -d '{"text":"Hallo, ich bin Shikigami!"}')
if [ "$code" = "404" ]; then  # piper 1.3.x endpoint
  code=$(curl -s -o "$OUT" -w "%{http_code}" -X POST "$BASE_URL/api/tts" \
    -H 'Content-Type: application/json' -d '{"text":"Hallo, ich bin Shikigami!"}')
fi
size=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
echo "HTTP $code, $size bytes -> $OUT"
if [ "$code" != "200" ]; then
  echo "synthesis failed — last log lines:" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

echo
echo "OK — piper is running on $BASE_URL (pid $SERVER_PID)"
echo "shikigami config: TTS_PROVIDER=piper-http and TTS_BASE_URL=http://<this-host>:${PORT}"
echo "logs: tail -f $LOG    stop: kill $SERVER_PID    (Ctrl-C also stops it)"
wait "$SERVER_PID"
