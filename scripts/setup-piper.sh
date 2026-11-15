#!/usr/bin/env bash
# Install piper-tts, download a (German) voice, and start the Piper HTTP server.
# Run this on jesktop (as planned) or locally on the Mac — then point
# TTS_BASE_URL at it (http://jesktop:5000 or http://localhost:5000).
#
# Usage: scripts/setup-piper.sh [voice] [port]
#   voice: e.g. de_DE-thorsten-medium (default), de_DE-ramona-medium, de_DE-karlsson-low
#   port:  5000 (default)
set -euo pipefail

VOICE="${1:-de_DE-thorsten-medium}"
PORT="${2:-5000}"
VENV="${PIPER_VENV:-$HOME/piper-venv}"
VOICES_DIR="${PIPER_VOICES_DIR:-$HOME/piper-voices}"

python3 -m venv "$VENV"
# [http] extra = flask, needed for `piper.http_server`; onnxruntime is pulled in automatically
# (GPU on jesktop: set PIPER_CUDA=1, or pip install onnxruntime-gpu instead of onnxruntime)
"$VENV/bin/pip" install -q --upgrade "piper-tts[http]>=1.3"

# voice files on HF: <lang>/<locale>/<speaker>/<quality>/<voice>.onnx[.json]
# e.g. de_DE-thorsten-medium -> de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx
quality="${VOICE##*-}"          # medium
base="${VOICE%-*}"              # de_DE-thorsten
locale="${base%%-*}"            # de_DE
speaker="${base#*-}"            # thorsten
lang="${locale%%_*}"            # de
prefix="https://huggingface.co/rhasspy/piper-voices/resolve/main/$lang/$locale/$speaker/$quality"

mkdir -p "$VOICES_DIR"
for f in "$VOICE.onnx" "$VOICE.onnx.json"; do
  if [ ! -s "$VOICES_DIR/$f" ]; then
    echo "downloading $prefix/$f"
    curl -sSL --fail -o "$VOICES_DIR/$f" "$prefix/$f"
  fi
done

CUDA_FLAG=""
if [ "${PIPER_CUDA:-0}" = "1" ]; then
  CUDA_FLAG="--cuda"
fi

echo "starting piper http server: voice=$VOICE port=$PORT cuda=${PIPER_CUDA:-0}"
# endpoint: POST /synthesize {"text": ...} (piper >= 1.8) or /api/tts (1.3.x) — client auto-detects
exec "$VENV/bin/python" -m piper.http_server \
  --model "$VOICES_DIR/$VOICE.onnx" --data-dir "$VOICES_DIR" --port "$PORT" $CUDA_FLAG
