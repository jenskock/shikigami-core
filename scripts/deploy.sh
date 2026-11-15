#!/usr/bin/env bash
# Placeholder deploy: rsync the code to the Raspberry Pi and optionally run it there.
# Configure via .env: DEPLOY_HOST (default pi@raspberrypi.local).
# Usage: make deploy            # copy only
#        DEPLOY_RUN=1 make deploy  # copy + start on the Pi
set -euo pipefail

HOST="${DEPLOY_HOST:-pi@raspberrypi.local}"
REMOTE_DIR="${DEPLOY_REMOTE_DIR:-/home/pi/shikigami-core}"
BACKEND="${SHIKIGAMI_BACKEND:-pidog}"

cd "$(dirname "$0")/.."

echo "==> rsync to ${HOST}:${REMOTE_DIR}"
rsync -avz --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.env' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude 'data' \
  ./ "${HOST}:${REMOTE_DIR}/"

echo "==> deployed. On the Pi:"
echo "    ssh ${HOST} 'cd ${REMOTE_DIR} && uv sync && SHIKIGAMI_BACKEND=${BACKEND} uv run shikigami'"

if [[ "${DEPLOY_RUN:-0}" == "1" ]]; then
  echo "==> starting on ${HOST} (backend=${BACKEND})"
  ssh "${HOST}" "cd ${REMOTE_DIR} && uv sync && SHIKIGAMI_BACKEND=${BACKEND} uv run shikigami"
fi
