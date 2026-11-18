#!/usr/bin/env bash
# Download the vision models into data/models (same files on Mac and Pi).
#   YuNet face detector (OpenCV Zoo, MIT) — perception.face_detector: yunet
#   SFace face recognizer (OpenCV Zoo, Apache-2.0) — perception.recognize: true (~37 MB)
# Without it perception falls back to Haar cascades and logs a warning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:-$ROOT/data/models}"
mkdir -p "$DIR"

# OpenCV Zoo stores models in Git LFS; media.githubusercontent.com serves the real file.
ZOO="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "ok   $(basename "$out") (already there)"
    return
  fi
  echo "get  $(basename "$out")"
  curl -fL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

fetch "$ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  "$DIR/face_detection_yunet_2023mar.onnx"
fetch "$ZOO/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
  "$DIR/face_recognition_sface_2021dec.onnx"

# Self-test: the model loads in this project's OpenCV.
cd "$ROOT"
uv run python - "$DIR" <<'EOF'
import sys
from pathlib import Path

import cv2
import numpy as np

model = Path(sys.argv[1]) / "face_detection_yunet_2023mar.onnx"
det = cv2.FaceDetectorYN.create(str(model), "", (320, 240), 0.7, 0.3, 50)
det.detect(np.zeros((240, 320, 3), dtype=np.uint8))
print(f"ok   YuNet loads (OpenCV {cv2.__version__})")
rec = cv2.FaceRecognizerSF.create(str(Path(sys.argv[1]) / "face_recognition_sface_2021dec.onnx"), "")
feat = rec.feature(np.zeros((112, 112, 3), dtype=np.uint8))
assert feat.shape == (1, 128), feat.shape
print("ok   SFace loads (128-d embeddings)")
EOF
