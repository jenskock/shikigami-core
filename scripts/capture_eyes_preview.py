#!/usr/bin/env python3
"""Capture a short eyes preview strip (PNG frames) for README docs.

Usage:
    uv run python scripts/capture_eyes_preview.py
    # writes docs/eyes_preview.png (contact sheet of emotions)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shikigami.expression.eyes import EyesEngine, EMOTION_NAMES  # noqa: E402
from shikigami.expression.eyes.render.numpy_draw import draw_frame_rgb  # noqa: E402
from shikigami.expression.eyes.state import EyesConfig  # noqa: E402


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("pillow required: uv add pillow", file=sys.stderr)
        return 1

    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    size = 160
    cols = 4
    names = list(EMOTION_NAMES)
    rows = (len(names) + cols - 1) // cols
    sheet = np.zeros((rows * size, cols * size, 3), dtype=np.uint8)

    for i, name in enumerate(names):
        eng = EyesEngine(EyesConfig(seed=7, personality="curious"))
        eng.set_emotion(name, intensity=1.0)
        # settle springs
        for _ in range(45):
            eng.update(1 / 60)
        frame = eng.frame
        rgb = draw_frame_rgb(frame, size, size, circular=True)
        r, c = divmod(i, cols)
        sheet[r * size : (r + 1) * size, c * size : (c + 1) * size] = rgb

    path = out_dir / "eyes_preview.png"
    Image.fromarray(sheet, "RGB").save(path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
