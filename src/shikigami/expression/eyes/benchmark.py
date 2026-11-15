"""Benchmark PIL/GC9A01 renderer FPS: `uv run eyes-bench`."""

from __future__ import annotations

import statistics
import time

from .controller import EyesEngine
from .render.pil_renderer import PILRenderer
from .state import EyesConfig


def main(frames: int = 300, size: int = 240) -> int:
    engine = EyesEngine(EyesConfig(seed=42, personality="curious"))
    renderer = PILRenderer(size=size, circular=True)
    # warm-up
    for i in range(30):
        engine.look_at((i % 20 - 10) / 10.0, 0.0)
        if i % 40 == 0:
            engine.blink()
        frame = engine.update(1 / 60)
        renderer.draw_rgb565(frame)

    dts: list[float] = []
    t0 = time.perf_counter()
    for i in range(frames):
        engine.look_at(((i * 3) % 40 - 20) / 20.0, ((i * 2) % 30 - 15) / 30.0)
        if i % 35 == 0:
            engine.set_emotion(["happy", "curious", "surprised", "sleepy"][i // 35 % 4])
        if i % 50 == 0:
            engine.blink()
        t1 = time.perf_counter()
        frame = engine.update(1 / 60)
        renderer.draw_rgb565(frame)
        dts.append(time.perf_counter() - t1)
    elapsed = time.perf_counter() - t0
    fps = frames / elapsed
    mean_ms = statistics.mean(dts) * 1000
    p95_ms = statistics.quantiles(dts, n=20)[18] * 1000  # ~95th
    print(f"eyes-bench size={size} frames={frames}")
    print(f"  throughput: {fps:.1f} fps  ({elapsed:.3f}s)")
    print(f"  frame time: mean={mean_ms:.2f} ms  p95={p95_ms:.2f} ms")
    print(f"  target: ≥30 fps on Pi 3B+  →  {'PASS' if fps >= 30 else 'CHECK ON DEVICE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
