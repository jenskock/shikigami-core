"""Learn, list or delete faces for recognition — without talking to the robot.

Uses the same camera, detector and SFace model as the running robot (Mac webcam,
Pi camera or a replay file) and writes into the same memory database, so a face
enrolled on the Mac is recognized on the Pi after copying data/memory.sqlite3.

  uv run python scripts/enroll_face.py --name Anna          # look at the camera
  uv run python scripts/enroll_face.py --name Anna --samples 10
  uv run python scripts/enroll_face.py --list
  uv run python scripts/enroll_face.py --delete Anna
  uv run python scripts/enroll_face.py --name Anna --camera-file anna.mp4

Stop the robot first: the camera can only be opened by one process.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from shikigami.config import REPO_ROOT, load_config
from shikigami.hal.cameras import make_camera
from shikigami.hal.picamera import picamera2_available
from shikigami.memory.store import MemoryStore
from shikigami.perception import Perception, build_detector
from shikigami.perception.people import PeopleSettings, PeopleTracker
from shikigami.perception.recognition import MAX_PER_NAME, FaceGallery, build_recognizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="learn the face in view under this name")
    g.add_argument("--list", action="store_true", help="list known faces")
    g.add_argument("--delete", metavar="NAME", help="forget a face")
    p.add_argument("--samples", type=int, default=None, help="embeddings to take (default: config)")
    p.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for a clear face")
    p.add_argument("--camera-file", default=None, help="enroll from a video / image folder")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


async def enroll(cfg, memory: MemoryStore, name: str, samples: int, timeout: float) -> int:
    log = logging.getLogger("enroll")
    recognizer = build_recognizer(cfg.perception, REPO_ROOT, log)
    if recognizer is None:
        print("face recognition unavailable — run `make setup-vision` (needs yunet + sface)")
        return 1
    cfg.sim.camera_preview = False
    camera = make_camera(cfg, on_robot=picamera2_available() and cfg.backend != "sim", log=log)
    await camera.start()
    gallery = FaceGallery.from_rows(memory.faces())
    perception = Perception(
        camera,
        build_detector(cfg.perception, REPO_ROOT, log),
        detect_hz=cfg.perception.detect_hz,
        log=log,
        people=PeopleTracker(recognizer, gallery, PeopleSettings.from_config(cfg.perception)),
    )
    perception.start()
    try:
        print(f"Look at the camera, {name} — taking {samples} samples…")
        req = await perception.enroll(samples, timeout)
    finally:
        perception.stop()
        await camera.stop()
    if not req.ok:
        print(f"failed: {req.reason or 'no clear face'} ({len(req.embeddings)}/{samples} samples)")
        return 1
    # Warn if this face already looks like someone else (wrong name, or a lookalike).
    other, score = gallery.match(sum(req.embeddings))
    if other and other.lower() != name.lower() and score >= cfg.perception.match_threshold:
        print(f"note: this face also matches {other!r} ({score:.2f})")
    for emb in req.embeddings:
        memory.add_face(name, emb.astype("float32").tobytes())
    memory.prune_faces(name, MAX_PER_NAME)
    print(f"learned {name} ({len(req.embeddings)} samples)")
    return 0


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.camera_file:
        cfg.camera.backend = "file"
        cfg.camera.file = args.camera_file
    memory = MemoryStore(REPO_ROOT / cfg.memory.path)
    try:
        if args.list:
            counts: dict[str, int] = {}
            for n, _ in memory.faces():
                counts[n] = counts.get(n, 0) + 1
            for n, c in sorted(counts.items()):
                print(f"{n}: {c} samples")
            if not counts:
                print("no faces learned yet")
            return 0
        if args.delete:
            removed = memory.forget_face(args.delete)
            print(f"forgot {args.delete} ({removed} samples)" if removed else "no such face")
            return 0 if removed else 1
        samples = args.samples or cfg.perception.enroll_samples
        return asyncio.run(enroll(cfg, memory, args.name.strip(), samples, args.timeout))
    finally:
        memory.close()


if __name__ == "__main__":
    sys.exit(main())
