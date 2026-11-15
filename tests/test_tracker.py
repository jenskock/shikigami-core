"""FaceTracker: detected face -> look_at + face_seen event; disappearance -> face_lost."""

import asyncio
from types import SimpleNamespace

import numpy as np

from shikigami.config import PerceptionConfig
from shikigami.perception.tracker import FaceTracker


class FakeCamera:
    def __init__(self, frame):
        self._frame = frame

    async def read_frame(self):
        return self._frame


def _make(detector):
    calls = {"look": [], "emote": [], "events": []}

    async def publish(topic, payload=None):
        calls["events"].append((topic, payload or {}))

    async def look_at(x, y):
        calls["look"].append((x, y))

    async def emote(name):
        calls["emote"].append(name)

    motion = SimpleNamespace(look_at=look_at)
    expr = SimpleNamespace(emote=emote)
    tracker = FaceTracker(
        FakeCamera(np.zeros((480, 640, 3), dtype=np.uint8)),
        motion,
        expr,
        publish,
        PerceptionConfig(track_faces=True, face_cooldown_s=0.0),
        detector=detector,
    )
    return tracker, calls


def test_face_seen_drives_look_at_and_event():
    detector = SimpleNamespace(
        detect_all=lambda frame: [("face", 300, 200, 40, 40), ("cat", 10, 10, 20, 20)]
    )
    tracker, calls = _make(detector)
    asyncio.run(tracker.poll())
    # face center: x=(300+20)/640*2-1=0.0, y=-((200+20)/480*2-1)=+0.0833 (inverted)
    assert len(calls["look"]) == 1
    lx, ly = calls["look"][0]
    assert abs(lx - 0.0) < 1e-6 and abs(ly - (1 - 2 * 220 / 480)) < 1e-6
    assert ("face_seen", {"x": 0.0, "y": round(1 - 2 * 220 / 480, 2), "faces": 1}) in calls[
        "events"
    ]
    assert calls["emote"] == ["curious"]


def test_face_lost_event_after_disappearance():
    state = {"boxes": [("face", 300, 200, 40, 40)]}
    detector = SimpleNamespace(detect_all=lambda frame: state["boxes"])
    tracker, calls = _make(detector)

    async def main():
        await tracker.poll()
        await asyncio.sleep(0.21)  # pass the 5 Hz poll throttle
        state["boxes"] = []
        await tracker.poll()

    asyncio.run(main())
    topics = [t for t, _ in calls["events"]]
    assert topics == ["face_seen", "face_lost"]


def test_no_frame_is_noop():
    tracker, calls = _make(SimpleNamespace(detect_all=lambda f: []))
    tracker._camera = FakeCamera(None)
    asyncio.run(tracker.poll())
    assert calls["look"] == [] and calls["events"] == []
