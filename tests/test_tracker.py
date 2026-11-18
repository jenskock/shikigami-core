"""FaceTracker: newest perception result -> look_at + face_seen; disappearance -> face_lost."""

import asyncio
import time
from types import SimpleNamespace

from shikigami.config import PerceptionConfig
from shikigami.perception.detectors import Detection
from shikigami.perception.pipeline import PerceptionResult
from shikigami.perception.tracker import STALE_S, FaceTracker


class FakePerception:
    def __init__(self):
        self.result = None
        self._seq = 0

    def set(self, *dets, t=None):
        self._seq += 1
        self.result = PerceptionResult(
            seq=self._seq, t=time.monotonic() if t is None else t, detections=tuple(dets)
        )

    def latest(self):
        return self.result


def _make():
    calls = {"look": [], "emote": [], "events": []}

    async def publish(topic, payload=None):
        calls["events"].append((topic, payload or {}))

    async def look_at(x, y):
        calls["look"].append((x, y))

    async def emote(name):
        calls["emote"].append(name)

    perception = FakePerception()
    tracker = FaceTracker(
        perception,
        SimpleNamespace(look_at=look_at),
        SimpleNamespace(emote=emote),
        publish,
        PerceptionConfig(track_faces=True, face_cooldown_s=0.0),
    )
    return tracker, perception, calls


FACE = Detection("face", 300 / 640, 200 / 480, 40 / 640, 40 / 480)


def test_face_seen_drives_look_at_and_event():
    tracker, perception, calls = _make()
    perception.set(FACE, Detection("cat", 0.01, 0.01, 0.03, 0.04))
    asyncio.run(tracker.poll())
    # face center: x=320/640 -> 0.0, y=220/480 -> -(2*220/480-1) (inverted)
    assert len(calls["look"]) == 1
    lx, ly = calls["look"][0]
    assert abs(lx) < 1e-6 and abs(ly - (1 - 2 * 220 / 480)) < 1e-6
    assert ("face_seen", {"x": 0.0, "y": round(1 - 2 * 220 / 480, 2), "faces": 1}) in calls[
        "events"
    ]
    assert calls["emote"] == ["curious"]


def test_same_result_is_not_reprocessed():
    tracker, perception, calls = _make()
    perception.set(FACE)

    async def main():
        await tracker.poll()
        await tracker.poll()

    asyncio.run(main())
    assert len(calls["look"]) == 1


def test_face_lost_event_after_disappearance():
    tracker, perception, calls = _make()

    async def main():
        perception.set(FACE)
        await tracker.poll()
        perception.set()
        await tracker.poll()

    asyncio.run(main())
    assert [t for t, _ in calls["events"]] == ["face_seen", "face_lost"]


def test_stale_results_count_as_face_lost():
    """Camera stops delivering: the last face must not stay 'present' forever."""
    tracker, perception, calls = _make()

    async def main():
        perception.set(FACE)
        await tracker.poll()
        perception.result = PerceptionResult(
            seq=perception.result.seq, t=time.monotonic() - STALE_S - 0.1, detections=(FACE,)
        )
        await tracker.poll()

    asyncio.run(main())
    assert [t for t, _ in calls["events"]] == ["face_seen", "face_lost"]


def test_no_result_is_noop():
    tracker, _perception, calls = _make()
    asyncio.run(tracker.poll())
    assert calls["look"] == [] and calls["events"] == []


def test_describe_view_states_only_what_is_detected():
    from shikigami.perception.pipeline import describe_view

    now = time.monotonic()

    def res(*dets, age=0.0):
        return PerceptionResult(seq=1, t=now - age, detections=tuple(dets))

    cat = Detection("cat", 0.1, 0.1, 0.1, 0.1)
    assert describe_view(res(FACE), now) == "1 face, who: unknown"
    assert describe_view(res(FACE, FACE, cat), now) == "2 faces, 1 cat, who: unknown"
    assert describe_view(res(cat), now) == "1 cat"
    assert describe_view(res(), now) == "nothing"
    assert describe_view(res(FACE, age=STALE_S + 0.1), now) == "camera not working"
    assert describe_view(None, now) == "camera not working"
    assert describe_view(None, now, camera_on=False) == "camera off"
