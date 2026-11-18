"""Perception: runs the detectors on the camera's low-res stream in its own thread.

One code path for Pi and Mac — it only sees ``Camera.latest()`` frames, at the
configured ``detect_hz``. Per cycle: detect on ``lores``, drop cat boxes sitting on
faces, then update people (tracks, recognition on ``main``, arrive/leave events) and
feed a pending face enrollment. Consumers read ``latest()`` / ``drain_events()``
without blocking.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..hal.interfaces import Camera, Frame
from .detectors import Detection, Detector
from .people import PeopleTracker, PersonView, largest_face

CAT_STABLE_FRAMES = 3  # a cat must be detected this many cycles in a row to count
ENROLL_SPACING_S = 0.15  # min time between enrollment samples (different frames/poses)


@dataclass(frozen=True)
class PerceptionResult:
    seq: int  # camera frame seq this came from
    t: float  # capture time (monotonic)
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    detect_ms: float = 0.0
    people: tuple[PersonView, ...] = field(default_factory=tuple)
    cats_in_view: int | None = None  # stable cats; None = count raw cat detections

    def of(self, label: str) -> list[Detection]:
        return [d for d in self.detections if d.label == label]

    def preview_detections(self) -> list[Detection]:
        """Boxes for the preview: faces labeled with who they are."""
        if not self.people:
            return list(self.detections)
        return [p.as_detection() for p in self.people] + [
            d for d in self.detections if d.label != "face"
        ]


STALE_S = 1.0  # a result older than this says nothing about now


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def describe_view(res: PerceptionResult | None, now: float, camera_on: bool = True) -> str:
    """What the camera detects right now, as the LLM may state it. Never more than this."""
    if not camera_on:
        return "camera off"
    if res is None or now - res.t > STALE_S:
        return "camera not working"
    names = [p.name for p in res.people] if res.people else [None] * len(res.of("face"))
    cats = res.cats_in_view if res.cats_in_view is not None else len(res.of("cat"))
    parts = []
    if names:
        parts.append(_plural(len(names), "face", "faces"))
    if cats:
        parts.append(_plural(cats, "cat", "cats"))
    if not parts:
        return "nothing"
    if names:
        known = [n for n in names if n]
        unknown = len(names) - len(known)
        who = known + ([f"{unknown} unknown" if known else "unknown"] if unknown else [])
        parts.append("who: " + " + ".join(who))
    return ", ".join(parts)


def suppress_cats_on_faces(dets: list[Detection]) -> list[Detection]:
    """Haar cat cascades fire on human faces: drop cat boxes that overlap a face."""
    faces = [d for d in dets if d.label == "face"]
    if not faces:
        return dets
    out = []
    for d in dets:
        if d.label == "cat" and any(_overlap(d, f) > 0.3 for f in faces):
            continue
        out.append(d)
    return out


def _overlap(a: Detection, b: Detection) -> float:
    """Intersection as a fraction of ``a``."""
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / a.area if a.area > 0 else 0.0


@dataclass
class EnrollRequest:
    samples: int
    deadline: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    track_id: int | None = None
    reason: str | None = None  # why it failed / why the last frame was skipped
    last_t: float = float("-inf")
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def ok(self) -> bool:
        return len(self.embeddings) >= self.samples


class Perception:
    def __init__(
        self,
        camera: Camera,
        detector: Detector,
        detect_hz: float = 10.0,
        log: logging.Logger | None = None,
        people: PeopleTracker | None = None,
    ) -> None:
        self._camera = camera
        self.detector = detector
        self.people = people
        self._events: collections.deque[tuple[str, dict]] = collections.deque(maxlen=100)
        self._enroll: EnrollRequest | None = None
        self._cat_streak = 0
        self._people_lock = threading.Lock()
        self._period = 1.0 / detect_hz if detect_hz > 0 else 0.0
        self._log = log or logging.getLogger("shikigami.perception")
        self._lock = threading.Lock()
        self._latest: PerceptionResult | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.stats = {"frames": 0, "detect_ms_avg": 0.0}

    def process(self, frame: Frame, now: float | None = None) -> PerceptionResult:
        """One cycle on one frame; publishes the result (synchronous; used by the thread)."""
        now = time.monotonic() if now is None else now
        t0 = time.perf_counter()
        dets = suppress_cats_on_faces(list(self.detector.detect(frame.lores)))
        ms = (time.perf_counter() - t0) * 1000
        faces = [d for d in dets if d.label == "face"]
        cats = [d for d in dets if d.label == "cat"]
        self._cat_streak = self._cat_streak + 1 if cats else 0
        people: list[PersonView] = []
        events: list[tuple[str, dict]] = []
        # People + enrollment have their own lock: recognition can take ~50 ms on the Pi
        # and must not block readers of latest() / drain_events() on the asyncio side.
        with self._people_lock:
            if self.people is not None:
                people, events = self.people.update(faces, frame.main, now)
            self._feed_enroll(faces, people, frame, now)
        res = PerceptionResult(
            seq=frame.seq,
            t=frame.t,
            detections=tuple(dets),
            detect_ms=ms,
            people=tuple(people),
            cats_in_view=len(cats) if self._cat_streak >= CAT_STABLE_FRAMES else 0,
        )
        with self._lock:
            self._latest = res
            self._events.extend(events)
        n = self.stats["frames"] = self.stats["frames"] + 1
        avg = self.stats["detect_ms_avg"]
        self.stats["detect_ms_avg"] = avg + (ms - avg) / min(n, 50)
        return res

    # -- events -----------------------------------------------------------------------

    def drain_events(self) -> list[tuple[str, dict]]:
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    # -- enrollment -------------------------------------------------------------------

    def request_enroll(self, samples: int = 5, timeout: float = 4.0) -> EnrollRequest:
        req = EnrollRequest(samples=samples, deadline=time.monotonic() + timeout)
        with self._people_lock:
            if self._enroll is not None:
                self._enroll.reason = "replaced by a newer request"
                self._enroll.done.set()
            if self.people is None or self.people.embedder is None:
                req.reason = "face recognition is not available (run make setup-vision)"
                req.done.set()
            else:
                self._enroll = req
        return req

    async def enroll(self, samples: int = 5, timeout: float = 4.0) -> EnrollRequest:
        """Collect ``samples`` embeddings of the one clear face in view."""
        req = self.request_enroll(samples, timeout)
        await asyncio.to_thread(req.done.wait, timeout + 1.0)
        with self._people_lock:
            if self._enroll is req:
                self._enroll = None
        if not req.done.is_set():
            req.reason = req.reason or "camera not delivering frames"
        return req

    def name_track(self, track_id: int, name: str) -> None:
        with self._people_lock:
            if self.people is not None:
                self.people.name_track(track_id, name, time.monotonic())

    def forget(self, name: str) -> None:
        with self._people_lock:
            if self.people is not None:
                self.people.forget(name)

    def _feed_enroll(
        self, faces: list[Detection], people: list[PersonView], frame: Frame, now: float
    ) -> None:
        req = self._enroll
        if req is None:
            return
        if now > req.deadline:
            req.reason = req.reason or "no clear face"
            self._enroll = None
            req.done.set()
            return
        if now - req.last_t < ENROLL_SPACING_S:
            return
        face, reason = largest_face(faces)
        if face is None:
            req.reason = reason
            return
        assert self.people is not None and self.people.embedder is not None
        emb = self.people.embedder.embed(frame.main, face)
        if emb is None:
            req.reason = "face could not be aligned"
            return
        req.reason, req.last_t = None, now
        req.embeddings.append(emb)
        for p in people:
            if p.box is face:
                req.track_id = p.track_id
        if req.ok:
            self._enroll = None
            req.done.set()

    def latest(self) -> PerceptionResult | None:
        with self._lock:
            return self._latest

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._thread.start()
        self._log.info(
            "perception: %s @ %.0f Hz",
            getattr(self.detector, "name", type(self.detector).__name__),
            1.0 / self._period if self._period else 0,
        )

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(1.0)
        self._thread = None

    def _run(self) -> None:
        last_seq = -1
        next_at = time.monotonic()
        last_report = time.monotonic()
        while self._running:
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(next_at - now, 0.05))
                continue
            frame: Any = self._camera.latest()
            if frame is None or frame.seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = frame.seq
            try:
                self.process(frame)
            except Exception:
                self._log.exception("detection failed")
            if self._period:
                next_at += self._period
                if next_at < now - self._period:
                    next_at = now + self._period
            if now - last_report > 60:
                last_report = now
                self._log.debug(
                    "perception: %d frames, detect %.1f ms avg",
                    self.stats["frames"],
                    self.stats["detect_ms_avg"],
                )
