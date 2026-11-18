"""People: face tracks across frames, who each one is, and arrive/leave events.

Pure logic driven by the perception thread (timestamps passed in, so it is testable
and identical on Mac and Pi):

- Tracks: faces are matched frame to frame by box overlap; a track dies after
  ``lost_s`` without its face.
- Recognition: at most one embedding per update (bounded cost on the Pi). New and
  unknown tracks are checked every ``recheck_unknown_s``, known ones every
  ``recheck_known_s``. A known track only changes its name after two disagreeing
  checks in a row.
- Presence (per name, not per track, so turning away does not re-greet):
  ``person_arrived`` once a named face has been visible ``arrive_after_s``;
  ``person_left`` after the name was not seen for ``leave_after_s``. Unknown faces
  announce ``person_arrived {"name": "unknown"}`` at most every ``unknown_cooldown_s``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .detectors import Detection
from .recognition import Embedder, FaceGallery

UNKNOWN = "unknown"


def iou(a: Detection, b: Detection) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: Detection
    first_t: float
    last_t: float
    name: str | None = None
    score: float = 0.0
    attempts: int = 0
    next_check_t: float = 0.0
    doubt: str | None = None  # a different name seen once; confirmed by a second check
    announced: bool = False


@dataclass(frozen=True)
class PersonView:
    track_id: int
    name: str | None  # None = not recognized (yet)
    box: Detection
    score: float = 0.0

    def as_detection(self) -> Detection:
        """Box labeled with the name, for the preview."""
        b = self.box
        return Detection(self.name or "face", b.x, b.y, b.w, b.h, b.score, b.landmarks)


@dataclass
class PeopleSettings:
    match_threshold: float = 0.4
    recheck_unknown_s: float = 1.5
    recheck_known_s: float = 10.0
    lost_s: float = 1.0
    arrive_after_s: float = 0.6
    leave_after_s: float = 10.0
    unknown_after_attempts: int = 3
    unknown_cooldown_s: float = 60.0

    @classmethod
    def from_config(cls, cfg) -> PeopleSettings:
        return cls(
            match_threshold=cfg.match_threshold,
            recheck_unknown_s=cfg.recheck_unknown_s,
            recheck_known_s=cfg.recheck_known_s,
            arrive_after_s=cfg.arrive_after_s,
            leave_after_s=cfg.leave_after_s,
        )


@dataclass
class PeopleTracker:
    embedder: Embedder | None
    gallery: FaceGallery
    settings: PeopleSettings = field(default_factory=PeopleSettings)
    tracks: dict[int, Track] = field(default_factory=dict)
    present: dict[str, float] = field(default_factory=dict)  # name -> last seen
    _next_id: int = 1
    _last_unknown_event: float = -1e9

    def update(
        self, faces: list[Detection], image: Any, now: float
    ) -> tuple[list[PersonView], list[tuple[str, dict]]]:
        """One detection cycle. ``image`` is the frame embeddings are taken from."""
        events: list[tuple[str, dict]] = []
        seen = self._match(faces, now)
        for tid in [t for t, tr in self.tracks.items() if now - tr.last_t > self.settings.lost_s]:
            del self.tracks[tid]
        self._recognize_one(seen, image, now)
        self._presence(seen, now, events)
        views = [
            PersonView(t.id, t.name, t.box, t.score)
            for t in sorted(seen, key=lambda t: -t.box.area)
        ]
        return views, events

    def name_track(self, track_id: int, name: str, now: float) -> None:
        """Enrollment just taught us this face: name it and count it as present."""
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        tr.name, tr.score, tr.doubt = name, 1.0, None
        tr.next_check_t = now + self.settings.recheck_known_s
        tr.announced = True
        self.present[name] = now  # they are talking to us already: no extra greeting

    def forget(self, name: str) -> None:
        for tr in self.tracks.values():
            if tr.name and tr.name.lower() == name.lower():
                tr.name, tr.score, tr.attempts, tr.next_check_t = None, 0.0, 0, 0.0
        for key in [k for k in self.present if k.lower() == name.lower()]:
            del self.present[key]

    # -- internals --------------------------------------------------------------------

    def _match(self, faces: list[Detection], now: float) -> list[Track]:
        pairs = sorted(
            (
                (iou(f, t.box), i, tid)
                for i, f in enumerate(faces)
                for tid, t in self.tracks.items()
            ),
            reverse=True,
        )
        used_f: set[int] = set()
        used_t: set[int] = set()
        seen: list[Track] = []
        for score, i, tid in pairs:
            if score < 0.2:
                break
            if i in used_f or tid in used_t:
                continue
            used_f.add(i)
            used_t.add(tid)
            tr = self.tracks[tid]
            tr.box, tr.last_t = faces[i], now
            seen.append(tr)
        for i, f in enumerate(faces):
            if i in used_f:
                continue
            tr = Track(self._next_id, f, now, now)
            self._next_id += 1
            self.tracks[tr.id] = tr
            seen.append(tr)
        return seen

    def _recognize_one(self, seen: list[Track], image: Any, now: float) -> None:
        due = [t for t in seen if now >= t.next_check_t]
        if not due:
            return
        tr = min(due, key=lambda t: (t.attempts, t.next_check_t))
        tr.attempts += 1
        s = self.settings
        name, score = None, 0.0
        if self.embedder is not None and len(self.gallery) and image is not None:
            emb = self.embedder.embed(image, tr.box)
            if emb is not None:
                name, score = self.gallery.match(emb)
        if score < s.match_threshold:
            name = None
        if name is None:
            if tr.name is not None and tr.doubt is None:
                tr.doubt = UNKNOWN  # one miss (blur, turned head) does not unname
            elif tr.name is not None:
                tr.name, tr.score, tr.doubt = None, 0.0, None
        elif tr.name is None or tr.name == name or tr.doubt == name:
            tr.name, tr.score, tr.doubt = name, score, None
        else:
            tr.doubt = name
        tr.next_check_t = now + (s.recheck_known_s if tr.name else s.recheck_unknown_s)

    def _presence(self, seen: list[Track], now: float, events: list[tuple[str, dict]]) -> None:
        s = self.settings
        for tr in seen:
            if now - tr.first_t < s.arrive_after_s:
                continue
            if tr.name:
                if tr.name not in self.present:
                    events.append(("person_arrived", {"name": tr.name}))
                self.present[tr.name] = now
                tr.announced = True
            elif not tr.announced and (
                self.embedder is None
                or not len(self.gallery)
                or tr.attempts >= s.unknown_after_attempts
            ):
                tr.announced = True
                if now - self._last_unknown_event >= s.unknown_cooldown_s:
                    self._last_unknown_event = now
                    events.append(("person_arrived", {"name": UNKNOWN}))
        for name in [n for n, t in self.present.items() if now - t > s.leave_after_s]:
            del self.present[name]
            events.append(("person_left", {"name": name}))


def largest_face(faces: list[Detection]) -> tuple[Detection | None, str | None]:
    """The face to enroll, or a reason why none is clear enough."""
    if not faces:
        return None, "no face in view"
    faces = sorted(faces, key=lambda d: -d.area)
    if len(faces) > 1 and faces[1].area > 0.5 * faces[0].area:
        return None, "several faces in view"
    return faces[0], None
