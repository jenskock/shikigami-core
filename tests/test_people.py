"""Phase 2: face tracks, recognition, presence events, enrollment, face tools, storage."""

import asyncio

import numpy as np
from pipecat.services.llm_service import FunctionCallParams

from shikigami.hal.interfaces import Frame
from shikigami.memory.store import MemoryStore
from shikigami.perception.detectors import Detection
from shikigami.perception.people import PeopleSettings, PeopleTracker, iou, largest_face
from shikigami.perception.pipeline import (
    CAT_STABLE_FRAMES,
    Perception,
    PerceptionResult,
    describe_view,
    suppress_cats_on_faces,
)
from shikigami.perception.recognition import FaceGallery, normalize
from shikigami.voice.tools import build_tool_schemas

JENS = normalize(np.eye(128, dtype=np.float32)[0])
ANNA = normalize(np.eye(128, dtype=np.float32)[1])
STRANGER = normalize(np.eye(128, dtype=np.float32)[2])


def face(x, y=0.3, w=0.15, h=0.2):
    lms = tuple((x + w * f, y + h * 0.5) for f in (0.3, 0.7, 0.5, 0.35, 0.65))
    return Detection("face", x, y, w, h, 0.9, lms)


class FakeEmbedder:
    """Who a face is depends on where it is: left half Jens, right half someone else."""

    def __init__(self, right=STRANGER):
        self.right = right
        self.calls = 0

    def embed(self, bgr, f):
        self.calls += 1
        return JENS if f.x < 0.5 else self.right


def tracker(embedder=None, gallery=None, **kw):
    g = gallery if gallery is not None else FaceGallery()
    if gallery is None:
        g.add("Jens", [JENS])
    s = PeopleSettings(**{"arrive_after_s": 0.5, "leave_after_s": 5.0, **kw})
    return PeopleTracker(embedder if embedder is not None else FakeEmbedder(), g, s)


def run(pt, frames):
    """frames: list of (t, [faces]) -> (views per step, all events)."""
    views, events = [], []
    for t, faces in frames:
        v, e = pt.update(faces, None if pt.embedder is None else np.zeros((4, 4, 3)), t)
        views.append(v)
        events += e
    return views, events


def test_iou():
    assert iou(face(0.1), face(0.1)) == 1.0
    assert iou(face(0.1), face(0.7)) == 0.0


def test_track_keeps_id_while_face_moves_and_drops_after_lost():
    pt = tracker()
    views, _ = run(pt, [(0.0, [face(0.10)]), (0.1, [face(0.12)]), (0.2, [face(0.14)])])
    assert len({v[0].track_id for v in views}) == 1
    run(pt, [(1.5, [])])
    assert pt.tracks == {}


def test_known_face_is_named_and_arrives_once():
    pt = tracker()
    views, events = run(pt, [(t / 10, [face(0.2)]) for t in range(10)])
    assert views[0][0].name == "Jens"
    assert events == [("person_arrived", {"name": "Jens"})]  # after 0.5 s, only once


def test_turning_away_does_not_regreet_but_leaving_says_goodbye():
    pt = tracker()
    _, events = run(
        pt,
        [(0.0, [face(0.2)]), (0.6, [face(0.2)])]  # arrives
        + [(1.0 + i / 10, []) for i in range(20)]  # turned away 2 s (new track after)
        + [(3.5, [face(0.2)]), (4.2, [face(0.2)])]  # back: same person, no new greeting
        + [(10.0, [])],  # gone > leave_after_s
    )
    assert events == [
        ("person_arrived", {"name": "Jens"}),
        ("person_left", {"name": "Jens"}),
    ]


def test_unknown_face_announced_once_after_attempts():
    pt = tracker(recheck_unknown_s=0.2)
    _, events = run(pt, [(i / 10, [face(0.7)]) for i in range(20)])
    assert events == [("person_arrived", {"name": "unknown"})]


def test_only_one_embedding_per_update():
    emb = FakeEmbedder()
    pt = tracker(emb)
    run(pt, [(0.0, [face(0.1), face(0.7)])])
    assert emb.calls == 1
    run(pt, [(0.1, [face(0.1), face(0.7)])])
    assert emb.calls == 2


def test_known_name_survives_one_bad_check():
    emb = FakeEmbedder()
    pt = tracker(emb, recheck_known_s=0.1)
    run(pt, [(0.0, [face(0.2)])])
    emb.embed = lambda bgr, f: STRANGER  # blur / turned head
    views, _ = run(pt, [(0.2, [face(0.2)])])
    assert views[0][0].name == "Jens"
    views, _ = run(pt, [(0.4, [face(0.2)])])
    assert views[0][0].name is None


def test_without_recognizer_faces_stay_unknown():
    pt = PeopleTracker(None, FaceGallery(), PeopleSettings(arrive_after_s=0.5))
    views, events = run(pt, [(0.0, [face(0.2)]), (0.6, [face(0.2)])])
    assert views[-1][0].name is None
    assert events == [("person_arrived", {"name": "unknown"})]


def test_describe_view_with_names():
    def res(*names):
        from shikigami.perception.people import PersonView

        people = tuple(PersonView(i, n, face(0.1 * i)) for i, n in enumerate(names))
        return PerceptionResult(seq=1, t=0.0, people=people, cats_in_view=0)

    assert describe_view(res("Jens"), 0.0) == "1 face, who: Jens"
    assert describe_view(res("Jens", None), 0.0) == "2 faces, who: Jens + 1 unknown"
    assert describe_view(res(None, None), 0.0) == "2 faces, who: unknown"


def test_cat_on_a_face_is_dropped():
    f = face(0.2)
    cat_on_face = Detection("cat", 0.22, 0.32, 0.1, 0.1)
    real_cat = Detection("cat", 0.7, 0.6, 0.1, 0.1)
    assert suppress_cats_on_faces([f, cat_on_face, real_cat]) == [f, real_cat]


class _Cam:
    def latest(self):
        return None


class _Det:
    name = "fake"

    def __init__(self, dets):
        self.dets = dets

    def detect(self, bgr):
        return list(self.dets)


def _frame(seq):
    return Frame(main=np.zeros((960, 1280, 3), np.uint8), lores=None, seq=seq, t=0.0)


def test_cat_counts_only_when_stable():
    det = _Det([Detection("cat", 0.7, 0.6, 0.1, 0.1)])
    p = Perception(_Cam(), det)
    counts = [p.process(_frame(i), now=i / 10).cats_in_view for i in range(CAT_STABLE_FRAMES)]
    assert counts[:-1] == [0] * (CAT_STABLE_FRAMES - 1) and counts[-1] == 1


def test_enrollment_collects_samples_and_names_the_track():
    det = _Det([face(0.2)])
    p = Perception(_Cam(), det, people=tracker(FakeEmbedder(), gallery=FaceGallery()))
    req = p.request_enroll(samples=3, timeout=60)
    for i in range(10):
        p.process(_frame(i), now=i * 0.2)
    assert req.done.is_set() and req.ok and len(req.embeddings) == 3
    assert req.track_id is not None


def test_enrollment_refuses_two_similar_faces_and_missing_model():
    p = Perception(
        _Cam(), _Det([face(0.1), face(0.6)]), people=tracker(FakeEmbedder(), FaceGallery())
    )
    req = p.request_enroll(samples=2, timeout=60)
    p.process(_frame(1), now=0.0)
    assert not req.done.is_set() and req.reason == "several faces in view"
    assert largest_face([]) == (None, "no face in view")
    p2 = Perception(_Cam(), _Det([face(0.1)]), people=PeopleTracker(None, FaceGallery()))
    req2 = p2.request_enroll()
    assert req2.done.is_set() and not req2.ok and "not available" in req2.reason


def test_memory_faces_roundtrip(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite3")
    for _ in range(5):
        m.add_face("Jens", JENS.tobytes())
    m.add_face("Anna", ANNA.tobytes())
    m.prune_faces("Jens", 3)
    g = FaceGallery.from_rows(m.faces())
    assert g.names() == ["Anna", "Jens"]
    assert g.match(JENS)[0] == "Jens" and g.match(ANNA)[0] == "Anna"
    assert m.forget_face("jens") == 3
    assert m.face_names() == ["Anna"]
    m.close()


def test_vision_tools_offered_and_failure_reruns_llm():
    calls = []

    class Hooks:
        def note_tool(self, a): ...
        async def execute(self, a): ...
        def tool_enter(self):
            return 0

        def tool_exit(self, turn=None): ...

        async def vision_tool(self, name, args):
            calls.append((name, args))
            return {"ok": False, "error": "no face in view"}

    schemas = build_tool_schemas(Hooks())
    remember = next(s for s in schemas if s.name == "remember_face")
    assert any(s.name == "forget_face" for s in schemas)
    results = []

    async def cb(result, *, properties=None):
        results.append((result, properties))

    asyncio.run(
        remember.handler(
            FunctionCallParams(
                function_name="remember_face",
                tool_call_id="r",
                arguments={"name": "Anna"},
                llm=None,  # type: ignore[arg-type]
                pipeline_worker=None,  # type: ignore[arg-type]
                context=None,  # type: ignore[arg-type]
                result_callback=cb,
            )
        )
    )
    assert calls == [("remember_face", {"name": "Anna"})]
    result, props = results[0]
    assert result["ok"] is False and props.run_llm is True
