"""Memory store round-trips (SQLite in tmp dir)."""

from shikigami.memory.store import MemoryStore


def test_facts_roundtrip(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite3")
    m.remember("favorite_color", "blue")
    m.remember("favorite_color", "teal")  # upsert
    m.remember("name", "Jens")
    facts = {f.key: f.value for f in m.facts()}
    assert facts == {"favorite_color": "teal", "name": "Jens"}
    m.forget("name")
    assert [f.key for f in m.facts()] == ["favorite_color"]
    m.close()


def test_events_and_mood(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite3")
    m.log_event("heard", {"text": "hi"})
    m.log_event("turn", {"reply": "yo"})
    events = m.recent_events(10)
    assert [e.kind for e in events] == ["heard", "turn"]  # oldest first
    assert events[0].payload == {"text": "hi"}
    assert m.get_mood() == "neutral"
    m.set_mood("curious")
    assert m.get_mood() == "curious"
    m.close()
