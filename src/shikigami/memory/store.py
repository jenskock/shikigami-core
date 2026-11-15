"""Small persistent store (SQLite via stdlib): user facts, recent events, mood."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Fact:
    key: str
    value: str
    category: str
    created_at: float


@dataclass
class Event:
    ts: float
    kind: str
    payload: dict


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'fact',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def remember(self, key: str, value: str, category: str = "fact") -> None:
        self._db.execute(
            "INSERT INTO facts(key, value, category, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, category=excluded.category, "
            "created_at=excluded.created_at",
            (key, value, category, time.time()),
        )
        self._db.commit()

    def forget(self, key: str) -> None:
        self._db.execute("DELETE FROM facts WHERE key=?", (key,))
        self._db.commit()

    def facts(self) -> list[Fact]:
        rows = self._db.execute("SELECT * FROM facts ORDER BY created_at").fetchall()
        return [Fact(r["key"], r["value"], r["category"], r["created_at"]) for r in rows]

    def log_event(self, kind: str, payload: dict | None = None) -> None:
        self._db.execute(
            "INSERT INTO events(ts, kind, payload) VALUES(?,?,?)",
            (time.time(), kind, json.dumps(payload or {})),
        )
        self._db.commit()

    def recent_events(self, n: int = 8) -> list[Event]:
        rows = self._db.execute(
            "SELECT ts, kind, payload FROM events ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [Event(r["ts"], r["kind"], json.loads(r["payload"])) for r in reversed(rows)]

    def set_mood(self, mood: str) -> None:
        self._db.execute(
            "INSERT INTO kv(key, value, updated_at) VALUES('mood', ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (mood, time.time()),
        )
        self._db.commit()

    def get_mood(self, default: str = "neutral") -> str:
        row = self._db.execute("SELECT value FROM kv WHERE key='mood'").fetchone()
        return row["value"] if row else default
