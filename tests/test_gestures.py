"""A small body gesture goes with each face, and never fights a requested move."""

import asyncio
import logging
from types import SimpleNamespace

from shikigami.app.bus import EventBus
from shikigami.app.loop import EMOTE_GESTURES, App
from shikigami.brain.actions import TOOL_SCHEMAS, EmoteAction, MoveAction
from shikigami.config import load_config
from shikigami.memory.store import MemoryStore


class _Motion:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.moving = False

    async def _do(self, name: str) -> None:
        self.moving = True
        self.calls.append(name)
        await asyncio.sleep(0.01)
        self.moving = False

    async def play_action(self, name: str) -> None:
        await self._do(name)

    async def stand(self) -> None:
        await self._do("stand")

    async def settle(self) -> None:
        await self._do("settle")

    async def lie(self) -> None:
        await self._do("lie")

    async def walk(self, direction, speed, duration_s) -> None:
        await self._do(f"walk {direction}")


def _app(tmp_path) -> tuple[App, _Motion]:
    cfg = load_config()
    cfg.memory.path = str(tmp_path / "m.db")
    motion = _Motion()
    faces: list[str] = []

    async def emote(name: str) -> None:
        faces.append(name)

    hal = SimpleNamespace(
        audio=SimpleNamespace(set_muted=lambda *_a, **_k: None),
        expression=SimpleNamespace(emote=emote),
        motion=motion,
        camera=None,
        backend="sim",
    )
    app = App(
        cfg,
        hal,
        brain=None,
        memory=MemoryStore(cfg.memory.path),
        bus=EventBus(),
        log=logging.getLogger("test"),
        use_voice=False,
    )
    return app, motion


async def _settle(app: App) -> None:
    while app._gestures:
        await asyncio.gather(*list(app._gestures))


def test_gestures_never_walk_or_turn():
    emotes = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "emote")
    known = set(emotes["function"]["parameters"]["properties"]["name"]["enum"])
    assert set(EMOTE_GESTURES) <= known
    for steps in EMOTE_GESTURES.values():
        assert not any(word in step for step in steps for word in ("forward", "backward", "turn"))


def test_emote_plays_its_gesture_without_holding_the_turn(tmp_path):
    app, motion = _app(tmp_path)

    async def run() -> None:
        await app.execute(EmoteAction(name="curious"))
        assert motion.calls == []  # the tool call returned before the body moved
        await _settle(app)

    asyncio.run(run())
    assert motion.calls == ["look left", "look right", "settle"]


def test_neutral_and_disabled_gestures_keep_the_body_still(tmp_path):
    app, motion = _app(tmp_path)

    async def run() -> None:
        await app.execute(EmoteAction(name="neutral"))
        app.cfg.reflexes.emote_gestures = False
        await app.execute(EmoteAction(name="happy"))
        await _settle(app)

    asyncio.run(run())
    assert motion.calls == []


def test_gesture_yields_to_a_requested_move(tmp_path):
    app, motion = _app(tmp_path)

    async def run() -> None:
        # Move first: the face that follows adds nothing to the body.
        move = asyncio.create_task(app.execute(MoveAction(motion="walk")))
        await asyncio.sleep(0)
        await app.execute(EmoteAction(name="happy"))
        await move
        await _settle(app)
        assert motion.calls == ["walk forward"]

        # Face first: the gesture finishes, then the move runs.
        motion.calls.clear()
        await app.execute(EmoteAction(name="surprised"))
        await asyncio.sleep(0)
        await app.execute(MoveAction(motion="walk"))
        await _settle(app)
        assert motion.calls == ["look up", "settle", "walk forward"]

    asyncio.run(run())


def test_no_gesture_while_lying_down(tmp_path):
    app, motion = _app(tmp_path)

    async def run() -> None:
        await app.execute(MoveAction(motion="lie"))
        await app.execute(EmoteAction(name="happy"))
        await _settle(app)
        assert motion.calls == ["lie"]  # a wave would stand the robot back up
        await app.execute(MoveAction(motion="stand"))
        await app.execute(EmoteAction(name="happy"))
        await _settle(app)
        assert motion.calls == ["lie", "stand", "wave"]

    asyncio.run(run())
