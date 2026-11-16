"""CLI: `uv run sim` (Mac buddy) or `uv run shikigami` (generic / Pi)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys

from .app.bus import EventBus
from .app.loop import App
from .brain.agent import Brain
from .config import REPO_ROOT, Config, load_config
from .hal import create_hal, resolve_backend
from .memory.store import MemoryStore
from .personality.loader import load_personality


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="shikigami", description="shikigami-core robot companion")
    p.add_argument("--config", default=None, help="YAML config path (default: config/default.yaml)")
    p.add_argument("--backend", choices=["sim", "pidog", "picrawler"], default=None)
    p.add_argument("--text", default=None, help="run one turn with this text, then exit")
    p.add_argument("--mock", action="store_true", help="use the canned mock LLM (no network)")
    p.add_argument("--no-window", action="store_true", help="disable the sim face window")
    p.add_argument("--no-camera", action="store_true", help="disable the camera preview window")
    p.add_argument("--no-tts", action="store_true", help="disable speech output")
    p.add_argument("--no-stt", action="store_true", help="text input (disable mic/STT)")
    p.add_argument("--mic", action="store_true", help="force mic input + STT (default for `sim`)")
    p.add_argument("--text-input", action="store_true", help="type instead of speaking")
    p.add_argument("--log-level", default=None)
    return p.parse_args(argv)


def build_app(cfg: Config, log: logging.Logger) -> App:
    backend = resolve_backend(cfg.backend)
    hal = create_hal(backend, cfg)
    personality = load_personality(REPO_ROOT / cfg.personality.path)
    if lang := os.environ.get("SHIKIGAMI_LANGUAGE"):
        personality.language = lang
    # steer STT toward the buddy's language when not set explicitly
    if not cfg.stt.language and personality.language:
        cfg.stt.language = personality.language
    memory = MemoryStore(REPO_ROOT / cfg.memory.path)
    brain = Brain(cfg.llm, personality, memory, log=logging.getLogger("shikigami.brain"))
    bus = EventBus()

    async def _persist(e) -> None:
        memory.log_event(e.kind, e.payload)

    bus.subscribe("*", _persist)
    return App(cfg, hal, brain, memory, bus, log=log)


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.backend = args.backend
    if args.mock:
        cfg.llm.mock = True
    if args.no_window:
        cfg.sim.window = False
    if args.no_camera:
        cfg.sim.camera_preview = False
        cfg.perception.track_faces = False
    if args.no_tts:
        cfg.tts.enabled = False
    if args.mic:
        cfg.sim.input = "mic"
        cfg.stt.enabled = True
    if args.text_input or args.no_stt:
        cfg.sim.input = "text"
        cfg.stt.enabled = False
    if args.log_level:
        cfg.log_level = args.log_level

    logging.basicConfig(
        level=getattr(
            logging, cfg.log_level.upper()
        ),  # AttributeError on a typo'd level = fail-fast
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("shikigami")
    log.info("backend=%s llm=%s@%s", resolve_backend(cfg.backend), cfg.llm.model, cfg.llm.base_url)

    app = build_app(cfg, log)

    async def run() -> None:
        await app.start()
        try:
            if args.text is not None:
                await app.handle_text(args.text)
            else:
                await app.run()
        finally:
            await app.stop()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())
    return 0


def main_sim(argv: list[str] | None = None) -> int:
    """Mac entrypoint: `uv run sim` → mic on; backend from config/env (default picrawler).

    Pass ``--backend sim`` for the label-only sim HAL. SoftPicrawler is the default
    Mac path so ``uv run sim`` shows real gait/servo behaviour without env vars.
    """
    rest = list(sys.argv[1:] if argv is None else argv)
    return main(["--mic", *rest])


if __name__ == "__main__":
    sys.exit(main())
