"""Run a blocking call on a daemon thread and await its result.

`asyncio.to_thread` parks the call on the loop's default executor, whose worker
threads are non-daemon and get joined at interpreter exit — so a stuck PortAudio
or httpx call would hang Ctrl-C forever. Daemon threads are never joined, so
shutdown stays prompt even if the call never returns.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


async def run_detached(fn: Callable[[], T]) -> T:
    fut: concurrent.futures.Future[T] = concurrent.futures.Future()

    def runner() -> None:
        # awaiter may have cancelled the Future (Ctrl-C / stop); ignore late set_*
        try:
            result = fn()
        except BaseException as exc:
            if not fut.cancelled():
                with contextlib.suppress(concurrent.futures.InvalidStateError):
                    fut.set_exception(exc)
            return
        if not fut.cancelled():
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                fut.set_result(result)

    name = getattr(fn, "__name__", "call")
    threading.Thread(target=runner, daemon=True, name=f"shikigami-{name}").start()
    return await asyncio.wrap_future(fut)
