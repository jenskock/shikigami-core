"""run_detached: blocking calls run on daemon threads, so cancelling the await
(or Ctrl-C at interpreter exit) is prompt even if the call itself never returns."""

import asyncio
import threading
import time

import pytest

from shikigami.concurrency import run_detached


def test_run_detached_returns_value():
    assert asyncio.run(run_detached(lambda: 42)) == 42


def test_run_detached_propagates_exception():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        asyncio.run(run_detached(boom))


def test_run_detached_cancelled_future_does_not_crash_thread():
    """Ctrl-C cancels the await; the daemon thread must not raise InvalidStateError."""

    async def main():
        started = threading.Event()
        release = threading.Event()

        def blocked():
            started.set()
            release.wait(2.0)
            return 1

        task = asyncio.create_task(run_detached(blocked))
        for _ in range(50):  # don't block the loop — let run_detached start the thread
            if started.is_set():
                break
            await asyncio.sleep(0.02)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.sleep(0.05)  # let the thread finish set_result on a cancelled fut

    asyncio.run(main())


def test_run_detached_cancellation_is_prompt():
    async def main():
        task = asyncio.create_task(run_detached(lambda: time.sleep(30)))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    t0 = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - t0 < 5  # did not wait out the 30 s blocking call
