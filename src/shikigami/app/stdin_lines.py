"""Async line reader for a file descriptor (stdin) via loop.add_reader.

Unlike `asyncio.to_thread(input, ...)`, this never leaves a blocked worker
thread behind, so Ctrl-C shuts down cleanly.
"""

from __future__ import annotations

import asyncio
import os


class StdinLines:
    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buf = b""
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._closed = False
        self._loop.add_reader(fd, self._on_readable)

    def _on_readable(self) -> None:
        if self._closed:
            return
        try:
            data = os.read(self._fd, 4096)
        except OSError:
            data = b""
        if not data:  # EOF
            self.close()
            self._queue.put_nowait(None)
            return
        self._buf += data
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            self._queue.put_nowait(raw.decode(errors="replace"))

    def __aiter__(self) -> StdinLines:
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._loop.remove_reader(self._fd)
