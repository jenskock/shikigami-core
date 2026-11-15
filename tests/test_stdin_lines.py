"""StdinLines: async line reading over a pipe fd, incl. EOF."""

import asyncio
import os

from shikigami.app.stdin_lines import StdinLines


def test_reads_lines_then_eof():
    async def main():
        rfd, wfd = os.pipe()
        lines = StdinLines(rfd)
        os.write(wfd, b"hello\nwor")
        await asyncio.sleep(0.05)  # partial line must wait for the newline
        os.write(wfd, b"ld\n")
        first = await lines.__anext__()
        second = await lines.__anext__()
        os.close(wfd)  # EOF
        try:
            await lines.__anext__()
            raise AssertionError("expected StopAsyncIteration")
        except StopAsyncIteration:
            pass
        lines.close()
        os.close(rfd)
        return first, second

    assert asyncio.run(main()) == ("hello", "world")
