"""When a new utterance may cut the robot off.

The robot's own voice never reaches this point: the transport's echo guard
keeps it out of the mic (see ``transport.py``). So a turn starts on plain
VAD, and the only question left is whether it interrupts what is running.
"""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import Frame
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy


def interruptions_enabled(
    *,
    barge_in: bool,
    interrupt_while_moving: bool,
    moving: bool,
) -> bool:
    """Barge-in is on, unless the body is moving and that is configured off."""
    if not barge_in:
        return False
    return interrupt_while_moving or not moving


class MotionAwareVADUserTurnStartStrategy(VADUserTurnStartStrategy):
    """VAD turn start whose interruption flag follows the body's motion."""

    def __init__(
        self,
        *,
        barge_in: bool,
        interrupt_while_moving: bool,
        moving: Callable[[], bool],
    ) -> None:
        super().__init__(enable_interruptions=barge_in)
        self._barge_in = barge_in
        self._interrupt_while_moving = interrupt_while_moving
        self._moving = moving

    async def process_frame(self, frame: Frame):
        self._enable_interruptions = interruptions_enabled(
            barge_in=self._barge_in,
            interrupt_while_moving=self._interrupt_while_moving,
            moving=self._moving(),
        )
        await super().process_frame(frame)
