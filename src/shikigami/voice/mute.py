"""When the user is allowed to interrupt the robot."""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import Frame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy


class MicMute:
    """Dashboard / Ctrl+M mute. Shared by the transport and the user aggregator."""

    def __init__(self) -> None:
        self.manual = False

    def set_muted(self, muted: bool) -> None:
        self.manual = bool(muted)


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


class MovingGateTurnStart(VADUserTurnStartStrategy):
    """VAD turn start whose interruption flag follows motion."""

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


class ManualMuteStrategy(BaseUserMuteStrategy):
    """Drop user audio and transcripts while the dashboard mic is muted.

    Typed text (``LLMMessagesAppendFrame``) is not in the muted set, so the
    chat field still works.
    """

    def __init__(self, mute: MicMute) -> None:
        super().__init__()
        self._mute = mute

    async def process_frame(self, frame: Frame) -> bool:
        return self._mute.manual
