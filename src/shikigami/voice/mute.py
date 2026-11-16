"""Manual mic mute for the dashboard / Ctrl+M."""

from __future__ import annotations

from pipecat.frames.frames import Frame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


class MicMute:
    """Mute shared by the transport and the user aggregator.

    Only ``manual`` (dashboard / Ctrl+M) mutes. Speaker echo is handled by
    echo-aware turn-start strategies, not by dropping mic samples.
    """

    def __init__(self) -> None:
        self.manual = False

    @property
    def active(self) -> bool:
        return self.manual

    def set_muted(self, muted: bool) -> None:
        self.manual = bool(muted)


class ManualMuteStrategy(BaseUserMuteStrategy):
    """Drop user audio and transcripts while manually muted.

    Typed text (``LLMMessagesAppendFrame``) is not in the muted set, so the
    chat field still works.
    """

    def __init__(self, mute: MicMute) -> None:
        super().__init__()
        self._mute = mute

    async def process_frame(self, frame: Frame) -> bool:
        return self._mute.active
