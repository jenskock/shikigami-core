"""Compact serial protocol to offload eyes to an ESP32 (GC9A01 / similar).

Wire format: newline-terminated ASCII commands, optional binary frame mode.

Commands (host → ESP32)::

    EMO <name> [intensity] [duration_s]
    BLEND <name>=<w> ...
    VA <valence> <arousal>
    GAZE <x> <y>
    TRACK <x> <y>
    BLINK
    WINK left|right
    REACT <event>
    SPEAK 0|1
    CFG color=<r>,<g>,<b> blink=<per_min> ...
    PING
    SYNC                 # request current state summary

Responses (ESP32 → host)::

    OK
    ERR <message>
    PONG
    STATE emotion=... gaze=...

Firmware stub: see ``firmware/eyes_esp32_stub.ino``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO


@dataclass(frozen=True, slots=True)
class EyesCommand:
    name: str
    args: tuple[str, ...] = ()

    def encode(self) -> bytes:
        if not self.args:
            return f"{self.name}\n".encode()
        return f"{self.name} {' '.join(self.args)}\n".encode()


def encode_emo(name: str, intensity: float = 1.0, duration: float | None = None) -> EyesCommand:
    args = [name, f"{intensity:.3f}"]
    if duration is not None:
        args.append(f"{duration:.3f}")
    return EyesCommand("EMO", tuple(args))


def encode_gaze(x: float, y: float, *, track: bool = False) -> EyesCommand:
    return EyesCommand("TRACK" if track else "GAZE", (f"{x:.4f}", f"{y:.4f}"))


def encode_blink() -> EyesCommand:
    return EyesCommand("BLINK")


def encode_wink(side: str) -> EyesCommand:
    return EyesCommand("WINK", (side,))


def encode_react(event: str) -> EyesCommand:
    return EyesCommand("REACT", (event.replace(" ", "_"),))


def encode_speak(active: bool) -> EyesCommand:
    return EyesCommand("SPEAK", ("1" if active else "0",))


def parse_command(line: str) -> EyesCommand | None:
    raw = line.strip()
    if not raw:
        return None
    parts = raw.split()
    return EyesCommand(parts[0].upper(), tuple(parts[1:]))


class SerialEyesClient:
    """Stub host-side client — writes commands to a file-like / serial port."""

    def __init__(self, port: BinaryIO) -> None:
        self._port = port

    def send(self, cmd: EyesCommand) -> None:
        self._port.write(cmd.encode())

    def send_many(self, cmds: Iterable[EyesCommand]) -> None:
        for c in cmds:
            self.send(c)
