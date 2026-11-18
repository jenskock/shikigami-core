"""Shared camera machinery: capture thread, Pi stream shaping, snapshots, preview.

Every camera (picamera2 on the Pi, webcam or file on the Mac) delivers the same
``Frame``: a ``main`` frame at ``main_size`` and a ``lores`` frame at
``detect_size``, at most ``fps`` times per second. Perception never knows which
camera it runs on — that is what makes the Mac a faithful stand-in for the Pi.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .interfaces import Camera, Frame

PREVIEW_EVERY = 2  # paint the preview every 2nd reflex tick (~10 Hz at 20 Hz)


@dataclass(frozen=True)
class StreamSpec:
    """What a camera must deliver. ``None`` sizes / ``fps <= 0`` mean native / uncapped."""

    main_size: tuple[int, int] | None = (1280, 960)
    lores_size: tuple[int, int] = (320, 240)
    fps: float = 15.0
    flip: str = "none"  # none | h | v | both

    @classmethod
    def from_config(cls, cam_cfg, *, emulate: bool) -> StreamSpec:
        """Pi spec from config; ``emulate`` = a Mac source, which honours ``pi_profile``."""
        if emulate and not cam_cfg.pi_profile:
            w, h = cam_cfg.detect_size
            return cls(main_size=None, lores_size=(w, h), fps=0.0, flip=cam_cfg.flip)
        return cls(
            main_size=tuple(cam_cfg.main_size),
            lores_size=tuple(cam_cfg.detect_size),
            fps=float(cam_cfg.fps),
            flip=cam_cfg.flip,
        )


def _flip(img: Any, how: str) -> Any:
    import cv2

    code = {"h": 1, "v": 0, "both": -1}.get(how)
    return img if code is None else cv2.flip(img, code)


def center_crop_to_aspect(img: Any, aspect: float) -> Any:
    """Crop the middle of ``img`` to width/height ``aspect`` (what the Pi sensor sees)."""
    h, w = img.shape[:2]
    if w / h > aspect + 1e-3:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return img[:, x0 : x0 + nw]
    if w / h < aspect - 1e-3:
        nh = int(round(w / aspect))
        y0 = (h - nh) // 2
        return img[y0 : y0 + nh]
    return img


def shape_frame(raw: Any, spec: StreamSpec, lores: Any | None = None) -> tuple[Any, Any]:
    """Raw BGR capture -> (main, lores) per spec. Pass ``lores`` if the hardware made one."""
    import cv2

    main = _flip(raw, spec.flip)
    if spec.main_size is not None:
        mw, mh = spec.main_size
        main = center_crop_to_aspect(main, mw / mh)
        if main.shape[1] != mw or main.shape[0] != mh:
            main = cv2.resize(main, (mw, mh), interpolation=cv2.INTER_AREA)
    main = np.ascontiguousarray(main)
    if lores is None:
        lw, lh = spec.lores_size
        if spec.main_size is None:  # native: keep the source aspect at detect width
            lh = max(1, int(round(main.shape[0] * lw / main.shape[1])))
        lores = cv2.resize(main, (lw, lh), interpolation=cv2.INTER_AREA)
    else:
        lores = _flip(lores, spec.flip)
    return main, np.ascontiguousarray(lores)


def draw_preview(frame: Any, out: Any, detections: list | None) -> None:
    """Letterbox ``frame`` into ``out`` and draw normalized detection boxes."""
    import cv2

    ph, pw = out.shape[:2]
    fh, fw = frame.shape[:2]
    scale = min(pw / fw, ph / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out[:] = 0
    oy, ox = (ph - nh) // 2, (pw - nw) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    for d in detections or ():
        bx, by = ox + int(d.x * nw), oy + int(d.y * nh)
        bw, bh = int(d.w * nw), int(d.h * nh)
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(
            out, d.label, (bx, max(14, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )


def draw_message(out: Any, msg: str) -> None:
    import cv2

    out[:] = 0
    cv2.putText(out, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def encode_jpeg(img: Any, max_side: int = 768, quality: int = 80) -> bytes | None:
    import cv2

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(
            img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return bytes(buf) if ok else None


class ThreadedCamera(Camera):
    """Base: a daemon thread grabs frames, shapes them to the spec, keeps only the newest.

    Subclasses implement ``_open`` / ``_grab`` / ``_close``. ``_grab`` blocks until the
    source has a frame (hardware-paced) or returns immediately (``paced = False``, e.g.
    files); the base sleeps or drops frames so at most ``spec.fps`` are delivered.
    """

    paced = True  # _grab waits for the hardware; False = the base sleeps to hold fps

    def __init__(
        self,
        spec: StreamSpec,
        dashboard: Any | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.spec = spec
        self._dash = dashboard
        self._log = log or logging.getLogger("shikigami.camera")
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._running = False
        self._opened = False
        self._unavailable: str | None = None
        self._tick = 0
        self._preview_buf: Any | None = None
        if dashboard is not None:
            self._preview_buf = np.zeros((dashboard.CAM_H, dashboard.CAM_W, 3), dtype=np.uint8)

    # -- subclass hooks ---------------------------------------------------------------

    @abc.abstractmethod
    def _open(self) -> None:
        """Open the source; raise with a helpful message if it cannot."""

    @abc.abstractmethod
    def _grab(self) -> tuple[Any, Any | None] | None:
        """One raw BGR frame (+ optional hardware lores), or None on a failed read."""

    @abc.abstractmethod
    def _close(self) -> None: ...

    # -- lifecycle --------------------------------------------------------------------

    async def start(self) -> None:
        try:
            self._open()
        except Exception as exc:
            self._unavailable = str(exc)
            raise
        self._opened = True
        self._unavailable = None
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"{type(self).__name__}-capture", daemon=True
        )
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            await asyncio.to_thread(t.join, 1.0)
        self._thread = None
        if self._opened:
            self._opened = False
            self._close()

    def _run(self) -> None:
        period = 1.0 / self.spec.fps if self.spec.fps > 0 else 0.0
        next_at = time.monotonic()
        failures = 0
        while self._running:
            try:
                got = self._grab()
            except Exception:
                self._log.exception("camera grab failed")
                got = None
            now = time.monotonic()
            if got is None:
                failures += 1
                if failures == 50:
                    self._log.warning("camera: no frames — is the source still there?")
                time.sleep(0.02)
                continue
            failures = 0
            if period and now < next_at - (0.2 * period if self.paced else 0.0):
                if self.paced:
                    continue  # hardware is faster than the Pi rate: drop this frame
                time.sleep(next_at - now)
                now = time.monotonic()
            if period:
                next_at += period
                if next_at < now - period:  # resync after a stall instead of bursting
                    next_at = now + period
            raw, lores = got
            main, lores = shape_frame(raw, self.spec, lores)
            with self._lock:
                self._seq += 1
                self._latest = Frame(main=main, lores=lores, seq=self._seq, t=now)

    # -- consumers --------------------------------------------------------------------

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    @property
    def last_frame(self) -> Any | None:
        f = self.latest()
        return None if f is None else f.main

    async def read_frame(self) -> Any | None:
        return self.last_frame

    async def snapshot_jpeg(self, max_side: int = 768, quality: int = 80) -> bytes | None:
        f = self.latest()
        if f is None:
            return None
        return await asyncio.to_thread(encode_jpeg, f.main, max_side, quality)

    async def preview_frame(self, detections: list | None = None) -> None:
        dash = self._dash
        if dash is None or not getattr(dash, "active", False) or self._preview_buf is None:
            return
        self._tick += 1
        if self._tick % PREVIEW_EVERY != 0:
            return
        out = self._preview_buf
        f = self.latest()
        if f is None:
            draw_message(out, self._unavailable or "no camera")
        else:
            draw_preview(f.main, out, detections)
        dash.set_camera(out)
