"""Pick the camera for a backend from ``cfg.camera``.

``auto`` means picamera2 on the robot and the webcam on the Mac. Whatever is picked
delivers the same ``Frame`` contract (see ``camera_base``).
"""

from __future__ import annotations

import logging
from typing import Any

from .camera_base import StreamSpec
from .interfaces import Camera

BACKENDS = ("auto", "picamera2", "webcam", "file")


def make_camera(
    cfg,
    *,
    on_robot: bool,
    dashboard: Any | None = None,
    log: logging.Logger | None = None,
) -> Camera:
    """``on_robot``: running on the Pi with the vendor libs (else the Mac emulation)."""
    cam_cfg = cfg.camera
    backend = (cam_cfg.backend or "auto").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"unknown camera.backend {backend!r}; expected one of {BACKENDS}")
    if backend == "auto":
        backend = "picamera2" if on_robot else "webcam"
    preview_dash = dashboard if cfg.sim.camera_preview else None

    if backend == "picamera2":
        from .picamera import PiCamera

        return PiCamera(StreamSpec.from_config(cam_cfg, emulate=False), log=log)
    if backend == "file":
        from .file_camera import FileCamera

        if not cam_cfg.file:
            raise ValueError("camera.backend=file needs camera.file (or SHIKIGAMI_CAMERA_FILE)")
        return FileCamera(
            cam_cfg.file,
            spec=StreamSpec.from_config(cam_cfg, emulate=True),
            dashboard=preview_dash,
            log=log,
        )
    from .sim.camera import SimCamera

    return SimCamera(
        cfg.sim.camera_index,
        preview=preview_dash is not None,
        dashboard=dashboard,
        log=log,
        spec=StreamSpec.from_config(cam_cfg, emulate=True),
    )
