"""Camera contract: Mac sources (webcam/file) deliver exactly the Pi's streams."""

import asyncio
import logging
import time

import cv2
import numpy as np
import pytest

from shikigami.config import Config
from shikigami.hal.camera_base import StreamSpec, center_crop_to_aspect, shape_frame
from shikigami.hal.cameras import make_camera
from shikigami.hal.file_camera import FileCamera
from shikigami.perception import Perception, build_detector
from shikigami.perception.detectors import CompositeDetector, Detection, HaarDetector


def _scene(w, h, box_frac=(0.45, 0.4, 0.1, 0.2)):
    """Gray frame with one bright box at a normalized position of the 4:3 center region."""
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    # place the box relative to the centered 4:3 region (what the Pi sensor would see)
    cw = min(w, int(round(h * 4 / 3)))
    x0 = (w - cw) // 2
    bx, by, bw, bh = box_frac
    cv2.rectangle(
        img,
        (x0 + int(bx * cw), int(by * h)),
        (x0 + int((bx + bw) * cw), int((by + bh) * h)),
        (255, 255, 255),
        -1,
    )
    return img


class BlobDetector:
    """Finds the bright box — stands in for a face detector, deterministic."""

    name = "blob"

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(gray > 200)
        if len(xs) == 0:
            return []
        h, w = gray.shape
        return [
            Detection(
                "face",
                xs.min() / w,
                ys.min() / h,
                (xs.max() - xs.min() + 1) / w,
                (ys.max() - ys.min() + 1) / h,
            )
        ]


def test_pi_and_emulated_specs_match():
    cfg = Config()
    assert StreamSpec.from_config(cfg.camera, emulate=True) == StreamSpec.from_config(
        cfg.camera, emulate=False
    )
    cfg.camera.pi_profile = False
    native = StreamSpec.from_config(cfg.camera, emulate=True)
    assert native.main_size is None and native.fps == 0


def test_webcam_frame_is_shaped_to_pi_streams():
    spec = StreamSpec(main_size=(1280, 960), lores_size=(320, 240), fps=15)
    main, lores = shape_frame(np.zeros((720, 1280, 3), np.uint8), spec)
    assert main.shape == (960, 1280, 3) and lores.shape == (240, 320, 3)


def test_center_crop_to_4_3():
    assert center_crop_to_aspect(np.zeros((720, 1280, 3)), 4 / 3).shape[:2] == (720, 960)
    assert center_crop_to_aspect(np.zeros((480, 640, 3)), 4 / 3).shape[:2] == (480, 640)


def test_flip_matches_mounting():
    raw = np.zeros((240, 320, 3), np.uint8)
    raw[:, :10] = 255  # bright left edge
    spec = StreamSpec(main_size=(320, 240), lores_size=(320, 240), flip="h")
    main, lores = shape_frame(raw, spec)
    assert main[:, -5:].mean() > 200 and lores[:, -5:].mean() > 200


def test_webcam_and_pi_frames_give_same_detections():
    """Parity: a 16:9 webcam frame and a 4:3 Pi frame of the same scene -> same boxes."""
    spec = StreamSpec(main_size=(1280, 960), lores_size=(320, 240))
    det = BlobDetector()
    _, mac_lores = shape_frame(_scene(1280, 720), spec)
    _, pi_lores = shape_frame(_scene(1280, 960), spec)
    (a,) = det.detect(mac_lores)
    (b,) = det.detect(pi_lores)
    for k in ("x", "y", "w", "h"):
        assert abs(getattr(a, k) - getattr(b, k)) < 0.02, k


def _write_frames(folder, n=5):
    folder.mkdir()
    for i in range(n):
        cv2.imwrite(str(folder / f"{i:03d}.png"), _scene(1280, 720))


def test_file_camera_delivers_pi_frames_at_capped_rate(tmp_path):
    _write_frames(tmp_path / "clip")
    spec = StreamSpec(main_size=(1280, 960), lores_size=(320, 240), fps=20)
    cam = FileCamera(tmp_path / "clip", spec=spec, log=logging.getLogger("test"))

    async def main():
        await cam.start()
        await asyncio.sleep(0.5)
        f = cam.latest()
        jpeg = await cam.snapshot_jpeg(max_side=640)
        await cam.stop()
        return f, jpeg

    f, jpeg = asyncio.run(main())
    assert f is not None
    assert f.main.shape == (960, 1280, 3) and f.lores.shape == (240, 320, 3)
    assert 5 <= f.seq <= 13  # ~10 frames in 0.5 s at 20 fps, not hundreds
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 640


def test_perception_runs_on_camera_frames(tmp_path):
    _write_frames(tmp_path / "clip")
    cam = FileCamera(
        tmp_path / "clip",
        spec=StreamSpec(main_size=(1280, 960), lores_size=(320, 240), fps=30),
        log=logging.getLogger("test"),
    )
    perception = Perception(cam, BlobDetector(), detect_hz=20)

    async def main():
        await cam.start()
        perception.start()
        deadline = time.monotonic() + 2
        while perception.latest() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        perception.stop()
        await cam.stop()

    asyncio.run(main())
    res = perception.latest()
    assert res is not None
    (face,) = res.of("face")
    assert abs(face.x - 0.45) < 0.02 and abs(face.y - 0.4) < 0.02


def test_make_camera_picks_backend(tmp_path):
    cfg = Config()
    cfg.sim.camera_preview = False
    assert type(make_camera(cfg, on_robot=False)).__name__ == "SimCamera"
    assert type(make_camera(cfg, on_robot=True)).__name__ == "PiCamera"
    cfg.camera.backend = "file"
    with pytest.raises(ValueError):
        make_camera(cfg, on_robot=False)
    cfg.camera.file = str(tmp_path)
    assert isinstance(make_camera(cfg, on_robot=True), FileCamera)


def test_detector_falls_back_to_haar_without_model(tmp_path):
    cfg = Config().perception
    det = build_detector(cfg, tmp_path)
    assert isinstance(det, CompositeDetector)
    assert [type(p) for p in det.parts] == [HaarDetector]
    assert det.parts[0].labels == ("face", "cat")
