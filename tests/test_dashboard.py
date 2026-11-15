"""Dashboard: device face buffer + camera annotate; Qt chrome optional."""

import asyncio
import logging

import cv2
import pytest

from shikigami.hal.sim.camera import SimCamera
from shikigami.hal.sim.dashboard import CAM_H, CAM_W, FACE_H, FACE_W, SimDashboard
from shikigami.hal.sim.expression import SimExpression
from shikigami.hal.sim.state import SimState

np = pytest.importorskip("numpy")


def test_face_and_cam_buffer_sizes():
    dash = SimDashboard(enabled=False)
    assert dash.face_buf.shape == (FACE_H, FACE_W, 3)
    assert dash.cam_buf.shape == (CAM_H, CAM_W, 3)


def test_expression_draws_into_device_face_buffer():
    dash = SimDashboard(enabled=False)
    state = SimState()
    state.last_latency = "llm=100ms total=200ms"
    expr = SimExpression(window=True, state=state, dashboard=dash)

    async def main():
        await expr.emote("happy")
        await expr.update(0.05)

    asyncio.run(main())
    face = dash.face_buf
    assert face[20, 20].mean() < 30
    assert face[260:285, 90:170].max() > 200  # bright white eye body
    assert dash.emotion == "happy"
    assert dash.cam_buf.max() == 0


def test_camera_draws_annotated_frame():
    dash = SimDashboard(enabled=False)
    cam = SimCamera(preview=True, dashboard=dash, log=logging.getLogger("test"))
    cam._cv2 = cv2
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    out = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cam._draw_frame(frame, out, [])
    ph, pw = out.shape[:2]
    scale = min(pw / 640, ph / 480)
    nw, nh = int(640 * scale), int(480 * scale)
    ox, oy = (pw - nw) // 2, (ph - nh) // 2
    # content region is gray; side/letterbox bars stay black
    assert abs(out[oy + 5, ox + 5].mean() - 200) < 5
    if ox > 0:
        assert out[oy + 5, 0].mean() == 0
    cam._draw_frame(frame, out, [("face", 320, 240, 100, 100)])
    green = ((out[:, :, 1] > 200) & (out[:, :, 0] < 100)).sum()
    assert green > 50
    assert dash.face_buf.max() == 0


def test_camera_letterboxes_to_cam_aspect():
    """Non-matching aspect keeps black bars; content centered."""
    dash = SimDashboard(enabled=False)
    cam = SimCamera(preview=True, dashboard=dash, log=logging.getLogger("test"))
    cam._cv2 = cv2
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # 4:3 into 16:9 cam buf
    out = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cam._draw_frame(frame, out, [])
    ph, pw = out.shape[:2]
    scale = min(pw / 640, ph / 480)
    nw = int(640 * scale)
    ox = (pw - nw) // 2
    assert ox > 0
    assert out[:, :ox].max() == 0
    assert abs(out[ph // 2, pw // 2].mean() - 200) < 5


def test_box_coords_are_scaled_to_panel():
    dash = SimDashboard(enabled=False)
    cam = SimCamera(preview=True, dashboard=dash, log=logging.getLogger("test"))
    cam._cv2 = cv2
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cam._draw_frame(frame, out, [("face", 0, 0, 1280, 720)])
    ph, pw = out.shape[:2]
    scale = min(pw / 1280, ph / 720)
    nw, nh = int(1280 * scale), int(720 * scale)
    oy, ox = (ph - nh) // 2, (pw - nw) // 2
    green = (out[:, :, 1] > 200) & (out[:, :, 0] < 100)
    assert green[oy, ox] and green[min(ph - 1, oy + nh - 1), min(pw - 1, ox + nw - 1)]


def test_submit_line_fires_on_submit():
    dash = SimDashboard(enabled=False)
    submitted = []
    dash.on_submit = submitted.append
    dash.submit_line("  hello  ")
    assert submitted == ["hello"]
    dash.submit_line("   ")
    assert submitted == ["hello"]


def test_set_camera_copies_into_cam_buf():
    dash = SimDashboard(enabled=False)
    frame = np.full((CAM_H, CAM_W, 3), 90, dtype=np.uint8)
    dash.set_camera(frame)
    assert dash.cam_buf.mean() == 90


def test_close_is_idempotent():
    dash = SimDashboard(enabled=False)
    closed = []
    dash.on_close = lambda: closed.append(True)
    dash._request_close()
    dash._request_close()
    assert closed == [True] and dash._closed
