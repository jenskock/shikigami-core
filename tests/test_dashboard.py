"""Dashboard: one canvas, panels at their rects; face + stats + camera compose correctly."""

import asyncio
import logging

import pytest

from shikigami.hal.sim.camera import SimCamera
from shikigami.hal.sim.dashboard import SimDashboard
from shikigami.hal.sim.expression import SimExpression
from shikigami.hal.sim.state import SimState

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")


def _dashboard():
    dash = SimDashboard(enabled=True, log=logging.getLogger("test"))
    assert dash.cv2 is not None and dash.canvas is not None
    dash.cv2.namedWindow = lambda *a, **k: None
    dash.cv2.imshow = lambda *a, **k: None
    dash.cv2.waitKey = lambda *a, **k: 0
    return dash


def test_panel_rects_are_disjoint_and_inside_canvas():
    d = SimDashboard(enabled=False)
    rects = [d.FACE_RECT, d.STATS_RECT, d.CAM_RECT, d.SPARE_RECT]
    for x, y, w, h in rects:
        assert x >= 0 and x + w <= d.WIDTH and y >= 0 and y + h <= d.HEIGHT
    # pairwise disjoint
    for i, (x1, y1, w1, h1) in enumerate(rects):
        for x2, y2, w2, h2 in rects[i + 1 :]:
            assert x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1


def test_expression_draws_face_and_stats_into_panels():
    dash = _dashboard()
    state = SimState()
    state.last_latency = "llm=100ms total=200ms"
    expr = SimExpression(window=True, state=state, dashboard=dash)

    async def main():
        await expr.emote("happy")
        await expr.update(0.05)

    asyncio.run(main())
    canvas = dash.canvas
    face = dash.panel(SimDashboard.FACE_RECT)
    stats = dash.panel(SimDashboard.STATS_RECT)
    cam = dash.panel(SimDashboard.CAM_RECT)
    gray_face = face.mean(axis=2)
    gray_stats = stats.mean(axis=2)
    # face panel: white head, black eyes; stats panel: white text pixels; cam untouched (black)
    assert gray_face[120, 240] > 200  # head
    assert gray_face[185, 168] < 50  # left eye
    assert gray_stats.max() > 200  # text drawn
    assert "llm=100ms" in state.last_latency
    assert cam.max() == 0  # camera panel not drawn by expression
    assert canvas.shape == (SimDashboard.HEIGHT, SimDashboard.WIDTH, 3)


def test_camera_draws_annotated_frame_into_cam_panel():
    dash = _dashboard()
    cam = SimCamera(preview=True, dashboard=dash, log=logging.getLogger("test"))
    cam._cv2 = dash.cv2
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # gray frame
    out = dash.panel(SimDashboard.CAM_RECT)
    assert out.shape[:2] == (SimDashboard.CAM_RECT[3], SimDashboard.CAM_RECT[2])
    cam._draw_frame(frame, out, [])  # no boxes: whole image area is the gray frame
    ph = out.shape[0]
    scale = min(out.shape[1] / 640, ph / 480)
    oy = (ph - int(480 * scale)) // 2
    assert abs(out[oy + 5].mean() - 200) < 5  # frame content fills the image area
    cam._draw_frame(frame, out, [("face", 320, 240, 100, 100)])  # boxes in frame coords
    green = ((out[:, :, 1] > 200) & (out[:, :, 0] < 100)).sum()
    assert green > 50  # detection box drawn
    # face panel untouched by camera
    assert dash.panel(SimDashboard.FACE_RECT).max() == 0


def test_camera_letterboxes_wide_frame():
    """A 16:9 frame in the cam panel must keep its aspect ratio (black bars)."""
    dash = _dashboard()
    cam = SimCamera(preview=True, dashboard=dash, log=logging.getLogger("test"))
    cam._cv2 = dash.cv2
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)  # 16:9
    out = dash.panel(SimDashboard.CAM_RECT)
    cam._draw_frame(frame, out, [])
    ph, pw = out.shape[:2]
    scale = min(pw / 1280, ph / 720)
    nh = int(720 * scale)
    top_bar = out[: (ph - nh) // 2]
    middle = out[ph // 2]
    assert top_bar.max() == 0  # letterbox bar is black
    assert abs(middle.mean() - 200) < 5  # image content in the middle


def test_box_coords_are_scaled_to_panel():
    """A box in full-res frame coords must land scaled+offset in the letterboxed panel."""
    dash = _dashboard()
    cam = SimCamera(preview=True, dashboard=dash, log=logging.getLogger("test"))
    cam._cv2 = dash.cv2
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = dash.panel(SimDashboard.CAM_RECT)
    cam._draw_frame(frame, out, [("face", 0, 0, 1280, 720)])  # full-frame box
    ph, pw = out.shape[:2]
    scale = min(pw / 1280, ph / 720)
    nw, nh = int(1280 * scale), int(720 * scale)
    oy, ox = (ph - nh) // 2, (pw - nw) // 2
    # corners of the drawn box should be green, outside the image area black
    green = (out[:, :, 1] > 200) & (out[:, :, 0] < 100)
    assert green[oy, ox] and green[oy + nh - 1, ox + nw - 1]
    assert not green[0, 0] if nh < ph else True


def test_window_keyboard_input_submits_line():
    dash = SimDashboard(enabled=False)
    submitted = []
    dash.on_submit = submitted.append
    for key in (104, 105, 8, 108, 111, 13):  # "hi", backspace, "lo", Enter -> "hlo"
        dash._handle_key(key)
    assert submitted == ["hlo"]
    assert dash.input_buffer == ""
    dash._handle_key(-1)  # no key pressed: no-op
    assert submitted == ["hlo"]
