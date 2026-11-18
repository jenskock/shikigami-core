"""Dashboard: device face buffer + camera annotate; Qt chrome optional."""

import asyncio

import pytest

from shikigami.hal.camera_base import draw_preview
from shikigami.hal.sim.dashboard import CAM_H, CAM_W, FACE_H, FACE_W, SimDashboard
from shikigami.hal.sim.expression import SimExpression
from shikigami.hal.sim.state import SimState
from shikigami.perception.detectors import Detection

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
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    out = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    draw_preview(frame, out, [])
    ph, pw = out.shape[:2]
    scale = min(pw / 640, ph / 480)
    nw, nh = int(640 * scale), int(480 * scale)
    ox, oy = (pw - nw) // 2, (ph - nh) // 2
    # content region is gray; side/letterbox bars stay black
    assert abs(out[oy + 5, ox + 5].mean() - 200) < 5
    if ox > 0:
        assert out[oy + 5, 0].mean() == 0
    draw_preview(frame, out, [Detection("face", 0.5, 0.5, 100 / 640, 100 / 480)])
    green = ((out[:, :, 1] > 200) & (out[:, :, 0] < 100)).sum()
    assert green > 50
    assert dash.face_buf.max() == 0


def test_camera_letterboxes_to_cam_aspect():
    """Non-matching aspect keeps black bars; content centered."""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # 4:3 into 16:9 cam buf
    out = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    draw_preview(frame, out, [])
    ph, pw = out.shape[:2]
    scale = min(pw / 640, ph / 480)
    nw = int(640 * scale)
    ox = (pw - nw) // 2
    assert ox > 0
    assert out[:, :ox].max() == 0
    assert abs(out[ph // 2, pw // 2].mean() - 200) < 5


def test_box_coords_are_scaled_to_panel():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    draw_preview(frame, out, [Detection("face", 0.0, 0.0, 1.0, 1.0)])
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


def test_toggle_mute_fires_callback():
    dash = SimDashboard(enabled=False)
    calls = []
    dash.on_mute_toggle = lambda: calls.append(True)
    dash.toggle_mute()
    assert calls == [True]


def test_sync_widgets_shows_muted():
    dash = SimDashboard(enabled=False)
    st = SimState()
    st.mic = "listening…"
    st.mic_level = 0.8
    st.mic_muted = True
    dash.state = st
    dash._labels = {
        "backend": _FakeLabel(),
        "mood": _FakeLabel(),
        "face": _FakeLabel(),
        "motion": _FakeLabel(),
        "status": _FakeLabel(),
        "heard": _FakeLabel(),
        "latency": _FakeLabel(),
        "mic": _FakeLabel(),
    }
    dash._mic_bar = _FakeBar()
    dash._mute_btn = _FakeMuteBtn()
    dash._sync_widgets()
    assert dash._labels["mic"].text() == "muted"
    assert dash._mic_bar.value() == 0
    assert dash._mute_btn.text().startswith("Unmute")


def test_sync_tools_shows_tool_log():
    dash = SimDashboard(enabled=False)
    st = SimState()
    st.append_tool("── turn · can you blink? ──")
    st.append_tool("emote(name='happy')")
    st.append_tool("say(text='Of course!')")
    dash.state = st
    dash._tools_view = _FakeToolsView()
    dash._sync_widgets()
    text = dash._tools_view.toPlainText()
    assert "emote(name='happy')" in text
    assert "say(text=" in text
    assert dash._tools_len == 3
    dash._sync_widgets()  # unchanged log → no rewrite
    assert dash._tools_view.writes == 1


def test_append_tool_trims():
    st = SimState()
    for i in range(250):
        st.append_tool(f"emote(name='n{i}')")
    assert len(st.tool_log) == 200
    assert st.tool_log[0] == "emote(name='n50')"


class _FakeLabel:
    def __init__(self) -> None:
        self._t = ""

    def text(self) -> str:
        return self._t

    def setText(self, t: str) -> None:
        self._t = t


class _FakeBar:
    def __init__(self) -> None:
        self._v = -1
        self._name = ""

    def value(self) -> int:
        return self._v

    def setValue(self, v: int) -> None:
        self._v = v

    def setObjectName(self, name: str) -> None:
        self._name = name

    def style(self):
        return self

    def unpolish(self, _w) -> None:
        return None

    def polish(self, _w) -> None:
        return None


class _FakeMuteBtn(_FakeLabel):
    def __init__(self) -> None:
        super().__init__()
        self._props: dict[str, str] = {}

    def property(self, key: str):
        return self._props.get(key)

    def setProperty(self, key: str, value: str) -> None:
        self._props[key] = value

    def style(self):
        return self

    def unpolish(self, _w) -> None:
        return None

    def polish(self, _w) -> None:
        return None


class _FakeToolsView:
    def __init__(self) -> None:
        self._t = ""
        self.writes = 0

    def toPlainText(self) -> str:
        return self._t

    def setPlainText(self, t: str) -> None:
        self._t = t
        self.writes += 1

    def verticalScrollBar(self):
        return None


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
