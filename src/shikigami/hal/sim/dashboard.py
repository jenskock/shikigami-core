"""Sim dashboard (Qt / PySide6): modern chrome around the device face buffer.

The face is a fixed-resolution BGR numpy framebuffer drawn with the same OpenCV
path the Pi OLED uses. Camera, stats, and chat are real Qt widgets.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from .body_view import BodyView
from .state import SimState

# How often the dedicated Qt pump runs. Keystrokes only appear when processEvents
# runs — it must not wait on the ~20 Hz reflex/camera tick.
_PUMP_DT_S = 0.01

# Device face framebuffer (OLED-sim). Eyes render into this 1:1; UI only displays it.
FACE_W, FACE_H = 480, 560
CAM_W, CAM_H = 400, 225

_BG = "#12141a"
_PANEL = "#1a1d26"
_TEXT = "#e8eaef"
_MUTED = "#8b90a0"
_ACCENT = "#3dd6c6"
_DANGER = "#e85d5d"
_FIELD = "#10121a"
_BORDER = "#2f3444"

_STYLESHEET = f"""
QWidget {{ background-color: {_BG}; color: {_TEXT}; font-family: Helvetica, Arial, sans-serif; }}
QFrame#panel {{ background-color: {_PANEL}; border-radius: 12px; }}
QLabel#section {{ color: {_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#brand {{ color: {_ACCENT}; font-size: 22px; font-weight: 700; }}
QLabel#muted {{ color: {_MUTED}; font-size: 12px; }}
QLabel#key {{ color: {_MUTED}; font-size: 12px; min-width: 64px; }}
QLabel#val {{ color: {_TEXT}; font-size: 13px; }}
QLineEdit {{
  background: {_FIELD}; border: 1px solid {_BORDER}; border-radius: 8px;
  padding: 8px 10px; color: {_TEXT}; selection-background-color: {_ACCENT};
}}
QPushButton#quit {{
  background: {_BORDER}; color: {_TEXT}; border: none; border-radius: 8px; padding: 10px;
}}
QPushButton#quit:hover {{ background: {_DANGER}; }}
QPushButton#mute {{
  background: {_BORDER}; color: {_TEXT}; border: none; border-radius: 8px; padding: 8px 10px;
}}
QPushButton#mute:hover {{ background: {_FIELD}; }}
QPushButton#mute[muted="true"] {{
  background: {_DANGER}; color: {_TEXT};
}}
QProgressBar {{
  background: {_BORDER}; border: none; border-radius: 4px; height: 8px; text-align: center;
}}
QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 4px; }}
QProgressBar#muted::chunk {{ background: {_MUTED}; border-radius: 4px; }}
QPlainTextEdit#tools {{
  background: {_FIELD}; border: 1px solid {_BORDER}; border-radius: 8px;
  padding: 6px 8px; color: {_TEXT}; font-family: Menlo, Monaco, monospace;
  font-size: 11px;
}}
QComboBox#speed {{
  background: {_FIELD}; border: 1px solid {_BORDER}; border-radius: 6px;
  padding: 4px 8px; color: {_TEXT}; min-width: 72px;
}}
QComboBox#speed::drop-down {{ border: none; }}
QComboBox#speed QAbstractItemView {{
  background: {_FIELD}; color: {_TEXT}; selection-background-color: {_ACCENT};
}}
"""


class SimDashboard:
    FACE_W = FACE_W
    FACE_H = FACE_H
    CAM_W = CAM_W
    CAM_H = CAM_H

    def __init__(self, enabled: bool, log: logging.Logger | None = None) -> None:
        self._log = log or logging.getLogger("shikigami.sim.dashboard")
        self._enabled = enabled
        self._closed = False
        self._app: Any | None = None
        self._win: Any | None = None
        self._face_label: Any | None = None
        self._cam_label: Any | None = None
        self._entry: Any | None = None
        self._mic_bar: Any | None = None
        self._mute_btn: Any | None = None
        self._labels: dict[str, Any] = {}
        self._body_view: BodyView | None = None
        self._speed_combo: Any | None = None
        self._tools_view: Any | None = None
        self._tools_len: int = -1
        self.state: SimState | None = None
        self.emotion: str = "neutral"
        self.face_buf = np.zeros((FACE_H, FACE_W, 3), dtype=np.uint8)
        self.cam_buf = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        self.on_submit: Callable[[str], None] | None = None
        self.on_close: Callable[[], None] | None = None
        self.on_mute_toggle: Callable[[], None] | None = None
        self._input_fallback = ""
        self._pump_task: asyncio.Task | None = None
        self._face_dirty = True
        self._cam_dirty = True
        self._mic_bar_muted: bool | None = None

        if enabled:
            try:
                from PySide6.QtWidgets import QApplication  # noqa: F401
            except ImportError:
                self._enabled = False
                self._log.warning("PySide6 missing — log-only dashboard (`uv sync`)")

    @property
    def active(self) -> bool:
        return self._enabled and not self._closed

    @property
    def input_buffer(self) -> str:
        if self._entry is not None:
            return self._entry.text()
        return self._input_fallback

    @input_buffer.setter
    def input_buffer(self, value: str) -> None:
        self._input_fallback = value
        if self._entry is not None:
            self._entry.setText(value)

    async def start(self) -> None:
        if not self._enabled:
            return
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import (
                QApplication,
                QComboBox,
                QFrame,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QPlainTextEdit,
                QProgressBar,
                QPushButton,
                QVBoxLayout,
                QWidget,
            )

            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            app.setStyleSheet(_STYLESHEET)
            self._app = app

            win = QWidget()
            win.setWindowTitle("shikigami")
            win.resize(1280, 720)
            win.setMinimumSize(1080, 560)
            root = QHBoxLayout(win)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(12)

            # Face — exact device pixels, centered in panel
            face_panel = QFrame()
            face_panel.setObjectName("panel")
            face_panel.setFixedWidth(FACE_W + 32)
            face_l = QVBoxLayout(face_panel)
            face_l.setContentsMargins(16, 16, 16, 16)
            face_l.setSpacing(8)
            face_title = QLabel("FACE")
            face_title.setObjectName("section")
            face_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            self._face_label = QLabel()
            self._face_label.setFixedSize(FACE_W, FACE_H)
            self._face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._face_label.setScaledContents(True)
            face_l.addStretch(1)
            face_l.addWidget(face_title, 0, Qt.AlignmentFlag.AlignHCenter)
            face_l.addWidget(self._face_label, 0, Qt.AlignmentFlag.AlignHCenter)
            face_l.addStretch(1)
            root.addWidget(face_panel, 0)

            # Stats + tools + input
            side = QFrame()
            side.setObjectName("panel")
            side.setMinimumWidth(300)
            side_l = QVBoxLayout(side)
            brand = QLabel("shikigami")
            brand.setObjectName("brand")
            side_l.addWidget(brand)
            self._labels["backend"] = QLabel("backend · sim")
            self._labels["backend"].setObjectName("muted")
            side_l.addWidget(self._labels["backend"])
            side_l.addSpacing(8)
            for key, title in (
                ("mood", "Mood"),
                ("face", "Face"),
                ("motion", "Motion"),
                ("status", "Status"),
                ("heard", "Heard"),
                ("latency", "Latency"),
                ("mic", "Mic"),
            ):
                row = QHBoxLayout()
                k = QLabel(title)
                k.setObjectName("key")
                v = QLabel("—")
                v.setObjectName("val")
                v.setWordWrap(True)
                row.addWidget(k)
                row.addWidget(v, 1)
                side_l.addLayout(row)
                self._labels[key] = v

            lvl = QLabel("LEVEL")
            lvl.setObjectName("section")
            side_l.addSpacing(10)
            side_l.addWidget(lvl)
            self._mic_bar = QProgressBar()
            self._mic_bar.setRange(0, 100)
            self._mic_bar.setTextVisible(False)
            self._mic_bar.setFixedHeight(8)
            side_l.addWidget(self._mic_bar)
            self._mute_btn = QPushButton("Mute mic  (Ctrl+M)")
            self._mute_btn.setObjectName("mute")
            self._mute_btn.setProperty("muted", "false")
            self._mute_btn.clicked.connect(self.toggle_mute)
            side_l.addWidget(self._mute_btn)

            tools_title = QLabel("TOOLS")
            tools_title.setObjectName("section")
            side_l.addSpacing(10)
            side_l.addWidget(tools_title)
            self._tools_view = QPlainTextEdit()
            self._tools_view.setObjectName("tools")
            self._tools_view.setReadOnly(True)
            self._tools_view.setPlaceholderText("tool calls appear here as the LLM fires them")
            self._tools_view.setMinimumHeight(160)
            side_l.addWidget(self._tools_view, 1)

            say = QLabel("SAY SOMETHING")
            say.setObjectName("section")
            side_l.addSpacing(10)
            side_l.addWidget(say)
            self._entry = QLineEdit()
            self._entry.setPlaceholderText("type + Enter")
            self._entry.returnPressed.connect(self._on_enter)
            side_l.addWidget(self._entry)

            quit_btn = QPushButton("Quit")
            quit_btn.setObjectName("quit")
            quit_btn.clicked.connect(self._request_close)
            side_l.addSpacing(8)
            side_l.addWidget(quit_btn)
            root.addWidget(side, 1)

            # Body + camera stacked on the right
            right = QVBoxLayout()
            right.setSpacing(12)

            body = QFrame()
            body.setObjectName("panel")
            body.setFixedWidth(CAM_W + 32)
            body_l = QVBoxLayout(body)
            body_l.setContentsMargins(16, 16, 16, 16)
            body_l.setSpacing(8)
            body_header = QHBoxLayout()
            body_title = QLabel("BODY")
            body_title.setObjectName("section")
            body_header.addWidget(body_title, 0, Qt.AlignmentFlag.AlignVCenter)
            body_header.addStretch(1)
            speed_label = QLabel("Speed")
            speed_label.setObjectName("muted")
            body_header.addWidget(speed_label, 0, Qt.AlignmentFlag.AlignVCenter)
            speed = QComboBox()
            speed.setObjectName("speed")
            for label, value in (("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0)):
                speed.addItem(label, value)
            speed.setCurrentIndex(2)  # 1×
            speed.currentIndexChanged.connect(self._on_playback_speed)
            self._speed_combo = speed
            body_header.addWidget(speed, 0, Qt.AlignmentFlag.AlignVCenter)
            body_l.addLayout(body_header)
            self._body_view = BodyView()
            self._body_view.setMinimumHeight(200)
            body_l.addWidget(self._body_view, 1)
            right.addWidget(body, 1)

            cam_panel = QFrame()
            cam_panel.setObjectName("panel")
            cam_panel.setFixedWidth(CAM_W + 32)
            cam_l = QVBoxLayout(cam_panel)
            cam_l.setContentsMargins(16, 16, 16, 16)
            cam_l.setSpacing(8)
            cam_title = QLabel("CAMERA")
            cam_title.setObjectName("section")
            cam_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            self._cam_label = QLabel()
            self._cam_label.setFixedSize(CAM_W, CAM_H)
            self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._cam_label.setScaledContents(True)
            cam_l.addWidget(cam_title, 0, Qt.AlignmentFlag.AlignHCenter)
            cam_l.addWidget(self._cam_label, 0, Qt.AlignmentFlag.AlignHCenter)
            right.addWidget(cam_panel, 0)

            root.addLayout(right, 0)

            win.closeEvent = self._on_close_event  # type: ignore[method-assign]
            from PySide6.QtGui import QKeySequence, QShortcut

            # Esc / Ctrl+M — bare letter shortcuts would steal keys from the chat field.
            QShortcut(QKeySequence(Qt.Key.Key_Escape), win, self._request_close)
            mute_sc = QShortcut(QKeySequence("Ctrl+M"), win)
            mute_sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            mute_sc.activated.connect(self.toggle_mute)
            self._win = win
            win.show()
            app.processEvents()
            self._pump_task = asyncio.create_task(self._pump_loop(), name="qt-pump")
            self._log.info("[sim] dashboard (Qt) opened")
        except Exception as exc:
            self._enabled = False
            self._win = None
            self._app = None
            self._log.warning("cannot open dashboard (%s) — log-only mode", exc)

    async def _pump_loop(self) -> None:
        """Keep Qt responsive and BODY pose at ~display rate during soft gait."""
        try:
            while not self._closed and self._app is not None:
                self.pump()
                # SoftPicrawler publishes poses on a worker thread; sync here so
                # interpolated frames paint even when the reflex loop is slower.
                st = self.state
                if st is not None:
                    self._sync_body(st)
                await asyncio.sleep(_PUMP_DT_S)
        except asyncio.CancelledError:
            raise

    def pump(self) -> None:
        """Flush Qt input/paint events. Safe to call often; does not push textures."""
        if self._closed or self._app is None:
            return
        self._app.processEvents()
        if self._win is not None and not self._win.isVisible():
            self._request_close()

    def mark_face_dirty(self) -> None:
        self._face_dirty = True

    def set_camera(self, bgr: np.ndarray) -> None:
        h, w = self.cam_buf.shape[:2]
        if bgr.shape[0] == h and bgr.shape[1] == w:
            np.copyto(self.cam_buf, bgr)
        else:
            self.cam_buf[:] = cv2.resize(bgr, (w, h))
        self._cam_dirty = True

    def show(self) -> None:
        """Push dirty textures + stats. Event pumping is handled by `_pump_loop`."""
        if self._closed or not self._enabled or self._app is None:
            return
        self._sync_widgets()
        self._push_images()
        self.pump()

    def _sync_widgets(self) -> None:
        st = self.state
        if st is None:
            return
        muted = bool(st.mic_muted)
        mic_text = "muted" if muted else (st.mic or "—")
        mapping = {
            "backend": f"backend · {st.backend}",
            "mood": st.mood or "—",
            "face": self.emotion or "—",
            "motion": st.last_motion or "—",
            "status": st.status or "—",
            "heard": (st.last_heard or "—")[:48],
            "latency": (st.last_latency or "—")[:48],
            "mic": mic_text,
        }
        for key, value in mapping.items():
            lab = self._labels.get(key)
            if lab is not None and lab.text() != value:
                lab.setText(value)
        if self._mic_bar is not None:
            level = 0.0 if muted else float(st.mic_level)
            self._mic_bar.setValue(int(max(0.0, min(1.0, level)) * 100))
            if self._mic_bar_muted is not muted:
                self._mic_bar_muted = muted
                self._mic_bar.setObjectName("muted" if muted else "")
                # Re-polish so #muted stylesheet applies without recreating the bar.
                self._mic_bar.style().unpolish(self._mic_bar)
                self._mic_bar.style().polish(self._mic_bar)
        self._sync_mute_button(muted)
        self._sync_body(st)
        self._sync_tools(st)

    def _sync_tools(self, st: SimState) -> None:
        view = self._tools_view
        if view is None:
            return
        lines = st.tool_log
        n = len(lines)
        if n == self._tools_len:
            return
        self._tools_len = n
        text = "\n".join(lines) if lines else ""
        # Preserve manual scroll unless the user is already at the bottom.
        bar = view.verticalScrollBar()
        at_bottom = bar is None or bar.value() >= bar.maximum() - 4
        view.setPlainText(text)
        if at_bottom and bar is not None:
            bar.setValue(bar.maximum())

    def _sync_mute_button(self, muted: bool) -> None:
        btn = self._mute_btn
        if btn is None:
            return
        label = "Unmute mic  (Ctrl+M)" if muted else "Mute mic  (Ctrl+M)"
        flag = "true" if muted else "false"
        if btn.text() != label:
            btn.setText(label)
        if btn.property("muted") != flag:
            btn.setProperty("muted", flag)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_playback_speed(self, _index: int = 0) -> None:
        combo = self._speed_combo
        st = self.state
        if combo is None or st is None:
            return
        value = combo.currentData()
        if value is None:
            return
        st.playback_speed = float(value)

    def _sync_body(self, st: SimState) -> None:
        view = self._body_view
        if view is None:
            return
        view.set_body_pose(getattr(st, "body_pose", None))

    def _push_images(self) -> None:
        if self._face_dirty and self._face_label is not None:
            self._face_label.setPixmap(self._bgr_to_pixmap(self.face_buf))
            self._face_dirty = False
        if self._cam_dirty and self._cam_label is not None:
            self._cam_label.setPixmap(self._bgr_to_pixmap(self.cam_buf))
            self._cam_dirty = False

    @staticmethod
    def _bgr_to_pixmap(bgr: np.ndarray) -> Any:
        from PySide6.QtGui import QImage, QPixmap

        # BGR888 avoids a full-frame channel swap; copy so Qt owns the pixels.
        buf = np.ascontiguousarray(bgr)
        h, w, _ = buf.shape
        qimg = QImage(buf.data, w, h, w * 3, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(qimg.copy())

    def _on_enter(self) -> None:
        if self._entry is None:
            return
        line = self._entry.text().strip()
        self._entry.clear()
        self._input_fallback = ""
        if line and self.on_submit is not None:
            self.on_submit(line)

    def toggle_mute(self) -> None:
        """Dashboard Mute button / Ctrl+M — app handles state + audio abort."""
        if self.on_mute_toggle is not None:
            self.on_mute_toggle()

    def _on_close_event(self, event: Any) -> None:
        self._request_close()
        event.accept()

    def _request_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.on_close is not None:
            self.on_close()
        if self._win is not None:
            self._win.hide()
            self._win = None

    def submit_line(self, line: str) -> None:
        line = line.strip()
        if line and self.on_submit is not None:
            self.on_submit(line)

    async def stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
            self._pump_task = None
        if self._closed and self._win is None:
            return
        self._closed = True
        if self._win is not None:
            self._win.close()
            self._win = None
