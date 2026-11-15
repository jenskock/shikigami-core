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

# How often the dedicated Qt pump runs. Keystrokes only appear when processEvents
# runs — it must not wait on the ~20 Hz reflex/camera tick.
_PUMP_DT_S = 0.01

# Device face framebuffer (OLED-sim). Eyes render into this 1:1; UI only displays it.
FACE_W, FACE_H = 480, 560
CAM_W, CAM_H = 640, 360

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
QProgressBar {{
  background: {_BORDER}; border: none; border-radius: 4px; height: 8px; text-align: center;
}}
QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 4px; }}
QProgressBar#servo {{
  background: {_BORDER}; border: none; border-radius: 3px; height: 6px; max-width: 120px;
}}
QProgressBar#servo::chunk {{ background: {_ACCENT}; border-radius: 3px; }}
QLabel#servo_key {{ color: {_MUTED}; font-size: 11px; min-width: 72px; }}
QLabel#servo_val {{ color: {_TEXT}; font-size: 11px; min-width: 40px; }}
"""

# Servo angle display range for progress bars (degrees).
_SERVO_DEG_MIN = -90.0
_SERVO_DEG_MAX = 90.0


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
        self._labels: dict[str, Any] = {}
        self._servo_bars: list[Any] = []
        self._servo_vals: list[Any] = []
        self._servo_keys: list[Any] = []
        self._servos_panel: Any | None = None
        self.state: Any | None = None
        self.emotion: str = "neutral"
        self.face_buf = np.zeros((FACE_H, FACE_W, 3), dtype=np.uint8)
        self.cam_buf = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        self.on_submit: Callable[[str], None] | None = None
        self.on_close: Callable[[], None] | None = None
        self._input_fallback = ""
        self._pump_task: asyncio.Task | None = None
        self._face_dirty = True
        self._cam_dirty = True

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
                QFrame,
                QHBoxLayout,
                QLabel,
                QLineEdit,
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
            win.resize(1520, 640)
            win.setMinimumSize(1100, 520)
            root = QHBoxLayout(win)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(12)

            # Face — exact device pixels
            face_panel = QFrame()
            face_panel.setObjectName("panel")
            face_l = QVBoxLayout(face_panel)
            face_title = QLabel("FACE")
            face_title.setObjectName("section")
            face_l.addWidget(face_title)
            self._face_label = QLabel()
            self._face_label.setFixedSize(FACE_W, FACE_H)
            self._face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            face_l.addWidget(self._face_label)
            root.addWidget(face_panel, 0)

            # Stats + input
            side = QFrame()
            side.setObjectName("panel")
            side.setMinimumWidth(260)
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
            side_l.addStretch(1)
            root.addWidget(side, 1)

            # Servos — 12 joints (picrawler); empty until motion publishes angles
            servos = QFrame()
            servos.setObjectName("panel")
            servos.setMinimumWidth(220)
            servos_l = QVBoxLayout(servos)
            servos_title = QLabel("SERVOS")
            servos_title.setObjectName("section")
            servos_l.addWidget(servos_title)
            self._servo_bars = []
            self._servo_vals = []
            self._servo_keys = []
            for i in range(12):
                row = QHBoxLayout()
                k = QLabel(f"S{i}")
                k.setObjectName("servo_key")
                bar = QProgressBar()
                bar.setObjectName("servo")
                bar.setRange(0, 100)
                bar.setTextVisible(False)
                bar.setFixedHeight(6)
                v = QLabel("—")
                v.setObjectName("servo_val")
                row.addWidget(k)
                row.addWidget(bar, 1)
                row.addWidget(v)
                servos_l.addLayout(row)
                self._servo_keys.append(k)
                self._servo_bars.append(bar)
                self._servo_vals.append(v)
            servos_l.addStretch(1)
            self._servos_panel = servos
            root.addWidget(servos, 0)

            # Camera
            cam_panel = QFrame()
            cam_panel.setObjectName("panel")
            cam_l = QVBoxLayout(cam_panel)
            cam_title = QLabel("CAMERA")
            cam_title.setObjectName("section")
            cam_l.addWidget(cam_title)
            self._cam_label = QLabel()
            self._cam_label.setFixedSize(CAM_W, CAM_H)
            self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cam_l.addWidget(self._cam_label, 1)
            root.addWidget(cam_panel, 2)

            win.closeEvent = self._on_close_event  # type: ignore[method-assign]
            from PySide6.QtGui import QKeySequence, QShortcut

            # Esc only — a "Q" shortcut would steal the letter from the chat field.
            QShortcut(QKeySequence(Qt.Key.Key_Escape), win, self._request_close)
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
        """Keep Qt responsive (typing, resize, quit) independent of reflex/camera cost."""
        try:
            while not self._closed and self._app is not None:
                self.pump()
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
        mapping = {
            "backend": f"backend · {st.backend}",
            "mood": st.mood or "—",
            "face": self.emotion or "—",
            "motion": st.last_motion or "—",
            "status": st.status or "—",
            "heard": (st.last_heard or "—")[:48],
            "latency": (st.last_latency or "—")[:48],
            "mic": st.mic or "—",
        }
        for key, value in mapping.items():
            lab = self._labels.get(key)
            if lab is not None and lab.text() != value:
                lab.setText(value)
        if self._mic_bar is not None:
            self._mic_bar.setValue(int(max(0.0, min(1.0, float(st.mic_level))) * 100))
        self._sync_servos(st)

    def _sync_servos(self, st: Any) -> None:
        if not self._servo_bars:
            return
        angles = getattr(st, "servo_angles", None) or []
        labels = getattr(st, "servo_labels", None) or []
        span = _SERVO_DEG_MAX - _SERVO_DEG_MIN
        for i in range(12):
            key = self._servo_keys[i]
            bar = self._servo_bars[i]
            val = self._servo_vals[i]
            label = labels[i] if i < len(labels) else f"S{i}"
            if key.text() != label:
                key.setText(label)
            if i < len(angles):
                deg = float(angles[i])
                pct = int(max(0.0, min(1.0, (deg - _SERVO_DEG_MIN) / span)) * 100)
                txt = f"{deg:+.0f}°"
                if bar.value() != pct:
                    bar.setValue(pct)
                if val.text() != txt:
                    val.setText(txt)
            else:
                if bar.value() != 0:
                    bar.setValue(0)
                if val.text() != "—":
                    val.setText("—")

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
