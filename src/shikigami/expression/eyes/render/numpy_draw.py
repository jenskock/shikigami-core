"""Fast RGB frame painter for pill eyes (shared by PIL / pygame / cv2 adapters)."""

from __future__ import annotations

import math

import numpy as np

from ..state import EyeState, FrameState

# Pupil / glint (RGB)
_PUPIL = np.array([25, 22, 5], dtype=np.uint8)
_GLINT = np.array([255, 255, 245], dtype=np.uint8)
_BG = np.array([0, 0, 0], dtype=np.uint8)


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, ::-1].copy()


def to_rgb565(rgb: np.ndarray) -> bytes:
    """Pack HxWx3 uint8 RGB into RGB565 big-endian bytes (GC9A01-friendly)."""
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    packed = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return packed.astype(">u2").tobytes()


def _circle_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = min(cx, cy) ** 2
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r2


# cache circular masks by size
_MASK_CACHE: dict[tuple[int, int], np.ndarray] = {}


def circular_mask(h: int, w: int) -> np.ndarray:
    key = (h, w)
    m = _MASK_CACHE.get(key)
    if m is None:
        m = _circle_mask(h, w)
        _MASK_CACHE[key] = m
    return m


def _fill_round_rect(
    img: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    color: tuple[int, int, int] | np.ndarray,
) -> None:
    h, w = img.shape[:2]
    x1i, y1i = int(math.floor(x1)), int(math.floor(y1))
    x2i, y2i = int(math.ceil(x2)), int(math.ceil(y2))
    if x2i <= x1i + 1 or y2i <= y1i + 1:
        return
    x1i, y1i = max(0, x1i), max(0, y1i)
    x2i, y2i = min(w, x2i), min(h, y2i)
    rw, rh = x2i - x1i, y2i - y1i
    if rw < 2 or rh < 2:
        return
    r = max(1, min(int(radius), rw // 2, rh // 2))
    # center rectangle + side slabs
    img[y1i:y2i, x1i + r : x2i - r] = color
    img[y1i + r : y2i - r, x1i:x2i] = color
    # corner circles
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    disk = xx * xx + yy * yy <= r * r
    corners = (
        (x1i + r, y1i + r),
        (x2i - r - 1, y1i + r),
        (x1i + r, y2i - r - 1),
        (x2i - r - 1, y2i - r - 1),
    )
    for cx, cy in corners:
        y0, y1c = cy - r, cy + r + 1
        x0, x1c = cx - r, cx + r + 1
        if y0 < 0 or x0 < 0 or y1c > h or x1c > w:
            # clipped corner
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy > r * r:
                        continue
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        img[py, px] = color
        else:
            patch = img[y0:y1c, x0:x1c]
            patch[disk] = color


def _blend(patch: np.ndarray, alpha: np.ndarray, color) -> None:
    """Alpha-composite a flat color onto an RGB uint8 patch, in place."""
    a = alpha[:, :, None]
    patch[:] = (patch * (1.0 - a) + np.asarray(color, dtype=np.float32) * a).astype(np.uint8)


def _draw_one_eye(
    img: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    eye: EyeState,
    *,
    heart: float = 0.0,
) -> None:
    """One eye: a rounded body, cut by a slanted upper lid and an arched lower lid.

    Everything is a signed distance on a small patch around the eye, so edges
    are smooth and a brow angle is a real slope, not a stack of rectangles.
    """
    h, w = img.shape[:2]
    rx, ry = max(2.0, rx), max(2.0, ry)
    margin = max(rx, ry) * 0.3 + 3.0
    x0, x1 = max(0, int(cx - rx - margin)), min(w, int(cx + rx + margin) + 1)
    y0, y1 = max(0, int(cy - ry - margin)), min(h, int(cy + ry + margin) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    patch = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    xx -= cx
    yy -= cy

    # Rounded rectangle body.
    r = min(rx, ry) * max(0.05, min(1.0, eye.corner_radius))
    qx = np.abs(xx) - (rx - r)
    qy = np.abs(yy) - (ry - r)
    dist = (
        np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - r
    )

    # Upper lid: a line through the eye, tilted by the brow angle.
    top_cov = max(0.0, min(1.0, eye.top_lid_coverage))
    lid_top = -ry + 2.0 * ry * top_cov + math.tan(math.radians(eye.top_lid_angle)) * xx
    # Lower lid: flat, or arched up in the middle like a smiling cheek.
    bot_cov = max(0.0, min(1.0, eye.bottom_lid_coverage))
    curve = max(0.0, min(1.5, eye.bottom_lid_curve))
    bottom_mid = ry - 2.0 * ry * bot_cov
    lid_bottom = bottom_mid + curve * 0.75 * ry * (xx / rx) ** 2
    if top_cov < 0.01 and abs(eye.top_lid_angle) < 0.5:
        lids = np.ones_like(xx)
    else:
        lids = np.clip(yy - lid_top + 0.5, 0.0, 1.0)
    if bot_cov >= 0.01 or curve >= 0.01:
        lids = lids * np.clip(lid_bottom - yy + 0.5, 0.0, 1.0)

    body = np.clip(0.5 - dist, 0.0, 1.0) * lids
    color = np.asarray(eye.color_rgb, dtype=np.float32)

    if eye.glow > 0.05:
        reach = max(rx, ry) * 0.3
        halo = np.clip(1.0 - np.maximum(dist, 0.0) / reach, 0.0, 1.0) ** 2
        soft_lids = np.clip((yy - lid_top) / reach + 1.0, 0.0, 1.0) * np.clip(
            (lid_bottom - yy) / reach + 1.0, 0.0, 1.0
        )
        _blend(patch, halo * soft_lids * 0.3 * eye.glow, color)
    _blend(patch, body, color)

    open_top = -ry + 2.0 * ry * top_cov
    opening = bottom_mid - open_top
    if opening < 4:
        return

    if heart > 0.4:
        _draw_heart(img, cx, cy + (open_top + bottom_mid) / 2, min(rx, ry) * 0.55, (20, 15, 30))
        return

    pupil = max(0.0, eye.pupil_scale)
    if pupil < 0.08:
        return
    prx = rx * 0.34 * pupil
    pry = min(ry * 0.4 * pupil, opening * 0.48)
    mid_y = (open_top + bottom_mid) / 2
    px = eye.pupil_offset_x * (rx - prx) * 0.7
    py = mid_y + eye.pupil_offset_y * max(0.0, opening / 2 - pry) * 0.8
    pd = np.hypot((xx - px) / max(1.0, prx), (yy - py) / max(1.0, pry))
    edge = 1.0 / max(1.0, min(prx, pry))
    _blend(patch, np.clip((1.0 - pd) / edge + 0.5, 0.0, 1.0) * body, _PUPIL)
    gr = max(1.5, prx * 0.3)
    gx = px + eye.highlight_offset_x * prx
    gy = py + eye.highlight_offset_y * pry
    gd = np.hypot(xx - gx, yy - gy)
    _blend(patch, np.clip(gr - gd + 0.5, 0.0, 1.0) * body, _GLINT)


def _draw_heart(
    img: np.ndarray, cx: float, cy: float, size: float, color: tuple[int, int, int]
) -> None:
    # two lobes + diamond — coarse but readable at 240px
    r = size * 0.42
    _fill_round_rect(
        img, cx - size * 0.75, cy - size * 0.35, cx - size * 0.05, cy + size * 0.35, r, color
    )
    _fill_round_rect(
        img, cx + size * 0.05, cy - size * 0.35, cx + size * 0.75, cy + size * 0.35, r, color
    )
    _fill_round_rect(
        img,
        cx - size * 0.55,
        cy - size * 0.05,
        cx + size * 0.55,
        cy + size * 0.75,
        size * 0.2,
        color,
    )


def draw_frame_rgb(
    frame: FrameState,
    width: int,
    height: int,
    *,
    circular: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Paint pill eyes into an RGB uint8 array of shape (height, width, 3)."""
    if out is None or out.shape != (height, width, 3):
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        img = out
        img[:] = 0

    # Two eyes around the centre, sized off the shorter side so the same
    # frame fits a 240 px round display and the 480x560 dashboard panel.
    cx = width / 2.0
    cy = height / 2.0
    base = min(width, height)
    gap = 0.185 * base * frame.gap
    eye_ry = 0.15 * base
    for eye, side in ((frame.left, -1.0), (frame.right, 1.0)):
        rx = 0.125 * base * eye.width * eye.scale
        ry = eye_ry * eye.height * eye.scale
        _draw_one_eye(
            img,
            cx + side * gap + eye.x * 0.125 * base,
            cy + eye.y * eye_ry,
            rx,
            ry,
            eye,
            heart=frame.heart_eyes,
        )

    if frame.sleep_zzz > 0.15:
        _draw_zzz(img, cx + gap * 1.35, cy - eye_ry * 2.2, frame.sleep_zzz, frame.time_s)

    if frame.dizzy > 0.2:
        _draw_dizzy_spiral(img, cx, cy, frame.dizzy, frame.time_s, base)

    if circular:
        mask = circular_mask(height, width)
        img[~mask] = 0

    return img


def _draw_zzz(img: np.ndarray, x: float, y: float, amp: float, t: float) -> None:
    # simple floating Z marks
    for i, ch_off in enumerate((0.0, 0.35, 0.7)):
        ox = x + i * 10
        oy = y - i * 12 - (t * 18 + ch_off * 10) % 20
        s = 4 + i * 2
        c = int(180 * amp)
        _fill_round_rect(img, ox, oy, ox + s, oy + 2, 1, (c, c, c))
        _fill_round_rect(img, ox + s - 2, oy, ox + s, oy + s, 1, (c, c, c))
        _fill_round_rect(img, ox, oy + s - 2, ox + s, oy + s, 1, (c, c, c))


def _draw_dizzy_spiral(
    img: np.ndarray, cx: float, cy: float, amp: float, t: float, base: float
) -> None:
    n = 18
    for i in range(n):
        a = t * 5.0 + i * 0.45
        r = (0.04 + i * 0.01) * base * amp
        px = cx + math.cos(a) * r
        py = cy + math.sin(a) * r
        c = int(200 * amp * (1.0 - i / n))
        _fill_round_rect(img, px - 1.5, py - 1.5, px + 1.5, py + 1.5, 1.5, (c, c, 40))
