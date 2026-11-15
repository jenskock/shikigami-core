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
    if ry < 2.5:
        _fill_round_rect(img, cx - rx, cy - 2, cx + rx, cy + 2, 2, eye.color_rgb)
        return

    # soft glow (slightly larger, dimmer)
    if eye.glow > 0.05:
        g = 1.0 + 0.22 * eye.glow
        glow_c = (
            max(0, int(eye.color_r * 0.28 * eye.glow)),
            max(0, int(eye.color_g * 0.28 * eye.glow)),
            max(0, int(eye.color_b * 0.28 * eye.glow)),
        )
        _fill_round_rect(
            img,
            cx - rx * g,
            cy - ry * g,
            cx + rx * g,
            cy + ry * g,
            ry * g * min(1.0, eye.corner_radius + 0.1),
            glow_c,
        )

    x1, y1, x2, y2 = cx - rx, cy - ry, cx + rx, cy + ry
    _fill_round_rect(img, x1, y1, x2, y2, ry * eye.corner_radius, eye.color_rgb)

    # lids (black carve) — approximate angled top lid with a sheared rect
    if eye.top_lid_coverage > 0.02:
        cover = ry * 2 * eye.top_lid_coverage
        # angle shifts left/right lid edge
        skew = math.tan(math.radians(eye.top_lid_angle)) * rx * 0.35
        _fill_round_rect(img, x1 - 2, y1 - 3, x2 + 2, y1 + cover, 4, (0, 0, 0))
        if abs(skew) > 0.5:
            # extra wedge for brow angle
            mid = y1 + cover * 0.55
            if skew > 0:
                _fill_round_rect(img, x1 - 2, y1 - 2, cx + skew, mid, 3, (0, 0, 0))
            else:
                _fill_round_rect(img, cx + skew, y1 - 2, x2 + 2, mid, 3, (0, 0, 0))

    if eye.bottom_lid_coverage > 0.02:
        cover = ry * 2 * eye.bottom_lid_coverage
        _fill_round_rect(img, x1 - 2, y2 - cover, x2 + 2, y2 + 3, 4, (0, 0, 0))

    open_top = y1 + ry * 2 * eye.top_lid_coverage
    open_bot = y2 - ry * 2 * eye.bottom_lid_coverage
    if open_bot - open_top < 4:
        return

    if heart > 0.4:
        _draw_heart(img, cx, (open_top + open_bot) / 2, min(rx, ry) * 0.55, eye.color_rgb)
        return

    prx = max(3.0, rx * 0.28)
    pry = max(3.0, min(ry * 0.7, (open_bot - open_top) * 0.35))
    mid_y = (open_top + open_bot) / 2
    px = cx + eye.pupil_offset_x * (rx - prx) * 0.65
    py = mid_y + eye.pupil_offset_y * ((open_bot - open_top) / 2 - pry) * 0.65
    _fill_round_rect(img, px - prx, py - pry, px + prx, py + pry, pry * 0.95, _PUPIL)
    gx = px + eye.highlight_offset_x * prx
    gy = py + eye.highlight_offset_y * pry
    gr = max(1.5, prx * 0.28)
    _fill_round_rect(img, gx - gr, gy - gr, gx + gr, gy + gr, gr, _GLINT)


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

    # layout: two capsules centered, matching prior sim proportions
    cx = width / 2.0
    cy = height / 2.0
    # base radii scale with the shorter dimension (works for 240 and 480×560)
    base = min(width, height)
    eye_rx = 0.135 * base * frame.left.width * frame.left.scale
    eye_ry = 0.072 * base * frame.left.height  # height already includes squash scale
    # use per-eye sizes
    gap = 0.185 * base * frame.gap

    _draw_one_eye(
        img,
        cx - gap,
        cy + frame.left.y * eye_ry,
        eye_rx,
        max(2.0, 0.072 * base * frame.left.height),
        frame.left,
        heart=frame.heart_eyes,
    )
    eye_rx_r = 0.135 * base * frame.right.width * frame.right.scale
    _draw_one_eye(
        img,
        cx + gap,
        cy + frame.right.y * eye_ry,
        eye_rx_r,
        max(2.0, 0.072 * base * frame.right.height),
        frame.right,
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
