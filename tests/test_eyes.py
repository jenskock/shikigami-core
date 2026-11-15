"""Procedural eyes: springs, emotion blends, deterministic blink scheduling."""

from __future__ import annotations

import pytest

from shikigami.expression.eyes.controller import EyesEngine
from shikigami.expression.eyes.emotion import (
    Emotion,
    blend_emotions,
    emotion_targets,
    from_valence_arousal,
    scale_intensity,
)
from shikigami.expression.eyes.life import LifeLayer
from shikigami.expression.eyes.protocol import encode_emo, encode_gaze, parse_command
from shikigami.expression.eyes.spring import Spring
from shikigami.expression.eyes.state import EyesConfig


def test_spring_reaches_target_without_linear_tween():
    s = Spring(value=0.0, target=1.0, omega=14.0)
    # critically damped: approaches asymptotically, never a fixed-step linear jump
    samples = [s.step(1 / 60) for _ in range(120)]
    assert samples[0] < 0.35  # not a big linear leap on frame 1
    assert samples[-1] == pytest.approx(1.0, abs=1e-3)
    # monotonic approach for a rest start
    assert all(samples[i] <= samples[i + 1] + 1e-9 for i in range(len(samples) - 1))


def test_spring_overshoots_with_initial_velocity():
    s = Spring(value=0.0, velocity=20.0, target=1.0, omega=12.0)
    vals = [s.step(1 / 60) for _ in range(90)]
    assert max(vals) > 1.0  # overshoot from velocity kick


def test_emotion_presets_cover_required_names():
    required = {
        "neutral",
        "happy",
        "sad",
        "angry",
        "surprised",
        "sleepy",
        "curious",
        "love",
        "confused",
        "scared",
        "focused",
        "mischievous",
    }
    assert required == {e.value for e in Emotion}
    happy = emotion_targets("happy")
    sleepy = emotion_targets("sleepy")
    assert happy.left.bottom_lid_coverage > 0.3  # raised cheeks
    assert sleepy.left.height < happy.left.height


def test_blend_emotions_happy_sleepy():
    blended = blend_emotions({"happy": 0.6, "sleepy": 0.4})
    happy = emotion_targets("happy")
    sleepy = emotion_targets("sleepy")
    # height should sit between the two presets
    h = blended.left.height
    lo, hi = sorted([happy.left.height, sleepy.left.height])
    assert lo - 1e-6 <= h <= hi + 1e-6
    # bottom lid likewise between
    b = blended.left.bottom_lid_coverage
    lo, hi = sorted([happy.left.bottom_lid_coverage, sleepy.left.bottom_lid_coverage])
    assert lo - 1e-6 <= b <= hi + 1e-6


def test_valence_arousal_happy_corner():
    tgt = from_valence_arousal(0.85, 0.55)
    # should lean happy-ish: raised cheeks (bottom lid)
    assert tgt.left.bottom_lid_coverage > 0.2


def test_scale_intensity_zero_is_neutral():
    neu = emotion_targets("neutral")
    scaled = scale_intensity(emotion_targets("angry"), 0.0)
    assert scaled.left.top_lid_angle == pytest.approx(neu.left.top_lid_angle)
    assert scaled.left.top_lid_coverage == pytest.approx(neu.left.top_lid_coverage)


def test_blink_schedule_is_deterministic_with_seed():
    def blink_times(seed: int) -> list[float]:
        life = LifeLayer(EyesConfig(seed=seed, blink_rate_per_min=30.0, double_blink_prob=0.0))
        t = 0.0
        times: list[float] = []
        saw_close = False
        for _ in range(5000):
            prev = life.blink.phase
            life.update(0.01)
            t += 0.01
            if prev == "idle" and life.blink.phase == "closing":
                times.append(round(t, 4))
                saw_close = True
            if len(times) >= 5:
                break
        assert saw_close
        return times

    a = blink_times(123)
    b = blink_times(123)
    c = blink_times(999)
    assert a == b
    assert a != c


def test_engine_look_at_and_blink_update_frame():
    eng = EyesEngine(EyesConfig(seed=1, personality="calm"))
    eng.set_emotion("curious", intensity=1.0)
    eng.look_at(0.8, -0.3)
    for _ in range(30):
        eng.update(1 / 60)
    assert abs(eng.frame.left.pupil_offset_x) > 0.05
    eng.blink()
    blinked = False
    for _ in range(40):
        f = eng.update(1 / 60)
        # soft blink: lids rise and height squashes, but eyes stay visible
        if f.left.top_lid_coverage > 0.25 or f.left.height < 0.85:
            blinked = True
            break
    assert blinked
    # never fully black out
    eng2 = EyesEngine(EyesConfig(seed=3, personality="calm"))
    eng2.blink()
    for _ in range(30):
        f = eng2.update(1 / 60)
        assert f.left.height >= 0.35
        assert f.left.top_lid_coverage <= 0.85


def test_react_startle_sets_surprised():
    eng = EyesEngine(EyesConfig(seed=2))
    eng.react("startle")
    for _ in range(10):
        eng.update(1 / 60)
    # surprised eyes are taller
    assert eng.frame.left.height >= emotion_targets("neutral").left.height


def test_protocol_roundtrip():
    assert encode_emo("happy", 0.8).encode() == b"EMO happy 0.800\n"
    assert encode_gaze(0.2, -0.1).encode() == b"GAZE 0.2000 -0.1000\n"
    cmd = parse_command("BLINK\n")
    assert cmd is not None and cmd.name == "BLINK"


def test_pil_renderer_rgb565_size():
    pytest.importorskip("PIL")
    from shikigami.expression.eyes.render.pil_renderer import PILRenderer

    eng = EyesEngine(EyesConfig(seed=0))
    frame = eng.update(0.016)
    raw = PILRenderer(240, circular=True).draw_rgb565(frame)
    assert len(raw) == 240 * 240 * 2


def test_draw_frame_has_white_pixels():
    from shikigami.expression.eyes.render.numpy_draw import draw_frame_rgb

    eng = EyesEngine(EyesConfig(seed=0))
    eng.set_emotion("happy")
    for _ in range(40):
        eng.update(1 / 60)
    rgb = draw_frame_rgb(eng.frame, 480, 560)
    assert rgb[250:310, 90:170].max() > 200  # bright white eye body
    assert rgb.mean() > 1.0
