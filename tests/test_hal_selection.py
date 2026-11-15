"""HAL selection: backend resolution + import guards for SunFounder libs."""

import sys

import pytest

from shikigami.config import load_config
from shikigami.hal import create_hal, resolve_backend
from shikigami.hal.errors import BackendUnavailable
from shikigami.hal.interfaces import HAL
from shikigami.hal.sim.backend import SimHAL


def test_resolve_backend_default():
    assert resolve_backend("sim") == "sim"
    assert resolve_backend(None) == "sim"


def test_resolve_backend_env_wins(monkeypatch):
    monkeypatch.setenv("SHIKIGAMI_BACKEND", "PiDog")
    assert resolve_backend("sim") == "pidog"


def test_resolve_backend_invalid(monkeypatch):
    monkeypatch.setenv("SHIKIGAMI_BACKEND", "nope")
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend(None)


def test_create_sim_hal():
    cfg = load_config()
    cfg.sim.window = False
    hal = create_hal("sim", cfg)
    assert isinstance(hal, SimHAL)
    assert isinstance(hal, HAL)
    assert hal.backend == "sim"
    assert hal.motion and hal.camera and hal.audio and hal.sensors and hal.expression


def test_vendor_libs_not_imported_by_sim():
    create_hal("sim", load_config())
    # the SunFounder packages must never be imported on a dev machine
    assert "pidog" not in sys.modules
    assert "picrawler" not in sys.modules
    assert "robot_hat" not in sys.modules
    assert "vilib" not in sys.modules


def test_create_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        create_hal("atlas", load_config())


def test_pidog_backend_guarded_when_libs_missing():
    if "pidog" in sys.modules:  # running on a Pi with the lib installed
        pytest.skip("pidog lib present on this host")
    with pytest.raises(BackendUnavailable, match="SunFounder"):
        create_hal("pidog", load_config())


def test_picrawler_backend_guarded_when_libs_missing():
    if "picrawler" in sys.modules:
        pytest.skip("picrawler lib present on this host")
    with pytest.raises(BackendUnavailable, match="SunFounder"):
        create_hal("picrawler", load_config())
