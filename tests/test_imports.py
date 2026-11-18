"""Entry points import cleanly in a fresh interpreter (no circular imports).

Inside pytest, earlier tests have already loaded most modules, which hides an
import cycle; a subprocess starts from nothing, like `uv run sim` does.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["shikigami.__main__", "shikigami.hal.sim.body_view", "shikigami.hal.picrawler.backend"],
)
def test_module_imports_from_scratch(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
