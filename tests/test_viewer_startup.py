"""Run the optional introduction's lifecycle checks without a browser dependency."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Startup behavior checks require Node >=18"
)
def test_viewer_startup_behavior():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [shutil.which("node"), "--test", "tests/viewer_startup.test.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
