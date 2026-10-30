"""Run dependency-free app state regressions when Node is available locally."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Frontend behavior checks require Node >=18"
)
@pytest.mark.parametrize(
    "suite", ["viewer_app.test.cjs", "benchmark_ui.test.cjs", "benchmark_second_pass.test.cjs"]
)
def test_viewer_app_behavior(suite):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [shutil.which("node"), "--test", f"tests/{suite}"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
