"""Dependency-free state tests for the auxiliary file inspector and redirects."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Inspector checks require Node >=18")
def test_inspector_ui_regressions():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [shutil.which("node"), "--test", "tests/inspector_ui.test.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
