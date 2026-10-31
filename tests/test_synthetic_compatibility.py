"""CPU-only recipe, weak-metric and kernel-build prerequisite checks."""

import subprocess
import sys
from pathlib import Path

import pytest

from experiments.vision_critic.synthetic_compatibility import agreement, prepare
from tools import vision_gpu_probe


def test_weak_label_metrics_count_invalid_outputs_without_claiming_real_quality():
    result = agreement(
        [
            {"valid": True, "matches_weak_target": True},
            {"valid": False, "matches_weak_target": False},
        ]
    )
    assert result["count"] == 2
    assert result["schema_valid_count"] == 1
    assert result["weak_label_choice_agreement"] == 0.5
    assert "no real design-quality" in result["scope"]
    assert agreement([])["weak_label_choice_agreement"] is None


@pytest.mark.parametrize(
    "revision,steps,seconds",
    [
        ("main", 24, 240),
        ("a" * 40, 1000, 240),
        ("a" * 40, 24, 301),
        ("a" * 40, 12, 0),
    ],
)
def test_synthetic_recipe_rejects_unpinned_or_unbounded_work_before_reading_files(
    revision, steps, seconds
):
    with pytest.raises(ValueError):
        prepare(Path("absent-collection"), Path("absent-labels"), revision, steps, seconds)


def test_synthetic_cli_requires_an_explicit_opt_in_before_reading_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.vision_critic.synthetic_compatibility",
            "--collection",
            "absent",
            "--labels",
            "absent",
            "--out",
            str(tmp_path),
            "--base-revision",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "Explicit --allow-synthetic" in result.stderr


def test_kernel_build_probe_reports_missing_python_header_actionably(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_gpu_probe.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(vision_gpu_probe.shutil, "which", lambda name: "/usr/bin/cc")
    report, status = vision_gpu_probe.build_prerequisites()
    assert status == 5 and not report["build_prerequisites_passed"]
    assert str(tmp_path / "Python.h") in report["missing"]
    assert "-dev" in report["error"] and "build-essential" in report["error"]
    assert report["model_training_validated"] is False


def test_kernel_build_probe_requires_both_header_and_compiler(tmp_path, monkeypatch):
    (tmp_path / "Python.h").write_text("/* infrastructure fixture */")
    monkeypatch.setattr(vision_gpu_probe.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(vision_gpu_probe.shutil, "which", lambda name: None)
    assert vision_gpu_probe.build_prerequisites()[1] == 5
    monkeypatch.setattr(vision_gpu_probe.shutil, "which", lambda name: "/usr/bin/cc")
    report, status = vision_gpu_probe.build_prerequisites()
    assert status == 0 and report["build_prerequisites_passed"]
    assert report["model_training_validated"] is False
