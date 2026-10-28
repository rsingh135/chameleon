"""Harness failures must reach its reports, previews and process exit status."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import harness


@pytest.fixture
def task():
    return {
        "task_id": "case",
        "family": "synthetic",
        "holdout": False,
        "source": {"canvas": {"width": 100, "height": 100}, "elements": []},
        "target_canvas": {"width": 100, "height": 100},
    }


@pytest.fixture
def fast_engine(monkeypatch):
    monkeypatch.setattr(harness, "build_context", lambda *args: object())
    monkeypatch.setattr(harness, "resize", lambda source, target: deepcopy(source))
    monkeypatch.setattr(
        harness,
        "score",
        lambda *args: {"violations": [], "total": 0.9, "metrics": {"collision_free": 1.0}},
    )
    monkeypatch.setattr(harness, "render_plate", lambda *args, **kwargs: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(harness, "render_image", lambda *args, **kwargs: Image.new("RGB", (1, 1)))


@pytest.fixture
def invocation(tmp_path, monkeypatch, task, fast_engine):
    inputs, out = tmp_path / "input", tmp_path / "out"
    inputs.mkdir()
    (inputs / "case.json").write_text(json.dumps(task), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["harness", "--input", str(inputs), "--out", str(out), "--no-render"]
    )
    return inputs, out


def fail_render(*args, **kwargs):
    raise RuntimeError("render backend failed")


def test_render_error_is_a_failed_row(task, fast_engine, monkeypatch):
    monkeypatch.setattr(harness, "render_image", fail_render)
    row, image = harness.run_one(task, "own", task["target_canvas"], render=True)
    assert image is None
    assert "render_error" in row["violation_codes"]
    assert row["total"] == 0
    report = harness.summarize([row])
    assert "valid: 0" in report
    assert "render_error" in report


def test_context_failure_is_recorded_as_crash(task, fast_engine, monkeypatch):
    monkeypatch.setattr(harness, "build_context", fail_render)
    row, image = harness.run_one(task, "own", task["target_canvas"], render=False)
    assert image is None
    assert row["crashed"]
    assert row["violation_codes"] == ["crash"]


def test_render_failure_contact_cell_is_marked_bad(task, fast_engine, monkeypatch):
    monkeypatch.setattr(harness, "render_plate", fail_render)
    row, image = harness.run_one(task, "own", task["target_canvas"], render=True)
    sheet = harness.contact_sheet(task, [row], [image], render=True)
    assert sheet.getpixel((harness.CELL_W + 1, 1)) == (200, 60, 60)


def test_empty_summary_has_zero_timings():
    report = harness.summarize([])
    assert "rows: 0" in report
    assert "mean 0.0ms" in report


def test_empty_input_returns_clear_failure(invocation, capsys):
    inputs, out = invocation
    (inputs / "case.json").unlink()
    assert harness.main() == 1
    assert "No tasks" in capsys.readouterr().err


def test_render_failure_returns_nonzero_and_bad_label(invocation, monkeypatch, capsys):
    inputs, out = invocation
    monkeypatch.setattr(
        "sys.argv", ["harness", "--input", str(inputs), "--out", str(out), "--no-sheets"]
    )
    monkeypatch.setattr(harness, "render_plate", fail_render)
    assert harness.main() == 1
    assert "BAD" in capsys.readouterr().out
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "render_error" in report[0]["violation_codes"]


def test_source_render_failure_preserves_report_and_sheet(invocation, monkeypatch):
    inputs, out = invocation
    monkeypatch.setattr("sys.argv", ["harness", "--input", str(inputs), "--out", str(out)])
    original = harness.contact_sheet

    def source_fails(task, rows, images, render):
        if render:
            raise RuntimeError("source preview failed")
        return original(task, rows, images, render=False)

    monkeypatch.setattr(harness, "contact_sheet", source_fails)
    assert harness.main() == 1
    assert (out / "sheet_case.png").exists()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "sheet_error" in report[0]["violation_codes"]


def test_valid_json_only_run_returns_zero(invocation, capsys):
    _, out = invocation
    assert harness.main() == 0
    assert "OK" in capsys.readouterr().out
    assert "valid: 1" in (out / "report.md").read_text(encoding="utf-8")


def test_contract_violation_returns_nonzero(invocation, monkeypatch, capsys):
    monkeypatch.setattr(
        harness,
        "score",
        lambda *args: {
            "violations": [SimpleNamespace(code="id_missing")],
            "total": 0.0,
            "metrics": {},
        },
    )
    assert harness.main() == 1
    assert "BAD" in capsys.readouterr().out


def test_crash_returns_nonzero(invocation, monkeypatch, capsys):
    monkeypatch.setattr(harness, "resize", fail_render)
    assert harness.main() == 1
    captured = capsys.readouterr()
    assert "BAD" in captured.out
    assert "crash" in captured.out
