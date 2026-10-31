"""The composition experiment reports declined states and partial failures honestly."""

import json
from copy import deepcopy

import pytest
from PIL import Image

from tools import composition_demo as demo


@pytest.fixture
def task_file(tmp_path):
    task = {
        "source": {
            "canvas": {"width": 100, "height": 100},
            "elements": [
                {
                    "id": "panel",
                    "type": "shape",
                    "x": 10,
                    "y": 10,
                    "width": 80,
                    "height": 80,
                    "shape_type": "rectangle",
                    "fill_color": "rgba(255,255,255,1)",
                }
            ],
        },
        "target_canvas": {"width": 200, "height": 100},
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    return path, task


def test_unsupported_structure_keeps_real_baseline_without_claiming_proposals(task_file, tmp_path):
    path, _task = task_file
    before = path.read_bytes()
    output = tmp_path / "result"
    assert demo.main(["--task", str(path), "--out", str(output)]) == 0
    report = json.loads((output / "states.json").read_text())
    assert report["success"]
    assert report["proposal_status"] == "unsupported_or_no_feasible_state"
    assert report["states"] == []
    assert report["baseline_kind"] == "current_default"
    assert "CURRENT DEFAULT" in report["baseline"]["name"]
    assert report["source_immutable"] and report["graph_immutable"]
    graph = json.loads((output / "graph.json").read_text())
    assert graph["source_digest"] == report["source_digest"]
    assert graph["graph_digest"] == report["graph_digest"]
    assert Image.open(output / "baseline.png").size == (200, 100)
    assert path.read_bytes() == before


def test_render_failure_returns_nonzero_and_removes_stale_preview(task_file, tmp_path, monkeypatch):
    path, _task = task_file
    output = tmp_path / "result"
    output.mkdir()
    (output / "source.png").write_bytes(b"stale preview")

    def broken_renderer(*_args, **_kwargs):
        raise RuntimeError("test renderer unavailable")

    monkeypatch.setattr(demo, "render_image", broken_renderer)
    assert demo.main(["--task", str(path), "--out", str(output)]) == 1
    report = json.loads((output / "states.json").read_text())
    assert not report["success"]
    assert "renderer unavailable" in report["source"]["error"]
    assert len(report["errors"]) == 2
    assert not (output / "source.png").exists()
    assert not (output / "baseline.png").exists()
    assert (output / "contact_sheet.png").is_file()


def test_saved_baseline_is_identified_and_must_match_target(task_file, tmp_path):
    path, task = task_file
    baseline = deepcopy(task["source"])
    baseline["canvas"] = task["target_canvas"]
    baseline_path = tmp_path / "saved.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    before = baseline_path.read_bytes()
    output = tmp_path / "result"
    argv = ["--task", str(path), "--baseline", str(baseline_path), "--out", str(output)]
    assert demo.main(argv) == 0
    report = json.loads((output / "states.json").read_text())
    assert report["baseline_kind"] == "saved_json"
    assert report["baseline"]["name"] == "SAVED BASELINE"
    assert report["baseline_digest"] == demo._digest(baseline)
    assert baseline_path.read_bytes() == before
    baseline["canvas"] = {"width": 123, "height": 100}
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert demo.main(argv) == 1
    report = json.loads((output / "states.json").read_text())
    assert not report["success"]
    assert "baseline canvas must match" in report["errors"][0]["error"]
    assert not (output / "contact_sheet.png").exists()
