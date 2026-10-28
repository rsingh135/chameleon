"""The batch CLI must report real failures and keep processing other tasks."""

import builtins
import json
from copy import deepcopy

import pytest

from engine import main as cli


@pytest.fixture
def task():
    return {
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
                    "fill_color": "#ffffff",
                }
            ],
        },
        "target_canvas": {"width": 100, "height": 100},
    }


@pytest.fixture
def batch(tmp_path, monkeypatch):
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    monkeypatch.setattr(cli, "INPUT_DIR", inputs)
    monkeypatch.setattr(cli, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(cli, "resize", lambda source, target: deepcopy(source))

    def render(design, assets_dir, png_path):
        png_path.write_bytes(b"rendered preview")

    monkeypatch.setattr(cli, "render_to_png", render)
    return inputs, outputs


def test_missing_renderer_is_an_explicit_error(tmp_path, monkeypatch, task):
    original_import = builtins.__import__

    def without_renderer(name, *args, **kwargs):
        if name == "renderer.converter.convert":
            raise ImportError("cr_renderer is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_renderer)
    preview = tmp_path / "preview.png"
    with pytest.raises(RuntimeError, match="requirements.txt"):
        cli.render_to_png(task["source"], tmp_path, preview)
    assert not preview.exists()


def test_malformed_task_does_not_stop_later_tasks(batch, task, capsys):
    inputs, outputs = batch
    (inputs / "a_broken.json").write_text("{invalid", encoding="utf-8")
    (inputs / "b_valid.json").write_text(json.dumps(task), encoding="utf-8")

    assert cli.main() == 1
    assert (outputs / "b_valid_output.json").exists()
    assert (outputs / "b_valid_output.png").exists()
    captured = capsys.readouterr()
    assert "a_broken.json" in captured.err
    assert "ERROR" in captured.err
    assert "b_valid.json" in captured.out


def test_contract_failure_is_not_reported_as_success(batch, task, monkeypatch, capsys):
    inputs, outputs = batch
    (inputs / "invalid.json").write_text(json.dumps(task), encoding="utf-8")

    def drop_element(source, target):
        return {"canvas": target, "elements": []}

    monkeypatch.setattr(cli, "resize", drop_element)
    assert cli.main() == 1
    assert not (outputs / "invalid_output.png").exists()
    assert "id_missing" in capsys.readouterr().err


def test_render_failure_clears_stale_preview_and_continues(batch, task, monkeypatch, capsys):
    inputs, outputs = batch
    outputs.mkdir()
    (outputs / "a_first_output.png").write_bytes(b"stale image")
    for name in ("a_first", "b_second"):
        (inputs / f"{name}.json").write_text(json.dumps(task), encoding="utf-8")

    def render(design, assets_dir, png_path):
        if png_path.stem.startswith("a_first"):
            raise RuntimeError("renderer failed")
        png_path.write_bytes(b"real preview")

    monkeypatch.setattr(cli, "render_to_png", render)
    assert cli.main() == 1
    assert not (outputs / "a_first_output.png").exists()
    assert (outputs / "b_second_output.png").exists()
    assert "renderer failed" in capsys.readouterr().err


def test_success_writes_both_artifacts(batch, task, capsys):
    inputs, outputs = batch
    (inputs / "valid.json").write_text(json.dumps(task), encoding="utf-8")

    assert cli.main() == 0
    result = json.loads((outputs / "valid_output.json").read_text(encoding="utf-8"))
    assert result == task["source"]
    assert (outputs / "valid_output.png").exists()
    assert "OK" in capsys.readouterr().out


def test_no_tasks_returns_nonzero(batch, capsys):
    assert cli.main() == 1
    assert "No task files" in capsys.readouterr().err
