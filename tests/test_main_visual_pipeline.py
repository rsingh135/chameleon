"""Headless dispatch, native pixel agreement and task-scoped visual receipts."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from PIL import Image

from engine import main as cli
from engine.eval import check_contract
from engine.eval.render import render_image
from engine.resize import resize_with_state
from engine.resources import DEFAULT_FONTS_DIR, using_resources
from engine.stage_one import strictly_in_bounds

ROOT = Path(__file__).resolve().parents[1]


def core():
    return json.loads((ROOT / "input/chameleon_sample_level2_core.json").read_text())["source"]


def test_real_unconfigured_batch_facade_enforces_strict_stage_one(monkeypatch):
    monkeypatch.delenv("CHAMELEON_VISUAL_PLANNER_CONFIG", raising=False)
    source, target = core(), {"width": 540, "height": 540}
    design, provenance = cli._resize_task(source, target)
    assert strictly_in_bounds(design)
    assert check_contract(source, design, target) == []
    assert "visual_model" not in provenance
    assert cli._TASK_PROVENANCE.get() is None


def test_configured_dispatch_writes_separate_bound_receipt_and_clears_stale_receipts(
    tmp_path, monkeypatch
):
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    source = {
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
    }
    task = {"source": source, "target_canvas": source["canvas"]}
    (inputs / "poster.json").write_text(json.dumps(task), encoding="utf-8")
    monkeypatch.setattr(cli, "INPUT_DIR", inputs)
    monkeypatch.setattr(cli, "OUTPUT_DIR", outputs)
    calls = []
    receipt = {
        "requested": True,
        "model_executed": False,
        "layout_changed": False,
        "fallback_reason": "Test configuration failure, not a model execution",
    }

    def configured(value, target, **kwargs):
        calls.append(kwargs)
        return {"design": deepcopy(value), "state": {"visual_model": deepcopy(receipt)}}

    monkeypatch.setattr(cli, "resize_configured", configured)
    assert cli.main() == 0
    assert calls == [{"assets_dir": cli.ASSETS_DIR, "strict_stage_one": True}]
    saved = json.loads((outputs / "poster_output.json").read_text())
    assert saved == source and set(saved) == {"canvas", "elements"}
    trace = json.loads((outputs / "poster_pipeline.json").read_text())
    assert trace["visual_model"] == receipt
    for kind in ("json", "png"):
        path = outputs / f"poster_output.{kind}"
        assert trace[f"output_{kind}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert cli._TASK_PROVENANCE.get() is None

    # Preserve the established two-argument resize replacement seam; no prior
    # configured receipt may leak into this independent plain-design task.
    monkeypatch.setattr(cli, "resize", lambda value, target: deepcopy(value))
    assert cli.main() == 0
    assert not (outputs / "poster_pipeline.json").exists()


def test_headless_final_guard_rejects_source_bleed_that_legacy_contract_allows(
    tmp_path, monkeypatch, capsys
):
    source, target = core(), {"width": 540, "height": 540}
    legacy = resize_with_state(source, target)["design"]
    assert check_contract(source, legacy, target) == []
    assert not strictly_in_bounds(legacy)
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    (inputs / "legacy.json").write_text(
        json.dumps({"source": source, "target_canvas": target}), encoding="utf-8"
    )
    monkeypatch.setattr(cli, "INPUT_DIR", inputs)
    monkeypatch.setattr(cli, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(cli, "resize", lambda *_a: legacy)
    assert cli.main() == 1
    assert "strict stage-one bounds" in capsys.readouterr().err
    assert not (outputs / "legacy_output.png").exists()
    assert not (outputs / "legacy_pipeline.json").exists()


def test_headless_export_matches_model_native_pixels_for_positioned_image_and_type(tmp_path):
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    picture = Image.new("RGB", (240, 80), "#ff0000")
    picture.paste("#00ff00", (80, 0, 160, 80))
    picture.paste("#0000ff", (160, 0, 240, 80))
    picture.save(assets / "stripes.png")
    (fonts / "face.ttf").write_bytes((DEFAULT_FONTS_DIR / "SolsticeSans-Regular.ttf").read_bytes())
    (fonts / "font-manifest.json").write_text(
        json.dumps(
            [
                {
                    "font_family": "NativeTestFace",
                    "file": "face.ttf",
                    "weight": "regular",
                    "style": "normal",
                }
            ]
        ),
        encoding="utf-8",
    )
    design = {
        "canvas": {"width": 160, "height": 140},
        "elements": [
            {
                "id": "photo",
                "type": "image",
                "asset_id": "stripes.png",
                "x": 0,
                "y": 0,
                "width": 80,
                "height": 80,
                "fit": "cover",
                "image_position": [1, 0.5],
            },
            {
                "id": "copy",
                "type": "text",
                "content": "Native type",
                "x": 5,
                "y": 95,
                "width": 150,
                "height": 40,
                "font_family": "NativeTestFace",
                "font_size": 19.25,
                "color": "#111111",
                "opacity": 0.65,
                "style_runs": [{"start": 0, "end": 6, "color": "#ff0000"}],
            },
        ],
    }
    original = deepcopy(design)
    with using_resources(assets, fonts):
        expected = render_image(design, max_size=160)
        cli.render_to_png(design, assets, tmp_path / "output.png")
    cli.render_to_png(design, assets, tmp_path / "standalone.png")
    assert (tmp_path / "standalone.png").read_bytes() == (tmp_path / "output.png").read_bytes()
    with Image.open(tmp_path / "output.png") as actual:
        assert actual.size == (160, 140)
        assert actual.convert("RGBA").tobytes() == expected.tobytes()
        assert actual.getpixel((20, 20))[:3] == (0, 0, 255)
    assert design == original
