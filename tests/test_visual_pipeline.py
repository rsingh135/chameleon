"""Production dispatch shares the audit phases and never fakes model execution."""

from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from engine import visual_config, visual_pipeline
from engine.resources import DEFAULT_FONTS_DIR


@pytest.fixture
def source():
    return {"canvas": {"width": 100, "height": 100}, "elements": []}


def test_unconfigured_dispatch_has_no_model_dependency_and_keeps_strict_profile(
    source, monkeypatch
):
    import engine.resize as resize

    calls = []
    monkeypatch.setattr(visual_config, "resolve_config", lambda _config: (None, {"enabled": False}))

    def baseline(value, target, **kwargs):
        calls.append(kwargs)
        return {"design": deepcopy(value), "state": {"name": "source"}}

    monkeypatch.setattr(resize, "resize_with_state", baseline)
    monkeypatch.setattr(
        visual_pipeline,
        "evaluate_visual_resize",
        lambda *_a, **_kw: pytest.fail("Model path invoked"),
    )
    result = visual_pipeline.resize_configured(source, source["canvas"])
    assert calls[0]["strict_bounds"] is True
    assert "visual_model" not in result["state"]


def test_invalid_requested_model_falls_back_with_visible_failure(source, monkeypatch):
    import engine.resize as resize

    def invalid(_config):
        raise ValueError("Missing pinned model configuration")

    monkeypatch.setattr(visual_config, "resolve_config", invalid)
    monkeypatch.setattr(
        visual_pipeline,
        "visual_execution_identity",
        lambda _config: {"enabled": True, "valid": False},
    )
    monkeypatch.setattr(
        resize,
        "resize_with_state",
        lambda value, target, **kw: {"design": deepcopy(value), "state": {}},
    )
    result = visual_pipeline.resize_configured(source, source["canvas"])
    trace = result["state"]["visual_model"]
    assert trace["requested"] and not trace["model_executed"]
    assert not trace["new_candidates_generated"] and trace["quality_improved"] is None
    assert "Missing pinned" in trace["fallback_reason"]


def test_invalid_configuration_type_falls_back_with_visible_failure(source, monkeypatch):
    import engine.resize as resize

    def invalid(_config):
        raise TypeError("Invalid provider configuration type")

    monkeypatch.setattr(visual_config, "resolve_config", invalid)
    monkeypatch.setattr(
        visual_pipeline,
        "visual_execution_identity",
        lambda _config: {"enabled": True, "valid": False},
    )
    monkeypatch.setattr(
        resize,
        "resize_with_state",
        lambda value, target, **kw: {"design": deepcopy(value), "state": {}},
    )
    result = visual_pipeline.resize_configured(source, source["canvas"])
    trace = result["state"]["visual_model"]
    assert trace["requested"] and not trace["model_executed"]
    assert "Invalid provider configuration type" in trace["fallback_reason"]


def test_configured_requests_share_phase_engine_and_do_not_cache_model_results(
    source, tmp_path, monkeypatch
):
    operator = {
        "enabled": True,
        "artifact_root": str(tmp_path),
        "provider_command": ["trusted"],
        "timeout_seconds": 240,
        "critic": True,
    }
    monkeypatch.setattr(
        visual_config, "resolve_config", lambda _config: (operator, {"enabled": True})
    )
    directories = []

    def phases(value, assets, fonts, out, **kwargs):
        directories.append(Path(out))
        assert kwargs["strict_stage_one"] is True and kwargs["critic_enabled"] is True
        folder = Path(out) / "100x100"
        folder.mkdir(parents=True)
        visual_pipeline._io._write(
            folder / "final-plan.json", {"design": value, "state": {"name": "source"}}
        )
        row = {
            "status": "contract_valid",
            "model_executed": True,
            "proposal_valid": True,
            "new_candidates_generated": False,
            "new_candidate_ids": [],
            "admitted_new_candidate_ids": [],
            "model_candidate_selected": False,
            "layout_changed": False,
            "critic_changed_selection": False,
            "quality_improved": None,
            "fallback_reason": "Model retained baseline",
            "planner": {"model_executed": True},
            "critic": {"model_executed": False},
            "selection_trace": {},
            "same_png": True,
        }
        return {"targets": [row], "implementation_unchanged": True}

    monkeypatch.setattr(visual_pipeline, "evaluate_visual_resize", phases)
    first = visual_pipeline.resize_configured(source, source["canvas"])
    second = visual_pipeline.resize_configured(source, source["canvas"])
    assert len(set(directories)) == 2
    assert first["state"]["visual_model"]["model_executed"]
    assert second["state"]["visual_model"]["quality_improved"] is None
    assert not second["state"]["visual_model"]["layout_changed"]


def test_resource_hashes_reject_relative_escape_and_bind_actual_bytes(source, tmp_path):
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    value = deepcopy(source)
    value["elements"] = [{"type": "image", "asset_id": "../private.bin"}]
    (tmp_path / "private.bin").write_bytes(b"source bytes")
    with pytest.raises(ValueError, match="escapes"):
        visual_pipeline._io.resource_hashes(value, assets, fonts)


@pytest.fixture
def missing_artwork_document(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "artwork-42",
                "type": "image",
                "asset_id": "missing-brand-art.png",
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 80,
                "z_index": 1,
            }
        ],
    }
    return source, assets


def test_visual_pipeline_renders_missing_asset_placeholder_without_replacing_artwork(
    missing_artwork_document, tmp_path
):
    source, assets = missing_artwork_document
    original = deepcopy(source)
    out = tmp_path / "evidence"
    report = visual_pipeline.evaluate_visual_resize(
        source,
        assets,
        DEFAULT_FONTS_DIR,
        out,
        targets=[source["canvas"]],
        command=["unused-provider"],
        provider=lambda *_args: pytest.fail("An unchanged single-image source needs no model"),
        implementation_hashes=lambda: {"test": "unchanged"},
    )
    assert report["targets"][0]["status"] == "contract_valid"
    assert report["resource_hashes"]["assets"] == {"missing-brand-art.png": {"status": "missing"}}
    assert source == original
    assert list(assets.iterdir()) == []
    with Image.open(out / "source.png") as preview:
        assert preview.size == (100, 100)
        assert preview.convert("RGB").getpixel((10, 10)) == (150, 40, 40)
    assert (out / "100x100" / "final.png").is_file()


@pytest.mark.parametrize("initially_present", [False, True])
def test_visual_pipeline_rejects_asset_presence_changes(
    missing_artwork_document, tmp_path, initially_present
):
    source, assets = missing_artwork_document
    asset = assets / source["elements"][0]["asset_id"]
    if initially_present:
        Image.new("RGB", (80, 80), "blue").save(asset)

    def snapshot(*_args, **_kwargs):
        if initially_present:
            asset.unlink()
        else:
            Image.new("RGB", (80, 80), "blue").save(asset)
        return {}

    with pytest.raises(RuntimeError, match="resource bytes changed"):
        visual_pipeline.evaluate_visual_resize(
            source,
            assets,
            DEFAULT_FONTS_DIR,
            tmp_path / "evidence",
            targets=[source["canvas"]],
            command=["unused-provider"],
            api={"snapshot": snapshot},
            implementation_hashes=lambda: {"test": "unchanged"},
        )


def test_complex_source_text_context_is_bounded_without_mutating_original(source):
    value = deepcopy(source)
    value["elements"] = [
        {
            "id": f"copy-{index}",
            "type": "text",
            "content": "Design copy: " + "\u00e9" * 500,
            "x": 0,
            "y": index,
            "width": 50,
            "height": 1,
            "font_size": 12,
        }
        for index in range(256)
    ]
    original = deepcopy(value)
    summary = visual_pipeline._source_summary(value)
    assert len(visual_pipeline._io._canonical(summary)) <= 10_000
    assert summary["element_count"] == 256
    assert summary["omitted_element_count"] == 256 - len(summary["elements"])
    assert summary["omitted_element_count"] > 0
    assert all(len(row["content"]) <= 180 for row in summary["elements"])
    assert value == original
