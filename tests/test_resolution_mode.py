"""Resolution-only requests must retain the source composition at one scale."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

import engine.resize as module
from engine.eval import check_contract
from engine.resources import using_resources
from engine.scene import analyze_scene

ROOT = Path(__file__).resolve().parents[1]
SCALED = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


def _uniform(source, target):
    scale = target["width"] / source["canvas"]["width"]
    result = deepcopy(source)
    result["canvas"] = dict(target)
    for element in result["elements"]:
        for key in SCALED & element.keys():
            element[key] *= scale
    return result


@pytest.mark.parametrize("level", ["level1_warmup", "level2_core"])
def test_actual_brand_resolution_change_preserves_all_source_geometry(level):
    source = json.loads(
        (ROOT / f"input/chameleon_sample_{level}.json").read_text(encoding="utf-8")
    )["source"]
    target = {key: value // 2 for key, value in source["canvas"].items()}
    before = deepcopy(source)
    result = module.resize_with_state(source, target)
    assert result["design"] == _uniform(source, target)
    assert check_contract(source, result["design"], target) == []
    assert source == before


def test_native_rich_text_resolution_change_preserves_style_and_original_identity():
    folder = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    target = {"width": 540, "height": 960}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        identity = module.resize_with_state(source, source["canvas"])
        result = module.resize_with_state(source, target)
        assert identity["design"] == source
        assert identity["state"]["name"] == "source"
        assert result["design"] == _uniform(source, target)
        assert check_contract(source, result["design"], target) == []
    for original, changed in zip(source["elements"], result["design"]["elements"], strict=True):
        assert {k: v for k, v in original.items() if k not in SCALED} == {
            k: v for k, v in changed.items() if k not in SCALED
        }


def test_resolution_result_cache_and_snapshot_share_fresh_source_faithful_designs():
    source = json.loads(
        (ROOT / "input/chameleon_sample_level2_core.json").read_text(encoding="utf-8")
    )["source"]
    target = {"width": 540, "height": 540}
    expected = _uniform(source, target)
    module._resize_cached.cache_clear()
    first = module.resize_with_state(source, target)
    first["design"]["elements"][0]["asset_id"] = "changed-by-caller.png"
    second = module.resize_with_state(source, target)
    assert second["design"] == expected
    snapshot = module.collect_candidate_snapshot(source, target)
    winner = next(
        row
        for row in snapshot["candidates"]
        if row["candidate_id"] == snapshot["baseline_candidate_id"]
    )
    assert winner["design"] == second["design"]
    assert winner["contract_valid"] and winner["admitted"]


def test_resolution_mode_keeps_model_binding_and_protection_trace():
    folder = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    target = {"width": 540, "height": 960}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        profile = analyze_scene(source)
        image = profile["images"][0]
        proposal = {
            "schema_version": 1,
            "source_digest": profile["source_digest"],
            "scene_digest": profile["scene_digest"],
            "images": [
                {
                    "element_id": image["element_id"],
                    "asset_sha256": image["asset_sha256"],
                    "proposed_role": "product",
                    "confidence": 0.99,
                    "protected_regions": [],
                }
            ],
        }
        result = module.resize_with_state(source, target, scene_proposal=proposal)
        assert result["design"] == _uniform(source, target)
        trace = result["state"]["scene_model"]
        assert trace["policy"] == "protect" and trace["images_reviewed"] == 1
        assert trace["protected_element_ids"] == [image["element_id"]]
        assert trace["candidates_vetoed"] == 0 and trace["proposal_digest"]
        with pytest.raises(ValueError):
            module.resize_with_state(
                source, target, scene_proposal={**proposal, "scene_digest": "0" * 64}
            )


def test_nonuniform_aspect_and_cropped_source_images_keep_the_normal_search(monkeypatch):
    source = json.loads(
        (ROOT / "input/chameleon_sample_level2_core.json").read_text(encoding="utf-8")
    )["source"]
    calls = []
    monkeypatch.setattr(module, "generate_candidates", lambda *_: calls.append(True) or [])
    for name in (
        "generate_background_states",
        "generate_composition_states",
        "generate_grouped_states",
        "generate_focal_states",
        "generate_source_guides",
    ):
        monkeypatch.setattr(module, name, lambda *_: ())
    module._candidate_pool(source, {"width": 540, "height": 541})
    assert calls == [True]
    cropped = deepcopy(source)
    next(e for e in cropped["elements"] if e["type"] == "image" and e["id"] != "bg")["x"] = -1
    module._candidate_pool(cropped, {"width": 540, "height": 540})
    assert calls == [True, True]


def test_resolution_mode_scales_stroke_and_radius_without_scaling_dimensionless_style():
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "rounded",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 80,
                "corner_radius": 12,
                "stroke_width": 4,
                "stroke_color": "#ffffff",
                "fill_color": "#223344",
                "opacity": 0.7,
            }
        ],
    }
    assert module.resize(source, {"width": 50, "height": 50}) == _uniform(
        source, {"width": 50, "height": 50}
    )


def test_uniform_candidate_cannot_waive_a_hard_font_floor():
    source = json.loads(
        (ROOT / "input/chameleon_sample_level2_core.json").read_text(encoding="utf-8")
    )["source"]
    assert module._resolution_state(source, {"width": 1, "height": 1}) is None
