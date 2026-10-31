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


def _odd_source():
    return {
        "canvas": {"width": 101, "height": 81},
        "elements": [
            {
                "id": "plate",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 4,
                "y": 4,
                "width": 93,
                "height": 73,
                "fill_color": "#223344",
                "corner_radius": 5,
                "stroke_width": 2,
                "stroke_color": "#ffffff",
                "opacity": 0.8,
                "z_index": 0,
            },
            {
                "id": "logo",
                "type": "image",
                "asset_id": "solstice_logo_lockup.png",
                "fit": "contain",
                "x": 12,
                "y": 12,
                "width": 32,
                "height": 10,
                "z_index": 1,
            },
            {
                "id": "copy",
                "type": "text",
                "content": "Hello",
                "font_family": "Solstice Sans",
                "font_size": 10,
                "font_weight": "bold",
                "color": "#ffffff",
                "text_align": "left",
                "line_height": 1.2,
                "x": 12,
                "y": 35,
                "width": 70,
                "height": 20,
                "group_id": "source-copy",
                "z_index": 2,
            },
        ],
    }


def _assert_padded_similarity(source, design, target):
    """All components retain one similarity, with at most a quarter-pixel inset."""
    scale = min(target[key] / source["canvas"][key] for key in ("width", "height"))
    offsets = {
        "x": (target["width"] - source["canvas"]["width"] * scale) / 2,
        "y": (target["height"] - source["canvas"]["height"] * scale) / 2,
    }
    assert design["canvas"] == target
    assert all(-1e-9 <= offset <= 0.25 + 1e-9 for offset in offsets.values())
    assert [e["id"] for e in design["elements"]] == [e["id"] for e in source["elements"]]
    for original, changed in zip(source["elements"], design["elements"], strict=True):
        for key in SCALED & original.keys():
            assert changed[key] == pytest.approx(original[key] * scale + offsets.get(key, 0))
        assert {k: v for k, v in changed.items() if k not in SCALED} == {
            k: v for k, v in original.items() if k not in SCALED
        }
    assert check_contract(source, design, target) == []


@pytest.mark.parametrize(
    ("canvas", "target"),
    [
        ({"width": 101, "height": 81}, {"width": 51, "height": 41}),
        ({"width": 1001, "height": 800}, {"width": 501, "height": 400}),
    ],
)
def test_rounded_resolution_preserves_all_components_at_one_scale(canvas, target):
    source = _odd_source()
    source["canvas"] = canvas
    before = deepcopy(source)
    state = module._resolution_state(source, target)
    assert state is not None
    assert state.name == "resolution-scale" and state.topology == "source-similarity"
    assert len(state.exceptions) == 1
    assert state.exceptions[0].startswith("resolution-rounding-padding:")
    _assert_padded_similarity(source, state.design, target)
    assert source == before


@pytest.mark.parametrize(
    ("name", "target"),
    [
        ("opensuse_trifold_front", {"width": 573, "height": 408}),
        ("opensuse_trifold_front", {"width": 286, "height": 204}),
        ("opensuse_cheatsheet", {"width": 421, "height": 144}),
    ],
)
def test_actual_reviewed_rounded_resolution_avoids_repacking_and_retains_contract(
    name, target, monkeypatch
):
    folder = ROOT / "fixtures/open_posters/prepared" / name
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)

    def unexpected_search(*_args):
        pytest.fail("A bounded rounded-resolution request must not enter composition search")

    monkeypatch.setattr(module, "generate_candidates", unexpected_search)
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        assert module._resolution_state(source, target) is not None
        result = module.resize_with_state(source, target)
        assert result["state"]["name"] == "resolution-scale"
        assert result["state"]["sizing_exceptions"][0].startswith("resolution-rounding-padding:")
        _assert_padded_similarity(source, result["design"], target)
    assert source == before


@pytest.mark.parametrize("target", [{"width": 51, "height": 42}, {"width": 121, "height": 96}])
def test_nearby_actual_aspect_changes_do_not_gain_a_resolution_exception(target):
    assert module._resolution_state(_odd_source(), target) is None


def test_rounded_resolution_still_declines_a_cropped_source_image():
    source = _odd_source()
    source["elements"][1]["x"] = -1
    assert module._resolution_state(source, {"width": 51, "height": 41}) is None
