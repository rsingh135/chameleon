"""Generic purpose-aware placement; manual acceptance posters stay out of tests."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.design_semantics import audit_semantic_preservation, describe_design
from engine.eval import check_contract
from engine.layout import semantic_anchors as module
from engine.layout.semantic_anchors import generate_semantic_anchor_states
from engine.stage_one import strictly_in_bounds


@pytest.fixture
def source():
    return {
        "canvas": {"width": 100, "height": 300},
        "elements": [
            {
                "id": "background",
                "type": "shape",
                "shape_type": "rectangle",
                "fill_color": "#224466",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 300,
            },
            {
                "id": "identity",
                "type": "image",
                "asset_id": "identity.png",
                "fit": "fill",
                "x": 72,
                "y": 8,
                "width": 20,
                "height": 10,
            },
            {
                "id": "title",
                "type": "text",
                "content": "Join",
                "font_family": "Solstice Sans",
                "font_size": 16,
                "color": "#ffffff",
                "x": 20,
                "y": 60,
                "width": 60,
                "height": 20,
                "group_id": "message-art",
            },
            {
                "id": "hero",
                "type": "image",
                "asset_id": "hero.png",
                "fit": "fill",
                "x": 20,
                "y": 90,
                "width": 60,
                "height": 100,
                "group_id": "message-art",
            },
            {
                "id": "destination",
                "type": "text",
                "content": "www.x.org",
                "font_family": "Solstice Sans",
                "font_size": 10,
                "color": "#ffffff",
                "x": 10,
                "y": 275,
                "width": 80,
                "height": 15,
            },
        ],
    }


def test_corner_footer_and_explicit_group_have_actual_valid_transformed_geometry(source):
    original = deepcopy(source)
    target = {"width": 600, "height": 150}
    semantics = describe_design(source)
    states = generate_semantic_anchor_states(source, target, semantics)
    assert states
    state = states[0]
    assert state.cluster_axes and state.topology == "source-semantic-anchors"
    assert strictly_in_bounds(state.design)
    assert check_contract(source, state.design, target) == []
    result = {e["id"]: e for e in state.design["elements"]}
    scale = result["hero"]["width"] / 60
    assert result["identity"]["width"] / 20 == pytest.approx(scale)
    assert result["destination"]["font_size"] / 10 == pytest.approx(scale)
    assert result["title"]["font_size"] / 16 == pytest.approx(scale)
    assert result["hero"]["y"] - result["title"]["y"] == pytest.approx(30 * scale)
    assert result["identity"]["y"] == pytest.approx(8 / 100 * 150)
    assert 600 - result["identity"]["x"] - result["identity"]["width"] == pytest.approx(
        8 / 100 * 150
    )
    assert 150 - result["destination"]["y"] - result["destination"]["height"] == pytest.approx(
        10 / 100 * 150
    )
    audit = audit_semantic_preservation(source, state.design, semantics)
    assert audit["metrics"]["corner_anchor_retention"] == pytest.approx(1)
    assert audit["metrics"]["footer_anchor_retention"] == pytest.approx(1)
    assert audit["metrics"]["explicit_group_retention"] == pytest.approx(1)
    assert not audit["violations"]
    assert source == original


def test_default_path_uses_describer_without_a_model(source):
    states = generate_semantic_anchor_states(source, {"width": 450, "height": 150})
    assert states
    assert all("semantic-source:" in " ".join(state.exceptions) for state in states)


def test_primary_uses_footer_free_band_and_can_start_above_a_nonoverlapping_corner(source):
    target = {"width": 600, "height": 150}
    states = generate_semantic_anchor_states(source, target, describe_design(source))
    result = {e["id"]: e for e in states[0].design["elements"]}
    assert result["title"]["y"] < result["identity"]["y"]
    assert result["hero"]["height"] > 80
    assert result["hero"]["y"] + result["hero"]["height"] < result["destination"]["y"]


def test_unrelated_copy_and_art_move_together_in_residual_primary_composition(source):
    value = deepcopy(source)
    value["elements"][2].pop("group_id")
    value["elements"][3].pop("group_id")
    target = {"width": 600, "height": 150}
    states = generate_semantic_anchor_states(value, target, describe_design(value))
    assert states
    result = {e["id"]: e for e in states[0].design["elements"]}
    scale = result["hero"]["width"] / 60
    assert result["hero"]["y"] - result["title"]["y"] == pytest.approx(30 * scale)


def test_no_shape_change_and_unanchored_scene_offer_no_anchor_alternative(source):
    assert generate_semantic_anchor_states(source, source["canvas"]) == []
    value = {
        "canvas": source["canvas"],
        "elements": deepcopy(source["elements"][:1] + source["elements"][2:4]),
    }
    assert (
        generate_semantic_anchor_states(
            value, {"width": 600, "height": 150}, describe_design(value)
        )
        == []
    )


def test_stale_and_mutated_descriptors_are_rejected(source):
    descriptor = describe_design(source)
    changed_source = deepcopy(source)
    changed_source["elements"][2]["content"] = "Different copy"
    with pytest.raises(ValueError, match="unchanged source"):
        generate_semantic_anchor_states(changed_source, {"width": 600, "height": 150}, descriptor)
    descriptor["elements"][1]["anchors"][0]["insets_short_axis"]["right"] = 0
    with pytest.raises(ValueError, match="descriptor digest"):
        generate_semantic_anchor_states(source, {"width": 600, "height": 150}, descriptor)


def test_footer_plate_keeps_leaf_anchor_and_atomic_relative_geometry(source):
    value = deepcopy(source)
    value["elements"][-1]["group_id"] = "footer-control"
    value["elements"].append(
        {
            "id": "footer-plate",
            "type": "shape",
            "shape_type": "rectangle",
            "fill_color": "#222222",
            "x": 7,
            "y": 272,
            "width": 86,
            "height": 21,
            "group_id": "footer-control",
        }
    )
    descriptor = describe_design(value)
    states = generate_semantic_anchor_states(value, {"width": 600, "height": 150}, descriptor)
    assert states
    result = {e["id"]: e for e in states[0].design["elements"]}
    factor = result["destination"]["font_size"] / 10
    assert result["destination"]["x"] - result["footer-plate"]["x"] == pytest.approx(3 * factor)
    assert result["destination"]["y"] - result["footer-plate"]["y"] == pytest.approx(3 * factor)
    audit = audit_semantic_preservation(value, states[0].design, descriptor)
    assert audit["metrics"]["footer_anchor_retention"] == pytest.approx(1)
    assert audit["metrics"]["explicit_group_retention"] == pytest.approx(1)


def test_core_near_aspect_change_declines_before_source_native_analysis(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "input/chameleon_sample_level2_core.json"
    source = json.loads(path.read_text(encoding="utf-8"))["source"]
    monkeypatch.setattr(
        module, "_descriptors", lambda *_a: pytest.fail("Native analysis must not run")
    )
    assert generate_semantic_anchor_states(source, {"width": 540, "height": 541}) == []


def test_ordinary_bottom_action_is_not_a_strong_automatic_footer(source, monkeypatch):
    value = deepcopy(source)
    value["elements"][-1].update(content="Shop now", role="action")
    with monkeypatch.context() as guard:
        guard.setattr(
            module, "_descriptors", lambda *_a: pytest.fail("Native analysis must not run")
        )
        assert generate_semantic_anchor_states(value, {"width": 600, "height": 150}) == []
    assert (
        generate_semantic_anchor_states(
            value, {"width": 600, "height": 150}, describe_design(value)
        )
        == []
    )


def test_source_group_containing_background_is_not_split_to_fill_canvas(source):
    value = deepcopy(source)
    value["elements"][0]["group_id"] = "message-art"
    assert (
        generate_semantic_anchor_states(
            value, {"width": 600, "height": 150}, describe_design(value)
        )
        == []
    )
