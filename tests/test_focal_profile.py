"""A requested focal anchor is source evidence, not a universal centering rule."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from engine.layout.appearance import unit_visible_box
from engine.layout.focal import analyze_focal, infer_focal_profile
from engine.layout.relationships import build_relationship_graph
from engine.resources import using_resources
from tools.composition_cases import generate_cases

ROOT = Path(__file__).resolve().parents[1]


def source(level="level2_core"):
    return json.loads((ROOT / f"input/chameleon_sample_{level}.json").read_text())["source"]


def centered_source():
    original = source()
    graph = build_relationship_graph(original)
    hero = next(unit for unit in graph.units if unit.role == "hero")
    box = unit_visible_box(
        [e for e in original["elements"] if e["id"] in hero.element_ids], graph.assets_dir
    )
    shift = original["canvas"]["width"] / 2 - box.x - box.width / 2
    for element in original["elements"]:
        if element["id"] in hero.element_ids:
            element["x"] += shift
    return original


def test_unchanged_source_has_full_center_span_and_hierarchy_retention():
    original = centered_source()
    result = analyze_focal(original, deepcopy(original))
    assert result["score"] == 1.0
    assert result["warnings"] == []
    assert result["metrics"] == {
        "active": True,
        "center_error": 0.0,
        "center_retained": True,
        "visible_span_retention": 1.0,
        "visible_span_ratio": 1.0,
        "occupied_balance_error": 0.0,
        "balance_score": 1.0,
        "focal_product_scale_vs_headline": 1.0,
        "focal_product_scale_vs_cta": 1.0,
    }


@pytest.mark.parametrize("level", ["level2_core", "level3_advanced"])
def test_supplied_single_hero_sources_evidence_only_a_horizontal_anchor(level):
    original = source(level)
    profile = infer_focal_profile(original)
    assert profile is not None
    assert profile.anchor_x == 0.5
    assert len(profile.supporting_units) >= 2
    assert abs(profile.source_center_x - 0.5) < 0.001
    assert "no y-center" in " ".join(profile.provenance)
    if level == "level3_advanced":
        box = profile.source_visible_box
        assert (box.y + box.height / 2) / original["canvas"]["height"] < 0.3


def test_frozen_profile_and_analysis_do_not_mutate_the_source_or_graph():
    original = source()
    graph = build_relationship_graph(original)
    before, graph_before = deepcopy(original), asdict(graph)
    profile = infer_focal_profile(original, graph)
    analyze_focal(original, deepcopy(original), graph, profile)
    assert original == before
    assert asdict(graph) == graph_before
    with pytest.raises(FrozenInstanceError):
        profile.anchor_x = 0.2


def test_opaque_ids_and_group_names_keep_the_same_evidence():
    original = source()
    expected = infer_focal_profile(original)
    names = {e["id"]: f"opaque-{index * 7919}" for index, e in enumerate(original["elements"])}
    changed = deepcopy(original)
    groups = {}
    for element in changed["elements"]:
        element["id"] = names[element["id"]]
        if element.get("group_id"):
            group = element["group_id"]
            element["group_id"] = groups.setdefault(group, f"group-{len(groups)}")
    actual = infer_focal_profile(changed)
    assert actual is not None
    assert actual.primary_element_ids == tuple(names[eid] for eid in expected.primary_element_ids)
    assert actual.source_center_x == expected.source_center_x
    assert actual.source_visible_height_fraction == expected.source_visible_height_fraction


def test_pushed_side_product_warns_even_when_every_size_and_font_is_unchanged():
    original = centered_source()
    graph = build_relationship_graph(original)
    profile = infer_focal_profile(original, graph)
    candidate = deepcopy(original)
    for element in candidate["elements"]:
        if element["id"] in profile.primary_element_ids:
            element["x"] -= original["canvas"]["width"] * 0.2
    result = analyze_focal(original, candidate, graph, profile)
    assert result["metrics"]["center_error"] == 0.2
    assert result["metrics"]["center_retained"] is False
    assert result["metrics"]["visible_span_retention"] == 1.0
    assert result["metrics"]["focal_product_scale_vs_headline"] == 1.0
    assert result["metrics"]["focal_product_scale_vs_cta"] == 1.0
    assert any(w["code"] == "focal_center_drift" for w in result["warnings"])


def test_y_movement_does_not_become_an_invented_vertical_center_requirement():
    original = centered_source()
    profile = infer_focal_profile(original)
    candidate = deepcopy(original)
    for element in candidate["elements"]:
        if element["id"] in profile.primary_element_ids:
            element["y"] -= 130
    result = analyze_focal(original, candidate, profile=profile)
    assert result["score"] == 1.0
    assert result["metrics"]["center_retained"] is True


def test_uniform_canvas_scale_preserves_normalized_anchor_and_size_diagnostics():
    original = centered_source()
    candidate = deepcopy(original)
    candidate["canvas"] = {key: value * 2 for key, value in original["canvas"].items()}
    for element in candidate["elements"]:
        for key in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in element:
                element[key] *= 2
    result = analyze_focal(original, candidate)
    assert result["score"] == 1.0
    assert result["metrics"]["visible_span_retention"] == 1.0
    assert result["metrics"]["focal_product_scale_vs_headline"] == 1.0
    assert result["metrics"]["focal_product_scale_vs_cta"] == 1.0


def test_asymmetric_source_is_not_assigned_a_canvas_center_requirement():
    original = source()
    for element in original["elements"]:
        if element["id"] in {"product", "logo", "headline", "cta_bg", "cta_label"}:
            element["x"] -= 180
    assert infer_focal_profile(original) is None
    assert analyze_focal(original, deepcopy(original))["metrics"]["active"] is False


@pytest.mark.parametrize("case", [0, 1], ids=["columns", "stacked"])
def test_multiple_product_campaigns_are_never_forced_to_one_focal_center(case):
    original = generate_cases()[case]["source"]
    assert infer_focal_profile(original) is None
    assert analyze_focal(original, deepcopy(original))["metrics"]["active"] is False


def test_ambiguous_relationship_graph_declines_even_if_the_pixels_are_centered():
    original = source()
    graph = build_relationship_graph(original)
    assert infer_focal_profile(original, replace(graph, ambiguities=(object(),))) is None


def test_missing_or_fully_transparent_image_is_not_focal_evidence(tmp_path):
    original = source()
    product = next(e for e in original["elements"] if e["id"] == "product")
    product["asset_id"] = "missing.png"
    with using_resources(tmp_path):
        assert infer_focal_profile(original) is None
        Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(tmp_path / "missing.png")
        assert infer_focal_profile(original) is None


def padded_source(tmp_path):
    image = Image.new("RGBA", (100, 100))
    ImageDraw.Draw(image).rectangle((0, 20, 39, 79), fill="orange")
    image.save(tmp_path / "opaque-padding.png")
    original = source()
    product = next(e for e in original["elements"] if e["id"] == "product")
    product.update(asset_id="opaque-padding.png", x=460, y=300, width=400, height=400)
    return original


def test_unequal_transparent_padding_uses_the_painted_center_not_the_frame(tmp_path):
    original = padded_source(tmp_path)
    with using_resources(tmp_path):
        graph = build_relationship_graph(original)
        profile = infer_focal_profile(original, graph)
    assert profile is not None
    assert profile.source_center_x == 0.5
    assert profile.source_visible_box.width == 160
    # Profile resources remain bound even outside the temporary resource context.
    result = analyze_focal(original, deepcopy(original), profile=profile)
    assert result["metrics"]["active"] is True
    assert result["metrics"]["center_retained"] is True
    assert result["score"] == 1.0


def test_centered_frame_with_off_center_ink_is_not_centered_source_evidence(tmp_path):
    original = padded_source(tmp_path)
    next(e for e in original["elements"] if e["id"] == "product")["x"] = 340
    with using_resources(tmp_path):
        assert infer_focal_profile(original) is None


def test_independent_text_and_cta_inflation_remains_a_separate_soft_diagnostic():
    original = centered_source()
    graph = build_relationship_graph(original)
    profile = infer_focal_profile(original, graph)
    candidate = deepcopy(original)
    headline_ids = {eid for eid, _ in profile.headline_font_sizes}
    action_ids = {eid for ids, _ in profile.cta_heights for eid in ids}
    for element in candidate["elements"]:
        if element["id"] in headline_ids:
            element["font_size"] *= 2
        if element["id"] in action_ids:
            element["height"] *= 2
    result = analyze_focal(original, candidate, graph, profile)
    assert result["metrics"]["center_retained"] is True
    assert result["metrics"]["focal_product_scale_vs_headline"] == 0.5
    assert result["metrics"]["focal_product_scale_vs_cta"] == 0.5
    assert {w["code"] for w in result["warnings"]} >= {
        "focal_product_scale_vs_headline",
        "focal_product_scale_vs_cta",
    }


def test_smaller_centered_product_cannot_claim_full_source_visible_span():
    original = centered_source()
    profile = infer_focal_profile(original)
    candidate = deepcopy(original)
    for element in candidate["elements"]:
        if element["id"] in profile.primary_element_ids:
            element["x"] += element["width"] / 4
            element["y"] += element["height"] / 4
            element["width"] *= 0.5
            element["height"] *= 0.5
    result = analyze_focal(original, candidate, profile=profile)
    assert result["metrics"]["center_retained"] is True
    assert result["metrics"]["visible_span_retention"] == 0.5
    assert any(w["code"] == "focal_visible_span" for w in result["warnings"])


def test_occupied_balance_preserves_the_source_centroid_rather_than_forcing_half():
    original = centered_source()
    profile = infer_focal_profile(original)
    # A source corner promotion makes occupied foreground mass asymmetric even
    # though its hero is centered. That authored asymmetry is the reference.
    assert abs(profile.source_occupied_centroid_x - 0.5) > 0.01
    result = analyze_focal(original, deepcopy(original), profile=profile)
    assert result["metrics"]["occupied_balance_error"] == 0.0
    assert result["metrics"]["balance_score"] == 1.0


def test_moving_all_copy_right_loses_soft_balance_with_the_same_centered_hero():
    original = centered_source()
    graph = build_relationship_graph(original)
    profile = infer_focal_profile(original, graph)
    shifted = deepcopy(original)
    copy_ids = {
        eid
        for unit in graph.units
        if unit.role in {"headline", "body", "action"}
        for eid in unit.element_ids
    }
    for element in shifted["elements"]:
        if element["id"] in copy_ids:
            element["x"] += 60
    unchanged = analyze_focal(original, original, graph, profile)
    result = analyze_focal(original, shifted, graph, profile)
    assert result["metrics"]["center_retained"] is True
    assert result["metrics"]["center_error"] == unchanged["metrics"]["center_error"]
    assert result["metrics"]["occupied_balance_error"] > 0.02
    assert result["metrics"]["balance_score"] < unchanged["metrics"]["balance_score"]
