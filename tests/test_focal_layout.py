"""Source-centered product placement must survive a landscape reflow."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract, score
from engine.layout.appearance import visible_box
from engine.layout.focal import analyze_focal
from engine.layout.focal_states import generate_focal_states
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.resize import resize_with_state

ROOT = Path(__file__).resolve().parents[1]


def test_centered_source_product_keeps_its_canvas_axis_in_landscape():
    task = json.loads((ROOT / "input/chameleon_sample_level2_core.json").read_text())
    source, target = task["source"], task["target_canvas"]
    plan = resize_with_state(source, target)
    product = next(e for e in plan["design"]["elements"] if e["id"] == "product")
    painted = visible_box(product)
    center_x = (painted.x + painted.width / 2) / target["width"]
    assert center_x == pytest.approx(0.5, abs=0.025)
    report = analyze_focal(source, plan["design"])
    assert report["metrics"]["occupied_balance_error"] <= 0.08
    assert report["metrics"]["focal_product_scale_vs_headline"] == pytest.approx(1, abs=0.05)
    assert report["metrics"]["focal_product_scale_vs_cta"] == pytest.approx(1, abs=0.05)
    context = build_context(source)
    assert check_contract(source, plan["design"], target, context) == []
    assert score(source, plan["design"], context)["metrics"]["raw_collisions"] == 0
    assert min(e["font_size"] for e in plan["design"]["elements"] if e["type"] == "text") >= 10


def test_focal_fit_handles_a_source_valid_unbreakable_headline():
    task = json.loads((ROOT / "input/chameleon_sample_level2_core.json").read_text())
    source = task["source"]
    headline = next(e for e in source["elements"] if e["id"] == "headline")
    headline.update(content="W" * 12, style_runs=[])
    original = deepcopy(source)
    target = {"width": 913, "height": 517}
    context = build_context(source)
    states = generate_focal_states(source, target)
    assert states
    for state in states:
        assert check_contract(source, state.design, target, context) == []
    assert source == original


def test_body_scale_exceptions_leave_controls_and_primary_hierarchy_rigid():
    task = json.loads((ROOT / "input/chameleon_sample_level2_core.json").read_text())
    source, target = task["source"], task["target_canvas"]
    original = {e["id"]: e for e in source["elements"]}
    states = generate_focal_states(source, target)
    assert len(states) <= 4
    for state in states:
        output = {e["id"]: e for e in state.design["elements"]}
        scale = output["product"]["width"] / original["product"]["width"]
        assert output["headline"]["font_size"] / original["headline"]["font_size"] == pytest.approx(
            scale
        )
        for eid in ("cta_bg", "cta_label", "badge_bg", "badge_label"):
            for key in ("width", "height", "font_size", "corner_radius"):
                if key in original[eid]:
                    assert output[eid][key] == pytest.approx(original[eid][key] * scale)
        assert output["subhead"]["font_size"] <= original["subhead"]["font_size"] * scale
        for element in state.design["elements"]:
            assert element["id"] in original
            for key in (
                "content",
                "font_family",
                "asset_id",
                "color",
                "opacity",
                "group_id",
                "z_index",
            ):
                assert element.get(key) == original[element["id"]].get(key)


def test_badges_follow_their_body_action_column_when_headline_moves_to_another_region():
    task = json.loads((ROOT / "input/chameleon_sample_level3_advanced.json").read_text())
    source = task["source"]
    graph = build_relationship_graph(source)
    state = next(
        state
        for state in generate_focal_states(source, {"width": 913, "height": 517}, graph)
        if state.name == "focal-balanced-1"
    )
    report = analyze_relationships(graph, state.design, dict(state.cluster_axes))
    assert not any(w["code"] == "row_attachment" for w in report["warnings"])
    detached = state.design
    for element in detached["elements"]:
        if element.get("group_id") in {"badge1", "badge2", "badge3"}:
            element["x"] -= 300
    report = analyze_relationships(graph, detached, dict(state.cluster_axes))
    assert any(w["code"] == "row_attachment" for w in report["warnings"])
