"""Advanced peer states must retain visible identity, local groups and real bounds."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract
from engine.layout.appearance import unit_visible_box, visible_box
from engine.layout.composition import generate_composition_states
from engine.layout.geometry import bounds
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.layout.text import get_measurer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "brand_kit/assets"
FONTS = ROOT / "brand_kit/fonts"
TARGETS = (
    {"width": 728, "height": 90},
    {"width": 1600, "height": 400},
    {"width": 913, "height": 517},
    {"width": 1080, "height": 1920},
)
MUTABLE = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


@pytest.fixture(scope="module")
def advanced_results():
    source = json.loads((ROOT / "input/chameleon_sample_level3_advanced.json").read_text())[
        "source"
    ]
    before = deepcopy(source)
    graph = build_relationship_graph(source, ASSETS)
    graph_before = asdict(graph)
    context = build_context(source, ASSETS, FONTS)
    states = {
        (target["width"], target["height"]): generate_composition_states(source, target, graph)
        for target in TARGETS
    }
    assert source == before
    assert asdict(graph) == graph_before
    return source, graph, context, states


@pytest.mark.parametrize("target", TARGETS)
def test_all_advanced_states_keep_every_frozen_field_rotated_bounds_and_visible_identity(
    advanced_results, target
):
    source, _graph, context, results = advanced_results
    states = results[target["width"], target["height"]]
    if min(target.values()) >= 400:
        assert states, "Representative targets need at least one compound alternative"
    originals = {element["id"]: element for element in source["elements"]}
    for state in states:
        design = state.design
        assert design["canvas"] == target
        assert check_contract(source, design, target, context) == []
        assert [element["id"] for element in design["elements"]] == list(originals)
        for element in design["elements"]:
            original = originals[element["id"]]
            assert {key: value for key, value in element.items() if key not in MUTABLE} == {
                key: value for key, value in original.items() if key not in MUTABLE
            }
            box = bounds(element)
            assert box.x >= -1e-6 and box.y >= -1e-6
            assert box.x + box.width <= target["width"] + 1e-6
            assert box.y + box.height <= target["height"] + 1e-6
        by_id = {element["id"]: element for element in design["elements"]}
        # Reject the observed failure where a stacked state's text survives
        # while its brand images collapse into invisible subpixel frames.
        product = visible_box(by_id["product"], ASSETS)
        logo = visible_box(by_id["logo"], ASSETS)
        assert min(product.width, product.height) >= 1
        assert min(logo.width, logo.height) >= 1
        if min(target.values()) >= 400:
            assert product.height >= min(target.values()) * 0.04


@pytest.mark.parametrize("target", [{"width": 1600, "height": 400}, {"width": 913, "height": 517}])
def test_representative_advanced_targets_offer_readable_badge_labels(advanced_results, target):
    _source, _graph, _context, results = advanced_results
    states = results[target["width"], target["height"]]
    assert any(
        min(
            element["font_size"]
            for element in state.design["elements"]
            if element["id"] in {"badge1_label", "badge2_label", "badge3_label"}
        )
        >= 10
        for state in states
    ), "A custom-size target must not collapse all peer labels while leaving unused space"


def test_portrait_badge_peers_stay_before_their_action_with_complete_footer(advanced_results):
    source, graph, context, results = advanced_results
    states = results[1080, 1920]
    assert states
    for state in states:
        design = state.design
        assert check_contract(source, design, design["canvas"], context) == []
        by_id = {element["id"]: element for element in design["elements"]}
        action = unit_visible_box([by_id["cta_bg"], by_id["cta_label"]], ASSETS)
        peer_boxes = [
            unit_visible_box([by_id[f"badge{index}_bg"], by_id[f"badge{index}_label"]], ASSETS)
            for index in (1, 2, 3)
        ]
        assert max(box.y + box.height for box in peer_boxes) <= action.y + 0.5
        for eid in ("legal_text", "holiday_watermark"):
            box = bounds(by_id[eid])
            assert box.y >= 0
            assert box.y + box.height <= 1920 + 1e-6
        assert by_id["holiday_watermark"]["asset_id"] == "solstice_holiday_badge_2025.png"
        diagnosis = analyze_relationships(graph, design, dict(state.cluster_axes))
        assert not any(
            warning["code"] in {"reading_order", "peer_order", "row_attachment"}
            for warning in diagnosis["warnings"]
        )


def test_group_members_keep_uniform_local_geometry_and_labels_fit_their_frames(advanced_results):
    source, _graph, _context, results = advanced_results
    groups = {}
    for element in source["elements"]:
        if element.get("group_id"):
            groups.setdefault(element["group_id"], []).append(element)
    for states in results.values():
        for state in states:
            by_id = {element["id"]: element for element in state.design["elements"]}
            for members in groups.values():
                anchor, placed = members[0], by_id[members[0]["id"]]
                scale = placed["width"] / anchor["width"]
                for original in members:
                    element = by_id[original["id"]]
                    assert element["width"] == pytest.approx(original["width"] * scale)
                    assert element["height"] == pytest.approx(original["height"] * scale)
                    assert element["x"] - placed["x"] == pytest.approx(
                        (original["x"] - anchor["x"]) * scale, abs=1e-6
                    )
                    assert element["y"] - placed["y"] == pytest.approx(
                        (original["y"] - anchor["y"]) * scale, abs=1e-6
                    )
                    if original["type"] == "text":
                        if original["id"] in {"badge1_label", "badge2_label", "badge3_label"}:
                            assert element["font_size"] >= original["font_size"] * scale - 1e-6
                        else:
                            assert element["font_size"] == pytest.approx(
                                original["font_size"] * scale
                            )
                        measured = get_measurer().measure(
                            element, element["font_size"], element["width"]
                        )
                        assert measured.width <= element["width"] + 1e-5
                        assert measured.height <= element["height"] + 1e-5


def test_advanced_states_use_structure_instead_of_element_or_group_names(advanced_results):
    source, _graph, _context, results = advanced_results
    renamed = deepcopy(source)
    groups = {}
    for index, element in enumerate(renamed["elements"]):
        element["id"] = f"opaque-{index * 193 + 7}"
        if element.get("group_id"):
            groups.setdefault(element["group_id"], f"opaque-group-{len(groups) * 131 + 3}")
            element["group_id"] = groups[element["group_id"]]
    before = deepcopy(renamed)
    graph = build_relationship_graph(renamed, ASSETS)
    actual = generate_composition_states(renamed, {"width": 1600, "height": 400}, graph)
    expected = results[1600, 400]
    assert actual and len(actual) == len(expected)
    for first, second in zip(expected, actual, strict=True):
        assert (first.name, first.topology) == (second.name, second.topology)
        for a, b in zip(first.design["elements"], second.design["elements"], strict=True):
            assert {key: value for key, value in a.items() if key not in {"id", "group_id"}} == {
                key: value for key, value in b.items() if key not in {"id", "group_id"}
            }
    assert renamed == before
