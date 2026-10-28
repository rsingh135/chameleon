"""Compound states preserve source relationships rather than isolated roles."""

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from engine.eval import check_contract, score
from engine.layout.appearance import unit_visible_box
from engine.layout.composition import generate_composition_states
from engine.layout.geometry import bounds
from engine.layout.relationships import analyze_relationships, build_relationship_graph

ROOT = Path(__file__).resolve().parents[1]


def source(level="level2_core"):
    return json.loads((ROOT / f"input/chameleon_sample_{level}.json").read_text())["source"]


@pytest.mark.parametrize(
    "target",
    [
        {"width": 1600, "height": 400},
        {"width": 913, "height": 517},
        {"width": 1080, "height": 1920},
        {"width": 320, "height": 50},
        {"width": 160, "height": 600},
        {"width": 1600, "height": 80},
        {"width": 10, "height": 10},
    ],
)
def test_every_returned_state_preserves_the_contract_and_complete_rotated_bounds(target):
    original = source()
    states = generate_composition_states(original, target)
    assert isinstance(states, tuple)
    assert len(states) <= 4
    if min(target.values()) >= 400:
        assert states, "Representative canvases should have a coherent compound alternative"
    for state in states:
        design = state.design
        assert design["canvas"] == target
        assert check_contract(original, design, target) == []
        assert [element["id"] for element in design["elements"]] == [
            element["id"] for element in original["elements"]
        ]
        for element in design["elements"]:
            box = bounds(element)
            assert box.x >= -1e-6 and box.y >= -1e-6
            assert box.x + box.width <= target["width"] + 1e-6
            assert box.y + box.height <= target["height"] + 1e-6


@pytest.mark.parametrize(
    "target",
    [
        {"width": 1600, "height": 400},
        {"width": 913, "height": 517},
        {"width": 1080, "height": 1920},
    ],
)
def test_core_states_keep_brand_axis_message_ownership_and_explicit_groups(target):
    original = source()
    graph = build_relationship_graph(original)
    brand = next(cluster for cluster in graph.clusters if cluster.kind == "brand")
    message = next(cluster for cluster in graph.clusters if cluster.kind == "message")
    states = generate_composition_states(original, target, graph)
    assert states
    assert any(
        min(
            element["font_size"]
            for element in state.design["elements"]
            if element["type"] == "text"
        )
        >= 10
        for state in states
    )
    for state in states:
        design = state.design
        result = analyze_relationships(graph, design, dict(state.cluster_axes))
        assert result["warnings"] == []
        assert result["metrics"]["explicit_group"] == pytest.approx(1)
        by_id = {element["id"]: element for element in design["elements"]}
        brand_boxes = [
            unit_visible_box([by_id[eid] for eid in graph.units[index].element_ids])
            for index in brand.unit_indices
        ]
        centers = [box.x + box.width / 2 for box in brand_boxes]
        assert centers[0] == pytest.approx(centers[1], abs=1e-6)
        assert brand_boxes[0].y + brand_boxes[0].height < brand_boxes[1].y
        message_boxes = [
            unit_visible_box([by_id[eid] for eid in graph.units[index].element_ids])
            for index in message.unit_indices
        ]
        assert [box.y for box in message_boxes] == sorted(box.y for box in message_boxes)
        assert (
            max(box.x + box.width / 2 for box in message_boxes)
            - min(box.x + box.width / 2 for box in message_boxes)
            < 1e-6
        )


def test_core_banner_has_readable_text_in_every_returned_state():
    original = source()
    states = generate_composition_states(original, {"width": 1600, "height": 400})
    assert states
    for state in states:
        metrics = score(original, state.design)["metrics"]
        assert metrics["raw_min_font_px"] >= 10
        assert metrics["raw_collisions"] == 0


def test_corner_accents_are_included_in_complete_state_bounds():
    original = source()
    for element in original["elements"]:
        if element.get("group_id") == "badge":
            element.update(x=0, width=530)
    target = {"width": 400, "height": 1700}
    states = generate_composition_states(original, target)
    assert states, "The smaller typography state fits; reject only infeasible alternatives"
    for state in states:
        assert check_contract(original, state.design, target) == []
        for element in state.design["elements"]:
            box = bounds(element)
            assert box.x >= -1e-6 and box.y >= -1e-6
            assert box.x + box.width <= target["width"] + 1e-6
            assert box.y + box.height <= target["height"] + 1e-6


def test_rotated_accent_group_keeps_identity_and_frame_bounds():
    original = source()
    for element in original["elements"]:
        if element.get("group_id") == "badge":
            element["rotation"] = -12
    target = {"width": 1600, "height": 400}
    states = generate_composition_states(original, target)
    assert states
    for state in states:
        assert check_contract(original, state.design, target) == []
        graph = build_relationship_graph(original)
        assert analyze_relationships(graph, state.design, dict(state.cluster_axes))["metrics"][
            "explicit_group"
        ] == pytest.approx(1)


def test_states_are_repeatable_immutable_and_tied_to_the_original_source():
    original = source()
    before = deepcopy(original)
    graph = build_relationship_graph(original)
    graph_before = asdict(graph)
    target = {"width": 1600, "height": 400}
    states = generate_composition_states(original, target, graph)
    assert states == generate_composition_states(original, target, graph)
    assert original == before
    assert asdict(graph) == graph_before
    digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    assert all(state.source_digest == digest for state in states)
    assert all((state.width, state.height) == (1600, 400) for state in states)
    design = states[0].design
    design["elements"][0]["x"] = 9000
    assert states[0].design["elements"][0]["x"] != 9000
    assert original == before
    with pytest.raises(FrozenInstanceError):
        states[0].topology = "other"


def test_opaque_element_and_group_names_do_not_change_geometry():
    original = source()
    renamed = deepcopy(original)
    groups = {}
    for index, element in enumerate(renamed["elements"]):
        element["id"] = f"opaque-{index}"
        if element.get("group_id"):
            groups.setdefault(element["group_id"], f"group-{len(groups)}")
            element["group_id"] = groups[element["group_id"]]
    target = {"width": 1600, "height": 400}
    first = generate_composition_states(original, target)
    second = generate_composition_states(renamed, target)
    assert first and len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert (a.name, a.topology, a.cluster_axes) == (b.name, b.topology, b.cluster_axes)
        for expected, actual in zip(a.design["elements"], b.design["elements"], strict=True):
            assert {
                key: value for key, value in expected.items() if key not in {"id", "group_id"}
            } == {key: value for key, value in actual.items() if key not in {"id", "group_id"}}


def test_unsupported_advanced_source_declines_instead_of_forcing_a_single_message_family():
    original = source("level3_advanced")
    before = deepcopy(original)
    assert generate_composition_states(original, {"width": 1600, "height": 400}) == ()
    assert original == before


def test_two_independent_product_pairs_decline_the_single_brand_family():
    original = source()
    original["canvas"]["width"] = 2160
    for element in list(original["elements"]):
        if element["id"] in {"logo", "product"}:
            duplicate = deepcopy(element)
            duplicate["id"] = "second-" + duplicate["id"]
            duplicate["x"] += 1080
            original["elements"].append(duplicate)
    before = deepcopy(original)
    assert generate_composition_states(original, {"width": 1600, "height": 400}) == ()
    assert original == before
