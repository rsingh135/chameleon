"""Coupled parent layouts preserve source size hierarchies across topology cuts."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract, score
from engine.layout.geometry import bounds
from engine.layout.grouped import generate_grouped_states
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from tools.composition_cases import generate_cases

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "brand_kit/assets"
FONTS = ROOT / "brand_kit/fonts"
MUTABLE = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


def _source(filename):
    return json.loads((ROOT / "input" / filename).read_text())["source"]


@pytest.fixture(scope="module")
def cases():
    core = _source("chameleon_sample_level2_core.json")
    advanced = _source("chameleon_sample_level3_advanced.json")
    result = []
    for source, target in (
        (core, {"width": 1600, "height": 400}),
        (advanced, {"width": 1600, "height": 400}),
        (advanced, {"width": 913, "height": 517}),
        (advanced, {"width": 1080, "height": 1920}),
    ):
        graph = build_relationship_graph(source, ASSETS)
        before, graph_before = deepcopy(source), asdict(graph)
        states = generate_grouped_states(source, target, graph)
        assert source == before and asdict(graph) == graph_before
        result.append((source, target, graph, states, build_context(source, ASSETS, FONTS)))
    return result


@pytest.mark.parametrize("case_index", range(4))
def test_parent_states_preserve_source_identity_containment_and_global_font_ratios(
    cases, case_index
):
    source, target, _graph, states, context = cases[case_index]
    assert 0 < len(states) <= 12
    original = {element["id"]: element for element in source["elements"]}
    for state in states:
        design = state.design
        assert check_contract(source, design, target, context) == []
        assert [e["id"] for e in design["elements"]] == list(original)
        footer_exceptions = {
            value.split(":")[1] for value in state.exceptions if value.startswith("footer-text:")
        }
        assert footer_exceptions <= {
            eid for unit in _graph.units if unit.role == "legal" for eid in unit.element_ids
        }
        scales = [
            e["font_size"] / original[e["id"]]["font_size"]
            for e in design["elements"]
            if e["type"] == "text" and e["id"] not in footer_exceptions
        ]
        assert max(scales) == pytest.approx(min(scales), abs=1e-10)
        for element in design["elements"]:
            old = original[element["id"]]
            assert {k: v for k, v in element.items() if k not in MUTABLE} == {
                k: v for k, v in old.items() if k not in MUTABLE
            }
            frame = bounds(element)
            assert frame.x >= -1e-6 and frame.y >= -1e-6
            assert frame.x + frame.width <= target["width"] + 1e-6
            assert frame.y + frame.height <= target["height"] + 1e-6


@pytest.mark.parametrize("case_index", range(4))
def test_source_atomic_controls_and_images_share_the_copy_scale(cases, case_index):
    source, _target, graph, states, _context = cases[case_index]
    original = {element["id"]: element for element in source["elements"]}
    groups = [unit for unit in graph.units if len(unit.members) > 1]
    for state in states:
        output = {element["id"]: element for element in state.design["elements"]}
        headline = next(e for e in source["elements"] if e["type"] == "text")
        scale = output[headline["id"]]["font_size"] / headline["font_size"]
        for unit in graph.units:
            for element_id in unit.element_ids:
                old, new = original[element_id], output[element_id]
                if old["type"] == "image" or any(
                    element_id in group.element_ids for group in groups
                ):
                    for key in ("width", "height", "corner_radius", "stroke_width"):
                        if key in old:
                            assert new[key] == pytest.approx(old[key] * scale)
        for group in groups:
            anchor = group.element_ids[0]
            for element_id in group.element_ids[1:]:
                for key in ("x", "y"):
                    assert output[element_id][key] - output[anchor][key] == pytest.approx(
                        (original[element_id][key] - original[anchor][key]) * scale
                    )


@pytest.mark.parametrize("case_index", range(4))
def test_supported_states_retain_ownership_and_offer_collision_free_choices(cases, case_index):
    source, target, graph, states, context = cases[case_index]
    essentials = {
        "reading_order",
        "explicit_group",
        "peer_order",
        "campaign_ownership",
        "row_attachment",
    }
    for state in states:
        warnings = analyze_relationships(graph, state.design, dict(state.cluster_axes))["warnings"]
        assert not essentials.intersection(w["code"] for w in warnings)
    assert any(
        score(source, state.design, context, target_canvas=target)["metrics"]["raw_collisions"] == 0
        for state in states
    )


def test_unequal_products_do_not_get_normalized_to_equal_slot_sizes():
    task = deepcopy(generate_cases()[0])
    source = task["source"]
    first, second = task["expected_campaigns"]
    originals = {element["id"]: element for element in source["elements"]}
    smaller = originals[second["product_ids"][0]]
    for position, size in (("x", "width"), ("y", "height")):
        center = smaller[position] + smaller[size] / 2
        smaller[size] *= 0.65
        smaller[position] = center - smaller[size] / 2
    graph = build_relationship_graph(source, ASSETS)
    states = generate_grouped_states(source, {"width": 1600, "height": 400}, graph)
    assert states
    for state in states:
        output = {element["id"]: element for element in state.design["elements"]}
        product_ids = (first["product_ids"][0], second["product_ids"][0])
        product_scales = [output[eid]["height"] / originals[eid]["height"] for eid in product_ids]
        assert product_scales[0] == pytest.approx(product_scales[1])
        assert output[product_ids[1]]["height"] / output[product_ids[0]]["height"] == pytest.approx(
            0.65
        )
        for campaign in (first, second):
            eid = campaign["headline_ids"][0]
            assert output[eid]["font_size"] / originals[eid]["font_size"] == pytest.approx(
                product_scales[0]
            )


def test_ambiguous_ownership_declines_parent_relayout():
    task = deepcopy(generate_cases()[1])
    remove = set(task["expected_campaigns"][1]["logo_ids"])
    source = task["source"]
    source["elements"] = [element for element in source["elements"] if element["id"] not in remove]
    graph = build_relationship_graph(source, ASSETS)
    assert graph.ambiguities
    assert generate_grouped_states(source, {"width": 1600, "height": 400}, graph) == ()


def test_bounded_states_are_deterministic_and_return_independent_copies(cases):
    source, target, graph, states, _context = cases[0]
    assert generate_grouped_states(source, target, graph) == states
    design = states[0].design
    design["elements"][0]["width"] = 1
    assert states[0].design["elements"][0]["width"] != 1


def test_tiny_target_never_breaks_proportions_to_force_a_font_floor():
    source = _source("chameleon_sample_level3_advanced.json")
    states = generate_grouped_states(source, {"width": 728, "height": 90})
    originals = {e["id"]: e for e in source["elements"]}
    for state in states:
        footer_exceptions = {value.split(":")[1] for value in state.exceptions}
        scales = [
            e["font_size"] / originals[e["id"]]["font_size"]
            for e in state.design["elements"]
            if e["type"] == "text" and e["id"] not in footer_exceptions
        ]
        assert max(scales) == pytest.approx(min(scales))


def test_independent_footer_readability_is_explicit_and_never_applied_to_campaign_leaves(cases):
    source, _target, graph, states, _context = cases[1]
    original = {element["id"]: element for element in source["elements"]}
    footers = {eid for unit in graph.units if unit.role == "legal" for eid in unit.element_ids}
    campaign_ids = {
        eid
        for cluster in graph.clusters
        if cluster.kind == "campaign"
        for eid in cluster.element_ids
    }
    assert states and footers
    for state in states:
        output = {element["id"]: element for element in state.design["elements"]}
        heading = next(eid for eid in campaign_ids if original[eid]["type"] == "text")
        scale = output[heading]["font_size"] / original[heading]["font_size"]
        exceptions = {value.split(":")[1] for value in state.exceptions}
        assert not exceptions.intersection(campaign_ids)
        for eid in footers:
            floor = min(10.0, original[eid]["font_size"])
            assert output[eid]["font_size"] >= floor
            if original[eid]["font_size"] * scale < floor - 1e-9:
                assert eid in exceptions


@pytest.mark.parametrize("task", generate_cases(), ids=lambda task: task["task_id"])
def test_custom_paired_target_has_readable_optical_packing_with_no_ownership_or_frame_loss(task):
    source, target = task["source"], {"width": 913, "height": 517}
    graph = build_relationship_graph(source, ASSETS)
    context = build_context(source, ASSETS, FONTS)
    states = generate_grouped_states(source, target, graph)
    readable = []
    for state in states:
        design = state.design
        assert check_contract(source, design, target, context) == []
        relation = analyze_relationships(graph, design, dict(state.cluster_axes))
        assert not relation["warnings"]
        by_id = {element["id"]: element for element in design["elements"]}
        original = {element["id"]: element for element in source["elements"]}
        scales = [
            element["font_size"] / original[element["id"]]["font_size"]
            for element in design["elements"]
            if element["type"] == "text"
        ]
        assert max(scales) == pytest.approx(min(scales), abs=1e-10)
        inset = min(target.values()) * 0.04
        for campaign in task["expected_campaigns"]:
            for eid in campaign["action_ids"]:
                box = bounds(by_id[eid])
                assert (
                    min(
                        box.x,
                        box.y,
                        target["width"] - box.x - box.width,
                        target["height"] - box.y - box.height,
                    )
                    >= inset - 1e-6
                )
        if min(e["font_size"] for e in design["elements"] if e["type"] == "text") >= 10:
            assert (
                score(source, design, context, target_canvas=target)["metrics"]["raw_collisions"]
                == 0
            )
            readable.append(state)
    assert readable, "Transparent image padding must not make a feasible custom layout unreadable"
    assert any(state.topology.startswith("grouped/1/") for state in states)
    assert any(state.topology.startswith("grouped/2/") for state in states)


def test_explicit_non_campaign_roots_pack_without_opening_local_geometry():
    source = {
        "canvas": {"width": 600, "height": 500},
        "elements": [
            {
                "id": "plate_a",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 30,
                "y": 20,
                "width": 180,
                "height": 100,
                "group_id": "node_a",
                "fill_color": "#222222",
                "corner_radius": 12,
            },
            {
                "id": "label_a",
                "type": "text",
                "x": 50,
                "y": 50,
                "width": 140,
                "height": 40,
                "group_id": "node_a",
                "font_family": "Solstice Sans",
                "font_size": 18,
                "content": "First station",
                "color": "#ffffff",
            },
            {
                "id": "plate_b",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 330,
                "y": 220,
                "width": 150,
                "height": 120,
                "group_id": "node_b",
                "fill_color": "#555555",
                "corner_radius": 9,
            },
            {
                "id": "label_b",
                "type": "text",
                "x": 345,
                "y": 260,
                "width": 120,
                "height": 40,
                "group_id": "node_b",
                "font_family": "Solstice Sans",
                "font_size": 15,
                "content": "Next station",
                "color": "#ffffff",
            },
        ],
    }
    graph = build_relationship_graph(source, ASSETS)
    assert not any(cluster.kind == "campaign" for cluster in graph.clusters)
    states = generate_grouped_states(source, {"width": 300, "height": 800}, graph)
    assert states
    originals = {e["id"]: e for e in source["elements"]}
    for state in states:
        assert state.topology.startswith("grouped-roots/")
        output = {e["id"]: e for e in state.design["elements"]}
        scales = [output[eid]["width"] / original["width"] for eid, original in originals.items()]
        assert max(scales) == pytest.approx(min(scales))
        assert output["label_a"]["x"] - output["plate_a"]["x"] == pytest.approx(20 * scales[0])
        assert output["label_b"]["y"] - output["plate_b"]["y"] == pytest.approx(40 * scales[0])


def test_a_loose_diagram_connector_prevents_independent_root_packing():
    source = {
        "canvas": {"width": 600, "height": 500},
        "elements": [
            {
                "id": "a",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 20,
                "y": 20,
                "width": 100,
                "height": 100,
                "group_id": "a",
                "fill_color": "#222222",
            },
            {
                "id": "connector",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 200,
                "y": 100,
                "width": 150,
                "height": 8,
                "fill_color": "#555555",
            },
        ],
    }
    assert generate_grouped_states(source, {"width": 300, "height": 800}) == ()
