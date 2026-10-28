"""Source relationships describe meaning that element-preservation misses."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from engine.eval import check_contract
from engine.eval.style import analyze_style
from engine.layout.relationships import analyze_relationships, build_relationship_graph

ROOT = Path(__file__).resolve().parents[1]


def source(level="level2_core"):
    return json.loads((ROOT / f"input/chameleon_sample_{level}.json").read_text())["source"]


@pytest.mark.parametrize("level", ["level1_warmup", "level2_core", "level3_advanced"])
def test_original_relationships_are_clean(level):
    original = source(level)
    graph = build_relationship_graph(original)
    result = analyze_relationships(graph, original)
    assert result["score"] == 1
    assert result["warnings"] == []


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_uniform_scaling_retains_relationships_and_group_geometry(scale):
    original = source()
    graph = build_relationship_graph(original)
    candidate = deepcopy(original)
    candidate["canvas"] = {key: round(value * scale) for key, value in original["canvas"].items()}
    for element in candidate["elements"]:
        for key in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in element:
                element[key] *= scale
    result = analyze_relationships(graph, candidate)
    assert result["warnings"] == []
    assert result["score"] == pytest.approx(1, abs=1e-5)


def test_message_cluster_contains_copy_and_its_action():
    graph = build_relationship_graph(source())
    message = next(cluster for cluster in graph.clusters if cluster.kind == "message")
    assert set(message.element_ids) == {"headline", "subhead", "cta_bg", "cta_label"}
    assert [graph.units[index].role for index in message.unit_indices] == [
        "headline",
        "body",
        "action",
    ]
    assert message.confidence < 1
    assert message.provenance
    assert {edge.code for edge in graph.edges if edge.cluster_id == message.id} == {
        "reading_order",
        "message_attachment",
        "shared_axis",
    }


def test_reversed_reading_order_is_detected_even_when_style_checks_pass():
    original = source()
    candidate = deepcopy(original)
    next(element for element in candidate["elements"] if element["id"] == "subhead")["y"] = 700
    assert check_contract(original, candidate, original["canvas"]) == []
    assert analyze_style(original, candidate)["warnings"] == []
    result = analyze_relationships(build_relationship_graph(original), candidate)
    assert "reading_order" in {warning["code"] for warning in result["warnings"]}
    assert result["metrics"]["reading_order"] < 1


def test_action_ownership_is_detected_even_when_style_checks_pass():
    original = source()
    candidate = deepcopy(original)
    for element in candidate["elements"]:
        if element.get("group_id") == "cta":
            element["x"] -= 280
    assert check_contract(original, candidate, original["canvas"]) == []
    assert analyze_style(original, candidate)["warnings"] == []
    result = analyze_relationships(build_relationship_graph(original), candidate)
    warnings = [warning for warning in result["warnings"] if warning["code"] == "shared_axis"]
    assert any("cta_label" in warning["element_ids"] for warning in warnings)


def test_complete_message_block_can_move_without_breaking_internal_relations():
    original = source()
    graph = build_relationship_graph(original)
    candidate = deepcopy(original)
    message = next(cluster for cluster in graph.clusters if cluster.kind == "message")
    for element in candidate["elements"]:
        if element["id"] in message.element_ids:
            element["x"] += 180
            element["y"] += 100
    result = analyze_relationships(graph, candidate)
    assert not [warning for warning in result["warnings"] if warning["cluster_id"] == message.id]


def test_brand_association_is_an_explicit_low_confidence_hypothesis():
    original = source()
    graph = build_relationship_graph(original)
    brand = next(cluster for cluster in graph.clusters if cluster.kind == "brand")
    assert set(brand.element_ids) == {"logo", "product"}
    assert brand.confidence == 0.7
    assert "hypothesis" in " ".join(brand.provenance)
    candidate = deepcopy(original)
    next(element for element in candidate["elements"] if element["id"] == "logo")["x"] += 250
    result = analyze_relationships(graph, candidate)
    assert "brand_association" in {warning["code"] for warning in result["warnings"]}


def test_unaligned_logo_is_not_forced_into_product_group():
    original = source()
    next(element for element in original["elements"] if element["id"] == "logo")["x"] = 80
    graph = build_relationship_graph(original)
    assert not [cluster for cluster in graph.clusters if cluster.kind == "brand"]


def test_explicit_group_detects_detached_label():
    original = source()
    candidate = deepcopy(original)
    next(element for element in candidate["elements"] if element["id"] == "cta_label")["y"] -= 150
    result = analyze_relationships(build_relationship_graph(original), candidate)
    assert "explicit_group" in {warning["code"] for warning in result["warnings"]}


def _text(name, x, y, size):
    return {
        "id": name,
        "type": "text",
        "x": x,
        "y": y,
        "width": 300,
        "height": 60,
        "font_size": size,
        "font_family": "Solstice Sans",
        "content": "Title" if size == 40 else "Body",
        "color": "#000000",
    }


def test_multiple_messages_keep_local_ownership_instead_of_global_role_buckets():
    original = {
        "canvas": {"width": 1100, "height": 500},
        "elements": [
            _text("a", 20, 20, 40),
            _text("b", 20, 100, 20),
            _text("c", 600, 20, 40),
            _text("d", 600, 100, 20),
        ],
    }
    graph = build_relationship_graph(original)
    messages = [cluster for cluster in graph.clusters if cluster.kind == "message"]
    assert {frozenset(cluster.element_ids) for cluster in messages} == {
        frozenset(("a", "b")),
        frozenset(("c", "d")),
    }


def test_flow_changes_require_an_explicit_cluster_state():
    original = {
        "canvas": {"width": 700, "height": 500},
        "elements": [_text("a", 20, 20, 40), _text("b", 20, 100, 20)],
    }
    graph = build_relationship_graph(original)
    message = graph.clusters[0]
    candidate = deepcopy(original)
    candidate["elements"][1].update(x=200, y=20)
    unchanged = analyze_relationships(graph, candidate)
    changed = analyze_relationships(graph, candidate, {message.id: "horizontal"})
    assert "shared_axis" in {warning["code"] for warning in unchanged["warnings"]}
    assert changed["warnings"] == []
    with pytest.raises(ValueError, match="known cluster"):
        analyze_relationships(graph, candidate, {"unknown": "horizontal"})


def test_graph_is_frozen_independent_of_source_mutations_and_json_serializable():
    original = source()
    graph = build_relationship_graph(original)
    before = asdict(graph)
    original["elements"][3]["x"] += 200
    assert asdict(graph) == before
    with pytest.raises(FrozenInstanceError):
        graph.units[0].role = "other"
    with pytest.raises(FrozenInstanceError):
        graph.units[0].box.x = 0
    json.dumps(asdict(graph), allow_nan=False)


def test_graph_is_deterministic_nonmutating_and_independent_of_sample_names():
    original = source()
    before = deepcopy(original)
    first = build_relationship_graph(original)
    assert first == build_relationship_graph(original)
    assert original == before
    renamed = deepcopy(original)
    group_names = {}
    for index, element in enumerate(renamed["elements"]):
        element["id"] = f"opaque-{index}"
        if element.get("group_id"):
            group_names.setdefault(element["group_id"], f"set-{len(group_names)}")
            element["group_id"] = group_names[element["group_id"]]
    second = build_relationship_graph(renamed)
    assert [(edge.code, edge.source_unit, edge.target_unit, edge.axis) for edge in first.edges] == [
        (edge.code, edge.source_unit, edge.target_unit, edge.axis) for edge in second.edges
    ]
    assert [(cluster.kind, cluster.unit_indices) for cluster in first.clusters] == [
        (cluster.kind, cluster.unit_indices) for cluster in second.clusters
    ]
    assert analyze_relationships(second, renamed)["warnings"] == []


def test_analysis_requires_source_identity_and_ignores_invisible_anchors():
    original = source()
    graph = build_relationship_graph(original)
    candidate = deepcopy(original)
    candidate["elements"].pop()
    with pytest.raises(ValueError, match="every original"):
        analyze_relationships(graph, candidate)
    for element in original["elements"]:
        if element["id"] == "logo":
            element["opacity"] = 0
    invisible = build_relationship_graph(original)
    assert not [cluster for cluster in invisible.clusters if cluster.kind == "brand"]
