"""Source block inference and exact local similarity transforms."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from engine.layout.geometry import Box, union
from engine.layout.hierarchy import fit_cluster, infer_hierarchy, transform_cluster
from engine.layout.relationships import build_relationship_graph
from tools.composition_cases import generate_cases

ROOT = Path(__file__).resolve().parents[1]


def source(level="level2_core"):
    return json.loads((ROOT / f"input/chameleon_sample_{level}.json").read_text())["source"]


def _changed_geometry(original, scale=1.0, dx=0.0, dy=0.0):
    changed = deepcopy(original)
    changed["canvas"] = {key: value * scale for key, value in original["canvas"].items()}
    for element in changed["elements"]:
        for key in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in element:
                element[key] *= scale
        element["x"] += dx
        element["y"] += dy
    return changed


def _membership(hierarchy):
    return {(node.kind, frozenset(node.element_ids)) for node in hierarchy.clusters}


@pytest.mark.parametrize("level", ["level1_warmup", "level2_core", "level3_advanced"])
def test_roots_partition_foreground_without_merging_background_layers(level):
    original = source(level)
    frozen = deepcopy(original)
    graph = build_relationship_graph(original)
    graph_snapshot = asdict(graph)
    hierarchy = infer_hierarchy(original, graph)
    member_ids = [eid for root in hierarchy.roots for eid in root.element_ids]
    foreground = {eid for unit in graph.units for eid in unit.element_ids}
    assert set(member_ids) == foreground
    assert len(member_ids) == len(set(member_ids))
    assert not foreground & set(hierarchy.background_ids + hierarchy.decoration_ids)
    assert foreground | set(hierarchy.background_ids + hierarchy.decoration_ids) == {
        element["id"] for element in original["elements"]
    }
    assert original == frozen
    assert asdict(graph) == graph_snapshot


def test_hierarchy_keeps_relationship_evidence_and_nested_atomic_controls():
    original = source()
    graph = build_relationship_graph(original)
    hierarchy = infer_hierarchy(original, graph)
    for inferred in graph.clusters:
        node = hierarchy.cluster(inferred.id)
        assert node.confidence == inferred.confidence
        assert node.provenance == inferred.provenance
    brand = hierarchy.cluster("brand:0")
    assert brand.confidence == 0.7
    assert "hypothesis" in " ".join(brand.provenance)
    message = hierarchy.cluster("message:0")
    controls = [hierarchy.cluster(name) for name in message.children]
    action = next(node for node in controls if "cta_bg" in node.element_ids)
    assert action.kind == "explicit"
    assert set(action.element_ids) == {"cta_bg", "cta_label"}
    assert action.children == ()
    campaign = next(node for node in hierarchy.roots if node.kind == "campaign")
    assert {"brand:0", "message:0"} <= set(campaign.children)


@pytest.mark.parametrize("case", [0, 1], ids=["columns", "stacked"])
def test_different_campaigns_cannot_become_one_similarity_block(case):
    original = generate_cases()[case]["source"]
    graph = build_relationship_graph(original)
    hierarchy = infer_hierarchy(original, graph)
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    assert len(campaigns) == 2
    assert {frozenset(root.element_ids) for root in hierarchy.roots} == {
        frozenset(campaign.element_ids) for campaign in campaigns
    }
    for node in hierarchy.clusters:
        owners = [
            campaign for campaign in campaigns if set(node.element_ids) & set(campaign.element_ids)
        ]
        assert len(owners) <= 1


def test_opaque_element_and_group_names_do_not_change_membership():
    original = source("level3_advanced")
    names = {
        element["id"]: f"u{number * 7919}" for number, element in enumerate(original["elements"])
    }
    renamed = deepcopy(original)
    for element in renamed["elements"]:
        element["id"] = names[element["id"]]
        if element.get("group_id"):
            element["group_id"] = "opaque-" + str(len(element["group_id"])) + element["group_id"]
    actual = _membership(infer_hierarchy(renamed))
    expected = {
        (kind, frozenset(names[eid] for eid in members))
        for kind, members in _membership(infer_hierarchy(original))
    }
    assert actual == expected


@pytest.mark.parametrize("scale", [0.5, 2.5])
def test_uniform_source_scaling_preserves_membership_and_normalized_geometry(scale):
    original = source()
    before = infer_hierarchy(original)
    after = infer_hierarchy(_changed_geometry(original, scale))
    assert _membership(before) == _membership(after)
    for node in before.clusters:
        changed = after.cluster(node.id)
        assert node.kind == changed.kind
        for first, second in zip(node.members, changed.members, strict=True):
            assert first.element_id == second.element_id
            for key in ("x", "y", "width", "height", "font_size"):
                a, b = getattr(first, key), getattr(second, key)
                if a is None:
                    assert b is None
                else:
                    assert a == pytest.approx(b, abs=1e-6)


def _text(name, y):
    return {
        "id": name,
        "type": "text",
        "x": 20,
        "y": y,
        "width": 300,
        "height": 25,
        "font_size": 20,
        "font_family": "Solstice Sans",
        "content": "Short label",
        "color": "#000000",
    }


def test_residual_complete_link_does_not_chain_three_adjacent_labels():
    original = {
        "canvas": {"width": 500, "height": 500},
        "elements": [_text("a", 20), _text("b", 50), _text("c", 80)],
    }
    before = infer_hierarchy(original)
    spatial = [node for node in before.clusters if node.kind == "spatial"]
    assert len(spatial) == 1
    assert len(spatial[0].element_ids) == 2
    assert spatial[0].confidence == 0.55
    assert "geometric hypothesis" in " ".join(spatial[0].provenance)
    # Translation changes absolute positions, not local grouping or coordinates.
    after = infer_hierarchy(_changed_geometry(original, dx=25, dy=25))
    assert _membership(before) == _membership(after)
    for root in before.roots:
        for first, second in zip(root.members, after.cluster(root.id).members, strict=True):
            assert asdict(first) == pytest.approx(asdict(second))


def test_repeated_images_do_not_establish_one_spatial_block():
    original = {
        "canvas": {"width": 500, "height": 500},
        "elements": [
            {"id": name, "type": "image", "x": x, "y": 20, "width": 100, "height": 100}
            for name, x in (("a", 20), ("b", 130))
        ],
    }
    assert len(infer_hierarchy(original).roots) == 2


@pytest.mark.parametrize("scale", [0.4, 1.0, 2.3])
def test_similarity_transform_preserves_every_member_font_size_and_gap_ratio(scale):
    original = source()
    cluster = infer_hierarchy(original).cluster("message:0")
    before = cluster.elements
    placed = transform_cluster(cluster, scale, 33, 47)
    frame = union(placed)
    assert frame.x == pytest.approx(33)
    assert frame.y == pytest.approx(47)
    assert frame.width == pytest.approx(cluster.frame_box.width * scale)
    assert frame.height == pytest.approx(cluster.frame_box.height * scale)
    mutable = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}
    for source_element, output in zip(before, placed, strict=True):
        assert output["id"] == source_element["id"]
        for key in ("width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in source_element:
                assert output[key] == pytest.approx(source_element[key] * scale)
        assert {key: value for key, value in output.items() if key not in mutable} == {
            key: value for key, value in source_element.items() if key not in mutable
        }
    for first, second in zip(range(len(before) - 1), range(1, len(before)), strict=True):
        for axis in ("x", "y"):
            assert placed[second][axis] - placed[first][axis] == pytest.approx(
                (before[second][axis] - before[first][axis]) * scale
            )


def test_rotated_explicit_group_scales_stroke_radius_and_offsets_together():
    original = source("level3_advanced")
    for element in original["elements"]:
        if element.get("group_id") == "ribbon":
            element["corner_radius"] = 8
            element["stroke_width"] = 2
    group = next(
        node
        for node in infer_hierarchy(original).clusters
        if node.kind == "explicit" and "ribbon_bg" in node.element_ids
    )
    placed = transform_cluster(group, 1.7, 100, 200)
    frame = union(placed)
    assert (frame.x, frame.y) == pytest.approx((100, 200))
    assert all(element["rotation"] == -12 for element in placed)
    assert all(element["corner_radius"] == pytest.approx(13.6) for element in placed)
    assert all(element["stroke_width"] == pytest.approx(3.4) for element in placed)


def test_cluster_snapshot_is_frozen_and_returned_elements_cannot_drift():
    original = source()
    hierarchy = infer_hierarchy(original)
    cluster = hierarchy.cluster("message:0")
    snapshot = cluster.elements
    with pytest.raises(FrozenInstanceError):
        cluster.confidence = 1
    original["elements"].clear()
    one = cluster.elements
    one[0]["x"] = -123
    assert cluster.elements == snapshot
    placed = transform_cluster(cluster, 1, cluster.frame_box.x, cluster.frame_box.y)
    assert placed == snapshot
    assert len(hierarchy.source_digest) == 64


@pytest.mark.parametrize("alignment", [("start", "end"), ("center", "center"), ("end", "start")])
def test_fit_contains_full_cluster_without_member_repairs(alignment):
    cluster = infer_hierarchy(source()).cluster("brand:0")
    target = Box(100, 200, 700, 330)
    placement = fit_cluster(cluster, target, alignment=alignment)
    assert placement is not None
    frame = union(placement.elements)
    assert frame.x >= target.x - 1e-6 and frame.y >= target.y - 1e-6
    assert frame.x + frame.width <= target.x + target.width + 1e-6
    assert frame.y + frame.height <= target.y + target.height + 1e-6
    ratios = [
        placed["width"] / original["width"]
        for original, placed in zip(cluster.elements, placement.elements, strict=True)
    ]
    assert ratios == pytest.approx([placement.scale] * len(ratios))
    changed = placement.elements
    changed.clear()
    assert placement.elements


def test_visible_anchor_keeps_source_visible_geometry_and_explicit_padding():
    cluster = infer_hierarchy(source()).cluster("brand:0")
    placed = transform_cluster(cluster, 0.8, 100, 120, anchor="visible")
    for original, output in zip(cluster.elements, placed, strict=True):
        assert output["x"] == pytest.approx(100 + (original["x"] - cluster.visible_box.x) * 0.8)
        assert output["y"] == pytest.approx(120 + (original["y"] - cluster.visible_box.y) * 0.8)
    no_upscale = fit_cluster(cluster, Box(0, 0, 2000, 2000), allow_upscale=False)
    assert no_upscale.scale == 1


def test_empty_visible_bounds_decline_visible_fit():
    original = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {"id": "a", "type": "image", "x": 10, "y": 10, "width": 20, "height": 20, "opacity": 0}
        ],
    }
    cluster = infer_hierarchy(original).roots[0]
    assert fit_cluster(cluster, Box(0, 0, 100, 100), anchor="visible") is None
    assert fit_cluster(cluster, Box(0, 0, 0, 10)) is None
    with pytest.raises(ValueError, match="empty visible"):
        transform_cluster(cluster, 1, 0, 0, anchor="visible")


def test_invalid_placement_and_foreign_graph_are_rejected():
    original = source()
    cluster = infer_hierarchy(original).roots[0]
    for scale in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive finite"):
            transform_cluster(cluster, scale, 0, 0)
    with pytest.raises(ValueError, match="anchor"):
        transform_cluster(cluster, 1, 0, 0, anchor="canvas")
    with pytest.raises(ValueError, match="alignment"):
        fit_cluster(cluster, Box(0, 0, 10, 10), alignment=("left", "top"))
    with pytest.raises(ValueError, match="finite"):
        fit_cluster(cluster, Box(0, 0, float("nan"), 10))
    with pytest.raises(ValueError, match="original source"):
        infer_hierarchy(original, build_relationship_graph(source("level3_advanced")))
    with pytest.raises(KeyError):
        infer_hierarchy(original).cluster("missing")
