"""Source-evidenced sibling rows move complete units without changing scale."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from engine.eval import check_contract, score
from engine.eval.proportions import analyze_proportions
from engine.layout.appearance import unit_visible_box
from engine.layout.grouped import generate_grouped_states
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.layout.sibling_rows import aligned_sibling_rows
from tools.composition_cases import generate_cases


@pytest.fixture(scope="module")
def paired():
    task = generate_cases()[0]
    source = task["source"]
    target = {"width": 1600, "height": 400}
    graph = build_relationship_graph(source)
    states = generate_grouped_states(source, target, graph)
    options = [
        (state, aligned_sibling_rows(source, target, graph, state.design, state.cluster_axes))
        for state in states
        if "aligned" not in state.name
    ]
    state, aligned = next((state, aligned) for state, aligned in options if aligned is not None)
    return task, source, target, graph, state, aligned


def test_source_heading_product_peer_and_action_rows_are_restored(paired):
    task, source, target, graph, state, aligned = paired
    by_id = {element["id"]: element for element in aligned["elements"]}
    for field, anchor in (
        ("headline_ids", "top"),
        ("body_ids", "top"),
        ("product_ids", "bottom"),
        ("action_ids", "bottom"),
    ):
        boxes = [
            unit_visible_box([by_id[eid] for eid in campaign[field]])
            for campaign in task["expected_campaigns"]
        ]
        values = [box.y if anchor == "top" else box.y + box.height for box in boxes]
        assert max(values) - min(values) < 1e-6, (field, values)
    peer_tops = [
        unit_visible_box([by_id[eid] for group in campaign["badge_groups"] for eid in group]).y
        for campaign in task["expected_campaigns"]
    ]
    assert max(peer_tops) - min(peer_tops) < 1e-6
    assert check_contract(source, aligned, target) == []
    assert analyze_relationships(graph, aligned, dict(state.cluster_axes))["warnings"] == []
    assert score(source, aligned)["metrics"]["raw_collisions"] == 0


def test_alignment_is_translation_only_for_every_atomic_unit(paired):
    _, source, target, graph, state, aligned = paired
    candidate = state.design
    before = {element["id"]: element for element in candidate["elements"]}
    after = {element["id"]: element for element in aligned["elements"]}
    for unit in graph.units:
        offsets = []
        for eid in unit.element_ids:
            first, second = before[eid], after[eid]
            assert {key: value for key, value in first.items() if key not in {"x", "y"}} == {
                key: value for key, value in second.items() if key not in {"x", "y"}
            }
            assert second["x"] == first["x"]
            offsets.append(second["y"] - first["y"])
        assert max(offsets) - min(offsets) < 1e-6
    aligned_metrics = analyze_proportions(source, aligned, graph)["metrics"]
    previous_metrics = analyze_proportions(source, candidate, graph)["metrics"]
    for field in (
        "rigid_group_error",
        "campaign_scale_rms_log",
        "sibling_product_scale_rms_log",
        "sibling_product_scale_ratio",
    ):
        assert aligned_metrics[field] == previous_metrics[field]
    assert aligned["canvas"] == target


def test_inputs_and_repeated_results_remain_independent(paired):
    _, source, target, graph, state, _ = paired
    candidate, axes = state.design, dict(state.cluster_axes)
    snapshots = deepcopy(source), deepcopy(candidate), deepcopy(axes), asdict(graph)
    first = aligned_sibling_rows(source, target, graph, candidate, axes)
    second = aligned_sibling_rows(source, target, graph, candidate, axes)
    assert first == second and first is not second
    first["elements"][0]["x"] = -999
    assert first != second
    assert (source, candidate, axes, asdict(graph)) == snapshots


def test_global_stack_is_not_mistaken_for_a_horizontal_comparison_row(paired):
    _, source, target, graph, _, _ = paired
    stacked = [
        state
        for state in generate_grouped_states(source, target, graph)
        if state.name.startswith("grouped-1-")
    ]
    assert stacked
    assert all(
        aligned_sibling_rows(source, target, graph, state.design, state.cluster_axes) is None
        for state in stacked
    )


@pytest.mark.parametrize(
    "changed", ["source-row", "target-portrait", "local-axis", "canvas-bounds"]
)
def test_unsupported_or_infeasible_rows_decline_without_repair(paired, changed):
    task, source, target, graph, state, _ = paired
    source, target, candidate, axes = (
        deepcopy(source),
        dict(target),
        state.design,
        dict(state.cluster_axes),
    )
    if changed == "source-row":
        ids = set(task["expected_campaigns"][1]["action_ids"])
        for element in source["elements"]:
            if element["id"] in ids:
                element["y"] += 80
        graph = build_relationship_graph(source)
    elif changed == "target-portrait":
        target = {"width": 1080, "height": 1920}
        candidate["canvas"] = target
    elif changed == "local-axis":
        for campaign in (cluster for cluster in graph.clusters if cluster.kind == "campaign"):
            axes[campaign.id] = "vertical"
    else:
        for element in candidate["elements"]:
            if element["id"] in task["expected_campaigns"][0]["product_ids"]:
                element["x"] = -1000
    before = deepcopy(candidate)
    assert aligned_sibling_rows(source, target, graph, candidate, axes) is None
    assert candidate == before


def test_originally_vertical_sources_and_narrow_detached_rows_are_not_forced():
    for case, target in ((1, {"width": 1600, "height": 400}), (0, {"width": 913, "height": 517})):
        source = generate_cases()[case]["source"]
        graph = build_relationship_graph(source)
        states = generate_grouped_states(source, target, graph)
        assert states
        assert all(
            aligned_sibling_rows(source, target, graph, state.design, state.cluster_axes) is None
            for state in states
        )


def test_unequal_source_product_bottoms_do_not_gain_a_false_shared_anchor():
    task = generate_cases()[0]
    source = deepcopy(task["source"])
    smaller = task["expected_campaigns"][1]["product_ids"][0]
    for element in source["elements"]:
        if element["id"] == smaller:
            element["x"] += element["width"] * 0.175
            element["y"] += element["height"] * 0.175
            element["width"] *= 0.65
            element["height"] *= 0.65
    target, graph = {"width": 1600, "height": 400}, build_relationship_graph(source)
    results = [
        aligned_sibling_rows(source, target, graph, state.design, state.cluster_axes)
        for state in generate_grouped_states(source, target, graph)
    ]
    supported = [result for result in results if result is not None]
    assert supported
    for result in supported:
        by_id = {element["id"]: element for element in result["elements"]}
        products = [
            unit_visible_box([by_id[eid] for eid in expected["product_ids"]])
            for expected in task["expected_campaigns"]
        ]
        assert abs(products[0].y + products[0].height - products[1].y - products[1].height) > 1
        first, second = (
            by_id[expected["product_ids"][0]] for expected in task["expected_campaigns"]
        )
        assert second["width"] / first["width"] == pytest.approx(0.65)


def test_opaque_ids_and_reversed_layer_creation_order_do_not_change_alignment(paired):
    _, source, target, graph, state, _ = paired
    mapping = {element["id"]: f"opaque-{i * 7919}" for i, element in enumerate(source["elements"])}
    renamed, candidate = deepcopy(source), state.design
    for design in (renamed, candidate):
        for element in design["elements"]:
            element["id"] = mapping[element["id"]]
        design["elements"].reverse()
    renamed_graph = build_relationship_graph(renamed)
    # Cluster names follow source unit indices, so declare flows on the newly
    # inferred graph rather than carrying stale original graph IDs across.
    axes = {
        cluster.id: "horizontal"
        if cluster.kind == "campaign"
        else "wrap"
        if cluster.kind == "badge_row"
        else "vertical"
        for cluster in renamed_graph.clusters
    }
    result = aligned_sibling_rows(renamed, target, renamed_graph, candidate, axes)
    assert result is not None
    assert [element["id"] for element in result["elements"]] == [
        element["id"] for element in candidate["elements"]
    ]
    assert analyze_relationships(renamed_graph, result, axes)["warnings"] == []
