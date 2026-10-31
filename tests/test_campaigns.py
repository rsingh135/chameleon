"""Independent source offers must retain their own product, message and action."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from engine.eval import check_contract, score
from engine.layout.appearance import unit_visible_box
from engine.layout.campaigns import MAX_STATES, generate_campaign_states
from engine.layout.geometry import bounds, intersection, union
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from tools.composition_cases import generate_cases


@pytest.fixture(params=[0, 1], ids=["source-columns", "source-stacked"])
def authored(request):
    return generate_cases()[request.param]


@pytest.mark.parametrize(
    "target",
    [
        {"width": 1600, "height": 400},
        {"width": 913, "height": 517},
        {"width": 1080, "height": 1920},
    ],
)
def test_real_asset_offers_have_complete_readable_owned_states(authored, target):
    source = authored["source"]
    graph = build_relationship_graph(source)
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    assert len(campaigns) == 2 and not graph.ambiguities
    states = generate_campaign_states(source, target, graph)
    assert states and len(states) <= MAX_STATES
    assert any(
        min(element.get("font_size", 1000) for element in state.design["elements"]) >= 10
        for state in states
    )
    source_by_id = {element["id"]: element for element in source["elements"]}
    for state in states:
        design = state.design
        assert check_contract(source, design, target) == []
        assert tuple(element["id"] for element in design["elements"]) == graph.source_ids
        by_id = {element["id"]: element for element in design["elements"]}
        for element in design["elements"]:
            original = source_by_id[element["id"]]
            mutable = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}
            assert {key: value for key, value in original.items() if key not in mutable} == {
                key: value for key, value in element.items() if key not in mutable
            }
            box = bounds(element)
            assert box.x >= -1e-6 and box.y >= -1e-6
            assert box.x + box.width <= target["width"] + 1e-6
            assert box.y + box.height <= target["height"] + 1e-6
        blocks = [union([by_id[eid] for eid in campaign.element_ids]) for campaign in campaigns]
        assert intersection(*blocks) == pytest.approx(0)
        relations = analyze_relationships(graph, design, dict(state.cluster_axes))
        assert not [
            warning
            for warning in relations["warnings"]
            if warning["code"]
            in {"campaign_ownership", "reading_order", "explicit_group", "peer_order"}
        ], relations["warnings"]
        assert score(source, design)["metrics"]["raw_collisions"] == 0


def test_long_copy_keeps_global_typographic_scale_and_its_own_action(authored):
    source = authored["source"]
    graph = build_relationship_graph(source)
    states = generate_campaign_states(source, {"width": 1600, "height": 400}, graph)
    assert states
    assert {state.topology.split("/")[0] for state in states} >= {
        "campaigns-columns",
        "campaigns-stacked",
    }
    for state in states:
        by_id = {element["id"]: element for element in state.design["elements"]}
        headline_scales = [
            by_id[expected["headline_ids"][0]]["font_size"] / 40
            for expected in authored["expected_campaigns"]
        ]
        assert headline_scales[0] == pytest.approx(headline_scales[1])
        for expected in authored["expected_campaigns"]:
            action = [by_id[eid] for eid in expected["action_ids"]]
            assert action[0]["group_id"] == action[1]["group_id"]
            assert action[1]["content"] in {"Discover the blend", "Bring everyone together"}
            badges = [
                unit_visible_box([by_id[eid] for eid in group])
                for group in expected["badge_groups"]
            ]
            body = unit_visible_box([by_id[eid] for eid in expected["body_ids"]])
            assert body.y + body.height <= min(box.y for box in badges) + 1e-6
            assert max(box.y + box.height for box in badges) <= unit_visible_box(action).y + 1e-6


def test_three_campaigns_can_wrap_in_order_without_crossing_ownership():
    task = generate_cases()[0]
    source = task["source"]
    source["canvas"]["width"] = 2400
    source["elements"][0]["width"] = 2400
    third = deepcopy(
        [
            element
            for element in source["elements"]
            if element["id"] in task["expected_campaigns"][1]["element_ids"]
        ]
    )
    for element in third:
        element["id"] += "-next"
        element["x"] += 800
        if element.get("group_id"):
            element["group_id"] += "-next"
    source["elements"].extend(third)
    graph = build_relationship_graph(source)
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    assert len(campaigns) == 3
    states = generate_campaign_states(source, {"width": 1600, "height": 1100}, graph)
    wrapped = [state for state in states if state.topology.startswith("campaigns-wrapped/")]
    assert wrapped
    for state in wrapped:
        design = state.design
        by_id = {element["id"]: element for element in design["elements"]}
        blocks = [union([by_id[eid] for eid in campaign.element_ids]) for campaign in campaigns]
        assert blocks[0].x < blocks[1].x
        assert blocks[2].y > max(blocks[0].y, blocks[1].y)
        assert blocks[2].x + blocks[2].width / 2 == pytest.approx(800)
        assert check_contract(source, design, design["canvas"]) == []
        report = analyze_relationships(graph, design, dict(state.cluster_axes))
        assert not [row for row in report["warnings"] if row["code"] == "campaign_ownership"]


def test_states_are_source_bound_repeatable_and_return_copies(authored, monkeypatch):
    import engine.layout.relationships as inference

    source = authored["source"]
    before = deepcopy(source)
    graph = build_relationship_graph(source)
    graph_before = asdict(graph)
    monkeypatch.setattr(
        inference, "build_relationship_graph", lambda *_: pytest.fail("Re-inferred source graph")
    )
    target = {"width": 1600, "height": 400}
    states = generate_campaign_states(source, target, graph)
    assert states == generate_campaign_states(source, target, graph)
    assert states and source == before and asdict(graph) == graph_before
    changed = states[0].design
    changed["elements"][0]["x"] = 99999
    assert states[0].design["elements"][0]["x"] != 99999
    with pytest.raises(FrozenInstanceError):
        states[0].topology = "other"


def test_unsupported_or_overlapping_ownership_declines(authored):
    source = authored["source"]
    graph = build_relationship_graph(source)
    campaign = next(cluster for cluster in graph.clusters if cluster.kind == "campaign")
    duplicate = replace(campaign, id="extra-owner")
    overlapping = replace(graph, clusters=graph.clusters + (duplicate,))
    assert generate_campaign_states(source, {"width": 1600, "height": 400}, overlapping) == ()
    other_source = deepcopy(source)
    other_source["canvas"]["height"] += 1
    assert generate_campaign_states(other_source, {"width": 1600, "height": 400}, graph) == ()


def test_unowned_foreground_is_not_mixed_into_a_global_role_bucket(authored):
    source = authored["source"]
    source["elements"].append(
        {
            "id": "unassigned",
            "type": "shape",
            "shape_type": "rectangle",
            "fill_color": "#ffffff",
            "x": 0,
            "y": 0,
            "width": 50,
            "height": 50,
            "z_index": 6,
        }
    )
    graph = build_relationship_graph(source)
    assert generate_campaign_states(source, {"width": 1600, "height": 400}, graph) == ()


def test_source_comparison_row_keeps_product_heading_and_action_anchors():
    target = {"width": 1600, "height": 400}
    task = generate_cases()[0]
    source = task["source"]
    graph = build_relationship_graph(source)
    states = generate_campaign_states(source, target, graph)
    aligned = [state for state in states if state.topology == "campaigns-columns/aligned"]
    assert len(aligned) == 1 and len(states) <= MAX_STATES
    state = aligned[0]
    by_id = {element["id"]: element for element in state.design["elements"]}
    for field, anchor in (
        ("product_ids", "bottom"),
        ("headline_ids", "top"),
        ("body_ids", "top"),
        ("action_ids", "bottom"),
    ):
        boxes = [
            unit_visible_box([by_id[eid] for eid in expected[field]])
            for expected in task["expected_campaigns"]
        ]
        values = [box.y if anchor == "top" else box.y + box.height for box in boxes]
        assert max(values) - min(values) < 1e-6, (field, values)
    assert check_contract(source, state.design, target) == []
    report = analyze_relationships(graph, state.design, dict(state.cluster_axes))
    assert report["warnings"] == []
    assert score(source, state.design)["metrics"]["raw_collisions"] == 0
    assert min(element.get("font_size", 1000) for element in state.design["elements"]) >= 10


def test_narrow_comparison_cells_do_not_force_a_detached_common_action_row():
    source = generate_cases()[0]["source"]
    graph = build_relationship_graph(source)
    states = generate_campaign_states(source, {"width": 913, "height": 517}, graph)
    assert states
    assert not any(state.topology == "campaigns-columns/aligned" for state in states)


@pytest.mark.parametrize(
    "case,target",
    [(1, {"width": 1600, "height": 400}), (0, {"width": 1080, "height": 1920})],
)
def test_vertical_sources_and_portrait_reflow_are_not_forced_into_shared_row(case, target):
    source = generate_cases()[case]["source"]
    graph = build_relationship_graph(source)
    states = generate_campaign_states(source, target, graph)
    assert states
    assert not any(state.topology == "campaigns-columns/aligned" for state in states)


def test_intentionally_different_source_action_rows_do_not_gain_comparison_state():
    task = generate_cases()[0]
    source = task["source"]
    shifted = set(task["expected_campaigns"][1]["action_ids"])
    for element in source["elements"]:
        if element["id"] in shifted:
            element["y"] += 80
    graph = build_relationship_graph(source)
    states = generate_campaign_states(source, {"width": 1600, "height": 400}, graph)
    assert states
    assert not any(state.topology == "campaigns-columns/aligned" for state in states)
