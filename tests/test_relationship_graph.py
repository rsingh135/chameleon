"""Source relationships describe meaning that element-preservation misses."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from engine.eval import check_contract
from engine.eval.style import analyze_style
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from tools.composition_cases import generate_cases

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
    first, second = (graph.units[index].box for index in message.unit_indices)
    source_gap = next(edge.source_gap for edge in graph.edges if edge.code == "message_attachment")
    # This test changes flow, not proximity. Preserve the dimensionless gap;
    # an absolute x=200 fell on different sides of the attachment threshold
    # with Windows and Linux font advances.
    candidate["elements"][1].update(
        x=first.x + first.width + source_gap * max(first.width, second.width), y=first.y
    )
    unchanged = analyze_relationships(graph, candidate)
    changed = analyze_relationships(graph, candidate, {message.id: "horizontal"})
    assert "shared_axis" in {warning["code"] for warning in unchanged["warnings"]}
    assert changed["warnings"] == []
    assert changed["metrics"]["message_attachment"] == pytest.approx(1)
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


@pytest.fixture
def full_bleed_scene(tmp_path):
    from PIL import Image

    Image.new("RGBA", (100, 160), "#885533").save(tmp_path / "photo.png")
    fade = Image.new("RGBA", (100, 80))
    fade.putdata([(0, 0, 0, round(255 * y / 79)) for y in range(80) for _ in range(100)])
    fade.save(tmp_path / "fade.png")
    return tmp_path, {
        "canvas": {"width": 1000, "height": 1600},
        "elements": [
            {
                "id": "opaque-a",
                "type": "image",
                "asset_id": "photo.png",
                "source_art": {"kind": "masked-photo"},
                "fit": "fill",
                "x": 0,
                "y": 0,
                "width": 1000,
                "height": 1600,
                "z_index": 0,
            },
            {
                "id": "opaque-b",
                "type": "image",
                "asset_id": "fade.png",
                "fit": "fill",
                "x": 0,
                "y": 800,
                "width": 1000,
                "height": 800,
                "z_index": 1,
            },
            {**_text("opaque-c", 100, 1200, 80), "z_index": 2},
            {**_text("opaque-d", 100, 1320, 30), "z_index": 3},
        ],
    }


def test_source_evidenced_full_bleed_photo_is_support_not_independent_product(full_bleed_scene):
    assets, original = full_bleed_scene
    before = deepcopy(original)
    graph = build_relationship_graph(original, assets)
    unit = next(unit for unit in graph.units if "opaque-a" in unit.element_ids)
    assert unit.role == "image_underlay"
    assert any(cluster.kind == "message" for cluster in graph.clusters)
    assert not any(cluster.kind == "campaign" for cluster in graph.clusters)
    assert graph.overlays[0].underlay_id == "opaque-a"
    assert analyze_relationships(graph, original)["warnings"] == []
    candidate = deepcopy(original)
    candidate["elements"][1]["x"] += 120
    assert "overlay_attachment" in {
        warning["code"] for warning in analyze_relationships(graph, candidate)["warnings"]
    }
    assert original == before


@pytest.mark.parametrize(
    "change",
    [
        {"source_art": {}},
        {"source_art": {"kind": "svg-nontext"}},
        {"width": 900},
        {"opacity": 0.8},
        {"rotation": 5},
        {"fit": "contain", "height": 1800},
    ],
)
def test_unproven_photos_and_vector_products_keep_foreground_role(full_bleed_scene, change):
    assets, original = full_bleed_scene
    original["elements"][0].update(change)
    graph = build_relationship_graph(original, assets)
    unit = next(unit for unit in graph.units if "opaque-a" in unit.element_ids)
    assert unit.role != "image_underlay"


def test_alpha_hole_outside_caption_region_is_not_full_bleed_support(full_bleed_scene):
    from PIL import Image

    assets, original = full_bleed_scene
    with Image.open(assets / "photo.png") as image:
        image.paste((0, 0, 0, 0), (10, 10, 30, 30))
        image.save(assets / "photo.png")
    graph = build_relationship_graph(original, assets)
    assert graph.overlays  # The smaller caption region remains opaque.
    unit = next(unit for unit in graph.units if "opaque-a" in unit.element_ids)
    assert unit.role != "image_underlay"


def test_actual_full_bleed_photo_campaign_declines_with_opaque_ids():
    from engine.resources import using_resources

    directory = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    original = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    for index, element in enumerate(original["elements"]):
        element["id"] = f"unrelated-{index}"
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        graph = build_relationship_graph(original)
        assert len(graph.overlays) == 1
        underlay = graph.overlays[0].underlay_id
        unit = next(unit for unit in graph.units if underlay in unit.element_ids)
        assert unit.role == "image_underlay"
        assert not [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
        assert analyze_relationships(graph, original)["warnings"] == []


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


def _peer_source():
    elements = []
    for index in range(3):
        x = 50 + index * 300
        elements.extend(
            [
                {
                    "id": f"p{index}",
                    "type": "shape",
                    "x": x,
                    "y": 50,
                    "width": 250,
                    "height": 50,
                    "shape_type": "rectangle",
                    "fill_color": "#d06030",
                    "group_id": f"g{index}",
                    "z_index": 1,
                },
                {
                    "id": f"l{index}",
                    "type": "text",
                    "x": x,
                    "y": 64,
                    "width": 250,
                    "height": 24,
                    "font_size": 18,
                    "font_family": "Solstice Sans",
                    "content": "Peer control",
                    "text_align": "center",
                    "color": "#ffffff",
                    "group_id": f"g{index}",
                    "z_index": 2,
                },
            ]
        )
    return {"canvas": {"width": 1000, "height": 500}, "elements": elements}


def _translate_ids(design, ids, dx=0, dy=0):
    for element in design["elements"]:
        if element["id"] in ids:
            element["x"] += dx
            element["y"] += dy


def test_advanced_peer_row_keeps_badges_separate_from_primary_action_and_ribbon():
    original = source("level3_advanced")
    graph = build_relationship_graph(original)
    rows = [cluster for cluster in graph.clusters if cluster.kind == "badge_row"]
    assert len(rows) == 1
    assert rows[0].element_ids == (
        "badge1_bg",
        "badge1_label",
        "badge2_bg",
        "badge2_label",
        "badge3_bg",
        "badge3_label",
    )
    campaign = next(cluster for cluster in graph.clusters if cluster.kind == "campaign")
    assert rows[0].id in campaign.member_cluster_ids
    assert "cta_bg" in campaign.element_ids
    assert "ribbon_bg" not in campaign.element_ids
    assert analyze_relationships(graph, original)["warnings"] == []


def test_peer_rows_allow_explicit_wrapping_without_reordering_members():
    original = _peer_source()
    graph = build_relationship_graph(original)
    row = next(cluster for cluster in graph.clusters if cluster.kind == "badge_row")
    candidate = deepcopy(original)
    _translate_ids(candidate, {"p2", "l2"}, dx=-600, dy=75)
    assert "peer_order" in {
        warning["code"] for warning in analyze_relationships(graph, candidate)["warnings"]
    }
    wrapped = analyze_relationships(graph, candidate, {row.id: "wrap"})
    assert wrapped["warnings"] == []
    assert wrapped["metrics"]["peer_order"] == 1
    _translate_ids(candidate, {"p0", "l0"}, dy=150)
    reordered = analyze_relationships(graph, candidate, {row.id: "wrap"})
    assert "peer_order" in {warning["code"] for warning in reordered["warnings"]}


def test_peer_rows_detect_detachment_and_disproportionate_label_prominence():
    original = _peer_source()
    graph = build_relationship_graph(original)
    candidate = deepcopy(original)
    _translate_ids(candidate, {"p2", "l2"}, dy=350)
    assert "peer_cohesion" in {
        warning["code"] for warning in analyze_relationships(graph, candidate)["warnings"]
    }
    candidate = deepcopy(original)
    next(element for element in candidate["elements"] if element["id"] == "l1")["font_size"] = 30
    assert "peer_prominence" in {
        warning["code"] for warning in analyze_relationships(graph, candidate)["warnings"]
    }


def test_minor_baseline_offsets_do_not_erase_a_peer_row():
    original = _peer_source()
    _translate_ids(original, {"p0", "l0"}, dy=2)
    graph = build_relationship_graph(original)
    rows = [cluster for cluster in graph.clusters if cluster.kind == "badge_row"]
    assert len(rows) == 1 and len(rows[0].unit_indices) == 3
    assert analyze_relationships(graph, original)["warnings"] == []


@pytest.mark.parametrize("case", generate_cases(), ids=lambda case: case["task_id"])
def test_local_campaign_ownership_covers_all_authored_members_without_global_role_buckets(case):
    original = case["source"]
    before = deepcopy(original)
    graph = build_relationship_graph(original)
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    assert len(campaigns) == 2
    assert graph.ambiguities == ()
    assert len(set(campaigns[0].unit_indices) & set(campaigns[1].unit_indices)) == 0
    assert {frozenset(cluster.element_ids) for cluster in campaigns} == {
        frozenset(expected["element_ids"]) for expected in case["expected_campaigns"]
    }
    for expected in case["expected_campaigns"]:
        campaign = next(
            cluster
            for cluster in campaigns
            if set(expected["product_ids"]) <= set(cluster.element_ids)
        )
        assert set(graph.units[campaign.primary_unit].element_ids) == set(expected["product_ids"])
        children = [
            cluster for cluster in graph.clusters if cluster.id in campaign.member_cluster_ids
        ]
        assert {cluster.kind for cluster in children} == {"brand", "message", "badge_row"}
        row = next(cluster for cluster in children if cluster.kind == "badge_row")
        assert set(row.element_ids) == set(sum(expected["badge_groups"], []))
        message = next(cluster for cluster in children if cluster.kind == "message")
        assert set(expected["action_ids"]) <= set(message.element_ids)
    assert original == before
    assert analyze_relationships(graph, original)["score"] == 1
    assert analyze_relationships(graph, original)["warnings"] == []


def test_secondary_product_and_upper_page_action_need_not_have_global_hero_or_footer_roles():
    case = generate_cases()[1]
    graph = build_relationship_graph(case["source"])
    upper_action = set(case["expected_campaigns"][0]["action_ids"])
    lower_product = set(case["expected_campaigns"][1]["product_ids"])
    assert (
        next(unit for unit in graph.units if set(unit.element_ids) == upper_action).role == "badge"
    )
    assert (
        next(unit for unit in graph.units if set(unit.element_ids) == lower_product).role
        == "support"
    )
    assert any(
        upper_action <= set(cluster.element_ids)
        for cluster in graph.clusters
        if cluster.kind == "message"
    )
    assert any(
        lower_product <= set(graph.units[cluster.primary_unit].element_ids)
        for cluster in graph.clusters
        if cluster.kind == "campaign"
    )


def test_prominent_peer_badges_are_not_consumed_as_the_local_action():
    case = generate_cases()[1]
    original = case["source"]
    expected = case["expected_campaigns"][0]
    by_id = {element["id"]: element for element in original["elements"]}
    for plate_id, label_id in expected["badge_groups"]:
        plate, label = by_id[plate_id], by_id[label_id]
        label.update(font_size=22, height=26.4, y=plate["y"] + (plate["height"] - 26.4) / 2)
    graph = build_relationship_graph(original)
    message = next(
        cluster
        for cluster in graph.clusters
        if cluster.kind == "message" and set(expected["headline_ids"]) <= set(cluster.element_ids)
    )
    assert set(expected["action_ids"]) <= set(message.element_ids)
    assert not set(sum(expected["badge_groups"], [])) & set(message.element_ids)
    assert any(
        set(sum(expected["badge_groups"], [])) == set(cluster.element_ids)
        for cluster in graph.clusters
        if cluster.kind == "badge_row"
    )


def test_swapped_messages_lose_product_ownership_even_when_all_copy_stays_readable():
    case = generate_cases()[0]
    original = case["source"]
    graph = build_relationship_graph(original)
    messages = [cluster for cluster in graph.clusters if cluster.kind == "message"]
    candidate = deepcopy(original)
    _translate_ids(candidate, set(messages[0].element_ids), dx=800)
    _translate_ids(candidate, set(messages[1].element_ids), dx=-800)
    assert check_contract(original, candidate, original["canvas"]) == []
    assert analyze_style(original, candidate)["warnings"] == []
    result = analyze_relationships(graph, candidate)
    assert result["metrics"]["campaign_ownership"] == 0
    assert "campaign_ownership" in {warning["code"] for warning in result["warnings"]}


def test_complete_campaigns_can_exchange_positions_without_reassigning_their_messages():
    original = generate_cases()[0]["source"]
    graph = build_relationship_graph(original)
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    candidate = deepcopy(original)
    _translate_ids(candidate, set(campaigns[0].element_ids), dx=800)
    _translate_ids(candidate, set(campaigns[1].element_ids), dx=-800)
    result = analyze_relationships(graph, candidate)
    assert result["warnings"] == []
    assert result["metrics"]["campaign_ownership"] == pytest.approx(1)


def test_ambiguous_equal_product_evidence_does_not_force_a_greedy_message_assignment():
    elements = []
    for index in range(2):
        elements.append(
            {
                "id": f"image{index}",
                "type": "image",
                "x": 350,
                "y": 100,
                "width": 300,
                "height": 300,
                "asset_id": "solstice_bottle.png",
                "fit": "contain",
            }
        )
        heading = _text(f"h{index}", 100 + index * 500, 600, 40)
        body = _text(f"b{index}", 100 + index * 500, 680, 20)
        heading["text_align"] = body["text_align"] = "center"
        elements.extend([heading, body])
    original = {"canvas": {"width": 1000, "height": 1000}, "elements": elements}
    graph = build_relationship_graph(original)
    assert not [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    assert len(graph.ambiguities) == 2
    assert all(
        ambiguity.candidate_cluster_ids and ambiguity.costs and ambiguity.provenance
        for ambiguity in graph.ambiguities
    )
    result = analyze_relationships(graph, original)
    assert result["ambiguities"] and result["warnings"] == []


@pytest.mark.parametrize("case", generate_cases(), ids=lambda case: case["task_id"])
def test_campaigns_and_peer_rows_keep_identity_under_uniform_scaling_and_opaque_names(case):
    original = case["source"]
    graph = build_relationship_graph(original)
    scaled = deepcopy(original)
    for key in scaled["canvas"]:
        scaled["canvas"][key] *= 2
    for element in scaled["elements"]:
        for key in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in element:
                element[key] *= 2
    result = analyze_relationships(graph, scaled)
    assert result["warnings"] == []
    assert result["score"] == pytest.approx(1, abs=1e-5)
    renamed = deepcopy(original)
    groups = {}
    for index, element in enumerate(renamed["elements"]):
        element["id"] = f"arbitrary-{index}"
        if element.get("group_id"):
            groups.setdefault(element["group_id"], f"set-{len(groups)}")
            element["group_id"] = groups[element["group_id"]]
    renamed_graph = build_relationship_graph(renamed)
    assert [
        (cluster.kind, cluster.unit_indices, cluster.primary_unit, cluster.member_cluster_ids)
        for cluster in graph.clusters
    ] == [
        (cluster.kind, cluster.unit_indices, cluster.primary_unit, cluster.member_cluster_ids)
        for cluster in renamed_graph.clusters
    ]
    assert renamed_graph.ambiguities == ()


def test_repeated_brand_section_boundaries_provide_evidence_for_vertical_ownership():
    case = generate_cases()[1]
    original = case["source"]
    assert (
        len(
            [
                cluster
                for cluster in build_relationship_graph(original).clusters
                if cluster.kind == "campaign"
            ]
        )
        == 2
    )
    uncertain = deepcopy(original)
    removed = set(case["expected_campaigns"][1]["logo_ids"])
    uncertain["elements"] = [
        element for element in uncertain["elements"] if element["id"] not in removed
    ]
    graph = build_relationship_graph(uncertain)
    assert graph.ambiguities, (
        "Without the repeated section header, preserve the competing spatial evidence"
    )


@pytest.mark.parametrize("case", generate_cases(), ids=lambda case: case["task_id"])
def test_source_layer_creation_order_does_not_change_campaign_reading_order(case):
    original = case["source"]
    graph = build_relationship_graph(original)
    reordered = deepcopy(original)
    reordered["elements"] = list(reversed(reordered["elements"]))
    second = build_relationship_graph(reordered)
    assert [
        frozenset(cluster.element_ids) for cluster in graph.clusters if cluster.kind == "campaign"
    ] == [
        frozenset(cluster.element_ids) for cluster in second.clusters if cluster.kind == "campaign"
    ]
    assert second.ambiguities == ()


@pytest.mark.parametrize("case", generate_cases(), ids=lambda case: case["task_id"])
def test_complete_peer_row_cannot_move_below_its_own_action(case):
    original = case["source"]
    graph = build_relationship_graph(original)
    row = next(cluster for cluster in graph.clusters if cluster.kind == "badge_row")
    assert any(edge.code == "row_attachment" and edge.cluster_id == row.id for edge in graph.edges)
    candidate = deepcopy(original)
    _translate_ids(candidate, set(row.element_ids), dy=175)
    assert check_contract(original, candidate, original["canvas"]) == []
    assert analyze_style(original, candidate, relationship_graph=graph)["warnings"] == []
    result = analyze_relationships(graph, candidate)
    assert result["metrics"]["peer_order"] == 1
    assert result["metrics"]["peer_cohesion"] == 1
    assert result["metrics"]["row_attachment"] < 1
    assert any(
        warning["code"] == "row_attachment" and warning["cluster_id"] == row.id
        for warning in result["warnings"]
    )


def test_swapping_complete_rows_between_offers_breaks_local_attachment():
    original = generate_cases()[0]["source"]
    graph = build_relationship_graph(original)
    rows = [cluster for cluster in graph.clusters if cluster.kind == "badge_row"]
    candidate = deepcopy(original)
    _translate_ids(candidate, set(rows[0].element_ids), dx=800)
    _translate_ids(candidate, set(rows[1].element_ids), dx=-800)
    assert check_contract(original, candidate, original["canvas"]) == []
    result = analyze_relationships(graph, candidate)
    assert result["metrics"]["peer_order"] == 1
    assert result["metrics"]["row_attachment"] == 0
    assert "row_attachment" in {warning["code"] for warning in result["warnings"]}


def test_peer_wrap_keeps_the_row_between_its_copy_and_action():
    original = generate_cases()[0]["source"]
    graph = build_relationship_graph(original)
    row = next(cluster for cluster in graph.clusters if cluster.kind == "badge_row")
    candidate = deepcopy(original)
    first, second = (graph.units[index] for index in row.unit_indices)
    _translate_ids(candidate, set(first.element_ids), dx=134, dy=-80)
    _translate_ids(candidate, set(second.element_ids), dx=-124, dy=-20)
    result = analyze_relationships(graph, candidate, {row.id: "wrap"})
    assert result["warnings"] == []
    assert result["metrics"]["row_attachment"] == pytest.approx(1)


def test_row_attachment_is_not_invented_when_the_source_row_follows_the_action():
    original = generate_cases()[0]["source"]
    initial = build_relationship_graph(original)
    row = next(cluster for cluster in initial.clusters if cluster.kind == "badge_row")
    _translate_ids(original, set(row.element_ids), dy=200)
    graph = build_relationship_graph(original)
    shifted_row = next(
        cluster for cluster in graph.clusters if set(cluster.element_ids) == set(row.element_ids)
    )
    assert not any(
        edge.code == "row_attachment" and edge.cluster_id == shifted_row.id for edge in graph.edges
    )
    assert analyze_relationships(graph, original)["warnings"] == []
