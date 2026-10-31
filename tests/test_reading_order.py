"""Geometry-only source reading order, including actual Grand regressions."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from engine.layout.reading_order import analyze_reading_order, infer_reading_order
from engine.layout.source_guides import generate_source_guides
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]


def text(eid, x, y, width=180, size=16, content="Identical copy"):
    return {
        "id": eid,
        "type": "text",
        "x": x,
        "y": y,
        "width": width,
        "height": size * 1.2,
        "font_family": "Solstice Sans",
        "font_size": size,
        "line_height": 1.2,
        "content": content,
        "color": "#111111",
    }


def design(elements, width=700, height=700):
    return {"canvas": {"width": width, "height": height}, "elements": elements}


def container_design():
    """Two source-proven opaque cards, with deliberately uneven text rows."""
    elements = [
        {
            "id": eid,
            "type": "shape",
            "shape_type": "rectangle",
            "x": x,
            "y": 150,
            "width": 400,
            "height": 700,
            "fill_color": "#ffffff",
            "z_index": 0,
        }
        for eid, x in (("frame-a", 50), ("frame-b", 550))
    ]
    elements.append(text("header", 100, 20, width=800))
    for prefix, x, rows in (("a", 75, (170, 250, 400, 600)), ("b", 575, (170, 220, 340, 760))):
        for index, y in enumerate(rows):
            leaf = text(f"{prefix}{index}", x, y, width=350, size=22 if index == 0 else 16)
            leaf["z_index"] = 1
            elements.append(leaf)
    elements.append(text("footer", 100, 930, width=800))
    return design(elements, 1000, 1000)


def move_owned_card(candidate, prefix, dx=0, dy=0):
    for element in candidate["elements"]:
        if element["id"] == f"frame-{prefix}" or element["id"] in {
            f"{prefix}{i}" for i in range(4)
        }:
            element["x"] += dx
            element["y"] += dy


def test_source_containers_keep_heading_and_body_contiguous_across_reflow():
    source = container_design()
    before = deepcopy(source)
    profile = infer_reading_order(source)
    assert profile.containers is not None
    assert len(profile.containers.containers) == 2
    assert profile.sequence == ("header", "a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3", "footer")
    candidate = deepcopy(source)
    candidate["canvas"]["height"] = 2000
    move_owned_card(candidate, "b", dx=-500, dy=850)
    candidate["elements"][-1]["y"] = 1930
    # Frozen source evidence is sufficient when the graph does not pass source.
    result = analyze_reading_order({}, candidate, profile)
    assert result["container_order_active"]
    assert result["candidate_sequence"] == list(profile.sequence)
    assert result["high_confidence_inversions"] == 0
    assert result["comparison_coverage"] == 1
    assert result["container_detachments"] == []
    assert source == before


def test_reversed_whole_containers_and_internal_text_reversals_remain_errors():
    source = container_design()
    profile = infer_reading_order(source)
    reversed_cards = deepcopy(source)
    move_owned_card(reversed_cards, "a", dx=500)
    move_owned_card(reversed_cards, "b", dx=-500)
    assert analyze_reading_order({}, reversed_cards, profile)["high_confidence_inversions"] > 0
    reversed_children = deepcopy(source)
    by_id = {element["id"]: element for element in reversed_children["elements"]}
    by_id["a0"]["y"], by_id["a2"]["y"] = by_id["a2"]["y"], by_id["a0"]["y"]
    result = analyze_reading_order({}, reversed_children, profile)
    assert any(
        pair["before"] == "a0" and pair["after"] == "a2" for pair in result["inversion_pairs"]
    )


def test_candidate_cannot_reassign_text_to_another_source_container():
    source = container_design()
    profile = infer_reading_order(source)
    candidate = deepcopy(source)
    next(element for element in candidate["elements"] if element["id"] == "a0")["x"] = 575
    result = analyze_reading_order({}, candidate, profile)
    assert result["container_order_active"]
    assert "source_container_detachment" in result["warnings"]
    assert any(
        item["frame_id"] == "frame-a" and item["element_id"] == "a0"
        for item in result["container_detachments"]
    )
    assert "a0" in result["source_containers"][0]["member_ids"]
    assert "a0" not in result["source_containers"][1]["member_ids"]


def test_candidate_must_retain_source_container_frames():
    source = container_design()
    profile = infer_reading_order(source)
    candidate = deepcopy(source)
    candidate["elements"] = [
        element for element in candidate["elements"] if element["id"] != "frame-a"
    ]
    with pytest.raises(ValueError, match="container frame"):
        analyze_reading_order({}, candidate, profile)
    candidate = deepcopy(source)
    candidate["elements"].append(deepcopy(candidate["elements"][0]))
    with pytest.raises(ValueError, match="exactly once"):
        analyze_reading_order({}, candidate, profile)


def test_unproven_card_shapes_do_not_override_plain_spatial_evidence():
    source = container_design()
    source["elements"] = [
        element for element in source["elements"] if element["id"] not in {"a2", "a3", "b2", "b3"}
    ]
    assert infer_reading_order(source).containers is None


def test_identity_and_source_are_immutable_and_profile_is_frozen():
    source = design([text("opaque-a", 20, 20), text("opaque-b", 20, 90), text("opaque-c", 20, 160)])
    before = deepcopy(source)
    profile = infer_reading_order(source)
    result = analyze_reading_order(source, source, profile)
    assert result["active"] and result["score"] == 1
    assert result["high_confidence_inversions"] == 0
    assert result["source_sequence"] == result["candidate_sequence"]
    assert source == before
    with pytest.raises(FrozenInstanceError):
        profile.sequence = ()
    with pytest.raises(FrozenInstanceError):
        profile.leaves[0].box.x = 99


def test_vertical_sequence_can_flow_into_proper_columns():
    source = design([text(str(i), 20, 20 + 70 * i) for i in range(6)])
    candidate = deepcopy(source)
    for index, element in enumerate(candidate["elements"]):
        element.update(x=20 if index < 3 else 350, y=20 + 70 * (index % 3))
    result = analyze_reading_order(source, candidate)
    assert result["candidate_sequence"] == [str(i) for i in range(6)]
    assert result["high_confidence_inversions"] == 0


def test_source_columns_use_column_major_order_instead_of_global_y():
    source = design(
        [
            text("a", 20, 20),
            text("b", 20, 90),
            text("c", 20, 160),
            text("d", 350, 25),
            text("e", 350, 95),
            text("f", 350, 165),
        ]
    )
    profile = infer_reading_order(source)
    assert profile.sequence == ("a", "b", "c", "d", "e", "f")
    candidate = deepcopy(source)
    for index, element in enumerate(candidate["elements"]):
        element.update(x=20, y=20 + 70 * index)
    assert analyze_reading_order(source, candidate, profile)["high_confidence_inversions"] == 0


def test_within_source_row_order_survives_vertical_flow_and_detects_reversal():
    source = design([text("a", 20, 20), text("b", 250, 20), text("c", 480, 20)])
    profile = infer_reading_order(source)
    assert ("a", "b", "c") in profile.row_groups
    candidate = deepcopy(source)
    for index, element in enumerate(candidate["elements"]):
        element.update(x=20, y=20 + 70 * index)
    assert analyze_reading_order(source, candidate, profile)["high_confidence_inversions"] == 0
    candidate["elements"][0]["y"], candidate["elements"][1]["y"] = 90, 20
    result = analyze_reading_order(source, candidate, profile)
    assert any(pair["before"] == "a" and pair["after"] == "b" for pair in result["inversion_pairs"])


def test_ids_copy_and_input_array_order_do_not_determine_reading_sequence():
    source = design([text("z", 20, 20), text("m", 20, 90), text("a", 20, 160)])
    reordered = deepcopy(source)
    reordered["elements"].reverse()
    for element in reordered["elements"]:
        element["id"] = {"z": "opaque-8", "m": "opaque-1", "a": "opaque-4"}[element["id"]]
        element["content"] = "Different words"
    assert infer_reading_order(source).sequence == ("z", "m", "a")
    assert infer_reading_order(reordered).sequence == ("opaque-8", "opaque-1", "opaque-4")


def test_translation_and_uniform_scale_keep_source_sequence_and_inversions():
    source = design([text("a", 20, 20), text("b", 20, 90), text("c", 350, 20), text("d", 350, 90)])
    transformed = deepcopy(source)
    transformed["canvas"] = {"width": 1500, "height": 1500}
    for element in transformed["elements"]:
        for field in ("x", "y", "width", "height", "font_size"):
            element[field] *= 2
        element["x"] += 30
        element["y"] += 40
    assert infer_reading_order(source).sequence == infer_reading_order(transformed).sequence
    assert analyze_reading_order(source, transformed)["high_confidence_inversions"] == 0


def test_cross_diagonal_and_overlapping_text_are_reported_as_ambiguous():
    source = design([text("left-low", 20, 160), text("right-high", 350, 20)])
    result = analyze_reading_order(source, source)
    assert result["ambiguity_count"] > 0
    assert result["high_confidence_inversions"] == 0
    overlap = design([text("one", 20, 20), text("two", 20, 20)])
    assert analyze_reading_order(overlap, overlap)["ambiguity_count"] > 0


def test_rotated_and_outside_text_never_become_confident_reading_evidence():
    source = design(
        [text("one", 20, 20), text("two", 20, 90), text("rotate", 20, 160), text("out", -200, 20)]
    )
    source["elements"][2]["rotation"] = 45
    result = analyze_reading_order(source, source)
    assert result["source_sequence"] == ["one", "two"]
    assert {item["element_id"] for item in result["excluded_elements"]} == {"rotate", "out"}
    assert "reading_order_uncertain" in result["warnings"]


def test_budget_declines_before_unbounded_font_measurement(monkeypatch):
    from engine.layout import reading_order

    source = design([text(str(index), 20, index * 30) for index in range(257)], height=8000)

    def must_not_measure(_element):
        raise AssertionError("Budget check must precede font measurement")

    monkeypatch.setattr(reading_order, "visible_box", must_not_measure)
    result = analyze_reading_order(source, source)
    assert not result["active"]
    assert result["comparison_coverage"] == 0
    assert result["high_confidence_inversions"] == 0
    assert any("budget" in item["reason"] for item in result["excluded_elements"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "replacement"])
def test_same_text_leaf_membership_is_required(change):
    source = design([text("one", 20, 20), text("two", 20, 90)])
    candidate = deepcopy(source)
    if change == "missing":
        candidate["elements"].pop()
    elif change == "duplicate":
        candidate["elements"].append(deepcopy(candidate["elements"][0]))
    else:
        candidate["elements"][0]["id"] = "another-leaf"
    with pytest.raises(ValueError, match="membership|unique"):
        analyze_reading_order(source, candidate)


def test_profile_restores_exact_document_fonts_and_actual_grand_negative():
    folder = ROOT / "fixtures/open_posters/prepared/doory_grand-opening-flyer"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    # Captured geometry from the actual first square output. Styles and leaf IDs
    # come from the retained original, not from semantic code/name matching.
    captured = [
        (48.6, 328.277311, 982.8, 24.982782, 18.422),
        (48.6, 48.6, 982.8, 260.237311, 230.285),
        (48.6, 372.700093, 982.8, 64.955233, 47.899),
        (48.6, 457.095326, 982.8, 54.962120, 40.529),
        (48.6, 911.705794, 982.8, 26.544206, 15.659),
        (48.6, 687.796520, 982.8, 53.088411, 15.659),
        (48.6, 531.497446, 982.8, 21.235365, 15.659),
        (48.6, 572.172810, 982.8, 96.183710, 77.375),
        (48.6, 832.853343, 982.8, 21.235365, 15.659),
        (48.6, 760.324932, 982.8, 53.088411, 15.659),
        (48.6, 873.528708, 982.8, 18.737086, 13.816),
    ]
    candidate = deepcopy(source)
    candidate["canvas"] = {"width": 1080, "height": 1080}
    texts = [element for element in candidate["elements"] if element["type"] == "text"]
    assert len(texts) == len(captured)
    for element, frame in zip(texts, captured, strict=True):
        element.update(zip(("x", "y", "width", "height", "font_size"), frame, strict=True))
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        profile = infer_reading_order(source)
    # Analysis after the resource context ends must still use those exact fonts.
    result = analyze_reading_order(source, candidate, profile)
    assert result["high_confidence_inversions"] >= 2
    assert any(
        pair["before"] == texts[0]["id"] and pair["after"] == texts[1]["id"]
        for pair in result["inversion_pairs"]
    )
    assert any(
        pair["before"] == texts[4]["id"] and pair["after"] == texts[5]["id"]
        for pair in result["inversion_pairs"]
    )


def test_actual_grand_good_column_proposals_retain_original_sequence():
    folder = ROOT / "fixtures/open_posters/prepared/doory_grand-opening-flyer"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        profile = infer_reading_order(source)
        states = [
            state
            for state in generate_source_guides(source, {"width": 1600, "height": 900})
            if state.name in {"source-guides/columns-4", "source-guides/columns-5"}
        ]
        assert states
        for state in states:
            result = analyze_reading_order(source, state.design, profile)
            assert result["high_confidence_inversions"] == 0, state.name
            assert result["candidate_sequence"] == list(profile.sequence), state.name


@pytest.mark.parametrize("case, count", [("opensuse_cheatsheet", 7), ("opensuse_trifold_front", 3)])
def test_actual_repeated_panels_do_not_interleave_separate_card_headings_and_bodies(case, count):
    folder = ROOT / "fixtures/open_posters/prepared" / case
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        profile = infer_reading_order(source)
        assert profile.containers is not None
        assert len(profile.containers.containers) == count
        states = [
            state
            for state in generate_source_guides(source, {"width": 1600, "height": 900})
            if state.topology == "source-panel-columns"
        ]
        assert states
    # Exact source font paths are frozen in the profile and restored here.
    for container in profile.containers.containers:
        positions = [
            index for index, eid in enumerate(profile.sequence) if eid in container.member_ids
        ]
        assert positions == list(range(min(positions), max(positions) + 1))
    for state in states:
        result = analyze_reading_order({}, state.design, profile)
        assert result["container_order_active"]
        assert result["high_confidence_inversions"] == 0, state.name
        assert result["container_detachments"] == [], state.name
        # Ambiguous ordering inside a card stays ambiguous. Whole owned cards
        # remain contiguous and retain their source order regardless of that.
        owners = {
            eid: index
            for index, container in enumerate(profile.containers.containers)
            for eid in container.member_ids
        }
        ordered = [owners[eid] for eid in result["candidate_sequence"] if eid in owners]
        blocks = [
            value
            for index, value in enumerate(ordered)
            if index == 0 or ordered[index - 1] != value
        ]
        assert blocks == list(range(count)), state.name
    assert source == before
