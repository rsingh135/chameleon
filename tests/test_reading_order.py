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
