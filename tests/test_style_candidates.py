"""Source-relative controls and corner accents must survive actual reflow."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import build_context as build_eval_context
from engine.eval import check_contract, score
from engine.layout.candidates import generate_candidates
from engine.layout.geometry import bounds
from engine.resize import resize

ROOT = Path(__file__).resolve().parents[1]


def task(level):
    return json.loads((ROOT / f"input/chameleon_sample_{level}.json").read_text(encoding="utf-8"))


def test_core_has_readable_style_preserving_corner_and_control_candidate():
    sample = task("level2_core")
    source, target = sample["source"], sample["target_canvas"]
    original = {element["id"]: element for element in source["elements"]}
    evaluation = build_eval_context(source)
    suitable = []
    for candidate in generate_candidates(source, target):
        elements = {element["id"]: element for element in candidate["elements"]}
        badge, action = elements["badge_bg"], elements["cta_bg"]
        source_ratio = original["badge_bg"]["width"] / original["cta_bg"]["width"]
        if abs(badge["width"] / action["width"] - source_ratio) > 0.01:
            continue
        aspect = badge["width"] / badge["height"]
        if (
            abs(aspect / (original["badge_bg"]["width"] / original["badge_bg"]["height"]) - 1)
            > 0.01
        ):
            continue
        if badge["x"] + badge["width"] / 2 > target["width"] * 0.25:
            continue
        if badge["y"] + badge["height"] / 2 > target["height"] * 0.25:
            continue
        if elements["product"]["height"] < target["height"] * 0.6:
            continue
        fonts = [
            element["font_size"] for element in candidate["elements"] if element["type"] == "text"
        ]
        if min(fonts) < 10:
            continue
        result = score(source, candidate, evaluation, target_canvas=target)
        if not result["violations"] and result["metrics"]["raw_collisions"] == 0:
            suitable.append(candidate)
    assert suitable, (
        "No candidate preserves the corner, control proportions and readable composition"
    )


def test_selected_core_keeps_corner_control_hierarchy_and_readability():
    sample = task("level2_core")
    result = resize(sample["source"], sample["target_canvas"])
    elements = {element["id"]: element for element in result["elements"]}
    badge, action = elements["badge_bg"], elements["cta_bg"]
    canvas = result["canvas"]
    assert badge["x"] + badge["width"] / 2 <= canvas["width"] * 0.25
    assert badge["y"] + badge["height"] / 2 <= canvas["height"] * 0.25
    assert badge["width"] / action["width"] == pytest.approx(220 / 420, abs=1e-5)
    assert elements["product"]["height"] >= canvas["height"] * 0.6
    assert (
        min(element["font_size"] for element in result["elements"] if element["type"] == "text")
        >= 10
    )
    assert check_contract(sample["source"], result, canvas) == []
    assert score(sample["source"], result)["metrics"]["raw_collisions"] == 0


def test_appended_style_variants_preserve_identity_and_bounds():
    from engine.layout.candidates import generate_style_candidates

    sample = task("level2_core")
    source = sample["source"]
    for element in source["elements"]:
        if element.get("group_id") == "badge":
            element["rotation"] = -12
    before = deepcopy(source)
    target = {"width": 1600, "height": 400}
    variants = generate_style_candidates(source, target)
    assert 1 <= len(variants) <= 8
    for candidate in variants:
        assert set(candidate) == {"canvas", "elements"}
        assert candidate["canvas"] == target
        assert [element["id"] for element in candidate["elements"]] == [
            element["id"] for element in source["elements"]
        ]
        assert check_contract(source, candidate, target) == []
        for original, element in zip(source["elements"], candidate["elements"], strict=True):
            mutable = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}
            assert {key: value for key, value in element.items() if key not in mutable} == {
                key: value for key, value in original.items() if key not in mutable
            }
            box = bounds(element)
            assert box.x >= -1e-6 and box.y >= -1e-6
            assert box.x + box.width <= target["width"] + 1e-6
            assert box.y + box.height <= target["height"] + 1e-6
    assert source == before


def test_style_variants_use_geometry_instead_of_sample_names():
    from engine.layout.candidates import generate_style_candidates

    sample = task("level2_core")
    source = sample["source"]
    renamed = deepcopy(source)
    group_map = {}
    for index, element in enumerate(renamed["elements"]):
        element["id"] = f"opaque-{index}"
        if element.get("group_id"):
            group_map.setdefault(element["group_id"], f"set-{len(group_map)}")
            element["group_id"] = group_map[element["group_id"]]
    expected = generate_style_candidates(source, sample["target_canvas"])
    actual = generate_style_candidates(renamed, sample["target_canvas"])
    assert actual and len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        for element, original in zip(a["elements"], b["elements"], strict=True):
            assert {
                key: value for key, value in element.items() if key not in {"id", "group_id"}
            } == {key: value for key, value in original.items() if key not in {"id", "group_id"}}


@pytest.mark.parametrize(
    "target", [{"width": 1600, "height": 400}, {"width": 1080, "height": 1920}]
)
def test_style_copy_remains_packed_and_controls_scale_together(target):
    from engine.layout.candidates import generate_style_candidates

    source = task("level2_core")["source"]
    for candidate in generate_style_candidates(source, target):
        elements = {element["id"]: element for element in candidate["elements"]}
        head, body = elements["headline"], elements["subhead"]
        gap = body["y"] - head["y"] - head["height"]
        assert 0 <= gap <= min(target.values()) * 0.04
        badge, action = elements["badge_bg"], elements["cta_bg"]
        assert badge["width"] / action["width"] == pytest.approx(220 / 420, abs=1e-5)
        assert badge["width"] / badge["height"] == pytest.approx(220 / 56)
        assert action["width"] / action["height"] == pytest.approx(420 / 90)
