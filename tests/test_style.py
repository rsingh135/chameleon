"""Source style checks expose concrete losses without freezing every position."""

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from engine.eval.style import analyze_style
from engine.layout import generate_candidates

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def core():
    return json.loads(
        (ROOT / "input/chameleon_sample_level2_core.json").read_text(encoding="utf-8")
    )["source"]


def _codes(result):
    return {warning["code"] for warning in result["warnings"]}


def _text(eid, size, x=10, y=10):
    return {
        "id": eid,
        "type": "text",
        "content": "Copy",
        "font_family": "Solstice Sans",
        "font_size": size,
        "color": "rgba(0,0,0,1)",
        "x": x,
        "y": y,
        "width": 100,
        "height": 50,
    }


@pytest.mark.parametrize("level", ["level1_warmup", "level2_core", "level3_advanced"])
def test_source_identity_is_clean(level):
    source = json.loads(
        (ROOT / f"input/chameleon_sample_{level}.json").read_text(encoding="utf-8")
    )["source"]
    result = analyze_style(source, deepcopy(source))
    assert result["score"] == 1
    assert result["warnings"] == []
    assert all(value == 1 for value in result["metrics"].values())


@pytest.mark.parametrize("scale", [0.5, 2])
def test_uniform_geometry_and_font_scaling_is_clean(core, scale):
    candidate = deepcopy(core)
    candidate["canvas"] = {key: value * scale for key, value in core["canvas"].items()}
    for element in candidate["elements"]:
        for key in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in element:
                element[key] *= scale
    result = analyze_style(core, candidate)
    assert result["score"] == 1
    assert result["warnings"] == []


def test_core_template_exposes_lost_corner_control_proportions_and_font_order(core):
    candidate = generate_candidates(core, {"width": 1600, "height": 400})[4]
    result = analyze_style(core, candidate)
    assert {"corner_affinity", "control_hierarchy", "text_hierarchy"} <= _codes(result)
    control_warning = next(w for w in result["warnings"] if w["code"] == "control_hierarchy")
    assert {"badge_bg", "badge_label", "cta_bg", "cta_label"} == set(control_warning["element_ids"])


def test_correcting_source_relationships_improves_style(core):
    target = {"width": 1600, "height": 400}
    weak = generate_candidates(core, target)[4]
    corrected = deepcopy(weak)
    original = {element["id"]: element for element in core["elements"]}
    scale = math.sqrt(target["width"] * target["height"] / (1080 * 1080))
    elements = {element["id"]: element for element in corrected["elements"]}
    for element in corrected["elements"]:
        source = original[element["id"]]
        if element["type"] == "text":
            element["font_size"] = source["font_size"] * scale
        if element.get("group_id") in ("badge", "cta"):
            origin = original[element["group_id"] + "_bg"]
            x, y = (16, 16) if element["group_id"] == "badge" else (900, 280)
            element.update(
                x=x + (source["x"] - origin["x"]) * scale,
                y=y + (source["y"] - origin["y"]) * scale,
                width=source["width"] * scale,
                height=source["height"] * scale,
            )
    for eid, x, y in (("product", 200, 14), ("logo", 650, 20)):
        elements[eid].update(
            x=x,
            y=y,
            width=original[eid]["width"] * scale,
            height=original[eid]["height"] * scale,
        )
    elements["headline"].update(x=600, y=130, width=700, height=50)
    elements["subhead"].update(x=600, y=190, width=700, height=32)
    better, worse = analyze_style(core, corrected), analyze_style(core, weak)
    assert better["score"] > worse["score"]
    assert better["score"] > 0.9
    assert better["warnings"] == []


def test_font_hierarchy_ignores_microscopic_source_ties():
    source = {
        "canvas": {"width": 400, "height": 400},
        "elements": [_text("a", 20), _text("b", 20.01)],
    }
    candidate = deepcopy(source)
    candidate["elements"][0]["font_size"] = 20.01
    candidate["elements"][1]["font_size"] = 20
    result = analyze_style(source, candidate)
    assert "text_hierarchy" not in _codes(result)
    assert result["metrics"]["text_hierarchy"] == 1


def test_meaningful_font_order_reversal_is_reported():
    source = {"canvas": {"width": 400, "height": 400}, "elements": [_text("a", 32), _text("b", 20)]}
    candidate = deepcopy(source)
    candidate["elements"][0]["font_size"] = 18
    result = analyze_style(source, candidate)
    assert "text_hierarchy" in _codes(result)
    assert result["metrics"]["text_hierarchy"] == 0


def test_adjacent_copy_can_change_orientation_while_remaining_close(core):
    candidate = deepcopy(core)
    elements = {element["id"]: element for element in candidate["elements"]}
    elements["headline"].update(x=10, y=500, width=450, height=100)
    elements["subhead"].update(x=472, y=500, width=590, height=70)
    assert "copy_cohesion" not in _codes(analyze_style(core, candidate))


def test_adjacent_copy_separation_is_reported(core):
    candidate = deepcopy(core)
    elements = {element["id"]: element for element in candidate["elements"]}
    elements["headline"].update(x=90, y=300, width=900, height=70)
    elements["subhead"]["y"] = 850
    result = analyze_style(core, candidate)
    assert "copy_cohesion" in _codes(result)


def test_inflating_a_text_frame_cannot_hide_separated_copy():
    source = {
        "canvas": {"width": 800, "height": 800},
        "elements": [
            {**_text("heading", 48, 100, 100), "content": "Headline", "width": 600, "height": 100},
            {**_text("body", 24, 100, 200), "content": "Body text", "width": 600, "height": 60},
        ],
    }
    candidate = deepcopy(source)
    candidate["elements"][0]["height"] = 500
    candidate["elements"][1]["y"] = 600
    result = analyze_style(source, candidate)
    assert "copy_cohesion" in _codes(result)
    assert result["metrics"]["copy_cohesion"] < 0.1


def test_foreground_cover_images_still_have_prominence_checks():
    source = {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            {
                "id": "product",
                "type": "image",
                "asset_id": "solstice_bottle.png",
                "fit": "cover",
                "x": 50,
                "y": 50,
                "width": 300,
                "height": 300,
            }
        ],
    }
    candidate = deepcopy(source)
    candidate["elements"][0].update(width=20, height=20)
    result = analyze_style(source, candidate)
    assert "image_prominence" in _codes(result)
    assert result["metrics"]["image_prominence"] == pytest.approx((20 / 300) ** 2, abs=1e-6)


def test_cover_prominence_accounts_for_alpha_content_lost_to_cropping(tmp_path):
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 50, 100))
    image.save(tmp_path / "edge-art.png")
    source = {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            {
                "id": "art",
                "type": "image",
                "asset_id": "edge-art.png",
                "fit": "cover",
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 100,
            }
        ],
    }
    candidate = deepcopy(source)
    candidate["elements"][0]["width"] = 100
    result = analyze_style(source, candidate, tmp_path)
    assert result["metrics"]["image_prominence"] == 0
    assert "image_prominence" in _codes(result)


def test_actual_cover_background_is_excluded_from_image_prominence():
    source = {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            {
                "id": "plate",
                "type": "image",
                "asset_id": "solstice_bg_gradient.png",
                "fit": "cover",
                "x": 0,
                "y": 0,
                "width": 400,
                "height": 400,
            }
        ],
    }
    candidate = deepcopy(source)
    candidate["canvas"] = {"width": 1600, "height": 400}
    candidate["elements"][0]["width"] = 1600
    result = analyze_style(source, candidate)
    assert result["metrics"]["image_prominence"] == 1
    assert result["warnings"] == []


def test_ordinary_content_is_not_pinned_to_its_source_corner():
    source = {"canvas": {"width": 400, "height": 400}, "elements": [_text("copy", 20)]}
    candidate = deepcopy(source)
    candidate["elements"][0].update(x=260, y=300)
    result = analyze_style(source, candidate)
    assert "corner_affinity" not in _codes(result)
    assert result["score"] == 1


def test_alpha_bounds_and_contain_letterboxing_measure_visible_image_mass(tmp_path):
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (20, 20, 80, 80))
    image.save(tmp_path / "asset.png")
    source = {
        "canvas": {"width": 800, "height": 400},
        "elements": [
            {
                "id": "art",
                "type": "image",
                "asset_id": "asset.png",
                "fit": "contain",
                "x": 100,
                "y": 100,
                "width": 400,
                "height": 200,
            }
        ],
    }
    candidate = deepcopy(source)
    candidate["elements"][0]["width"] = 200
    # Both boxes draw the same 200x200 contained image. Structural image-aspect
    # constraints are checked elsewhere; this test isolates visible style mass.
    measured = analyze_style(source, candidate, tmp_path)
    fallback = analyze_style(source, candidate, tmp_path / "missing")
    assert measured["metrics"]["image_prominence"] == 1
    assert "image_prominence" not in _codes(measured)
    assert fallback["metrics"]["image_prominence"] == 0.5
    assert "image_prominence" in _codes(fallback)


def test_missing_assets_fall_back_to_geometry_without_crashing(tmp_path):
    source = {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            {
                "id": "missing",
                "type": "image",
                "asset_id": "missing.png",
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 200,
            }
        ],
    }
    candidate = deepcopy(source)
    candidate["elements"][0].update(width=100, height=100)
    result = analyze_style(source, candidate, tmp_path)
    assert result["metrics"]["image_prominence"] == 0.25
    assert "geometry" in result["warnings"][0]["message"]


@pytest.mark.parametrize(
    "elements",
    [
        [],
        [
            {
                "id": "shape",
                "type": "shape",
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 20,
                "shape_type": "rectangle",
                "fill_color": "#fff",
            }
        ],
    ],
)
def test_designs_without_text_or_images_are_supported(elements):
    source = {"canvas": {"width": 100, "height": 100}, "elements": elements}
    assert analyze_style(source, deepcopy(source))["score"] == 1


def test_style_analysis_is_deterministic_and_does_not_mutate_inputs(core):
    candidate = generate_candidates(core, {"width": 1600, "height": 400})[4]
    before = deepcopy((core, candidate))
    first = analyze_style(core, candidate)
    assert analyze_style(core, candidate) == first
    assert (core, candidate) == before


def test_style_checks_do_not_depend_on_sample_ids(core):
    candidate = generate_candidates(core, {"width": 1600, "height": 400})[4]
    expected = analyze_style(core, candidate)
    renamed_source, renamed_candidate = deepcopy(core), deepcopy(candidate)
    names = {element["id"]: f"element-{index}" for index, element in enumerate(core["elements"])}
    for design in (renamed_source, renamed_candidate):
        for element in design["elements"]:
            element["id"] = names[element["id"]]
    actual = analyze_style(renamed_source, renamed_candidate)
    assert actual["score"] == expected["score"]
    assert [warning["code"] for warning in actual["warnings"]] == [
        warning["code"] for warning in expected["warnings"]
    ]
