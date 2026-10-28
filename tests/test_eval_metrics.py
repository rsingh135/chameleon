"""Proxy metrics must prefer the designer's layout over degraded versions of
it. Each test degrades one property and checks the matching metric drops."""

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from engine.eval import build_context, score
from engine.eval.catalogue import CATALOGUE, by_aspect_distance
from engine.eval.render import render_plate
from engine.eval.text import get_measurer

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "brand_kit/assets"
LEVEL2 = json.loads((REPO_ROOT / "input/chameleon_sample_level2_core.json").read_text())["source"]
LEVEL3 = json.loads((REPO_ROOT / "input/chameleon_sample_level3_advanced.json").read_text())[
    "source"
]


def _el(design, el_id):
    return next(el for el in design["elements"] if el["id"] == el_id)


@pytest.fixture(scope="module")
def ctx2():
    return build_context(LEVEL2, ASSETS)


@pytest.fixture(scope="module")
def ctx3():
    return build_context(LEVEL3, ASSETS)


def test_identity_scores_high_json_only(ctx2):
    r = score(LEVEL2, LEVEL2, ctx2)
    assert r["violations"] == []
    assert r["total"] >= 0.9
    assert r["metrics"]["raw_collisions"] == 0


def test_identity_scores_high_with_pixels(ctx3):
    plate = render_plate(LEVEL3, ASSETS, max_size=512)
    r = score(LEVEL3, LEVEL3, ctx3, plate)
    assert r["violations"] == []
    assert r["total"] >= 0.9
    assert r["metrics"]["raw_collisions"] == 0
    assert r["metrics"]["raw_min_contrast"] > 3.0  # white text on dark gradient / orange


def test_violation_zeroes_total(ctx2):
    cand = copy.deepcopy(LEVEL2)
    _el(cand, "logo")["width"] *= 2
    r = score(LEVEL2, cand, ctx2)
    assert r["total"] == 0.0 and r["metrics"] == {}


def test_new_collision_is_penalised(ctx2):
    cand = copy.deepcopy(LEVEL2)
    head = _el(cand, "headline")
    prod = _el(cand, "product")
    head["y"] = prod["y"] + prod["height"] / 2  # headline now sits on the product
    base = score(LEVEL2, LEVEL2, ctx2)["metrics"]
    degraded = score(LEVEL2, cand, ctx2)["metrics"]
    assert degraded["raw_collisions"] >= 1
    assert degraded["collision_free"] < base["collision_free"]


def test_intentional_overlap_is_not_a_collision(ctx2):
    # deco_circle_1 overlaps the product in the source on purpose.
    assert frozenset(("deco_circle_1", "product")) in ctx2.intentional_overlaps
    assert score(LEVEL2, LEVEL2, ctx2)["metrics"]["raw_collisions"] == 0


@pytest.fixture
def padded_product(tmp_path):
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (45, 20, 55, 80))
    image.save(tmp_path / "product.png")
    Image.new("RGBA", (20, 20), (0, 0, 255, 255)).save(tmp_path / "small.png")
    source = {
        "canvas": {"width": 800, "height": 800},
        "elements": [
            {
                "id": "product",
                "type": "image",
                "x": 100,
                "y": 100,
                "width": 400,
                "height": 400,
                "asset_id": "product.png",
                "fit": "contain",
            },
            {
                "id": "copy",
                "type": "text",
                "x": 150,
                "y": 200,
                "width": 200,
                "height": 120,
                "font_size": 24,
                "content": "Copy",
                "font_family": "Solstice Sans",
                "color": "#000000",
            },
        ],
    }
    return source, tmp_path


def test_transparent_image_padding_and_text_frame_overlap_are_not_collisions(padded_product):
    source, assets = padded_product
    context = build_context(source, assets)
    assert frozenset(("product", "copy")) in context.intentional_overlaps
    result = score(source, source, context)
    assert result["violations"] == []
    assert result["metrics"]["raw_collisions"] == 0
    assert result["metrics"]["collision_free"] == 1


@pytest.mark.parametrize("move", ["text", "image"])
def test_source_padding_overlap_cannot_authorize_new_visible_collision(padded_product, move):
    source, assets = padded_product
    context = build_context(source, assets)
    candidate = copy.deepcopy(source)
    if move == "text":
        _el(candidate, "copy")["x"] = 280
    else:
        _el(candidate, "product")["x"] = 0
    result = score(source, candidate, context)
    assert result["violations"] == []
    assert result["metrics"]["raw_collisions"] == 1
    assert result["metrics"]["collision_free"] < 1


def test_visible_image_to_image_collision_uses_alpha_not_padded_frames(padded_product):
    source, assets = padded_product
    source["elements"][1] = {
        "id": "small",
        "type": "image",
        "asset_id": "small.png",
        "fit": "contain",
        "x": 150,
        "y": 200,
        "width": 50,
        "height": 50,
    }
    context = build_context(source, assets)
    assert score(source, source, context)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    _el(candidate, "small")["x"] = 290
    assert score(source, candidate, context)["metrics"]["raw_collisions"] == 1


def test_actual_visible_layering_is_retained_without_allowing_more_overlap(padded_product):
    source, assets = padded_product
    source["elements"][1] = {
        "id": "plate",
        "type": "shape",
        "shape_type": "rectangle",
        "fill_color": "#ffffff",
        "x": 230,
        "y": 200,
        "width": 60,
        "height": 40,
    }
    context = build_context(source, assets)
    assert score(source, source, context)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    _el(candidate, "plate")["x"] = 270
    assert score(source, candidate, context)["metrics"]["raw_collisions"] == 1


def test_visible_image_text_overlay_can_scale_with_its_source_layering(padded_product):
    source, assets = padded_product
    _el(source, "copy")["x"] = 280
    context = build_context(source, assets)
    candidate = copy.deepcopy(source)
    candidate["canvas"] = {"width": 1600, "height": 1600}
    for element in candidate["elements"]:
        for key in ("x", "y", "width", "height", "font_size"):
            if key in element:
                element[key] *= 2
    result = score(source, candidate, context)
    assert result["violations"] == []
    assert result["metrics"]["raw_collisions"] == 0


def test_image_inside_unused_multiline_text_area_is_not_a_collision(padded_product):
    source, assets = padded_product
    _el(source, "copy").update(content="Copy\n\nCopy", x=280, y=100, height=120)
    _el(source, "product").update(x=220, y=131, width=20, height=20, asset_id="small.png")
    context = build_context(source, assets)
    assert score(source, source, context)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    _el(candidate, "product").update(x=285, y=135)
    assert score(source, candidate, context)["metrics"]["raw_collisions"] == 0


@pytest.mark.parametrize("content", ["image", "text"])
def test_source_decoration_below_foreground_is_an_underlay_not_a_collision(padded_product, content):
    source, assets = padded_product
    _el(source, "product")["z_index"] = 3
    _el(source, "copy")["z_index"] = 4
    source["elements"].append(
        {
            "id": "ornament",
            "type": "shape",
            "shape_type": "ellipse",
            "fill_color": "#ffffff",
            "opacity": 0.15,
            "z_index": 1,
            "x": 550,
            "y": 500,
            "width": 120,
            "height": 120,
        }
    )
    context = build_context(source, assets)
    assert score(source, source, context)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    _el(candidate, "ornament").update(x=270 if content == "image" else 140, y=190)
    result = score(source, candidate, context)
    assert result["violations"] == []
    assert result["metrics"]["raw_collisions"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"opacity": 1},
        {"opacity": 0.5},
        {"z_index": 5},
        {"z_index": 3},
        {"group_id": "diagram"},
        {"type": "image", "asset_id": "small.png", "fit": "contain"},
    ],
)
def test_opaque_raised_grouped_or_image_elements_are_not_underlays(padded_product, overrides):
    source, assets = padded_product
    _el(source, "product")["z_index"] = 3
    _el(source, "copy")["z_index"] = 4
    _el(source, "copy")["y"] = 20
    ornament = {
        "id": "ornament",
        "type": "shape",
        "shape_type": "ellipse",
        "fill_color": "#ffffff",
        "opacity": 0.15,
        "z_index": 1,
        "x": 550,
        "y": 500,
        "width": 120,
        "height": 120,
        **overrides,
    }
    source["elements"].append(ornament)
    context = build_context(source, assets)
    assert score(source, source, context)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    _el(candidate, "ornament").update(x=270, y=190)
    result = score(source, candidate, context)
    assert result["violations"] == []
    assert result["metrics"]["raw_collisions"] == 1


def test_ungrouped_translucent_label_plate_does_not_gain_underlay_exemption(padded_product):
    source, assets = padded_product
    _el(source, "product")["z_index"] = 3
    _el(source, "copy")["y"] = 20
    source["elements"].extend(
        [
            {
                "id": "plate",
                "type": "shape",
                "shape_type": "rectangle",
                "fill_color": "#ffffff",
                "opacity": 0.15,
                "z_index": 1,
                "x": 550,
                "y": 500,
                "width": 120,
                "height": 100,
            },
            {
                "id": "label",
                "type": "text",
                "font_family": "Solstice Sans",
                "font_size": 18,
                "content": "Label",
                "color": "#000000",
                "z_index": 2,
                "x": 560,
                "y": 520,
                "width": 100,
                "height": 30,
            },
        ]
    )
    context = build_context(source, assets)
    assert score(source, source, context)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    _el(candidate, "plate").update(x=270, y=190)
    assert score(source, candidate, context)["metrics"]["raw_collisions"] == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_overlapping_source_text_frames_do_not_allow_new_line_collisions(grouped):
    source = {
        "canvas": {"width": 800, "height": 800},
        "elements": [
            {
                "id": name,
                "type": "text",
                "x": 100,
                "y": y,
                "width": 600,
                "height": height,
                "font_size": size,
                "content": content,
                "font_family": "Solstice Sans",
                "color": "#000000",
                **({"group_id": "copy"} if grouped else {}),
            }
            for name, y, height, size, content in (
                ("h", 100, 200, 48, "Headline"),
                ("b", 200, 60, 24, "Body text"),
            )
        ],
    }
    context = build_context(source, ASSETS)
    assert frozenset(("h", "b")) in context.intentional_overlaps
    assert score(source, source, context)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    candidate["elements"][1]["y"] = 100
    result = score(source, candidate, context)
    assert result["violations"] == []
    assert result["metrics"]["raw_collisions"] == 1
    assert result["metrics"]["collision_free"] < 1


def test_text_collision_line_bounds_respect_alignment_and_source_layering():
    source = {
        "canvas": {"width": 600, "height": 300},
        "elements": [
            {
                "id": name,
                "type": "text",
                "x": 20,
                "y": 20,
                "width": 550,
                "height": 100,
                "font_size": 24,
                "content": "Short",
                "font_family": "Solstice Sans",
                "color": "#000000",
                "text_align": alignment,
            }
            for name, alignment in (("left", "left"), ("right", "right"))
        ],
    }
    assert score(source, source)["metrics"]["raw_collisions"] == 0
    source["elements"][1]["text_align"] = "left"
    assert score(source, source)["metrics"]["raw_collisions"] == 0
    candidate = copy.deepcopy(source)
    candidate["elements"][1]["y"] += 100
    assert score(source, candidate)["metrics"]["raw_collisions"] == 0


def test_group_split_is_penalised(ctx2):
    cand = copy.deepcopy(LEVEL2)
    _el(cand, "cta_label")["y"] -= 300  # label leaves its button
    m = score(LEVEL2, cand, ctx2)["metrics"]
    assert m["group_integrity"] < 1.0
    assert m["containment"] < 1.0


def test_hierarchy_inversion_is_penalised(ctx2):
    cand = copy.deepcopy(LEVEL2)
    head, sub = _el(cand, "headline"), _el(cand, "subhead")
    head["font_size"], sub["font_size"] = 20.0, 24.0
    assert score(LEVEL2, cand, ctx2)["metrics"]["hierarchy"] < 1.0


def test_tiny_text_is_penalised(ctx2):
    cand = copy.deepcopy(LEVEL2)
    for el in cand["elements"]:
        if el["type"] == "text":
            el["font_size"] = 6.0
    assert score(LEVEL2, cand, ctx2)["metrics"]["text_size"] < 0.6


def test_fit_font_size_fits_and_is_monotone():
    m = get_measurer(REPO_ROOT / "brand_kit/fonts")
    head = _el(LEVEL3, "headline")
    small = {**head, "width": 300, "height": 120}
    fitted = m.fit_font_size(small)
    assert fitted is not None and fitted <= head["font_size"]
    assert m.fits(small, fitted)
    assert not m.fits(small, fitted * 1.5)


def test_catalogue_ranking():
    nearest = by_aspect_distance({"width": 1080, "height": 1080}, n=1)[0]
    assert CATALOGUE[nearest]["width"] == CATALOGUE[nearest]["height"]
    assert by_aspect_distance({"width": 728, "height": 90})[-1] in ("wide_skyscraper", "half_page")


def test_moving_a_corner_element_across_the_canvas_is_penalised(ctx3):
    cand = copy.deepcopy(LEVEL3)
    for el_id in ("ribbon_bg", "ribbon_label"):
        el = _el(cand, el_id)
        el["x"], el["y"] = 60, 1100  # top-right ribbon -> bottom-left
    base = score(LEVEL3, LEVEL3, ctx3)["metrics"]["position_affinity"]
    moved = score(LEVEL3, cand, ctx3)["metrics"]["position_affinity"]
    assert base == 1.0 and moved < base
