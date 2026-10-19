"""Proxy metrics must prefer the designer's layout over degraded versions of
it. Each test degrades one property and checks the matching metric drops."""

import copy
import json
from pathlib import Path

import pytest

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


def test_identity_scores_high_with_pixels(ctx3):
    plate = render_plate(LEVEL3, ASSETS, max_size=512)
    r = score(LEVEL3, LEVEL3, ctx3, plate)
    assert r["violations"] == []
    assert r["total"] >= 0.9
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
