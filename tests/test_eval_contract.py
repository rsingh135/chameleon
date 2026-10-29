"""engine.eval.contract must catch every hard-rule violation and accept a
design that respects them. Mutations are applied to the sample sources
themselves (a valid design) so each test isolates one rule."""

import copy
import json
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = sorted((REPO_ROOT / "input").glob("*.json"))


@pytest.fixture(params=SAMPLES, ids=lambda p: p.stem.removeprefix("chameleon_sample_"))
def sample(request):
    task = json.loads(request.param.read_text())
    return task["source"], build_context(task["source"], REPO_ROOT / "brand_kit/assets")


def _codes(source, candidate, ctx, target=None):
    return sorted({v.code for v in check_contract(source, candidate, target, ctx)})


def _text(design):
    return next(el for el in design["elements"] if el["type"] == "text")


def test_identity_is_clean(sample):
    source, ctx = sample
    assert check_contract(source, copy.deepcopy(source), source["canvas"], ctx) == []


def test_canvas_mismatch(sample):
    source, ctx = sample
    assert "canvas_mismatch" in _codes(source, source, ctx, {"width": 1, "height": 1})


def test_dropped_and_added_elements(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    dropped = cand["elements"].pop()
    cand["elements"].append({**dropped, "id": "ghost"})
    codes = _codes(source, cand, ctx)
    assert "id_missing" in codes and "id_unexpected" in codes


def test_duplicate_id(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    cand["elements"].append(copy.deepcopy(cand["elements"][0]))
    assert "id_duplicate" in _codes(source, cand, ctx)


def test_frozen_fields(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    _text(cand)["content"] = "rewritten"
    _text(cand)["font_family"] = "Comic Sans"
    assert _codes(source, cand, ctx) == ["frozen_field_changed"]


@pytest.mark.parametrize("field", ["native_text_effects", "source_art"])
def test_imported_style_metadata_cannot_disappear_or_change(sample, field):
    original, _ = sample
    source = copy.deepcopy(original)
    _text(source)[field] = {"version": 1, "source": "authored identity"}
    context = build_context(source)
    changed = copy.deepcopy(source)
    _text(changed)[field]["source"] = "altered"
    assert _codes(source, changed, context) == ["frozen_field_changed"]
    del _text(changed)[field]
    assert _codes(source, changed, context) == ["frozen_field_changed"]


def test_off_canvas(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    _text(cand)["x"] = source["canvas"]["width"] - 1
    assert _codes(source, cand, ctx) == ["off_canvas"]


def test_missing_required_field(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    del _text(cand)["color"]
    assert "missing_field" in _codes(source, cand, ctx)


def test_aspect_drift_on_contain_image(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    img = next(el for el in cand["elements"] if el["type"] == "image" and el.get("fit") != "cover")
    img["width"] *= 1.5
    assert _codes(source, cand, ctx) == ["aspect_drift"]


def test_cover_image_may_change_aspect(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    bg = next(el for el in cand["elements"] if el.get("fit") == "cover")
    bg["height"] = bg["height"] * 0.5
    assert "aspect_drift" not in _codes(source, cand, ctx)


def test_font_size_floor(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    _text(cand)["font_size"] = 0.5
    assert _codes(source, cand, ctx) == ["font_size_below_min"]


def test_text_overflow_height(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    _text(cand)["font_size"] = 400
    assert "text_overflow_height" in _codes(source, cand, ctx)


def test_text_overflow_width_unbreakable_word(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    t = _text(cand)
    t["width"] = 8  # narrower than any single word; renderer will not hyphenate
    assert "text_overflow_width" in _codes(source, cand, ctx)


def test_degenerate_size(sample):
    source, ctx = sample
    cand = copy.deepcopy(source)
    _text(cand)["height"] = 0
    assert "degenerate_size" in _codes(source, cand, ctx)


def test_shape_geometry_may_scale(sample):
    """corner_radius / stroke_width are px geometry, not identity: a pill
    button that shrinks keeps its pill shape by scaling its radius."""
    source, ctx = sample
    cand = copy.deepcopy(source)
    shapes = [el for el in cand["elements"] if el["type"] == "shape"]
    if not shapes:
        pytest.skip("no shapes in this sample")
    for el in shapes:
        el["corner_radius"] = el.get("corner_radius", 0) * 0.5 + 1
        el["stroke_width"] = 3
    assert "frozen_field_changed" not in _codes(source, cand, ctx)
