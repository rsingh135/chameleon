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


def _numeric_geometry_source(kind="shape", out_of_view=False):
    element = {
        "id": "source-leaf",
        "type": kind,
        "x": 250 if out_of_view else 10,
        "y": 10,
        "width": 100,
        "height": 50,
    }
    if kind == "text":
        element.update(
            content="Readable", font_family="Solstice Sans", font_size=16, color="#111111"
        )
    elif kind == "image":
        element.update(asset_id="unavailable-contract-fixture.png", fit="contain")
    else:
        element.update(shape_type="rectangle", fill_color="#112233")
    return {"canvas": {"width": 200, "height": 100}, "elements": [element]}


@pytest.mark.parametrize("kind", ["shape", "text", "image"])
@pytest.mark.parametrize("out_of_view", [False, True])
@pytest.mark.parametrize("field", ["x", "y", "width", "height"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_geometry_never_uses_authored_crop_exemption(kind, out_of_view, field, value):
    source = _numeric_geometry_source(kind, out_of_view)
    context = build_context(source)
    candidate = copy.deepcopy(source)
    candidate["elements"][0][field] = value
    violations = check_contract(source, candidate, source["canvas"], context)
    assert [v.code for v in violations] == ["invalid_geometry"]
    assert violations[0].element_id == "source-leaf"
    assert field in violations[0].message


@pytest.mark.parametrize("field", ["font_size", "rotation", "corner_radius", "stroke_width"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_optional_nonfinite_geometry_is_rejected_before_measurement(field, value):
    source = _numeric_geometry_source("text" if field == "font_size" else "shape", True)
    context = build_context(source)
    candidate = copy.deepcopy(source)
    candidate["elements"][0][field] = value
    assert _codes(source, candidate, context) == ["invalid_geometry"]


@pytest.mark.parametrize(
    "field", ["x", "y", "width", "height", "font_size", "rotation", "corner_radius", "stroke_width"]
)
@pytest.mark.parametrize("value", [True, False, "12", None])
def test_geometry_requires_real_numbers_and_never_accepts_boolean_coercion(field, value):
    source = _numeric_geometry_source("text" if field == "font_size" else "shape", True)
    context = build_context(source)
    candidate = copy.deepcopy(source)
    candidate["elements"][0][field] = value
    assert _codes(source, candidate, context) == ["invalid_geometry"]


def test_mixed_valid_and_invalid_leaves_report_only_invalid_geometry():
    source = _numeric_geometry_source(out_of_view=True)
    good = copy.deepcopy(source["elements"][0])
    good.update(id="valid-leaf", x=10)
    source["elements"].append(good)
    context = build_context(source)
    candidate = copy.deepcopy(source)
    candidate["elements"][0].update(x=float("nan"), rotation=float("inf"))
    violations = check_contract(source, candidate, source["canvas"], context)
    assert len(violations) == 2
    assert {v.element_id for v in violations} == {"source-leaf"}
    assert {v.code for v in violations} == {"invalid_geometry"}
    assert candidate["elements"][1] == good


def test_invalid_geometry_stops_text_measurement_and_quality_scoring(monkeypatch):
    import importlib

    contract = importlib.import_module("engine.eval.contract")
    scoring = importlib.import_module("engine.eval.score")
    source = _numeric_geometry_source("text", True)
    context = build_context(source)
    candidate = copy.deepcopy(source)
    candidate["elements"][0]["font_size"] = float("nan")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError(
            "Invalid candidate geometry reached text measurement or quality metrics"
        )

    class UnusedMeasurer:
        layout = must_not_run

    monkeypatch.setattr(contract, "get_measurer", lambda *_: UnusedMeasurer())
    monkeypatch.setattr(scoring, "compute_metrics", must_not_run)
    result = scoring.score(source, candidate, context, target_canvas=source["canvas"])
    assert result["total"] == 0
    assert result["metrics"] == {}
    assert [v.code for v in result["violations"]] == ["invalid_geometry"]


def test_finite_authored_crop_and_image_aspect_rules_remain_unchanged():
    source = _numeric_geometry_source("image", True)
    context = build_context(source)
    candidate = copy.deepcopy(source)
    candidate["elements"][0].update(x=-20.5, y=0.5, width=160.0, height=80.0)
    assert check_contract(source, candidate, source["canvas"], context) == []
    candidate["elements"][0]["width"] = 190.0
    assert _codes(source, candidate, context) == ["aspect_drift"]
    source["elements"][0]["fit"] = "cover"
    candidate["elements"][0]["fit"] = "cover"
    assert check_contract(source, candidate, source["canvas"], build_context(source)) == []


def test_bad_candidate_does_not_prevent_valid_candidates_from_using_shared_context(monkeypatch):
    import importlib

    scoring = importlib.import_module("engine.eval.score")
    source = _numeric_geometry_source(out_of_view=True)
    context = build_context(source)
    invalid = copy.deepcopy(source)
    invalid["elements"][0]["x"] = float("nan")
    evaluated = []

    def metrics(_source, candidate, _context, **_kwargs):
        evaluated.append(candidate["elements"][0]["x"])
        return {"containment": 1.0}

    monkeypatch.setattr(scoring, "compute_metrics", metrics)
    results = [scoring.score(source, candidate, context) for candidate in (source, invalid, source)]
    assert [result["total"] for result in results] == [1.0, 0.0, 1.0]
    assert evaluated == [250, 250]
    assert [v.code for v in results[1]["violations"]] == ["invalid_geometry"]


def test_finite_optional_geometry_is_still_permitted():
    source = _numeric_geometry_source(out_of_view=True)
    source["elements"][0]["rotation"] = 12.5
    candidate = copy.deepcopy(source)
    candidate["elements"][0].update(corner_radius=4.5, stroke_width=1.25)
    assert check_contract(source, candidate, source["canvas"]) == []
