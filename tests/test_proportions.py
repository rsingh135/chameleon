"""Proportion diagnostics catch distortions that ownership checks cannot."""

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval.proportions import analyze_proportions
from engine.layout.relationships import build_relationship_graph
from tools.composition_cases import generate_cases

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source():
    return json.loads((ROOT / "input/chameleon_sample_level2_core.json").read_text())["source"]


def transformed(source, scale=1.0, dx=0.0, dy=0.0):
    output = deepcopy(source)
    for element in output["elements"]:
        element["x"] = element["x"] * scale + dx
        element["y"] = element["y"] * scale + dy
        for field in ("width", "height", "font_size", "corner_radius", "stroke_width"):
            if field in element:
                element[field] *= scale
    return output


def test_one_similarity_transform_has_zero_deformation_without_mutating_source(source):
    before = deepcopy(source)
    result = analyze_proportions(source, transformed(source, 1.7, 91, -23))
    assert result["score"] == 1.0
    assert result["warnings"] == []
    assert result["metrics"]["rigid_group_error"] == 0.0
    assert result["metrics"]["campaign_scale_rms_log"] == 0.0
    assert source == before


def test_independent_label_font_boost_is_not_rigid_group_preservation(source):
    output = deepcopy(source)
    next(e for e in output["elements"] if e["id"] == "badge_label")["font_size"] *= 1.5
    result = analyze_proportions(source, output)
    assert result["metrics"]["rigid_group_error"] == pytest.approx(math.log(1.5), abs=1e-6)
    assert any(w["code"] == "rigid_group_deformation" for w in result["warnings"])


def test_separately_filled_brand_and_message_slots_report_scale_drift(source):
    output = deepcopy(source)
    for element in output["elements"]:
        if element["id"] in {"logo", "product"}:
            element["width"] *= 0.5
            element["height"] *= 0.5
    result = analyze_proportions(source, output)
    campaign = next(c for c in result["clusters"] if c["kind"] == "campaign")
    assert campaign["largest_to_smallest_scale"] == 2.0
    assert any(w["code"] == "campaign_scale_drift" for w in result["warnings"])


def unequal_products():
    task = generate_cases()[0]
    source = deepcopy(task["source"])
    smaller = task["expected_campaigns"][1]["product_ids"][0]
    for element in source["elements"]:
        if element["id"] == smaller:
            element["x"] += element["width"] * 0.175
            element["y"] += element["height"] * 0.175
            element["width"] *= 0.65
            element["height"] *= 0.65
    return source, smaller


def test_equalizing_two_different_source_products_is_measurable():
    source, smaller = unequal_products()
    output = deepcopy(source)
    for element in output["elements"]:
        if element["id"] == smaller:
            element["width"] /= 0.65
            element["height"] /= 0.65
    result = analyze_proportions(source, output)
    assert result["metrics"]["sibling_product_scale_ratio"] == pytest.approx(1 / 0.65, abs=1e-6)
    assert any(w["code"] == "sibling_product_scale_drift" for w in result["warnings"])


def test_shared_scale_keeps_unequal_product_prominence_unequal():
    source, _ = unequal_products()
    result = analyze_proportions(source, transformed(source, 0.8, 20, 40))
    assert result["metrics"]["sibling_product_scale_ratio"] == 1.0
    assert result["score"] == 1.0


def test_standalone_text_reflow_does_not_masquerade_as_font_ratio_drift(source):
    output = deepcopy(source)
    for element in output["elements"]:
        if element["type"] == "text" and not element.get("group_id"):
            element["width"] *= 0.7
            element["height"] *= 2
    result = analyze_proportions(source, output)
    assert result["score"] == 1.0
    assert result["warnings"] == []


def test_proportion_analysis_reuses_the_original_graph(source, monkeypatch):
    import engine.eval.proportions as module

    graph = build_relationship_graph(source)
    monkeypatch.setattr(module, "build_relationship_graph", lambda *_: pytest.fail("rebuilt graph"))
    assert analyze_proportions(source, deepcopy(source), graph)["score"] == 1.0


def test_incomplete_candidate_does_not_get_a_perfect_proportion_score(source):
    output = deepcopy(source)
    output["elements"].pop()
    assert analyze_proportions(source, output)["score"] == 0.0


def test_removing_rounded_geometry_is_not_an_unchanged_group(source):
    output = deepcopy(source)
    for element in output["elements"]:
        element.pop("corner_radius", None)
    result = analyze_proportions(source, output)
    assert result["score"] < 1.0
    assert any(w["code"] == "rigid_group_deformation" for w in result["warnings"])


def test_duplicate_candidate_ids_do_not_collapse_into_a_perfect_score(source):
    output = deepcopy(source)
    output["elements"].append(deepcopy(output["elements"][-1]))
    assert analyze_proportions(source, output)["score"] == 0.0


def test_singleton_image_aspect_drift_is_measured_even_if_area_is_unchanged(source):
    output = deepcopy(source)
    image = next(e for e in output["elements"] if e["id"] == "product")
    image["width"] *= 1.02
    image["height"] /= 1.02
    result = analyze_proportions(source, output)
    assert result["score"] < 1.0
    assert result["metrics"]["rigid_group_error"] > 0
