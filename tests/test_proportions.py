"""Proportion diagnostics catch distortions that ownership checks cannot."""

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval.proportions import analyze_proportions
from engine.layout.relationships import build_relationship_graph
from engine.resources import using_resources
from tools.composition_cases import generate_cases

ROOT = Path(__file__).resolve().parents[1]


def test_full_bleed_photo_cannot_hide_compressed_foreground_caption():
    folder = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        graph = build_relationship_graph(source)
        photo_ids = {
            eid for unit in graph.units if unit.role == "image_underlay" for eid in unit.element_ids
        }
        assert photo_ids
        excluded = photo_ids | {overlay.overlay_id for overlay in graph.overlays}
        foreground_ids = {
            eid for unit in graph.units for eid in unit.element_ids if eid not in excluded
        }
        candidate = deepcopy(source)
        for element in candidate["elements"]:
            if element["id"] not in foreground_ids:
                continue
            element["x"] = source["canvas"]["width"] / 4 + element["x"] * 0.5
            element["y"] = source["canvas"]["height"] / 4 + element["y"] * 0.5
            for field in ("width", "height", "font_size", "corner_radius", "stroke_width"):
                if field in element:
                    element[field] *= 0.5
        result = analyze_proportions(source, candidate, graph)
    assert not photo_ids.intersection(result["composition_span"]["element_ids"])
    assert result["composition_span"]["kind"] == "foreground-bands"
    assert result["metrics"]["composition_span_retention"] == pytest.approx(0.25, abs=0.005)
    assert source == before


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


def test_source_span_detects_an_undersized_composition_on_a_taller_canvas(source):
    output = deepcopy(source)
    output["canvas"] = {"width": 1080, "height": 1920}
    result = analyze_proportions(source, output)
    assert result["metrics"]["composition_span_retention"] == pytest.approx(1080 / 1920)


def test_source_span_is_capped_instead_of_rewarding_unlimited_growth(source):
    result = analyze_proportions(source, transformed(source, 1.4))
    assert result["metrics"]["composition_span_retention"] == 1.0


def _type_poster():
    def text(eid, y, content, size=30, x=250, width=500):
        return {
            "id": eid,
            "type": "text",
            "x": x,
            "y": y,
            "width": width,
            "height": size * 1.3,
            "content": content,
            "font_family": "Solstice Sans",
            "font_size": size,
            "color": "rgba(20,20,20,1)",
            "text_align": "center",
            "line_height": 1.2,
        }

    return {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [
            {
                "id": "page",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 0,
                "y": 0,
                "width": 1000,
                "height": 1000,
                "fill_color": "rgba(255,255,255,1)",
            },
            text("title", 200, "Create", 80),
            text("body", 480, "Details stay connected"),
            {
                "id": "event-plate",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 300,
                "y": 700,
                "width": 400,
                "height": 100,
                "fill_color": "rgba(240,240,240,1)",
                "group_id": "event",
            },
            {**text("event-copy", 730, "Saturday 14", 34, 320, 360), "group_id": "event"},
        ],
    }


def test_text_only_source_bands_detect_tall_target_compression_without_campaigns():
    source = _type_poster()
    graph = build_relationship_graph(source)
    assert not any(cluster.kind == "campaign" for cluster in graph.clusters)
    candidate = deepcopy(source)
    candidate["canvas"]["height"] = 2000
    candidate["elements"][0]["height"] = 2000
    result = analyze_proportions(source, candidate, graph)
    assert result["composition_span"]["active"]
    assert result["composition_span"]["kind"] == "foreground-bands"
    assert result["metrics"]["composition_span_retention"] == 0.5
    assert result["score"] == 1


def test_spreading_only_parent_bands_improves_span_without_changing_rigid_event():
    source = _type_poster()
    before = deepcopy(source)
    compressed = deepcopy(source)
    compressed["canvas"]["height"] = 2000
    spread = deepcopy(compressed)
    offsets = {"title": 200, "body": 480, "event-plate": 700, "event-copy": 700}
    for element in spread["elements"]:
        element["y"] += offsets.get(element["id"], 0)
    first, second = analyze_proportions(source, compressed), analyze_proportions(source, spread)
    assert (
        second["metrics"]["composition_span_retention"]
        > first["metrics"]["composition_span_retention"]
    )
    assert second["score"] == first["score"] == 1
    assert second["metrics"]["rigid_group_error"] == 0
    assert source == before


def test_source_span_preserves_authored_negative_space_and_caps_expansion():
    source = _type_poster()
    for element in source["elements"][1:]:
        element["y"] = 300 + element["y"] * 0.4
    original = analyze_proportions(source, deepcopy(source))
    larger = deepcopy(source)
    for element in larger["elements"][1:]:
        element["y"] = (element["y"] - 300) * 2
    expanded = analyze_proportions(source, larger)
    assert original["metrics"]["composition_span_retention"] == 1
    assert expanded["metrics"]["composition_span_retention"] == 1


def test_fine_footer_and_small_corner_support_cannot_disguise_compressed_body():
    source = _type_poster()
    source["elements"].extend(
        [
            {
                "id": "fine-footer",
                "type": "text",
                "x": 50,
                "y": 945,
                "width": 900,
                "height": 15,
                "content": "Fine footer",
                "font_size": 10,
                "font_family": "Solstice Sans",
                "color": "rgba(0,0,0,1)",
            },
            {
                "id": "corner",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 960,
                "y": 0,
                "width": 40,
                "height": 40,
                "fill_color": "rgba(255,0,0,1)",
            },
        ]
    )
    candidate = deepcopy(source)
    candidate["canvas"]["height"] = 2000
    next(e for e in candidate["elements"] if e["id"] == "fine-footer")["y"] = 1945
    result = analyze_proportions(source, candidate)
    assert result["metrics"]["composition_span_retention"] == 0.5
    assert not {"fine-footer", "corner"} & set(result["composition_span"]["element_ids"])


def test_small_middle_body_copy_remains_primary_even_when_role_inference_says_legal():
    source = _type_poster()
    body = next(e for e in source["elements"] if e["id"] == "body")
    body.update(
        content="This authored body remains part of the central message despite small type.",
        font_size=16,
        height=48,
    )
    result = analyze_proportions(source, deepcopy(source))
    assert "body" in result["composition_span"]["element_ids"]


def test_a_real_source_action_plate_is_not_discarded_as_corner_fine_print():
    source = _type_poster()
    source["elements"].extend(
        [
            {
                "id": "action-plate",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 800,
                "y": 940,
                "width": 180,
                "height": 60,
                "fill_color": "rgba(10,10,10,1)",
                "group_id": "action",
            },
            {
                "id": "action-label",
                "type": "text",
                "x": 815,
                "y": 950,
                "width": 150,
                "height": 45,
                "content": "Join",
                "font_size": 34,
                "font_family": "Solstice Sans",
                "color": "rgba(255,255,255,1)",
                "group_id": "action",
            },
        ]
    )
    result = analyze_proportions(source, deepcopy(source))
    assert {"action-plate", "action-label"} <= set(result["composition_span"]["element_ids"])


def test_actual_grand_distributed_bands_retain_more_portrait_span_after_resource_context():
    from engine.layout.source_guides import generate_source_guides
    from engine.resources import using_resources

    prepared = ROOT / "fixtures/open_posters/prepared/doory_grand-opening-flyer"
    source = json.loads((prepared / "source.json").read_text(encoding="utf-8"))
    with using_resources(
        assets_dir=prepared / "resources/assets", fonts_dir=prepared / "resources/fonts"
    ):
        graph = build_relationship_graph(source)
        states = {
            state.name: state.design
            for state in generate_source_guides(source, {"width": 1080, "height": 1920}, graph)
        }
    canonical = analyze_proportions(source, states["source-guides/source-fit"], graph)
    distributed = analyze_proportions(source, states["source-guides/bands-distributed"], graph)
    assert canonical["composition_span"]["active"] and distributed["composition_span"]["active"]
    assert (
        distributed["metrics"]["composition_span_retention"]
        > canonical["metrics"]["composition_span_retention"]
    )
    assert distributed["score"] == canonical["score"] == 1


def test_noncampaign_rows_can_become_columns_without_losing_authored_visual_weight():
    source = {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            dict(
                id="a",
                type="shape",
                shape_type="rectangle",
                x=50,
                y=50,
                width=100,
                height=100,
                fill_color="#123456",
            ),
            dict(
                id="b",
                type="shape",
                shape_type="rectangle",
                x=50,
                y=250,
                width=100,
                height=100,
                fill_color="#654321",
            ),
        ],
    }
    columns = deepcopy(source)
    columns["elements"][1].update(x=250, y=50)
    result = analyze_proportions(source, columns)
    assert result["score"] == 1
    assert result["composition_span"]["basis"] == "occupied-area"
    assert result["metrics"]["composition_span_retention"] == 1
    smaller = deepcopy(columns)
    for element in smaller["elements"]:
        for key in ("x", "y", "width", "height"):
            element[key] *= 0.5
    assert analyze_proportions(source, smaller)["metrics"]["composition_span_retention"] == 0.25
