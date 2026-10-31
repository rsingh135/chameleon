"""The graph verifier distinguishes invalid output, native proof and proxies."""

import json
import shutil
from copy import deepcopy

import pytest

from engine.resources import DEFAULT_ASSETS_DIR, DEFAULT_FONTS_DIR
from experiments.agentic_graph import verification


@pytest.fixture
def source():
    return {
        "canvas": {"width": 320, "height": 180},
        "elements": [
            {
                "id": "paper",
                "type": "shape",
                "x": 0,
                "y": 0,
                "width": 320,
                "height": 180,
                "shape_type": "rectangle",
                "fill_color": "rgba(255,255,255,1)",
                "z_index": 0,
            },
            {
                "id": "headline",
                "type": "text",
                "x": 20,
                "y": 20,
                "width": 280,
                "height": 40,
                "content": "Keep the light",
                "font_family": "Solstice Sans",
                "font_weight": "bold",
                "font_size": 24,
                "color": "rgba(0,0,0,1)",
                "group_id": "message",
                "z_index": 1,
            },
            {
                "id": "copy",
                "type": "text",
                "x": 20,
                "y": 100,
                "width": 260,
                "height": 40,
                "content": "Original words stay.",
                "font_family": "Solstice Sans",
                "font_size": 14,
                "color": "rgba(30,30,30,1)",
                "group_id": "message",
                "z_index": 2,
            },
        ],
    }


def codes(result):
    return {row["code"] for row in result["failures"]}


def test_real_native_evidence_is_json_and_does_not_claim_visual_approval(source):
    original = deepcopy(source)
    result = verification.verify_design(source, deepcopy(source), source["canvas"])
    assert result["status"] == "passed" and result["hard_pass"]
    assert result["render"]["width"] == 320 and result["render"]["height"] == 180
    assert result["render"]["source_png_sha256"] == result["render"]["png_sha256"]
    assert len(result["render"]["png_sha256"]) == 64
    assert result["quality"]["status"] == "diagnostic_only"
    assert result["acceptance_scope"] == "hard_contract_and_native_render_only"
    assert "fundamental_pass" not in result
    assert result["diagnostics"]["metrics"]["raw_collisions"] == 0
    assert result["diagnostics"]["metrics"]["raw_min_contrast"] > 4
    assert source == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "Invented copy"),
        ("font_family", "Solstice Serif"),
        ("color", "rgba(250,0,0,1)"),
        ("group_id", "invented-group"),
    ],
)
def test_fixed_identity_changes_are_actionable_failures(source, field, value):
    candidate = deepcopy(source)
    candidate["elements"][1][field] = value
    result = verification.verify_design(source, candidate, source["canvas"])
    assert not result["hard_pass"] and result["status"] == "failed"
    assert "frozen_field_changed" in codes(result)
    assert any(row["element_ids"] == ["headline"] for row in result["failures"])
    assert result["render"]["status"] == "skipped"


def test_implicit_opacity_cannot_hide_source_content(source):
    candidate = deepcopy(source)
    candidate["elements"][1]["opacity"] = 0
    result = verification.verify_design(source, candidate, source["canvas"])
    assert "frozen_field_added" in codes(result)
    assert result["failures"][0]["evidence"]["field"] == "opacity"


@pytest.mark.parametrize(
    "change,expected",
    [
        ("drop", "id_missing"),
        ("duplicate", "id_duplicate"),
        ("add", "id_unexpected"),
    ],
)
def test_element_membership_is_fixed(source, change, expected):
    candidate = deepcopy(source)
    if change == "drop":
        candidate["elements"].pop()
    else:
        extra = deepcopy(candidate["elements"][-1])
        if change == "add":
            extra["id"] = "new-object"
        candidate["elements"].append(extra)
    assert expected in codes(verification.verify_design(source, candidate, source["canvas"]))


def test_distorted_noncover_image_fails_even_when_it_still_fits(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    source = {
        "canvas": {"width": 200, "height": 100},
        "elements": [
            {
                "id": "product",
                "type": "image",
                "x": 20,
                "y": 10,
                "width": 20,
                "height": 80,
                "asset_id": "intentional-missing.png",
            }
        ],
    }
    candidate = deepcopy(source)
    candidate["elements"][0]["width"] = 36
    result = verification.verify_design(
        source, candidate, source["canvas"], assets_dir=assets, fonts_dir=DEFAULT_FONTS_DIR
    )
    assert "aspect_drift" in codes(result)


def test_authored_source_bleed_is_still_rejected_by_strict_stage_one(source):
    source["elements"][0].update(x=-4, width=328)
    result = verification.verify_design(source, deepcopy(source), source["canvas"])
    assert "strict_bounds" in codes(result)
    assert any(row["element_ids"] == ["paper"] for row in result["failures"])


def test_strict_bounds_measures_rotated_extents():
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "rotated",
                "type": "shape",
                "x": 5,
                "y": 5,
                "width": 90,
                "height": 90,
                "rotation": 45,
                "shape_type": "rectangle",
                "fill_color": "rgba(30,30,30,1)",
            }
        ],
    }
    result = verification.verify_design(source, deepcopy(source), source["canvas"])
    assert "strict_bounds" in codes(result)


def test_real_glyph_measurement_rejects_word_and_line_overflow(source):
    candidate = deepcopy(source)
    candidate["elements"][1]["width"] = 8
    result = verification.verify_design(source, candidate, source["canvas"])
    assert "text_overflow_width" in codes(result)
    assert "text_overflow_height" in codes(result)


def test_available_json_font_name_cannot_silently_substitute(source):
    source["elements"][1]["font_family"] = "Unavailable Source Face"
    result = verification.verify_design(source, deepcopy(source), source["canvas"])
    assert codes(result) == {"font_face_unavailable"}
    assert not result["hard_pass"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["elements"][1].pop("font_family"),
        lambda d: d["elements"][1].update(type=[]),
        lambda d: d["elements"][1].update(x=float("nan")),
        lambda d: d["elements"][1].update(x=10**400),
    ],
)
def test_malformed_candidate_is_a_report_not_a_crash(source, mutation):
    candidate = deepcopy(source)
    mutation(candidate)
    result = verification.verify_design(source, candidate, source["canvas"])
    assert result["status"] == "failed" and not result["hard_pass"]
    assert result["failures"]
    json.dumps(result, allow_nan=False)


def test_unavailable_real_renderer_never_becomes_a_pass(source, monkeypatch):
    def unavailable(_design):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(verification, "_native_png", unavailable)
    result = verification.verify_design(source, deepcopy(source), source["canvas"])
    assert result["status"] == "unverified" and not result["hard_pass"]
    assert "native_render" in codes(result)
    assert result["render"]["status"] == "unverified"


def test_tiny_type_and_new_text_collision_are_diagnostics_not_automatic_acceptance(source):
    candidate = deepcopy(source)
    candidate["elements"][2].update(x=20, y=20)
    result = verification.verify_design(source, candidate, source["canvas"])
    assert result["hard_pass"]
    assert result["diagnostics"]["metrics"]["raw_collisions"] > 0
    collision = next(
        row for row in result["quality"]["checks"] if row["code"] == "unwanted_collisions"
    )
    assert collision["status"] == "warning"
    assert set(collision["element_ids"]) == {"headline", "copy"}
    candidate["elements"][2].update(y=100, font_size=4)
    result = verification.verify_design(source, candidate, source["canvas"])
    assert result["hard_pass"]
    readable = next(row for row in result["quality"]["checks"] if row["code"] == "text_readability")
    assert readable["status"] == "warning" and readable["element_ids"] == ["copy"]


def test_deliberately_missing_asset_keeps_native_placeholder(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    source = {
        "canvas": {"width": 100, "height": 80},
        "elements": [
            {
                "id": "artwork",
                "type": "image",
                "x": 10,
                "y": 10,
                "width": 70,
                "height": 60,
                "asset_id": "absent-on-purpose.png",
            }
        ],
    }
    result = verification.verify_design(
        source, deepcopy(source), source["canvas"], assets_dir=assets, fonts_dir=DEFAULT_FONTS_DIR
    )
    assert result["hard_pass"] and result["render"]["status"] == "passed"
    assert any("missing" in warning.lower() for warning in result["render"]["warnings"])
    assert not list(assets.iterdir())


def test_resource_byte_change_is_rejected_even_if_pixels_are_unchanged(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    asset = assets / "supplied.png"
    shutil.copyfile(next(DEFAULT_ASSETS_DIR.glob("*.png")), asset)
    source = {
        "canvas": {"width": 80, "height": 80},
        "elements": [
            {
                "id": "artwork",
                "type": "image",
                "x": 10,
                "y": 10,
                "width": 60,
                "height": 60,
                "asset_id": asset.name,
            }
        ],
    }
    native = verification._native_png
    calls = 0

    def render(design):
        nonlocal calls
        png = native(design)
        calls += 1
        if calls == 1:
            asset.write_bytes(asset.read_bytes() + b"\n")
        return png

    monkeypatch.setattr(verification, "_native_png", render)
    result = verification.verify_design(
        source, deepcopy(source), source["canvas"], assets_dir=assets, fonts_dir=DEFAULT_FONTS_DIR
    )
    assert result["status"] == "failed" and "resource_mutation" in codes(result)
