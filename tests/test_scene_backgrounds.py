"""Source-backed crop viewports must fix real pillarboxing without stretching products."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract
from engine.eval.proportions import analyze_proportions
from engine.layout.backgrounds import generate_background_states
from engine.layout.reading_order import analyze_reading_order
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.resources import using_resources
from engine.scene import (
    analyze_scene,
    approved_cover,
    approved_support,
    background_coverage,
    support_retained,
)

PREPARED = (
    Path(__file__).resolve().parents[1] / "fixtures/open_posters/prepared/doory_rooftop-event-story"
)
GEOMETRY = {
    "x",
    "y",
    "width",
    "height",
    "font_size",
    "corner_radius",
    "stroke_width",
    "image_position",
}
TARGETS = [{"width": 1080, "height": 1080}, {"width": 1600, "height": 900}]


@pytest.fixture
def rooftop():
    source = json.loads((PREPARED / "source.json").read_text())
    with using_resources(PREPARED / "resources/assets", PREPARED / "resources/fonts"):
        yield source


def test_initial_scene_distinguishes_background_and_caption_support(rooftop):
    original = deepcopy(rooftop)
    scene = analyze_scene(rooftop)
    assert scene["model_applied"] is False
    assert [r["role"] for r in scene["images"]] == ["background", "background_support"]
    assert scene["images"][0]["allow_cover"]
    assert scene["images"][1]["allow_support_reflow"]
    assert not scene["images"][1]["allow_cover"]
    assert all(r["evidence"] for r in scene["images"])
    for row in scene["images"]:
        asset = next(e["asset_id"] for e in rooftop["elements"] if e["id"] == row["element_id"])
        assert (
            row["asset_sha256"]
            == hashlib.sha256((PREPARED / "resources/assets" / asset).read_bytes()).hexdigest()
        )
    assert rooftop == original
    scene["images"][0]["allow_cover"] = False
    assert analyze_scene(rooftop)["images"][0]["allow_cover"] is True


@pytest.mark.parametrize("role", ["product", "hero", "logo"])
def test_explicit_foreground_role_cannot_be_overridden_by_background_geometry(rooftop, role):
    source = deepcopy(rooftop)
    source["elements"][1]["role"] = role
    assert not any(r["allow_cover"] for r in analyze_scene(source)["images"])
    assert generate_background_states(source, TARGETS[0]) == []
    attempted = deepcopy(source)
    attempted["elements"][1]["fit"] = "cover"
    assert "frozen_field_changed" in {v.code for v in check_contract(source, attempted)}


@pytest.mark.parametrize("mutation", ["non_full_bleed", "unknown_art", "rotation", "group"])
def test_unproven_background_cannot_gain_cover_permission(rooftop, mutation):
    source = deepcopy(rooftop)
    photo = source["elements"][1]
    if mutation == "non_full_bleed":
        photo["width"] *= 0.8
    elif mutation == "unknown_art":
        photo.pop("source_art")
    elif mutation == "rotation":
        photo["rotation"] = 5
    else:
        photo["group_id"] = "foreground-composite"
    assert not any(r["allow_cover"] for r in analyze_scene(source)["images"])


@pytest.mark.parametrize("target", TARGETS)
def test_actual_rooftop_cover_candidates_fill_without_changing_identity(rooftop, target):
    before = deepcopy(rooftop)
    profile = analyze_scene(rooftop)
    graph = build_relationship_graph(rooftop)
    context = build_context(rooftop)
    states = generate_background_states(rooftop, target, profile)
    assert len(states) == 3
    for state in states:
        design = state.design
        assert check_contract(rooftop, design, target, context) == []
        assert background_coverage(rooftop, design, profile) == 1
        assert [e["id"] for e in design["elements"]] == [e["id"] for e in rooftop["elements"]]
        for old, new in zip(rooftop["elements"], design["elements"], strict=True):
            mutable = GEOMETRY | ({"fit"} if old["type"] == "image" else set())
            assert {k: v for k, v in old.items() if k not in mutable} == {
                k: v for k, v in new.items() if k not in mutable
            }
        images = [e for e in design["elements"] if e["type"] == "image"]
        assert approved_cover(profile, images[0], target)
        assert approved_support(profile, images[1], target)
        assert images[0]["fit"] == "cover"
        assert (images[0]["x"], images[0]["y"], images[0]["width"], images[0]["height"]) == (
            0,
            0,
            target["width"],
            target["height"],
        )
        assert support_retained(rooftop, design, profile, images[1]["id"])
        assert analyze_relationships(graph, design)["warnings"] == []
        assert analyze_proportions(rooftop, design, graph)["warnings"] == []
        assert analyze_reading_order(rooftop, design)["high_confidence_inversions"] == 0
        texts = [e for e in design["elements"] if e["type"] == "text"]
        body = next(e for e in texts if e["content"].startswith("Live music"))
        day = next(e for e in texts if e["content"] == "18")
        assert day["font_size"] / body["font_size"] == pytest.approx(32 / 30)
    assert rooftop == before


def test_cover_exception_does_not_authorize_stretching_or_partial_frames(rooftop):
    target = TARGETS[0]
    candidate = generate_background_states(rooftop, target)[0].design
    stretched = deepcopy(candidate)
    stretched["elements"][1]["fit"] = "fill"
    assert "aspect_drift" in {v.code for v in check_contract(rooftop, stretched, target)}
    partial = deepcopy(candidate)
    partial["elements"][1]["width"] *= 0.5
    codes = {v.code for v in check_contract(rooftop, partial, target)}
    assert {"aspect_drift", "frozen_field_changed"} <= codes


def test_moving_fade_away_from_caption_is_still_a_relationship_failure(rooftop):
    target = TARGETS[0]
    candidate = generate_background_states(rooftop, target)[0].design
    fade = candidate["elements"][2]
    fade["y"], fade["height"] = target["height"] - 20, 20
    profile = analyze_scene(rooftop)
    assert not support_retained(rooftop, candidate, profile, fade["id"])
    warnings = analyze_relationships(build_relationship_graph(rooftop), candidate)["warnings"]
    assert "overlay_attachment" in {w["code"] for w in warnings}


def test_unchanged_canvas_keeps_original_viewport(rooftop):
    assert generate_background_states(rooftop, rooftop["canvas"]) == []
    assert generate_background_states(rooftop, {"width": 540, "height": 960}) == []


def test_wide_crop_retains_source_headroom_and_default_square_crop(rooftop):
    from engine.scene import cover_position

    photo = analyze_scene(rooftop)["images"][0]
    assert 0.2 < photo["content_top_fraction"] < 0.7
    assert cover_position(photo, TARGETS[0]) == [0.5, 0.5]
    wide = cover_position(photo, TARGETS[1])
    assert wide[0] == 0.5 and 0 < wide[1] < 0.5
    source_w, source_h = photo["image_size"]
    scale = TARGETS[1]["width"] / source_w
    cropped_top = (source_h * scale - TARGETS[1]["height"]) * wide[1]
    onset = photo["content_top_fraction"] * source_h * scale - cropped_top
    assert onset == pytest.approx(TARGETS[1]["height"] * 0.08)


@pytest.mark.parametrize("position", [[0, float("nan")], [True, 0.5], [-1, 0.5], [0.5], "top"])
def test_bad_crop_position_cannot_pass_contract(rooftop, position):
    candidate = generate_background_states(rooftop, TARGETS[0])[0].design
    candidate["elements"][1]["image_position"] = position
    assert "invalid_image_position" in {v.code for v in check_contract(rooftop, candidate)}


def test_crop_position_does_not_authorize_foreground_crop(rooftop):
    source = deepcopy(rooftop)
    source["elements"][1]["role"] = "product"
    candidate = deepcopy(source)
    candidate["elements"][1]["image_position"] = [0.5, 0.25]
    assert "unapproved_image_position" in {v.code for v in check_contract(source, candidate)}


def test_renderer_crop_position_moves_viewport_without_stretching_or_changing_default(tmp_path):
    from PIL import Image, ImageChops

    from engine.image_viewport import positioned_image
    from renderer.converter.assets import _fit_image

    asset = Image.new("RGB", (80, 160))
    asset.putdata([(y, x, 20) for y in range(160) for x in range(80)])
    asset = asset.convert("RGBA")
    asset.save(tmp_path / "source.png")
    element = {"asset_id": "source.png", "width": 80, "height": 80, "fit": "cover"}
    centered = _fit_image(asset, 80, 80, "cover")
    explicit = positioned_image({**element, "image_position": [0.5, 0.5]}, tmp_path)
    upper = positioned_image({**element, "image_position": [0.5, 0.25]}, tmp_path)
    assert ImageChops.difference(centered, explicit).getbbox() is None
    assert ImageChops.difference(centered, asset.crop((0, 40, 80, 120))).getbbox() is None
    assert ImageChops.difference(upper, asset.crop((0, 20, 80, 100))).getbbox() is None


def test_dropping_authored_foreground_crop_position_is_rejected(rooftop):
    source = deepcopy(rooftop)
    source["elements"][1].update(role="product", fit="cover", image_position=[0.5, 0.25])
    candidate = deepcopy(source)
    candidate["elements"][1].pop("image_position")
    assert "unapproved_image_position" in {v.code for v in check_contract(source, candidate)}


@pytest.mark.parametrize("target", TARGETS)
def test_background_reflow_preserves_header_body_and_calendar_font_hierarchy(rooftop, target):
    from engine.eval.style import analyze_style

    originals = {e["id"]: e for e in rooftop["elements"]}
    for state in generate_background_states(rooftop, target):
        scales = [
            e["font_size"] / originals[e["id"]]["font_size"]
            for e in state.design["elements"]
            if e["type"] == "text"
        ]
        assert max(scales) / min(scales) == pytest.approx(1, abs=1e-9)
        assert analyze_style(rooftop, state.design)["warnings"] == []
