"""Source-backed crop viewports must fix real pillarboxing without stretching products."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract
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
GEOMETRY = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}
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
