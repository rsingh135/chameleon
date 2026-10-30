"""Adaptive frame permission requires intact, source-corroborated support."""

import json
import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from engine.eval.proportions import analyze_proportions
from engine.layout.source_containers import analyze_source_containers, infer_source_containers
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def panels():
    elements = []
    for column in range(2):
        x = column * 520
        elements.append(
            dict(
                id=f"opaque-frame-{column}",
                type="shape",
                shape_type="rectangle",
                x=x,
                y=0,
                width=400,
                height=600,
                fill_color="#eeeeee",
                corner_radius=10,
                stroke_width=2,
                stroke_color="#222222",
                z_index=0,
            )
        )
        for index in range(4):
            elements.append(
                dict(
                    id=f"opaque-child-{column}-{index}",
                    type="text",
                    content=f"Item {index + 1}",
                    x=x + 30,
                    y=50 + index * 100,
                    width=300,
                    height=40,
                    font_size=24,
                    font_family="Solstice Sans",
                    color="#222222",
                    z_index=1,
                )
            )
    return {"canvas": {"width": 1000, "height": 700}, "elements": elements}


def _legal_native_adaptation(source):
    candidate = deepcopy(source)
    for element in candidate["elements"]:
        if element["id"] == "opaque-frame-0":
            element.update(width=450, height=500)
        elif element["id"] == "opaque-frame-1":
            element.update(x=550, width=450, height=500)
        elif element["id"].startswith("opaque-child-1"):
            element["x"] += 30
    return candidate


def test_native_frame_can_adapt_without_exempting_any_child_geometry(panels):
    source = panels
    before = deepcopy(source)
    profile = infer_source_containers(source)
    assert len(profile.containers) == 2
    candidate = _legal_native_adaptation(source)
    analysis = analyze_source_containers(source, candidate, profile)
    assert analysis["accepted"]
    assert set(analysis["accepted_frame_ids"]) == {"opaque-frame-0", "opaque-frame-1"}
    result = analyze_proportions(source, candidate)
    assert result["metrics"]["rigid_group_error"] == 0
    assert result["metrics"]["adaptive_frame_count"] == 2
    assert result["containers"] == analysis
    assert source == before
    with pytest.raises(FrozenInstanceError):
        profile.containers[0].adaptive = False


@pytest.mark.parametrize("kind", ["frame", "child", "padding"])
def test_detached_frame_escaped_child_or_lost_source_padding_revokes_permission(panels, kind):
    candidate = _legal_native_adaptation(panels)
    if kind == "frame":
        candidate["elements"][0]["x"] += 200
    elif kind == "child":
        candidate["elements"][1]["x"] += 450
    else:
        candidate["elements"][0]["x"] = 30
    result = analyze_source_containers(panels, candidate)
    assert not result["accepted"]
    assert result["accepted_frame_ids"] == []
    assert "source_container_detached" in {w["code"] for w in result["warnings"]}
    assert analyze_proportions(panels, candidate)["metrics"]["rigid_group_error"] > 0


@pytest.mark.parametrize("field", ["corner_radius", "stroke_width"])
def test_frame_geometry_follows_children_instead_of_stretched_frame_axis(panels, field):
    candidate = _legal_native_adaptation(panels)
    candidate["elements"][0][field] *= 2
    result = analyze_source_containers(panels, candidate)
    assert result["accepted_frame_ids"] == []
    assert result["metrics"]["max_frame_style_log_error"] == pytest.approx(math.log(2))
    assert "source_container_unsupported_deformation" in {w["code"] for w in result["warnings"]}


@pytest.mark.parametrize("value", [float("nan"), -1, "invalid"])
def test_invalid_native_shape_geometry_cannot_gain_an_adaptive_exception(panels, value):
    candidate = _legal_native_adaptation(panels)
    candidate["elements"][0]["corner_radius"] = value
    result = analyze_source_containers(panels, candidate)
    assert not result["accepted"]
    assert result["accepted_frame_ids"] == []


def test_new_cross_container_overlap_is_not_legal_enclosure(panels):
    candidate = deepcopy(panels)
    for element in candidate["elements"]:
        if element["id"] == "opaque-frame-1" or element["id"].startswith("opaque-child-1"):
            element["x"] -= 200
    result = analyze_source_containers(panels, candidate)
    assert not result["accepted"]
    assert result["metrics"]["max_new_container_overlap"] == pytest.approx(0.2)
    assert "source_container_detached" in {w["code"] for w in result["warnings"]}


@pytest.mark.parametrize("change", [{"y": 140}, {"y": 40, "height": 400}])
def test_nested_color_section_cannot_shift_to_other_leaves_inside_one_text_group(panels, change):
    for element in panels["elements"]:
        if element["type"] == "text":
            element["group_id"] = "paragraph-" + element["id"].split("-")[2]
    section = dict(
        id="opaque-section",
        type="shape",
        shape_type="rectangle",
        fill_color="#448822",
        x=20,
        y=240,
        width=360,
        height=70,
        z_index=0.5,
    )
    panels["elements"].append(section)
    profile = infer_source_containers(panels)
    assert "opaque-section" in profile.containers[0].frame_ids
    assert analyze_source_containers(panels, panels, profile)["accepted"]
    candidate = deepcopy(panels)
    candidate["elements"][-1].update(change)
    result = analyze_source_containers(panels, candidate, profile)
    assert not result["accepted"]
    assert result["accepted_frame_ids"] == []
    assert "source_container_detached" in {w["code"] for w in result["warnings"]}


@pytest.mark.parametrize("change", ["missing", "duplicate", "copy", "group", "paint"])
def test_incomplete_or_changed_membership_and_identity_cannot_claim_frame_permission(
    panels, change
):
    candidate = _legal_native_adaptation(panels)
    if change == "missing":
        candidate["elements"].pop()
    elif change == "duplicate":
        candidate["elements"].append(deepcopy(candidate["elements"][-1]))
    elif change == "copy":
        candidate["elements"][1]["content"] = "Different text"
    elif change == "group":
        candidate["elements"][1]["group_id"] = "different-owner"
    else:
        candidate["elements"][0]["fill_color"] = "#111111"
    result = analyze_source_containers(panels, candidate)
    assert not result["accepted"]
    assert result["accepted_frame_ids"] == []
    assert "source_container_unsupported_deformation" in {w["code"] for w in result["warnings"]}


def test_source_mutation_cannot_reuse_a_frozen_container_profile(panels):
    profile = infer_source_containers(panels)
    changed = deepcopy(panels)
    changed["elements"][0]["x"] += 1
    result = analyze_source_containers(changed, changed, profile)
    assert not result["accepted"]
    assert result["accepted_frame_ids"] == []


def test_actual_shape_label_group_remains_rigid_inside_an_adaptive_container(panels):
    label = panels["elements"][1]
    label["group_id"] = "control"
    support = dict(
        id="opaque-control",
        type="shape",
        shape_type="rectangle",
        group_id="control",
        x=label["x"] - 10,
        y=label["y"] - 10,
        width=320,
        height=60,
        fill_color="#ffffff",
        z_index=0.5,
    )
    panels["elements"].append(support)
    candidate = _legal_native_adaptation(panels)
    next(e for e in candidate["elements"] if e["id"] == label["id"])["font_size"] *= 1.5
    result = analyze_proportions(panels, candidate)
    assert result["metrics"]["rigid_group_error"] > math.log(1.05)
    assert "rigid_group_deformation" in {w["code"] for w in result["warnings"]}


def _vector_panels(source, directory, *, patterned):
    image = Image.new("RGBA", (100, 150), "#eeeeee")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 0, 99, 149), outline="#222222", width=2)
    if patterned:
        drawing.line((20, 20, 80, 130), fill="#aa0000", width=8)
    image.save(directory / "vector.png")
    output = deepcopy(source)
    for element in output["elements"]:
        if element["type"] == "shape":
            for key in (
                "shape_type",
                "fill_color",
                "corner_radius",
                "stroke_width",
                "stroke_color",
            ):
                element.pop(key, None)
            element.update(
                type="image", asset_id="vector.png", fit="fill", source_art={"kind": "svg-nontext"}
            )
    return output


def test_flat_border_raster_still_cannot_violate_immutable_image_aspect(panels, tmp_path):
    source = _vector_panels(panels, tmp_path, patterned=False)
    with using_resources(assets_dir=tmp_path):
        assert len(infer_source_containers(source).containers) == 2
        candidate = deepcopy(source)
        candidate["elements"][0]["width"] *= 1.1
        result = analyze_source_containers(source, candidate)
        assert not result["accepted"]
        assert result["accepted_frame_ids"] == []
        assert "source_container_unsupported_deformation" in {w["code"] for w in result["warnings"]}
        assert analyze_proportions(source, candidate)["metrics"]["rigid_group_error"] > 0


def test_arbitrary_vector_art_lookalike_receives_no_adaptive_permission(panels, tmp_path):
    source = _vector_panels(panels, tmp_path, patterned=True)
    with using_resources(assets_dir=tmp_path):
        assert infer_source_containers(source).containers == ()
        candidate = deepcopy(source)
        candidate["elements"][0]["width"] *= 1.2
        result = analyze_proportions(source, candidate)
        assert result["metrics"]["adaptive_frame_count"] == 0
        assert result["metrics"]["rigid_group_error"] > 0


@pytest.mark.parametrize("fill", ["#11223380", "#1234", "rgba(0,0,0,0.8)"])
def test_translucent_native_shape_is_not_a_source_container(panels, fill):
    for element in panels["elements"]:
        if element["type"] == "shape":
            element["fill_color"] = fill
    assert infer_source_containers(panels).containers == ()


@pytest.mark.parametrize("case", ["opensuse_trifold_front", "opensuse_cheatsheet"])
def test_genuine_reviewed_panel_states_keep_support_and_only_legal_frames_adapt(case):
    from engine.layout.source_guides import generate_source_guides

    directory = ROOT / "fixtures/open_posters/prepared" / case
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        original = analyze_source_containers(source, source)
        assert original["accepted"]
        assert original["metrics"]["accepted_frame_count"] > 0
        states = [
            s
            for s in generate_source_guides(source, {"width": 1600, "height": 900})
            if s.name.startswith("source-guides/panels")
        ]
        assert states
        accepted = [analyze_proportions(source, state.design) for state in states]
        assert any(r["containers"]["accepted"] for r in accepted)
        assert any(r["metrics"]["adaptive_frame_count"] > 0 for r in accepted)
    assert source == before
