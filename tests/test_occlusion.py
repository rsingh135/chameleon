"""Visible source-layer occlusion never waives unknown or textual collisions."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from cr_renderer import text_utils
from PIL import Image, ImageStat

from engine.eval import build_context
from engine.eval.metrics import compute_metrics
from engine.eval.occlusion import analyze_occlusion, infer_occlusion_profile
from engine.eval.render import render_image
from engine.layout.source_guides import generate_source_guides
from engine.resources import get_renderer, using_resources

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "brand_kit/fonts"


def _text(eid, x, y, content="SALE", z=3):
    return dict(
        id=eid,
        type="text",
        x=x,
        y=y,
        width=120,
        height=50,
        content=content,
        font_family="Solstice Sans",
        font_size=24,
        line_height=1.2,
        color="rgba(255,255,255,1)",
        z_index=z,
    )


@pytest.fixture
def case(tmp_path):
    Image.new("RGBA", (160, 240), (220, 40, 40, 255)).save(tmp_path / "photo.png")
    Image.new("RGBA", (60, 60), (30, 180, 60, 255)).save(tmp_path / "unknown.png")
    source = {
        "canvas": {"width": 400, "height": 300},
        "elements": [
            dict(
                id="photo",
                type="image",
                x=220,
                y=0,
                width=160,
                height=240,
                asset_id="photo.png",
                fit="fill",
                z_index=1,
                source_art={"kind": "photo"},
            ),
            dict(
                id="panel",
                type="shape",
                shape_type="rectangle",
                x=0,
                y=200,
                width=400,
                height=100,
                fill_color="rgba(0,0,80,1)",
                z_index=2,
            ),
            _text("label", 20, 220),
        ],
    }
    return source, tmp_path


@pytest.mark.parametrize(
    "target,name",
    [
        ({"width": 1600, "height": 900}, "source-guides/media-columns-0.62"),
        ({"width": 1080, "height": 1920}, "source-guides/media-stacked"),
    ],
)
def test_real_sale_opaque_source_panels_explain_actual_image_intersections(target, name):
    folder = ROOT / "fixtures/open_posters/prepared/doory_bold-sale-announcement"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        state = next(s for s in generate_source_guides(source, target) if s.name == name)
        candidate = state.design
        frozen = deepcopy(candidate)
        context = build_context(source)
        raw_before = compute_metrics(source, candidate, context)["raw_collisions"]
        pixels_before = render_image(candidate, max_size=max(target.values())).tobytes()
        vendor = text_utils.make_text_blob
        profile = infer_occlusion_profile(source)
        result = analyze_occlusion(source, candidate, profile)
        assert result["raw_pair_count"] == raw_before > 0
        assert result["visible_pair_count"] == 0
        assert len(result["source_supported_discounts"]) == result["raw_pair_count"]
        assert result["visibility_retention"] >= 0.85
        assert result["coverage"]["complete"] and not result["warnings"]
        assert all(
            p["opaque_support_coverage"] >= 0.985 and p["visible_alpha_overlap"] <= 0.02
            for p in result["source_supported_discounts"]
        )
        assert compute_metrics(source, candidate, context)["raw_collisions"] == raw_before
        assert render_image(candidate, max_size=max(target.values())).tobytes() == pixels_before
        assert text_utils.make_text_blob is vendor
    assert source == before and candidate == frozen


def test_newly_hidden_photo_blocks_all_discounts(case):
    source, assets = case
    candidate = deepcopy(source)
    candidate["elements"][0]["y"] = 60
    profile = infer_occlusion_profile(source, assets, FONTS)
    result = analyze_occlusion(source, candidate, profile)
    assert result["raw_pair_count"] > 0
    assert result["visible_pair_count"] == result["raw_pair_count"]
    assert result["visibility_retention"] < 0.85
    assert any(w["code"] == "photo_visibility_lost" for w in result["warnings"])
    assert not result["source_supported_discounts"]


def test_unknown_image_occlusion_is_never_excused_by_z_order_alone(case):
    source, assets = case
    source["elements"].append(
        dict(
            id="unknown",
            type="image",
            x=0,
            y=0,
            width=60,
            height=60,
            fit="fill",
            asset_id="unknown.png",
            z_index=4,
        )
    )
    candidate = deepcopy(source)
    candidate["elements"][-1].update(x=250, y=80)
    result = analyze_occlusion(source, candidate, infer_occlusion_profile(source, assets, FONTS))
    assert result["visibility_retention"] >= 0.85
    assert result["raw_pair_count"] == result["visible_pair_count"] > 0
    assert not result["source_supported_discounts"]


def test_actual_glyph_moved_outside_its_source_support_remains_a_collision(case):
    source, assets = case
    candidate = deepcopy(source)
    candidate["elements"][-1].update(x=240, y=80)
    result = analyze_occlusion(source, candidate, infer_occlusion_profile(source, assets, FONTS))
    assert result["raw_pair_count"] == result["visible_pair_count"] > 0
    assert any(
        p["status"] == "retained_foreground_outside_source_plate" for p in result["bounded_pairs"]
    )
    assert not result["source_supported_discounts"]


def test_text_text_collisions_are_unchanged_even_with_source_plates(case):
    source, assets = case
    source["elements"].append(_text("other", 20, 20, "OTHER", 4))
    candidate = deepcopy(source)
    candidate["elements"][-1]["y"] = 220
    context = build_context(source, assets, FONTS)
    raw = compute_metrics(source, candidate, context)["raw_collisions"]
    result = analyze_occlusion(source, candidate, infer_occlusion_profile(source, assets, FONTS))
    assert result["raw_pair_count"] == raw == result["visible_pair_count"] > 0
    assert any(p["status"] == "retained_nonimage_pair" for p in result["bounded_pairs"])


def test_masks_preserve_fit_rotation_and_are_bounded(case):
    source, assets = case
    source["canvas"] = {"width": 600, "height": 500}
    source["elements"][0].update(x=200, y=150, width=200, height=100, rotation=35, fit="contain")
    Image.new("RGBA", (100, 100), (220, 40, 40, 255)).save(assets / "photo.png")
    source["elements"][1].update(y=220, width=600, height=280)
    source["elements"][-1]["y"] = 300
    contained = infer_occlusion_profile(source, assets, FONTS, max_size=9000)
    cover = deepcopy(source)
    cover["elements"][0]["fit"] = "cover"
    covered = infer_occlusion_profile(cover, assets, FONTS, max_size=9000)

    def mass(profile):
        mask = dict(profile.source_masks)["photo"]
        assert max(mask.width, mask.height) <= 512
        return ImageStat.Stat(mask.image()).sum[0]

    assert contained.max_size == covered.max_size == 512
    assert mass(covered) / mass(contained) == pytest.approx(2, rel=0.04)
    unrotated = deepcopy(source)
    unrotated["elements"][0]["rotation"] = 0
    flat = infer_occlusion_profile(unrotated, assets, FONTS, max_size=9000)
    assert dict(flat.source_masks)["photo"].alpha != dict(contained.source_masks)["photo"].alpha


def test_profiles_are_cached_frozen_and_refuse_stale_sources_or_assets(case):
    source, assets = case
    before = deepcopy(source)
    default = get_renderer()
    profile = infer_occlusion_profile(source, assets, FONTS)
    assert infer_occlusion_profile(source, assets, FONTS) is profile
    with pytest.raises(FrozenInstanceError):
        profile.max_size = 123
    changed = deepcopy(source)
    changed["elements"][-1]["content"] = "CHANGED"
    with pytest.raises(ValueError, match="source"):
        analyze_occlusion(changed, changed, profile)
    Image.new("RGBA", (160, 240), (220, 40, 40, 120)).save(assets / "photo.png")
    with pytest.raises(ValueError, match="resources changed"):
        analyze_occlusion(source, source, profile)
    assert get_renderer() is default and source == before


def test_unsafe_resource_paths_are_rejected_before_any_asset_read(case):
    source, assets = case
    source["elements"][0]["asset_id"] = "../outside.png"
    with pytest.raises(ValueError, match="inside"):
        infer_occlusion_profile(source, assets, FONTS)


@pytest.mark.parametrize("complete,raw,expected", [(True, 3, 0), (False, 3, 3), (True, 4, 4)])
def test_only_complete_matching_evidence_can_discount_official_collisions(complete, raw, expected):
    from engine.eval.occlusion import collision_safety

    report = dict(
        coverage={"complete": complete},
        raw_pair_count=3,
        visible_pair_count=0,
        visibility_retention=1,
    )
    assert collision_safety(report, raw) == (expected, True)


def test_photo_visibility_guard_applies_even_without_any_collision():
    from engine.eval.occlusion import collision_safety

    report = dict(
        coverage={"complete": True},
        raw_pair_count=0,
        visible_pair_count=0,
        visibility_retention=0.4,
    )
    assert collision_safety(report, 0) == (0, False)


@pytest.mark.parametrize(
    "target,name",
    [
        ({"width": 1600, "height": 900}, "source-guides/media-columns-0.62"),
        ({"width": 1080, "height": 1920}, "source-guides/media-stacked"),
    ],
)
def test_selector_accepts_proven_sale_layering_over_compact_source_fit(target, name):
    from engine.resize import _select_candidate

    folder = ROOT / "fixtures/open_posters/prepared/doory_bold-sale-announcement"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        states = generate_source_guides(source, target)
        compact = next(s for s in states if s.name == "source-guides/source-fit")
        reflow = next(s for s in states if s.name == name)
        axes = {
            json.dumps(s.design, sort_keys=True): dict(s.cluster_axes) for s in (compact, reflow)
        }
        assert (
            _select_candidate(source, target, [compact.design, reflow.design], axes)
            == reflow.design
        )
