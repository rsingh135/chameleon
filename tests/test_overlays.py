"""Only source-supported translucent support layers become image attachments."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from engine.layout.overlays import infer_source_overlays


@pytest.fixture
def scene(tmp_path):
    Image.new("RGBA", (64, 64), "#885533").save(tmp_path / "photo.png")
    fade = Image.new("RGBA", (64, 64))
    fade.putdata([(0, 0, 0, round(255 * y / 63)) for y in range(64) for _ in range(64)])
    fade.save(tmp_path / "fade.png")
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "photo",
                "type": "image",
                "asset_id": "photo.png",
                "fit": "fill",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 100,
                "z_index": 0,
            },
            {
                "id": "fade",
                "type": "image",
                "asset_id": "fade.png",
                "fit": "fill",
                "x": 0,
                "y": 20,
                "width": 100,
                "height": 80,
                "z_index": 1,
            },
            {
                "id": "caption",
                "type": "text",
                "content": "Caption",
                "x": 10,
                "y": 80,
                "width": 80,
                "height": 15,
                "z_index": 2,
            },
        ],
    }
    return tmp_path, source


def test_source_pixels_attach_fade_to_opaque_underlay_without_mutating_source(scene):
    assets, source = scene
    original = deepcopy(source)
    attachments = infer_source_overlays(source, assets)
    assert source == original
    assert len(attachments) == 1
    attachment = attachments[0]
    assert (attachment.overlay_id, attachment.underlay_id) == ("fade", "photo")
    assert attachment.opaque_visible_ratio < 0.05
    assert attachment.underlay_opaque_ratio == 1
    assert attachment.relative_frame == (0, 0.2, 1, 0.8)
    assert infer_source_overlays(source, assets) == attachments


def test_opaque_vector_product_art_is_never_removed_as_overlay(scene):
    assets, source = scene
    source["elements"][1]["source_art"] = {"kind": "svg-nontext"}
    Image.new("RGBA", (64, 64), "#000000").save(assets / "fade.png")
    assert infer_source_overlays(source, assets) == ()


def test_colorful_translucent_artwork_is_not_a_modulation_layer(scene):
    assets, source = scene
    image = Image.new("RGBA", (64, 64))
    image.putdata([(x * 4, y * 4, 128, round(255 * y / 63)) for y in range(64) for x in range(64)])
    image.save(assets / "fade.png")
    assert infer_source_overlays(source, assets) == ()


def test_flat_monochrome_translucent_product_is_not_a_fade(scene):
    assets, source = scene
    Image.new("RGBA", (64, 64), (0, 0, 0, 120)).save(assets / "fade.png")
    assert infer_source_overlays(source, assets) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"x": -1},
        {"y": 21},
        {"rotation": 10},
        {"z_index": -1},
        {"z_index": 3},
        {"asset_id": "missing.png"},
        {"asset_id": "../outside.png"},
    ],
)
def test_unproven_geometry_resources_or_paint_order_declines_inference(scene, change):
    assets, source = scene
    source["elements"][1].update(change)
    assert infer_source_overlays(source, assets) == ()


def test_actual_alpha_hole_beneath_fade_declines_attachment(scene):
    assets, source = scene
    photo = Image.open(assets / "photo.png")
    photo.paste((0, 0, 0, 0), (10, 30, 30, 50))
    photo.save(assets / "photo.png")
    assert infer_source_overlays(source, assets) == ()


def test_intervening_partial_photo_requires_more_than_one_owner(scene):
    assets, source = scene
    source["elements"].insert(
        1, {**source["elements"][0], "id": "second-photo", "width": 50, "z_index": 0.5}
    )
    assert infer_source_overlays(source, assets) == ()


def test_equal_frame_translucent_image_is_not_guessed_as_smaller_overlay(scene):
    assets, source = scene
    source["elements"][1].update(y=0, height=100)
    assert infer_source_overlays(source, assets) == ()


def test_real_rooftop_source_fade_has_exactly_one_physical_underlay():
    root = (
        Path(__file__).resolve().parents[1]
        / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    )
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    attachments = infer_source_overlays(source, root / "resources/assets")
    assert len(attachments) == 1
    by_id = {element["id"]: element for element in source["elements"]}
    assert by_id[attachments[0].overlay_id]["source_art"]["kind"] == "svg-nontext"
    assert by_id[attachments[0].underlay_id]["source_art"]["kind"] == "masked-photo"
    assert attachments[0].opaque_visible_ratio < 0.05


def test_relationships_keep_photo_overlay_attached_even_when_photo_is_background():
    from engine.layout.relationships import analyze_relationships, build_relationship_graph
    from engine.resources import using_resources

    root = (
        Path(__file__).resolve().parents[1]
        / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    )
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    with using_resources(root / "resources/assets", root / "resources/fonts"):
        graph = build_relationship_graph(source)
        attachment = graph.overlays[0]
        for cluster in graph.clusters:
            if cluster.kind == "campaign":
                assert attachment.overlay_id not in graph.units[cluster.primary_unit].element_ids
        candidate = deepcopy(source)
        candidate["canvas"] = {key: value * 0.75 for key, value in source["canvas"].items()}
        for element in candidate["elements"]:
            for key in ("x", "y", "width", "height", "font_size"):
                if key in element:
                    element[key] *= 0.75
        good = analyze_relationships(graph, candidate)
        assert good["metrics"]["overlay_attachment"] == 1
        next(e for e in candidate["elements"] if e["id"] == attachment.overlay_id)["x"] += 120
        bad = analyze_relationships(graph, candidate)
        warning = next(w for w in bad["warnings"] if w["code"] == "overlay_attachment")
        assert warning["confidence"] >= 0.8
        assert warning["element_ids"] == [attachment.underlay_id, attachment.overlay_id]
