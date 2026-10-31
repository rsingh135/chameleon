"""Isolated native alpha measures paint, while fallbacks remain explicit."""

import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from engine import semantic_ink
from engine.resources import get_renderer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_fonts(tmp_path):
    entries = []
    for name, weight in (
        ("SolsticeSans-Regular.ttf", "regular"),
        ("SolsticeSans-Bold.ttf", "bold"),
    ):
        (tmp_path / name).write_bytes((ROOT / "brand_kit/fonts" / name).read_bytes())
        entries.append(
            {"font_family": "ExactSans", "file": name, "weight": weight, "style": "normal"}
        )
    (tmp_path / "font-manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


def _text(**changes):
    return {
        "id": "synthetic-footer",
        "type": "text",
        "x": 100.25,
        "y": 240.5,
        "width": 500,
        "height": 160,
        "content": "www.example.org",
        "font_family": "ExactSans",
        "font_size": 40,
        "font_weight": "regular",
        "font_style": "normal",
        "text_align": "left",
        "line_height": 1.0,
        "color": "#ffffff",
        **changes,
    }


def test_native_footer_ink_is_not_its_padded_frame_and_input_stays_immutable(native_fonts):
    element = _text(style_runs=[{"start": 0, "end": 3, "font_weight": "bold", "color": "#ff0000"}])
    before = deepcopy(element)
    result = semantic_ink.element_ink_bounds(element, fonts_dir=native_fonts)
    assert result["provenance"] == "native_alpha", result
    assert result["raster_scale"] == result["confidence"] == 1.0
    x, y, width, height = result["box"]
    assert x >= element["x"] - 1 and y > element["y"]
    assert 0 < width < element["width"] and 0 < height < element["height"] / 2
    assert element == before


def test_native_alpha_translation_maps_back_to_source_coordinates(native_fonts):
    element = _text()
    before = semantic_ink.element_ink_bounds(element, fonts_dir=native_fonts)
    moved = semantic_ink.element_ink_bounds(
        {**element, "x": element["x"] + 37, "y": element["y"] - 81}, fonts_dir=native_fonts
    )
    assert before["provenance"] == moved["provenance"] == "native_alpha"
    assert moved["box"] == pytest.approx(
        [before["box"][0] + 37, before["box"][1] - 81, *before["box"][2:]], abs=1e-6
    )


def test_equivalent_proportional_text_retains_ink_geometry_with_raster_tolerance(native_fonts):
    element = _text()
    original = deepcopy(element)
    scale_factor = 2
    first = semantic_ink.element_ink_bounds(element, fonts_dir=native_fonts)
    doubled = deepcopy(element)
    for key in ("x", "y", "width", "height", "font_size"):
        doubled[key] *= scale_factor
    doubled_original = deepcopy(doubled)
    second = semantic_ink.element_ink_bounds(doubled, fonts_dir=native_fonts)
    assert first["provenance"] == second["provenance"] == "native_alpha"
    assert first["raster_scale"] == second["raster_scale"] == 1
    assert first["confidence"] == second["confidence"] == 1
    assert element == original and doubled == doubled_original
    assert all(result["box"][2] > 0 and result["box"][3] > 0 for result in (first, second))

    def edges(box):
        x, y, width, height = box
        return x, y, x + width, y + height

    # The original edge has one raster pixel of uncertainty, scaled to the
    # result coordinates; the separately painted result has one more pixel.
    # Compare edges because opposing edge errors accumulate in width/height.
    pixel_quantization_budget = scale_factor / first["raster_scale"] + 1 / second["raster_scale"]
    assert edges(second["box"]) == pytest.approx(
        [value * scale_factor for value in edges(first["box"])],
        abs=pixel_quantization_budget,
        rel=0,
    )


def test_native_outline_effect_is_included_without_changing_the_descriptor(native_fonts):
    plain = semantic_ink.element_ink_bounds(_text(), fonts_dir=native_fonts)
    element = _text(
        native_text_effects={
            "version": 1,
            "outline": {"color": "#ffffff", "width_em": 0.4, "fill": True},
        }
    )
    before = deepcopy(element)
    outlined = semantic_ink.element_ink_bounds(element, fonts_dir=native_fonts)
    assert outlined["provenance"] == "native_alpha", outlined
    assert outlined["box"][2] > plain["box"][2]
    assert outlined["box"][3] > plain["box"][3]
    assert element == before


def test_actual_raster_is_bounded_in_both_pixels_and_axis(native_fonts, monkeypatch):
    renderer = get_renderer(native_fonts)
    captured = []

    def render(example, **kwargs):
        assert kwargs["canvas_color"] is None
        encoded = renderer.render(example, **kwargs)
        with Image.open(BytesIO(encoded)) as image:
            captured.append(image.size)
        return encoded

    monkeypatch.setattr(
        semantic_ink,
        "get_renderer",
        lambda _fonts: SimpleNamespace(font_manager=renderer.font_manager, render=render),
    )
    monkeypatch.setattr(semantic_ink, "MAX_RASTER_PIXELS", 4096)
    monkeypatch.setattr(semantic_ink, "MAX_RASTER_AXIS", 128)
    result = semantic_ink.element_ink_bounds(_text(), fonts_dir=native_fonts)
    assert result["provenance"] == "native_alpha", result
    assert 0 < result["raster_scale"] < 1
    assert len(captured) == 1
    assert captured[0][0] * captured[0][1] <= 4096 and max(captured[0]) <= 128


@pytest.mark.parametrize(
    "change", [{"font_family": "NotAvailable"}, {"native_text_effects": {"version": 99}}]
)
def test_missing_exact_font_or_unsupported_effect_returns_a_reasoned_frame(native_fonts, change):
    element = _text(**change)
    result = semantic_ink.element_ink_bounds(element, fonts_dir=native_fonts)
    assert result["provenance"] == "frame_fallback"
    assert result["box"] == [element[key] for key in ("x", "y", "width", "height")]
    assert result["reason"] and result["raster_scale"] == 0


def test_renderer_failure_cannot_be_labelled_native_ink(monkeypatch):
    def fail(_fonts):
        raise RuntimeError("injected renderer failure")

    monkeypatch.setattr(semantic_ink, "get_renderer", fail)
    result = semantic_ink.element_ink_bounds(_text())
    assert result["provenance"] == "frame_fallback"
    assert "injected renderer failure" in result["reason"]


def test_empty_native_alpha_does_not_invent_a_visible_frame(native_fonts):
    result = semantic_ink.element_ink_bounds(
        _text(color="rgba(255,255,255,0)"), fonts_dir=native_fonts
    )
    assert result["provenance"] == "native_alpha", result
    assert result["box"][2:] == [0, 0]


def test_brand_renderer_can_measure_known_exact_font_without_a_manifest():
    result = semantic_ink.element_ink_bounds(_text(font_family="Solstice Sans"))
    assert result["provenance"] == "native_alpha", result
    assert 0 < result["box"][3] < 80


def test_approximate_image_alpha_is_distinguished_from_native_text_paint(tmp_path):
    image = Image.new("RGBA", (20, 10))
    image.paste((255, 255, 255, 255), (5, 2, 15, 8))
    path = tmp_path / "padded.png"
    image.save(path)
    before = path.read_bytes()
    result = semantic_ink.element_ink_bounds(
        {
            "id": "image",
            "type": "image",
            "x": 10,
            "y": 20,
            "width": 200,
            "height": 100,
            "asset_id": "padded.png",
            "fit": "fill",
        },
        assets_dir=tmp_path,
    )
    assert result["box"] == [60, 40, 100, 60]
    assert result["provenance"] == "frame_fallback" and "Approximate" in result["reason"]
    assert result["raster_scale"] == 0 and path.read_bytes() == before
