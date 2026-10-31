"""Visible source geometry must reflect content rather than padded frames."""

import builtins
import os
from copy import deepcopy

import pytest
from PIL import Image

from engine.layout import appearance
from engine.layout.appearance import unit_visible_box, visible_box
from engine.layout.geometry import Box
from engine.layout.text import TextMeasurer, get_measurer


def _image(**values):
    return {
        "id": "art",
        "type": "image",
        "asset_id": "padded.png",
        "fit": "contain",
        "x": 10,
        "y": 20,
        "width": 400,
        "height": 200,
        **values,
    }


def _text(**values):
    return {
        "id": "copy",
        "type": "text",
        "content": "Copy",
        "font_family": "Solstice Sans",
        "font_size": 20,
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 100,
        **values,
    }


@pytest.fixture
def assets(tmp_path):
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (20, 30, 80, 90))
    image.save(tmp_path / "padded.png")
    return tmp_path


@pytest.mark.parametrize(
    "fit,expected",
    [
        ("contain", Box(150, 80, 120, 120)),
        ("cover", Box(90, 40, 240, 180)),
        ("fill", Box(90, 80, 240, 120)),
    ],
)
def test_image_fit_maps_real_alpha_extent_into_frame(assets, fit, expected):
    assert visible_box(_image(fit=fit), assets) == expected


def test_cover_can_crop_all_visible_pixels_away(tmp_path):
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 30, 100))
    image.save(tmp_path / "padded.png")
    result = visible_box(_image(width=100, height=100, fit="cover"), tmp_path)
    assert result.width * result.height == 0


def test_rotated_padded_image_rotates_about_frame_center(assets):
    result = visible_box(_image(rotation=90), assets)
    assert (result.x, result.y, result.width, result.height) == pytest.approx((130, 60, 120, 120))


def test_shape_uses_full_rotated_bounds():
    element = {"type": "shape", "x": 10, "y": 20, "width": 100, "height": 50, "rotation": 90}
    result = visible_box(element)
    assert (result.x, result.y, result.width, result.height) == pytest.approx((35, -5, 50, 100))


def test_asset_cache_invalidates_when_file_changes(assets):
    element = _image()
    first = visible_box(element, assets)
    path = assets / element["asset_id"]
    previous = path.stat().st_mtime_ns
    Image.new("RGBA", (100, 100), (255, 0, 0, 255)).save(path)
    os.utime(path, ns=(previous + 1_000_000, previous + 1_000_000))
    assert visible_box(element, assets) == Box(110, 20, 200, 200)
    assert visible_box(element, assets) != first


@pytest.mark.parametrize("kind", ["missing", "unreadable"])
def test_unavailable_assets_fall_back_to_geometry(tmp_path, kind):
    if kind == "unreadable":
        (tmp_path / "padded.png").write_bytes(b"not an image")
    assert visible_box(_image(), tmp_path) == Box(10, 20, 400, 200)


@pytest.mark.parametrize("absolute", [False, True])
def test_asset_path_cannot_escape_the_supplied_asset_root(tmp_path, absolute):
    assets = tmp_path / "assets"
    assets.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(outside)
    asset_id = str(outside) if absolute else "../outside.png"
    assert visible_box(_image(asset_id=asset_id), assets) == Box(10, 20, 400, 200)


def test_transparent_asset_has_no_occupied_area(tmp_path):
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(tmp_path / "padded.png")
    result = visible_box(_image(), tmp_path)
    assert result.width * result.height == 0


@pytest.mark.parametrize("element", [_image(opacity=0), _text(opacity=0), _text(content=" \n\n")])
def test_invisible_or_blank_elements_have_zero_occupied_area(element):
    result = visible_box(element)
    assert result.width * result.height == 0


@pytest.mark.parametrize("align,offset", [("left", 0), ("center", 0.5), ("right", 1)])
def test_text_alignment_uses_occupied_line_width(align, offset):
    element = _text(text_align=align)
    measured = get_measurer().measure(element, element["font_size"], element["width"])
    result = visible_box(element)
    assert result.x == pytest.approx(element["x"] + (element["width"] - measured.width) * offset)
    assert result.width == pytest.approx(measured.width)
    assert result.y == element["y"]
    assert result.height == pytest.approx(24)


def test_text_overflow_is_not_clipped_to_the_frame():
    element = _text(content="Unbreakableword", width=20, height=2, text_align="right")
    result = visible_box(element)
    assert result.width > element["width"]
    assert result.height > element["height"]
    assert result.x < element["x"]


def test_text_empty_lines_affect_positions_but_not_trailing_occupied_extent():
    result = visible_box(_text(content="Title\n\nBody\n", height=10))
    assert result.y == 100
    assert result.height == pytest.approx(72)


def test_text_rotation_uses_original_frame_center():
    element = _text(rotation=90)
    width = visible_box(_text()).width
    result = visible_box(element)
    assert (result.x, result.y, result.width, result.height) == pytest.approx((226, 50, 24, width))


def test_unit_union_ignores_empty_elements_and_does_not_mutate_inputs(assets):
    elements = [_image(), _text(), _text(opacity=0, x=-999, y=-999)]
    before = deepcopy(elements)
    a, b = visible_box(elements[0], assets), visible_box(elements[1], assets)
    result = unit_visible_box(elements, assets)
    assert result.x == min(a.x, b.x)
    assert result.y == min(a.y, b.y)
    assert result.x + result.width == max(a.x + a.width, b.x + b.width)
    assert result.y + result.height == max(a.y + a.height, b.y + b.height)
    assert unit_visible_box(elements, assets) == result
    assert elements == before
    assert unit_visible_box([]) == Box(0, 0, 0, 0)


def test_text_measurement_remains_available_without_renderer(monkeypatch):
    original_import = builtins.__import__

    def block_renderer(name, *args, **kwargs):
        if name.startswith(("renderer", "skia", "cr_renderer")):
            raise ImportError("renderer deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_renderer)
    fallback = TextMeasurer()
    assert fallback.backend == "estimated"
    monkeypatch.setattr(appearance, "get_measurer", lambda: fallback)
    result = visible_box(_text())
    assert result.width > 0 and result.height > 0
