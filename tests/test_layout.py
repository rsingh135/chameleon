"""Behavioral checks for reflow, grouping and renderer-compatible text fitting."""

from copy import deepcopy

import pytest

from engine.layout import generate_candidates, get_measurer
from engine.layout.geometry import bounds
from engine.resize import resize


def test_rotated_visual_bounds_fit_canvas(source, target_canvas):
    result = resize(source, target_canvas)
    for element in result["elements"]:
        box = bounds(element)
        assert box.x >= -1e-6
        assert box.y >= -1e-6
        assert box.x + box.width <= target_canvas["width"] + 1e-6
        assert box.y + box.height <= target_canvas["height"] + 1e-6


def test_reflow_is_deterministic_and_does_not_mutate_target(source, target_canvas):
    original = deepcopy(target_canvas)
    first = resize(source, target_canvas)
    assert resize(source, target_canvas) == first
    assert target_canvas == original


def test_selected_text_fits_boxes(source, target_canvas):
    result = resize(source, target_canvas)
    for element in result["elements"]:
        if element["type"] == "text":
            measured = get_measurer().measure(element, element["font_size"], element["width"])
            assert measured.width <= element["width"] + 0.05, element["id"]
            assert measured.height <= element["height"] + 0.05, element["id"]


def test_reflow_does_not_depend_on_sample_ids(source, target_canvas):
    changed = deepcopy(source)
    for index, element in enumerate(changed["elements"]):
        element["id"] = f"arbitrary-{index}"
    expected = resize(source, target_canvas)
    actual = resize(changed, target_canvas)
    for a, b in zip(actual["elements"], expected["elements"], strict=True):
        assert {k: v for k, v in a.items() if k != "id"} == {
            k: v for k, v in b.items() if k != "id"
        }


@pytest.mark.parametrize(
    "content", ["UnbreakableWord" * 12, "Line one\nLine two", "你好世界 " * 10]
)
def test_font_fit_handles_long_words_newlines_and_unicode(content):
    element = {"type": "text", "content": content, "font_family": "Solstice Sans", "font_size": 48}
    size = get_measurer().fit(element, 260, 80, 48)
    measured = get_measurer().measure(element, size, 260)
    assert 1 <= size <= 48
    assert measured.width <= 260.05
    assert measured.height <= 80.05


def test_each_candidate_preserves_opacity_style_and_grouping(source):
    for candidate in generate_candidates(source, {"width": 728, "height": 90}):
        for original, element in zip(source["elements"], candidate["elements"], strict=True):
            for field in ("opacity", "group_id", "style_runs", "rotation", "asset_id"):
                assert element.get(field) == original.get(field)
