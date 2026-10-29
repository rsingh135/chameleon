"""Fine authored rules cannot win selection as new heavy bars."""

from copy import deepcopy

import pytest

from engine.layout.rules import analyze_rules


def test_uniform_scaling_preserves_both_horizontal_and_rotated_dividers():
    source = {
        "canvas": {"width": 800, "height": 1000},
        "elements": [
            {
                "id": "one",
                "type": "shape",
                "shape_type": "rectangle",
                "width": 69,
                "height": 1,
                "rotation": 0,
            },
            {
                "id": "two",
                "type": "shape",
                "shape_type": "rectangle",
                "width": 145,
                "height": 1,
                "rotation": 90,
            },
        ],
    }
    result = deepcopy(source)
    for element in result["elements"]:
        element["width"] *= 1.7
        element["height"] *= 1.7
    report = analyze_rules(source, result)
    assert report["rule_count"] == 2 and report["distorted_rules"] == []
    result["elements"][0]["height"] *= 30
    bad = analyze_rules(source, result)
    assert bad["distorted_rules"][0]["element_id"] == "one"
    assert bad["score"] < 0.2
    assert source["elements"][0]["height"] == 1


@pytest.mark.parametrize("width,height", [(200, 40), (500, 20)])
def test_filled_buttons_and_heavy_authored_bars_are_not_fine_rules(width, height):
    source = {
        "canvas": {"width": 800, "height": 1000},
        "elements": [
            {
                "id": "shape",
                "type": "shape",
                "shape_type": "rectangle",
                "width": width,
                "height": height,
            }
        ],
    }
    assert not analyze_rules(source, source)["active"]
