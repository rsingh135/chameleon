"""Composition regressions that canvas containment alone cannot detect."""

import json
from pathlib import Path

import pytest

from engine.layout import generate_candidates
from engine.layout.candidates import _place
from engine.layout.geometry import Box, union
from engine.layout.model import Unit
from engine.resize import resize


def test_rotated_group_keeps_relative_geometry_in_a_wide_slot():
    elements = [
        {
            "id": "plate",
            "type": "shape",
            "x": 100,
            "y": 100,
            "width": 300,
            "height": 54,
            "rotation": -12,
            "shape_type": "rectangle",
            "fill_color": "rgba(200,30,30,1)",
        },
        {
            "id": "label",
            "type": "text",
            "x": 100,
            "y": 110,
            "width": 300,
            "height": 34,
            "rotation": -12,
            "content": "SALE",
            "font_size": 20,
            "font_family": "Solstice Sans",
        },
    ]
    output = _place(Unit(elements, union(elements)), Box(0, 0, 600, 60), 1)
    scale_x = output[0]["width"] / elements[0]["width"]
    scale_y = output[0]["height"] / elements[0]["height"]
    assert scale_x == pytest.approx(scale_y)
    assert output[1]["y"] - output[0]["y"] == pytest.approx(10 * scale_x)


def test_wide_compositions_offer_a_prominent_product():
    task = json.loads(
        (
            Path(__file__).resolve().parents[1] / "input/chameleon_sample_level2_core.json"
        ).read_text()
    )
    target = {"width": 1600, "height": 400}
    candidates = generate_candidates(task["source"], target)
    products = [
        element
        for candidate in candidates
        for element in candidate["elements"]
        if element.get("asset_id") == "solstice_bottle.png"
    ]
    assert max(element["height"] for element in products) >= target["height"] * 0.6


def test_wide_resize_selects_a_prominent_product():
    task_path = Path(__file__).resolve().parents[1] / "input/chameleon_sample_level2_core.json"
    source = json.loads(task_path.read_text())["source"]
    images = [e for e in source["elements"] if e["type"] == "image" and e.get("fit") != "cover"]
    hero = max(images, key=lambda element: element["width"] * element["height"])
    target = {"width": 1600, "height": 400}
    output = resize(source, target)
    resized_hero = next(element for element in output["elements"] if element["id"] == hero["id"])
    assert resized_hero["height"] >= target["height"] * 0.6
