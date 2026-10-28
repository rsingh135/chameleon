"""The experimental path keeps original context and guards final proxy quality."""

from copy import deepcopy

import pytest

from engine.eval import build_context as build_eval_context
from engine.eval import check_contract
from engine.layout import continuation


@pytest.fixture
def design():
    return {
        "canvas": {"width": 600, "height": 800},
        "elements": [
            {
                "id": "background",
                "type": "shape",
                "x": 0,
                "y": 0,
                "width": 600,
                "height": 800,
                "shape_type": "rectangle",
                "fill_color": "rgba(20,30,40,1)",
            },
            {
                "id": "headline",
                "type": "text",
                "x": 50,
                "y": 70,
                "width": 500,
                "height": 120,
                "content": "Chase the Light",
                "font_family": "Solstice Serif",
                "font_size": 46,
                "color": "rgba(255,255,255,1)",
            },
            {
                "id": "body",
                "type": "text",
                "x": 50,
                "y": 230,
                "width": 500,
                "height": 100,
                "content": "Botanical sparkling infusions for golden hour.",
                "font_family": "Solstice Sans",
                "font_size": 22,
                "color": "rgba(255,255,255,1)",
            },
            {
                "id": "button",
                "type": "shape",
                "x": 50,
                "y": 650,
                "width": 250,
                "height": 60,
                "shape_type": "rectangle",
                "fill_color": "rgba(220,100,30,1)",
                "group_id": "cta",
            },
            {
                "id": "label",
                "type": "text",
                "x": 65,
                "y": 665,
                "width": 220,
                "height": 30,
                "content": "Shop the collection",
                "font_family": "Solstice Sans",
                "font_size": 18,
                "color": "rgba(255,255,255,1)",
                "group_id": "cta",
            },
        ],
    }


def test_canvases_follow_log_dimensions_and_keep_endpoints():
    canvases = continuation.orientation_canvases(
        {"width": 600, "height": 800}, {"width": 1200, "height": 400}, 3
    )
    assert canvases == [
        {"width": 600, "height": 800},
        {"width": 849, "height": 566},
        {"width": 1200, "height": 400},
    ]


def test_path_preserves_identity_content_and_inputs(design):
    original = deepcopy(design)
    target = {"width": 1200, "height": 400}
    result = continuation.continue_layout(design, target, steps=4, beam_width=2)
    assert design == original
    assert target == {"width": 1200, "height": 400}
    assert result["frames"][0] == original
    assert result["frames"][-1]["canvas"] == target
    context = build_eval_context(original)
    for frame in result["frames"]:
        assert check_contract(original, frame, frame["canvas"], context) == []
        assert [element["id"] for element in frame["elements"]] == [
            element["id"] for element in original["elements"]
        ]
        for source_element, element in zip(original["elements"], frame["elements"], strict=True):
            for field in (
                "content",
                "font_family",
                "asset_id",
                "group_id",
                "opacity",
                "style_runs",
            ):
                assert element.get(field) == source_element.get(field)


def test_path_is_deterministic(design):
    first = continuation.continue_layout(
        design, {"width": 1000, "height": 400}, steps=3, beam_width=2
    )
    second = continuation.continue_layout(
        design, {"width": 1000, "height": 400}, steps=3, beam_width=2
    )
    assert first == second


def test_every_anchor_uses_original_source_not_previous_result(design, monkeypatch):
    original = deepcopy(design)
    generate, resize = continuation.generate_candidates, continuation.resize
    seen = []

    def generate_from_original(source, canvas):
        assert source == original
        seen.append(canvas)
        return generate(source, canvas)

    def resize_from_original(source, canvas):
        assert source == original
        return resize(source, canvas)

    monkeypatch.setattr(continuation, "generate_candidates", generate_from_original)
    monkeypatch.setattr(continuation, "resize", resize_from_original)
    result = continuation.continue_layout(
        design, {"width": 1200, "height": 400}, steps=4, beam_width=2
    )
    assert seen == [frame["canvas"] for frame in result["frames"][1:]]


def test_propagation_restores_original_font_preferences(design):
    context = continuation.build_context(design)
    shrunken = deepcopy(design)
    for element in shrunken["elements"]:
        if element["type"] == "text":
            element["font_size"] = 1
    recovered = continuation.propagated_candidate(design, context, shrunken, design["canvas"])
    sizes = [element["font_size"] for element in recovered["elements"] if element["type"] == "text"]
    assert min(sizes) > 10


def test_endpoint_stays_exactly_baseline_despite_extreme_movement_weight(design):
    target = {"width": 1200, "height": 300}
    result = continuation.continue_layout(
        design,
        target,
        steps=3,
        beam_width=2,
        movement_weight=10000,
    )
    assert result["frames"][-1] == result["baseline_frames"][-1]
    assert result["comparison"]["endpoint_design_matches_baseline"]
    assert result["anchors"][-1]["quality"] == result["baseline_anchors"][-1]["quality"]


def test_smooth_valid_seeds_prevent_deferring_all_motion_to_final_anchor(monkeypatch):
    source = {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            {
                "id": "marker",
                "type": "shape",
                "x": 20,
                "y": 20,
                "width": 80,
                "height": 80,
                "shape_type": "rectangle",
                "fill_color": "rgba(40,80,120,1)",
            }
        ],
    }

    def endpoint(original, canvas):
        design = deepcopy(original)
        design["canvas"] = dict(canvas)
        element = design["elements"][0]
        element.update(
            x=canvas["width"] * 0.7,
            y=canvas["height"] * 0.4,
            width=canvas["width"] * 0.2,
            height=canvas["height"] * 0.2,
        )
        return design

    def templates(original, canvas):
        held = deepcopy(original)
        held["canvas"] = dict(canvas)
        element = held["elements"][0]
        element.update(
            x=canvas["width"] * 0.05,
            y=canvas["height"] * 0.05,
            width=canvas["width"] * 0.2,
            height=canvas["height"] * 0.2,
        )
        return [held, endpoint(original, canvas)]

    monkeypatch.setattr(continuation, "resize", endpoint)
    monkeypatch.setattr(continuation, "generate_candidates", templates)
    result = continuation.continue_layout(source, {"width": 800, "height": 200}, steps=7)
    jumps = [anchor["movement"]["max_center_jump"] for anchor in result["anchors"]]
    assert jumps[1] > 0
    assert max(jumps) < 0.2
    assert jumps[-1] <= max(jumps[1:-1]) * 1.1
    assert result["frames"][-1] == result["baseline_frames"][-1]
    for anchor, baseline in zip(result["anchors"], result["baseline_anchors"], strict=True):
        assert not anchor["violations"]
        assert anchor["quality"] >= baseline["quality"] - 0.06 - 1e-9


@pytest.mark.parametrize("steps", [1, 26, 2.5])
def test_anchor_count_is_bounded(steps):
    with pytest.raises(ValueError, match="steps"):
        continuation.orientation_canvases(
            {"width": 100, "height": 100}, {"width": 200, "height": 100}, steps
        )
