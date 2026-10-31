"""Relationships should survive changes in headline scale and canvas shape."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.layout import build_context, generate_candidates
from engine.layout.candidates import _flow
from engine.layout.geometry import Box

ROOT = Path(__file__).resolve().parents[1]


def _source(level: str) -> dict:
    return json.loads((ROOT / f"input/chameleon_sample_{level}.json").read_text(encoding="utf-8"))[
        "source"
    ]


def _unit(context, element_id):
    return next(unit for unit in context.units if any(e["id"] == element_id for e in unit.elements))


@pytest.mark.parametrize("headline_size", [56, 120])
def test_footer_control_role_does_not_depend_on_headline_size(headline_size):
    source = _source("level2_core")
    next(e for e in source["elements"] if e["id"] == "headline")["font_size"] = headline_size
    context = build_context(source)
    assert _unit(context, "cta_label").role == "action"
    assert _unit(context, "badge_label").role == "badge"


def test_inferred_control_groups_work_with_renamed_ids():
    source = _source("level2_core")
    labels = {}
    for index, element in enumerate(source["elements"]):
        renamed = f"item-{index}"
        labels[element["id"]] = renamed
        element["id"] = renamed
        element.pop("group_id", None)
    context = build_context(source)
    action = _unit(context, labels["cta_label"])
    assert action.role == "action"
    assert {e["id"] for e in action.elements} == {labels["cta_label"], labels["cta_bg"]}
    assert _unit(context, labels["badge_label"]).role == "badge"


def test_advanced_badges_keep_their_roles_above_the_footer_control():
    context = build_context(_source("level3_advanced"))
    assert _unit(context, "cta_label").role == "action"
    for label in ("badge1_label", "badge2_label", "badge3_label", "ribbon_label"):
        assert _unit(context, label).role == "badge"


@pytest.mark.parametrize("height", [200, 600])
def test_headline_and_body_keep_the_configured_gap_when_space_is_available(height):
    context = build_context(_source("level2_core"))
    units = [_unit(context, "headline"), _unit(context, "subhead")]
    gap = 12
    elements = _flow(units, Box(0, 0, 1000, height), font_scale=1, gap=gap)
    headline, body = elements
    actual_gap = body["y"] - (headline["y"] + headline["height"])
    assert actual_gap == pytest.approx(gap)
    assert headline["y"] > 0
    assert body["y"] + body["height"] < height


@pytest.mark.parametrize("level", ["level2_core", "level3_advanced"])
def test_relationship_reflow_preserves_all_content_and_identity(level):
    source = _source(level)
    before = deepcopy(source)
    mutable = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}
    for candidate in generate_candidates(source, {"width": 1600, "height": 400}):
        assert [e["id"] for e in candidate["elements"]] == [e["id"] for e in source["elements"]]
        for original, resized in zip(source["elements"], candidate["elements"], strict=True):
            assert {key: value for key, value in resized.items() if key not in mutable} == {
                key: value for key, value in original.items() if key not in mutable
            }
    assert source == before
