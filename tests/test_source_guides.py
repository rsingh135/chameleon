"""Source order, rule geometry and real row ownership survive reflow."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract
from engine.layout.source_guides import MAX_STATES, generate_source_guides
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]
MUTABLE = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


def _text(eid, content, y, font=20, **extra):
    return dict(
        id=eid,
        type="text",
        x=80,
        y=y,
        width=440,
        height=font * 1.2,
        content=content,
        font_family="Solstice Sans",
        font_size=font,
        line_height=1.2,
        text_align="center",
        color="#000000",
        **extra,
    )


def _simple():
    return {
        "canvas": {"width": 600, "height": 900},
        "elements": [
            _text("brand", "Small brand", 50),
            _text("headline", "A large title", 140, 48),
            _text("body", "Supporting information", 320),
            _text("footer", "Further details", 720),
        ],
    }


def test_canonical_uses_one_real_scale_for_geometry_and_fonts_including_photos():
    source = _simple()
    source["elements"].append(
        dict(
            id="photo",
            type="image",
            x=200,
            y=450,
            width=160,
            height=220,
            asset_id="solstice_bottle.png",
            fit="contain",
        )
    )
    before = deepcopy(source)
    states = generate_source_guides(source, {"width": 1200, "height": 600})
    assert states and len(states) <= MAX_STATES
    canonical = next(s.design for s in states if s.name == "source-guides/source-fit")
    scale = 600 / 900
    for old, new in zip(source["elements"], canonical["elements"], strict=True):
        assert new["height"] / old["height"] == pytest.approx(scale)
        if old["type"] == "text":
            assert new["font_size"] / old["font_size"] == pytest.approx(scale)
        else:
            assert new["width"] / old["width"] == pytest.approx(scale)
    assert all(s.name == "source-guides/source-fit" for s in states)
    assert source == before


def test_real_flyer_keeps_business_order_rule_thinness_and_complete_event_row():
    folder = ROOT / "fixtures/open_posters/prepared/doory_grand-opening-flyer"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    target = {"width": 1600, "height": 900}
    before = deepcopy(source)
    originals = {e["id"]: e for e in source["elements"]}
    header = next(e for e in source["elements"] if e.get("content") == "CORNER BLOOM CAFÉ")
    title = next(e for e in source["elements"] if e.get("content") == "Grand")
    event = [
        e
        for e in source["elements"]
        if e.get("content")
        in {"FROM 8:00\nUNTIL 14:00", "SATURDAY", "12th", "JULY", "24 ORCHARD LANE\nRIVERSIDE"}
        or e.get("rotation") == 90
    ]
    assert len(event) == 7
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        states = generate_source_guides(source, target)
        context = build_context(source)
        assert 1 < len(states) <= MAX_STATES
        assert any("columns" in s.name for s in states)
        for state in states:
            design = state.design
            elements = {e["id"]: e for e in design["elements"]}
            assert elements[header["id"]]["y"] < elements[title["id"]]["y"]
            assert check_contract(source, design, target, context) == []
            for old in source["elements"]:
                new = elements[old["id"]]
                assert {k: v for k, v in old.items() if k not in MUTABLE} == {
                    k: v for k, v in new.items() if k not in MUTABLE
                }
                if old["type"] == "shape" and old["height"] == 1:
                    assert new["width"] / new["height"] == pytest.approx(old["width"])
            anchor = event[0]
            scale = elements[anchor["id"]]["height"] / anchor["height"]
            for old in event:
                new = elements[old["id"]]
                assert new["x"] - elements[anchor["id"]]["x"] == pytest.approx(
                    (old["x"] - anchor["x"]) * scale
                )
                assert new["y"] - elements[anchor["id"]]["y"] == pytest.approx(
                    (old["y"] - anchor["y"]) * scale
                )
                assert new["width"] == pytest.approx(old["width"] * scale)
                assert new["height"] == pytest.approx(old["height"] * scale)
            assert [e["id"] for e in design["elements"]] == list(originals)
    assert source == before


def test_tracking_expands_standalone_frame_without_changing_font_or_tracking():
    source = {
        "canvas": {"width": 600, "height": 900},
        "elements": [
            dict(_text("tracked", "TRACKING", 200, 30, letter_spacing=9), x=190, width=220)
        ],
    }
    states = generate_source_guides(source, {"width": 1000, "height": 450})
    state = next(s for s in states if s.name == "source-guides/source-fit")
    text = state.design["elements"][0]
    assert text["font_size"] == 15
    assert text["letter_spacing"] == 9
    assert text["width"] > 110
    assert text["x"] + text["width"] / 2 == pytest.approx(500)
    assert state.exceptions


def test_vector_provenance_allows_pair_rows_but_unmarked_photo_does_not():
    source = _simple()
    for i, y in enumerate([320, 450, 580]):
        source["elements"].append(
            dict(
                id=f"icon{i}",
                type="image",
                x=20,
                y=y,
                width=40,
                height=40,
                asset_id=f"icon{i}.png",
                fit="fill",
                source_art={"kind": "svg-nontext", "curve_editable": False},
            )
        )
        source["elements"].append(dict(_text(f"label{i}", f"Description {i}", y), x=80, width=440))
    assert any(
        "columns" in s.name for s in generate_source_guides(source, {"width": 1600, "height": 900})
    )
    source["elements"][-2].pop("source_art")
    assert all(
        s.name == "source-guides/source-fit"
        for s in generate_source_guides(source, {"width": 1600, "height": 900})
    )


def test_bounded_deterministic_and_fresh_without_mutating_source():
    source = _simple()
    before = deepcopy(source)
    states = generate_source_guides(source, {"width": 1600, "height": 900})
    assert states == generate_source_guides(source, {"width": 1600, "height": 900})
    assert len(states) <= MAX_STATES
    if states:
        changed = states[0].design
        changed["elements"][0]["content"] = "changed"
        assert states[0].design["elements"][0]["content"] != "changed"
    assert source == before
    assert generate_source_guides(source, {"width": 0, "height": 900}) == []
