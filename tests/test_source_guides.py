"""Source order, rule geometry and real row ownership survive reflow."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract
from engine.layout.model import build_context as layout_context
from engine.layout.source_guides import MAX_STATES, _owned_units, generate_source_guides
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
    assert any(s.topology == "source-media-columns" for s in states)
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


@pytest.mark.parametrize(
    "target", [{"width": 1600, "height": 900}, {"width": 1080, "height": 1920}]
)
def test_original_sale_has_real_format_proposals_with_rigid_photo_and_plate(target):
    folder = ROOT / "fixtures/open_posters/prepared/doory_bold-sale-announcement"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        _, plates, media = _owned_units(layout_context(source))
        assert len(plates) == len(media) == 1
        states = generate_source_guides(source, target)
        adapted = [s for s in states if s.topology.startswith("source-media-")]
        assert adapted and len(states) <= MAX_STATES
        context = build_context(source)
        for state in adapted:
            design = state.design
            by_id = {e["id"]: e for e in design["elements"]}
            assert check_contract(source, design, target, context) == []
            assert [e["id"] for e in design["elements"]] == [e["id"] for e in source["elements"]]
            for component in [plates[0], media[0]]:
                anchor = component.elements[0]
                new_anchor = by_id[anchor["id"]]
                scale = new_anchor["height"] / anchor["height"]
                for old in component.elements:
                    new = by_id[old["id"]]
                    assert new["width"] == pytest.approx(old["width"] * scale)
                    assert new["height"] == pytest.approx(old["height"] * scale)
                    assert new["x"] - new_anchor["x"] == pytest.approx(
                        (old["x"] - anchor["x"]) * scale
                    )
                    assert new["y"] - new_anchor["y"] == pytest.approx(
                        (old["y"] - anchor["y"]) * scale
                    )
                    if old["type"] == "text":
                        assert new["font_size"] == pytest.approx(old["font_size"] * scale)
            plate = next(e for e in plates[0].elements if e["type"] == "image")
            assert by_id[plate["id"]]["y"] + by_id[plate["id"]]["height"] == pytest.approx(
                target["height"]
            )
            for old, new in zip(source["elements"], design["elements"], strict=True):
                assert {k: v for k, v in old.items() if k not in MUTABLE} == {
                    k: v for k, v in new.items() if k not in MUTABLE
                }
    assert source == before


@pytest.mark.parametrize(
    "target", [{"width": 1600, "height": 900}, {"width": 1080, "height": 1080}]
)
def test_photo_dependent_dark_headers_and_fade_keep_original_bright_support(target):
    folder = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        ordinary, plates, media = _owned_units(layout_context(source))
        assert ordinary and not plates and len(media) == 1
        headers = [e for e in source["elements"] if e["type"] == "text" and e["y"] < 200]
        assert len(headers) == 3
        assert all(e["id"] in {m["id"] for m in media[0].elements} for e in headers)
        photo = next(
            e for e in media[0].elements if e.get("source_art", {}).get("kind") == "masked-photo"
        )
        states = [
            s
            for s in generate_source_guides(source, target)
            if s.topology == "source-media-columns"
        ]
        assert states
        for state in states:
            elements = {e["id"]: e for e in state.design["elements"]}
            frame = elements[photo["id"]]
            for old in headers:
                new = elements[old["id"]]
                assert (
                    frame["x"] <= new["x"]
                    and new["x"] + new["width"] <= frame["x"] + frame["width"] + 0.01
                )
            for unit in ordinary:
                for old in unit.text:
                    new = elements[old["id"]]
                    assert new["x"] + new["width"] < frame["x"]


def test_story_distributes_source_whitespace_without_detaching_event_internals():
    folder = ROOT / "fixtures/open_posters/prepared/doory_grand-opening-flyer"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    target = {"width": 1080, "height": 1920}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        state = next(
            s
            for s in generate_source_guides(source, target)
            if s.name == "source-guides/bands-distributed"
        )
    text = [e for e in state.design["elements"] if e["type"] == "text"]
    assert (
        max(e["y"] + e["height"] for e in text) - min(e["y"] for e in text) > target["height"] * 0.8
    )
    scales = [
        new["font_size"] / old["font_size"]
        for old, new in zip(source["elements"], state.design["elements"], strict=True)
        if old["type"] == "text"
    ]
    assert max(scales) == pytest.approx(min(scales))
