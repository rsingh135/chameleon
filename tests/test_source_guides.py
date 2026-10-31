"""Source order, rule geometry and real row ownership survive reflow."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract
from engine.layout.model import build_context as layout_context
from engine.layout.reading_order import analyze_reading_order
from engine.layout.source_containers import (
    _native_frame,
    analyze_source_containers,
    infer_source_containers,
)
from engine.layout.source_guides import MAX_STATES, _owned_units, generate_source_guides
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]
MUTABLE = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


def _panel_failure_details(source, target):
    """Failure-only observation of real proposal declines; never alter policy."""
    import linecache
    import platform
    import sys
    from collections import Counter

    import engine.layout.source_guides as guides
    from engine.layout.text import get_measurer

    counts, failures, seen = Counter(), [], set()
    observed = {"_panel_place", "_state", "_valid", "_tracking_frame", "_text_fits"}

    def trace(frame, event, result):
        name = frame.f_code.co_name
        if frame.f_code.co_filename != guides.__file__ or name not in observed:
            return None
        if event != "return" or result is not None and result is not False:
            return trace
        counts[name] += 1
        local = frame.f_locals
        element = local.get("element")
        eid = element.get("id") if isinstance(element, dict) else None
        key = (name, eid, frame.f_lineno)
        if key in seen or len(failures) >= 12:
            return trace
        seen.add(key)
        record = {
            "function": name,
            "line": frame.f_lineno,
            "guard": linecache.getline(guides.__file__, frame.f_lineno).strip(),
        }
        for field in ("scale", "content", "free", "minimum", "width"):
            if isinstance(local.get(field), (int, float)):
                record[field] = local[field]
        for field in ("cell", "inner", "row_inner"):
            box = local.get(field)
            if box is not None:
                record[field] = {k: getattr(box, k) for k in ("x", "y", "width", "height")}
        if isinstance(element, dict):
            record["element"] = {
                k: element.get(k)
                for k in (
                    "id",
                    "type",
                    "font_family",
                    "font_size",
                    "x",
                    "y",
                    "width",
                    "height",
                    "max_lines",
                    "group_id",
                    "line_height",
                    "letter_spacing",
                )
            }
            record["copy"] = str(element.get("content", ""))[:100]
            measured = local.get("measured")
            if measured is not None and hasattr(measured, "wrapped"):
                record["measured"] = {
                    "width": measured.width,
                    "height": measured.height,
                    "lines": measured.lines,
                    "wrapped": measured.wrapped[:140],
                }
        panel = local.get("panel")
        if panel is not None:
            record["panel"] = panel.container.elements[0]["id"]
            cell = local.get("cell")
            if cell is not None:
                bad = []
                for child in local.get("result", []):
                    box = guides.bounds(child)
                    if (
                        box.x < cell.x - guides._EPS
                        or box.y < cell.y - guides._EPS
                        or box.x + box.width > cell.x + cell.width + guides._EPS
                        or box.y + box.height > cell.y + cell.height + guides._EPS
                    ):
                        bad.append({"id": child["id"], "bounds": vars(box)})
                record["bad_bounds"] = bad[:3]
        failures.append(record)
        return trace

    measurer = get_measurer()
    details = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "backend": measurer.backend,
        "native_document": measurer.native_document,
    }
    try:
        import skia

        details["skia"] = skia.__version__
    except ImportError:
        details["skia"] = None
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        states = generate_source_guides(source, target)
    finally:
        sys.settrace(previous)
    details.update(
        observed_states=[state.name for state in states],
        decline_counts=dict(counts),
        first_declines=failures,
    )
    return json.dumps(details, indent=2)


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


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_native_svg_grouped_line_repairs_actual_advances_without_changing_ink_anchor(align):
    from engine.layout.geometry import Box
    from engine.layout.source_guides import _text_fits, _tracking_frame
    from engine.layout.text import get_measurer

    package = ROOT / "fixtures/open_posters/prepared/opensuse_trifold_front"
    source = json.loads((package / "source.json").read_text(encoding="utf-8"))
    # A genuine grouped SVG line, with a fractional native font size. Recreate
    # the measured Linux failure: glyphs need slightly more room than the
    # similarity-scaled frame, although the authored line is still one line.
    element = deepcopy(
        next(e for e in source["elements"] if e.get("group_id") and e["type"] == "text")
    )
    with using_resources(package / "resources/assets", package / "resources/fonts"):
        advance = get_measurer().measure(element, element["font_size"], float("inf")).width
        element.update(x=200, y=40, width=advance / 1.02, text_align=align)
        original = deepcopy(element)
        assert not _text_fits(element)
        assert _tracking_frame(element, Box(0, 0, 1000, 300), allow_svg_group=True)
        factor = {"left": 0, "center": 0.5, "right": 1}[align]
        assert element["x"] + element["width"] * factor == pytest.approx(
            original["x"] + original["width"] * factor
        )
        assert advance <= element["width"] < original["width"] * 1.05
        assert _text_fits(element)
        assert (
            get_measurer().measure(element, element["font_size"], element["width"]).wrapped
            == element["content"]
        )
        assert {k: v for k, v in element.items() if k not in {"x", "width"}} == {
            k: v for k, v in original.items() if k not in {"x", "width"}
        }
        fitted = deepcopy(element)
        assert _tracking_frame(element, Box(0, 0, 1000, 300), allow_svg_group=True) is False
        assert element == fitted


@pytest.mark.parametrize(
    "change",
    [
        {"rotation": 1},
        {"max_lines": 2},
        {"svg_font_family": None},
        {"text_align": "justify"},
        {"native_text_effects": {"gradient": {"kind": "linear"}}},
        {"content": "One\ntwo"},
    ],
)
def test_native_svg_group_frame_repair_rejects_unproven_or_frame_dependent_lines(change):
    from engine.layout.geometry import Box
    from engine.layout.source_guides import _tracking_frame
    from engine.layout.text import get_measurer

    package = ROOT / "fixtures/open_posters/prepared/opensuse_trifold_front"
    element = dict(
        _text("line", "A native grouped line", 40, 18, group_id="authored-group"),
        font_family="OpenSans-Regular",
        svg_font_family="Open Sans",
        max_lines=1,
    )
    element.update(change)
    with using_resources(package / "resources/assets", package / "resources/fonts"):
        advance = get_measurer().measure(element, element["font_size"], float("inf")).width
        element["width"] = advance / 1.02
        before = deepcopy(element)
        assert _tracking_frame(element, Box(0, 0, 1000, 300), allow_svg_group=True) is None
        assert element == before


def test_native_svg_group_frame_repair_keeps_strict_cell_and_group_limits():
    from engine.layout.geometry import Box
    from engine.layout.source_guides import _tracking_frame
    from engine.layout.text import get_measurer

    package = ROOT / "fixtures/open_posters/prepared/opensuse_trifold_front"
    element = dict(
        _text("line", "A native grouped line", 40, 18, group_id="authored-group"),
        x=10,
        text_align="left",
        font_family="OpenSans-Regular",
        svg_font_family="Open Sans",
        max_lines=1,
    )
    with using_resources(package / "resources/assets", package / "resources/fonts"):
        advance = get_measurer().measure(element, element["font_size"], float("inf")).width
        element["width"] = advance / 1.02
        before = deepcopy(element)
        assert _tracking_frame(element, Box(0, 0, 1000, 300)) is None
        assert _tracking_frame(element, Box(10, 0, advance - 1, 300), allow_svg_group=True) is None
        assert element == before
        element["width"] = advance / 1.06
        before = deepcopy(element)
        assert _tracking_frame(element, Box(0, 0, 1000, 300), allow_svg_group=True) is None
        assert element == before


@pytest.mark.parametrize("height,floor", [(1080, 13.7), (1920, 16.8)])
def test_native_integer_advances_fit_proven_panels_without_changing_fonts_or_insets(
    monkeypatch, height, floor
):
    import math

    from engine.layout.source_guides import _panel_states
    from engine.layout.text import get_measurer

    package = ROOT / "fixtures/open_posters/prepared/opensuse_cheatsheet"
    source = json.loads((package / "source.json").read_text(encoding="utf-8"))
    original = deepcopy(source)
    target = {"width": 1080, "height": height}
    with using_resources(package / "resources/assets", package / "resources/fonts"):
        # Native backends can quantize individual hinted advances. Exercise
        # that real measurement boundary rather than granting wider tolerances
        # to the fit, group, enclosure or collision checks.
        measurer = get_measurer()
        widths = measurer._widths
        monkeypatch.setattr(
            measurer, "_widths", lambda *args: tuple(math.floor(w + 0.5) for w in widths(*args))
        )
        states = _panel_states(source, target, layout_context(source))
        assert states
        state = max(
            states,
            key=lambda s: min(e["font_size"] for e in s.design["elements"] if e["type"] == "text"),
        )
        design = state.design
        assert min(e["font_size"] for e in design["elements"] if e["type"] == "text") >= floor
        assert check_contract(source, design, target, build_context(source)) == []
        assert analyze_source_containers(source, design)["warnings"] == []
        assert source == original


def test_native_panel_frame_correction_cannot_overlap_an_originally_separate_label():
    from engine.layout.geometry import Box, bounds, union
    from engine.layout.model import Unit
    from engine.layout.source_guides import _Band, _Panel, _panel_dimensions, _panel_place
    from engine.layout.text import get_measurer

    package = ROOT / "fixtures/open_posters/prepared/opensuse_trifold_front"
    with using_resources(package / "resources/assets", package / "resources/fonts"):
        label = dict(
            _text("label", "Service status", 30, 18, group_id="authored-line"),
            x=20,
            text_align="left",
            font_family="OpenSans-Regular",
            svg_font_family="Open Sans",
            max_lines=1,
        )
        advance = get_measurer().measure(label, label["font_size"], float("inf")).width
        label["width"] = advance / 1.02
        neighbor = dict(label, id="neighbor", x=20 + label["width"] + 0.1, content="Next")
        root = dict(
            id="panel",
            type="shape",
            x=0,
            y=0,
            width=500,
            height=200,
            shape_type="rectangle",
            fill_color="#ffffff",
        )
        row = _Band((label, neighbor), union([label, neighbor]), union([label, neighbor]))
        panel = _Panel(
            Unit([root], bounds(root)),
            (Unit([label, neighbor], row.box),),
            (row,),
            frozenset(),
            True,
            False,
        )
        before = deepcopy(panel)
        _, height = _panel_dimensions(panel)
        assert _panel_place(panel, Box(0, 0, 500, height), 1) is None
        assert panel == before


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
                    assert (
                        new["x"] + new["width"] < frame["x"]
                        or new["x"] > frame["x"] + frame["width"]
                    )


@pytest.mark.parametrize("width,height", [(1600, 900), (1080, 1080)])
def test_photo_caption_source_order_guides_fullheight_media_and_live_copy_wrap(width, height):
    folder = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    target = {"width": width, "height": height}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        state = next(
            s
            for s in generate_source_guides(source, target)
            if s.name == "source-guides/media-columns-leading-fullheight"
        )
        design = state.design
        assert check_contract(source, design, target, build_context(source)) == []
        order = analyze_reading_order(source, design)
        assert order["high_confidence_inversions"] == 0
        assert order["comparison_coverage"] == 1
        assert min(e["font_size"] for e in design["elements"] if e["type"] == "text") >= (
            min(width, height) * 0.012
        )
        original_photo = next(
            e for e in source["elements"] if e.get("source_art", {}).get("kind") == "masked-photo"
        )
        photo = next(e for e in design["elements"] if e["id"] == original_photo["id"])
        assert photo["x"] == photo["y"] == 0
        assert photo["height"] == height
        assert photo["width"] / original_photo["width"] == pytest.approx(
            photo["height"] / original_photo["height"]
        )
        for old, new in zip(source["elements"], design["elements"], strict=True):
            assert {k: v for k, v in old.items() if k not in MUTABLE} == {
                k: v for k, v in new.items() if k not in MUTABLE
            }
    assert source == before


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


@pytest.mark.parametrize(
    "ident,count,square_floor,story_floor",
    [
        ("opensuse_trifold_front", 3, 12.96, 16.0),
        ("opensuse_cheatsheet", 7, 13.7, 16.8),
    ],
)
@pytest.mark.parametrize("height", [1080, 1920])
def test_real_source_panels_reflow_at_one_readable_foreground_scale(
    ident, count, square_floor, story_floor, height
):
    folder = ROOT / "fixtures/open_posters/prepared" / ident
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    target = {"width": 1080, "height": height}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        profile = infer_source_containers(source)
        assert len(profile.containers) == count
        assert profile.source_digest
        with pytest.raises(FrozenInstanceError):
            profile.containers[0].confidence = 0
        claimed = [eid for c in profile.containers for eid in (c.frame_id, *c.member_ids)]
        assert len(claimed) == len(set(claimed))
        backgrounds = {e["id"] for e in layout_context(source).backgrounds}
        assert set(claimed) | set(profile.loose_ids) | backgrounds == {
            e["id"] for e in source["elements"]
        }
        states = generate_source_guides(source, target)
        assert len(states) <= MAX_STATES
        panels = [s for s in states if s.topology == "source-panel-columns"]
        assert panels, _panel_failure_details(source, target)
        state = max(
            panels,
            key=lambda s: min(e["font_size"] for e in s.design["elements"] if e["type"] == "text"),
        )
        design = state.design
        assert check_contract(source, design, target, build_context(source)) == []
        assert analyze_source_containers(source, design, profile)["accepted"]
        assert min(e["font_size"] for e in design["elements"] if e["type"] == "text") >= (
            square_floor if height == 1080 else story_floor
        )
        frame_ids = {eid for c in profile.containers for eid in c.frame_ids}
        common_scale = next(
            e["font_size"] for e in design["elements"] if e["type"] == "text"
        ) / next(e["font_size"] for e in source["elements"] if e["type"] == "text")
        scales = []
        for old, new in zip(source["elements"], design["elements"], strict=True):
            assert {k: v for k, v in old.items() if k not in MUTABLE} == {
                k: v for k, v in new.items() if k not in MUTABLE
            }
            if old["type"] == "text":
                scales.append(new["font_size"] / old["font_size"])
            if old["id"] not in frame_ids | backgrounds:
                assert new["height"] / old["height"] == pytest.approx(common_scale)
                if old["type"] != "text":
                    assert new["width"] / old["width"] == pytest.approx(
                        new["height"] / old["height"]
                    )
        assert max(scales) == pytest.approx(min(scales))
    assert source == before


@pytest.mark.parametrize("fill", ["#fff0", "#ffffff00", "rgba(1,2,3,0.5)", "url(#paint)"])
def test_adaptive_native_frame_requires_proven_opaque_paint(fill):
    frame = dict(type="shape", shape_type="rectangle", fill_color=fill)
    assert not _native_frame(frame)
    frame["fill_color"] = "#ffffff"
    assert _native_frame(frame)
    frame["fill_color"] = "rgba(115,186,37,1)"
    assert _native_frame(frame)


def test_vector_container_proof_rejects_internal_art_and_unknown_photos(tmp_path):
    from PIL import Image, ImageDraw

    from engine.layout.source_containers import _flat_vector_container

    image = Image.new("RGBA", (100, 180), "#003342")
    ImageDraw.Draw(image).rectangle((0, 0, 99, 179), outline="#66bf2e")
    path = tmp_path / "plate.png"
    image.save(path)
    element = dict(
        type="image", asset_id="plate.png", fit="fill", source_art={"kind": "svg-nontext"}
    )
    with using_resources(tmp_path):
        assert _flat_vector_container(element)
        ImageDraw.Draw(image).rectangle((40, 80, 60, 100), fill="#ffffff")
        image.save(path)
        assert not _flat_vector_container(element)
        element.pop("source_art")
        assert not _flat_vector_container(element)


def test_synthetic_columns_keep_venue_and_guest_together_in_real_concert_poster():
    folder = ROOT / "fixtures/open_posters/prepared/doory_night-bloom-concert-poster"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        states = generate_source_guides(source, {"width": 1600, "height": 900})
    names = {state.name for state in states}
    assert "source-guides/columns-5" not in names
    assert "source-guides/columns-4" in names


@pytest.mark.parametrize(
    "ident,kept,rejected",
    [
        ("doory_grand-opening-flyer", "columns-4", ("columns-2", "columns-3", "columns-5")),
        ("doory_botanical-celebration-invitation", "columns-5", ("columns-3", "columns-4")),
    ],
)
def test_close_primary_title_bands_stay_attached_in_synthetic_columns(ident, kept, rejected):
    folder = ROOT / "fixtures/open_posters/prepared" / ident
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        names = {
            state.name for state in generate_source_guides(source, {"width": 1600, "height": 900})
        }
    assert "source-guides/" + kept in names
    assert all("source-guides/" + name not in names for name in rejected)


def test_ornament_only_column_is_not_substantive_copy():
    from engine.layout.geometry import Box
    from engine.layout.source_guides import _balanced_columns, _Band

    text = _Band((dict(type="text", content="Details"),), Box(0, 0, 200, 100), Box(0, 0, 200, 100))
    art = _Band((dict(type="shape"),), Box(0, 0, 200, 100), Box(0, 0, 200, 100))
    assert not _balanced_columns([[text], [art]])
    tail = _Band((dict(type="text", content="Tail"),), Box(0, 0, 200, 20), Box(0, 0, 200, 20))
    assert not _balanced_columns([[text], [tail]])
    assert _balanced_columns([[text], [text]])


def test_full_candidate_pool_selects_connected_concert_venue_and_guest(tmp_path):
    from engine.eval.render import render_image
    from engine.resize import resize_with_state

    folder = ROOT / "fixtures/open_posters/prepared/doory_night-bloom-concert-poster"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    target = {"width": 1600, "height": 900}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        selected = resize_with_state(source, target)
        design = selected["design"]
        venue = next(e for e in design["elements"] if e.get("content") == "RIVERSIDE PARK")
        guest = next(e for e in design["elements"] if e.get("content") == "with very special guest")
        artist = next(e for e in design["elements"] if e.get("content") == "PAPER TIGERS")
        venue_center = venue["x"] + venue["width"] / 2
        for element in (guest, artist):
            assert abs(element["x"] + element["width"] / 2 - venue_center) < target["width"] * 0.08
            assert element["y"] > venue["y"] + venue["height"]
        assert check_contract(source, design, target, build_context(source)) == []
        native = render_image(design)
        assert native.size == (1600, 900)
        native.save(tmp_path / "selected-concert-landscape.png")
    assert source == before


def test_native_invitation_columns_use_visible_type_width_instead_of_empty_frames():
    folder = ROOT / "fixtures/open_posters/prepared/doory_botanical-celebration-invitation"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    originals = {e["id"]: e for e in source["elements"]}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        state = next(
            s
            for s in generate_source_guides(source, {"width": 1600, "height": 900})
            if s.name == "source-guides/columns-5"
        )
    scales = [
        e["font_size"] / originals[e["id"]]["font_size"]
        for e in state.design["elements"]
        if e["type"] == "text"
    ]
    assert min(scales) > 1.75
    assert max(scales) == pytest.approx(min(scales))


def test_normalized_native_frames_keep_exact_source_pixels_lines_and_identity(tmp_path):
    from engine.eval.render import render_image
    from engine.layout.source_guides import _normalize_column_rows, _rows
    from engine.layout.text import get_measurer

    folder = ROOT / "fixtures/open_posters/prepared/doory_botanical-celebration-invitation"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        context = layout_context(source)
        rows, signatures = _normalize_column_rows(_rows(context), context)
        assert signatures
        changed = {element["id"]: element for row in rows for element in row.elements}
        normalized = {**source, "elements": [changed.get(e["id"], e) for e in source["elements"]]}
        for old, new in zip(source["elements"], normalized["elements"], strict=True):
            assert {k: v for k, v in old.items() if k not in {"x", "width"}} == {
                k: v for k, v in new.items() if k not in {"x", "width"}
            }
            if old["id"] in signatures:
                assert (
                    get_measurer().measure(new, new["font_size"], new["width"]).wrapped
                    == signatures[old["id"]]
                )
        original_png, normalized_png = render_image(source), render_image(normalized)
        assert original_png.tobytes() == normalized_png.tobytes()
        original_png.save(tmp_path / "source.png")
        normalized_png.save(tmp_path / "normalized-source.png")
    assert source == before


@pytest.mark.parametrize("protection", ["group", "rotation", "justify", "gradient"])
def test_column_frame_normalization_skips_protected_text(protection):
    from engine.layout.source_guides import _normalize_column_rows, _rows

    folder = ROOT / "fixtures/open_posters/prepared/doory_botanical-celebration-invitation"
    original = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    element = deepcopy(next(e for e in original["elements"] if e.get("content") == "AVA"))
    if protection == "group":
        element["group_id"] = "authored-group"
    elif protection == "rotation":
        element["rotation"] = 10
    elif protection == "justify":
        element["text_align"] = "justify"
    else:
        element["native_text_effects"]["gradient"] = {
            "x1": 0,
            "y1": 0,
            "x2": 1,
            "y2": 0,
            "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}],
        }
    source = {"canvas": original["canvas"], "elements": [element]}
    before = deepcopy(source)
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        context = layout_context(source)
        rows, signatures = _normalize_column_rows(_rows(context), context)
    assert not signatures
    assert rows[0].elements[0] == element
    assert source == before


def test_column_frame_normalization_preserves_default_legacy_context():
    from engine.layout.source_guides import _normalize_column_rows, _rows
    from engine.resources import DEFAULT_ASSETS_DIR, DEFAULT_FONTS_DIR

    source = _simple()
    with using_resources(DEFAULT_ASSETS_DIR, DEFAULT_FONTS_DIR):
        context = layout_context(source)
        original = _rows(context)
        normalized, signatures = _normalize_column_rows(original, context)
    assert normalized == original
    assert not signatures


def test_full_candidate_pool_selects_readable_connected_invitation_columns(tmp_path):
    from engine.eval.render import render_image
    from engine.resize import resize_with_state

    folder = ROOT / "fixtures/open_posters/prepared/doory_botanical-celebration-invitation"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        plan = resize_with_state(source, {"width": 1600, "height": 900})
        design = plan["design"]
        names = [e for e in design["elements"] if e.get("content") in {"AVA", "THEODORE"}]
        venue = next(e for e in design["elements"] if e.get("content") == "The Orchard House")
        centers = [e["x"] + e["width"] / 2 for e in names]
        assert max(centers) - min(centers) < 10
        assert venue["x"] + venue["width"] / 2 - max(centers) > 1600 * 0.2
        assert min(e["font_size"] for e in design["elements"] if e["type"] == "text") > 18
        native = render_image(design)
        assert native.size == (1600, 900)
        native.save(tmp_path / "selected-invitation-landscape.png")
