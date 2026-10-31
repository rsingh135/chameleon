"""Region copy may wrap; native controls and source rows retain proportions."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.layout.geometry import Box, bounds
from engine.layout.media_text import MAX_ROWS, pack_information_region
from engine.layout.text import get_measurer
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]
MUTABLE = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


@pytest.fixture
def exact_fonts(tmp_path):
    filename = "SolsticeSans-Regular.ttf"
    (tmp_path / filename).write_bytes((ROOT / "brand_kit/fonts" / filename).read_bytes())
    (tmp_path / "font-manifest.json").write_text(
        json.dumps(
            {
                "fonts": [
                    {
                        "font_family": "ExactFace",
                        "file": filename,
                        "weight": "regular",
                        "style": "normal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _text(eid, content, x, y, width, height, size):
    return dict(
        id=eid,
        type="text",
        content=content,
        x=x,
        y=y,
        width=width,
        height=height,
        font_family="ExactFace",
        font_size=size,
        line_height=1.2,
        color="#222222",
        native_text_effects={"version": 1, "tracking_em": 0.04},
        z_index=2,
    )


@pytest.fixture
def scene():
    heading = _text("opaque-a", "Source title keeps its meaning", 0, 80, 800, 72, 60)
    body = _text(
        "opaque-b",
        "Live editable copy keeps its original wording and face while wrapping "
        "into a narrower information region. Bring a friend.",
        0,
        200,
        800,
        108,
        30,
    )
    body["style_runs"] = [{"start": 0, "end": 4, "color": "#aa0000"}]
    calendar = []
    for index in range(7):
        label = _text(f"opaque-{index + 2}", str(index + 18), index * 120, 406, 90, 40, 32)
        if index == 3:
            label["group_id"] = "native-control"
            calendar.append(
                dict(
                    id="opaque-support",
                    type="shape",
                    shape_type="rectangle",
                    x=index * 120,
                    y=394,
                    width=90,
                    height=64,
                    corner_radius=32,
                    fill_color="#eeeeee",
                    group_id="native-control",
                    z_index=1,
                )
            )
        calendar.append(label)
    rows = [[heading], [body], calendar]
    return {"canvas": {"width": 1000, "height": 1600}, "elements": [heading, body, *calendar]}, rows


def test_shared_font_budget_wraps_copy_and_preserves_all_source_identity(scene, exact_fonts):
    source, rows = scene
    before = deepcopy(source)
    cell = Box(10, 20, 420, 600)
    with using_resources(fonts_dir=exact_fonts):
        packed = pack_information_region(source, rows, cell, preferred_scale=0.6, min_font_size=13)
        again = pack_information_region(source, rows, cell, preferred_scale=0.6, min_font_size=13)
        assert packed is not None
        assert packed == again
        assert packed.font_scale < 0.6
        assert "opaque-a" in packed.wrapped_element_ids
        assert packed.exceptions
        assert [e["id"] for e in packed.elements] == [e["id"] for e in source["elements"]]
        by_id = {e["id"]: e for e in packed.elements}
        for original in source["elements"]:
            output = by_id[original["id"]]
            assert {k: v for k, v in output.items() if k not in MUTABLE} == {
                k: v for k, v in original.items() if k not in MUTABLE
            }
            box = bounds(output)
            assert cell.x <= box.x <= box.x + box.width <= cell.x + cell.width + 0.015
            assert cell.y <= box.y <= box.y + box.height <= cell.y + cell.height + 0.015
            if output["type"] == "text":
                measured = get_measurer().measure(output, output["font_size"], output["width"])
                assert measured.width <= output["width"] + 0.015
                assert measured.height <= output["height"] + 0.015
        for row in rows[:2]:
            original = row[0]
            assert by_id[original["id"]]["font_size"] / original["font_size"] == packed.font_scale
        scales = [
            by_id[e["id"]]["font_size"] / e["font_size"]
            for e in source["elements"]
            if e["type"] == "text"
        ]
        assert max(scales) == pytest.approx(min(scales), rel=1e-12)
    assert source == before


def test_complete_calendar_row_retains_every_frame_font_gap_and_radius_ratio(scene, exact_fonts):
    source, rows = scene
    with using_resources(fonts_dir=exact_fonts):
        packed = pack_information_region(source, rows, Box(0, 0, 420, 600), preferred_scale=0.6)
    assert packed is not None
    by_id = {e["id"]: e for e in packed.elements}
    row, scale = rows[-1], packed.rigid_row_scales[-1]
    assert scale == packed.font_scale
    reference = row[0]
    for original in row:
        output = by_id[original["id"]]
        for field in ("width", "height", "font_size", "corner_radius"):
            if field in original:
                assert output[field] == pytest.approx(original[field] * scale)
        for axis in ("x", "y"):
            assert output[axis] - by_id[reference["id"]][axis] == pytest.approx(
                (original[axis] - reference[axis]) * scale
            )


def test_height_search_stays_inside_font_budget_and_above_every_font_floor(scene, exact_fonts):
    source, rows = scene
    with using_resources(fonts_dir=exact_fonts):
        packed = pack_information_region(
            source, rows, Box(0, 0, 360, 170), preferred_scale=0.8, min_font_size=13
        )
    assert packed is not None
    assert 13 / 30 <= packed.font_scale < 0.8
    assert min(e["font_size"] for e in packed.elements if e["type"] == "text") >= 13 - 1e-9


def test_rigid_row_cannot_fit_by_shrinking_its_labels_below_floor(scene, exact_fonts):
    source, rows = scene
    with using_resources(fonts_dir=exact_fonts):
        assert (
            pack_information_region(
                source, rows, Box(0, 0, 250, 1000), preferred_scale=0.6, min_font_size=13
            )
            is None
        )


def test_unbreakable_word_or_max_lines_never_overflows_for_a_larger_font(scene, exact_fonts):
    source, rows = scene
    rows[0][0].update(content="M" * 100, max_lines=1)
    with using_resources(fonts_dir=exact_fonts):
        assert (
            pack_information_region(
                source, rows, Box(0, 0, 420, 1000), preferred_scale=0.6, min_font_size=13
            )
            is None
        )


@pytest.mark.parametrize("change", [{"group_id": "singleton"}, {"rotation": 5}])
def test_explicit_or_rotated_singleton_text_stays_rigid(scene, exact_fonts, change):
    source, rows = scene
    rows[0][0].update(content="Source title", **change)
    original = deepcopy(rows[0][0])
    with using_resources(fonts_dir=exact_fonts):
        packed = pack_information_region(source, rows, Box(0, 0, 420, 800), preferred_scale=0.6)
    assert packed is not None
    output = next(e for e in packed.elements if e["id"] == original["id"])
    scale = packed.rigid_row_scales[0]
    assert output["width"] == pytest.approx(original["width"] * scale)
    assert output["height"] == pytest.approx(original["height"] * scale)
    assert original["id"] not in packed.wrapped_element_ids


def test_split_or_partial_native_group_is_rejected(scene, exact_fonts):
    source, rows = scene
    support = next(e for e in rows[-1] if e["type"] == "shape")
    rows[-1].remove(support)
    with using_resources(fonts_dir=exact_fonts):
        assert (
            pack_information_region(source, rows, Box(0, 0, 420, 600), preferred_scale=0.6) is None
        )
    rows.append([support])
    with using_resources(fonts_dir=exact_fonts):
        assert (
            pack_information_region(source, rows, Box(0, 0, 420, 600), preferred_scale=0.6) is None
        )


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_budget_declines_without_touching_source(scene, invalid):
    source, rows = scene
    before = deepcopy(source)
    assert (
        pack_information_region(source, rows, Box(0, 0, 420, 600), preferred_scale=invalid) is None
    )
    assert source == before


def test_budget_exceeded_is_rejected_before_font_measurement(scene, monkeypatch):
    source, rows = scene

    def unexpected():
        raise AssertionError("Budget must be checked before expensive font measurement")

    monkeypatch.setattr("engine.layout.media_text.get_measurer", unexpected)
    assert (
        pack_information_region(
            source, rows * (MAX_ROWS + 1), Box(0, 0, 420, 600), preferred_scale=0.6
        )
        is None
    )


def test_actual_rooftop_information_wraps_at_common_scale_with_readable_native_calendar():
    from engine.layout.model import build_context
    from engine.layout.source_guides import _owned_units, _unit_rows

    directory = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        context = build_context(source)
        ordinary, plates, _ = _owned_units(context)
        rows = _unit_rows(context, ordinary + plates)
        packed = pack_information_region(
            source,
            [row.elements for row in rows],
            Box(582, 38, 460, 1004),
            preferred_scale=0.6,
            min_font_size=12.96,
        )
        assert packed is not None
        assert packed.font_scale <= 0.6
        assert packed.wrapped_element_ids
        assert min(e["font_size"] for e in packed.elements if e["type"] == "text") >= 12.96
        for original in (row.elements[0] for row in rows if len(row.elements) == 1):
            if original["type"] == "text":
                output = next(e for e in packed.elements if e["id"] == original["id"])
                assert output["font_size"] == pytest.approx(
                    original["font_size"] * packed.font_scale
                )
        by_id = {e["id"]: e for e in source["elements"]}
        scales = [
            e["font_size"] / by_id[e["id"]]["font_size"]
            for e in packed.elements
            if e["type"] == "text"
        ]
        assert max(scales) == pytest.approx(min(scales), rel=1e-12)
    assert source == before


def test_large_preferred_copy_cannot_diminish_calendar_prominence(scene, exact_fonts):
    source, rows = scene
    with using_resources(fonts_dir=exact_fonts):
        packed = pack_information_region(
            source, rows, Box(0, 0, 420, 1000), preferred_scale=0.95, min_font_size=13
        )
    assert packed is not None
    originals = {e["id"]: e for e in source["elements"]}
    scales = [
        e["font_size"] / originals[e["id"]]["font_size"]
        for e in packed.elements
        if e["type"] == "text"
    ]
    assert max(scales) == pytest.approx(min(scales), rel=1e-12)
    assert min(e["font_size"] for e in packed.elements if e["type"] == "text") >= 13


def test_common_scale_floor_failure_declines_instead_of_inflating_copy(scene, exact_fonts):
    source, rows = scene
    with using_resources(fonts_dir=exact_fonts):
        assert (
            pack_information_region(
                source, rows, Box(0, 0, 350, 1000), preferred_scale=0.8, min_font_size=13
            )
            is None
        )


def test_native_square_fade_stack_enlarges_complete_caption_without_crop_or_hierarchy_loss():
    from engine.eval import check_contract
    from engine.layout.geometry import union
    from engine.layout.reading_order import analyze_reading_order
    from engine.layout.relationships import analyze_relationships, build_relationship_graph
    from engine.layout.source_guides import generate_source_guides

    directory = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        target = {"width": 1080, "height": 1080}
        graph = build_relationship_graph(source)
        caption_ids = set(next(c for c in graph.clusters if c.kind == "caption").element_ids)
        states = [
            s
            for s in generate_source_guides(source, target)
            if s.topology == "source-media-stacked-fade"
        ]
        assert 1 <= len(states) <= 2
        for state in states:
            design = state.design
            assert check_contract(source, design, target) == []
            relation = analyze_relationships(graph, design)
            assert relation["warnings"] == []
            caption = next(r for r in relation["relations"] if r["code"] == "caption_hierarchy")
            assert caption["font_scale_ratio"] == 1
            assert (
                min(
                    e["font_size"]
                    for e in design["elements"]
                    if e["id"] in caption_ids and e["type"] == "text"
                )
                >= 20
            )
            order = analyze_reading_order(source, design)
            assert order["high_confidence_inversions"] == 0
            assert order["comparison_coverage"] == 1
            old = {e["id"]: e for e in source["elements"]}
            for element in design["elements"]:
                original = old[element["id"]]
                if element["type"] == "image":
                    assert element["width"] / original["width"] == pytest.approx(
                        element["height"] / original["height"]
                    )
            assert union([e for e in design["elements"] if e["id"] in caption_ids]).width > 650
    assert source == before


@pytest.mark.parametrize("unsupported", ["no-fade", "no-matte", "light-matte", "product"])
def test_fade_stack_declines_unsupported_photo_or_matte_evidence(unsupported):
    from engine.layout.source_guides import generate_source_guides

    directory = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    photo = next(
        e for e in source["elements"] if e.get("source_art", {}).get("kind") == "masked-photo"
    )
    if unsupported == "no-fade":
        next(e for e in source["elements"] if e.get("source_art", {}).get("kind") == "svg-nontext")[
            "asset_id"
        ] = photo["asset_id"]
    elif unsupported == "product":
        photo["role"] = "product"
    else:
        background = next(
            e
            for e in source["elements"]
            if e["type"] == "shape" and e["width"] >= source["canvas"]["width"]
        )
        if unsupported == "no-matte":
            source["elements"].remove(background)
        else:
            background["fill_color"] = "rgba(255,255,255,1)"
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        assert not [
            s
            for s in generate_source_guides(source, {"width": 1080, "height": 1080})
            if s.topology == "source-media-stacked-fade"
        ]


def test_fade_stack_is_square_specific_and_does_not_replace_landscape_media_columns():
    from engine.eval import check_contract
    from engine.layout.source_guides import generate_source_guides

    directory = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        target = {"width": 1600, "height": 900}
        states = generate_source_guides(source, target)
        assert not [s for s in states if s.topology == "source-media-stacked-fade"]
        leading = next(
            s for s in states if s.name == "source-guides/media-columns-leading-fullheight"
        )
        assert check_contract(source, leading.design, target) == []
