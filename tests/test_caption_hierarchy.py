"""A shared photo caption keeps source prominence across all candidate families."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from engine.layout.model import build_context
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene(tmp_path):
    Image.new("RGBA", (100, 160), "#778899").save(tmp_path / "photo.png")
    fade = Image.new("RGBA", (100, 80))
    fade.putdata([(0, 0, 0, round(255 * y / 79)) for y in range(80) for _ in range(100)])
    fade.save(tmp_path / "fade.png")

    def text(eid, copy, y, size, height):
        return dict(
            id=eid,
            type="text",
            content=copy,
            x=100,
            y=y,
            width=800,
            height=height,
            font_family="Solstice Sans",
            font_size=size,
            color="#ffffff",
            z_index=3,
        )

    elements = [
        dict(
            id="opaque-photo",
            type="image",
            asset_id="photo.png",
            fit="fill",
            source_art={"kind": "masked-photo"},
            x=0,
            y=0,
            width=1000,
            height=1600,
            z_index=0,
        ),
        dict(
            id="opaque-fade",
            type="image",
            asset_id="fade.png",
            fit="fill",
            x=0,
            y=800,
            width=1000,
            height=800,
            z_index=1,
        ),
        text("opaque-header", "Independent photo heading", 100, 26, 50),
        text("opaque-title", "Event", 1000, 108, 140),
        text(
            "opaque-body",
            "Supporting event information stays connected to its date and controls.",
            1200,
            30,
            80,
        ),
        dict(
            id="opaque-plate",
            type="shape",
            shape_type="rectangle",
            group_id="date",
            x=90,
            y=1400,
            width=120,
            height=64,
            fill_color="#ffffff",
            z_index=2,
        ),
        {
            **text("opaque-date", "21", 1410, 32, 43),
            "width": 100,
            "group_id": "date",
            "color": "#111111",
        },
        text(
            "opaque-legal",
            "Legal information stays distinct from the main event copy and dates.",
            1535,
            12,
            36,
        ),
    ]
    return tmp_path, {"canvas": {"width": 1000, "height": 1600}, "elements": elements}


def _cohort(graph):
    return next(cluster for cluster in graph.clusters if cluster.kind == "caption")


def _caption_warnings(graph, candidate):
    return [
        w
        for w in analyze_relationships(graph, candidate)["warnings"]
        if w["code"] == "caption_hierarchy"
    ]


def _scaled(source, scale):
    candidate = deepcopy(source)
    candidate["canvas"] = {key: value * scale for key, value in source["canvas"].items()}
    for element in candidate["elements"]:
        for field in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"):
            if field in element:
                element[field] *= scale
    return candidate


def test_source_fade_cohort_includes_main_caption_and_date_but_not_header_or_fine_legal(scene):
    assets, source = scene
    before = deepcopy(source)
    graph = build_relationship_graph(source, assets)
    assert set(_cohort(graph).element_ids) == {
        "opaque-title",
        "opaque-body",
        "opaque-date",
        "opaque-plate",
    }
    assert _caption_warnings(graph, source) == []
    assert not [c for c in graph.clusters if c.kind == "campaign"]
    assert source == before


@pytest.mark.parametrize("scale", [0.5, 1.7])
def test_caption_source_ratios_survive_a_common_scale_and_translation(scene, scale):
    assets, source = scene
    candidate = _scaled(source, scale)
    for element in candidate["elements"]:
        element["x"] += 43
        element["y"] += 17
    report = analyze_relationships(build_relationship_graph(source, assets), candidate)
    assert not [w for w in report["warnings"] if w["code"] == "caption_hierarchy"]
    relation = next(r for r in report["relations"] if r["code"] == "caption_hierarchy")
    assert relation["font_scale_ratio"] == 1


@pytest.mark.parametrize("factor,warns", [(1.15, False), (1.151, True), (2.0, True)])
def test_large_body_or_small_calendar_cannot_pass_only_an_absolute_font_floor(scene, factor, warns):
    assets, source = scene
    candidate = deepcopy(source)
    candidate["elements"][4]["font_size"] *= factor
    graph = build_relationship_graph(source, assets)
    assert bool(_caption_warnings(graph, candidate)) is warns
    assert min(e["font_size"] for e in candidate["elements"] if e["type"] == "text") >= 12


def test_photo_header_and_real_fine_legal_can_retain_their_separate_font_floor(scene):
    assets, source = scene
    candidate = _scaled(source, 0.5)
    by_id = {e["id"]: e for e in candidate["elements"]}
    by_id["opaque-legal"]["font_size"] = 10
    by_id["opaque-header"].update(font_size=26, height=34)
    assert _caption_warnings(build_relationship_graph(source, assets), candidate) == []


def test_calendar_misclassified_as_legal_is_kept_by_supporting_font_median(scene, monkeypatch):
    assets, source = scene
    context = build_context(source)
    date = next(u for u in context.units if any(e["id"] == "opaque-date" for e in u.elements))
    date.role = "legal"
    monkeypatch.setattr("engine.layout.relationships.build_context", lambda _source: context)
    graph = build_relationship_graph(source, assets)
    assert "opaque-date" in _cohort(graph).element_ids
    assert "opaque-legal" not in _cohort(graph).element_ids
    candidate = deepcopy(source)
    candidate["elements"][6]["font_size"] *= 0.6
    assert _caption_warnings(graph, candidate)


def test_source_names_and_copy_do_not_define_caption_membership(scene):
    assets, source = scene
    expected = [u for u in _cohort(build_relationship_graph(source, assets)).unit_indices]
    for index, element in enumerate(source["elements"]):
        element["id"] = f"renamed-{index}"
        if element["type"] == "text":
            element["content"] = "".join(
                "A" if char.isalpha() else char for char in element["content"]
            )
    assert list(_cohort(build_relationship_graph(source, assets)).unit_indices) == expected


@pytest.mark.parametrize("change", [{"source_art": {"kind": "svg-nontext"}}, {"width": 900}])
def test_unproven_vector_or_non_full_bleed_underlay_does_not_invent_caption_ownership(
    scene, change
):
    assets, source = scene
    source["elements"][0].update(change)
    graph = build_relationship_graph(source, assets)
    assert not [c for c in graph.clusters if c.kind == "caption"]


def test_same_pixel_evidence_supports_a_photo_already_classified_as_canvas_background(scene):
    assets, source = scene
    source["elements"][0]["fit"] = "cover"
    graph = build_relationship_graph(source, assets)
    assert set(_cohort(graph).element_ids) >= {"opaque-title", "opaque-body", "opaque-date"}


@pytest.mark.parametrize(
    "target", [{"width": 1080, "height": 1080}, {"width": 1600, "height": 900}]
)
def test_actual_rooftop_full_candidate_pool_keeps_calendar_source_prominence(target):
    from engine.resize import resize

    directory = ROOT / "fixtures/open_posters/prepared/doory_rooftop-event-story"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        graph = build_relationship_graph(source)
        selected = resize(source, target)
        assert _caption_warnings(graph, selected) == []
        relation = next(
            r
            for r in analyze_relationships(graph, selected)["relations"]
            if r["code"] == "caption_hierarchy"
        )
        assert relation["font_scale_ratio"] <= 1.15
        assert min(e["font_size"] for e in selected["elements"] if e["type"] == "text") >= (
            max(12, min(target.values()) * 0.012) - 1e-6
        )
    assert source == before
