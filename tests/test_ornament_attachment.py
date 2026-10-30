"""Source ornaments retain their physical heading and relative prominence."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene(tmp_path):
    image = Image.new("RGBA", (20, 100))
    ImageDraw.Draw(image).line([(10, 0), (2, 25), (18, 50), (2, 75), (10, 99)], fill="black")
    image.save(tmp_path / "vector.png")
    source = {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [
            dict(
                id="opaque-a",
                type="image",
                asset_id="vector.png",
                fit="fill",
                source_art={"kind": "svg-nontext"},
                x=490,
                y=100,
                width=20,
                height=100,
            ),
            dict(
                id="opaque-b",
                type="text",
                content="Source heading",
                font_family="Solstice Sans",
                font_size=80,
                color="#222222",
                text_align="center",
                x=150,
                y=250,
                width=700,
                height=100,
            ),
            dict(
                id="opaque-c",
                type="text",
                content="Supporting copy",
                font_family="Solstice Sans",
                font_size=24,
                color="#222222",
                text_align="center",
                x=150,
                y=380,
                width=700,
                height=40,
            ),
        ],
    }
    return tmp_path, source


def _ornaments(graph):
    return [edge for edge in graph.edges if edge.code == "ornament_attachment"]


def test_sparse_source_vector_has_physical_attachment_without_product_semantics(scene):
    assets, source = scene
    before = deepcopy(source)
    graph = build_relationship_graph(source, assets)
    assert len(_ornaments(graph)) == 1
    edge = _ornaments(graph)[0]
    assert graph.units[edge.target_unit].element_ids == ("opaque-b",)
    assert edge.confidence == 0.9
    assert edge.provenance
    assert not [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    assert analyze_relationships(graph, source)["metrics"]["ornament_attachment"] == 1
    assert source == before


@pytest.mark.parametrize("scale", [0.5, 1.8])
def test_complete_ornament_heading_block_can_translate_and_scale(scene, scale):
    assets, source = scene
    candidate = deepcopy(source)
    for element in candidate["elements"]:
        for key in ("x", "y", "width", "height", "font_size"):
            if key in element:
                element[key] *= scale
        element["x"] += 73
        element["y"] += 44
    report = analyze_relationships(build_relationship_graph(source, assets), candidate)
    assert not [w for w in report["warnings"] if w["code"] == "ornament_attachment"]
    relation = next(r for r in report["relations"] if r["code"] == "ornament_attachment")
    assert relation["relative_scale_ratio"] == 1
    assert relation["normalized_gap_error"] == 0
    assert relation["normalized_alignment_error"] == 0


@pytest.mark.parametrize("change", ["inflate", "detach", "misalign", "reverse"])
def test_independent_ornament_inflation_or_detachment_is_rejected(scene, change):
    assets, source = scene
    candidate = deepcopy(source)
    ornament = candidate["elements"][0]
    if change == "inflate":
        ornament["width"] *= 4
        ornament["height"] *= 4
    elif change == "detach":
        candidate["elements"][1]["y"] += 200
    elif change == "misalign":
        ornament["x"] -= 300
    else:
        ornament["y"] = 500
    report = analyze_relationships(build_relationship_graph(source, assets), candidate)
    assert "ornament_attachment" in {w["code"] for w in report["warnings"]}


def test_opaque_names_and_changed_copy_do_not_change_source_attachment(scene):
    assets, source = scene
    first = build_relationship_graph(source, assets)
    for index, element in enumerate(source["elements"]):
        element["id"] = f"unrelated-{index}"
        if element["type"] == "text":
            element["content"] = "Other heading" if index == 1 else "Different copy"
    second = build_relationship_graph(source, assets)
    assert [(e.source_unit, e.target_unit, e.axis, e.alignment) for e in _ornaments(first)] == [
        (e.source_unit, e.target_unit, e.axis, e.alignment) for e in _ornaments(second)
    ]


def test_declared_product_and_opaque_vector_art_do_not_become_ornaments(scene):
    assets, source = scene
    source["elements"][0]["role"] = "product"
    assert _ornaments(build_relationship_graph(source, assets)) == []
    source["elements"][0].pop("role")
    Image.new("RGBA", (20, 100), "black").save(assets / "vector.png")
    assert _ornaments(build_relationship_graph(source, assets)) == []


def test_ambiguous_equal_heading_owners_are_not_guessed(scene):
    assets, source = scene
    source["elements"].append({**source["elements"][1], "id": "another-heading"})
    assert _ornaments(build_relationship_graph(source, assets)) == []


def test_alignment_across_remote_blank_space_is_not_physical_attachment(scene):
    assets, source = scene
    source["elements"][1]["y"] = 570
    source["elements"][2]["y"] = 730
    assert _ornaments(build_relationship_graph(source, assets)) == []


def test_equal_area_does_not_hide_ornament_shape_deformation(scene):
    assets, source = scene
    candidate = deepcopy(source)
    candidate["elements"][0].update(x=460, y=175, width=80, height=25)
    report = analyze_relationships(build_relationship_graph(source, assets), candidate)
    relation = next(r for r in report["relations"] if r["code"] == "ornament_attachment")
    assert relation["relative_scale_ratio"] == 1
    assert relation["relative_width_scale"] == 4
    assert relation["relative_height_scale"] == 0.25
    assert "ornament_attachment" in {w["code"] for w in report["warnings"]}


def _botanical():
    directory = ROOT / "fixtures/open_posters/prepared/doory_botanical-celebration-invitation"
    return directory, json.loads((directory / "source.json").read_text(encoding="utf-8"))


def test_actual_botanical_source_columns_keep_ornament_with_its_heading():
    from engine.layout.source_guides import generate_source_guides

    directory, source = _botanical()
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        graph = build_relationship_graph(source)
        assert len(_ornaments(graph)) == 1
        state = next(
            s
            for s in generate_source_guides(source, {"width": 1600, "height": 900})
            if s.name == "source-guides/columns-5"
        )
        report = analyze_relationships(graph, state.design)
        assert not [w for w in report["warnings"] if w["code"] == "ornament_attachment"]
        assert not [cluster for cluster in graph.clusters if cluster.kind == "campaign"]


def test_actual_botanical_full_candidate_pool_rejects_giant_detached_ornament():
    from engine.resize import resize

    directory, source = _botanical()
    with using_resources(directory / "resources/assets", directory / "resources/fonts"):
        graph = build_relationship_graph(source)
        selected = resize(source, {"width": 1600, "height": 900})
        report = analyze_relationships(graph, selected)
        assert not [w for w in report["warnings"] if w["code"] == "ornament_attachment"]
        relation = next(r for r in report["relations"] if r["code"] == "ornament_attachment")
        assert 1 / 1.15 <= relation["relative_scale_ratio"] <= 1.15
        assert relation["normalized_alignment_error"] <= 0.18
