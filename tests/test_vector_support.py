"""Tiny sparse vector support does not establish independent product semantics."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from engine.layout.focal import infer_focal_profile
from engine.layout.relationships import build_relationship_graph
from engine.layout.source_containers import infer_source_containers
from engine.resources import using_resources


def _source(assets, *, size=80, dense=False, colored=False, kind="svg-nontext", role=None):
    image = Image.new("RGBA", (80, 80))
    draw = ImageDraw.Draw(image)
    if dense:
        draw.ellipse((10, 10, 70, 70), fill="#333333")
    else:
        draw.line((10, 70, 40, 10, 70, 70), fill="#ff0000" if colored else "#333333", width=2)
        if colored:
            draw.line((10, 30, 70, 30), fill="#0000ff", width=2)
    image.save(assets / "graphic.png")
    graphic = {
        "id": "graphic",
        "type": "image",
        "x": 500 - size / 2,
        "y": 300,
        "width": size,
        "height": size,
        "asset_id": "graphic.png",
        "fit": "fill",
        "source_art": {"kind": kind},
    }
    if role:
        graphic["role"] = role

    def text(eid, y, content, size):
        return {
            "id": eid,
            "type": "text",
            "x": 250,
            "y": y,
            "width": 500,
            "height": 100,
            "content": content,
            "font_size": size,
            "font_family": "Solstice Sans",
            "text_align": "center",
            "color": "rgba(0,0,0,1)",
        }

    return {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [
            graphic,
            text("headline", 100, "Title", 80),
            text("body", 530, "Product details", 32),
            {
                "id": "action",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 350,
                "y": 780,
                "width": 300,
                "height": 80,
                "fill_color": "rgba(200,200,200,1)",
                "group_id": "action",
            },
            {
                **text("action-label", 790, "Join", 30),
                "group_id": "action",
                "x": 375,
                "width": 250,
                "height": 55,
            },
        ],
    }


def test_small_sparse_vector_keeps_literal_group_but_does_not_infer_product(tmp_path):
    source = _source(tmp_path)
    graph = build_relationship_graph(source, tmp_path)
    assert not any(cluster.kind == "campaign" for cluster in graph.clusters)
    assert any("graphic" in unit.element_ids for unit in graph.units)
    assert infer_focal_profile(source, graph) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"size": 400},
        {"dense": True},
        {"colored": True},
        {"kind": "photo"},
        {"kind": "masked-photo"},
        {"role": "product"},
    ],
)
def test_large_dense_colored_declared_and_cutout_products_retain_ownership(tmp_path, changes):
    source = _source(tmp_path, **changes)
    graph = build_relationship_graph(source, tmp_path)
    assert len([cluster for cluster in graph.clusters if cluster.kind == "campaign"]) == 1
    assert infer_focal_profile(source, graph) is not None


def test_unknown_image_bytes_cannot_prove_sparse_support(tmp_path):
    source = _source(tmp_path)
    (tmp_path / "graphic.png").unlink()
    graph = build_relationship_graph(source, tmp_path)
    assert any(cluster.kind == "campaign" for cluster in graph.clusters)


def test_actual_grouped_botanical_sprig_is_not_an_independent_product():
    root = (
        Path(__file__).resolve().parents[1]
        / "fixtures/open_posters/prepared/doory_botanical-celebration-invitation"
    )
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    with using_resources(assets_dir=root / "resources/assets", fonts_dir=root / "resources/fonts"):
        graph = build_relationship_graph(source)
        assert not any(cluster.kind == "campaign" for cluster in graph.clusters)
        assert infer_focal_profile(source, graph) is None
    assert any(
        len(unit.element_ids) == 2
        and all(member.element_type == "image" for member in unit.members)
        for unit in graph.units
    )


def test_proven_source_panel_frames_do_not_acquire_unrelated_product_messages():
    root = (
        Path(__file__).resolve().parents[1] / "fixtures/open_posters/prepared/opensuse_cheatsheet"
    )
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    with using_resources(assets_dir=root / "resources/assets", fonts_dir=root / "resources/fonts"):
        profile = infer_source_containers(source)
        frames = {container.frame_id for container in profile.containers}
        assert len(frames) == 7
        graph = build_relationship_graph(source)
        campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
        assert not any(frames.intersection(cluster.element_ids) for cluster in campaigns)
        assert frames.issubset({eid for unit in graph.units for eid in unit.element_ids})

        # The exception requires actual source-supported frame evidence. A
        # photo with the same bounding box still carries product ownership.
        unproven = deepcopy(source)
        for element in unproven["elements"]:
            if element["id"] in frames:
                element["source_art"]["kind"] = "photo"
        assert not infer_source_containers(unproven).containers
        unknown_graph = build_relationship_graph(unproven)
        assert any(
            frames.intersection(cluster.element_ids)
            for cluster in unknown_graph.clusters
            if cluster.kind == "campaign"
        )

        declared = deepcopy(source)
        for element in declared["elements"]:
            if element["id"] in frames:
                element["role"] = "product"
        declared_graph = build_relationship_graph(declared)
        assert any(
            frames.intersection(cluster.element_ids)
            for cluster in declared_graph.clusters
            if cluster.kind == "campaign"
        )
    assert source == before
