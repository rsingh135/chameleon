"""Actual resvg/font imports, explicit unsupported cases and immutable originals."""

import hashlib
import io
import json
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont
from PIL import Image

from tools.poster_import import (
    ImportBlock,
    main,
    parse_transform,
    prepare_poster,
    rectangle_path_bounds,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def face(tmp_path):
    original = sorted((ROOT / "brand_kit/fonts").glob("*.ttf"))[0]
    data = original.read_bytes()
    with TTFont(original) as font:
        names = font["name"]
        family = next(n.toUnicode() for n in names.names if n.nameID == 1)
        ps = next(n.toUnicode() for n in names.names if n.nameID == 6)
        weight = font["OS/2"].usWeightClass
        italic = bool(font["head"].macStyle & 2)
    (tmp_path / "face.ttf").write_bytes(data)
    return {
        "family": family,
        "weight": weight,
        "style": "italic" if italic else "normal",
        "path": "face.ttf",
        "postscript_name": ps,
        "resolution": "exact",
    }


def svg(body, style="", extra=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="360" height="260" '
        f'viewBox="0 0 360 260" {extra}><style>{style}</style>{body}</svg>'
    ).encode()


def text(face, content="Exact face", **extra):
    attrs = " ".join(f'{key}="{value}"' for key, value in extra.items())
    return (
        f'<text id="title" x="30" y="65" font-family="{face["family"]}" '
        f'font-weight="{face["weight"]}" font-style="{face["style"]}" '
        f'font-size="24" {attrs}>{content}</text>'
    )


def prepare(tmp_path, raw, fonts=()):
    (tmp_path / "original.svg").write_bytes(raw)
    entry = {
        "id": "synthetic-case",
        "title": "Importer test, not a public poster",
        "split": "reviewed",
        "author": "test",
        "source_url": "https://example.invalid/test.svg",
        "license_id": "test",
        "license_url": "https://example.invalid/license",
        "license_path": "LICENSE",
        "original_path": "original.svg",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "font_requirements": list(fonts),
    }
    report = prepare_poster(entry, tmp_path, tmp_path / "prepared")
    return report, tmp_path / "prepared"


def test_real_svg_text_keeps_exact_font_bytes_and_live_text(tmp_path, face):
    raw = svg(
        '<rect width="360" height="260" fill="white"/>'
        + text(face)
        + '<g id="icon"><path d="M30 120L80 120L60 170Z" fill="#e45330"/>'
        '<path d="M85 120L130 170" stroke="black" stroke-width="4"/></g>'
    )
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    assert (out / "original.svg").read_bytes() == raw
    assert report["editable_text_count"] == 1
    assert report["vector_art_count"] == 1
    source = json.loads((out / "source.json").read_text())
    title = next(e for e in source["elements"] if e["type"] == "text")
    assert title["content"] == "Exact face"
    assert title["font_family"] == face["postscript_name"]
    assert title["font_size"] == 24
    manifest = json.loads((out / "resources/fonts/font-manifest.json").read_text())
    font_file = out / "resources/fonts" / manifest["fonts"][0]["file"]
    assert font_file.read_bytes() == (tmp_path / "face.ttf").read_bytes()
    assert manifest["fonts"][0]["font_family"] == face["postscript_name"]
    assert Image.open(out / "native.png").size == (360, 260)
    assert report["native_imported_normalized_mae"] < 0.02
    assert report["parity_review"] == "pending"


def test_solid_shapes_use_native_schema_and_keep_transform_and_z_order(tmp_path):
    raw = svg(
        '<rect x="1" y="2" width="350" height="250" fill="white"/>'
        '<g transform="translate(40 30) rotate(30)">'
        '<rect id="tile" x="0" y="0" width="50" height="20" fill="red"/></g>'
        '<circle cx="200" cy="150" r="20" fill="blue"/>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text())
    assert [e["z_index"] for e in source["elements"]] == [0, 1, 2]
    assert all(e["type"] == "shape" for e in source["elements"])
    assert source["elements"][1]["rotation"] == pytest.approx(30)
    assert source["elements"][1]["width"] == 50
    assert report["editable_text_count"] == 0


def test_native_reference_keeps_transparency_without_inventing_a_paper_element(tmp_path):
    report, out = prepare(
        tmp_path,
        svg(
            '<rect x="20" y="20" width="40" height="40" fill="red"/>'
            '<circle cx="100" cy="100" r="20" fill="blue"/>'
        ),
    )
    assert report["status"] == "ready", report["failures"]
    assert Image.open(out / "native.png").convert("RGBA").getpixel((0, 0))[3] == 0
    source = json.loads((out / "source.json").read_text())
    assert len(source["elements"]) == 2
    assert "white paper" in report["comparison_paper"]


@pytest.mark.parametrize(
    "feature",
    [
        '<g clip-path="url(#clip)"><rect width="80" height="80"/></g>',
        '<path filter="url(#blur)" d="M10 10L70 70"/>',
        '<text><textPath href="#curve">Do not rasterize me</textPath></text>',
        '<foreignObject width="10" height="10"/>',
        '<unknownPaint><rect width="20" height="20"/></unknownPaint>',
        '<use href="#shape"/>',
        '<image href="https://example.invalid/image.png" width="20" height="20"/>',
    ],
)
def test_unsupported_features_produce_a_report_and_no_runnable_bundle(tmp_path, feature):
    report, out = prepare(tmp_path, svg(feature))
    assert report["status"] == "unsupported", report
    assert report["failures"]
    assert not (out / "bundle.json").exists()
    assert (out / "original.svg").is_file()


def test_missing_or_wrong_font_never_becomes_rasterized_editable_text(tmp_path, face):
    report, out = prepare(tmp_path, svg(text(face)), [])
    assert report["status"] == "unsupported"
    assert any(f["code"] == "missing_font" for f in report["failures"])
    assert not (out / "bundle.json").exists()
    wrong = {**face, "postscript_name": "Invented-Substitute"}
    report, _ = prepare(tmp_path, svg(text(face)), [wrong])
    assert any(f["code"] == "font_face_mismatch" for f in report["failures"])


def test_unused_font_styles_and_defs_do_not_require_or_import_hidden_text(tmp_path):
    raw = svg(
        '<defs><text font-family="Unavailable">unused</text></defs>'
        '<rect x="20" y="20" width="40" height="40"/>'
        '<circle cx="100" cy="100" r="20"/>',
        ".unused { font-family: Missing; }",
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    assert report["editable_text_count"] == 0
    assert len(json.loads((out / "source.json").read_text())["elements"]) == 2


def test_whole_document_and_inkscape_layer_wrappers_are_not_atomic(tmp_path, face):
    raw = svg(
        '<g id="whole-document"><g xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        'inkscape:groupmode="layer" id="layer1">'
        + text(face)
        + '<rect x="30" y="120" width="70" height="40" fill="red"/>'
        '<circle cx="240" cy="170" r="20" fill="blue"/></g></g>'
    )
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    assert all(
        not e.get("group_id") for e in json.loads((out / "source.json").read_text())["elements"]
    )
    assert any(g["whole_document"] for g in report["authored_groups"])
    assert any(g["layer"] for g in report["authored_groups"])


def test_compact_authored_label_component_remains_one_group(tmp_path, face):
    raw = svg(
        '<rect width="360" height="260" fill="white"/><g id="badge">'
        '<rect x="20" y="20" width="200" height="70" fill="orange"/>'
        + text(face, "Badge")
        + '</g><circle cx="280" cy="180" r="20"/>'
    )
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text())
    grouped = [e for e in source["elements"] if e.get("group_id")]
    assert len(grouped) == 2
    assert len({e["group_id"] for e in grouped}) == 1


def test_whole_poster_raster_fallback_is_explicitly_refused(tmp_path):
    raw = svg('<g><path d="M20 20L300 20L300 240Z" fill="red"/></g>')
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "unsupported"
    assert report["failures"][0]["code"] == "whole_poster_raster"
    assert not (out / "bundle.json").exists()


def test_authored_position_lists_become_exact_editable_character_objects(tmp_path, face):
    raw = svg(text(face, '<tspan x="20 40" y="60 60">AB</tspan>'))
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert [element["content"] for element in source["elements"]] == ["A", "B"]
    assert all(element["font_size"] == 24 for element in source["elements"])
    assert {mapping["source_complete_text"] for mapping in report["node_mapping"]} == {"AB"}
    assert all(
        mapping["segmentation"] == "authored_character_positions"
        for mapping in report["node_mapping"]
    )
    assert report["native_imported_normalized_mae"] < 0.01


def test_hash_mismatch_is_not_a_different_poster_in_disguise(tmp_path):
    raw = svg('<rect width="10" height="10"/>')
    (tmp_path / "original.svg").write_bytes(raw)
    report = prepare_poster(
        {"id": "wrong", "original_path": "original.svg", "sha256": "0" * 64},
        tmp_path,
        tmp_path / "prepared",
    )
    assert report["status"] == "unsupported"
    assert report["failures"][0]["code"] == "source_hash_mismatch"


def test_transform_order_is_svg_order_and_invalid_syntax_is_rejected():
    assert parse_transform("translate(10,20) scale(2)") == (2, 0, 0, 2, 10, 20)
    with pytest.raises(ImportBlock):
        parse_transform("translate(10) garbage")


def test_internal_use_and_nontext_clips_keep_their_actual_pixels(tmp_path):
    raw = svg(
        '<defs><g id="art"><path d="M0 0H40V40H0Z" fill="blue"/></g>'
        '<clipPath id="clip"><rect x="100" y="100" width="25" height="30"/></clipPath></defs>'
        '<rect width="360" height="260" fill="white"/>'
        '<use href="#art" x="30" y="30"/>'
        '<g clip-path="url(#clip)"><rect x="100" y="100" width="60" height="60" fill="red"/></g>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    assert report["vector_art_count"] == 1
    assert report["native_shape_count"] == 2
    assert report["native_imported_normalized_mae"] < 0.005
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert all(
        element["source_art"]["kind"] == "svg-nontext"
        for element in source["elements"]
        if element["type"] == "image"
    )


def test_text_clip_may_only_be_omitted_when_it_changes_no_source_pixels(tmp_path, face):
    definition = '<defs><clipPath id="clip"><rect width="360" height="260"/></clipPath></defs>'
    report, _ = prepare(
        tmp_path, svg(definition + text(face, **{"clip-path": "url(#clip)"})), [face]
    )
    assert report["status"] == "ready", report["failures"]
    definition = (
        '<defs><clipPath id="clip"><rect x="30" y="30" width="20" height="80"/></clipPath></defs>'
    )
    report, out = prepare(
        tmp_path, svg(definition + text(face, **{"clip-path": "url(#clip)"})), [face]
    )
    assert report["status"] == "unsupported"
    assert any(failure["code"] == "clipped_text" for failure in report["failures"])
    assert not (out / "bundle.json").exists()


def test_declared_adapted_reference_does_not_replace_author_original(tmp_path, face):
    original = svg(text(face, "Author original"))
    reference = svg(text(face, "Disclosed adapted reference"))
    (tmp_path / "original.svg").write_bytes(original)
    (tmp_path / "adapted.svg").write_bytes(reference)
    entry = {
        "id": "adaptation",
        "original_path": "original.svg",
        "sha256": hashlib.sha256(original).hexdigest(),
        "prepared_reference_path": "adapted.svg",
        "font_requirements": [face],
        "baseline_kind": "disclosed_open_font_adaptation",
        "adaptations": [{"type": "font_change", "reason": "explicit test adaptation"}],
    }
    report = prepare_poster(entry, tmp_path, tmp_path / "prepared")
    assert report["status"] == "ready", report["failures"]
    assert (tmp_path / "prepared/original.svg").read_bytes() == original
    assert (tmp_path / "prepared/reference.svg").read_bytes() == reference
    assert report["original_sha256"] != report["reference_sha256"]
    bundle = json.loads((tmp_path / "prepared/bundle.json").read_text(encoding="utf-8"))
    assert bundle["task_id"] == bundle["poster_id"] == "adaptation"
    assert bundle["attribution"]["original_sha256"] == entry["sha256"]


def test_default_svg_cli_leaves_other_source_adapters_alone(tmp_path, monkeypatch):
    entries = [{"id": "svg", "original_path": "a.svg"}, {"id": "json", "original_path": "b.json"}]
    manifest = tmp_path / "catalogue.json"
    manifest.write_text(json.dumps({"examples": entries}), encoding="utf-8")
    seen = []

    def record(entry, *_):
        seen.append(entry["id"])
        return {"status": "ready", "failures": []}

    monkeypatch.setattr("tools.poster_import.prepare_poster", record)
    assert main(["--manifest", str(manifest)]) == 0
    assert seen == ["svg"]


def test_root_viewport_crop_preserves_native_pixels_and_complete_editable_frame(tmp_path, face):
    import resvg_py

    raw = svg(text(face, "Original edge crop").replace('x="30"', 'x="-25"'))
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert source["elements"][0]["content"] == "Original edge crop"
    assert source["elements"][0]["x"] < 0
    assert report["source_warnings"][0]["code"] == "root_viewport_clipped_text"
    native = resvg_py.svg_to_bytes(
        svg_string=raw.decode(),
        width=360,
        height=260,
        dpi=96,
        skip_system_fonts=True,
        font_files=[str(tmp_path / "face.ttf")],
    )
    assert Image.open(io.BytesIO(native)).convert("RGBA").tobytes() == (
        Image.open(out / "native.png").convert("RGBA").tobytes()
    )
    assert report["native_imported_normalized_mae"] < 0.01


def test_adapted_reference_hash_mismatch_fails_before_import(tmp_path):
    raw = svg('<rect width="80" height="50"/>')
    (tmp_path / "original.svg").write_bytes(raw)
    (tmp_path / "reference.svg").write_bytes(raw)
    report = prepare_poster(
        {
            "id": "mismatch",
            "original_path": "original.svg",
            "prepared_reference_path": "reference.svg",
            "prepared_reference_sha256": "0" * 64,
        },
        tmp_path,
        tmp_path / "prepared",
    )
    assert report["status"] == "unsupported"
    assert report["failures"][0]["code"] == "reference_hash_mismatch"
    assert not (tmp_path / "prepared/bundle.json").exists()


@pytest.mark.parametrize(
    "commands",
    [
        "M10 20H50V40H10Z",
        "m10 20h40v20h-40z",
        "M10,20 50,20 50,40 10,40z",
        "m10 20 40 0 0 20 -40 0 z m0 0",
        "M1e1 2e1H5e1V4e1H1e1z",
        "M10 20V40H50V20Z",
        "M30 20H50V40H10V20H30Z M0 0 M9 9L9 9Z",
    ],
)
def test_proven_rectangle_paths_accept_relative_implicit_and_degenerate_subpaths(commands):
    assert rectangle_path_bounds(commands) == pytest.approx((10, 20, 40, 20))


def test_genuine_rectangle_path_example_is_proven_without_source_ids():
    assert rectangle_path_bounds(
        "m 0,430.86719 1261.418,0 L 1261.418,0 0,0 Z m 0,0"
    ) == pytest.approx((0, 0, 1261.418, 430.86719))


@pytest.mark.parametrize(
    "commands",
    [
        "M0 0L20 20L0 20Z",  # diagonal
        "M0 0H40V10H10V40H0Z",  # rectilinear L-shape
        "M0 0H40V20H0",  # open contour
        "M0 0H40V20V0H0Z",  # boundary backtracking, no rectangular fill
        "M0 0H40V20H0Z M10 5H20V10H10Z",  # hole/compound area
        "M0 0H40V20H0Z M0 0H40V20H0Z",  # repeated contours depend on fill-rule
        "M0 0H40V20H0Z M100 100L101 100",  # extra line is not a point subpath
        "M0 0C0 0 40 0 40 20H0Z",  # curve commands are never simplified
        "M0 0Q20 0 40 20H0Z",
        "M0 0A20 20 0 0 1 40 20H0Z",
        "M0 0H40V20H0Z M",  # incomplete data
        "M0 0H40V20H0Z nonsense",
        "M0 0L0 0Z",  # no area
        "M-1e308 -1e308H1e308V1e308H-1e308Z",  # overflow cannot be proved
    ],
)
def test_general_or_uncertain_paths_are_not_promoted_to_rectangles(commands):
    assert rectangle_path_bounds(commands) is None


def test_proven_path_and_native_rect_have_identical_editable_geometry_and_pixels(tmp_path):
    attributes = 'fill="#003342" fill-opacity=".7"'
    path = f'<path d="M10 20h80v50h-80z" {attributes}/>'
    rect = f'<rect x="10" y="20" width="80" height="50" {attributes}/>'
    wrapper = '<g transform="translate(30 20) rotate(25) scale(2 1.5)">{}</g>'
    report, out = prepare(tmp_path, svg(wrapper.format(path)))
    assert report["status"] == "ready", report["failures"]
    from_path = json.loads((out / "source.json").read_text(encoding="utf-8"))
    rendered_path = Image.open(out / "imported.png").convert("RGBA").tobytes()
    assert report["node_mapping"][0]["rectangle_path_proven"]
    assert report["native_shape_count"] == 1
    assert report["vector_art_count"] == 0
    assert report["native_imported_normalized_mae"] < 0.01
    report, out = prepare(tmp_path, svg(wrapper.format(rect)))
    assert report["status"] == "ready", report["failures"]
    assert json.loads((out / "source.json").read_text(encoding="utf-8")) == from_path
    assert Image.open(out / "imported.png").convert("RGBA").tobytes() == rendered_path


def test_rectangle_path_in_authored_text_component_keeps_group_and_z_order(tmp_path, face):
    raw = svg(
        '<g id="badge"><path d="M20 20h200v70H20Z" fill="orange"/>'
        + text(face, "Badge")
        + '</g><circle cx="280" cy="180" r="20"/>'
    )
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert source["elements"][0]["type"] == "shape"
    assert source["elements"][0]["group_id"] == source["elements"][1]["group_id"]
    assert [element["z_index"] for element in source["elements"]] == [0, 1, 2]


def test_stroked_rectangle_path_retains_art_when_native_stroke_is_not_supported(tmp_path):
    raw = svg(
        '<rect width="360" height="260" fill="white"/>'
        '<path d="M30 30h80v60H30Z" fill="red" stroke="blue" stroke-width="8"/>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    assert report["vector_art_count"] == 1
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert source["elements"][1]["type"] == "image"
    assert report["native_imported_normalized_mae"] < 0.005


def test_rectangle_path_with_vertex_marker_retains_complete_art(tmp_path):
    raw = svg(
        '<defs><marker id="mark" markerWidth="12" markerHeight="12" refX="6" refY="6">'
        '<circle cx="6" cy="6" r="6" fill="blue"/></marker></defs>'
        '<rect width="360" height="260" fill="white"/>'
        '<path d="M50 50h80v60H50Z" fill="red" marker-start="url(#mark)"/>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    assert report["vector_art_count"] == 1
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert source["elements"][1]["type"] == "image"
    assert report["native_imported_normalized_mae"] < 0.005


@pytest.mark.parametrize("clip_width, nonbinding", [(120, True), (20, False)])
def test_native_rectangle_preserves_a_binding_clip_as_exact_intersection(
    tmp_path, clip_width, nonbinding
):
    raw = svg(
        '<defs><clipPath id="clip"><rect x="30" y="20" '
        f'width="{clip_width}" height="100"/></clipPath></defs>'
        '<rect width="360" height="260" fill="white"/>'
        '<g clip-path="url(#clip)"><path d="M30 30h80v60H30Z" fill="red"/></g>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert source["elements"][1]["type"] == "shape"
    if nonbinding:
        assert report["node_mapping"][1]["nonbinding_clip_released"]
    else:
        assert not report["node_mapping"][1]["nonbinding_clip_released"]
        assert source["elements"][1]["width"] == 20
        assert report["node_mapping"][1]["source_frame"]["width"] == 80
        assert report["node_mapping"][1]["rectangular_clip_chain"]
        assert report["node_mapping"][1]["native_normalized_mae"] < 0.005
    assert report["native_imported_normalized_mae"] < 0.005


def test_clip_and_element_transforms_are_composed_exactly_once(tmp_path):
    raw = svg(
        '<defs><clipPath id="clip" transform="translate(5 3)">'
        '<rect x="10" y="0" width="20" height="50"/></clipPath></defs>'
        '<rect width="360" height="260" fill="white"/>'
        '<g transform="translate(40 20) scale(2)" clip-path="url(#clip)">'
        '<path transform="translate(5 0)" d="M0 0h80v60h-80z" fill="red"/></g>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    element = json.loads((out / "source.json").read_text(encoding="utf-8"))["elements"][1]
    assert [element[field] for field in ("x", "y", "width", "height")] == pytest.approx(
        [70, 26, 40, 100]
    )
    mapping = report["node_mapping"][1]
    assert mapping["source_frame"] == {"x": 50, "y": 20, "width": 160, "height": 120, "rotation": 0}
    assert mapping["source_path_data"] == "M0 0h80v60h-80z"
    assert mapping["rectangular_clip_chain"][0]["element_local_bounds"] == pytest.approx(
        [10, 3, 20, 50]
    )


def test_nested_rectangular_clips_retain_the_complete_intersection_chain(tmp_path):
    raw = svg(
        '<defs><clipPath id="outer"><rect x="20" y="20" width="80" height="80"/></clipPath>'
        '<clipPath id="inner"><rect x="40" y="40" width="20" height="20"/></clipPath></defs>'
        '<rect width="360" height="260" fill="white"/>'
        '<g clip-path="url(#outer)"><g clip-path="url(#inner)">'
        '<path d="M10 10h120v120h-120z" fill="red"/></g></g>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    element = json.loads((out / "source.json").read_text(encoding="utf-8"))["elements"][1]
    assert [element[field] for field in ("x", "y", "width", "height")] == [40, 40, 20, 20]
    chain = report["node_mapping"][1]["rectangular_clip_chain"]
    assert [record["clip_id"] for record in chain] == ["inner", "outer"]
    assert report["node_mapping"][1]["source_frame"]["width"] == 120


def test_stylesheet_clip_is_disabled_for_the_nonbinding_proof_not_silently_released(tmp_path):
    raw = svg(
        '<defs><clipPath id="clip"><rect x="30" y="20" width="20" height="100"/></clipPath></defs>'
        '<rect width="360" height="260" fill="white"/>'
        '<g class="cut"><path d="M30 30h80v60H30Z" fill="red"/></g>',
        style=".cut { clip-path:url(#clip); }",
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    element = json.loads((out / "source.json").read_text(encoding="utf-8"))["elements"][1]
    assert element["type"] == "shape" and element["width"] == 20
    assert not report["node_mapping"][1]["nonbinding_clip_released"]
    assert report["node_mapping"][1]["rectangular_clip_chain"]


def test_binding_stylesheet_text_clip_does_not_get_ignored(tmp_path, face):
    raw = svg(
        '<defs><clipPath id="clip"><rect x="30" y="20" width="20" height="100"/></clipPath></defs>'
        + text(face, **{"class": "cut"}),
        style=".cut { clip-path:url(#clip); }",
    )
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "unsupported"
    assert any(failure["code"] == "clipped_text" for failure in report["failures"])
    assert not (out / "bundle.json").exists()


@pytest.mark.parametrize(
    "clip_geometry,target",
    [
        ('<circle cx="50" cy="50" r="25"/>', '<path d="M30 30h80v60H30Z" fill="red"/>'),
        (
            '<rect x="30" y="30" width="20" height="20"/>'
            '<rect x="70" y="30" width="20" height="20"/>',
            '<path d="M30 30h80v60H30Z" fill="red"/>',
        ),
        (
            '<rect x="30" y="30" width="20" height="60" transform="rotate(25 40 40)"/>',
            '<path d="M30 30h80v60H30Z" fill="red"/>',
        ),
        (
            '<rect x="30" y="30" width="20" height="60" stroke="blue"/>',
            '<path d="M30 30h80v60H30Z" fill="red"/>',
        ),
        (
            '<rect x="30" y="30" width="20" height="60" rx="8"/>',
            '<path d="M30 30h80v60H30Z" fill="red"/>',
        ),
        (
            '<rect x="30" y="30" width="20" height="60" style="rx:8;ry:8"/>',
            '<path d="M30 30h80v60H30Z" fill="red"/>',
        ),
        (
            '<rect x="30" y="30" width="20" height="60"/>',
            '<rect x="30" y="30" width="80" height="60" rx="8" fill="red"/>',
        ),
    ],
)
def test_unproven_clip_geometry_and_rounded_edge_cuts_remain_exact_art(
    tmp_path, clip_geometry, target
):
    raw = svg(
        '<defs><clipPath id="clip">' + clip_geometry + "</clipPath></defs>"
        '<rect width="360" height="260" fill="white"/>'
        '<g clip-path="url(#clip)">' + target + "</g>"
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert source["elements"][1]["type"] == "image"
    assert report["native_imported_normalized_mae"] < 0.005


def test_genuine_prepared_background_is_an_editable_shape_not_a_full_canvas_image():
    folder = ROOT / "fixtures/open_posters/prepared/opensuse_cheatsheet"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    canvas = source["canvas"]
    assert any(
        element["type"] == "shape"
        and element["shape_type"] == "rectangle"
        and element["width"] * element["height"] >= canvas["width"] * canvas["height"] * 0.99
        and abs(element["x"]) < 1
        and abs(element["y"]) < 1
        for element in source["elements"]
    )


def test_root_level_clip_definitions_are_preserved_and_never_drawn_as_content(tmp_path):
    raw = svg(
        '<clipPath id="clip"><path d="M20 20h20v60H20Z"/></clipPath>'
        '<rect width="360" height="260" fill="white"/>'
        '<path d="M20 20h80v60H20Z" fill="red" clip-path="url(#clip)"/>'
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == "ready", report["failures"]
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert len(source["elements"]) == 2
    assert source["elements"][1]["type"] == "shape"
    assert source["elements"][1]["width"] == 20
    assert report["native_imported_normalized_mae"] < 0.005


@pytest.mark.parametrize("content,ready", [("", True), ("Do not silently drop live copy", False)])
def test_only_empty_inkscape_flow_containers_are_nonpainting(tmp_path, content, ready):
    raw = svg(
        '<rect x="20" y="20" width="40" height="40" fill="red"/>'
        '<flowRoot style="font-family:Missing"><flowRegion>'
        '<rect x="30" y="30" width="100" height="100"/></flowRegion>'
        "<flowPara>" + content + "</flowPara></flowRoot>"
    )
    report, out = prepare(tmp_path, raw)
    assert report["status"] == ("ready" if ready else "unsupported"), report["failures"]
    if ready:
        source = json.loads((out / "source.json").read_text(encoding="utf-8"))
        assert len(source["elements"]) == 1
        assert any("empty nonpainting" in record["reason"] for record in report["ignored_nodes"])
    else:
        assert not (out / "bundle.json").exists()
        assert report["native_reference_complete"] is False
        assert report["failures"][0]["code"] == "unsupported_flow_text"
        assert report["failures"][0]["source_text"] == content


def test_provider_alias_rewrite_with_empty_tspan_style_paints_the_native_reference(tmp_path):
    catalogue = json.loads(
        (ROOT / "fixtures/open_posters/catalogue.json").read_text(encoding="utf-8")
    )
    face = next(
        dict(face)
        for entry in catalogue["examples"]
        for face in entry.get("font_requirements", [])
        if face.get("family") == "Source Sans Pro" and str(face.get("weight")) == "300"
    )
    (tmp_path / "face.ttf").write_bytes(
        (ROOT / "fixtures/open_posters" / face["path"]).read_bytes()
    )
    face["path"] = "face.ttf"
    raw = svg(text(face, '<tspan x="30" y="65">Visible alias headline</tspan>'))
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    assert report["font_resolutions"][0]["resolution"] == "provider_evidenced_family_alias"
    native = Image.open(out / "native.png").convert("RGBA")
    assert native.getchannel("A").getbbox() is not None
    assert report["editable_text_count"] == 1
    assert report["native_imported_normalized_mae"] < 0.005


def test_single_text_owner_opacity_is_preserved_in_the_editable_leaf(tmp_path, face):
    raw = svg(text(face, '<tspan x="30" y="65">Half opacity</tspan>', opacity=".5"))
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "ready", report["failures"]
    element = json.loads((out / "source.json").read_text(encoding="utf-8"))["elements"][0]
    assert element["opacity"] == 0.5
    assert report["node_mapping"][0]["effective_text_opacity"] == 0.5
    assert Image.open(out / "native.png").convert("RGBA").getchannel("A").getextrema()[1] == 128
    # Independent PNG compositors can differ by a channel value or two at .5.
    assert 125 <= Image.open(out / "imported.png").convert("RGB").getextrema()[0][0] <= 129


def test_multi_run_owner_opacity_does_not_silently_become_separate_leaf_opacity(tmp_path, face):
    raw = svg(text(face, "<tspan>A</tspan><tspan>B</tspan>", opacity=".5"))
    report, out = prepare(tmp_path, raw, [face])
    assert report["status"] == "unsupported"
    assert any(failure["code"] == "text_group_compositing" for failure in report["failures"])
    assert not (out / "bundle.json").exists()
