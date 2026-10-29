"""Actual resvg/font imports, explicit unsupported cases and immutable originals."""

import hashlib
import io
import json
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont
from PIL import Image

from tools.poster_import import ImportBlock, main, parse_transform, prepare_poster

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
    assert report["vector_art_count"] == 2
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
