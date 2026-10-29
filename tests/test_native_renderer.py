"""Fractional imported type is isolated from existing bundled rendering."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
from cr_renderer import text_utils
from cr_renderer.schema import TextElement

from engine.eval.render import render_image
from engine.native_renderer import NativeDocumentRenderer, make_native_text_blob
from engine.resources import (
    DEFAULT_ASSETS_DIR,
    DEFAULT_FONTS_DIR,
    get_renderer,
    load_font_manager,
    native_render_example,
    render_input,
    using_resources,
)
from renderer.converter.convert import convert_chameleon_to_crello

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_fonts(tmp_path):
    source = sorted((ROOT / "brand_kit/fonts").glob("*.ttf"))[0]
    (tmp_path / "face.ttf").write_bytes(source.read_bytes())
    (tmp_path / "font-manifest.json").write_text(
        json.dumps(
            {
                "fonts": [
                    {
                        "font_family": "ExactDocumentFace",
                        "weight": "regular",
                        "style": "normal",
                        "file": "face.ttf",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def design(family, size=12.9):
    return {
        "canvas": {"width": 360, "height": 120},
        "elements": [
            {
                "id": "text",
                "type": "text",
                "x": 20,
                "y": 20,
                "width": 320,
                "height": 60,
                "content": "Fractional native typography",
                "font_family": family,
                "font_size": size,
                "color": "rgba(0,0,0,1)",
                "font_weight": "regular",
                "font_style": "normal",
            }
        ],
    }


def test_native_blob_keeps_fractional_metrics_and_vendor_function_identity(native_fonts):
    manager = load_font_manager(native_fonts)
    element = TextElement(
        uuid="",
        type="textElement",
        width=320,
        height=60,
        text="Native",
        font="ExactDocumentFace",
        fontSize=12.9,
        lineHeight=1,
        textAlign="left",
        capitalize=False,
        letterSpacing=0,
        colorMap=text_utils.generate_map(["rgba(0,0,0,1)"] * len("Native")),
    )
    span = next(text_utils.make_text_spans(element))[0]
    vendor_fn = text_utils.make_text_blob
    native = make_native_text_blob(element, span, manager)
    vendor = vendor_fn(element, span, manager)
    assert native.metrics.fAscent != vendor.metrics.fAscent
    assert native.metrics.fAscent / vendor.metrics.fAscent == pytest.approx(12.9 / 12, rel=1e-5)
    assert text_utils.make_text_blob is vendor_fn


def test_only_manifest_renderer_uses_float_adapter(native_fonts):
    default = get_renderer(DEFAULT_FONTS_DIR)
    assert not isinstance(default, NativeDocumentRenderer)
    assert isinstance(get_renderer(native_fonts), NativeDocumentRenderer)
    a = render_image(design("ExactDocumentFace", 12.1), fonts_dir=native_fonts)
    b = render_image(design("ExactDocumentFace", 12.9), fonts_dir=native_fonts)
    assert a.tobytes() != b.tobytes()
    assert get_renderer(DEFAULT_FONTS_DIR) is default


def test_concurrent_native_and_default_documents_do_not_change_default_pixels(native_fonts):
    # A small independent brand face avoids importing any native aliases into it.
    brand = design("Solstice Sans")
    before = render_image(brand).tobytes()
    vendor_fn = text_utils.make_text_blob

    def draw_native(_):
        return render_image(design("ExactDocumentFace"), fonts_dir=native_fonts).tobytes()

    def draw_brand(_):
        return render_image(brand).tobytes()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: draw_native(i) if i % 2 else draw_brand(i), range(8)))
    assert all(results[i] == before for i in range(0, 8, 2))
    assert render_image(brand).tobytes() == before
    assert text_utils.make_text_blob is vendor_fn


def test_native_paint_metadata_follows_validated_converter_z_order(native_fonts):
    source = design("ExactDocumentFace")
    late = source["elements"][0]
    late.update(id="late", z_index=9, x=130, native_text_effects={"version": 1, "baseline_em": 0.9})
    early = {
        **deepcopy(late),
        "id": "early",
        "z_index": -2,
        "x": 10,
        "native_text_effects": {"version": 1, "baseline_em": 0.8},
    }
    source["elements"].append(early)
    before = deepcopy(source)
    with using_resources(fonts_dir=native_fonts):
        example, _ = convert_chameleon_to_crello(
            render_input(source), load_font_manager(), str(DEFAULT_ASSETS_DIR)
        )
        attached = native_render_example(example, source)
        assert [descriptor["baseline_em"] for descriptor in attached["native_text_effects"]] == [
            0.8,
            0.9,
        ]
        assert attached["native_text_effects"][0]["source_text"] == early["content"]
        assert "native_text_effects" not in example
        assert source == before
        damaged = {**example, "left": list(reversed(example["left"]))}
        with pytest.raises(ValueError, match="frame order"):
            native_render_example(damaged, source)
