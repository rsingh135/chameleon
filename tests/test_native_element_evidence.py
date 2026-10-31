"""Native evidence keeps authored pixels and makes incomplete paint explicit."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from engine.resources import get_renderer
from tools import native_element_evidence as evidence
from viewer.resize_cache import ResourceBindings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh_identity_pins(monkeypatch):
    # Only helper metadata is isolated; shared native renderer caches are never cleared.
    monkeypatch.setattr(evidence, "_BINDINGS", ResourceBindings())
    monkeypatch.setattr(evidence, "_WORKER_THREAD", None)
    monkeypatch.setattr(evidence, "_CODE_PIN", None)


@pytest.fixture
def resources(tmp_path):
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    entries = []
    for name, weight in (
        ("SolsticeSans-Regular.ttf", "regular"),
        ("SolsticeSans-Bold.ttf", "bold"),
    ):
        (fonts / name).write_bytes((ROOT / "brand_kit/fonts" / name).read_bytes())
        entries.append(
            {"font_family": "ExactSans", "file": name, "weight": weight, "style": "normal"}
        )
    (fonts / "font-manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return assets, fonts


@pytest.fixture
def cheap_identity(monkeypatch):
    # Native identity has a separate real test. Failure/geometry tests need no repeated DLL hashes.
    monkeypatch.setattr(
        evidence, "_implementation", lambda: ({"test_native": "frozen"}, {"cpu": True})
    )


def text(**changes):
    return {
        "id": "native-footer",
        "type": "text",
        "x": 100.25,
        "y": 240.5,
        "width": 500,
        "height": 160,
        "content": "www.example.org",
        "font_family": "ExactSans",
        "font_size": 40,
        "font_weight": "regular",
        "font_style": "normal",
        "text_align": "left",
        "line_height": 1.0,
        "color": "#ffffff",
        **changes,
    }


def shape(**changes):
    return {
        "id": "shape",
        "type": "shape",
        "shape_type": "rectangle",
        "x": 20,
        "y": 30,
        "width": 80,
        "height": 40,
        "fill_color": "rgba(255,0,0,1)",
        **changes,
    }


def picture(**changes):
    return {
        "id": "image",
        "type": "image",
        "x": 10,
        "y": 20,
        "width": 60,
        "height": 60,
        "asset_id": "original.png",
        "fit": "contain",
        **changes,
    }


def run(element, resources):
    assets, fonts = resources
    return evidence.render_isolated_element(element, assets_dir=assets, fonts_dir=fonts)


def decoded(result, key="alpha_png"):
    with Image.open(BytesIO(result[key])) as image:
        return image.copy()


def unavailable(result, reason):
    assert result["status"] == "unavailable", result["receipt"]
    assert reason in result["reason"]
    assert result["alpha_png"] is result["checker_png"] is None
    assert result["receipt"]["resources_unchanged"] is False
    assert result["receipt"]["frame_or_asset_fallback_used"] is False


def test_real_native_receipt_binds_exact_bytes_resources_runtime_and_input(resources):
    element = text(
        opacity=0.5, style_runs=[{"start": 0, "end": 3, "font_weight": "bold", "color": "#ff0000"}]
    )
    original = deepcopy(element)
    result = run(element, resources)
    assert result["status"] == "available", result["reason"]
    assert element == original
    receipt = result["receipt"]
    alpha, checker = decoded(result), decoded(result, "checker_png")
    assert alpha.mode == "RGBA" and checker.mode == "RGB" and alpha.size == checker.size
    assert alpha.getchannel("A").getextrema() == (0, 128)
    assert checker.getpixel((0, 0)) == (224, 224, 224)
    assert checker.getpixel((8, 0)) == (192, 192, 192)
    assert any(
        r > g and a
        for r, g, _b, a in (
            alpha.getpixel((x, y)) for y in range(alpha.height) for x in range(alpha.width)
        )
    )
    assert receipt["resources_unchanged"] and not receipt["paint_empty"]
    assert receipt["source_visible_alpha_box_xywh"][2] < element["width"]
    assert receipt["policy"] == {
        "version": evidence.ISOLATION_V1,
        "max_native_axis": 1024,
        "max_native_pixels": 4_000_000,
        "max_logical_extent": 1_000_000,
        "max_thumbnail_axis": 512,
        "alpha_padding": 8,
        "checker": {"colors_rgba": [[224, 224, 224, 255], [192, 192, 192, 255]], "cell_pixels": 8},
    }
    assert receipt["source_element_sha256"] == evidence._digest(original)
    assert "engine/resources.py" in receipt["implementation_hashes"]
    assert "tools/native_element_evidence.py" in receipt["implementation_hashes"]
    assert receipt["runtime"]["Pillow"]
    assert receipt["resource_identity"]["hashes"]["fonts"]["font-manifest.json"]
    for key in ("alpha_png", "checker_png"):
        assert hashlib.sha256(result[key]).hexdigest() == receipt[key + "_sha256"]
    assert receipt["receipt_digest"] == evidence._digest(
        {k: v for k, v in receipt.items() if k != "receipt_digest"}
    )


def test_translation_preserves_pixels_and_maps_source_ink(resources, cheap_identity):
    first = run(text(), resources)
    moved = run(text(x=137.25, y=159.5), resources)
    assert first["status"] == moved["status"] == "available"
    assert first["alpha_png"] == moved["alpha_png"]
    box = first["receipt"]["source_visible_alpha_box_xywh"]
    assert moved["receipt"]["source_visible_alpha_box_xywh"] == pytest.approx(
        [box[0] + 37, box[1] - 81, *box[2:]]
    )


def test_outline_and_rotated_shape_are_native_paint(resources, cheap_identity):
    plain = run(text(), resources)
    outline = run(
        text(
            native_text_effects={
                "version": 1,
                "outline": {"color": "#ffffff", "width_em": 0.4, "fill": True},
            }
        ),
        resources,
    )
    assert outline["status"] == "available", outline["reason"]
    assert (
        outline["receipt"]["source_visible_alpha_box_xywh"][3]
        > plain["receipt"]["source_visible_alpha_box_xywh"][3]
    )
    rotated = run(shape(rotation=45, opacity=0.25), resources)
    assert rotated["status"] == "available", rotated["reason"]
    assert rotated["receipt"]["original_rotation"] == 45
    assert decoded(rotated).getchannel("A").getextrema()[1] == 64
    assert decoded(rotated).getpixel(
        (decoded(rotated).width // 2, decoded(rotated).height // 2)
    ) == (255, 0, 0, 64)
    assert rotated["receipt"]["source_visible_alpha_box_xywh"][3] > 40


def test_empty_real_paint_keeps_transparent_rgba_and_checker(resources, cheap_identity):
    result = run(text(color="rgba(255,255,255,0)"), resources)
    assert result["status"] == "available", result["reason"]
    image = decoded(result)
    assert image.size == (32, 32) and image.getchannel("A").getbbox() is None
    assert result["receipt"]["paint_empty"]
    assert result["receipt"]["source_visible_alpha_box_xywh"] is None
    checker = decoded(result, "checker_png")
    assert {checker.getpixel((x, y)) for y in range(32) for x in range(32)} == {
        (224, 224, 224),
        (192, 192, 192),
    }


def test_original_rgba_contain_cover_and_viewport_rotation(resources, cheap_identity):
    assets, _fonts = resources
    image = Image.new("RGBA", (40, 20))
    image.paste((255, 0, 0, 128), (0, 0, 20, 20))
    image.paste((0, 0, 255, 255), (20, 0, 40, 20))
    image.save(assets / "original.png")
    original = (assets / "original.png").read_bytes()
    contain = run(picture(), resources)
    left = run(picture(fit="cover", image_position=[0, 0]), resources)
    right = run(picture(fit="cover", image_position=[1, 1]), resources)
    rotated = run(picture(fit="cover", image_position=[1, 1], rotation=30), resources)
    assert all(r["status"] == "available" for r in (contain, left, right, rotated)), [
        r["reason"] for r in (contain, left, right, rotated)
    ]
    assert (
        contain["receipt"]["source_visible_alpha_box_xywh"][3]
        < left["receipt"]["source_visible_alpha_box_xywh"][3]
    )
    assert decoded(left).getchannel("A").getextrema()[1] < 255
    assert decoded(right).getchannel("A").getextrema()[1] == 255
    assert (
        rotated["receipt"]["source_visible_alpha_box_xywh"][2]
        > right["receipt"]["source_visible_alpha_box_xywh"][2]
    )
    assert (assets / "original.png").read_bytes() == original


def test_native_axis_and_final_thumbnail_bounded_without_frame_mutation(resources, cheap_identity):
    element = shape(width=4000, height=1000)
    original = deepcopy(element)
    result = run(element, resources)
    assert result["status"] == "available", result["reason"]
    receipt = result["receipt"]
    assert max(receipt["native_raster_size"]) == 1024
    assert receipt["native_raster_size"][0] * receipt["native_raster_size"][1] <= 4_000_000
    assert max(decoded(result).size) == 512
    assert receipt["raster_scale"] < 1 and receipt["thumbnail_scale"] < 1
    assert element == original


@pytest.mark.parametrize(
    ("element", "reason"),
    [
        (picture(asset_id="missing.png"), "missing"),
        (picture(asset_id="../escape.png"), "path"),
        (text(font_family="MissingFace"), "font"),
        (text(native_text_effects={"version": 99}), "version"),
        (text(font_size=9000), "numeric"),
        (text(content="x" * 20001), "content"),
        (shape(width=1000000, height=1000000), "pixel budget"),
        (shape(rotation=float("nan")), "JSON"),
        (shape(x=10**400), "numeric"),
        (shape(shadow={"blur": 1}), "Unsupported"),
        (shape(_unrecognized_paint={"shadow": 1}), "Unsupported"),
        (shape(fill_color="#ff0000"), "color syntax"),
        (shape(stroke_width=2, stroke_color="rgb(0,0,0)"), "authored stroke"),
        (shape(shape_type="ellipse", stroke_width=2), "authored stroke"),
        (shape(shape_type="line", stroke_color="rgba(255,0,0,0.5)"), "stroke alpha"),
        (text(style_runs=[{"start": 0, "end": 3, "font_style": "oblique"}]), "run face"),
        (picture(image_position=[0, 0]), "cover"),
    ],
)
def test_unavailable_never_returns_fallback_png(element, reason, resources, cheap_identity):
    unavailable(run(element, resources), reason)


def test_bundled_text_opacity_unproved_without_manifest(resources, cheap_identity):
    assets, _fonts = resources
    result = evidence.render_isolated_element(
        text(font_family="Solstice Sans", opacity=0.5),
        assets_dir=assets,
        fonts_dir=ROOT / "brand_kit/fonts",
    )
    unavailable(result, "opacity")


@pytest.mark.parametrize("failure", ["clipped", "RGB", "size", "warnings"])
def test_renderer_or_conversion_cannot_hide_partial_paint(
    failure, resources, cheap_identity, monkeypatch
):
    assets, fonts = resources
    renderer = get_renderer(fonts)
    if failure == "warnings":
        original_convert = evidence.convert_chameleon_to_crello

        def convert(*args):
            example, _warnings = original_convert(*args)
            return example, ["unsupported original paint"]

        monkeypatch.setattr(evidence, "convert_chameleon_to_crello", convert)
    else:

        def render(example, **kwargs):
            encoded = renderer.render(example, **kwargs)
            with Image.open(BytesIO(encoded)) as image:
                if failure == "clipped":
                    replaced = Image.new("RGBA", image.size, (1, 2, 3, 255))
                elif failure == "RGB":
                    replaced = image.convert("RGB")
                else:
                    replaced = Image.new("RGBA", (1, 1))
                out = BytesIO()
                replaced.save(out, format="PNG")
                return out.getvalue()

        monkeypatch.setattr(
            evidence,
            "get_renderer",
            lambda _fonts: SimpleNamespace(font_manager=renderer.font_manager, render=render),
        )
    result = evidence.render_isolated_element(shape(), assets_dir=assets, fonts_dir=fonts)
    unavailable(
        result,
        "clipped" if failure == "clipped" else "warnings" if failure == "warnings" else "RGBA",
    )


def test_asset_bytes_changed_at_same_path_rejected_across_calls(resources, cheap_identity):
    assets, _fonts = resources
    path = assets / "original.png"
    Image.new("RGBA", (20, 20), (255, 0, 0, 255)).save(path)
    assert run(picture(), resources)["status"] == "available"
    stat = path.stat()
    Image.new("RGBA", (20, 20), (0, 255, 0, 255)).save(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    unavailable(run(picture(), resources), "Resource bytes changed")


def test_font_inventory_and_bytes_are_pinned(resources, cheap_identity):
    _assets, fonts = resources
    assert run(text(), resources)["status"] == "available"
    path = fonts / "SolsticeSans-Regular.ttf"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    unavailable(run(text(), resources), "Resource bytes changed")
    path.write_bytes(original)
    (fonts / "extra.json").write_text("{}", encoding="utf-8")
    unavailable(run(text(), resources), "Font inventory changed")


def test_resource_change_during_render_discards_both_artifacts(
    resources, cheap_identity, monkeypatch
):
    assets, fonts = resources
    path = assets / "original.png"
    Image.new("RGBA", (20, 20), (255, 0, 0, 255)).save(path)
    renderer = get_renderer(fonts)

    def render(*args, **kwargs):
        encoded = renderer.render(*args, **kwargs)
        Image.new("RGBA", (20, 20), (0, 255, 0, 255)).save(path)
        return encoded

    monkeypatch.setattr(
        evidence,
        "get_renderer",
        lambda _fonts: SimpleNamespace(font_manager=renderer.font_manager, render=render),
    )
    unavailable(run(picture(), resources), "Resource bytes changed")


def test_worker_affinity_and_changed_implementation_explicitly_unavailable(
    resources, cheap_identity, monkeypatch
):
    assert run(shape(), resources)["status"] == "available"
    with ThreadPoolExecutor(max_workers=1) as executor:
        unavailable(
            executor.submit(run, shape(), resources).result(), "original single worker thread"
        )
    monkeypatch.setattr(
        evidence, "_implementation", lambda: ({"test_native": "changed"}, {"cpu": True})
    )
    unavailable(run(shape(), resources), "implementation changed")


def test_invalid_policy_and_missing_resource_directory_are_explicit(resources):
    assets, fonts = resources
    unavailable(
        evidence.render_isolated_element(
            shape(), assets_dir=assets, fonts_dir=fonts, policy="other"
        ),
        "policy",
    )
    unavailable(
        evidence.render_isolated_element(shape(), assets_dir=assets / "missing", fonts_dir=fonts),
        "directories",
    )


@pytest.mark.parametrize("manifest", [False, True])
def test_font_inventory_byte_bounds_precede_hashing_or_native_load(
    resources, monkeypatch, manifest
):
    _assets, fonts = resources
    original_stat = Path.stat

    def bounded_stat(path, *args, **kwargs):
        actual = original_stat(path, *args, **kwargs)
        if path.parent == fonts and path.suffix == (".json" if manifest else ".ttf"):
            fields = {key: getattr(actual, key) for key in dir(actual) if key.startswith("st_")}
            fields["st_size"] = 2 * 1024 * 1024 if manifest else 36 * 1024 * 1024
            return SimpleNamespace(**fields)
        return actual

    def must_not_hash(*args, **kwargs):
        pytest.fail("Excessive font bytes reached hashing/native loading")

    monkeypatch.setattr(Path, "stat", bounded_stat)
    monkeypatch.setattr(evidence._BINDINGS, "bind", must_not_hash)
    unavailable(run(text(), resources), "one-MiB" if manifest else "aggregate 64-MiB")


def test_image_decompression_bomb_is_explicitly_unavailable(resources, monkeypatch):
    assets, _fonts = resources
    (assets / "original.png").write_bytes(b"unsafe image header")

    def bomb(*args, **kwargs):
        raise Image.DecompressionBombError("oversized original image")

    monkeypatch.setattr(evidence.Image, "open", bomb)
    unavailable(run(picture(), resources), "DecompressionBombError")


def test_nonmanifest_viewport_and_style_ranges_cannot_silently_use_legacy_paint(
    resources, cheap_identity
):
    assets, _fonts = resources
    original = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    original.paste((0, 0, 255, 255), (20, 0, 40, 20))
    original.save(assets / "original.png")
    bundled_fonts = ROOT / "brand_kit/fonts"
    unavailable(
        evidence.render_isolated_element(
            picture(fit="cover", image_position=[0, 0]), assets_dir=assets, fonts_dir=bundled_fonts
        ),
        "viewport requires manifest",
    )
    unavailable(
        evidence.render_isolated_element(
            text(
                font_family="Solstice Sans", style_runs=[{"start": 0, "end": 3, "color": "#ff0000"}]
            ),
            assets_dir=assets,
            fonts_dir=bundled_fonts,
        ),
        "style runs require manifest",
    )


def test_line_supported_matching_alpha_is_preserved(resources, cheap_identity):
    result = run(
        shape(
            shape_type="line",
            fill_color="rgba(255,0,0,0.5)",
            stroke_color="rgba(0,0,255,0.5)",
            stroke_width=3,
        ),
        resources,
    )
    assert result["status"] == "available", result["reason"]
    image = decoded(result)
    assert image.getpixel((image.width // 2, image.height // 2)) == (0, 0, 255, 128)
