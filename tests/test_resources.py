"""Resource contexts and exact native document font aliases."""

import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier

import pytest
from PIL import Image

from engine import resources

FONT_FILES = {
    ("regular", "normal"): "SolsticeSans-Regular.ttf",
    ("bold", "normal"): "SolsticeSans-Bold.ttf",
    ("regular", "italic"): "SolsticeSans-Italic.ttf",
    ("bold", "italic"): "SolsticeSans-BoldItalic.ttf",
}


@pytest.fixture(autouse=True)
def isolated_resource_caches():
    resources.clear_resource_caches()
    yield
    resources.clear_resource_caches()


def _manifest(directory, entries):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "font-manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return directory


def _font(directory, family="Native-Poster-PSAlias-9", weight="regular", style="normal"):
    path = directory / "nested" / "opaque-native-file.ttf"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (resources.DEFAULT_FONTS_DIR / FONT_FILES[weight, style]).read_bytes()
    path.write_bytes(data)
    _manifest(
        directory,
        [
            {
                "font_family": family,
                "file": "nested/opaque-native-file.ttf",
                "weight": weight,
                "style": style,
            }
        ],
    )
    return data


def test_default_resources_are_absolute_and_independent_of_working_directory(tmp_path, monkeypatch):
    before = resources.get_resources()
    monkeypatch.chdir(tmp_path)
    assert resources.get_resources() is before
    assert before.assets_dir.is_absolute() and before.assets_dir.is_dir()
    assert before.fonts_dir.is_absolute() and before.fonts_dir.is_dir()
    with pytest.raises(FrozenInstanceError):
        before.assets_dir = tmp_path


def test_context_restores_nested_overrides_and_exception(tmp_path):
    original = resources.get_resources()
    with resources.using_resources(assets_dir=tmp_path / "assets") as outer:
        assert resources.get_resources() is outer
        assert outer.fonts_dir == original.fonts_dir
        with pytest.raises(RuntimeError), resources.using_resources(fonts_dir=tmp_path / "fonts"):
            assert resources.get_resources().assets_dir == outer.assets_dir
            assert resources.get_resources().fonts_dir == (tmp_path / "fonts").resolve()
            raise RuntimeError("document failed")
        assert resources.get_resources() is outer
    assert resources.get_resources() is original


def test_independent_threads_do_not_share_document_context(tmp_path):
    original = resources.get_resources()
    barrier = Barrier(2)

    def worker(name):
        assert resources.get_resources() is original
        with resources.using_resources(tmp_path / name / "assets", tmp_path / name / "fonts"):
            barrier.wait(timeout=5)
            seen = resources.get_resources()
            barrier.wait(timeout=5)
        assert resources.get_resources() is original
        return seen

    with resources.using_resources(tmp_path / "parent"), ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, ["first", "second"]))
        assert resources.get_resources().assets_dir == (tmp_path / "parent").resolve()
    assert [context.assets_dir.parent.name for context in results] == ["first", "second"]
    assert resources.get_resources() is original


def test_async_tasks_restore_their_own_context(tmp_path):
    original = resources.get_resources()

    async def worker(name):
        with resources.using_resources(fonts_dir=tmp_path / name):
            await asyncio.sleep(0)
            return resources.get_resources().fonts_dir

    async def run():
        return await asyncio.gather(worker("first"), worker("second"))

    assert asyncio.run(run()) == [(tmp_path / "first").resolve(), (tmp_path / "second").resolve()]
    assert resources.get_resources() is original


def test_concurrent_documents_with_same_alias_keep_their_own_font_bytes(tmp_path):
    alias = "Same-Native-PS-Alias"
    first = tmp_path / "first"
    second = tmp_path / "second"
    expected = {
        "first": _font(first, alias),
        "second": _font(second, alias, weight="bold"),
    }
    barrier = Barrier(2)

    def worker(name):
        weight = "bold" if name == "second" else "regular"
        style = "bold" if weight == "bold" else "regular"
        with resources.using_resources(fonts_dir=tmp_path / name):
            barrier.wait(timeout=5)
            manager = resources.load_font_manager()
            renderer = resources.get_renderer()
            barrier.wait(timeout=5)
            assert renderer.font_manager is manager
            return manager.lookup(alias, weight, style)

    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(worker, ["first", "second"]))
    assert actual == [expected["first"], expected["second"]]
    assert actual[0] != actual[1]


def test_exact_postscript_alias_preserves_original_font_bytes_and_context(tmp_path):
    from cr_renderer.fonts import FontManager, normalize_family

    alias = "Opaque-Adobe-Poster-Face-9"
    expected = _font(tmp_path, alias)
    with resources.using_resources(fonts_dir=tmp_path):
        manager = resources.load_font_manager()
        assert manager is resources.load_font_manager(tmp_path / ".")
    assert isinstance(manager, FontManager)
    assert manager.lookup(alias) == expected
    assert manager.lookup(alias.lower()) == expected
    assert manager._fonts[normalize_family(alias)][0]["fontFamily"] == alias
    with pytest.raises(LookupError, match="family is not available"):
        manager.lookup("Opaque")
    with pytest.raises(LookupError, match="face is not available"):
        manager.lookup(alias, "bold", "bold")


def test_manifest_styles_are_explicit_and_not_inferred_from_alias(tmp_path):
    alias = "Document-Face"
    entries, expected = [], {}
    for i, ((weight, style), filename) in enumerate(FONT_FILES.items()):
        data = (resources.DEFAULT_FONTS_DIR / filename).read_bytes()
        (tmp_path / f"opaque-{i}.otf").write_bytes(data)
        entries.append(
            {"font_family": alias, "file": f"opaque-{i}.otf", "weight": weight, "style": style}
        )
        renderer_style = (
            "bolditalic"
            if weight == "bold" and style == "italic"
            else ("bold" if weight == "bold" else "italic" if style == "italic" else "regular")
        )
        expected[weight, renderer_style] = data
    _manifest(tmp_path, {"version": 1, "fonts": entries})
    manager = resources.load_font_manager(tmp_path)
    for (weight, style), data in expected.items():
        assert manager.lookup(alias, weight, style) == data


def test_missing_manifest_font_is_an_error(tmp_path):
    _manifest(
        tmp_path,
        [{"font_family": "Native", "file": "missing.ttf", "weight": "regular", "style": "normal"}],
    )
    with pytest.raises(FileNotFoundError, match="Document font file is missing"):
        resources.load_font_manager(tmp_path)


@pytest.mark.parametrize("filename", ["../escape.ttf", "nested/../../escape.ttf", "C:\\escape.ttf"])
def test_manifest_font_paths_cannot_escape_document_directory(tmp_path, filename):
    _manifest(
        tmp_path,
        [{"font_family": "Native", "file": filename, "weight": "regular", "style": "normal"}],
    )
    with pytest.raises(ValueError, match="relative|escapes"):
        resources.load_font_manager(tmp_path)


@pytest.mark.parametrize(
    "filename,data",
    [
        ("font.woff2", b"wOF2" + b"0" * 20),
        ("font.ttf", b"not a native font"),
        ("font.otf", b"OTTO" + b"0" * 32),
    ],
)
def test_manifest_fonts_require_native_extension_and_valid_font_data(tmp_path, filename, data):
    (tmp_path / filename).write_bytes(data)
    _manifest(
        tmp_path,
        [{"font_family": "Native", "file": filename, "weight": "regular", "style": "normal"}],
    )
    with pytest.raises(ValueError, match="font|Font"):
        resources.load_font_manager(tmp_path)


@pytest.mark.parametrize(
    "entries",
    [
        {},
        {"fonts": "not a list"},
        ["not an entry"],
        [{"font_family": "", "file": "font.ttf", "weight": "regular", "style": "normal"}],
        [{"font_family": "Native", "file": "font.ttf", "weight": "black", "style": "normal"}],
    ],
)
def test_malformed_manifest_never_falls_back_to_another_family(tmp_path, entries):
    _manifest(tmp_path, entries)
    with pytest.raises(ValueError):
        resources.load_font_manager(tmp_path)


def test_duplicate_normalized_alias_face_is_declined(tmp_path):
    _font(tmp_path)
    entry = {
        "font_family": "Document_Font",
        "file": "nested/opaque-native-file.ttf",
        "weight": "regular",
        "style": "normal",
    }
    _manifest(tmp_path, [entry, {**entry, "font_family": "Document Font"}])
    with pytest.raises(ValueError, match="Duplicate"):
        resources.load_font_manager(tmp_path)


def test_default_and_legacy_font_folders_keep_the_existing_loader(tmp_path, monkeypatch):
    from renderer.converter import brand_fonts

    original_loader = brand_fonts.load_brand_font_manager
    calls = []

    def observe(directory):
        calls.append(directory)
        return original_loader(directory)

    monkeypatch.setattr(brand_fonts, "load_brand_font_manager", observe)
    default = resources.load_font_manager()
    assert (
        default.lookup("Solstice Sans")
        == (resources.DEFAULT_FONTS_DIR / "SolsticeSans-Regular.ttf").read_bytes()
    )
    (tmp_path / "NativeSans-Regular.ttf").write_bytes(default.lookup("Solstice Sans"))
    legacy = resources.load_font_manager(tmp_path)
    assert legacy.lookup("Native Sans") == default.lookup("Solstice Sans")
    assert calls == [str(resources.DEFAULT_FONTS_DIR), str(tmp_path.resolve())]


def test_document_renderer_shares_manifest_manager_and_does_not_mutate_default(tmp_path):
    from renderer.converter.convert import convert_chameleon_to_crello

    alias = "Opaque-Poster-Native-PS"
    _font(tmp_path, alias)
    default = resources.get_renderer(resources.DEFAULT_FONTS_DIR)
    original_manager = default.font_manager
    with resources.using_resources(fonts_dir=tmp_path):
        renderer = resources.get_renderer()
        assert renderer is resources.get_renderer(tmp_path)
        assert renderer.font_manager is resources.load_font_manager()
    assert renderer is not default
    assert resources.get_renderer(resources.DEFAULT_FONTS_DIR).font_manager is original_manager
    example, warnings = convert_chameleon_to_crello(
        {
            "canvas": {"width": 300, "height": 100},
            "elements": [
                {
                    "id": "native-title",
                    "type": "text",
                    "x": 10,
                    "y": 10,
                    "width": 280,
                    "height": 70,
                    "content": "Native poster",
                    "font_family": alias,
                    "font_size": 32,
                    "color": "#111111",
                }
            ],
        },
        renderer.font_manager,
    )
    result = Image.open(io.BytesIO(renderer.render(example, max_size=300, format="png")))
    assert warnings == []
    assert result.size == (300, 100)
    assert len(result.convert("RGB").getcolors(maxcolors=30000)) > 1


def test_empty_manifest_supports_no_text_but_never_supplies_a_fallback_font(tmp_path):
    _manifest(tmp_path, [])
    manager = resources.load_font_manager(tmp_path)
    assert manager._fonts == {}
    assert resources.get_renderer(tmp_path).font_manager is manager
    with pytest.raises(LookupError, match="family is not available"):
        manager.lookup("Solstice Sans")


def test_document_caches_are_bounded_and_eviction_preserves_default(tmp_path):
    from renderer.converter import convert

    default = resources.get_renderer()
    original_manager = default.font_manager
    first_key = None
    for i in range(resources._CACHE_SIZE + 1):
        directory = _manifest(tmp_path / str(i), [])
        renderer = resources.get_renderer(directory)
        assert renderer.font_manager is resources.load_font_manager(directory)
        first_key = first_key or str(directory.resolve())
    assert resources._load_font_manager.cache_info().currsize == resources._CACHE_SIZE
    assert len(resources._renderers) == resources._CACHE_SIZE
    assert first_key not in convert._renderer_cache
    assert resources.get_renderer() is default
    assert default.font_manager is original_manager
    resources.clear_resource_caches()
    assert len(resources._renderers) == 0
    assert resources._load_font_manager.cache_info().currsize == 0
    assert resources.get_renderer().font_manager is original_manager


def test_font_cache_churn_keeps_active_renderer_manager_identity(tmp_path):
    active = _manifest(tmp_path / "active", [])
    renderer = resources.get_renderer(active)
    for i in range(resources._CACHE_SIZE + 1):
        resources.load_font_manager(_manifest(tmp_path / str(i), []))
    assert resources.load_font_manager(active) is renderer.font_manager
    assert resources.get_renderer(active) is renderer


def test_context_constructor_normalizes_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = resources.ResourceContext(Path("assets"), Path("fonts"))
    assert context.assets_dir == tmp_path / "assets"
    assert context.fonts_dir == tmp_path / "fonts"


def test_native_render_ranges_are_end_exclusive_without_changing_source(tmp_path):
    from copy import deepcopy

    from renderer.converter.convert import convert_chameleon_to_crello

    _font(tmp_path)
    design = _native_design()
    text = design["elements"][0]
    text.update(
        content="ABCD",
        style_runs=[
            {"start": 1, "end": 2, "color": "#ff0000"},
            {"start": 2, "end": 2, "color": "#00ff00"},
            {"start": 3, "end": 4, "color": "#0000ff"},
        ],
    )
    original = deepcopy(design)
    with resources.using_resources(fonts_dir=tmp_path):
        rendered = resources.render_input(design)
        converted, _ = convert_chameleon_to_crello(rendered, resources.load_font_manager())
    assert rendered is not design
    assert rendered["elements"][0]["style_runs"] == [
        {"start": 1, "end": 1, "color": "#ff0000"},
        {"start": 3, "end": 3, "color": "#0000ff"},
    ]
    assert converted["text_color"][0] == ["#111111", "#ff0000", "#111111", "#0000ff"]
    assert design == original
    assert resources.render_input(design) is design


def test_legacy_brand_render_ranges_retain_existing_behavior(tmp_path):
    design = _native_design()
    design["elements"][0]["style_runs"] = [{"start": 1, "end": 2, "color": "#ff0000"}]
    assert resources.render_input(design, resources.DEFAULT_FONTS_DIR) is design
    assert resources.render_input(design, tmp_path) is design


def test_both_native_render_entry_points_use_only_a_render_copy(tmp_path, monkeypatch):
    from copy import deepcopy

    from engine.eval import render as evaluation_render
    from viewer import server

    _font(tmp_path)
    design = _native_design()
    design["elements"][0]["style_runs"] = [{"start": 1, "end": 2, "color": "#ff0000"}]
    original = deepcopy(design)
    converted_ranges = []

    def record_converter(rendered, *_args):
        converted_ranges.append(rendered["elements"][0]["style_runs"])
        content = rendered["elements"][0]["content"]
        return {"length": 1, "text": [content], "text_line": [[0] * len(content)]}, []

    class Renderer:
        font_manager = object()

        def render(self, *_args, **_kwargs):
            result = io.BytesIO()
            Image.new("RGBA", (400, 200), "white").save(result, format="PNG")
            return result.getvalue()

    for module in [evaluation_render, server]:
        monkeypatch.setattr(module, "get_renderer", lambda *_: Renderer())
        monkeypatch.setattr(module, "convert_chameleon_to_crello", record_converter)
    with resources.using_resources(fonts_dir=tmp_path):
        assert evaluation_render.render_image(design).size == (400, 200)
        assert server._render_png(design)[0].startswith("data:image/png;base64,")
    assert converted_ranges == [[{"start": 1, "end": 1, "color": "#ff0000"}]] * 2
    assert design == original


def _native_design(alias="Native-Poster-PSAlias-9"):
    return {
        "canvas": {"width": 400, "height": 200},
        "elements": [
            {
                "id": "title",
                "type": "text",
                "x": 40,
                "y": 40,
                "width": 300,
                "height": 60,
                "content": "Native message",
                "font_family": alias,
                "font_size": 32,
                "color": "#111111",
                "rotation": 8,
            }
        ],
    }


def test_layout_and_eval_measurements_follow_inherited_document_fonts(tmp_path):
    from engine.eval.text import get_measurer as eval_measurer
    from engine.layout.text import get_measurer as layout_measurer

    first, second = tmp_path / "first", tmp_path / "second"
    _font(first)
    _font(second)
    (second / "nested/opaque-native-file.ttf").write_bytes(
        (resources.DEFAULT_FONTS_DIR / "SolsticeSans-Bold.ttf").read_bytes()
    )
    element = _native_design()["elements"][0]
    measured = []
    for directory in [first, second]:
        with resources.using_resources(fonts_dir=directory):
            layout, evaluation = layout_measurer(), eval_measurer()
            assert layout._manager is resources.load_font_manager()
            assert evaluation._font_manager is resources.load_font_manager()
            width = layout.measure(element, 32, 300).width
            assert width == pytest.approx(evaluation.layout(element).max_line_width)
            measured.append(width)
    assert measured[0] != measured[1]
    assert layout_measurer().fonts_dir == resources.DEFAULT_FONTS_DIR
    assert Path(eval_measurer().fonts_dir) == resources.DEFAULT_FONTS_DIR


def test_stored_eval_context_and_graph_restore_document_resources_after_exit(tmp_path):
    from engine.eval import build_context, score
    from engine.eval.render import render_image
    from engine.eval.style import analyze_style
    from engine.layout.relationships import analyze_relationships, build_relationship_graph

    fonts = tmp_path / "fonts"
    _font(fonts)
    assets = tmp_path / "assets"
    assets.mkdir()
    source = _native_design()
    original = resources.get_resources()
    with resources.using_resources(assets, fonts):
        context = build_context(source)
        graph = build_relationship_graph(source)
        expected_score = score(source, source, context)
        expected_relationships = analyze_relationships(graph, source)
        expected_style = analyze_style(source, source, relationship_graph=graph)
        expected_pixels = render_image(source).tobytes()
    assert context.assets_dir == str(assets.resolve())
    assert context.fonts_dir == str(fonts.resolve())
    assert graph.assets_dir == context.assets_dir and graph.fonts_dir == context.fonts_dir
    assert score(source, source, context) == expected_score
    assert analyze_relationships(graph, source) == expected_relationships
    assert analyze_style(source, source, relationship_graph=graph) == expected_style
    assert render_image(source, context.assets_dir, context.fonts_dir).tobytes() == expected_pixels
    assert resources.get_resources() is original


def test_implicit_alpha_resources_and_resize_cache_are_isolated_by_both_paths(
    tmp_path, monkeypatch
):
    from copy import deepcopy

    import engine.resize as resize_module
    from engine.layout.appearance import visible_box
    from engine.layout.text import get_measurer

    first_assets, second_assets = tmp_path / "first-assets", tmp_path / "second-assets"
    for directory, extent in [(first_assets, 20), (second_assets, 60)]:
        directory.mkdir()
        image = Image.new("RGBA", (100, 100))
        image.paste((255, 0, 0, 255), (0, 0, extent, 100))
        image.save(directory / "same.png")
    first_fonts, second_fonts = tmp_path / "first-fonts", tmp_path / "second-fonts"
    _font(first_fonts)
    _font(second_fonts)
    (second_fonts / "nested/opaque-native-file.ttf").write_bytes(
        (resources.DEFAULT_FONTS_DIR / "SolsticeSans-Bold.ttf").read_bytes()
    )
    source = _native_design()
    image = {
        "id": "product",
        "type": "image",
        "x": 20,
        "y": 100,
        "width": 100,
        "height": 100,
        "asset_id": "same.png",
    }
    calls = []

    def generate(original, target):
        calls.append(resources.get_resources())
        result = deepcopy(original)
        result["canvas"] = target
        result["elements"][0]["x"] = visible_box(image).width
        result["elements"][0]["width"] = (
            get_measurer().measure(original["elements"][0], 32, 300).width
        )
        return [result]

    monkeypatch.setattr(resize_module, "generate_candidates", generate)
    monkeypatch.setattr(resize_module, "generate_composition_states", lambda *_: ())
    monkeypatch.setattr(resize_module, "generate_grouped_states", lambda *_: ())
    monkeypatch.setattr(
        resize_module, "_select_candidate", lambda _s, _t, candidates, *_: candidates[0]
    )
    resize_module._resize_cached.cache_clear()
    try:
        with resources.using_resources(first_assets, first_fonts):
            first = resize_module.resize(source, {"width": 500, "height": 200})
            second = resize_module.resize(
                source, {"width": 500, "height": 200}, assets_dir=second_assets
            )
            third = resize_module.resize_with_state(
                source, {"width": 500, "height": 200}, fonts_dir=second_fonts
            )["design"]
            assert resize_module.resize(source, {"width": 500, "height": 200}) == first
        assert len(calls) == 3
        assert first["elements"][0]["x"] == 20
        assert second["elements"][0]["x"] == 60
        assert third["elements"][0]["x"] == 20
        assert first["elements"][0]["width"] != third["elements"][0]["width"]
        assert calls[0].fonts_dir == calls[1].fonts_dir == first_fonts.resolve()
        assert calls[0].assets_dir == calls[2].assets_dir == first_assets.resolve()
    finally:
        resize_module._resize_cached.cache_clear()
