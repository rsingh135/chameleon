"""Native document wraps include tracking and exact live run faces."""

import json
from copy import deepcopy
from itertools import accumulate
from pathlib import Path

import pytest

from engine.eval.render import render_image
from engine.eval.text import TextMeasurer as EvalMeasurer
from engine.layout.text import TextMeasurer as LayoutMeasurer
from engine.native_typography import character_advances, effective_tracking, rewrap_example
from engine.resources import (
    DEFAULT_FONTS_DIR,
    load_font_manager,
    native_render_example,
    render_input,
    using_resources,
)
from renderer.converter.convert import convert_chameleon_to_crello
from renderer.converter.wrap import char_widths, wrap_text

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_fonts(tmp_path):
    faces = []
    for alias, filename, weight, style in (
        ("ExactSans", "SolsticeSans-Regular.ttf", "regular", "normal"),
        ("ExactSans", "SolsticeSans-Bold.ttf", "bold", "normal"),
        ("ExactSans", "SolsticeSans-Italic.ttf", "regular", "italic"),
        ("ExactItalic", "SolsticeSerif-Italic.ttf", "regular", "normal"),
    ):
        (tmp_path / filename).write_bytes((ROOT / "brand_kit/fonts" / filename).read_bytes())
        faces.append(dict(font_family=alias, file=filename, weight=weight, style=style))
    (tmp_path / "font-manifest.json").write_text(json.dumps({"fonts": faces}), encoding="utf-8")
    return tmp_path


def _tracked(native_fonts):
    manager = load_font_manager(native_fonts)
    size, tracking = 32, 20
    widths = char_widths(manager, "ExactSans", "regular", "regular", size, "A A A")
    return dict(
        id="tracked",
        type="text",
        x=10,
        y=10,
        width=sum(widths[:3]) + 2 * tracking + 0.01,
        height=150,
        content="A A A",
        font_family="ExactSans",
        font_size=size,
        line_height=1.2,
        letter_spacing=tracking,
        color="#000000",
    )


def test_native_measurers_wrap_using_positive_tracking(native_fonts):
    element = _tracked(native_fonts)
    layout = LayoutMeasurer(native_fonts).measure(element, element["font_size"], element["width"])
    evaluated = EvalMeasurer(native_fonts).layout(element)
    assert layout.wrapped == "A A\nA"
    assert evaluated.lines == ("A A", "A")
    single = char_widths(
        load_font_manager(native_fonts), "ExactSans", "regular", "regular", 32, "A"
    )[0]
    assert evaluated.line_widths == pytest.approx((layout.width, single))
    assert evaluated.height == pytest.approx(layout.height)


def test_original_grand_body_retains_authored_two_line_break():
    folder = ROOT / "fixtures/open_posters/prepared/doory_grand-opening-flyer"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    element = next(
        e for e in source["elements"] if e.get("content", "").startswith("FOR A MORNING")
    )
    expected = ("FOR A MORNING OF COFFEE AND PASTRIES", "CELEBRATING THE OPENING OF OUR NEW CAFÉ")
    evaluated = EvalMeasurer(folder / "resources/fonts").layout(element)
    layout = LayoutMeasurer(folder / "resources/fonts").measure(
        element, element["font_size"], element["width"]
    )
    assert evaluated.lines == expected
    assert layout.wrapped == "\n".join(expected)
    assert max(evaluated.line_widths) <= element["width"]


def test_rewrap_changes_only_text_lines_and_retains_source_run_indices(native_fonts):
    element = _tracked(native_fonts)
    element["content"] = "A A A\nA A A"
    element["style_runs"] = [{"start": 2, "end": 5, "font_weight": "bold", "color": "#ff0000"}]
    element["native_text_effects"] = {
        "version": 1,
        "runs": [{"start": 8, "end": 11, "font_family": "ExactItalic"}],
    }
    design = {"canvas": {"width": 220, "height": 170}, "elements": [element]}
    source_before = deepcopy(design)
    manager = load_font_manager(native_fonts)
    with using_resources(fonts_dir=native_fonts):
        example, warnings = convert_chameleon_to_crello(render_input(design), manager)
    assert warnings == []
    example_before = deepcopy(example)
    wrapped = rewrap_example(example, design, manager)
    assert len(wrapped["text"][0]) == len(element["content"])
    assert wrapped["text"][0][5] == "\n"
    assert all(
        a == b or (a == " " and b == "\n")
        for a, b in zip(element["content"], wrapped["text"][0], strict=True)
    )
    assert wrapped["text_line"][0] == list(accumulate(int(c == "\n") for c in wrapped["text"][0]))
    assert {k: v for k, v in wrapped.items() if k not in {"text", "text_line"}} == {
        k: v for k, v in example.items() if k not in {"text", "text_line"}
    }
    assert example == example_before and design == source_before
    wrapped["text_line"][0][0] = 99
    assert example["text_line"][0][0] == 0


def test_exact_effect_run_face_and_ordinary_style_faces_change_native_measurement(native_fonts):
    manager = load_font_manager(native_fonts)
    element = dict(_tracked(native_fonts), content="WW WW", letter_spacing=3, font_size=40)
    base = char_widths(manager, "ExactSans", "regular", "regular", 40, element["content"])
    italic = char_widths(manager, "ExactItalic", "regular", "regular", 40, "WW")
    styled_width = sum(base[:3]) + sum(italic) + 4 * 3
    base_width = sum(base) + 4 * 3
    assert abs(styled_width - base_width) > 1
    element["width"] = (styled_width + base_width) / 2
    element["native_text_effects"] = {
        "version": 1,
        "runs": [{"start": 3, "end": 5, "font_family": "ExactItalic"}],
    }
    advances = character_advances(element, manager)
    assert advances[3:] == pytest.approx(italic)
    evaluated = EvalMeasurer(native_fonts).layout(element)
    measured = LayoutMeasurer(native_fonts).measure(element, 40, element["width"])
    assert evaluated.line_count == (1 if styled_width < base_width else 2)
    assert measured.wrapped == "\n".join(evaluated.lines)

    # An ordinary bold style run is measured with its exact bold face too.
    element.pop("native_text_effects")
    element["style_runs"] = [{"start": 3, "end": 5, "font_weight": "bold"}]
    bold = char_widths(manager, "ExactSans", "bold", "bold", 40, "WW")
    assert character_advances(element, manager)[3:] == pytest.approx(bold)


@pytest.mark.parametrize("tracking", [-10, -100])
def test_signed_negative_tracking_is_not_clamped_or_nonfinite(native_fonts, tracking):
    element = dict(_tracked(native_fonts), letter_spacing=tracking, width=60)
    advances = char_widths(
        load_font_manager(native_fonts), "ExactSans", "regular", "regular", 32, "A A A"
    )
    expected = max(0, sum(advances) + tracking * 4)
    measured = LayoutMeasurer(native_fonts).measure(element, 32, 60)
    evaluated = EvalMeasurer(native_fonts).layout(element)
    assert measured.wrapped == element["content"]
    assert measured.width == pytest.approx(expected)
    assert evaluated.line_widths == pytest.approx((expected,))
    assert all(w >= 0 for w in evaluated.line_widths)


def test_final_glyph_has_no_extra_tracking_and_long_words_are_not_split(native_fonts):
    element = dict(_tracked(native_fonts), content="A", width=1)
    evaluated = EvalMeasurer(native_fonts).layout(element)
    expected = char_widths(
        load_font_manager(native_fonts), "ExactSans", "regular", "regular", 32, "A"
    )[0]
    assert evaluated.lines == ("A",)
    assert evaluated.line_widths == pytest.approx((expected,))
    assert evaluated.widest_unbreakable == pytest.approx(expected)


def test_native_em_tracking_preserves_source_pixels_and_scales_with_type(native_fonts):
    element = _tracked(native_fonts)
    manager = load_font_manager(native_fonts)
    design = {"canvas": {"width": 220, "height": 170}, "elements": [element]}
    source_pixels = render_image(design, fonts_dir=native_fonts).tobytes()
    pixel = EvalMeasurer(native_fonts).layout(element)
    element["native_text_effects"] = {
        "version": 1,
        "tracking_em": element["letter_spacing"] / element["font_size"],
    }
    element["letter_spacing"] = 0
    assert effective_tracking(element) == 20
    assert EvalMeasurer(native_fonts).layout(element).line_widths == pytest.approx(
        pixel.line_widths
    )
    assert render_image(design, fonts_dir=native_fonts).tobytes() == source_pixels
    full_glyph_widths = [
        sum(char_widths(manager, "ExactSans", "regular", "regular", 32, line))
        for line in pixel.lines
    ]
    half_glyph_widths = [
        sum(char_widths(manager, "ExactSans", "regular", "regular", 16, line))
        for line in pixel.lines
    ]
    # Fit the same authored line at the actual half-size hinted advance rather
    # than assuming that different platform font rasterizers halve every glyph.
    half_width = half_glyph_widths[0] + 10 * (len(pixel.lines[0]) - 1) + 0.01
    half = EvalMeasurer(native_fonts).layout(element, font_size=16, width=half_width)
    assert half.lines == pixel.lines
    full_tracking = [a - b for a, b in zip(pixel.line_widths, full_glyph_widths, strict=True)]
    half_tracking = [a - b for a, b in zip(half.line_widths, half_glyph_widths, strict=True)]
    assert full_tracking == pytest.approx([20 * (len(line) - 1) for line in pixel.lines])
    assert half_tracking == pytest.approx([value / 2 for value in full_tracking])
    assert effective_tracking(element, 16) == 10


def test_pixel_tracking_remains_fixed_when_no_native_em_semantics_are_declared(native_fonts):
    element = _tracked(native_fonts)
    assert effective_tracking(element, 16) == element["letter_spacing"] == 20


def test_native_rendered_lines_match_measurers_and_default_path_is_unchanged(native_fonts):
    element = _tracked(native_fonts)
    design = {"canvas": {"width": 220, "height": 170}, "elements": [element]}
    with using_resources(fonts_dir=native_fonts):
        manager = load_font_manager()
        example, _ = convert_chameleon_to_crello(render_input(design), manager)
        native = native_render_example(example, design)
        image = render_image(design)
    assert native["text"][0] == "A A\nA"
    mask = image.convert("L").point(lambda value: 255 if value < 100 else 0)
    assert mask.getbbox()[3] > element["y"] + element["font_size"] * element["line_height"]

    brand = dict(element, font_family="Solstice Sans")
    manager = load_font_manager(DEFAULT_FONTS_DIR)
    widths = char_widths(manager, "Solstice Sans", "regular", "regular", 32, brand["content"])
    expected = wrap_text(brand["content"], widths, brand["width"])
    assert LayoutMeasurer(DEFAULT_FONTS_DIR).measure(brand, 32, brand["width"]).wrapped == expected
    assert EvalMeasurer(DEFAULT_FONTS_DIR).layout(brand).lines == tuple(expected.split("\n"))
    brand_design = {**design, "elements": [brand]}
    example, _ = convert_chameleon_to_crello(brand_design, manager)
    assert native_render_example(example, brand_design, DEFAULT_FONTS_DIR) is example
