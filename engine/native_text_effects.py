"""Live document text paints, using exact faces and current editable glyphs.

Descriptors are immutable source metadata, not pre-rendered images. Outline
widths are em values so a resized message keeps its authored stroke ratio.
Declared native tracking is also in em; other documents retain pixel tracking.
Gradient endpoints use the current text frame. Run indices are end-exclusive
offsets in the current source content, before the converter inserts line breaks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import skia
from cr_renderer import text_utils
from cr_renderer.color_utils import parse_color

from engine.native_typography import native_font


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"native text {name} must be finite")
    return float(value)


def validate_descriptor(descriptor, content=None):
    """Reject unsupported effects rather than quietly rendering ordinary text."""
    if not isinstance(descriptor, Mapping) or descriptor.get("version") != 1:
        raise ValueError("native_text_effects requires version 1")
    if set(descriptor) - {
        "version",
        "outline",
        "gradient",
        "runs",
        "source_text",
        "baseline_em",
        "tracking_em",
    }:
        raise ValueError("unsupported native text effect")
    if "tracking_em" in descriptor:
        _finite(descriptor["tracking_em"], "tracking")
    if "baseline_em" in descriptor and not 0 <= _finite(descriptor["baseline_em"], "baseline") <= 4:
        raise ValueError("invalid native text baseline")
    outline = descriptor.get("outline")
    if outline is not None:
        if not isinstance(outline, Mapping) or set(outline) != {"color", "width_em", "fill"}:
            raise ValueError("invalid native text outline")
        if _finite(outline["width_em"], "outline width") < 0 or type(outline["fill"]) is not bool:
            raise ValueError("invalid native text outline width/fill")
        parse_color(outline["color"])
    gradient = descriptor.get("gradient")
    if gradient is not None:
        if not isinstance(gradient, Mapping) or set(gradient) - {
            "x1",
            "y1",
            "x2",
            "y2",
            "stops",
            "height_factor",
        }:
            raise ValueError("invalid native text gradient")
        coords = tuple(_finite(gradient.get(k), k) for k in ("x1", "y1", "x2", "y2"))
        if coords[:2] == coords[2:]:
            raise ValueError("native text gradient endpoints coincide")
        if _finite(gradient.get("height_factor", 1), "gradient height") <= 0:
            raise ValueError("invalid native text gradient height")
        stops = gradient.get("stops")
        if not isinstance(stops, list) or len(stops) < 2:
            raise ValueError("native text gradient requires at least two stops")
        previous = -1.0
        for stop in stops:
            if not isinstance(stop, Mapping) or set(stop) != {"offset", "color"}:
                raise ValueError("invalid native text gradient stop")
            offset = _finite(stop["offset"], "gradient offset")
            if not previous <= offset <= 1 or offset < 0:
                raise ValueError("native text gradient stops must be ordered inside 0..1")
            previous = offset
            parse_color(stop["color"])
    runs = descriptor.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("native text runs must be an array")
    previous = 0
    for run in runs:
        if not isinstance(run, Mapping) or set(run) != {"start", "end", "font_family"}:
            raise ValueError("invalid native text font run")
        start, end = run["start"], run["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < previous
            or end <= start
            or (content is not None and end > len(content))
            or not isinstance(run["font_family"], str)
            or not run["font_family"]
        ):
            raise ValueError("native text runs must be ordered nonoverlapping source ranges")
        previous = end
    return descriptor


def _source_positions(wrapped, source):
    # The converter substitutes whitespace with line breaks; it does not invent
    # glyphs. Reject changed copy instead of painting a face on the wrong words.
    if len(wrapped) == len(source) and all(
        a == b or (a.isspace() and b.isspace()) for a, b in zip(wrapped, source, strict=True)
    ):
        return list(range(len(source)))
    positions, cursor = [], 0
    for char in wrapped:
        if char.isspace():
            positions.append(cursor)
            if cursor < len(source) and source[cursor].isspace():
                cursor += 1
            continue
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source) or char != source[cursor]:
            raise ValueError("native text run source differs from current wrapped content")
        positions.append(cursor)
        cursor += 1
    if source[cursor:].strip():
        raise ValueError("native text run source contains unrendered glyphs")
    return positions


def _face(font_manager, family, size, bold=False, italic=False):
    style = (
        "bolditalic" if bold and italic else "bold" if bold else "italic" if italic else "regular"
    )
    return native_font(font_manager, family, "bold" if bold else "regular", style, size)


def draw_native_text(canvas, font_manager, element, descriptor):
    """Draw current editable text with live gradient/outline and exact face runs."""
    source = descriptor.get("source_text", element.text)
    if not isinstance(source, str):
        raise ValueError("native text source_text must be text")
    validate_descriptor(descriptor, source)
    positions = _source_positions(element.text, source)
    families = [None] * len(element.text)
    for run in descriptor.get("runs", []):
        for index, source_index in enumerate(positions):
            if run["start"] <= source_index < run["end"]:
                families[index] = run["font_family"]
    colors = text_utils.make_dense_map(element.colorMap, len(element.text), "rgba(0,0,0,1)")
    bolds = text_utils.make_dense_map(element.boldMap, len(element.text), False)
    italics = text_utils.make_dense_map(element.italicMap, len(element.text), False)
    opacities = text_utils.make_dense_map(element.opacityMap, len(element.text), 1.0)
    base = _face(font_manager, element.font, element.fontSize)
    baseline = (
        descriptor["baseline_em"] * element.fontSize
        if "baseline_em" in descriptor
        else -base.getMetrics().fAscent
    )
    tracking = (
        descriptor["tracking_em"] * element.fontSize
        if "tracking_em" in descriptor
        else element.letterSpacing
    )
    gradient, outline = descriptor.get("gradient"), descriptor.get("outline")
    shader = None
    if gradient:
        width, height = element.width, element.height * gradient.get("height_factor", 1)
        shader = skia.GradientShader.MakeLinear(
            [
                (gradient["x1"] * width, gradient["y1"] * height),
                (gradient["x2"] * width, gradient["y2"] * height),
            ],
            [parse_color(stop["color"]) for stop in gradient["stops"]],
            [stop["offset"] for stop in gradient["stops"]],
        )
    lines = element.lineMap or text_utils.make_default_map(len(element.text))
    for line_index, line in enumerate(lines):
        indices = [
            i
            for i in range(line.startIndex, line.endIndex + 1)
            if i < len(element.text) and element.text[i] != "\n"
        ]
        glyphs = []
        for index in indices:
            exact_family = families[index]
            font = _face(
                font_manager,
                exact_family or element.font,
                element.fontSize,
                False if exact_family else bolds[index],
                False if exact_family else italics[index],
            )
            char = element.text[index].upper() if element.capitalize else element.text[index]
            glyph = font.textToGlyphs(char)
            builder = skia.TextBlobBuilder()
            builder.allocRunPosH(font, glyph, font.getXPos(glyph), 0)
            glyphs.append((builder.make(), sum(font.getWidths(glyph)), index))
        advance = sum(width for _, width, _ in glyphs) + tracking * max(0, len(glyphs) - 1)
        x = (element.width - advance) * ({"left": 0, "center": 0.5, "right": 1}[element.textAlign])
        y = line_index * element.fontSize * (element.lineHeight or 1) + baseline
        for blob, width, index in glyphs:
            if not outline or outline["fill"]:
                paint = skia.Paint(AntiAlias=True, Color=parse_color(colors[index]))
                if shader is not None:
                    # Stop alpha belongs to the shader; applying the fallback
                    # first-stop alpha again would hide the entire gradient.
                    paint.setColor(skia.ColorWHITE)
                    paint.setShader(shader)
                paint.setAlphaf(paint.getAlphaf() * float(element.opacity) * opacities[index])
                canvas.drawTextBlob(blob, x, y, paint)
            if outline:
                paint = skia.Paint(
                    AntiAlias=True,
                    Style=skia.Paint.kStroke_Style,
                    StrokeWidth=outline["width_em"] * element.fontSize,
                    Color=parse_color(outline["color"]),
                )
                paint.setAlphaf(paint.getAlphaf() * float(element.opacity) * opacities[index])
                canvas.drawTextBlob(blob, x, y, paint)
            x += width + tracking
