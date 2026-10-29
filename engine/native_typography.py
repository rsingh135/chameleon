"""Document-local wrapping using painted face advances and authored tracking.

Only spaces become newlines, so source style/effect run offsets remain valid.
The bundled renderer and its default font directory keep their existing path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from renderer.converter.convert import _is_italic, _normalize_weight
from renderer.converter.wrap import char_widths

_BOLD = {"bold", "semibold", "extrabold", "black"}


def uses_native_typography(fonts_dir: str | Path) -> bool:
    """Exact document manifests opt in; the bundled kit never does."""
    from engine.resources import DEFAULT_FONTS_DIR

    directory = Path(fonts_dir).resolve()
    return directory != DEFAULT_FONTS_DIR and (directory / "font-manifest.json").is_file()


def _face(family, bold, italic):
    style = (
        "bolditalic" if bold and italic else "bold" if bold else "italic" if italic else "regular"
    )
    return family, "bold" if bold else "regular", style


def _range(run, length):
    start, end = run.get("start"), run.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= length:
        raise ValueError("Native font run ranges must be inside the text content.")
    return range(start, end)


def character_advances(element, font_manager, size=None, width_lookup=None):
    """Return untracked advances for the exact face painted at each source index.

    Ordinary style runs use end-exclusive source ranges. Native effect runs
    already identify a complete PostScript face and request its regular slot,
    exactly as the native effect painter does; no alias substitution is allowed.
    """
    text = str(element.get("content", ""))
    size = float(element.get("font_size", 24) if size is None else size)
    family = element.get("font_family", "")
    bold = _normalize_weight(element.get("font_weight")) in _BOLD
    italic = _is_italic(element.get("font_style") or element.get("italic"))
    families, bolds, italics = [family] * len(text), [bold] * len(text), [italic] * len(text)
    for run in element.get("style_runs") or []:
        indices = _range(run, len(text))
        for index in indices:
            if "font_family" in run:
                families[index] = run["font_family"]
            if "font_weight" in run:
                bolds[index] = _normalize_weight(run["font_weight"]) in _BOLD
            if "font_style" in run or "italic" in run:
                italics[index] = _is_italic(run.get("font_style") or run.get("italic"))
    faces = [_face(f, b, i) for f, b, i in zip(families, bolds, italics, strict=True)]
    for run in (element.get("native_text_effects") or {}).get("runs", []):
        if not isinstance(run.get("font_family"), str) or not run["font_family"]:
            raise ValueError("Native font run requires an exact font_family alias.")
        for index in _range(run, len(text)):
            faces[index] = (run["font_family"], "regular", "regular")

    lookup = width_lookup or (
        lambda family, weight, style, size, text: char_widths(
            font_manager, family, weight, style, size, text
        )
    )
    widths = []
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and faces[end] == faces[start]:
            end += 1
        segment = text[start:end]
        if element.get("capitalize"):
            # Keep one advance per source character even when uppercasing one
            # character produces several glyphs. Native effect runs retain the
            # same original-index semantics when painting their current copy.
            measured = [sum(lookup(*faces[start], size, char.upper())) for char in segment]
        else:
            measured = list(lookup(*faces[start], size, segment))
        if len(measured) != end - start:
            raise ValueError("Native glyph advances must retain source character indices.")
        widths.extend(
            0.0 if char == "\n" else float(w) for char, w in zip(segment, measured, strict=True)
        )
        start = end
    return tuple(widths)


@dataclass(frozen=True)
class NativeLayout:
    wrapped: str
    line_widths: tuple[float, ...]
    word_widths: tuple[float, ...]


def effective_tracking(element, size=None):
    """Native em-based tracking follows type size; pixel-based tracking stays fixed."""
    descriptor = element.get("native_text_effects") or {}
    if "tracking_em" in descriptor:
        em = float(descriptor["tracking_em"])
        font_size = float(element.get("font_size", 24) if size is None else size)
        spacing = em * font_size
    else:
        spacing = float(element.get("letter_spacing", 0))
    if not math.isfinite(spacing):
        raise ValueError("Native letter_spacing must be finite.")
    return spacing


def _wrap(content, advances, limit):
    """Greedy space-only wrap; trailing whitespace cannot eject a fitting word."""
    chars = list(content)
    prefix = [0.0, *accumulate(advances)]
    width, last_space = 0.0, None
    for index, char in enumerate(content):
        if char == "\n":
            width, last_space = 0.0, None
        elif char == " ":
            # Defer a break decision until the next word. Breaking on this
            # whitespace would unnecessarily move the preceding fitting word.
            width += advances[index]
            last_space = index
        elif width + advances[index] > limit and last_space is not None:
            chars[last_space] = "\n"
            width = prefix[index + 1] - prefix[last_space + 1]
            last_space = None
        else:
            width += advances[index]
    return "".join(chars)


def layout_native_text(element, font_manager, size=None, width=None, width_lookup=None):
    """Wrap with signed tracking while omitting tracking after each final glyph."""
    content = str(element.get("content", ""))
    width = float(element.get("width", 1) if width is None else width)
    spacing = effective_tracking(element, size)
    advances = character_advances(element, font_manager, size, width_lookup)
    adjusted = [
        advance + spacing if char != "\n" else 0.0
        for char, advance in zip(content, advances, strict=True)
    ]
    # Adding spacing to both sides compares n advances + n tracking values
    # against width + tracking: only n-1 spaces belong between the glyphs.
    # Extreme negative tracking can compress a logical advance below zero;
    # retain its signed behavior while bounding the wrap limit and line width.
    limit = max(1e-9, width + spacing)
    wrapped = _wrap(content, adjusted, limit) if width > 0 else content
    lines, words = [], []
    advance, count, word_advance, word_count = 0.0, 0, 0.0, 0
    for char, glyph in zip(wrapped + "\n", (*advances, 0.0), strict=True):
        if char in {" ", "\n"}:
            words.append(max(0.0, word_advance + spacing * max(0, word_count - 1)))
            word_advance, word_count = 0.0, 0
        else:
            word_advance += glyph
            word_count += 1
        if char == "\n":
            lines.append(max(0.0, advance + spacing * max(0, count - 1)))
            advance, count = 0.0, 0
        else:
            advance += glyph
            count += 1
    return NativeLayout(wrapped, tuple(lines), tuple(words))


def rewrap_example(example, design, font_manager):
    """Replace native text/line columns on a copy, preserving all source indices.

    Call only after the resource layer has selected a document manifest. Other
    converter columns, including paint/style maps and image pixels, are retained.
    """
    document = design.get("source", design)
    elements = sorted(document["elements"], key=lambda element: element.get("z_index", 0))
    if (
        example.get("length") != len(elements)
        or len(example.get("text", [])) != len(elements)
        or len(example.get("text_line", [])) != len(elements)
    ):
        raise ValueError("Native wrapping does not match converted element count.")
    result = {
        **example,
        "text": list(example["text"]),
        "text_line": [list(line) for line in example["text_line"]],
    }
    for index, element in enumerate(elements):
        if element.get("type") != "text":
            continue
        source = str(element.get("content", ""))
        converted = example["text"][index]
        if len(source) != len(converted) or any(
            a != b and not (a == " " and b == "\n") for a, b in zip(source, converted, strict=True)
        ):
            raise ValueError("Native wrapping differs from converted source character indices.")
        wrapped = layout_native_text(element, font_manager).wrapped
        result["text"][index] = wrapped
        result["text_line"][index] = list(accumulate(int(char == "\n") for char in wrapped))
    return result
