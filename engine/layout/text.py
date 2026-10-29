"""Renderer-compatible text measurement and bounded font fitting.

The fallback keeps contract tests usable without the rendering dependencies.
Production installs use the same Skia font manager and wrapping as the renderer.
"""

import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from engine.resources import DEFAULT_FONTS_DIR, get_resources, load_font_manager

FONTS_DIR = DEFAULT_FONTS_DIR


@dataclass(frozen=True)
class TextMeasurement:
    width: float
    height: float
    lines: int
    wrapped: str


class TextMeasurer:
    def __init__(self, fonts_dir: str | Path | None = None):
        self.fonts_dir = (
            get_resources().fonts_dir if fonts_dir is None else Path(fonts_dir).resolve()
        )
        self._widths = lru_cache(maxsize=8192)(self._measure_widths)
        self.backend = "estimated"
        self._manager = None
        self._char_widths = None
        self._wrap = None
        try:
            from renderer.converter.wrap import char_widths, wrap_text

            self._manager = load_font_manager(self.fonts_dir)
        except ImportError:
            return
        self._char_widths, self._wrap = char_widths, wrap_text
        self.backend = "skia"

    def _measure_widths(
        self, family: str, weight: str, style: str, size: float, content: str
    ) -> tuple:
        if self._manager is not None:
            return tuple(self._char_widths(self._manager, family, weight, style, size, content))
        return tuple(
            size * (1.0 if unicodedata.east_asian_width(c) in {"F", "W"} else 0.65)
            if c != "\n"
            else 0.0
            for c in content
        )

    def measure(self, element: dict, size: float, width: float) -> TextMeasurement:
        content = str(element.get("content", ""))
        bold = element.get("font_weight", "regular") == "bold"
        italic = element.get("font_style", "normal") == "italic"
        style = (
            "bolditalic"
            if bold and italic
            else "bold"
            if bold
            else "italic"
            if italic
            else "regular"
        )
        widths = self._widths(
            element.get("font_family", "Solstice Sans"),
            "bold" if bold else "regular",
            style,
            round(size, 5),
            content,
        )
        if self._wrap is not None:
            wrapped = self._wrap(content, list(widths), width)
        else:
            # Match the renderer's space-only greedy wrapping, including long words.
            chars, advance, last_space = list(content), 0.0, None
            for i, char in enumerate(chars):
                if char == "\n":
                    advance, last_space = 0.0, None
                    continue
                if advance + widths[i] > width and last_space is not None:
                    chars[last_space] = "\n"
                    advance = sum(widths[last_space + 1 : i + 1])
                    last_space = None
                else:
                    advance += widths[i]
                if char == " ":
                    last_space = i
            wrapped = "".join(chars)
        line_widths, advance, count = [], 0.0, 0
        spacing = max(0.0, float(element.get("letter_spacing", 0)))
        for i, char in enumerate(wrapped):
            if char == "\n":
                line_widths.append(advance + spacing * max(0, count - 1))
                advance, count = 0.0, 0
            else:
                advance += widths[i]
                count += 1
        line_widths.append(advance + spacing * max(0, count - 1))
        lines = len(line_widths)
        return TextMeasurement(
            max(line_widths, default=0.0),
            lines * size * element.get("line_height", 1.2),
            lines,
            wrapped,
        )

    def fit(self, element: dict, width: float, height: float, preferred_size: float) -> float:
        """Largest size up to the preferred size with width and line-height room."""
        lower, upper = 1.0, max(1.0, preferred_size)
        for _ in range(15):
            middle = (lower + upper) / 2
            measured = self.measure(element, middle, max(0.01, width))
            max_lines = element.get("max_lines")
            if (
                measured.width <= width + 1e-6
                and measured.height <= height + 1e-6
                and (max_lines is None or measured.lines <= max_lines)
            ):
                lower = middle
            else:
                upper = middle
        return max(1.0, math.floor(lower * 1000) / 1000)


@lru_cache(maxsize=16)
def _get_measurer(fonts_dir: str) -> TextMeasurer:
    return TextMeasurer(fonts_dir)


def get_measurer(fonts_dir: str | Path | None = None) -> TextMeasurer:
    directory = get_resources().fonts_dir if fonts_dir is None else Path(fonts_dir).resolve()
    return _get_measurer(str(directory))


get_measurer.cache_clear = _get_measurer.cache_clear
get_measurer.cache_info = _get_measurer.cache_info


def fit_text(
    element: dict, width: float, height: float, preferred_size: float | None = None
) -> float:
    return get_measurer().fit(element, width, height, preferred_size or element["font_size"])
