"""Text measurement that matches the official renderer.

cr-renderer wraps text greedily at spaces using Skia glyph advances
(`renderer/converter/wrap.py`), draws each line at
`y = line_index * font_size * line_height`, never clips to the element box,
and never breaks a word that is wider than the box. This module reuses that
exact code path so what we measure is what gets drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from engine.native_typography import layout_native_text, uses_native_typography
from engine.resources import DEFAULT_FONTS_DIR as DEFAULT_FONTS_DIR
from engine.resources import get_resources, load_font_manager
from renderer.converter.convert import _normalize_weight, _weight_style_to_font_style
from renderer.converter.wrap import char_widths, wrap_text

DEFAULT_LINE_HEIGHT = 1.2


@dataclass(frozen=True)
class TextLayout:
    lines: tuple[str, ...]
    line_widths: tuple[float, ...]
    font_size: float
    line_height: float

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def max_line_width(self) -> float:
        return max(self.line_widths, default=0.0)

    @property
    def height(self) -> float:
        """Vertical extent the renderer will draw, from the top of the box."""
        return self.line_count * self.font_size * self.line_height

    @property
    def widest_unbreakable(self) -> float:
        """Width of the widest single word. If this exceeds the box width the
        renderer draws it past the right edge rather than breaking it."""
        return max(self._word_widths, default=0.0)

    _word_widths: tuple[float, ...] = ()


class TextMeasurer:
    """Measures text elements with the renderer's own font manager and
    wrapping. One instance per fonts directory; results are cached."""

    def __init__(self, fonts_dir: str | Path | None = None):
        directory = get_resources().fonts_dir if fonts_dir is None else Path(fonts_dir).resolve()
        self.fonts_dir = str(directory)
        self.native_document = uses_native_typography(directory)
        self._font_manager = load_font_manager(directory)
        self._width_cache: dict[tuple, tuple[float, ...]] = {}

    @staticmethod
    def _style(el: dict) -> tuple[str, str]:
        weight = _normalize_weight(el.get("font_weight"))
        is_bold = weight == "bold"
        is_italic = el.get("font_style") == "italic"
        return weight, _weight_style_to_font_style(is_bold, is_italic)

    def _widths(self, family: str, weight: str, style: str, size: float, text: str) -> tuple:
        key = (family, weight, style, size, text)
        hit = self._width_cache.get(key)
        if hit is None:
            if len(self._width_cache) > 8192:
                self._width_cache.clear()
            hit = tuple(char_widths(self._font_manager, family, weight, style, size, text))
            self._width_cache[key] = hit
        return hit

    def layout(self, el: dict, font_size: float | None = None, width: float | None = None):
        """Lay out a text element as the renderer would. `font_size` and
        `width` override the element's own values (for fit searches)."""
        content = str(el.get("content", ""))
        size = float(font_size if font_size is not None else el.get("font_size", 24))
        box_w = float(width if width is not None else el.get("width", 1.0))
        if self.native_document:
            layout = layout_native_text(el, self._font_manager, size, box_w, self._widths)
            return TextLayout(
                lines=tuple(layout.wrapped.split("\n")),
                line_widths=layout.line_widths,
                font_size=size,
                line_height=float(el.get("line_height") or DEFAULT_LINE_HEIGHT),
                _word_widths=layout.word_widths,
            )
        weight, style = self._style(el)
        family = el.get("font_family", "")
        widths = self._widths(family, weight, style, size, content)
        wrapped = wrap_text(content, list(widths), box_w)

        lines: list[str] = []
        line_widths: list[float] = []
        start = 0
        for i, ch in enumerate(wrapped + "\n"):
            if ch == "\n":
                lines.append(wrapped[start:i])
                line_widths.append(sum(widths[start:i]))
                start = i + 1

        word_widths: list[float] = []
        acc = 0.0
        for i, ch in enumerate(content):
            if ch in (" ", "\n"):
                word_widths.append(acc)
                acc = 0.0
            else:
                acc += widths[i]
        word_widths.append(acc)

        return TextLayout(
            lines=tuple(lines),
            line_widths=tuple(line_widths),
            font_size=size,
            line_height=float(el.get("line_height") or DEFAULT_LINE_HEIGHT),
            _word_widths=tuple(word_widths),
        )

    def fits(self, el: dict, font_size: float | None = None, slack: float = 0.02) -> bool:
        """True if the text stays inside its box at `font_size`."""
        lay = self.layout(el, font_size=font_size)
        box_w = float(el.get("width", 0.0))
        box_h = float(el.get("height", 0.0))
        return lay.widest_unbreakable <= box_w * (1 + slack) + 0.5 and (
            lay.height <= box_h * (1 + slack) + 0.5
        )

    def fit_font_size(self, el: dict, min_size: float = 1.0, max_size: float | None = None):
        """Largest font size in [min_size, max_size] at which the text fits
        its box, or None if even `min_size` overflows. Binary search on a
        0.5px grid; fit is monotone in size for a fixed box."""
        hi = float(max_size if max_size is not None else el.get("font_size", 24))
        lo = float(min_size)
        if hi < lo:
            return None
        if self.fits(el, hi):
            return hi
        if not self.fits(el, lo):
            return None
        while hi - lo > 0.5:
            mid = (lo + hi) / 2
            if self.fits(el, mid):
                lo = mid
            else:
                hi = mid
        return lo


@lru_cache(maxsize=16)
def _get_measurer(fonts_dir: str) -> TextMeasurer:
    return TextMeasurer(fonts_dir)


def get_measurer(fonts_dir: str | Path | None = None) -> TextMeasurer:
    directory = get_resources().fonts_dir if fonts_dir is None else Path(fonts_dir).resolve()
    return _get_measurer(str(directory))


get_measurer.cache_clear = _get_measurer.cache_clear
get_measurer.cache_info = _get_measurer.cache_info
