"""Reuse exact font faces without changing glyph advances or native paint."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import skia

from engine import native_typography as typography

FONT_DIR = Path(__file__).resolve().parents[1] / "brand_kit/fonts"


@pytest.fixture
def fonts():
    return [
        (FONT_DIR / name).read_bytes()
        for name in ("SolsticeSans-Regular.ttf", "SolsticeSans-Bold.ttf")
    ]


@pytest.fixture(autouse=True)
def clear_cache():
    if hasattr(typography, "clear_native_typeface_cache"):
        typography.clear_native_typeface_cache()
    yield
    if hasattr(typography, "clear_native_typeface_cache"):
        typography.clear_native_typeface_cache()


class Manager:
    def __init__(self, data):
        self.data = data

    def lookup(self, family, weight, style):
        return self.data


def instrument(monkeypatch):
    original = skia.Typeface.MakeFromData
    calls = []

    def make(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(typography.skia, "Typeface", SimpleNamespace(MakeFromData=make))
    return calls, original


def test_repeated_exact_bytes_parse_once_but_return_fresh_fonts(monkeypatch, fonts):
    calls, _ = instrument(monkeypatch)
    manager = Manager(fonts[0])
    created = [
        typography.native_font(manager, "same", "regular", "regular", size)
        for size in (12, 12, 12.125, 31.387, 48)
    ]
    assert len(calls) == 1
    assert all(created[i] is not created[j] for i in range(len(created)) for j in range(i))
    created[0].setScaleX(1.7)
    assert created[1].getScaleX() == 1
    assert created[2].getSize() == pytest.approx(12.125, abs=0)


def test_same_alias_changed_font_bytes_miss_and_match_legacy_widths(monkeypatch, fonts):
    calls, original = instrument(monkeypatch)
    manager = Manager(fonts[0])
    text = "office Wé—!"
    for data in fonts:
        manager.data = data
        measured = typography.native_char_widths(
            manager, "same", "regular", "regular", 17.135, text
        )
        expected_font = skia.Font(original(data), 17.135)
        expected_font.setLinearMetrics(True)
        expected = tuple(expected_font.getWidths(expected_font.textToGlyphs(text)))
        assert measured == expected
    assert len(calls) == 2


def test_native_png_bytes_match_uncached_face_creation(fonts):
    def png(font):
        surface = skia.Surface(280, 80)
        canvas = surface.getCanvas()
        canvas.clear(skia.ColorWHITE)
        canvas.drawString("office Wé—!", 8, 45, font, skia.Paint(Color=skia.ColorBLACK))
        return bytes(surface.makeImageSnapshot().encodeToData())

    manager = Manager(fonts[0])
    cached = typography.native_font(manager, "same", "regular", "regular", 21.375)
    legacy = skia.Font(skia.Typeface.MakeFromData(fonts[0]), 21.375)
    legacy.setLinearMetrics(True)
    assert png(cached) == png(legacy)


def test_count_bound_evicts_lru_and_oversize_bytes_are_not_retained(monkeypatch, fonts):
    monkeypatch.setattr(typography, "_TYPEFACE_CACHE_LIMIT", 1)
    calls, _ = instrument(monkeypatch)
    manager = Manager(fonts[0])
    for data in (fonts[0], fonts[1], fonts[0]):
        manager.data = data
        typography.native_font(manager, "same", "regular", "regular", 20)
    assert len(calls) == 3
    assert len(typography._typefaces) == 1
    typography.clear_native_typeface_cache()
    monkeypatch.setattr(typography, "_TYPEFACE_CACHE_BYTE_LIMIT", len(fonts[0]) - 1)
    for _ in range(2):
        typography.native_font(manager, "same", "regular", "regular", 20)
    assert len(calls) == 5
    assert len(typography._typefaces) == 0
    assert typography._typeface_bytes == 0


def test_byte_budget_evicts_faces_even_below_count_bound(monkeypatch, fonts):
    monkeypatch.setattr(typography, "_TYPEFACE_CACHE_BYTE_LIMIT", max(map(len, fonts)))
    calls, _ = instrument(monkeypatch)
    manager = Manager(fonts[0])
    for data in (fonts[0], fonts[1], fonts[0]):
        manager.data = data
        typography.native_font(manager, "same", "regular", "regular", 20)
    assert len(calls) == 3
    assert typography._typeface_bytes <= typography._TYPEFACE_CACHE_BYTE_LIMIT


def test_mutable_input_is_snapshotted_and_failed_faces_are_not_cached(monkeypatch, fonts):
    calls, _ = instrument(monkeypatch)
    manager = Manager(bytearray(fonts[0]))
    typography.native_font(manager, "same", "regular", "regular", 20)
    manager.data[:] = fonts[1]
    typography.native_font(manager, "same", "regular", "regular", 20)
    assert len(calls) == 2
    size = len(typography._typefaces)
    manager.data = b"not a font"
    for _ in range(2):
        with pytest.raises(ValueError, match="Invalid native text face"):
            typography.native_font(manager, "same", "regular", "regular", 20)
    assert len(calls) == 4
    assert len(typography._typefaces) == size


def test_missing_face_lookup_still_fails_without_fallback(fonts):
    class Missing(Manager):
        def lookup(self, *args):
            raise LookupError("missing exact face")

    with pytest.raises(LookupError, match="missing exact face"):
        typography.native_font(Missing(fonts[0]), "missing", "regular", "regular", 20)
