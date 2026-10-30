"""Document text effects remain live glyph paints with exact font resources."""

from copy import deepcopy

import pytest
import skia
from cr_renderer import text_utils
from cr_renderer.schema import TextElement

from engine.native_text_effects import draw_native_text, validate_descriptor
from engine.resources import load_font_manager


def _element(**changes):
    return TextElement(
        uuid="fixture",
        type="textElement",
        width=300,
        height=110,
        text="MMMM",
        font="Solstice Sans",
        fontSize=70.75,
        lineHeight=1.2,
        textAlign="left",
        capitalize=False,
        letterSpacing=0,
        colorMap=[{"startIndex": 0, "endIndex": 3, "value": "rgba(0,0,0,1)"}],
        **changes,
    )


def _render(descriptor, element=None):
    surface = skia.Surface(400, 180)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)
    draw_native_text(canvas, load_font_manager(), element or _element(), descriptor)
    return surface.makeImageSnapshot().toarray()


def test_outline_has_hollow_live_glyphs_and_does_not_mutate_metadata():
    descriptor = {"version": 1, "outline": {"color": "#ff0000", "width_em": 0.04, "fill": False}}
    frozen = deepcopy(descriptor)
    outline = _render(descriptor)
    fill = _render({"version": 1})
    assert outline[:, :, 3].sum() < fill[:, :, 3].sum()
    assert outline[:, :, 3].max() == 255
    assert descriptor == frozen


def test_gradient_follows_current_frame_and_live_content():
    descriptor = {
        "version": 1,
        "gradient": {
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 1,
            "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}],
        },
    }
    pixels = _render(descriptor)
    opaque = pixels[:, :, 3] > 250
    ys, xs = opaque.nonzero()
    top = pixels[ys[ys < 30], xs[ys < 30], :3].mean(axis=0)
    bottom = pixels[ys[ys > 55], xs[ys > 55], :3].mean(axis=0)
    # Skia's N32 array is BGRA: blue rises toward the lower endpoint.
    assert top[2] > bottom[2] and top[0] < bottom[0]
    changed = _element()
    changed.text = "I"
    assert _render(descriptor, changed)[:, :, 3].sum() < pixels[:, :, 3].sum()


def test_outline_width_uses_current_fractional_font_size():
    class Canvas:
        def __init__(self):
            self.widths = []

        def drawTextBlob(self, blob, x, y, paint):
            self.widths.append(paint.getStrokeWidth())

    descriptor = {"version": 1, "outline": {"color": "#000000", "width_em": 0.05, "fill": False}}
    canvas = Canvas()
    draw_native_text(canvas, load_font_manager(), _element(), descriptor)
    assert canvas.widths == pytest.approx([70.75 * 0.05] * 4)


def test_transparent_first_gradient_stop_does_not_hide_later_opaque_stops():
    descriptor = {
        "version": 1,
        "gradient": {
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 1,
            "stops": [
                {"offset": 0, "color": "rgba(255,0,0,0)"},
                {"offset": 1, "color": "rgba(0,0,255,1)"},
            ],
        },
    }
    element = _element()
    element.colorMap = text_utils.make_default_map(len(element.text), "rgba(255,0,0,0)")
    pixels = _render(descriptor, element)
    assert pixels[:, :, 3].max() > 100


def test_explicit_native_frame_baseline_scales_without_changing_adobe_default():
    class Canvas:
        def __init__(self):
            self.ys = []

        def drawTextBlob(self, blob, x, y, paint):
            self.ys.append(y)

    canvas = Canvas()
    element = _element()
    draw_native_text(canvas, load_font_manager(), element, {"version": 1, "baseline_em": 0.87914})
    assert canvas.ys == pytest.approx([70.75 * 0.87914] * 4)
    default = Canvas()
    draw_native_text(default, load_font_manager(), element, {"version": 1})
    assert default.ys != pytest.approx(canvas.ys)


def test_em_tracking_preserves_source_pixels_and_scales_with_current_glyphs():
    import numpy as np

    element = _element()
    element.letterSpacing = 7.075
    previous = _render({"version": 1}, element)
    element.letterSpacing = 0
    current = _render({"version": 1, "tracking_em": 0.1}, element)
    assert np.array_equal(previous, current)

    class Canvas:
        def __init__(self):
            self.xs = []

        def drawTextBlob(self, blob, x, y, paint):
            self.xs.append(x)

    full, full_untracked, half, half_untracked = Canvas(), Canvas(), Canvas(), Canvas()
    manager = load_font_manager()
    draw_native_text(full_untracked, manager, element, {"version": 1})
    draw_native_text(full, manager, element, {"version": 1, "tracking_em": 0.3})
    full_size = element.fontSize
    element.fontSize /= 2
    element.width /= 2
    element.height /= 2
    draw_native_text(half_untracked, manager, element, {"version": 1})
    draw_native_text(half, manager, element, {"version": 1, "tracking_em": 0.3})
    # Native outline advances and authored tracking share one scale, while the
    # glyphs remain live rasterized text. Check both components independently.
    assert half_untracked.xs == pytest.approx(
        [value / 2 for value in full_untracked.xs], rel=0, abs=1e-5
    )
    full_tracking = [a - b for a, b in zip(full.xs, full_untracked.xs, strict=True)]
    half_tracking = [a - b for a, b in zip(half.xs, half_untracked.xs, strict=True)]
    assert full_tracking == pytest.approx([index * 0.3 * full_size for index in range(4)])
    assert half_tracking == pytest.approx([value / 2 for value in full_tracking])


def test_exact_run_face_is_requested_after_wrapping_without_italic_fallback():
    data = load_font_manager().lookup("Solstice Sans", "regular", "regular")

    class Manager:
        def __init__(self):
            self.requests = []

        def lookup(self, family, weight, style):
            self.requests.append((family, weight, style))
            if family not in {"Solstice Sans", "Exact-Italic"}:
                raise KeyError(family)
            return data

    manager = Manager()
    element = _element()
    element.text = "AA\nBB"
    element.lineMap = text_utils.generate_map([0, 0, 1, 1, 1])
    descriptor = {
        "version": 1,
        "source_text": "AA BB",
        "runs": [{"start": 3, "end": 5, "font_family": "Exact-Italic"}],
    }
    draw_native_text(skia.Surface(400, 180).getCanvas(), manager, element, descriptor)
    assert manager.requests.count(("Exact-Italic", "regular", "regular")) == 2
    assert not any(style == "italic" for _, _, style in manager.requests)


@pytest.mark.parametrize(
    "descriptor",
    [
        {"version": 2},
        {"version": 1, "blur": 4},
        {"version": 1, "runs": [{"start": 0, "end": 10, "font_family": "Face"}]},
        {"version": 1, "outline": {"color": "#fff", "width_em": -1, "fill": False}},
    ],
)
def test_unknown_or_invalid_effects_fail_explicitly(descriptor):
    with pytest.raises(ValueError):
        validate_descriptor(descriptor, "abc")
