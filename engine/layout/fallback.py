"""A layout that is always valid, for when every proposal fails the contract.

Stage-one scoring gives a task zero credit if the engine produces no output.
A uniformly scaled, centred copy of the source with text re-fitted to its box
is never beautiful, but it always satisfies the structural rules: same ids,
nothing off-canvas, images in proportion, text inside its box. Every pipeline
failure degrades to this instead of to nothing.
"""

from __future__ import annotations

from copy import deepcopy

from .geometry import bounds, contain
from .text import get_measurer


def safe_candidate(source: dict, target: dict) -> dict:
    """Contain-scale the whole design into `target` and re-fit every text box."""
    width, height = float(target["width"]), float(target["height"])
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    scale = min(width / sw, height / sh)
    dx, dy = (width - sw * scale) / 2, (height - sh * scale) / 2
    measurer = get_measurer()
    elements = []
    for original in source["elements"]:
        element = deepcopy(original)
        full_bleed = original["width"] >= sw * 0.9 and original["height"] >= sh * 0.9
        if (
            original["type"] != "text"
            and full_bleed
            and original.get("fit", "contain") != "contain"
        ):
            # Backgrounds keep covering the canvas rather than leaving bars.
            element.update(x=0.0, y=0.0, width=width, height=height)
        else:
            element["x"] = original["x"] * scale + dx
            element["y"] = original["y"] * scale + dy
            element["width"] = max(0.001, original["width"] * scale)
            element["height"] = max(0.001, original["height"] * scale)
        for key in ("corner_radius", "stroke_width"):
            if key in original:
                element[key] = original[key] * scale
        if element["type"] == "text":
            element["font_size"] = max(1.0, original["font_size"] * scale)
            element["font_size"] = measurer.fit(
                element, element["width"], element["height"], element["font_size"]
            )
            _widen_for_unbreakable(element, measurer, width)
            _grow_for_lines(element, measurer, width, height)
        contain(element, width, height)
        elements.append(element)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": elements}


def _widen_for_unbreakable(element: dict, measurer, canvas_width: float) -> None:
    """The renderer never hyphenates: a word wider than its box spills past the
    right edge. Widen the box up to the canvas, then shrink the font."""
    for _ in range(8):
        measured = measurer.measure(element, element["font_size"], element["width"])
        if measured.width <= element["width"] + 0.5:
            return
        if element["width"] < canvas_width - 1:
            grow = min(canvas_width, measured.width + 1.0)
            element["x"] = max(0.0, min(element["x"], canvas_width - grow))
            element["width"] = grow
        else:
            element["font_size"] = max(1.0, element["font_size"] * 0.85)
    box = bounds(element)
    if box.width > canvas_width:
        element["font_size"] = max(1.0, element["font_size"] * canvas_width / box.width)


def _grow_for_lines(element: dict, measurer, canvas_width: float, canvas_height: float) -> None:
    """At the 1px font floor a box can still be shorter than its wrapped
    lines. Widen the box (fewer lines), then give it the height it needs."""
    for _ in range(6):
        measured = measurer.measure(element, element["font_size"], element["width"])
        if measured.height <= element["height"] + 0.5:
            return
        if element["width"] < canvas_width - 1:
            element["x"], element["width"] = 0.0, canvas_width
            continue
        needed = min(canvas_height, measured.height + 0.5)
        element["y"] = max(0.0, min(element["y"], canvas_height - needed))
        element["height"] = needed
