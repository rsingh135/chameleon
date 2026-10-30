"""Hard requirements a resized design must meet (RULES.md, the schema, and
the resize() docstring). A candidate with any violation is disqualified
before quality is considered, so the search never prefers a pretty layout
that would be thrown out by the structural check.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from numbers import Real

from engine.eval.context import EDGE_SLACK, Context, build_context
from engine.eval.geometry import canvas_box, contains, element_box
from engine.eval.text import get_measurer

ASPECT_TOLERANCE = 0.05
MIN_FONT_SIZE = 1.0
TEXT_SLACK = 0.02  # fraction of box the renderer may spill before we call it overflow

REQUIRED_COMMON = ("id", "type", "x", "y", "width", "height")
REQUIRED_BY_TYPE = {
    "text": ("content", "font_family", "font_size", "color"),
    "image": ("asset_id",),
    "shape": ("shape_type", "fill_color"),
}
NUMERIC_GEOMETRY = (
    "x",
    "y",
    "width",
    "height",
    "font_size",
    "rotation",
    "corner_radius",
    "stroke_width",
)
# Fields the engine may not change: everything that is the design's identity
# rather than its geometry. Position, size, font_size, and the px-valued shape
# geometry (`corner_radius`, `stroke_width`) may scale with the element.
FROZEN_FIELDS = (
    "type",
    "content",
    "font_family",
    "font_weight",
    "font_style",
    "color",
    "text_align",
    "line_height",
    "letter_spacing",
    "style_runs",
    "native_text_effects",
    "source_art",
    "asset_id",
    "fit",
    "shape_type",
    "fill_color",
    "stroke_color",
    "group_id",
    "z_index",
    "opacity",
    "rotation",
)


@dataclass(frozen=True)
class Violation:
    code: str
    element_id: str | None
    message: str

    def __str__(self) -> str:
        where = f"[{self.element_id}] " if self.element_id else ""
        return f"{self.code}: {where}{self.message}"


def _finite_real(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_contract(
    source: dict,
    candidate: dict,
    target_canvas: dict | None = None,
    context: Context | None = None,
) -> list[Violation]:
    out: list[Violation] = []
    if not isinstance(candidate, dict) or "canvas" not in candidate or "elements" not in candidate:
        return [Violation("structure", None, "output lacks top-level 'canvas' or 'elements'")]

    canvas = candidate["canvas"]
    if target_canvas is not None and (
        canvas.get("width") != target_canvas["width"]
        or canvas.get("height") != target_canvas["height"]
    ):
        out.append(Violation("canvas_mismatch", None, f"canvas {canvas} != target {target_canvas}"))

    src_ids = [el["id"] for el in source["elements"]]
    out_ids = [el.get("id") for el in candidate["elements"]]
    src_set, out_set = set(src_ids), set(out_ids)
    for missing in sorted(src_set - out_set):
        out.append(Violation("id_missing", missing, "element dropped from output"))
    for extra in sorted(out_set - src_set):
        out.append(Violation("id_unexpected", extra, "element not in source"))
    for dup, n in Counter(out_ids).items():
        if n > 1:
            out.append(Violation("id_duplicate", dup, f"appears {n} times"))
    if out_set != src_set or len(out_ids) != len(out_set):
        return out  # further per-element checks would be noise

    context = context or build_context(source)
    cbox = canvas_box(canvas)
    measurer = get_measurer(context.fonts_dir)
    scene = None

    for el in candidate["elements"]:
        eid = el["id"]
        src = context.by_id[eid]

        missing = [f for f in REQUIRED_COMMON if f not in el]
        missing += [f for f in REQUIRED_BY_TYPE.get(el.get("type", ""), ()) if f not in el]
        if missing:
            out.append(Violation("missing_field", eid, f"missing {missing}"))
            continue

        invalid = [f for f in NUMERIC_GEOMETRY if f in el and not _finite_real(el[f])]
        if invalid:
            out.extend(
                Violation("invalid_geometry", eid, f"{f} must be a finite real number: {el[f]!r}")
                for f in invalid
            )
            # Authored off-canvas content may retain its crop, but invalid
            # numbers never reach bounds, aspect ratios or text measurement.
            continue

        cover, support = False, False
        if el["type"] == "image" and src.get("fit") == "fill":
            from engine.scene import analyze_scene, approved_cover, approved_support

            if scene is None:
                scene = analyze_scene(
                    source, assets_dir=context.assets_dir, fonts_dir=context.fonts_dir
                )
            cover = approved_cover(scene, el, canvas)
            support = approved_support(scene, el, canvas)
        for f in FROZEN_FIELDS:
            if f in src and el.get(f) != src[f]:
                if f == "fit" and cover:
                    continue
                out.append(
                    Violation("frozen_field_changed", eid, f"{f}: {src[f]!r} -> {el.get(f)!r}")
                )

        if el["width"] <= 0 or el["height"] <= 0:
            out.append(Violation("degenerate_size", eid, f"{el['width']}x{el['height']}"))
            continue

        if eid in context.in_bounds_ids and not contains(cbox, element_box(el), EDGE_SLACK):
            out.append(
                Violation(
                    "off_canvas",
                    eid,
                    f"({el['x']:.0f},{el['y']:.0f},{el['width']:.0f},{el['height']:.0f}) "
                    f"outside {canvas['width']}x{canvas['height']}",
                )
            )

        if (
            el["type"] == "image"
            and src.get("fit", "contain") != "cover"
            and not (cover or support)
        ):
            src_ratio = src["width"] / src["height"] if src["height"] else 0
            if src_ratio:
                drift = abs(el["width"] / el["height"] - src_ratio) / src_ratio
                if drift > ASPECT_TOLERANCE:
                    out.append(
                        Violation("aspect_drift", eid, f"{drift:.1%} > {ASPECT_TOLERANCE:.0%}")
                    )

        if el["type"] == "text":
            if el["font_size"] < MIN_FONT_SIZE:
                out.append(Violation("font_size_below_min", eid, f"{el['font_size']}"))
                continue
            lay = measurer.layout(el)
            if lay.widest_unbreakable > el["width"] * (1 + TEXT_SLACK) + EDGE_SLACK:
                msg = f"unbreakable word {lay.widest_unbreakable:.0f}px > box {el['width']:.0f}px"
                out.append(Violation("text_overflow_width", eid, msg))
            if lay.height > el["height"] * (1 + TEXT_SLACK) + EDGE_SLACK:
                out.append(
                    Violation(
                        "text_overflow_height",
                        eid,
                        f"{lay.line_count} lines = {lay.height:.0f}px > box {el['height']:.0f}px",
                    )
                )

    return out
