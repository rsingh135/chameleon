"""Proxy quality metrics for a resized design, mirroring the rubric in the
brief: no unwanted collisions, everything readable, on brand, intentional,
holds up everywhere. Each metric is a float in [0, 1] where 1 is best, so
they can be weighted and compared across candidates. Raw diagnostics that
are not scores (counts, pixel sizes) are prefixed `raw_`.

JSON-only metrics need no rendering. Pixel metrics (`text_contrast`,
`text_busyness`) need a rendered plate; they are skipped when none is given.
"""

from __future__ import annotations

import math
import re
from itertools import combinations

from PIL import Image

from engine.eval.context import Context
from engine.eval.geometry import (
    Box,
    canvas_box,
    contains,
    element_box,
    overlap_fraction,
    rotated_aabb,
    union,
)
from engine.eval.render import region_stats
from engine.eval.text import get_measurer
from engine.layout.appearance import visible_box
from engine.layout.model import build_context as build_layout_context

# Minimum comfortably readable text: a fraction of the shorter canvas side,
# floored in px. 1080-wide -> 13px; 728x90 -> the 10px floor applies.
MIN_FONT_FRACTION = 0.012
MIN_FONT_PX = 10.0
# Target margin from the canvas edge for foreground elements that were not
# bleeding in the source, as a fraction of the shorter side.
MARGIN_FRACTION = 0.03
ALIGN_SNAP = 0.01  # snap tolerance for alignment, as a fraction of canvas size
_RGBA_RE = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+))?\s*\)")


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _lin(c: float) -> float:
    """sRGB channel (0..1) -> linear light, per WCAG 2."""
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(color: str) -> tuple[float, float] | None:
    """(relative luminance 0..1, alpha 0..1) of a CSS rgba() string."""
    m = _RGBA_RE.match(color or "")
    if not m:
        return None
    r, g, b = (float(m.group(i)) / 255.0 for i in (1, 2, 3))
    alpha = float(m.group(4)) if m.group(4) is not None else 1.0
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b), max(0.0, min(1.0, alpha))


def _contrast_ratio(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _text_line_boxes(element: dict, layout) -> list[Box]:
    """Occupied line bounds, rather than the possibly oversized text frame.

    These bounds use glyph advances and line height, not rasterized glyph ink.
    The caller uses them only for unrotated text.
    """
    align = {"left": 0, "center": 0.5, "right": 1}.get(element.get("text_align"), 0)
    height = layout.font_size * layout.line_height
    boxes = []
    for index, (line, width) in enumerate(zip(layout.lines, layout.line_widths, strict=True)):
        if not line.strip() or width <= 0:
            continue
        x = element["x"] + (element["width"] - width) * align
        y = element["y"] + index * height
        boxes.append(Box(x, y, x + width, y + height))
    return boxes


def _line_overlap(first: list[Box], second: list[Box]) -> float:
    """Largest occupied-line intersection; both lists are ordered vertically."""
    i = j = 0
    maximum = 0.0
    while i < len(first) and j < len(second):
        a, b = first[i], second[j]
        maximum = max(maximum, overlap_fraction(a, b))
        if a.y1 <= b.y1:
            i += 1
        else:
            j += 1
    return maximum


def _occupied_boxes(element: dict, text_lines: dict[str, list[Box]], assets_dir: str) -> list[Box]:
    """Approximate visible rectangles, not a pixel-level collision mask.

    Use individual occupied lines for unrotated text. Rotated text, images and
    shapes use their visible axis-aligned bounds, including image alpha/fitting.
    """
    if element.get("opacity", 1) <= 0:
        return []
    if element["type"] == "text" and not element.get("rotation", 0):
        return text_lines[element["id"]]
    box = visible_box(element, assets_dir)
    if box.width <= 0 or box.height <= 0:
        return []
    return [Box(box.x, box.y, box.x + box.width, box.y + box.height)]


def _decorative_underlays(source: dict, context: Context) -> set[str]:
    """Reuse source ornament classification without treating label plates as art."""
    decorations = {element["id"] for element in build_layout_context(source).decorations}
    label_plates = {
        outer
        for inner, outer in context.containment
        if context.by_id[inner]["type"] == "text" and not context.by_id[inner].get("group_id")
    }
    return decorations - label_plates


def compute_metrics(
    source: dict,
    candidate: dict,
    context: Context,
    plate: Image.Image | None = None,
) -> dict[str, float]:
    m: dict[str, float] = {}
    cand_by_id = {el["id"]: el for el in candidate["elements"]}
    ccanvas = canvas_box(candidate["canvas"])
    scanvas = context.canvas
    fg = [i for i in context.foreground_ids if i in cand_by_id]
    cboxes = {i: rotated_aabb(cand_by_id[i]) for i in cand_by_id}
    sboxes = {i: rotated_aabb(context.by_id[i]) for i in context.by_id}
    short_side = min(ccanvas.width, ccanvas.height)
    measurer = get_measurer(context.fonts_dir)
    texts = [i for i in fg if cand_by_id[i]["type"] == "text"]
    candidate_layouts = {i: measurer.layout(cand_by_id[i]) for i in texts}
    candidate_lines = {i: _text_line_boxes(cand_by_id[i], candidate_layouts[i]) for i in texts}
    source_lines = {i: _text_line_boxes(context.by_id[i], context.text_layouts[i]) for i in texts}
    candidate_occupied = {
        i: _occupied_boxes(cand_by_id[i], candidate_lines, context.assets_dir) for i in fg
    }
    source_occupied = {
        i: _occupied_boxes(context.by_id[i], source_lines, context.assets_dir) for i in fg
    }
    underlays = _decorative_underlays(source, context)

    # --- collisions: overlaps that did not exist in the source ---------------
    worst = 0.0
    total = 0.0
    count = 0
    for a, b in combinations(fg, 2):
        if any(
            decoration in underlays
            and other not in underlays
            and context.by_id[decoration].get("z_index", 0) < context.by_id[other].get("z_index", 0)
            and cand_by_id[decoration].get("z_index", 0) < cand_by_id[other].get("z_index", 0)
            for decoration, other in ((a, b), (b, a))
        ):
            # Source low-opacity ungrouped ornaments may sit behind foreground
            # content. Their rectangular bounds do not model the painted shape;
            # rendered contrast/busyness must still assess text legibility.
            continue
        unrotated_text = cand_by_id[a]["type"] == cand_by_id[b]["type"] == "text" and not any(
            element.get("rotation", 0)
            for element in (cand_by_id[a], cand_by_id[b], context.by_id[a], context.by_id[b])
        )
        if "image" in (cand_by_id[a]["type"], cand_by_id[b]["type"]):
            # An overlapping padded source frame does not establish visible
            # layering. Allow only the overlap measured in the source itself.
            baseline = _line_overlap(source_occupied[a], source_occupied[b])
            frac = _line_overlap(candidate_occupied[a], candidate_occupied[b]) - baseline
            if frac <= 0.02:
                continue
        elif unrotated_text:
            # Overlapping source frames do not authorize a new collision of
            # their drawn lines, including labels within the same group.
            baseline = _line_overlap(source_lines[a], source_lines[b])
            frac = _line_overlap(candidate_lines[a], candidate_lines[b]) - baseline
            if frac <= 0.02:
                continue
        else:
            if context.same_group(a, b) or frozenset((a, b)) in context.intentional_overlaps:
                continue
            frac = overlap_fraction(cboxes[a], cboxes[b])
        if frac <= 0.0:
            continue
        weight = 2.0 if "text" in (cand_by_id[a]["type"], cand_by_id[b]["type"]) else 1.0
        total += frac * weight
        worst = max(worst, frac * weight)
        count += 1
    m["raw_collisions"] = float(count)
    m["collision_free"] = _clamp(1.0 - total / max(1, len(fg)) * 2 - (0.5 if worst >= 1.0 else 0))

    # --- text readability --------------------------------------------------
    if texts:
        threshold = max(MIN_FONT_PX, MIN_FONT_FRACTION * short_side)
        min_font = min(cand_by_id[i]["font_size"] for i in texts)
        m["raw_min_font_px"] = float(min_font)
        m["text_size"] = _clamp(min_font / threshold)

        growth = []
        for i in texts:
            src_lines = max(1, context.text_layouts[i].line_count)
            lay = candidate_layouts[i]
            growth.append(lay.line_count / src_lines)
        m["raw_line_growth"] = float(sum(growth) / len(growth))
        m["text_lines_stable"] = _clamp(1.0 - max(0.0, (m["raw_line_growth"] - 1.0)) / 2.0)

        # Hierarchy: font-size ordering between text pairs preserved. A pair
        # whose source sizes differ by 1.5x or more and comes out inverted is
        # a visible design error (legal line larger than the headline).
        kept = tot = 0
        major_inversions = 0
        for a, b in combinations(texts, 2):
            sa, sb = context.by_id[a]["font_size"], context.by_id[b]["font_size"]
            if sa == sb:
                continue
            tot += 1
            ca, cb = cand_by_id[a]["font_size"], cand_by_id[b]["font_size"]
            if (sa > sb) == (ca > cb) and ca != cb:
                kept += 1
            elif max(sa, sb) / min(sa, sb) >= 1.5 and (sa > sb) != (ca > cb):
                major_inversions += 1
        m["hierarchy"] = kept / tot if tot else 1.0
        m["raw_hierarchy_inversions"] = float(major_inversions)
    else:
        m["text_size"] = m["text_lines_stable"] = m["hierarchy"] = 1.0

    # --- relationships ----------------------------------------------------
    if context.containment:
        kept = sum(
            1
            for inner, outer in context.containment
            if contains(cboxes[outer], cboxes[inner], slack=1.0)
        )
        m["containment"] = kept / len(context.containment)
    else:
        m["containment"] = 1.0

    if context.groups:
        scores = []
        for members in context.groups.values():
            su = union([sboxes[i] for i in members])
            cu = union([cboxes[i] for i in members])
            if su is None or cu is None or su.area == 0 or cu.area == 0:
                continue
            err = 0.0
            for i in members:
                s, c = sboxes[i], cboxes[i]
                # Normalised position of the member's centre inside the group box.
                sx = (s.cx - su.x0) / max(su.width, 1e-6)
                sy = (s.cy - su.y0) / max(su.height, 1e-6)
                cx = (c.cx - cu.x0) / max(cu.width, 1e-6)
                cy = (c.cy - cu.y0) / max(cu.height, 1e-6)
                err += abs(sx - cx) + abs(sy - cy)
            scores.append(_clamp(1.0 - err / len(members)))
        m["group_integrity"] = sum(scores) / len(scores) if scores else 1.0
    else:
        m["group_integrity"] = 1.0

    # --- hero prominence ---------------------------------------------------
    imgs = [i for i in fg if cand_by_id[i]["type"] == "image"]
    if imgs:
        hero = max(imgs, key=lambda i: sboxes[i].area)
        src_share = sboxes[hero].area / scanvas.area
        cand_share = cboxes[hero].area / ccanvas.area
        still_largest = max(imgs, key=lambda i: cboxes[i].area) == hero
        ratio = cand_share / src_share if src_share else 1.0
        # On a strip or tower the hero cannot keep its area share; filling
        # the short side is the most prominent it can be, so credit that too.
        src_short = max(sboxes[hero].width, sboxes[hero].height) / min(
            scanvas.width, scanvas.height
        )
        cand_short = max(cboxes[hero].width, cboxes[hero].height) / min(
            ccanvas.width, ccanvas.height
        )
        short_ratio = cand_short / src_short if src_short else 1.0
        m["hero"] = _clamp(min(max(ratio, short_ratio), 1.0)) * (1.0 if still_largest else 0.6)
    else:
        m["hero"] = 1.0

    # --- position affinity: elements stay in the region they came from -----
    # A corner ribbon should still be in a corner, a footer watermark still in
    # the footer. Compare each element's normalised centre (0..1 in each axis)
    # between source and candidate; distances beyond 0.35 count as moved.
    if fg:
        drift = []
        for i in fg:
            s, c = sboxes[i], cboxes[i]
            dx = s.cx / scanvas.width - c.cx / ccanvas.width
            dy = s.cy / scanvas.height - c.cy / ccanvas.height
            drift.append(_clamp(1.0 - math.hypot(dx, dy) / 0.35))
        affinity = sum(drift) / len(drift)
        # When the canvas shape changes a lot, elements *must* move; blend the
        # metric toward neutral so a strip relayout is not punished for it.
        aspect_change = abs(
            math.log((ccanvas.width / ccanvas.height) / (scanvas.width / scanvas.height))
        )
        relax = _clamp(aspect_change / math.log(6.0))
        m["position_affinity"] = affinity * (1 - relax) + relax
    else:
        m["position_affinity"] = 1.0

    # --- relative scale consistency ---------------------------------------
    if len(fg) >= 2:
        logs = []
        for a, b in combinations(fg, 2):
            sa, sb = sboxes[a].area, sboxes[b].area
            ca, cb = cboxes[a].area, cboxes[b].area
            if min(sa, sb, ca, cb) <= 0:
                continue
            logs.append(abs(math.log((ca / cb) / (sa / sb))))
        m["scale_consistency"] = (
            _clamp(1.0 - (sum(logs) / len(logs)) / math.log(4)) if logs else 1.0
        )
    else:
        m["scale_consistency"] = 1.0

    # --- alignment, balance, margin, coverage ------------------------------
    if len(fg) >= 2:
        tol = ALIGN_SNAP * short_side
        anchors = {
            i: (
                (0, cboxes[i].x0),
                (1, cboxes[i].cx),
                (2, cboxes[i].x1),
                (3, cboxes[i].y0),
                (4, cboxes[i].cy),
                (5, cboxes[i].y1),
            )
            for i in fg
        }
        aligned = 0
        for i in fg:
            others = [j for j in fg if j != i]
            if any(
                abs(va - vb) <= tol
                for ka, va in anchors[i]
                for j in others
                for kb, vb in anchors[j]
                if ka == kb
            ):
                aligned += 1
        m["alignment"] = aligned / len(fg)
    else:
        m["alignment"] = 1.0

    if fg:
        tot_area = sum(cboxes[i].area for i in fg) or 1.0
        com_x = sum(cboxes[i].cx * cboxes[i].area for i in fg) / tot_area
        com_y = sum(cboxes[i].cy * cboxes[i].area for i in fg) / tot_area
        off = math.hypot(
            (com_x - ccanvas.cx) / ccanvas.width, (com_y - ccanvas.cy) / ccanvas.height
        )
        m["balance"] = _clamp(1.0 - off * 2.0)

        target = MARGIN_FRACTION * short_side
        margin_scores = []
        for i in fg:
            if i not in context.in_bounds_ids:
                continue
            b = element_box(cand_by_id[i])
            gap = min(b.x0, b.y0, ccanvas.x1 - b.x1, ccanvas.y1 - b.y1)
            margin_scores.append(_clamp(gap / target) if target > 0 else 1.0)
        m["margin"] = sum(margin_scores) / len(margin_scores) if margin_scores else 1.0

        scov = _union_coverage([sboxes[i] for i in fg], scanvas)
        ccov = _union_coverage([cboxes[i] for i in fg], ccanvas)
        m["raw_coverage"] = ccov
        m["coverage"] = _clamp(1.0 - abs(ccov - scov) / max(scov, 0.2))
    else:
        m["balance"] = m["margin"] = m["coverage"] = 1.0

    # --- pixels under text -------------------------------------------------
    if plate is not None and texts:
        contrasts, busy = [], []
        for i in texts:
            el = cand_by_id[i]
            parsed = _luminance(el.get("color", ""))
            mean_srgb, busyness = region_stats(plate, cboxes[i], candidate["canvas"])
            plate_lum = _lin(mean_srgb)
            if parsed is not None:
                lum_text, alpha = parsed
                alpha *= float(el.get("opacity", 1.0) or 1.0)
                effective = alpha * lum_text + (1 - alpha) * plate_lum
                contrasts.append(_contrast_ratio(effective, plate_lum))
            busy.append(busyness)
        if contrasts:
            per_text = [_clamp((c - 1.0) / 3.0) for c in contrasts]  # 4:1 -> 1.0
            m["raw_min_contrast"] = min(contrasts)
            m["text_contrast"] = 0.5 * min(per_text) + 0.5 * sum(per_text) / len(per_text)
        m["raw_max_busyness"] = max(busy)
        m["text_busyness"] = _clamp(1.0 - max(busy) * 4.0)

    return m


def _union_coverage(boxes: list[Box], canvas: Box, grid: int = 64) -> float:
    """Fraction of the canvas covered by the union of boxes, on a coarse grid."""
    if not boxes or canvas.area <= 0:
        return 0.0
    cw, ch = canvas.width / grid, canvas.height / grid
    hit = 0
    for gy in range(grid):
        y = canvas.y0 + (gy + 0.5) * ch
        for gx in range(grid):
            x = canvas.x0 + (gx + 0.5) * cw
            if any(b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1 for b in boxes):
                hit += 1
    return hit / (grid * grid)
