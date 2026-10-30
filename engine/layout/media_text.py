"""Bounded live-copy wrapping inside an already chosen information region.

The caller supplies source rows and a typography budget. This module changes
only geometry: singleton ungrouped text may wrap, while every other source row
retains one similarity transform. It does not infer media ownership or choose
a layout, and never changes copy, font faces, paint, tracking or source objects.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from .geometry import Box, bounds, union
from .text import get_measurer

MAX_ELEMENTS = 128
MAX_ROWS = 32
MAX_CHARACTERS = 20_000
SEARCH_STEPS = 24
REFINE_STEPS = 12
_EPS = 0.015
_FIELDS = ("width", "height", "font_size", "corner_radius", "stroke_width")


@dataclass(frozen=True)
class MediaTextPacking:
    elements: tuple[dict, ...]
    font_scale: float
    # One scale per input row. Flexible rows use font_scale; rigid rows can
    # uniformly shrink to fit their complete original width, including gaps.
    rigid_row_scales: tuple[float, ...]
    wrapped_element_ids: tuple[str, ...]
    exceptions: tuple[str, ...]


@dataclass(frozen=True)
class _Row:
    elements: tuple[dict, ...]
    box: Box
    flexible: bool


def _finite_positive(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _prepare(source, rows):
    if not rows or len(rows) > MAX_ROWS:
        return None
    originals = {element["id"]: element for element in source["elements"]}
    if len(originals) != len(source["elements"]):
        return None
    seen, prepared, groups = set(), [], {}
    characters = 0
    for row in rows:
        if not row:
            return None
        elements = tuple(row)
        for element in elements:
            eid = element.get("id")
            if eid in seen or eid not in originals or element != originals[eid]:
                return None
            if not all(_finite_positive(element.get(key)) for key in ("width", "height")):
                return None
            if not all(math.isfinite(element.get(key, 0)) for key in ("x", "y", "rotation")):
                return None
            if element["type"] == "text":
                if not _finite_positive(element.get("font_size")):
                    return None
                characters += len(str(element.get("content", "")))
            if element.get("group_id"):
                groups.setdefault(element["group_id"], set()).add(len(prepared))
            seen.add(eid)
        single = elements[0]
        flexible = (
            len(elements) == 1
            and single["type"] == "text"
            and not single.get("group_id")
            and not single.get("rotation", 0)
            and bool(str(single.get("content", "")).strip())
        )
        prepared.append(_Row(elements, union(list(elements)), flexible))
    if (
        len(seen) > MAX_ELEMENTS
        or characters > MAX_CHARACTERS
        or not any(row.flexible for row in prepared)
    ):
        return None
    # A source group cannot be split across rows or partially passed to this
    # helper. Geometry supplied by a candidate is never treated as source evidence.
    if any(len(indices) != 1 for indices in groups.values()) or any(
        element.get("group_id") in groups and element["id"] not in seen
        for element in source["elements"]
    ):
        return None
    # Row order is supplied by source inference, not sorted by mutable IDs.
    if any(b.box.y < a.box.y - _EPS for a, b in zip(prepared, prepared[1:], strict=False)):
        return None
    return tuple(prepared)


def _transform(row, scale, x, y):
    result = []
    for original in row.elements:
        element = deepcopy(original)
        element["x"] = x + (original["x"] - row.box.x) * scale
        element["y"] = y + (original["y"] - row.box.y) * scale
        for field in _FIELDS:
            if field in original:
                element[field] = original[field] * scale
        result.append(element)
    return result


def _text_fits(element, measurer, floor):
    if element["type"] != "text":
        return True
    if element["font_size"] < floor - 1e-9:
        return False
    measured = measurer.measure(element, element["font_size"], element["width"])
    return (
        measured.width <= element["width"] + _EPS
        and measured.height <= element["height"] + _EPS
        and (element.get("max_lines") is None or measured.lines <= element["max_lines"])
    )


def _at_scale(rows, source_box, cell, scale, floor, measurer):
    packed, heights, scales, wrapped = [], [], [], []
    for row in rows:
        if row.flexible:
            original = row.elements[0]
            element = deepcopy(original)
            element["font_size"] = original["font_size"] * scale
            align = original.get("text_align", "left")
            fraction = (
                original["x"]
                + original["width"]
                * (0.5 if align == "center" else 1.0 if align in {"right", "end"} else 0.0)
                - source_box.x
            ) / source_box.width
            anchor = cell.x + max(0.0, min(1.0, fraction)) * cell.width
            available = (
                2 * min(anchor - cell.x, cell.x + cell.width - anchor)
                if align == "center"
                else anchor - cell.x
                if align in {"right", "end"}
                else cell.x + cell.width - anchor
            )
            width = min(original["width"] * scale, available)
            if width <= _EPS:
                return None
            measured = measurer.measure(element, element["font_size"], width)
            element["width"] = width
            element["height"] = max(original["height"] * scale, measured.height)
            element["x"] = (
                anchor - width / 2
                if align == "center"
                else anchor - width
                if align in {"right", "end"}
                else anchor
            )
            element["y"] = 0.0
            if not _text_fits(element, measurer, floor):
                return None
            if measured.lines > str(original.get("content", "")).count("\n") + 1:
                wrapped.append(element["id"])
            packed.append([element])
            heights.append(element["height"])
            scales.append(scale)
        else:
            row_scale = min(scale, cell.width / row.box.width)
            centre = (
                cell.x
                + ((row.box.x + row.box.width / 2 - source_box.x) / source_box.width) * cell.width
            )
            x = max(
                cell.x,
                min(
                    centre - row.box.width * row_scale / 2,
                    cell.x + cell.width - row.box.width * row_scale,
                ),
            )
            elements = _transform(row, row_scale, x, 0.0)
            if any(not _text_fits(element, measurer, floor) for element in elements):
                return None
            packed.append(elements)
            heights.append(row.box.height * row_scale)
            scales.append(row_scale)
    gaps = [
        max(0.0, b.box.y - a.box.y - a.box.height) * scale
        for a, b in zip(rows, rows[1:], strict=False)
    ]
    content = sum(heights)
    # Only inter-row whitespace may compress. Preserve at least half of each
    # positive authored gap; do not fit by collapsing connected labels together.
    if content + sum(gaps) * 0.5 > cell.height + _EPS:
        return None
    spare = max(0.0, cell.height - content)
    spread = min(spare, sum(gaps) * 2.5)
    y = cell.y + (spare - spread) / 2
    result = []
    for index, elements in enumerate(packed):
        for element in elements:
            element["y"] += y
        result.extend(elements)
        y += heights[index]
        if index < len(gaps) and sum(gaps):
            y += spread * gaps[index] / sum(gaps)
    if any(
        (box := bounds(element)).x < cell.x - _EPS
        or box.y < cell.y - _EPS
        or box.x + box.width > cell.x + cell.width + _EPS
        or box.y + box.height > cell.y + cell.height + _EPS
        for element in result
    ):
        return None
    return result, tuple(scales), tuple(wrapped)


def pack_information_region(
    source: dict,
    rows,
    cell: Box,
    *,
    preferred_scale: float,
    min_font_size: float | None = None,
) -> MediaTextPacking | None:
    """Fit immutable source rows with a bounded common copy-font budget.

    ``rows`` is a sequence of source-element sequences, for example
    ``[row.elements for row in info_rows]``. The caller owns source grouping,
    media placement, target bounds and the scale budget. Only ungrouped,
    unrotated singleton live text can change its frame aspect. Other complete
    rows retain relative positions, dimensions and every font/shape ratio.

    Exact installed document fonts and native wrapping come from the active
    resource context. Missing rendering dependencies decline this proposal.
    Input dictionaries and all non-geometric style fields remain unchanged.
    """
    if (
        not _finite_positive(preferred_scale)
        or not all(math.isfinite(value) for value in (cell.x, cell.y, cell.width, cell.height))
        or min(cell.width, cell.height) <= 0
    ):
        return None
    floor = 1.0 if min_font_size is None else min_font_size
    if not _finite_positive(floor):
        return None
    prepared = _prepare(source, rows)
    if prepared is None:
        return None
    # The row's labels must keep their prominence relative to the copy. A
    # readable calendar does not justify independently enlarged body/headline
    # type beside it. Complete rigid rows still use one similarity transform.
    rigid_limits = [cell.width / row.box.width for row in prepared if not row.flexible]
    if rigid_limits:
        preferred_scale = min(preferred_scale, min(rigid_limits))
    measurer = get_measurer()
    if measurer.backend != "skia":
        return None
    elements = [element for row in prepared for element in row.elements]
    source_box = union(elements)
    fonts = [element["font_size"] for element in elements if element["type"] == "text"]
    lower = max(floor / size for size in fonts)
    if preferred_scale < lower - 1e-9:
        return None

    def attempt(scale):
        return _at_scale(prepared, source_box, cell, scale, floor, measurer)

    packed, chosen = attempt(preferred_scale), preferred_scale
    if packed is None:
        # Native glyph hinting need not be mathematically scale-linear. Locate
        # a fitting sampled interval before the bounded local refinement.
        upper = preferred_scale
        for index in range(1, SEARCH_STEPS + 1):
            scale = preferred_scale - (preferred_scale - lower) * index / SEARCH_STEPS
            fit = attempt(scale)
            if fit is not None:
                packed, chosen = fit, scale
                for _ in range(REFINE_STEPS):
                    middle = (chosen + upper) / 2
                    fit = attempt(middle)
                    if fit is not None:
                        packed, chosen = fit, middle
                    else:
                        upper = middle
                break
            upper = scale
    if packed is None:
        return None
    result, scales, wrapped = packed
    changed = {element["id"]: element for element in result}
    ordered = tuple(
        changed[element["id"]] for element in source["elements"] if element["id"] in changed
    )
    exceptions = tuple(
        "text-frame:" + row.elements[0]["id"] + ": information-region reflow"
        for row in prepared
        if row.flexible
    )
    return MediaTextPacking(ordered, chosen, scales, wrapped, exceptions)
