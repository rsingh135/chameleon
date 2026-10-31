"""Fit authored document columns to a new canvas using measured live copy.

A source-wide heading band and long column rule provide the layout evidence.
Text keeps one common font scale; only ungrouped live text frames can rewrap.
Explicit groups and artwork retain similarity geometry. The source's ordered
sections can flow into additional columns; no copy or artwork is synthesized.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from statistics import median

from .geometry import Box, bounds, union
from .grouped import GroupedState
from .model import build_context
from .source_guides import _transform
from .text import get_measurer

_EPS = 0.02


@dataclass(frozen=True)
class _Row:
    elements: tuple[dict, ...]
    source_left: float
    flexible: bool

    @property
    def box(self):
        return union(list(self.elements))


@dataclass(frozen=True)
class _SourceColumns:
    header: tuple[tuple[dict, ...], ...]
    columns: tuple[tuple[tuple[_Row, ...], ...], ...]
    separators: tuple[dict, ...]
    header_rule: dict
    context: object


def _text_fits(element):
    if element["type"] != "text":
        return True
    measured = get_measurer().measure(element, element["font_size"], element["width"])
    return (
        element["font_size"] >= 1
        and measured.width <= element["width"] + _EPS
        and measured.height <= element["height"] + _EPS
        and (element.get("max_lines") is None or measured.lines <= element["max_lines"])
    )


def _header_groups(units, width):
    texts = [u for u in units if u.text]
    groups, anchors = [], []
    for unit in sorted(texts, key=lambda u: (u.box.x, u.box.y)):
        first = unit.text[0]
        right = first.get("text_align") in {"right", "end"}
        anchor = first["x"] + (first["width"] if right else 0)
        index = next(
            (
                i
                for i, (x, align) in enumerate(anchors)
                if align == right and abs(x - anchor) < width * 0.06
            ),
            None,
        )
        if index is None:
            groups.append(list(unit.elements))
            anchors.append((anchor, right))
        else:
            groups[index].extend(unit.elements)
    for unit in (u for u in units if not u.text):
        if unit.box.width > width * 0.08 and unit.box.height > width * 0.08:
            groups.append(list(unit.elements))
            anchors.append((unit.box.x, False))
        elif groups:
            index = min(
                range(len(groups)),
                key=lambda i: abs(
                    (union(groups[i]).x + union(groups[i]).width / 2)
                    - (unit.box.x + unit.box.width / 2)
                ),
            )
            groups[index].extend(unit.elements)
        else:
            return ()
    return tuple(tuple(g) for g in sorted(groups, key=lambda g: union(g).x))


def _sections(units):
    texts = [u for u in units if u.text]
    sizes = [e["font_size"] for u in texts for e in u.text]
    if sum(len(unit.text) for unit in texts) < 4:
        return ()
    typical = median(sorted(sizes)[: max(1, len(sizes) // 2)])
    left = min(u.box.x for u in units)
    rows = [list(u.elements) for u in sorted(texts, key=lambda u: (u.box.y, u.box.x))]
    for unit in (u for u in units if not u.text):
        if min(unit.box.width, unit.box.height) > typical * 2.5:
            return ()
        choices = [
            (abs(unit.box.y + unit.box.height / 2 - union(row).y - union(row).height / 2), i)
            for i, row in enumerate(rows)
            if abs(unit.box.y + unit.box.height / 2 - union(row).y - union(row).height / 2)
            <= max(typical, unit.box.height)
            and abs(unit.box.x - union(row).x) <= typical * 5
        ]
        if not choices:
            return ()
        rows[min(choices)[1]].extend(unit.elements)
    sections, current = [], []
    for elements in rows:
        text = next(e for e in elements if e["type"] == "text")
        # Font prominence, rather than bold body labels, establishes a source
        # section boundary. Skills/tools and job labels remain owned by their
        # parent heading when the whole section moves to a different column.
        heading = text["font_size"] >= typical * 1.2
        # A section label keeps its first content row instead of becoming an
        # isolated column heading. Subsequent typographic subheads can flow.
        if (
            heading
            and current
            and (
                len(current) >= 2
                or sum(e["type"] == "text" for row in current for e in row.elements) > 1
            )
        ):
            sections.append(tuple(current))
            current = []
        flexible = (
            sum(e["type"] == "text" for e in elements) == 1
            and not any(e.get("group_id") for e in elements)
            and not text.get("rotation", 0)
        )
        current.append(_Row(tuple(elements), left, flexible))
    if current:
        sections.append(tuple(current))
    return tuple(sections)


def _infer(source):
    context = build_context(source)
    if context.decorations or not context.backgrounds or not 12 <= len(context.units) <= 128:
        return None
    rules = [
        u
        for u in context.units
        if len(u.elements) == 1
        and u.elements[0]["type"] == "shape"
        and not u.elements[0].get("group_id")
    ]
    vertical = [
        u
        for u in rules
        if u.box.height >= context.height * 0.35
        and u.box.width <= context.width * 0.015
        and context.width * 0.15 < u.box.x < context.width * 0.85
    ]
    horizontal = [
        u
        for u in rules
        if u.box.width >= context.width * 0.7
        and u.box.height <= context.height * 0.025
        and context.height * 0.08 < u.box.y < context.height * 0.4
    ]
    if not 1 <= len(vertical) <= 2 or not horizontal:
        return None
    header_rule = max(horizontal, key=lambda u: u.box.y)
    body_top = min(u.box.y for u in vertical)
    if header_rule.box.y + header_rule.box.height >= body_top:
        return None
    separators = sorted(vertical, key=lambda u: u.box.x)
    cuts = [u.box.x + u.box.width / 2 for u in separators]
    special = {e["id"] for u in [*separators, header_rule] for e in u.elements}
    header, columns = [], [[] for _ in range(len(cuts) + 1)]
    for unit in context.units:
        if all(e["id"] in special for e in unit.elements):
            continue
        if unit.box.y + unit.box.height <= body_top + _EPS:
            header.append(unit)
        elif unit.box.y >= body_top - context.height * 0.015:
            columns[bisect_right(cuts, unit.box.x + unit.box.width / 2)].append(unit)
        else:
            return None
    grouped = _header_groups(header, context.width)
    sections = tuple(_sections(column) for column in columns)
    if not grouped or not all(sections):
        return None
    return _SourceColumns(
        grouped,
        sections,
        tuple(u.elements[0] for u in separators),
        header_rule.elements[0],
        context,
    )


def _row_at(row, cell, scale):
    box = row.box
    moved = _transform(list(row.elements), scale, cell.x - row.source_left * scale, -box.y * scale)
    if row.flexible:
        text = next(e for e in moved if e["type"] == "text")
        text["width"] = cell.x + cell.width - text["x"]
        if text["width"] <= 0:
            return None
        measured = get_measurer().measure(text, text["font_size"], text["width"])
        text["height"] = max(measured.height + 0.02, text["font_size"])
    if any(not _text_fits(e) for e in moved):
        return None
    frame = union(moved)
    if frame.x < cell.x - _EPS or frame.x + frame.width > cell.x + cell.width + _EPS:
        return None
    return moved, frame.height


def _pack(sections, cell, scale, spread=False):
    prepared, minimum = [], 0.0
    typical = (
        median(
            e["font_size"]
            for section in sections
            for row in section
            for e in row.elements
            if e["type"] == "text"
        )
        * scale
    )
    for section_index, section in enumerate(sections):
        if section_index:
            minimum += typical * 1.1
        for row_index, row in enumerate(section):
            item = _row_at(row, cell, scale)
            if item is None:
                return None
            gap = 0.0
            if row_index:
                previous = section[row_index - 1].box
                gap = max(
                    typical * 0.12, max(0, row.box.y - previous.y - previous.height) * scale * 0.45
                )
            prepared.append((section_index, gap, item[0], item[1]))
            minimum += gap + item[1]
    if minimum > cell.height + _EPS:
        return None
    extra = max(0, cell.height - minimum)
    section_extra = (
        min(extra / (len(sections) - 1), typical * 2) if spread and len(sections) > 1 else 0
    )
    y, previous, output = cell.y, 0, []
    for index, gap, elements, height in prepared:
        if index != previous:
            y += typical * 1.1 + section_extra
        y += gap
        for element in elements:
            element["y"] += y
        output.extend(elements)
        y += height
        previous = index
    return output, minimum


def _partition(sections, cells, scale):
    if len(cells) == 1:
        packed = _pack(sections, cells[0], scale, spread=True)
        return packed[0] if packed else None
    best = None
    for cut in range(1, len(sections)):
        groups = [sections[:cut], sections[cut:]]
        packed = [_pack(group, cell, scale) for group, cell in zip(groups, cells, strict=True)]
        if any(p is None for p in packed):
            continue
        balance = abs(packed[0][1] - packed[1][1])
        if best is None or balance < best[0]:
            best = balance, groups
    if best is None:
        return None
    return [
        e
        for group, cell in zip(best[1], cells, strict=True)
        for e in _pack(group, cell, scale, spread=True)[0]
    ]


def _horizontal_rule(original, x, y, width, thickness):
    element = deepcopy(original)
    element.update(x=x, y=y, width=width, height=thickness)
    return element


def _vertical_rule(original, centre, top, height, scale):
    element = deepcopy(original)
    box = bounds(original)
    thickness = max(0.01, box.width * scale)
    if abs(float(original.get("rotation", 0))) % 180 == 90:
        element.update(
            x=centre - height / 2,
            y=top + height / 2 - thickness / 2,
            width=height,
            height=thickness,
        )
    elif not original.get("rotation", 0):
        element.update(x=centre - thickness / 2, y=top, width=thickness, height=height)
    else:
        return None
    return element


def _attempt(source, target, profile, count, header_fraction, scale):
    width, height = target["width"], target["height"]
    pad, gap = min(width, height) * 0.035, min(width, height) * 0.04
    header_height = height * header_fraction
    groups, required = [], []
    for group in profile.header:
        box = union(list(group))
        media = all(e["type"] != "text" for e in group)
        local_scale = min(scale, header_height / box.height) if media else scale
        if box.height * local_scale > header_height + _EPS:
            return None
        required.append(box.width * local_scale)
        groups.append((group, box, local_scale))
    if sum(required) + gap * (len(groups) - 1) > width - 2 * pad:
        return None
    elements, x = [], pad
    for index, ((group, box, local_scale), required_width) in enumerate(
        zip(groups, required, strict=True)
    ):
        if index == len(groups) - 1:
            x = width - pad - required_width
        elements.extend(
            _transform(
                list(group),
                local_scale,
                x - box.x * local_scale,
                pad + (header_height - box.height * local_scale) / 2 - box.y * local_scale,
            )
        )
        x += required_width + gap
    if any(not _text_fits(e) for e in elements):
        return None
    bar = bounds(profile.header_rule)
    thickness = max(0.5, bar.height * scale)
    rule_y = pad + header_height + gap * 0.5
    elements.append(_horizontal_rule(profile.header_rule, 0, rule_y, width, thickness))
    top, bottom = rule_y + thickness + gap, height - pad
    if bottom <= top:
        return None
    source_widths = [
        max(row.box.x + row.box.width for section in sections for row in section)
        - min(row.source_left for section in sections for row in section)
        for sections in profile.columns
    ]
    total = sum(source_widths)
    available = width - 2 * pad - gap * (count - 1)
    x, boundaries = pad, []
    per_source = count // len(profile.columns)
    for index, (sections, source_width) in enumerate(
        zip(profile.columns, source_widths, strict=True)
    ):
        column_width = available * source_width / total / per_source
        cells = [
            Box(x + n * (column_width + gap), top, column_width, bottom - top)
            for n in range(per_source)
        ]
        packed = _partition(sections, cells, scale)
        if packed is None:
            return None
        elements.extend(packed)
        x += per_source * (column_width + gap)
        if index < len(profile.columns) - 1:
            boundaries.append(x - gap / 2)
    for original, centre in zip(profile.separators, boundaries, strict=True):
        element = _vertical_rule(original, centre, top, bottom - top, scale)
        if element is None:
            return None
        elements.append(element)
    originals = {e["id"]: e for e in elements}
    for original in profile.context.backgrounds:
        originals[original["id"]] = dict(
            deepcopy(original), x=0.0, y=0.0, width=width, height=height
        )
    if set(originals) != {e["id"] for e in source["elements"]}:
        return None
    design = {"canvas": dict(target), "elements": [originals[e["id"]] for e in source["elements"]]}
    from engine.stage_one import strictly_in_bounds

    if not strictly_in_bounds(design):
        return None
    return design


def generate_fitted_column_states(source, target, graph=None):
    """Return bounded, source-derived column fits; decline unsupported designs."""
    del graph
    if any(
        not isinstance(canvas, dict)
        or any(
            isinstance(canvas.get(key), bool)
            or not isinstance(canvas.get(key), (int, float))
            or not math.isfinite(canvas[key])
            or canvas[key] <= 0
            for key in ("width", "height")
        )
        for canvas in (source.get("canvas"), target)
    ):
        return []
    if (target["width"] / target["height"]) / (
        source["canvas"]["width"] / source["canvas"]["height"]
    ) < 1.35:
        return []
    profile = _infer(source)
    if profile is None or get_measurer().backend != "skia":
        return []
    states = []
    for count, fraction in (
        (len(profile.columns), 0.18),
        (len(profile.columns), 0.24),
        (len(profile.columns) * 2, 0.24),
        (len(profile.columns) * 2, 0.30),
    ):
        if count > 4:
            continue
        lower, upper, design = (
            0.05,
            min(
                3.0,
                math.sqrt(
                    target["width"]
                    * target["height"]
                    / (profile.context.width * profile.context.height)
                )
                * 1.5,
            ),
            None,
        )
        for _ in range(18):
            scale = (lower + upper) / 2
            trial = _attempt(source, target, profile, count, fraction, scale)
            if trial is None:
                upper = scale
            else:
                lower, design = scale, trial
        if design is None:
            continue
        from engine.eval import check_contract

        if check_contract(source, design, target):
            continue
        name = f"fitted-columns/{count}-header-{fraction:g}"
        states.append(
            GroupedState(
                name,
                "fitted-source-columns",
                (),
                json.dumps(design, sort_keys=True, separators=(",", ":")),
                (
                    f"source-column-fit: measured live frames; common font scale {lower:.8g}",
                    f"source-column-count:{count}",
                ),
            )
        )
    return states
