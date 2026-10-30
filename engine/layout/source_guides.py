"""Bounded proposals derived from authored rows instead of inferred role order.

Source rows and media/information regions retain a shared internal geometry/font
scale: rules, icons, text and their authored offsets move together. Singleton
frames may widen for immutable tracking. Imported SVG single-line frames may
also correct a small native glyph-advance overflow within their panel cell.
These proposals supplement the other families; they do not select a winner.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from statistics import median

from engine.resources import get_resources

from .appearance import unit_visible_box
from .geometry import Box, bounds, intersection, union
from .grouped import GroupedState
from .media_text import pack_information_region
from .model import Unit, build_context
from .overlays import infer_source_overlays
from .reading_order import infer_reading_order
from .source_containers import _native_frame, infer_source_containers
from .text import get_measurer

MAX_STATES = 10
_EPS = 0.015


@dataclass(frozen=True)
class _Band:
    elements: tuple[dict, ...]
    box: Box
    affinity: Box


@dataclass(frozen=True)
class _Panel:
    container: Unit
    children: tuple[Unit, ...]
    rows: tuple[_Band, ...]
    stretch_ids: frozenset[str]
    flexible: bool
    bordered: bool


def _source_panels(context):
    """Reuse the same immutable enclosure evidence as diagnostics."""
    profile = infer_source_containers(context=context)
    panels = []
    for record in profile.containers:
        root = next(u for u in context.units if any(e["id"] == record.frame_id for e in u.elements))
        members = set(record.member_ids)
        children = tuple(u for u in context.units if all(e["id"] in members for e in u.elements))
        panels.append(
            _Panel(
                root,
                children,
                tuple(_unit_rows(context, children)),
                frozenset(record.frame_ids) - {record.frame_id},
                record.adaptive,
                "pixel-proven-flat-rectangular-vector-plate" in record.provenance,
            )
        )
    loose_ids = set(profile.loose_ids)
    loose = tuple(u for u in context.units if all(e["id"] in loose_ids for e in u.elements))
    return tuple(panels), loose


def _panel_dimensions(panel):
    if not panel.flexible:
        elements = [*panel.container.elements, *(e for u in panel.children for e in u.elements)]
        box = union(elements)
        return box.width, box.height
    if panel.bordered:
        # A rasterized border remains subject to the original image aspect
        # contract. Its children can use optical gaps; its frame keeps its
        # source height, rather than turning the border into a short fat card.
        return panel.container.box.width, panel.container.box.height
    widths = []
    for row in panel.rows:
        elements = [e for e in row.elements if e["id"] not in panel.stretch_ids]
        frame = union(elements) if elements else row.box
        source = panel.container.box
        widths.append(
            frame.width
            + (max(0, frame.x - source.x) + max(0, source.x + source.width - frame.x - frame.width))
            * 0.3
        )
    first, last = panel.rows[0], panel.rows[-1]
    overhang = max(0, first.affinity.y - first.box.y) + max(
        0, last.box.y + last.box.height - last.affinity.y - last.affinity.height
    )
    _, compact_gaps = _panel_gaps(panel)
    gaps = sum(compact_gaps)
    left, top, right, bottom = _panel_insets(panel)
    return max(widths), (
        sum(r.affinity.height for r in panel.rows) + overhang + gaps + top + bottom
    )


def _panel_insets(panel):
    frame = union(
        [e for unit in panel.children for e in unit.elements if e["id"] not in panel.stretch_ids]
    )
    source = panel.container.box
    # Native color-panel margins can compact along with authored row gaps.
    # Keep positive space around the foreground; never use edge-flush packing.
    # Raster borders retain all source insets and their original aspect limit.
    factor = 1 if panel.bordered else 0.3
    return (
        max(0, frame.x - source.x) * factor,
        max(0, frame.y - source.y) * factor,
        max(0, source.x + source.width - frame.x - frame.width) * factor,
        max(0, source.y + source.height - frame.y - frame.height) * factor,
    )


def _panel_gaps(panel):
    """Compact source whitespace without introducing text-frame/plate overlap.

    Only already-proven following native background frames qualify. Authored
    frame overlaps and dense plain-text rows retain their existing behavior;
    neither collision diagnostics nor actual ink masks are discounted.
    """
    authored, minimum = [], []
    for before, after in zip(panel.rows, panel.rows[1:], strict=False):
        gap = max(0, after.affinity.y - before.affinity.y - before.affinity.height)
        floor = gap * 0.3
        for plate in after.elements:
            if plate["id"] not in panel.stretch_ids or plate["type"] != "shape":
                continue
            plate_top = bounds(plate).y
            for previous in before.elements:
                if previous["type"] != "text":
                    continue
                frame = bounds(previous)
                if frame.y + frame.height > plate_top + _EPS:
                    continue
                clearance = (
                    frame.y
                    + frame.height
                    - before.affinity.y
                    - before.affinity.height
                    + after.affinity.y
                    - plate_top
                )
                floor = max(floor, clearance + 0.02)
        authored.append(gap)
        minimum.append(max(0, floor))
    return authored, minimum


def _panel_min_width(panel, scale):
    """Reserve immutable fixed tracking without independently shrinking titles."""
    required = _panel_dimensions(panel)[0] * scale
    for unit in panel.children:
        for original in unit.text:
            if (
                original.get("max_lines") != 1
                or original.get("group_id")
                or original.get("letter_spacing", 0) * (1 - scale) <= 0
            ):
                continue
            changed = _transform([original], scale, 0, 0)[0]
            measured = get_measurer().measure(changed, changed["font_size"], 100000).width
            # Native rectangular containers may accommodate their original
            # lettering by changing frame width; the source insets stay scaled.
            padding = max(0, original["x"] - panel.container.box.x) + max(
                0,
                panel.container.box.x
                + panel.container.box.width
                - original["x"]
                - original["width"],
            )
            if not panel.bordered:
                padding = 0
            required = max(required, measured + padding * scale + 0.04)
    return required


def _panel_place(panel, cell, scale):
    if not panel.flexible:
        elements = [*panel.container.elements, *(e for u in panel.children for e in u.elements)]
        box = union(elements)
        moved = _transform(
            elements,
            scale,
            cell.x + (cell.width - box.width * scale) / 2 - box.x * scale,
            cell.y + (cell.height - box.height * scale) / 2 - box.y * scale,
        )
        frame = Box(
            cell.x + (cell.width - box.width * scale) / 2,
            cell.y + (cell.height - box.height * scale) / 2,
            box.width * scale,
            box.height * scale,
        )
        exceptions = []
        for element in moved:
            if element["type"] == "text" and not _text_fits(element):
                if _tracking_frame(element, frame, allow_svg_group=True) is None:
                    return None
                exceptions.append("tracking-frame:" + element["id"])
        return moved, exceptions
    root = panel.container.elements[0]
    source = panel.container.box
    background = _transform([root], scale, 0, 0)[0]
    background.update(x=cell.x, y=cell.y, width=cell.width, height=cell.height)
    result = [background]
    exceptions = ["container-frame:" + root["id"] + ": source panel"]
    rows = panel.rows
    left_inset, top_inset, right_inset, bottom_inset = _panel_insets(panel)
    inner = Box(
        cell.x + left_inset * scale,
        cell.y + top_inset * scale,
        cell.width - (left_inset + right_inset) * scale,
        cell.height - (top_inset + bottom_inset) * scale,
    )
    top = max(0, rows[0].affinity.y - rows[0].box.y) * scale
    bottom = (
        max(
            0, rows[-1].box.y + rows[-1].box.height - rows[-1].affinity.y - rows[-1].affinity.height
        )
        * scale
    )
    gaps, minimum_gaps = _panel_gaps(panel)
    content = sum(r.affinity.height for r in rows) * scale + top + bottom
    if content > inner.height + _EPS:
        return None
    free = inner.height - content
    minimum = sum(minimum_gaps) * scale
    if free + _EPS < minimum:
        return None
    spread = max(minimum, min(free, sum(gaps) * scale * 3))
    gap_sizes = [
        base * scale + (spread - minimum) * (authored / sum(gaps) if sum(gaps) else 1 / len(gaps))
        for authored, base in zip(gaps, minimum_gaps, strict=True)
    ]
    remainder = (free - spread) / 2
    if not panel.bordered:
        # A source title at the top of a color panel stays top-affiliated when
        # that panel becomes taller than a neighboring stack of two panels.
        remainder = min(remainder, max(0, rows[0].box.y - source.y) * scale)
    y = inner.y + top + remainder
    for index, row in enumerate(rows):
        foreground = [e for e in row.elements if e["id"] not in panel.stretch_ids]
        frame = union(foreground) if foreground else row.box
        width = frame.width * scale
        factor = 1 if panel.bordered else 0.3
        row_left = max(0, frame.x - source.x) * factor * scale
        row_right = max(0, source.x + source.width - frame.x - frame.width) * factor * scale
        row_inner = Box(cell.x + row_left, inner.y, cell.width - row_left - row_right, inner.height)
        if width > row_inner.width + _EPS:
            return None
        centre = cell.x + (frame.x + frame.width / 2 - source.x) / source.width * cell.width
        alignment = {e.get("text_align", "left") for e in foreground if e["type"] == "text"}
        if alignment == {"left"}:
            preferred = cell.x + max(0, frame.x - source.x) * scale
        elif alignment <= {"right", "end"} and alignment:
            preferred = (
                cell.x
                + cell.width
                - max(0, source.x + source.width - frame.x - frame.width) * scale
                - width
            )
        else:
            preferred = centre - width / 2
        x = max(row_inner.x, min(preferred, row_inner.x + row_inner.width - width))
        moved = _transform(row.elements, scale, x - frame.x * scale, y - row.affinity.y * scale)
        for element, original in zip(moved, row.elements, strict=True):
            if element["id"] in panel.stretch_ids:
                left = max(0, (original["x"] - source.x) / source.width)
                right = min(1, (original["x"] + original["width"] - source.x) / source.width)
                element["x"] = cell.x + left * cell.width
                element["width"] = (right - left) * cell.width
                exceptions.append("container-frame:" + element["id"] + ": source section")
            elif element["type"] == "text" and not _text_fits(element):
                if _tracking_frame(element, row_inner, allow_svg_group=True) is None:
                    return None
                exceptions.append("tracking-frame:" + element["id"])
        result.extend(moved)
        y += row.affinity.height * scale
        if index < len(gap_sizes):
            y += gap_sizes[index]
    # Inset native plates must continue to support the original source ink.
    # Their authored root-relative width is only an initial proposal: compact
    # parent margins cannot clip copy against a rounded white background.
    originals = {e["id"]: e for unit in panel.children for e in unit.elements}
    changed = {e["id"]: e for e in result}
    for eid in panel.stretch_ids:
        original = originals[eid]
        old_box = bounds(original)
        supported = []
        for child in originals.values():
            if child["type"] != "text" or child.get("z_index", 0) <= original.get("z_index", 0):
                continue
            ink = unit_visible_box([child])
            if intersection(ink, old_box) > ink.width * ink.height * 0.02:
                supported.append(child)
        if not supported:
            continue
        old_ink = unit_visible_box(supported)
        new_ink = unit_visible_box([changed[e["id"]] for e in supported])
        left_pad = max(0, old_ink.x - old_box.x) * scale * 0.3
        right_pad = max(0, old_box.x + old_box.width - old_ink.x - old_ink.width) * scale * 0.3
        element = changed[eid]
        left = min(element["x"], new_ink.x - left_pad)
        right = max(element["x"] + element["width"], new_ink.x + new_ink.width + right_pad)
        element["x"] = max(cell.x, left)
        element["width"] = min(cell.x + cell.width, right) - element["x"]
    if any(
        bounds(e).x < cell.x - _EPS
        or bounds(e).x + bounds(e).width > cell.x + cell.width + _EPS
        or bounds(e).y < cell.y - _EPS
        or bounds(e).y + bounds(e).height > cell.y + cell.height + _EPS
        for e in result
    ):
        return None
    return result, exceptions


def _panel_states(source, target, context):
    panels, loose = _source_panels(context)
    if not panels:
        return []
    top = [u for u in loose if u.box.y + u.box.height / 2 < context.height / 2]
    bottom = [u for u in loose if u not in top]
    top_box = union([e for u in top for e in u.elements]) if top else None
    bottom_box = union([e for u in bottom for e in u.elements]) if bottom else None
    extra = (top_box.height if top_box else 0) + (bottom_box.height if bottom_box else 0)
    sizes = [_panel_dimensions(panel) for panel in panels]
    width, height = float(target["width"]), float(target["height"])
    plans = []
    for columns in range(1, min(4, len(panels)) + 1):
        for cuts in combinations(range(1, len(panels)), columns - 1):
            edges = (0, *cuts, len(panels))
            parts = [panels[a:b] for a, b in zip(edges, edges[1:], strict=False)]
            ranges = [range(a, b) for a, b in zip(edges, edges[1:], strict=False)]
            for margin in (0.012, 0.0) if panels[0].flexible else (0.018,):
                inset = min(width, height) * margin
                gap = max(2.0, min(width, height) * margin)
                required_widths = [max(sizes[i][0] for i in group) for group in ranges]
                required_heights = [sum(sizes[i][1] for i in group) for group in ranges]
                available_width = width - 2 * inset - gap * (columns - 1)
                available_height = (
                    height - 2 * inset - gap * (max(len(group) for group in parts) - 1)
                )
                if top:
                    available_height -= gap
                if bottom:
                    available_height -= gap
                scale = min(
                    available_width / sum(required_widths),
                    available_height / (extra + max(required_heights)),
                )
                for _ in range(6):
                    actual_widths = [
                        max(_panel_min_width(panels[i], scale) for i in group) for group in ranges
                    ]
                    actual_heights = [
                        sum(
                            max(
                                sizes[i][1] * scale,
                                actual_widths[column]
                                / panels[i].container.box.width
                                * panels[i].container.box.height
                                / 1.049,
                            )
                            if panels[i].bordered
                            else sizes[i][1] * scale
                            for i in group
                        )
                        for column, group in enumerate(ranges)
                    ]
                    shrink = min(
                        available_width / sum(actual_widths),
                        available_height / (extra * scale + max(actual_heights)),
                    )
                    if shrink >= 1 - 1e-8:
                        break
                    scale *= shrink * 0.999
                required_widths = actual_widths
                if scale <= 0:
                    continue
                plans.append((scale, columns, cuts, margin, inset, gap, parts, required_widths))
    # Retain diverse contiguous partitions rather than many trivial duplicates.
    plans.sort(key=lambda p: (-p[0], p[1], p[2], p[3]))
    chosen = []
    keys = set()
    for plan in plans:
        key = (plan[1], plan[2])
        if key not in keys:
            chosen.append(plan)
            keys.add(key)
        if len(chosen) == 4:
            break
    states = []
    for scale, columns, _cuts, _margin, inset, gap, parts, required_widths in chosen:
        body_top = inset + (top_box.height * scale + gap if top_box else 0)
        body_bottom = height - inset - (bottom_box.height * scale + gap if bottom_box else 0)
        available_width = width - 2 * inset - gap * (columns - 1)
        elements = []
        exceptions = []
        bordered = all(panel.bordered for panel in panels)
        x = inset + (available_width - sum(required_widths)) / 2 if bordered else inset
        valid = True
        for group, required in zip(parts, required_widths, strict=True):
            column_width = (
                required if bordered else available_width * required / sum(required_widths)
            )
            heights = [
                max(
                    _panel_dimensions(panel)[1] * scale,
                    column_width / panel.container.box.width * panel.container.box.height / 1.049,
                )
                if panel.bordered
                else _panel_dimensions(panel)[1] * scale
                for panel in group
            ]
            spare = body_bottom - body_top - sum(heights) - gap * (len(group) - 1)
            if spare < -_EPS:
                valid = False
                break
            y = body_top if group[0].flexible and not group[0].bordered else body_top + spare / 2
            for panel, panel_height in zip(group, heights, strict=True):
                allocated = panel_height + (
                    spare / len(group) if panel.flexible and not panel.bordered else 0
                )
                moved = _panel_place(panel, Box(x, y, column_width, allocated), scale)
                if moved is None:
                    valid = False
                    break
                elements.extend(moved[0])
                exceptions.extend(moved[1])
                y += allocated + gap
            if not valid:
                break
            x += column_width + gap
        if not valid:
            continue
        for units, box, y in [
            (top, top_box, inset),
            (bottom, bottom_box, height - inset - (bottom_box.height * scale if bottom_box else 0)),
        ]:
            if not units:
                continue
            centre = box.x + box.width / 2
            x = max(
                inset,
                min(
                    centre / context.width * width - box.width * scale / 2,
                    width - inset - box.width * scale,
                ),
            )
            elements.extend(
                _transform(
                    [e for u in units for e in u.elements],
                    scale,
                    x - box.x * scale,
                    y - box.y * scale,
                )
            )
        for element in elements:
            if element["type"] == "text" and not _text_fits(element):
                if (
                    _tracking_frame(
                        element, Box(inset, 0, width - 2 * inset, height), allow_svg_group=True
                    )
                    is None
                ):
                    valid = False
                    break
                exceptions.append("tracking-frame:" + element["id"])
        if not valid:
            continue
        name = "panels-" + str(columns) + "-" + "-".join(str(len(group)) for group in parts)
        state = _state(source, target, context, name, "source-panel-columns", elements, exceptions)
        if state is not None:
            states.append(state)
    return states


def _transform(elements, scale, dx, dy):
    result = []
    for original in elements:
        element = deepcopy(original)
        element["x"] = original["x"] * scale + dx
        element["y"] = original["y"] * scale + dy
        for field in ("width", "height", "font_size", "corner_radius", "stroke_width"):
            if field in original:
                element[field] = original[field] * scale
        result.append(element)
    return result


def _rows(context):
    """Connected y-overlap components, retaining explicit units as indivisible."""
    units = sorted(context.units, key=lambda unit: (unit.box.y, unit.box.x))
    rows = []
    for unit in units:
        support = unit_visible_box(unit.elements)
        # Imported text often retains a deliberately generous frame below its
        # final line. Empty frame padding must not connect the next unrelated
        # row, while all actual frames still participate in rigid placement.
        overlap = (
            (
                min(support.y + support.height, rows[-1].affinity.y + rows[-1].affinity.height)
                - max(support.y, rows[-1].affinity.y)
            )
            if rows
            else 0
        )
        if rows and overlap > min(support.height, rows[-1].affinity.height) * 0.08:
            elements = rows[-1].elements + tuple(unit.elements)
            rows[-1] = _Band(elements, union(elements), unit_visible_box(list(elements)))
        else:
            rows.append(_Band(tuple(unit.elements), unit.box, support))
    return rows


def _owned_units(context):
    """Retain original low-layer plates and their contained foreground together.

    A color panel can overlap a photograph without making the whole page one
    immutable object. Ownership is directional: the panel owns later elements
    contained inside it, never the earlier photograph behind it.
    """
    units = context.units
    area = context.width * context.height
    candidates = []
    for index, unit in enumerate(units):
        art = all(
            e["type"] == "shape"
            or (e["type"] == "image" and e.get("source_art", {}).get("kind") == "svg-nontext")
            for e in unit.elements
        )
        overlay = len(unit.elements) == 1 and _translucent_overlay(unit.elements[0])
        if (
            not art
            or overlay
            or unit.box.width < context.width * 0.35
            or unit.box.height < context.height * 0.035
            or unit.box.width * unit.box.height < area * 0.04
        ):
            continue
        z = max(e.get("z_index", 0) for e in unit.elements)
        if any(
            other.text
            and min(e.get("z_index", 0) for e in other.elements) > z
            and intersection(unit.box, other.box) >= other.box.width * other.box.height * 0.95
            for other in units
        ):
            candidates.append(index)
    owner = {}
    for index, unit in enumerate(units):
        z = min(e.get("z_index", 0) for e in unit.elements)
        containers = [
            i
            for i in candidates
            if i != index
            and max(e.get("z_index", 0) for e in units[i].elements) < z
            and intersection(units[i].box, unit.box) >= unit.box.width * unit.box.height * 0.95
        ]
        if containers:
            owner[index] = min(containers, key=lambda i: units[i].box.width * units[i].box.height)
    grouped = {}
    for index, unit in enumerate(units):
        root = index
        while root in owner:
            root = owner[root]
        grouped.setdefault(root, []).extend(unit.elements)
    plate_roots = set(candidates) - set(owner)
    plates = [
        Unit(elements, union(elements), "plate")
        for root, elements in grouped.items()
        if root in plate_roots
    ]
    remaining = [
        Unit(elements, union(elements))
        for root, elements in grouped.items()
        if root not in plate_roots
    ]

    # Strongly overlapping original art/photo images form one visual composite,
    # such as a photograph and its shaped colored surround. Do not pull unrelated
    # text into that composite just because the image frame has a padded corner.
    image_units = [
        u
        for u in remaining
        if any(e["type"] == "image" for e in u.elements)
        and u.box.width * u.box.height >= area * 0.04
    ]
    parents = list(range(len(image_units)))

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for i, first in enumerate(image_units):
        for j in range(i):
            second = image_units[j]
            smaller = min(first.box.width * first.box.height, second.box.width * second.box.height)
            if intersection(first.box, second.box) >= smaller * 0.55:
                parents[root(i)] = root(j)
    composites = {}
    for index, unit in enumerate(image_units):
        composites.setdefault(root(index), []).extend(unit.elements)
    media = [Unit(elements, union(elements), "media") for elements in composites.values()]
    claimed = {id(u) for u in image_units}
    ordinary = [u for u in remaining if id(u) not in claimed]
    # A dark label readable on the original bright photo cannot be moved onto
    # a dark page background. Retain that original photographic support, while
    # compatible white message copy can form a separate information region.
    backgrounds = [e for e in context.backgrounds if e["type"] == "shape"]
    if backgrounds:
        background = _luminance(backgrounds[-1].get("fill_color", "#ffffff"))
        independent = []
        for unit in ordinary:
            bad_support = (
                unit.text
                and all(e["type"] == "text" for e in unit.elements)
                and any(
                    _contrast(_luminance(e.get("color", "#000000")), background) < 3
                    for e in unit.text
                )
            )
            owners = [
                m
                for m in media
                if intersection(m.box, unit.box) >= unit.box.width * unit.box.height * 0.95
            ]
            if bad_support and owners:
                owner = min(owners, key=lambda m: m.box.width * m.box.height)
                owner.elements.extend(unit.elements)
                owner.box = union(owner.elements)
            else:
                independent.append(unit)
        ordinary = independent
    return ordinary, plates, media


def _luminance(color):
    from PIL import ImageColor

    try:
        rgb = ImageColor.getrgb(color)[:3]
    except (ValueError, TypeError):
        return 0.5
    channels = [c / 255 for c in rgb]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def _contrast(first, second):
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


@lru_cache(maxsize=128)
def _alpha_overlay(path, modified, size):
    from PIL import Image

    del modified, size
    try:
        with Image.open(path) as image:
            histogram = image.convert("RGBA").getchannel("A").histogram()
        visible = sum(histogram[1:])
        return visible > sum(histogram) * 0.3 and histogram[255] / max(1, visible) < 0.65
    except (OSError, ValueError):
        return False


def _translucent_overlay(element):
    if element["type"] != "image" or element.get("source_art", {}).get("kind") != "svg-nontext":
        return False
    root = get_resources().assets_dir.resolve()
    path = (root / element.get("asset_id", "")).resolve()
    if not path.is_relative_to(root):
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    return _alpha_overlay(str(path), stat.st_mtime_ns, stat.st_size)


def _spread(rows, scale, cell, source_width):
    """Adapt only inter-region whitespace; keep every complete region rigid."""
    content = sum(row.box.height for row in rows) * scale
    if content > cell.height + _EPS:
        return None
    gaps = [max(0, b.box.y - a.box.y - a.box.height) for a, b in zip(rows, rows[1:], strict=False)]
    if gaps and not sum(gaps):
        gaps = [1.0] * len(gaps)
    available = cell.height - content
    # Expand meaningful inter-region whitespace, with a bound that prevents a
    # tight message from becoming four detached lines across a large canvas.
    free = min(available, sum(gaps) * scale * 2.5)
    y, result, exceptions = cell.y + (available - free) / 2, [], []
    for index, row in enumerate(rows):
        centre = cell.x + (row.box.x + row.box.width / 2) / source_width * cell.width
        x = max(
            cell.x,
            min(centre - row.box.width * scale / 2, cell.x + cell.width - row.box.width * scale),
        )
        elements = _transform(row.elements, scale, x - row.box.x * scale, y - row.box.y * scale)
        if len(elements) == 1 and elements[0]["type"] == "text":
            expanded = _tracking_frame(elements[0], cell)
            if expanded is None:
                return None
            if expanded:
                exceptions.append("text-frame:" + elements[0]["id"] + ": immutable tracking")
        elif any(not _text_fits(e) for e in elements):
            return None
        result.extend(elements)
        y += row.box.height * scale
        if index < len(gaps):
            y += free * gaps[index] / sum(gaps)
    return result, exceptions


def _unit_rows(context, units):
    from types import SimpleNamespace

    return _rows(SimpleNamespace(units=units))


def _distributed(source, target, context, rows, cell):
    scale = min(
        cell.width / max(row.box.width for row in rows),
        cell.height * 0.9 / sum(row.box.height for row in rows),
    )
    packed = _spread(rows, scale, cell, context.width)
    return (
        _state(
            source,
            target,
            context,
            "bands-distributed",
            "source-bands-vertical",
            packed[0],
            packed[1],
        )
        if packed
        else None
    )


def _square_fade_stacks(source, target, context, photo, rows):
    """Keep the complete photo and enlarge its source-supported lower caption.

    A portrait photo cannot become a landscape strip without cropping. Its
    existing fade can instead support a wider caption over the original matte.
    This proposal keeps the complete media transform, source paint and one
    common information scale; it never adds a plate or changes image fitting.
    """
    width, height = float(target["width"]), float(target["height"])
    if not 0.95 <= width / height <= 1.1 or photo.box.height < photo.box.width * 1.3:
        return []
    originals = {e["id"]: e for e in source["elements"]}
    photo_ids = {
        e["id"]
        for e in photo.elements
        if e["type"] == "image"
        and e.get("source_art", {}).get("kind") == "masked-photo"
        and e.get("role") not in {"product", "hero"}
    }
    if not photo_ids:
        return []
    foreground = [e for row in rows for e in row.elements]
    info = union(foreground)
    media_ids = {e["id"] for e in photo.elements}
    overlays = [
        overlay
        for overlay in infer_source_overlays(source)
        if overlay.underlay_id in photo_ids
        and overlay.overlay_id in media_ids
        and intersection(info, bounds(originals[overlay.overlay_id]))
        >= info.width * info.height * 0.98
    ]
    backgrounds = [e for e in context.backgrounds if _native_frame(e)]
    if not overlays or not backgrounds:
        return []
    matte = max(backgrounds, key=lambda e: e.get("z_index", 0))
    luminance = _luminance(matte["fill_color"])
    # Reuse the tested opaque-paint proof. Unknown or translucent text cannot
    # establish readable support outside the photographic frame.
    for row in rows:
        for text in (e for e in row.elements if e["type"] == "text"):
            opaque = _native_frame(
                {
                    "type": "shape",
                    "shape_type": "rectangle",
                    "fill_color": text["color"],
                    "opacity": text.get("opacity", 1),
                }
            )
            local_support = any(
                _native_frame(plate)
                and plate.get("z_index", 0) < text.get("z_index", 0)
                and intersection(bounds(text), bounds(plate))
                >= text["width"] * text["height"] * 0.98
                and _contrast(_luminance(plate["fill_color"]), _luminance(text["color"])) >= 4.5
                for plate in row.elements
            )
            if not opaque or (
                _contrast(luminance, _luminance(text["color"])) < 4.5 and not local_support
            ):
                return []
    try:
        from engine.eval.metrics import MIN_FONT_FRACTION, MIN_FONT_PX
    except ImportError:
        return []
    padding = min(width, height) * 0.035
    media_scale = min(height / photo.box.height, (width - 2 * padding) / photo.box.width)
    moved = _transform(
        photo.elements,
        media_scale,
        (width - photo.box.width * media_scale) / 2 - photo.box.x * media_scale,
        -photo.box.y * media_scale,
    )
    by_id = {e["id"]: e for e in moved}
    overlay = max(overlays, key=lambda a: originals[a.overlay_id].get("z_index", 0))
    fade = by_id[overlay.overlay_id]
    # Keep a source-derived inset into the fade; do not move copy into the
    # unsupported upper photo simply to gain more typography space.
    support_top = (
        fade["y"] + max(0, info.y - originals[overlay.overlay_id]["y"]) * media_scale * 0.5
    )
    area_scale = math.sqrt(width * height / (context.width * context.height))
    floor = max(MIN_FONT_PX, MIN_FONT_FRACTION * min(width, height))
    states = []
    for multiplier in (1.0, 1.15):
        scale = min(
            area_scale * multiplier,
            (width - 2 * padding) / info.width,
            (height - 2 * padding) / info.height,
        )
        cell = Box(
            (width - info.width * scale) / 2,
            height - padding - info.height * scale,
            info.width * scale,
            info.height * scale,
        )
        if cell.y < support_top - _EPS:
            continue
        packed = pack_information_region(
            source, [row.elements for row in rows], cell, preferred_scale=scale, min_font_size=floor
        )
        if packed is None:
            continue
        state = _state(
            source,
            target,
            context,
            f"media-fade-stack-{multiplier:g}",
            "source-media-stacked-fade",
            moved + list(packed.elements),
            [
                *packed.exceptions,
                "source-photo-fade: lower caption support",
                f"region-scale:media:{media_scale:.8g}",
                f"region-scale:information:{packed.font_scale:.8g}",
            ],
        )
        if state is not None:
            states.append(state)
    return states


def _photo_regions(source, target, context):
    """Source-derived media/info topology with rigid photo and plate owners."""
    ordinary, plates, media = _owned_units(context)
    # Competing main photos require the existing campaign/ownership families.
    # Small source art remains in the information column, with its source row.
    if len(media) != 1 or not ordinary + plates:
        return []
    photo = media[0]
    # Vector-only posters use the row family, not a guessed photographic hero.
    if all(e.get("source_art", {}).get("kind") == "svg-nontext" for e in photo.elements):
        return []
    width, height = float(target["width"]), float(target["height"])
    aspect_change = (width / height) / (context.width / context.height)
    if 0.8 <= aspect_change <= 1.25:
        return []
    states = []
    info_rows = _unit_rows(context, ordinary + plates)
    if width / height >= 0.95:
        padding = min(width, height) * 0.035
        photo_scale = min(height / photo.box.height, width * 0.48 / photo.box.width)
        photo_width = photo.box.width * photo_scale
        photo_elements = _transform(
            photo.elements,
            photo_scale,
            width - photo_width - photo.box.x * photo_scale,
            (height - photo.box.height * photo_scale) / 2 - photo.box.y * photo_scale,
        )
        # Original media side determines the new side; only that topology is tried.
        on_right = photo.box.x + photo.box.width / 2 >= context.width / 2
        for fraction in (0.56, 0.62):
            info_width = width * fraction
            if not plates:
                # Without an opaque owned panel, the target photo cannot sit
                # behind light message copy: reserve genuinely clear support.
                info_width = min(info_width, width - photo_width - padding)
            if info_width <= 0:
                continue
            cell = Box(0 if on_right else width - info_width, padding, info_width, height - padding)
            scale = min(
                info_width / max(context.width, max(r.box.width for r in info_rows)),
                cell.height * 0.86 / sum(r.box.height for r in info_rows),
            )
            packed = _spread(info_rows, scale, cell, context.width)
            if packed is None:
                continue
            moved = deepcopy(photo_elements)
            if not on_right:
                for e in moved:
                    e["x"] -= width - photo_width
            state = _state(
                source,
                target,
                context,
                f"media-columns-{fraction:g}",
                "source-media-columns",
                moved + packed[0],
                [
                    *packed[1],
                    f"region-scale:media:{photo_scale:.8g}",
                    f"region-scale:information:{scale:.8g}",
                ],
            )
            if state is not None:
                states.append(state)
        # A source photo may own live captions before the main copy. Infer that
        # order once from the source, then offer the corresponding left/right
        # topology with the whole available media height. Clear information
        # copy can wrap; existing multi-object rows keep rigid geometry.
        caption_ids = {e["id"] for e in photo.elements if e["type"] == "text"}
        info_ids = {e["id"] for row in info_rows for e in row.elements if e["type"] == "text"}
        if caption_ids and info_ids and not plates:
            profile = infer_reading_order(source)
            positions = {eid: index for index, eid in enumerate(profile.sequence)}
            captions = [positions[eid] for eid in caption_ids if eid in positions]
            information = [positions[eid] for eid in info_ids if eid in positions]
            leading = (
                len(captions) == len(caption_ids)
                and len(information) == len(info_ids)
                and max(captions) < min(information)
            )
            trailing = (
                len(captions) == len(caption_ids)
                and len(information) == len(info_ids)
                and min(captions) > max(information)
            )
            if leading or trailing:
                media_scale = min(height / photo.box.height, width * 0.565 / photo.box.width)
                media_width = photo.box.width * media_scale
                clear_width = width - media_width - 2 * padding
                if clear_width >= width * 0.3:
                    cell = Box(
                        media_width + padding if leading else padding,
                        padding,
                        clear_width,
                        height - 2 * padding,
                    )
                    preferred = min(
                        width / context.width,
                        cell.height * 0.86 / sum(r.box.height for r in info_rows),
                    )
                    packed = pack_information_region(
                        source,
                        [row.elements for row in info_rows],
                        cell,
                        preferred_scale=preferred,
                    )
                    if packed is not None:
                        moved = _transform(
                            photo.elements,
                            media_scale,
                            (0 if leading else width - media_width) - photo.box.x * media_scale,
                            -photo.box.y * media_scale,
                        )
                        state = _state(
                            source,
                            target,
                            context,
                            "media-columns-"
                            + ("leading" if leading else "trailing")
                            + "-fullheight",
                            "source-media-columns",
                            moved + list(packed.elements),
                            [
                                *packed.exceptions,
                                f"region-scale:media:{media_scale:.8g}",
                                f"region-scale:information:{packed.font_scale:.8g}",
                            ],
                        )
                        if state is not None:
                            states.append(state)
                if leading:
                    states.extend(_square_fade_stacks(source, target, context, photo, info_rows))
    elif height / width >= 1.2 and len(plates) == 1 and ordinary:
        plate = plates[0]
        if (
            plate.box.width >= context.width * 0.75
            and plate.box.y + plate.box.height >= context.height * 0.95
        ):
            plate_scale = width / context.width
            plate_h = plate.box.height * plate_scale
            plate_y = height - plate_h
            top_elements = [e for unit in ordinary for e in unit.elements]
            top = union(top_elements)
            padding = min(width, height) * 0.035
            overlap = max(
                0,
                min(photo.box.y + photo.box.height, plate.box.y + plate.box.height)
                - max(photo.box.y, plate.box.y),
            )
            visible_height = max(photo.box.height - overlap, photo.box.height * 0.25)
            photo_scale = min(
                width / photo.box.width,
                max(0.01, height - plate_h - top.height * plate_scale - padding) / visible_height,
            )
            photo_y = plate_y - visible_height * photo_scale
            photo_x = (width - photo.box.width * photo_scale) * (
                1 if photo.box.x >= context.width * 0.25 else 0.5
            )
            elements = _transform(
                photo.elements,
                photo_scale,
                photo_x - photo.box.x * photo_scale,
                photo_y - photo.box.y * photo_scale,
            )
            elements += _transform(
                plate.elements, plate_scale, 0, plate_y - plate.box.y * plate_scale
            )
            elements += _transform(top_elements, plate_scale, 0, padding - top.y * plate_scale)
            state = _state(
                source,
                target,
                context,
                "media-stacked",
                "source-media-stacked",
                elements,
                [
                    f"region-scale:media:{photo_scale:.8g}",
                    f"region-scale:information:{plate_scale:.8g}",
                ],
            )
            if state is not None:
                states.append(state)
    return states


def _text_fits(element):
    if element["type"] != "text":
        return True
    if element["font_size"] < 1:
        return False
    measured = get_measurer().measure(element, element["font_size"], element["width"])
    return (
        measured.width <= element["width"] + _EPS
        and measured.height <= element["height"] + _EPS
        and (element.get("max_lines") is None or measured.lines <= element["max_lines"])
    )


def _tracking_frame(element, cell, *, allow_svg_group=False):
    """Widen a frame about its authored text anchor; never change its font.

    A grouped SVG line has an imported calibrated frame, not a wrapping choice.
    Native hinted advances can differ slightly between renderers at the current
    scaled size. Only panel callers may repair that frame, below the existing
    rigid-group deformation bound and inside the assigned row/panel cell.
    """
    if _text_fits(element):
        return False
    grouped = bool(element.get("group_id"))
    if element.get("rotation", 0):
        return None
    if grouped and (
        not allow_svg_group
        or not get_measurer().native_document
        or not element.get("svg_font_family")
        or element.get("max_lines") != 1
        or "\n" in element.get("content", "")
        or element.get("text_align", "left") not in {"left", "center", "right", "end"}
        or (element.get("native_text_effects") or {}).get("gradient") is not None
    ):
        return None
    left, right = cell.x, cell.x + cell.width
    align = element.get("text_align", "left")
    if align == "center":
        anchor = element["x"] + element["width"] / 2
        maximum = 2 * min(anchor - left, right - anchor)
    elif align in {"right", "end"}:
        anchor = element["x"] + element["width"]
        maximum = anchor - left
    else:
        anchor = element["x"]
        maximum = right - anchor
    if grouped:
        # Width and alignment-origin changes remain below the unchanged 5%
        # source-group guard. Do not grow a frame to hide a genuine reflow or
        # squeeze unrelated text into its neighbor's row/panel.
        maximum = min(maximum, element["width"] * 1.049)
    if maximum <= element["width"] + _EPS:
        return None
    # Tracking is frozen by the document contract. Its advances can therefore
    # require more than the similarity-scaled frame, especially when shrinking.
    original_width = element["width"]
    if grouped:
        measured = get_measurer().measure(element, element["font_size"], math.inf)
        preferred = max(original_width, measured.width + 0.02)
        if preferred > maximum or measured.wrapped != element.get("content", ""):
            return None
        repaired = dict(element, width=preferred)
        repaired["x"] = (
            anchor - preferred / 2
            if align == "center"
            else anchor - preferred
            if align in {"right", "end"}
            else anchor
        )
        if not _text_fits(repaired):
            return None
        element.update(x=repaired["x"], width=preferred)
        return True
    measured = get_measurer().measure(element, element["font_size"], maximum)
    preferred = max(original_width, min(maximum, measured.width + 0.02))
    widths = sorted(
        {
            preferred,
            maximum,
            *(original_width + (maximum - original_width) * i / 12 for i in range(1, 12)),
        }
    )
    for width in widths:
        element["width"] = width
        element["x"] = (
            anchor - width / 2
            if align == "center"
            else anchor - width
            if align in {"right", "end"}
            else anchor
        )
        if _text_fits(element):
            return True
    return None


def _design(source, target, background_ids, foreground):
    changed = {e["id"]: e for e in foreground}
    ordered = []
    for original in source["elements"]:
        if original["id"] in background_ids:
            element = deepcopy(original)
            element.update(
                x=0.0, y=0.0, width=float(target["width"]), height=float(target["height"])
            )
        else:
            element = changed[original["id"]]
        ordered.append(element)
    return {"canvas": dict(target), "elements": ordered}


def _valid(elements, target):
    for element in elements:
        if not all(math.isfinite(float(element[field])) for field in ("x", "y", "width", "height")):
            return False
        box = bounds(element)
        if (
            min(box.width, box.height) <= 0
            or box.x < -_EPS
            or box.y < -_EPS
            or box.x + box.width > target["width"] + _EPS
            or box.y + box.height > target["height"] + _EPS
            or not _text_fits(element)
        ):
            return False
    return True


def _state(source, target, context, name, topology, elements, exceptions=()):
    if not _valid(elements, target):
        return None
    design = _design(source, target, {e["id"] for e in context.backgrounds}, elements)
    return GroupedState(
        "source-guides/" + name,
        topology,
        (),
        json.dumps(design, sort_keys=True, separators=(",", ":")),
        tuple(exceptions),
    )


def _source_fit(source, target, context, foreground):
    width, height = float(target["width"]), float(target["height"])
    source_width, source_height = context.width, context.height
    scale = min(width / source_width, height / source_height)
    # Retain the authored canvas centre even if a foreground frame extends
    # beyond that canvas. Reduce the whole arrangement rather than clamping leaves.
    for element in foreground:
        box = bounds(element)
        horizontal = max(abs(box.x - source_width / 2), abs(box.x + box.width - source_width / 2))
        vertical = max(abs(box.y - source_height / 2), abs(box.y + box.height - source_height / 2))
        if horizontal:
            scale = min(scale, width / (2 * horizontal))
        if vertical:
            scale = min(scale, height / (2 * vertical))
    elements = _transform(
        foreground, scale, (width - source_width * scale) / 2, (height - source_height * scale) / 2
    )
    singletons = {
        u.elements[0]["id"]
        for u in context.units
        if len(u.elements) == 1 and not u.elements[0].get("group_id")
    }
    exceptions = []
    for element in elements:
        if element["type"] == "text" and not _text_fits(element):
            if element["id"] not in singletons:
                return None
            expanded = _tracking_frame(element, Box(0, 0, width, height))
            if expanded is None:
                return None
            if expanded:
                exceptions.append("text-frame:" + element["id"] + ": immutable tracking")
    return _state(source, target, context, "source-fit", "source-similarity", elements, exceptions)


def _extent(rows, gap_factor):
    return sum(row.box.height for row in rows) + sum(
        max(0, after.box.y - before.box.y - before.box.height) * gap_factor
        for before, after in zip(rows, rows[1:], strict=False)
    )


def _normalize_column_rows(rows, context):
    """Remove only empty horizontal native text-frame space, preserving paint.

    The source's current wrapped strings, exact face advances and alignment
    anchor must remain identical. Grouped/rotated/justified and frame-gradient
    text stays untouched. This helper is used only by synthetic band columns.
    """
    measurer = get_measurer()
    if not measurer.native_document or measurer.backend != "skia":
        return rows, {}
    singletons = {unit.elements[0]["id"] for unit in context.units if len(unit.elements) == 1}

    def gradient(value):
        if isinstance(value, dict):
            return any(
                (key == "gradient" and child is not None)
                or (key in {"type", "kind"} and isinstance(child, str) and "gradient" in child)
                or gradient(child)
                for key, child in value.items()
            )
        if isinstance(value, (tuple, list)):
            return any(gradient(child) for child in value)
        return False

    normalized, signatures = [], {}
    for row in rows:
        # A multi-label calendar/control row is an authored rigid cohort. Only
        # one independent text leaf (possibly flanked by simple rules) qualifies.
        if sum(e["type"] == "text" for e in row.elements) != 1:
            normalized.append(row)
            continue
        elements = []
        for original in row.elements:
            element = original
            align = original.get("text_align", "left")
            if (
                original["type"] == "text"
                and original["id"] in singletons
                and not original.get("group_id")
                and not original.get("rotation", 0)
                and align in {"left", "center", "right"}
                and not gradient(original)
            ):
                measured = measurer.measure(original, original["font_size"], original["width"])
                width = measured.width + max(0.05, original["font_size"] * 0.01)
                if 0 < width < original["width"] - _EPS:
                    candidate = deepcopy(original)
                    candidate["width"] = width
                    candidate["x"] += (original["width"] - width) * {
                        "left": 0,
                        "center": 0.5,
                        "right": 1,
                    }[align]
                    checked = measurer.measure(candidate, candidate["font_size"], width)
                    if (
                        checked.wrapped == measured.wrapped
                        and abs(checked.width - measured.width) <= 1e-7
                    ):
                        element = candidate
                        signatures[original["id"]] = measured.wrapped
            elements.append(element)
        # Occupied pixels do not move: only the source frame envelope changes.
        normalized.append(_Band(tuple(elements), union(elements), row.affinity))
    return normalized, signatures


def _column(rows, scale, cell, gap_factor, source_width, preserved_lines=None):
    total = _extent(rows, gap_factor) * scale
    y = cell.y + (cell.height - total) / 2
    result, exceptions = [], []
    for i, row in enumerate(rows):
        # Source-normalized centres preserve centred headers and left/right
        # alignments while each column supplies the new containing width.
        centre = cell.x + (row.box.x + row.box.width / 2) / source_width * cell.width
        x = centre - row.box.width * scale / 2
        x = max(cell.x, min(x, cell.x + cell.width - row.box.width * scale))
        elements = _transform(row.elements, scale, x - row.box.x * scale, y - row.box.y * scale)
        if len(elements) == 1 and elements[0]["type"] == "text":
            expanded = _tracking_frame(elements[0], cell)
            if expanded is None:
                return None
            if expanded:
                exceptions.append("text-frame:" + elements[0]["id"] + ": immutable tracking")
        elif any(not _text_fits(e) for e in elements):
            return None
        for element in elements:
            expected = (preserved_lines or {}).get(element["id"])
            if (
                expected is not None
                and get_measurer().measure(element, element["font_size"], element["width"]).wrapped
                != expected
            ):
                return None
        result.extend(elements)
        if i < len(rows) - 1:
            gap = max(0, rows[i + 1].box.y - row.box.y - row.box.height) * gap_factor
            y += (row.box.height + gap) * scale
    return result, exceptions


def _packed(source, target, context, sections, cells, gap_factor, name, topology):
    preserved_lines = {}
    if topology in {"source-bands-columns", "source-bands-header-columns"}:
        normalized = []
        for index, rows in enumerate(sections):
            if topology == "source-bands-header-columns" and index == 0:
                normalized.append(rows)
                continue
            changed, signatures = _normalize_column_rows(rows, context)
            normalized.append(changed)
            preserved_lines.update(signatures)
        sections = normalized
    upper = min(
        min(
            cell.width / max(row.box.width for row in rows), cell.height / _extent(rows, gap_factor)
        )
        for rows, cell in zip(sections, cells, strict=True)
    )
    # Frozen tracking can impose a lower feasible scale, so do not assume a
    # conventional binary search with a feasible lower endpoint. Try larger
    # shared scales first; this fixed scan is deliberately bounded.
    for fraction in (1.0, 0.99, 0.97, 0.94, 0.90, 0.85, 0.78, 0.68, 0.55):
        scale = upper * fraction
        parts = [
            _column(rows, scale, cell, gap_factor, context.width, preserved_lines)
            for rows, cell in zip(sections, cells, strict=True)
        ]
        if any(part is None for part in parts):
            continue
        elements = [e for part in parts for e in part[0]]
        exceptions = [e for part in parts for e in part[1]] + [
            "text-frame:" + eid + ": native line advances" for eid in preserved_lines
        ]
        state = _state(source, target, context, name, topology, elements, exceptions)
        if state is not None:
            return state
    return None


def _balanced_columns(sections):
    """Synthetic columns need meaningful copy and comparable occupied content.

    This applies only to newly invented band columns, never authored panels,
    photo/information regions or another family's asymmetric source topology.
    Empty frame padding and source whitespace do not count as occupied height.
    """
    if not sections or any(
        not any(
            e["type"] == "text" and e.get("content", "").strip() for r in rows for e in r.elements
        )
        for rows in sections
    ):
        return False
    occupied = [sum(row.affinity.height for row in rows) for rows in sections]
    return min(occupied) >= max(occupied) * 0.45


def _source_cut_allowed(rows, cut, body_size):
    """Keep adjacent prominent title lines and their close dividers attached."""
    before, after = rows[cut - 1], rows[cut]

    def prominent(row):
        return (
            max((e.get("font_size", 0) for e in row.elements if e["type"] == "text"), default=0)
            >= body_size * 1.6
        )

    gap = max(0, after.affinity.y - before.affinity.y - before.affinity.height)
    if (
        prominent(before)
        and prominent(after)
        and gap <= min(before.affinity.height, after.affinity.height) * 0.5
    ):
        return False
    for index in (cut - 1, cut):
        row = rows[index]
        if (
            not all(e["type"] == "shape" for e in row.elements)
            or row.affinity.height > body_size * 0.35
            or row.affinity.width < row.affinity.height * 4
        ):
            continue
        previous = (
            max(0, row.affinity.y - rows[index - 1].affinity.y - rows[index - 1].affinity.height)
            if index > 0
            else math.inf
        )
        following = (
            max(0, rows[index + 1].affinity.y - row.affinity.y - row.affinity.height)
            if index + 1 < len(rows)
            else math.inf
        )
        if previous < following * 0.7 and cut == index:
            return False
        if following < previous * 0.7 and cut == index + 1:
            return False
    return True


def generate_source_guides(source: dict, target: dict, graph=None) -> list[GroupedState]:
    """Return at most ten independent source-derived candidate proposals.

    Optional ``graph`` is accepted for factory compatibility and is not changed.
    Rigid region owners allow photo-aware format changes. Separate band topology
    is limited to text/shape/vector rows with unambiguous occupied y gaps.
    """
    del graph
    if not source.get("elements") or any(
        not math.isfinite(float(target.get(k, 0))) or float(target.get(k, 0)) <= 0
        for k in ("width", "height")
    ):
        return []
    context = build_context(source)
    if min(context.width, context.height) <= 0:
        return []
    background_ids = {e["id"] for e in context.backgrounds}
    foreground = [e for e in source["elements"] if e["id"] not in background_ids]
    if not foreground:
        return []
    states = []
    canonical = _source_fit(source, target, context, foreground)
    if canonical is not None:
        states.append(canonical)
    if not context.decorations:
        states.extend(_photo_regions(source, target, context))
        states.extend(_panel_states(source, target, context))
    if context.decorations or any(
        e["type"] == "image" and e.get("source_art", {}).get("kind") != "svg-nontext"
        for e in foreground
    ):
        return states
    rows = _rows(context)
    if not 2 <= len(rows) <= 20:
        return states
    width, height = float(target["width"]), float(target["height"])
    inset = min(width, height) * 0.035
    cell = Box(inset, inset, width - 2 * inset, height - 2 * inset)
    for gap_factor in (1.0, 0.35, 0.08):
        state = _packed(
            source,
            target,
            context,
            [rows],
            [cell],
            gap_factor,
            f"bands-{gap_factor:g}",
            "source-bands-vertical",
        )
        if state is not None:
            states.append(state)
    distributed = _distributed(source, target, context, rows, cell)
    if distributed is not None:
        states.append(distributed)
    if len(rows) >= 4 and width / height >= 0.85:
        body_sizes = [e["font_size"] for e in foreground if e["type"] == "text"]
        body_size = median(body_sizes) if body_sizes else 1.0
        gap = min(width, height) * 0.055
        column_width = (cell.width - gap) / 2
        cells = [
            Box(cell.x, cell.y, column_width, cell.height),
            Box(cell.x + column_width + gap, cell.y, column_width, cell.height),
        ]
        cuts = sorted(
            (
                cut
                for cut in range(2, len(rows) - 1)
                if _balanced_columns([rows[:cut], rows[cut:]])
                and _source_cut_allowed(rows, cut, body_size)
            ),
            key=lambda i: (
                -(rows[i].affinity.y - rows[i - 1].affinity.y - rows[i - 1].affinity.height),
                abs(i - len(rows) / 2),
                i,
            ),
        )[:2]
        for cut in cuts:
            state = _packed(
                source,
                target,
                context,
                [rows[:cut], rows[cut:]],
                cells,
                0.35,
                f"columns-{cut}",
                "source-bands-columns",
            )
            if state is not None:
                states.append(state)
        # A wide introductory band can retain its spanning role above a source-
        # ordered two-column body. No source names or role guesses are involved.
        if len(rows) >= 6:
            prefix = 1
            remaining = rows[prefix:]
            cut = (len(remaining) + 1) // 2
            if not _balanced_columns([remaining[:cut], remaining[cut:]]) or not _source_cut_allowed(
                rows, prefix + cut, body_size
            ):
                unique = {}
                for state in states:
                    unique.setdefault(state.design_json, state)
                return list(unique.values())[:MAX_STATES]
            header_height = min(
                cell.height * 0.32, rows[0].box.height * cell.width / rows[0].box.width
            )
            header = Box(cell.x, cell.y, cell.width, header_height)
            body_y = cell.y + header_height + gap
            body_cells = [Box(c.x, body_y, c.width, cell.y + cell.height - body_y) for c in cells]
            state = _packed(
                source,
                target,
                context,
                [rows[:prefix], remaining[:cut], remaining[cut:]],
                [header, *body_cells],
                0.35,
                "header-columns",
                "source-bands-header-columns",
            )
            if state is not None:
                states.append(state)
    unique = {}
    for state in states:
        unique.setdefault(state.design_json, state)
    return list(unique.values())[:MAX_STATES]
