"""Bounded proposals derived from authored rows instead of inferred role order.

Source rows and media/information regions retain a shared internal geometry/font
scale: rules, icons, text and their authored offsets move together. Only an
ungrouped singleton text frame may widen to accommodate immutable tracking.
These proposals supplement the other families; they do not select a winner.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

from engine.resources import get_resources

from .appearance import unit_visible_box
from .geometry import Box, bounds, intersection, union
from .grouped import GroupedState
from .model import Unit, build_context
from .text import get_measurer

MAX_STATES = 10
_EPS = 0.015


@dataclass(frozen=True)
class _Band:
    elements: tuple[dict, ...]
    box: Box
    affinity: Box


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


def _tracking_frame(element, cell):
    """Widen a singleton about its authored text anchor; never change its font."""
    if _text_fits(element):
        return False
    if element.get("rotation", 0) or element.get("group_id"):
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
    if maximum <= element["width"] + _EPS:
        return None
    # Tracking is frozen by the document contract. Its advances can therefore
    # require more than the similarity-scaled frame, especially when shrinking.
    original_width = element["width"]
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


def _column(rows, scale, cell, gap_factor, source_width):
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
        result.extend(elements)
        if i < len(rows) - 1:
            gap = max(0, rows[i + 1].box.y - row.box.y - row.box.height) * gap_factor
            y += (row.box.height + gap) * scale
    return result, exceptions


def _packed(source, target, context, sections, cells, gap_factor, name, topology):
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
            _column(rows, scale, cell, gap_factor, context.width)
            for rows, cell in zip(sections, cells, strict=True)
        ]
        if any(part is None for part in parts):
            continue
        elements = [e for part in parts for e in part[0]]
        exceptions = [e for part in parts for e in part[1]]
        state = _state(source, target, context, name, topology, elements, exceptions)
        if state is not None:
            return state
    return None


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
        gap = min(width, height) * 0.055
        column_width = (cell.width - gap) / 2
        cells = [
            Box(cell.x, cell.y, column_width, cell.height),
            Box(cell.x + column_width + gap, cell.y, column_width, cell.height),
        ]
        cuts = sorted(
            range(2, len(rows) - 1),
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
