"""Shared local-coordinate placement for compound composition states."""

from copy import deepcopy

from .appearance import unit_visible_box
from .geometry import union
from .model import Unit
from .text import get_measurer


def _translate(elements: list[dict], dx: float, dy: float) -> None:
    for element in elements:
        element["x"] += dx
        element["y"] += dy


def _uniform(unit: Unit, scale: float, cx: float, top: float, assets, visible=None) -> list[dict]:
    visible = visible if visible is not None else unit_visible_box(unit.elements, assets)
    dx = cx - (visible.x + visible.width / 2) * scale
    dy = top - visible.y * scale
    elements = deepcopy(unit.elements)
    for element in elements:
        element["x"] = element["x"] * scale + dx
        element["y"] = element["y"] * scale + dy
        element["width"] *= scale
        element["height"] *= scale
        for field in ("font_size", "corner_radius", "stroke_width"):
            if field in element:
                element[field] *= scale
    return elements


def _brand(units: list[Unit], width: float, height: float, gap: float, assets) -> list[dict]:
    """Fit full frames while arranging their visible silhouettes on one axis."""
    visible = [unit_visible_box(unit.elements, assets) for unit in units]

    def arrange(scale):
        elements, y = [], 0.0
        for unit, box in zip(units, visible, strict=True):
            elements.extend(_uniform(unit, scale, 0, y, assets, visible=box))
            y += box.height * scale + gap
        return elements

    upper = min(
        width / max(unit.box.width for unit in units),
        height / max(1, sum(box.height for box in visible)),
    )
    lower = 0.0
    for _ in range(24):
        scale = (lower + upper) / 2
        frame = union(arrange(scale))
        if frame.width <= width and frame.height <= height:
            lower = scale
        else:
            upper = scale
    result = arrange(lower)
    frame = union(result)
    _translate(result, 0, -frame.y)
    return result


def _peer_rows(
    units: list[Unit],
    width: float,
    scale: float,
    gap: float,
    assets,
    alignment: str = "center",
    minimum_label_size: float = 0.0,
) -> list[dict]:
    """Balance ordered peers across rows without detaching labels or stretching plates."""
    if not units:
        return []
    widths = [unit.box.width * scale for unit in units]
    if max(widths) > width + 1e-6:
        return []
    costs, breaks = [0.0] + [float("inf")] * len(units), [0] * (len(units) + 1)
    for end in range(1, len(units) + 1):
        used = 0.0
        for start in range(end - 1, -1, -1):
            used += widths[start] + (gap if start < end - 1 else 0)
            if used > width + 1e-6:
                break
            cost = costs[start] + ((width - used) / max(width, 1)) ** 2
            if cost < costs[end]:
                costs[end], breaks[end] = cost, start
    slices, end = [], len(units)
    while end:
        start = breaks[end]
        slices.append((start, end))
        end = start
    elements, y = [], 0.0
    for start, end in reversed(slices):
        used = sum(widths[start:end]) + gap * (end - start - 1)
        x = (
            -width / 2
            if alignment == "start"
            else width / 2 - used
            if alignment == "end"
            else -used / 2
        )
        height = max(unit.box.height for unit in units[start:end]) * scale
        for index in range(start, end):
            unit = units[index]
            placed = _uniform(unit, scale, 0, y + (height - unit.box.height * scale) / 2, assets)
            for element in placed:
                if element["type"] == "text" and element["font_size"] < minimum_label_size:
                    measured = get_measurer().measure(element, minimum_label_size, element["width"])
                    if (
                        measured.width <= element["width"]
                        and measured.height <= element["height"]
                        and (
                            element.get("max_lines") is None
                            or measured.lines <= element["max_lines"]
                        )
                    ):
                        element["font_size"] = minimum_label_size
                    else:
                        element["font_size"] = get_measurer().fit(
                            element, element["width"], element["height"], minimum_label_size
                        )
            frame = union(placed)
            _translate(placed, x - frame.x, 0)
            elements.extend(placed)
            x += widths[index] + gap
        y += height + gap
    return elements


def _message(
    units: list[Unit],
    width: float,
    scale: float,
    short: float,
    assets,
    rows=(),
    alignment: str = "center",
    minimum_label_size: float = 0.0,
) -> list[dict]:
    elements, y = [], 0.0
    previous = None
    entries = [(unit_visible_box(unit.elements, assets), unit) for unit in units]
    entries.extend(
        (unit_visible_box([e for unit in row for e in unit.elements], assets), row) for row in rows
    )
    if rows:
        entries.sort(key=lambda item: item[0].y)
    for source_box, entry in entries:
        if previous is not None:
            source_gap = max(0, source_box.y - previous.y - previous.height)
            y += max(short * 0.02, min(short * 0.06, source_gap * scale))
        if isinstance(entry, (tuple, list)):
            placed = _peer_rows(
                entry, width, scale, short * 0.02, assets, alignment, minimum_label_size
            )
            if not placed:
                return []
            _translate(placed, 0, y)
        elif len(entry.elements) == 1 and entry.elements[0]["type"] == "text":
            unit = entry
            element = deepcopy(unit.elements[0])
            element["font_size"] = max(1.0, element["font_size"] * scale)
            element.update(x=-width / 2, y=y, width=width)
            measured = get_measurer().measure(element, element["font_size"], width)
            element["height"] = max(1.0, measured.height + 0.01)
            if element.get("max_lines") is not None and measured.lines > element["max_lines"]:
                return []
            placed = [element]
        else:
            placed = _uniform(entry, scale, 0, y, assets)
            box = unit_visible_box(placed, assets)
            if alignment != "center":
                x = -width / 2 if alignment == "start" else width / 2 - box.width
                _translate(placed, x - box.x, 0)
        elements.extend(placed)
        box = unit_visible_box(placed, assets)
        y = box.y + box.height
        previous = source_box
    return elements
