"""Bounded composition search over preserved, stacked and split layouts."""

import math
from copy import deepcopy

from .geometry import Box, contain
from .model import DesignContext, Unit, build_context
from .text import get_measurer

_ORDER = {
    "logo": 0,
    "hero": 1,
    "headline": 2,
    "body": 3,
    "badge": 4,
    "action": 5,
    "support": 6,
    "legal": 7,
}


def _place(unit: Unit, box: Box, font_scale: float) -> list[dict]:
    sx, sy = box.width / unit.box.width, box.height / unit.box.height
    # Rotating after different horizontal/vertical scales changes a group's
    # geometry. Move rotated labels and their plates with one shared scale.
    if any(element.get("rotation", 0) for element in unit.elements):
        scale = min(sx, sy)
        width, height = unit.box.width * scale, unit.box.height * scale
        box = Box(box.x + (box.width - width) / 2, box.y + (box.height - height) / 2, width, height)
        sx = sy = scale
    result = []
    for source in unit.elements:
        element = deepcopy(source)
        element["x"] = box.x + (source["x"] - unit.box.x) * sx
        element["y"] = box.y + (source["y"] - unit.box.y) * sy
        element["width"] = max(0.001, source["width"] * sx)
        element["height"] = max(0.001, source["height"] * sy)
        if source["type"] == "image" and source.get("fit") != "cover":
            scale = min(sx, sy)
            width, height = source["width"] * scale, source["height"] * scale
            element["x"] += (element["width"] - width) / 2
            element["y"] += (element["height"] - height) / 2
            element["width"], element["height"] = width, height
        if source["type"] == "text":
            preferred = source["font_size"] * font_scale
            if len(unit.elements) > 1:
                preferred = max(preferred, min(element["height"] / 1.2, box.height * 0.5))
            element["font_size"] = get_measurer().fit(
                element, element["width"], element["height"], preferred
            )
        if "corner_radius" in source:
            element["corner_radius"] = source["corner_radius"] * min(sx, sy)
        if "stroke_width" in source:
            element["stroke_width"] = source["stroke_width"] * min(sx, sy)
        result.append(element)
    return result


def _base(context: DesignContext, target: dict) -> list[dict]:
    width, height = target["width"], target["height"]
    elements = []
    for original in context.backgrounds:
        element = deepcopy(original)
        element.update(x=0.0, y=0.0, width=float(width), height=float(height))
        elements.append(element)
    scale = min(width / context.width, height / context.height)
    for original in context.decorations:
        element = deepcopy(original)
        element["width"] *= scale
        element["height"] *= scale
        element["x"] = original["x"] / context.width * width
        element["y"] = original["y"] / context.height * height
        contain(element, width, height)
        elements.append(element)
    return elements


def _finish(source: dict, target: dict, elements: list[dict]) -> dict:
    by_id = {element["id"]: element for element in elements}
    ordered = []
    for original in source["elements"]:
        element = by_id[original["id"]]
        contain(element, target["width"], target["height"])
        if element["type"] == "text":
            element["font_size"] = get_measurer().fit(
                element, element["width"], element["height"], element["font_size"]
            )
        ordered.append(element)
    return {"canvas": dict(target), "elements": ordered}


def _preferred_height(unit: Unit, width: float, height: float, font_scale: float) -> float:
    if unit.role == "logo":
        return min(height * 0.12, width / (unit.box.width / unit.box.height))
    if unit.role == "hero":
        return min(height * 0.42, width / (unit.box.width / unit.box.height))
    if not unit.text:
        return min(height * 0.09, width / (unit.box.width / unit.box.height))
    measured = max(
        get_measurer().measure(e, max(1.0, e["font_size"] * font_scale), width).height
        for e in unit.text
    )
    if len(unit.elements) > 1:
        measured *= 1.8
    return max(2.0, measured)


def _flow(units: list[Unit], box: Box, font_scale: float, gap: float) -> list[dict]:
    """Allocate vertical space by measured content, retaining all units."""
    if not units:
        return []
    heights = [_preferred_height(u, box.width, box.height, font_scale) for u in units]
    available = max(0.1, box.height - gap * (len(units) - 1))
    factor = min(1.0, available / max(0.1, sum(heights)))
    heights = [height * factor for height in heights]
    extra = max(0.0, available - sum(heights))
    spacing = gap + extra / max(1, len(units) + 1)
    y, elements = box.y + extra / max(1, len(units) + 1), []
    for unit, height in zip(units, heights, strict=True):
        unit_width = box.width
        if unit.role == "logo":
            unit_width = min(box.width * 0.65, height * unit.box.width / unit.box.height)
        elif not unit.text:
            unit_width = min(box.width, height * unit.box.width / unit.box.height)
        elements.extend(
            _place(
                unit, Box(box.x + (box.width - unit_width) / 2, y, unit_width, height), font_scale
            )
        )
        y += height + spacing
    return elements


def _row(units: list[Unit], box: Box, font_scale: float, gap: float) -> list[dict]:
    if not units:
        return []
    weights = [max(1.0, math.sqrt(u.box.width / u.box.height)) for u in units]
    available = max(0.1, box.width - gap * (len(units) - 1))
    x, elements = box.x, []
    for unit, weight in zip(units, weights, strict=True):
        width = available * weight / sum(weights)
        elements.extend(_place(unit, Box(x, box.y, width, box.height), font_scale))
        x += width + gap
    return elements


def _media(units: list[Unit], box: Box, font_scale: float, gap: float) -> list[dict]:
    """Give the main image the space left after the brand lockup is placed."""
    if not units:
        return []
    logos = [unit for unit in units if unit.role == "logo"]
    heroes = [unit for unit in units if unit.role != "logo"]
    if not heroes:
        return _flow(logos, box, font_scale, gap)
    elements = []
    logo_height = min(box.height * 0.18, box.width * 0.25) if logos else 0.0
    if logos:
        elements.extend(_row(logos, Box(box.x, box.y, box.width, logo_height), font_scale, gap))
    top = box.y + logo_height + (gap if logos else 0)
    available = max(0.1, box.height - logo_height - (gap if logos else 0))
    elements.extend(_row(heroes, Box(box.x, top, box.width, available), font_scale, gap))
    return elements


def _preserved(source: dict, context: DesignContext, target: dict, stretch: bool) -> dict:
    width, height = target["width"], target["height"]
    sx, sy = width / context.width, height / context.height
    scale = min(sx, sy)
    elements = _base(context, target)
    for unit in context.units:
        if stretch:
            box = Box(unit.box.x * sx, unit.box.y * sy, unit.box.width * sx, unit.box.height * sy)
        else:
            box = Box(
                unit.box.x * scale + (width - context.width * scale) / 2,
                unit.box.y * scale + (height - context.height * scale) / 2,
                unit.box.width * scale,
                unit.box.height * scale,
            )
        elements.extend(_place(unit, box, min(math.sqrt(sx * sy), scale * 1.3)))
    return _finish(source, target, elements)


def _stacked(source: dict, context: DesignContext, target: dict, typography: float) -> dict:
    width, height = target["width"], target["height"]
    pad, gap = min(width, height) * 0.045, min(width, height) * 0.018
    inner = Box(pad, pad, width - 2 * pad, height - 2 * pad)
    scale = math.sqrt(width * height / (context.width * context.height)) * typography
    units = sorted(context.units, key=lambda u: (_ORDER[u.role], u.box.y, u.box.x))
    badges = [u for u in units if u.role == "badge"]
    rest = [u for u in units if u.role != "badge"]
    elements = _base(context, target)
    if len(badges) > 1:
        row_height = min(inner.height * 0.08, inner.width * 0.075)
        elements.extend(
            _row(
                badges,
                Box(inner.x, inner.y + inner.height - row_height, inner.width, row_height),
                scale,
                gap,
            )
        )
        inner = Box(inner.x, inner.y, inner.width, inner.height - row_height - gap)
    else:
        rest.extend(badges)
        rest.sort(key=lambda u: (_ORDER[u.role], u.box.y, u.box.x))
    elements.extend(_flow(rest, inner, scale, gap))
    return _finish(source, target, elements)


def _split(
    source: dict,
    context: DesignContext,
    target: dict,
    media_share: float,
    sidebar: bool,
    typography: float,
) -> dict:
    width, height = target["width"], target["height"]
    pad, gap = min(width, height) * 0.04, min(width, height) * 0.025
    inner_width, inner_height = width - 2 * pad, height - 2 * pad
    scale = math.sqrt(width * height / (context.width * context.height)) * typography
    media = [u for u in context.units if u.role in {"logo", "hero"}]
    copy = [u for u in context.units if u.role in {"headline", "body"}]
    badges = [u for u in context.units if u.role == "badge"]
    actions = [u for u in context.units if u.role in {"action", "support"}]
    legal = [u for u in context.units if u.role == "legal"]
    footer_height = min(inner_height * 0.13, max(4.0, height * 0.045)) if legal else 0.0
    content_height = inner_height - footer_height - (gap if legal else 0)
    elements = _base(context, target)
    media_width = inner_width * media_share if media else 0
    media.sort(key=lambda u: (_ORDER[u.role], u.box.y))
    elements.extend(_media(media, Box(pad, pad, media_width, content_height), scale, gap))
    content_x = pad + media_width + (gap if media else 0)
    content_width = width - pad - content_x
    if sidebar and actions + badges:
        side_width = content_width * 0.27
        elements.extend(
            _flow(
                badges + actions,
                Box(width - pad - side_width, pad, side_width, content_height),
                scale,
                gap,
            )
        )
        content_width -= side_width + gap
        elements.extend(_flow(copy, Box(content_x, pad, content_width, content_height), scale, gap))
    else:
        row_height = min(content_height * 0.17, height * 0.14) if badges else 0
        action_height = min(content_height * 0.22, height * 0.18) if actions else 0
        copy_height = content_height - row_height - action_height
        copy_height -= gap * (bool(badges) + bool(actions))
        elements.extend(
            _flow(copy, Box(content_x, pad, content_width, max(0.1, copy_height)), scale, gap)
        )
        row_y = pad + copy_height + (gap if badges else 0)
        elements.extend(_row(badges, Box(content_x, row_y, content_width, row_height), scale, gap))
        action_y = pad + content_height - action_height
        elements.extend(
            _row(actions, Box(content_x, action_y, content_width, action_height), scale, gap)
        )
    elements.extend(
        _flow(legal, Box(pad, height - pad - footer_height, inner_width, footer_height), scale, 0)
    )
    return _finish(source, target, elements)


def generate_candidates(source: dict, target_canvas: dict) -> list[dict]:
    """Generate deterministic complete designs, preserving input order and values."""
    context = build_context(source)
    candidates = [
        _preserved(source, context, target_canvas, False),
        _preserved(source, context, target_canvas, True),
    ]
    candidates.extend(_stacked(source, context, target_canvas, t) for t in (0.9, 1.15))
    if target_canvas["width"] / target_canvas["height"] >= 1.35:
        for media_share in (0.18, 0.28):
            for sidebar in (False, True):
                candidates.append(_split(source, context, target_canvas, media_share, sidebar, 1.1))
    return candidates
