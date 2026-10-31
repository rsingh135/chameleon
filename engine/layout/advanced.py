"""Realize a campaign with ordered badge peers, corner accents and a footer.

Typography is fitted as a block; labels never leave their plates. Missing assets
keep their original references and render as the existing warned placeholders.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass

from .appearance import unit_visible_box
from .blocks import _brand, _message, _translate, _uniform
from .candidates import _base
from .geometry import intersection, union
from .model import build_context
from .text import get_measurer


@dataclass(frozen=True)
class AdvancedState:
    name: str
    topology: str
    cluster_axes: tuple[tuple[str, str], ...]
    design_json: str

    @property
    def design(self):
        return json.loads(self.design_json)


def _footer(context, legal, supports, target, scale, assets):
    width, height = target["width"], target["height"]
    short = min(width, height)
    pad, gap = short * 0.04, short * 0.025
    elements, right = [], width - pad
    occupied = []
    for unit in reversed(supports):
        local_scale = min(scale, short * 0.14 / unit.box.height, width * 0.18 / unit.box.width)
        placed = _uniform(unit, local_scale, 0, 0, assets)
        frame = union(placed)
        _translate(placed, right - frame.width - frame.x, height - pad - frame.height - frame.y)
        elements.extend(placed)
        occupied.append(union(placed))
        right -= frame.width + gap
    y = height - pad
    legal_elements = []
    for unit in reversed(legal):
        if len(unit.elements) != 1 or unit.elements[0]["type"] != "text":
            return None
        element = deepcopy(unit.elements[0])
        element["font_size"] = max(min(12.0, short * 0.11), element["font_size"] * scale)
        element.update(x=pad, width=right - pad)
        if element["width"] <= 0:
            return None
        measured = get_measurer().measure(element, element["font_size"], element["width"])
        element.update(y=y - measured.height, height=measured.height + 0.01)
        y = element["y"] - gap
        legal_elements.append(element)
    elements.extend(reversed(legal_elements))
    # Side icons may share the footer's empty horizontal space. Actual visible
    # conflicts are checked later rather than reserving their height globally.
    top = y + gap if legal_elements else height - pad
    return elements, top


def _trial(
    source,
    target,
    graph,
    brand,
    message,
    rows,
    legal,
    supports,
    accents,
    topology,
    copy_share,
    typography,
):
    context = build_context(source)
    width, height = target["width"], target["height"]
    short = min(width, height)
    pad, gap = short * 0.04, short * 0.04
    base = math.sqrt(width * height / (graph.width * graph.height))
    assets = graph.assets_dir
    footer = _footer(context, legal, supports, target, base, assets)
    if footer is None:
        return None
    footer_elements, footer_top = footer
    bottom = footer_top - (gap if footer_elements else 0)
    elements, accent_boxes = [], []
    for unit in accents:
        placed = _uniform(unit, typography, 0, 0, assets)
        box = union(placed)
        left = unit.box.x + unit.box.width / 2 < graph.width / 2
        _translate(placed, (pad if left else width - pad - box.width) - box.x, pad - box.y)
        elements.extend(placed)
        accent_boxes.append(union(placed))

    brand_units = [context.units[i] for i in brand.unit_indices]
    message_units = [context.units[i] for i in message.unit_indices]
    peer_units = [[context.units[i] for i in row.unit_indices] for row in rows]
    anchors = [
        e.alignment for e in graph.edges if e.cluster_id == message.id and e.code == "shared_axis"
    ]
    alignment = anchors[0] if anchors else "center"
    copy_width = (width - 2 * pad) * copy_share
    content = _message(
        message_units,
        copy_width,
        typography,
        short,
        assets,
        rows=peer_units,
        alignment=alignment,
        minimum_label_size=min(10.0, short * 0.11),
    )
    if not content:
        return None
    copy_frame = union(content)
    if copy_frame.width > copy_width + 0.1:
        return None

    def clear_top(x, w):
        return max(
            [pad]
            + [
                b.y + b.height + gap
                for b in accent_boxes
                if min(x + w, b.x + b.width) > max(x, b.x)
            ]
        )

    if topology == "columns":
        identity = _brand(
            brand_units,
            width * (1 - copy_share) * 0.94,
            (bottom - pad) * 0.96,
            short * 0.035,
            assets,
        )
        brand_frame, visible = union(identity), unit_visible_box(identity, assets)
        copy_visible = unit_visible_box(content, assets)
        total = visible.width + short * 0.10 + copy_visible.width
        left = (width - total) / 2
        # A corner promotion can occupy the outer whitespace beside a compact
        # campaign. Keep the campaign left of it when full frames still fit.
        right_accents = [b for b in accent_boxes if b.x >= width / 2]
        minimum_left = pad + visible.x - brand_frame.x
        maximum_left = (
            width
            - pad
            - copy_frame.width
            - visible.width
            - short * 0.10
            + copy_visible.x
            - copy_frame.x
        )
        if minimum_left > maximum_left:
            return None
        if right_accents:
            reserved = min(b.x - gap for b in right_accents) - total
            if reserved >= minimum_left:
                left = min(left, reserved)
        left = max(minimum_left, min(left, maximum_left))
        dx = left - visible.x
        brand_top = clear_top(brand_frame.x + dx, brand_frame.width)
        if brand_top >= bottom:
            return None
        if brand_top + brand_frame.height > bottom:
            identity = _brand(
                brand_units,
                width * (1 - copy_share) * 0.94,
                bottom - brand_top,
                short * 0.035,
                assets,
            )
            brand_frame, visible = union(identity), unit_visible_box(identity, assets)
            total = visible.width + short * 0.10 + copy_visible.width
            left = (width - total) / 2
            minimum_left = pad + visible.x - brand_frame.x
            maximum_left = (
                width
                - pad
                - copy_frame.width
                - visible.width
                - short * 0.10
                + copy_visible.x
                - copy_frame.x
            )
            if minimum_left > maximum_left:
                return None
            if right_accents:
                reserved = min(b.x - gap for b in right_accents) - total
                if reserved >= minimum_left:
                    left = min(left, reserved)
            left = max(minimum_left, min(left, maximum_left))
            dx = left - visible.x
        _translate(identity, dx, brand_top + (bottom - brand_top - brand_frame.height) / 2)
        copy_x = left + visible.width + short * 0.10 - copy_visible.x
        copy_top = clear_top(copy_visible.x + copy_x, copy_visible.width)
        if copy_top + copy_frame.height > bottom:
            return None
        _translate(
            content, copy_x, copy_top + (bottom - copy_top - copy_frame.height) / 2 - copy_frame.y
        )
    else:
        top = clear_top(pad, width - 2 * pad)
        available = bottom - top - gap - copy_frame.height
        if available < short * 0.38:
            return None
        identity = _brand(
            brand_units, width * 0.72, min(available, (bottom - top) * 0.52), short * 0.035, assets
        )
        brand_frame = union(identity)
        total = brand_frame.height + gap + copy_frame.height
        y = top + max(0, (bottom - top - total) / 2)
        _translate(identity, width / 2, y - brand_frame.y)
        _translate(content, width / 2, y + brand_frame.height + gap - copy_frame.y)
    foreground = elements + identity + content + footer_elements
    if min(union(identity).width, union(identity).height) < 1:
        return None
    # Check optically meaningful accent/footer conflicts before the renderer.
    for ornament in elements + footer_elements:
        a = unit_visible_box([ornament], assets)
        for other in identity + content:
            if intersection(a, unit_visible_box([other], assets)) > 0.5:
                return None
    elements = _base(context, target) + foreground
    frame = union(elements)
    if (
        frame.x < -1e-6
        or frame.y < -1e-6
        or frame.x + frame.width > width + 1e-6
        or frame.y + frame.height > height + 1e-6
    ):
        return None
    if any(e.get("font_size", 1) < 1 or min(e["width"], e["height"]) <= 0.001 for e in elements):
        return None
    by_id = {e["id"]: e for e in elements}
    if set(by_id) != set(graph.source_ids):
        return None
    return {"canvas": dict(target), "elements": [by_id[e["id"]] for e in source["elements"]]}


def generate_advanced_states(source, target, graph):
    """Fit one campaign's peer rows and accessories; never invent ownership."""
    brands = [c for c in graph.clusters if c.kind == "brand"]
    messages = [c for c in graph.clusters if c.kind == "message"]
    rows = [c for c in graph.clusters if c.kind == "badge_row"]
    if len(brands) != 1 or len(messages) != 1 or not rows:
        return ()
    brand, message = brands[0], messages[0]
    used = set(brand.unit_indices + message.unit_indices)
    used.update(i for row in rows for i in row.unit_indices)
    context = build_context(source)
    legal, supports, accents = [], [], []
    for index, unit in enumerate(context.units):
        if index in used:
            continue
        cx = (unit.box.x + unit.box.width / 2) / graph.width
        cy = (unit.box.y + unit.box.height / 2) / graph.height
        if unit.role == "legal":
            legal.append(unit)
        elif unit.role in {"support", "logo"} and cy >= 0.75:
            supports.append(unit)
        elif unit.role == "badge" and cy <= 0.25 and (cx <= 0.25 or cx >= 0.75):
            accents.append(unit)
        else:
            return ()
    base = math.sqrt(target["width"] * target["height"] / (graph.width * graph.height))
    source_fonts = [e["font_size"] for u in context.units if u.role != "legal" for e in u.text]
    minimum = 1 / min(source_fonts, default=1)
    states = []
    topologies = (
        ("columns", "stacked") if target["width"] / target["height"] >= 1.3 else ("stacked",)
    )
    for topology in topologies:
        shares = (0.68, 0.72) if topology == "columns" else (0.84, 0.94)
        for share in shares:

            def fit(scale, topology=topology, share=share):
                return _trial(
                    source,
                    target,
                    graph,
                    brand,
                    message,
                    rows,
                    legal,
                    supports,
                    accents,
                    topology,
                    share,
                    scale,
                )

            if fit(minimum) is None:
                continue
            low, high = minimum, max(minimum, base * 1.5)
            for _ in range(14):
                middle = (low + high) / 2
                if fit(middle) is not None:
                    low = middle
                else:
                    high = middle
            scales = sorted(set((min(base, low), low * 0.9, low)))
            for index, scale in enumerate(scales):
                design = fit(scale)
                if design is None:
                    continue
                axes = {
                    brand.id: "vertical",
                    message.id: "vertical",
                    **{row.id: "wrap" for row in rows},
                }
                for campaign in (c for c in graph.clusters if c.kind == "campaign"):
                    axes[campaign.id] = "horizontal" if topology == "columns" else "vertical"
                states.append(
                    AdvancedState(
                        f"peers-{topology}-{share:g}-{index}",
                        topology,
                        tuple(axes.items()),
                        json.dumps(design, sort_keys=True),
                    )
                )
    return tuple(states)
