"""Landscape proposals that preserve a source-evidenced central product axis.

Focal importance and rigid membership are separate constraints. Native atomic
groups keep their internals; optional standalone supporting copy may rewrap or
use a declared smaller scale. Nothing is removed or substituted.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace

from .appearance import unit_visible_box
from .blocks import _translate
from .candidates import _base
from .focal import infer_focal_profile
from .geometry import intersection, union
from .grouped import GroupedState, _accessories, _copy, _identity, _prepare
from .hierarchy import infer_hierarchy
from .model import build_context
from .relationships import build_relationship_graph
from .text import get_measurer


def _message(offer, context, graph, hierarchy, scale, width, factor, *, heading=False):
    local = deepcopy(context)
    exceptions = []
    for index in offer.message.unit_indices:
        unit = local.units[index]
        if unit.role in {"body", "legal"} and len(unit.elements) == 1 and unit.text:
            element = unit.elements[0]
            if factor < 1:
                element["font_size"] *= factor
                exceptions.append(f"support-text:{element['id']}: relative scale {factor:g}")
    if heading:
        indices = tuple(
            index for index in offer.message.unit_indices if context.units[index].role == "headline"
        )
        rows = ()
    else:
        indices, rows = offer.message.unit_indices, offer.rows
    if not indices:
        return None
    message = replace(offer.message, unit_indices=indices)
    placed = _copy(
        replace(offer, message=message, rows=rows), local, graph, hierarchy, scale, width
    )
    return (placed, tuple(exceptions)) if placed else None


def _realize(source, target, graph, hierarchy, offer, claimed, profile, mode, factor, scale):
    context = build_context(source)
    accessories = _accessories(source, context, hierarchy, claimed, target, scale)
    if accessories is None:
        return None
    fixed, fixed_groups, bottom, exceptions = accessories
    pad = min(target.values()) * 0.04
    height = bottom - pad
    identity = _identity(offer, hierarchy, scale, "vertical", graph.assets_dir)
    if not identity:
        return None
    frame = union(identity)
    if frame.height > height:
        return None
    primary_ids = set(profile.primary_element_ids)
    primary = unit_visible_box([e for e in identity if e["id"] in primary_ids], graph.assets_dir)
    _translate(
        identity,
        target["width"] * profile.anchor_x - primary.x - primary.width / 2,
        pad + (height - frame.height) / 2 - frame.y,
    )
    primary = unit_visible_box([e for e in identity if e["id"] in primary_ids], graph.assets_dir)
    font = (
        max(
            e["font_size"]
            for index in offer.message.unit_indices
            for e in context.units[index].text
        )
        * scale
    )
    gap = max(pad, font * 0.8)
    left_edge = primary.x - gap
    right_edge = primary.x + primary.width + gap
    width = target["width"] - pad - right_edge
    if width <= 0:
        return None
    if mode == "side":
        copy = _message(offer, context, graph, hierarchy, scale, width, factor)
        if copy is None:
            return None
        content, support_exceptions = copy
        frame = union(content)
        if frame.height > height or frame.width > width + 0.01:
            return None
        _translate(content, right_edge - frame.x, pad + (height - frame.height) / 2 - frame.y)
        campaign_axis = "horizontal"
    else:
        heading_indices = tuple(
            index for index in offer.message.unit_indices if context.units[index].role == "headline"
        )
        remaining = tuple(
            index for index in offer.message.unit_indices if index not in heading_indices
        )
        if len(heading_indices) != 1 or not remaining or left_edge <= pad:
            return None
        heading_copy = _message(
            offer, context, graph, hierarchy, scale, left_edge - pad, factor, heading=True
        )
        right_offer = replace(offer, message=replace(offer.message, unit_indices=remaining))
        following_copy = _message(right_offer, context, graph, hierarchy, scale, width, factor)
        if heading_copy is None or following_copy is None:
            return None
        heading, _ = heading_copy
        following, support_exceptions = following_copy
        hb, fb = union(heading), union(following)
        if (
            max(hb.height, fb.height) > height
            or hb.width > left_edge - pad + 0.01
            or fb.width > width + 0.01
        ):
            return None
        following_y = pad + (height - fb.height) * 0.58
        heading_y = min(pad + (height - hb.height) * 0.45, following_y)
        _translate(heading, pad - hb.x, heading_y - hb.y)
        _translate(following, right_edge - fb.x, following_y - fb.y)
        content = heading + following
        # Keep the product before its narrative in the original vertical order;
        # the two copy regions may share a reading row but never reverse it.
        message_box = unit_visible_box(content, graph.assets_dir)
        shift = max(
            0.0,
            primary.y + primary.height / 2 + font * 0.2 - message_box.y - message_box.height / 2,
        )
        if union(content).y + union(content).height + shift > bottom + 1e-6:
            return None
        _translate(content, 0, shift)
        campaign_axis = "vertical"
    placed = identity + content
    all_elements = _base(context, target) + placed + fixed
    frame = union(all_elements)
    if (
        frame.x < -1e-6
        or frame.y < -1e-6
        or frame.x + frame.width > target["width"] + 1e-6
        or frame.y + frame.height > target["height"] + 1e-6
    ):
        return None
    if any(
        e.get("font_size", 1) < 1 or min(e["width"], e["height"]) <= 0.001 for e in all_elements
    ):
        return None
    measurer = get_measurer()
    for element in all_elements:
        if element["type"] != "text":
            continue
        measured = measurer.measure(element, element["font_size"], element["width"])
        if measured.width > element["width"] + 0.01 or measured.height > element["height"] + 0.01:
            return None
    by_id = {e["id"]: e for e in all_elements}
    if len(all_elements) != len(source["elements"]) or set(by_id) != set(graph.source_ids):
        return None
    # Check separate atomic leaves, including corner accessories, against actual
    # painted support. Padded image frames may overlap empty space only.
    blocks = [[by_id[eid] for eid in unit.element_ids] for unit in graph.units]
    for index, first in enumerate(blocks):
        for second in blocks[index + 1 :]:
            a, b = (
                unit_visible_box(first, graph.assets_dir),
                unit_visible_box(second, graph.assets_dir),
            )
            if intersection(a, b) > 0.5:
                return None
    axes = {offer.message.id: "vertical", offer.campaign.id: campaign_axis}
    if offer.brand:
        axes[offer.brand.id] = "vertical"
    axes.update((row.id, "wrap") for row in offer.rows)
    return (
        {"canvas": dict(target), "elements": [by_id[e["id"]] for e in source["elements"]]},
        tuple(axes.items()),
        exceptions + support_exceptions,
    )


def generate_focal_states(source, target, graph=None, profile=None):
    """Offer at most four focal-axis layouts for an evidenced single product."""
    graph = graph or build_relationship_graph(source)
    profile = profile or infer_focal_profile(source, graph)
    if profile is None or target["width"] < target["height"] * 1.4:
        return ()
    prepared = _prepare(graph)
    if prepared is None or len(prepared[0]) != 1:
        return ()
    offers, claimed = prepared
    hierarchy = infer_hierarchy(source, graph)
    fonts = [e["font_size"] for e in source["elements"] if e["type"] == "text"]
    minimum = 1 / min(fonts, default=1) / 0.85
    upper = max(
        minimum, math.sqrt(target["width"] * target["height"] / (graph.width * graph.height)) * 2
    )
    states = []
    for mode in ("balanced", "side"):
        for factor in (1.0, 0.85):

            def fit(scale, mode=mode, factor=factor):
                return _realize(
                    source,
                    target,
                    graph,
                    hierarchy,
                    offers[0],
                    claimed,
                    profile,
                    mode,
                    factor,
                    scale,
                )

            if fit(minimum) is None:
                continue
            low, high = minimum, upper
            for _ in range(18):
                middle = (low + high) / 2
                if fit(middle) is None:
                    high = middle
                else:
                    low = middle
            candidate = fit(low * 0.995)
            if candidate is None:
                continue
            design, axes, exceptions = candidate
            states.append(
                GroupedState(
                    f"focal-{mode}-{factor:g}",
                    f"focal/{mode}",
                    axes,
                    json.dumps(design, sort_keys=True),
                    exceptions,
                )
            )
    return tuple(states)
