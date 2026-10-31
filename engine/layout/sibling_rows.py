"""Translation-only sibling anchors supported by the original source rows.

This is a bounded alternative for comparison-like campaigns, not a universal
card template. It preserves each candidate unit's shape, font and local geometry
and declines when shared rows would detach an offer or fail canvas containment.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from statistics import median

from .appearance import unit_visible_box
from .geometry import Box, bounds, intersection, union
from .relationships import SourceRelationshipGraph, analyze_relationships


@dataclass(frozen=True)
class _Offer:
    campaign: object
    message: object
    identity: tuple[int, ...]
    parts: tuple[tuple[int, ...], ...]
    source_boxes: tuple[Box, ...]


def _box_union(boxes):
    left, top = min(box.x for box in boxes), min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return Box(left, top, right - left, bottom - top)


def _shared(values, tolerance):
    return max(values) - min(values) <= tolerance


def _source_offers(graph):
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    if len(campaigns) < 2 or graph.ambiguities:
        return None
    by_cluster = {cluster.id: cluster for cluster in graph.clusters}
    offers, claimed, row_count = [], set(), None
    for campaign in campaigns:
        if campaign.primary_unit not in campaign.unit_indices:
            return None
        children = [by_cluster[key] for key in campaign.member_cluster_ids]
        messages = [cluster for cluster in children if cluster.kind == "message"]
        if len(messages) != 1 or len(messages[0].unit_indices) < 3:
            return None
        message = messages[0]
        if graph.units[message.unit_indices[0]].role != "headline" or any(
            edge.axis != "vertical"
            for edge in graph.edges
            if edge.cluster_id == message.id and edge.code == "reading_order"
        ):
            return None
        action = graph.units[message.unit_indices[-1]]
        if not {"shape", "text"} <= {member.element_type for member in action.members}:
            return None
        rows = sorted(
            (cluster for cluster in children if cluster.kind == "badge_row"),
            key=lambda cluster: _box_union([graph.units[i].box for i in cluster.unit_indices]).y,
        )
        if row_count is not None and row_count != len(rows):
            return None
        row_count = len(rows)
        parts = (
            message.unit_indices[:1],
            message.unit_indices[1:-1],
            *(row.unit_indices for row in rows),
            message.unit_indices[-1:],
        )
        copy_indices = [index for part in parts for index in part]
        identity = tuple(index for index in campaign.unit_indices if index not in copy_indices)
        if (
            not identity
            or campaign.primary_unit not in identity
            or len(copy_indices) != len(set(copy_indices))
            or claimed.intersection(campaign.unit_indices)
        ):
            return None
        claimed.update(campaign.unit_indices)
        offers.append(
            _Offer(
                campaign,
                message,
                identity,
                parts,
                tuple(_box_union([graph.units[i].box for i in part]) for part in parts),
            )
        )
    offers.sort(
        key=lambda offer: _box_union([graph.units[i].box for i in offer.campaign.unit_indices]).x
    )
    tolerance = graph.height * 0.015
    # Source comparison headings, peer rows and action bottoms must actually
    # share anchors. Body alignment and product bottoms are separately optional.
    if not _shared([offer.source_boxes[0].y for offer in offers], tolerance) or not _shared(
        [offer.source_boxes[-1].y + offer.source_boxes[-1].height for offer in offers], tolerance
    ):
        return None
    if any(
        not _shared([offer.source_boxes[index].y for offer in offers], tolerance)
        for index in range(2, len(offers[0].parts) - 1)
    ):
        return None
    frames = [
        _box_union([graph.units[i].frame_box for i in offer.campaign.unit_indices])
        for offer in offers
    ]
    if any(
        first.x + first.width > second.x + min(first.width, second.width) * 0.1
        for first, second in zip(frames, frames[1:], strict=False)
    ):
        return None
    return tuple(offers)


def _translate(elements, dy):
    for element in elements:
        element["y"] += dy


def aligned_sibling_rows(
    source: dict,
    target: dict,
    graph: SourceRelationshipGraph,
    design: dict,
    axes: dict | tuple,
) -> dict | None:
    """Return an independently copied shared-row alternative, or decline.

    Only translations are permitted: every atomic unit retains all candidate
    dimensions, fonts and offsets. Product bottoms align only when they shared
    a source bottom; unequal source product prominence is never equalized.
    Full bounds and original relationship evidence are checked after placement.
    """
    if (
        target["width"] < target["height"]
        or design.get("canvas") != target
        or tuple(element["id"] for element in source["elements"]) != graph.source_ids
        or source["canvas"] != {"width": graph.width, "height": graph.height}
    ):
        return None
    offers = _source_offers(graph)
    if offers is None:
        return None
    axes = dict(axes)
    if any(
        axes.get(offer.campaign.id) != "horizontal"
        or axes.get(offer.message.id, "vertical") != "vertical"
        for offer in offers
    ):
        return None
    ids = [element["id"] for element in design["elements"]]
    if len(ids) != len(set(ids)) or set(ids) != set(graph.source_ids):
        return None
    result = deepcopy(design)
    by_id = {element["id"]: element for element in result["elements"]}
    original = {element["id"]: element for element in source["elements"]}

    def elements(indices):
        return [by_id[eid] for index in indices for eid in graph.units[index].element_ids]

    sections = [[elements(part) for part in offer.parts] for offer in offers]
    boxes = [[unit_visible_box(part, graph.assets_dir) for part in parts] for parts in sections]
    if any(min(box.width, box.height) <= 0 for row in boxes for box in row):
        return None
    campaign_boxes = [
        unit_visible_box(elements(offer.campaign.unit_indices), graph.assets_dir)
        for offer in offers
    ]
    for first, second in zip(campaign_boxes, campaign_boxes[1:], strict=False):
        vertical_overlap = max(
            0.0, min(first.y + first.height, second.y + second.height) - max(first.y, second.y)
        )
        if (
            first.x + first.width > second.x + min(first.width, second.width) * 0.1
            or vertical_overlap < min(first.height, second.height) * 0.5
        ):
            return None
    scales = []
    for offer in offers:
        for eid in offer.campaign.element_ids:
            first, second = original[eid], by_id[eid]
            scales.append(
                second["font_size"] / first["font_size"]
                if first["type"] == "text"
                else math.sqrt(
                    second["width"] * second["height"] / (first["width"] * first["height"])
                )
            )
    scale = median(scales)
    short = min(target["height"], target["width"] / len(offers))

    def gap(before, after):
        source_gap = max(
            max(
                0.0,
                offer.source_boxes[after].y
                - offer.source_boxes[before].y
                - offer.source_boxes[before].height,
            )
            for offer in offers
        )
        return 0.0 if source_gap == 0 else max(short * 0.02, min(short * 0.06, source_gap * scale))

    heights = [max(row[index].height for row in boxes) for index in range(len(boxes[0]))]
    starts = [0.0]
    for index in range(1, len(heights)):
        starts.append(starts[-1] + heights[index - 1] + gap(index - 1, index))
    shared_body = _shared([offer.source_boxes[1].y for offer in offers], graph.height * 0.015)
    shared_product = _shared(
        [
            graph.units[offer.campaign.primary_unit].box.y
            + graph.units[offer.campaign.primary_unit].box.height
            for offer in offers
        ],
        graph.height * 0.015,
    )
    identities = [elements(offer.identity) for offer in offers]
    product_boxes = [
        unit_visible_box(elements((offer.campaign.primary_unit,)), graph.assets_dir)
        for offer in offers
    ]
    product_baseline = max(
        product.y + product.height - row[0].y
        for product, row in zip(product_boxes, boxes, strict=True)
    )
    for parts, row, identity, product in zip(
        sections, boxes, identities, product_boxes, strict=True
    ):
        _translate(
            identity,
            product_baseline - product.y - product.height if shared_product else -row[0].y,
        )
        for index, (part, box) in enumerate(zip(parts, row, strict=True)):
            top = starts[index] + (heights[index] - box.height if index == len(parts) - 1 else 0)
            if index == 1 and not shared_body:
                # A different source body offset is not evidence for a new row.
                top = row[0].height + max(0.0, box.y - row[0].y - row[0].height)
            _translate(part, top - box.y)
    moved = [element for offer in offers for element in elements(offer.campaign.unit_indices)]
    frame = union(moved)
    if frame.height > target["height"] + 1e-6:
        return None
    _translate(moved, (target["height"] - frame.height) / 2 - frame.y)
    for element in result["elements"]:
        box = bounds(element)
        if (
            box.x < -1e-6
            or box.y < -1e-6
            or box.x + box.width > target["width"] + 1e-6
            or box.y + box.height > target["height"] + 1e-6
        ):
            return None
    candidate_boxes = [
        unit_visible_box(elements((unit.index,)), graph.assets_dir) for unit in graph.units
    ]
    for i, first in enumerate(graph.units):
        for second in graph.units[i + 1 :]:
            if (
                intersection(first.box, second.box) <= 1e-6
                and intersection(candidate_boxes[first.index], candidate_boxes[second.index]) > 1e-6
            ):
                return None
    if analyze_relationships(graph, result, axes)["warnings"]:
        return None
    return result
