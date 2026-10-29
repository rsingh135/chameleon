"""Bounded compound-layout experiment built from an immutable source graph.

The first realizer handles one inferred brand/product cluster and one message
cluster, plus source corner accents. Ambiguous or unsupported structures decline
instead of forcing every design into this family. It is not a general optimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .advanced import generate_advanced_states
from .appearance import unit_visible_box
from .blocks import _brand, _message, _translate, _uniform
from .campaigns import generate_campaign_states
from .candidates import _base
from .geometry import union
from .model import build_context
from .relationships import SourceRelationshipGraph, build_relationship_graph
from .text import get_measurer


@dataclass(frozen=True)
class CompositionState:
    name: str
    topology: str
    source_digest: str
    width: int
    height: int
    cluster_axes: tuple[tuple[str, str], ...]
    design_json: str

    @property
    def design(self) -> dict:
        """Each caller gets a copy; fitted frames never become the new source."""
        return json.loads(self.design_json)


def _realize(source, target, graph, brand, message, accents, topology, typography):
    context = build_context(source)
    assets = graph.assets_dir
    width, height = target["width"], target["height"]
    short = min(width, height)
    pad, gap = short * 0.04, short * 0.10
    scale = math.sqrt(width * height / (graph.width * graph.height)) * typography
    message_units = [context.units[index] for index in message.unit_indices]
    brand_units = [context.units[index] for index in brand.unit_indices]
    # Preserve source typography/control ratios; search wraps rather than
    # inflating every copy frame to consume all available width.
    text_widths = [
        get_measurer().measure(e, e["font_size"] * scale, width * 0.65).width
        for unit in message_units
        for e in unit.text
    ]
    control_width = max(
        (unit.box.width * scale for unit in message_units if len(unit.elements) > 1), default=0
    )
    copy_width = max(control_width, max(text_widths, default=width * 0.4))
    copy_width = min(width - 2 * pad, max(short * 0.65, copy_width + short * 0.06))
    if topology == "columns":
        copy_width = min(copy_width, width * 0.64)
    content = _message(message_units, copy_width, scale, short, assets)
    if not content:
        return None
    content_frame = union(content)
    if content_frame.height > height - 2 * pad or content_frame.width > width - 2 * pad:
        return None
    brand_width = width * (0.36 if topology == "columns" else 0.72)
    brand_height = (height - 2 * pad) * (0.94 if topology == "columns" else 0.56)
    identity = _brand(brand_units, brand_width, brand_height, short * 0.035, assets)
    brand_frame = union(identity)
    brand_visible = unit_visible_box(identity, assets)
    content_visible = unit_visible_box(content, assets)
    if topology == "columns":
        total = brand_visible.width + gap + content_visible.width
        left = (width - total) / 2
        _translate(identity, left - brand_visible.x, (height - brand_frame.height) / 2)
        _translate(
            content,
            left + brand_visible.width + gap - content_visible.x,
            (height - content_frame.height) / 2 - content_frame.y,
        )
    else:
        total = brand_frame.height + gap + content_frame.height
        if total > height - 2 * pad:
            return None
        top = (height - total) / 2
        _translate(identity, width / 2, top)
        _translate(content, width / 2, top + brand_frame.height + gap - content_frame.y)
    elements = _base(context, target) + identity + content
    for index in accents:
        unit = context.units[index]
        box = unit.box
        left = box.x + box.width / 2 < graph.width / 2
        top = box.y + box.height / 2 < graph.height / 2
        placed = _uniform(unit, scale, 0, 0, assets)
        frame = union(placed)
        x = pad if left else width - pad - frame.width
        y = pad if top else height - pad - frame.height
        _translate(placed, x - frame.x, y - frame.y)
        elements.extend(placed)
    # Never repair elements independently: that would silently tear clusters.
    for element in elements:
        if element.get("font_size", 1) < 1:
            return None
    frame = union(elements)
    if (
        frame.x < 0
        or frame.y < 0
        or frame.x + frame.width > width
        or frame.y + frame.height > height
    ):
        return None
    by_id = {element["id"]: element for element in elements}
    if set(by_id) != set(graph.source_ids):
        return None
    return {"canvas": dict(target), "elements": [by_id[e["id"]] for e in source["elements"]]}


def generate_composition_states(
    source: dict, target: dict, graph: SourceRelationshipGraph | None = None
) -> tuple[CompositionState, ...]:
    """Try coherent compound states; decline unsupported/ambiguous structures."""
    graph = graph or build_relationship_graph(source)
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    compound = generate_campaign_states(source, target, graph)
    if not compound:
        compound = generate_advanced_states(source, target, graph)
    if compound:
        return tuple(
            CompositionState(
                state.name,
                state.topology,
                digest,
                target["width"],
                target["height"],
                state.cluster_axes,
                state.design_json,
            )
            for state in compound
        )
    brands = [cluster for cluster in graph.clusters if cluster.kind == "brand"]
    messages = [cluster for cluster in graph.clusters if cluster.kind == "message"]
    if len(brands) != 1 or len(messages) != 1:
        return ()
    brand, message = brands[0], messages[0]
    if any(
        edge.axis != "vertical" for edge in graph.edges if edge.cluster_id in {brand.id, message.id}
    ):
        return ()
    owned = set(brand.unit_indices) | set(message.unit_indices)
    accents = []
    for unit in graph.units:
        if unit.index in owned:
            continue
        cx = (unit.frame_box.x + unit.frame_box.width / 2) / graph.width
        cy = (unit.frame_box.y + unit.frame_box.height / 2) / graph.height
        if unit.role != "badge" or not ((cx < 0.25 or cx > 0.75) and cy < 0.25):
            return ()
        accents.append(unit.index)
    states = []
    topologies = (
        ("columns", "stacked") if target["width"] / target["height"] >= 1.35 else ("stacked",)
    )
    for topology in topologies:
        axes = tuple((cluster.id, "vertical") for cluster in (brand, message))
        owned_campaigns = [
            c for c in graph.clusters if c.kind == "campaign" and set(c.unit_indices) <= owned
        ]
        axes += tuple(
            (c.id, "horizontal" if topology == "columns" else "vertical") for c in owned_campaigns
        )
        for typography in (0.85, 1.0):
            design = _realize(source, target, graph, brand, message, accents, topology, typography)
            if design is not None:
                states.append(
                    CompositionState(
                        f"{topology}-{typography:g}",
                        topology,
                        digest,
                        target["width"],
                        target["height"],
                        axes,
                        json.dumps(design, sort_keys=True),
                    )
                )
    return tuple(states)
