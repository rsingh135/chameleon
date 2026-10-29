"""Immutable source relationships for experiments in coherent graphic reflow.

Geometry is evidence for a relationship, not proof of human meaning. Explicit
groups are strong; inferred message and brand clusters retain their confidence
and provenance. Candidates are measured against the original graph only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from engine.resources import get_resources, using_resources

from .appearance import unit_visible_box
from .geometry import Box, bounds, union
from .model import build_context

_CODES = (
    "reading_order",
    "message_attachment",
    "shared_axis",
    "brand_association",
    "explicit_group",
    "peer_order",
    "peer_cohesion",
    "peer_prominence",
    "campaign_ownership",
    "row_attachment",
)


@dataclass(frozen=True)
class ElementGeometry:
    element_id: str
    element_type: str
    box: Box
    font_size: float | None


@dataclass(frozen=True)
class RelationshipUnit:
    index: int
    element_ids: tuple[str, ...]
    role: str
    box: Box
    frame_box: Box
    members: tuple[ElementGeometry, ...]


@dataclass(frozen=True)
class RelationshipCluster:
    id: str
    kind: str
    unit_indices: tuple[int, ...]
    element_ids: tuple[str, ...]
    confidence: float
    provenance: tuple[str, ...]
    member_cluster_ids: tuple[str, ...] = ()
    primary_unit: int | None = None


@dataclass(frozen=True)
class RelationshipEdge:
    code: str
    source_unit: int
    target_unit: int
    cluster_id: str
    axis: str
    alignment: str
    source_gap: float
    source_alignment: float
    confidence: float
    provenance: tuple[str, ...]
    target_cluster_id: str = ""


@dataclass(frozen=True)
class RelationshipAmbiguity:
    source_unit: int
    candidate_cluster_ids: tuple[str, ...]
    costs: tuple[float, ...]
    reason: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class SourceRelationshipGraph:
    width: float
    height: float
    units: tuple[RelationshipUnit, ...]
    edges: tuple[RelationshipEdge, ...]
    clusters: tuple[RelationshipCluster, ...]
    source_ids: tuple[str, ...]
    assets_dir: str | None
    ambiguities: tuple[RelationshipAmbiguity, ...] = ()
    fonts_dir: str | None = None


def _primary(box: Box, axis: str) -> tuple[float, float]:
    return (box.y, box.height) if axis == "vertical" else (box.x, box.width)


def _cross(box: Box, axis: str) -> tuple[float, float]:
    return (box.x, box.width) if axis == "vertical" else (box.y, box.height)


def _anchor(box: Box, axis: str, alignment: str) -> float:
    start, size = _cross(box, axis)
    return start + size * {"start": 0.0, "center": 0.5, "end": 1.0}[alignment]


def _pair(first: Box, second: Box, axis: str, alignment: str) -> tuple[float, float]:
    a, size_a = _primary(first, axis)
    b, size_b = _primary(second, axis)
    gap = (b - a - size_a) / max(size_a, size_b, 1.0)
    cross_scale = max(_cross(first, axis)[1], _cross(second, axis)[1], 1.0)
    offset = (_anchor(second, axis, alignment) - _anchor(first, axis, alignment)) / cross_scale
    return gap, offset


def _connection(first: Box, second: Box, width: float, height: float):
    """Return a plausible forward axis/anchor, without examining text content."""
    if min(first.width, first.height, second.width, second.height) <= 0:
        return None
    choices = []
    for axis in ("vertical", "horizontal"):
        start, size = _primary(first, axis)
        following, _ = _primary(second, axis)
        extent = height if axis == "vertical" else width
        # Source container overlap is allowed; a following unit still starts later.
        if following < start or following - start - size > extent * 0.4:
            continue
        alignment = min(
            ("center", "start", "end"),
            key=lambda value: abs(_pair(first, second, axis, value)[1]),
        )
        gap, offset = _pair(first, second, axis, alignment)
        if abs(offset) <= 0.18:
            choices.append((max(0.0, gap) + abs(offset) * 4, axis, alignment))
    return min(choices)[1:] if choices else None


def _box_union(boxes: list[Box]) -> Box:
    left, top = min(box.x for box in boxes), min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return Box(left, top, right - left, bottom - top)


def _ownership_cost(first: Box, second: Box, width: float, height: float):
    """Spatial evidence for ownership, independent of IDs, language and canvas pose."""
    dx = (second.x + second.width / 2 - first.x - first.width / 2) / width
    dy = (second.y + second.height / 2 - first.y - first.height / 2) / height
    axis = "vertical" if abs(dy) >= abs(dx) else "horizontal"
    _, offset = _pair(first, second, axis, "center")
    return math.hypot(dx, dy) + abs(offset) * 0.3, axis


def _separated(best: float, other: float) -> bool:
    # A near tie is uncertainty, not permission for source-order greedy matching.
    return other - best >= 0.12


def _peer_neighbors(first: Box, second: Box) -> bool:
    if first.x > second.x:
        first, second = second, first
    gap = second.x - first.x - first.width
    return (
        min(first.height, second.height) > 0
        and abs(first.y + first.height / 2 - second.y - second.height / 2)
        <= max(first.height, second.height) * 0.3
        and max(first.height, second.height) <= min(first.height, second.height) * 1.5
        and -min(first.width, second.width) * 0.05
        <= gap
        <= max(max(first.height, second.height) * 3, min(first.width, second.width) * 0.5)
    )


def build_relationship_graph(
    source: dict, assets_dir: str | Path | None = None
) -> SourceRelationshipGraph:
    """Snapshot source units and infer a sparse graph with explicit uncertainty.

    Clusters expose ``id``, ``kind``, ``unit_indices`` and ``element_ids``.
    ``message`` joins a headline to aligned following copy and a nearby action.
    ``brand`` is a confidence-0.7 logo/product association hypothesis. It must not
    be treated as an unconditional hard group by a solver.
    """
    ids = tuple(element["id"] for element in source["elements"])
    if len(ids) != len(set(ids)):
        raise ValueError("Source relationship graph requires unique element IDs")
    context = build_context(source)
    resources = get_resources()
    assets = str(Path(assets_dir).resolve() if assets_dir is not None else resources.assets_dir)
    units = tuple(
        RelationshipUnit(
            index,
            tuple(element["id"] for element in unit.elements),
            unit.role,
            unit_visible_box(unit.elements, assets),
            unit.box,
            tuple(
                ElementGeometry(
                    element["id"], element["type"], bounds(element), element.get("font_size")
                )
                for element in unit.elements
            ),
        )
        for index, unit in enumerate(context.units)
    )
    clusters, edges, ambiguities = [], [], []

    def connect(
        code,
        first,
        second,
        cluster,
        axis,
        alignment,
        confidence,
        provenance,
        target_cluster_id="",
        target_box=None,
    ):
        gap, offset = _pair(units[first].box, target_box or units[second].box, axis, alignment)
        edges.append(
            RelationshipEdge(
                code,
                first,
                second,
                cluster,
                axis,
                alignment,
                gap,
                offset,
                confidence,
                provenance,
                target_cluster_id,
            )
        )

    # Explicit source groups remain atomic even when their parent topology changes.
    for index, original in enumerate(context.units):
        group_ids = {element.get("group_id") for element in original.elements}
        if len(original.elements) > 1 and None not in group_ids and len(group_ids) == 1:
            connect(
                "explicit_group",
                index,
                index,
                "",
                "vertical",
                "center",
                1.0,
                ("explicit source group_id",),
            )

    controls = [
        unit
        for unit in units
        if any(member.element_type == "shape" for member in unit.members)
        and any(member.element_type == "text" for member in unit.members)
    ]
    peer_controls = set()
    for index, first in enumerate(controls):
        for second in controls[index + 1 :]:
            fonts = [
                max(member.font_size or 0 for member in unit.members) for unit in (first, second)
            ]
            if (
                min(fonts) > 0
                and max(fonts) / min(fonts) <= 1.2
                and _peer_neighbors(first.box, second.box)
            ):
                peer_controls.update((first.index, second.index))
    claimed = set()
    for heading in (unit for unit in units if unit.role == "headline"):
        following = []
        for body in (unit for unit in units if unit.role == "body" and unit.index not in claimed):
            relation = _connection(heading.box, body.box, context.width, context.height)
            if relation:
                axis, anchor = relation
                gap, _ = _pair(heading.box, body.box, axis, anchor)
                following.append((max(0, gap), body.index, axis, anchor))
        if not following:
            continue
        _, body_index, axis, anchor = min(following)
        members = [heading.index, body_index]
        claimed.add(body_index)
        # Add an action only when its geometric ownership is supported by the source.
        possible = []
        body_font = max((member.font_size or 0 for member in units[body_index].members), default=0)
        for action in (unit for unit in units if unit.index not in claimed):
            control = any(member.element_type == "shape" for member in action.members) and any(
                member.element_type == "text" for member in action.members
            )
            font = max((member.font_size or 0 for member in action.members), default=0)
            # A local offer's action may sit halfway down a multi-offer page.
            # Infer prominence relative to its local copy, not global footer y.
            if action.role != "action" and not (
                control and action.index not in peer_controls and font >= body_font * 0.85
            ):
                continue
            relation = _connection(units[body_index].box, action.box, context.width, context.height)
            if relation and relation[0] == axis:
                gap, offset = _pair(units[body_index].box, action.box, axis, anchor)
                if abs(offset) <= 0.2:
                    possible.append((max(0, gap), action.index))
        if possible:
            action = min(possible)[1]
            members.append(action)
            claimed.add(action)
        cluster_id = f"message:{sum(cluster.kind == 'message' for cluster in clusters)}"
        provenance = (
            "source headline role",
            "aligned forward source adjacency",
            "no candidate-derived inference",
        )
        clusters.append(
            RelationshipCluster(
                cluster_id,
                "message",
                tuple(members),
                tuple(eid for index in members for eid in units[index].element_ids),
                0.85,
                provenance,
            )
        )
        for first, second in zip(members, members[1:], strict=False):
            for code in ("reading_order", "message_attachment", "shared_axis"):
                connect(code, first, second, cluster_id, axis, anchor, 0.85, provenance)

    images = [
        unit
        for unit in units
        if all(member.element_type == "image" for member in unit.members)
        and unit.box.width * unit.box.height > 0
    ]
    largest = max((unit.frame_box.width * unit.frame_box.height for unit in images), default=0)
    logos = [
        unit
        for unit in images
        if unit.role == "logo"
        or (
            unit.frame_box.width / max(unit.frame_box.height, 1e-9) > 2
            and unit.frame_box.width * unit.frame_box.height <= largest * 0.45
        )
    ]
    logo_ids = {unit.index for unit in logos}
    other_images = [unit for unit in images if unit.index not in logo_ids]
    largest_product = max(
        (unit.frame_box.width * unit.frame_box.height for unit in other_images), default=0
    )
    products = [
        unit
        for unit in other_images
        if unit.frame_box.width * unit.frame_box.height >= largest_product * 0.25
    ]
    claimed_products = set()
    for logo in logos:
        hypotheses = []
        for hero in (unit for unit in products if unit.index not in claimed_products):
            relation = _connection(logo.box, hero.box, context.width, context.height)
            if relation:
                axis, alignment = relation
                # Brand association requires center alignment, not merely adjacent edges.
                gap, offset = _pair(logo.box, hero.box, axis, "center")
                if abs(offset) <= 0.15:
                    hypotheses.append((max(0, gap), hero.index, axis))
        if not hypotheses:
            continue
        _, hero_index, axis = min(hypotheses)
        claimed_products.add(hero_index)
        cluster_id = f"brand:{sum(cluster.kind == 'brand' for cluster in clusters)}"
        provenance = (
            "inferred logo and hero roles",
            "shared visible source center axis",
            "association hypothesis, not proven semantics",
        )
        members = (logo.index, hero_index)
        clusters.append(
            RelationshipCluster(
                cluster_id,
                "brand",
                members,
                tuple(eid for index in members for eid in units[index].element_ids),
                0.7,
                provenance,
            )
        )
        connect(
            "brand_association", logo.index, hero_index, cluster_id, axis, "center", 0.7, provenance
        )

    # Peer rows contain grouped shape/label controls, not every global badge.
    # Nearby source rows remain separate when another campaign is far away.
    peer_units = [
        unit
        for unit in units
        if unit.role == "badge"
        and unit.index not in claimed
        and any(member.element_type == "shape" for member in unit.members)
        and any(member.element_type == "text" for member in unit.members)
        and not any(element.get("rotation", 0) for element in context.units[unit.index].elements)
    ]
    peer_units.sort(key=lambda unit: (unit.box.x, unit.box.y, unit.index))
    groups = []
    for unit in peer_units:
        attached = False
        for group in groups:
            previous = group[-1]
            a, b = previous.box, unit.box
            if _peer_neighbors(a, b):
                group.append(unit)
                attached = True
                break
        if not attached:
            groups.append([unit])
    for group in groups:
        if len(group) < 2:
            continue
        group.sort(key=lambda unit: (unit.box.x, unit.index))
        cluster_id = f"badge_row:{sum(cluster.kind == 'badge_row' for cluster in clusters)}"
        provenance = (
            "source peer control baseline",
            "similar control heights and nearby spacing",
            "ordered row may wrap without changing membership",
        )
        clusters.append(
            RelationshipCluster(
                cluster_id,
                "badge_row",
                tuple(unit.index for unit in group),
                tuple(eid for unit in group for eid in unit.element_ids),
                0.85,
                provenance,
            )
        )
        for first, second in zip(group, group[1:], strict=False):
            for code in ("peer_order", "peer_cohesion", "peer_prominence"):
                connect(
                    code,
                    first.index,
                    second.index,
                    cluster_id,
                    "horizontal",
                    "center",
                    0.85,
                    provenance,
                )

    messages = [cluster for cluster in clusters if cluster.kind == "message"]
    message_boxes = {
        cluster.id: _box_union([units[index].box for index in cluster.unit_indices])
        for cluster in messages
    }
    costs = {
        (product.index, message.id): _ownership_cost(
            product.box, message_boxes[message.id], context.width, context.height
        )
        for product in products
        for message in messages
    }
    brands = [cluster for cluster in clusters if cluster.kind == "brand"]
    if len(brands) > 1:
        # Repeated local wordmarks provide section boundaries. A preceding
        # offer's complete message should not steal the next section's product
        # simply because a long preceding caption has a nearby center.
        for brand in brands:
            header, product_index = brand.unit_indices
            brand_edge = next(
                edge
                for edge in edges
                if edge.cluster_id == brand.id and edge.code == "brand_association"
            )
            header_start, _ = _primary(units[header].box, brand_edge.axis)
            for message in messages:
                start, size = _primary(message_boxes[message.id], brand_edge.axis)
                if start + size < header_start:
                    value, axis = costs[product_index, message.id]
                    costs[product_index, message.id] = (value + 2.0, axis)
    assignments = []
    for product in products:
        ranked = sorted(
            ((costs[product.index, message.id][0], message.id) for message in messages),
        )
        if not ranked:
            continue
        best, chosen = ranked[0]
        competing = sorted((costs[other.index, chosen][0], other.index) for other in products)
        margin = len(ranked) == 1 or _separated(best, ranked[1][0])
        mutual = competing[0][1] == product.index and (
            len(competing) == 1 or _separated(best, competing[1][0])
        )
        if not margin or not mutual:
            ambiguities.append(
                RelationshipAmbiguity(
                    product.index,
                    tuple(name for _, name in ranked),
                    tuple(round(cost, 6) for cost, _ in ranked),
                    "Product/message ownership lacks separated mutual spatial evidence.",
                    ("all source message alternatives compared", "no greedy fallback assignment"),
                )
            )
            continue
        assignments.append((product.index, chosen, costs[product.index, chosen][1]))

    row_owners = {}
    for row in (cluster for cluster in clusters if cluster.kind == "badge_row"):
        row_box = _box_union([units[index].box for index in row.unit_indices])
        ranked = sorted(
            (
                _ownership_cost(row_box, message_boxes[message.id], context.width, context.height)[
                    0
                ],
                message.id,
            )
            for message in messages
        )
        if ranked and (len(ranked) == 1 or _separated(ranked[0][0], ranked[1][0])):
            row_owners[row.id] = ranked[0][1]
        elif ranked:
            ambiguities.append(
                RelationshipAmbiguity(
                    row.unit_indices[0],
                    tuple(name for _, name in ranked),
                    tuple(round(cost, 6) for cost, _ in ranked),
                    "Peer row ownership is ambiguous between source messages.",
                    ("source row remains independent", "no forced campaign ownership"),
                )
            )

    by_cluster = {cluster.id: cluster for cluster in clusters}
    for product_index, message_id, axis in assignments:
        message = by_cluster[message_id]
        brand = next(
            (
                cluster
                for cluster in clusters
                if cluster.kind == "brand" and product_index in cluster.unit_indices
            ),
            None,
        )
        rows = [by_cluster[name] for name, owner in row_owners.items() if owner == message_id]
        children = ([brand.id] if brand else []) + [message_id] + [row.id for row in rows]
        owned = {product_index, *message.unit_indices}
        if brand:
            owned.update(brand.unit_indices)
        for row in rows:
            owned.update(row.unit_indices)
        ordered = tuple(
            sorted(owned, key=lambda index: (units[index].box.y, units[index].box.x, index))
        )
        cluster_id = f"campaign:{product_index}"
        provenance = (
            "comparable foreground product image",
            "separated mutual local message ownership",
            "source spatial evidence, not semantic certainty",
        )
        clusters.append(
            RelationshipCluster(
                cluster_id,
                "campaign",
                ordered,
                tuple(eid for index in ordered for eid in units[index].element_ids),
                0.8,
                provenance,
                tuple(children),
                product_index,
            )
        )
        connect(
            "campaign_ownership",
            product_index,
            message.unit_indices[0],
            cluster_id,
            axis,
            "center",
            0.8,
            provenance,
            target_cluster_id=message_id,
            target_box=message_boxes[message_id],
        )
        if len(message.unit_indices) >= 3:
            body_index, action_index = message.unit_indices[-2:]
            body, action = units[body_index].box, units[action_index].box
            message_axis = next(
                edge.axis
                for edge in edges
                if edge.cluster_id == message.id and edge.code == "reading_order"
            )
            body_start, body_size = _primary(body, message_axis)
            action_start, _ = _primary(action, message_axis)
            for row in rows:
                row_box = _box_union([units[index].box for index in row.unit_indices])
                row_start, row_size = _primary(row_box, message_axis)
                if (
                    row_start >= body_start + body_size - 0.5
                    and row_start + row_size <= action_start + 0.5
                ):
                    before, _ = _pair(body, row_box, message_axis, "center")
                    _, offset = _pair(row_box, message_boxes[message_id], message_axis, "center")
                    edges.append(
                        RelationshipEdge(
                            "row_attachment",
                            row.unit_indices[0],
                            action_index,
                            row.id,
                            message_axis,
                            "center",
                            before,
                            offset,
                            0.85,
                            (
                                "source peer row lies between supporting copy and local action",
                                "row ownership inferred from original campaign",
                                "wrapping preserves local reading position",
                            ),
                            message.id,
                        )
                    )

    # Layer creation order is not authored reading order. Canonicalize campaign
    # rows from their original compound geometry before a realizer packs them.
    campaign_boxes = {
        cluster.id: _box_union([units[index].box for index in cluster.unit_indices])
        for cluster in clusters
        if cluster.kind == "campaign"
    }
    source_rows = []
    for campaign in sorted(
        (cluster for cluster in clusters if cluster.kind == "campaign"),
        key=lambda cluster: (campaign_boxes[cluster.id].y, campaign_boxes[cluster.id].x),
    ):
        box = campaign_boxes[campaign.id]
        for row in source_rows:
            other = campaign_boxes[row[0].id]
            overlap = max(0, min(box.y + box.height, other.y + other.height) - max(box.y, other.y))
            if (
                abs(box.y - other.y) <= min(box.height, other.height) * 0.2
                or overlap >= min(box.height, other.height) * 0.7
            ):
                row.append(campaign)
                break
        else:
            source_rows.append([campaign])
    ordered_campaigns = [
        cluster
        for row in source_rows
        for cluster in sorted(row, key=lambda cluster: campaign_boxes[cluster.id].x)
    ]
    clusters = [cluster for cluster in clusters if cluster.kind != "campaign"] + ordered_campaigns

    return SourceRelationshipGraph(
        context.width,
        context.height,
        units,
        tuple(edges),
        tuple(clusters),
        ids,
        assets,
        tuple(ambiguities),
        str(resources.fonts_dir),
    )


def analyze_relationships(
    graph: SourceRelationshipGraph, candidate: dict, state: dict[str, str] | None = None
) -> dict:
    """Measure original relationships under an explicitly declared cluster flow.

    ``state`` optionally maps a cluster ID to ``vertical`` or ``horizontal``;
    peer badge rows additionally permit the explicit ``wrap`` state.
    A complete block may move/scale without penalty; changing its flow is an
    explicit state change, never an interpretation inferred from the candidate.
    Structural and immutable appearance validation remains the caller's job.
    """
    by_id = {element["id"]: element for element in candidate["elements"]}
    if set(by_id) != set(graph.source_ids) or len(by_id) != len(candidate["elements"]):
        raise ValueError("Relationship analysis requires every original element exactly once")
    state = state or {}
    cluster_map = {cluster.id: cluster for cluster in graph.clusters}
    if any(
        key not in cluster_map
        or (
            value not in {"vertical", "horizontal"}
            and not (value == "wrap" and cluster_map[key].kind == "badge_row")
        )
        for key, value in state.items()
    ):
        raise ValueError("Relationship states must map known cluster IDs to permitted flow states")
    with using_resources(assets_dir=graph.assets_dir, fonts_dir=graph.fonts_dir):
        boxes = {
            unit.index: unit_visible_box([by_id[eid] for eid in unit.element_ids], graph.assets_dir)
            for unit in graph.units
        }
    values = {code: [] for code in _CODES}
    message_boxes = {
        cluster.id: _box_union([boxes[index] for index in cluster.unit_indices])
        for cluster in graph.clusters
        if cluster.kind == "message"
    }
    products = [
        cluster.primary_unit
        for cluster in graph.clusters
        if cluster.kind == "campaign" and cluster.primary_unit is not None
    ]
    warnings, relations = [], []
    for edge in graph.edges:
        first, second = graph.units[edge.source_unit], graph.units[edge.target_unit]
        flow = state.get(
            edge.target_cluster_id if edge.code == "row_attachment" else edge.cluster_id,
            edge.axis,
        )
        axis = "horizontal" if flow == "wrap" else flow
        a, b = boxes[first.index], boxes[second.index]
        if edge.target_cluster_id:
            b = message_boxes[edge.target_cluster_id]
        gap, alignment = _pair(a, b, axis, edge.alignment)
        lost, message = False, ""
        if edge.code == "explicit_group":
            members = [by_id[eid] for eid in first.element_ids]
            frame = union(members)
            error = 0.0
            for original, element in zip(first.members, members, strict=True):
                box = bounds(element)
                for (
                    start,
                    size,
                    source_start,
                    source_size,
                    frame_start,
                    frame_size,
                    source_frame_start,
                    source_frame_size,
                ) in (
                    (
                        box.x,
                        box.width,
                        original.box.x,
                        original.box.width,
                        frame.x,
                        frame.width,
                        first.frame_box.x,
                        first.frame_box.width,
                    ),
                    (
                        box.y,
                        box.height,
                        original.box.y,
                        original.box.height,
                        frame.y,
                        frame.height,
                        first.frame_box.y,
                        first.frame_box.height,
                    ),
                ):
                    error = max(
                        error,
                        abs(
                            (start - frame_start) / max(frame_size, 1e-9)
                            - (source_start - source_frame_start) / max(source_frame_size, 1e-9)
                        ),
                        abs(
                            size / max(frame_size, 1e-9)
                            - source_size / max(source_frame_size, 1e-9)
                        ),
                    )
            fidelity = math.exp(-error * 4)
            lost = error > 0.1
            message = "Members of an explicit source group lost their internal geometry."
        elif edge.code == "reading_order":
            before, _ = _primary(a, axis)
            after, _ = _primary(b, axis)
            fidelity = 1.0 if after >= before - 0.5 else 0.0
            lost = fidelity == 0
            message = "Related content no longer follows its source reading order."
        elif edge.code == "message_attachment":
            growth = max(0.0, gap - edge.source_gap)
            fidelity = math.exp(-growth)
            lost = growth > 0.6
            message = "An associated message or action moved away from its source cluster."
        elif edge.code == "shared_axis":
            drift = abs(alignment - edge.source_alignment)
            fidelity = math.exp(-drift * 3)
            lost = drift > 0.2
            message = "Related message content lost its shared source alignment."
        elif edge.code == "brand_association":
            # Brand ownership is a soft hypothesis, not an identity constraint.
            growth = max(0.0, gap - edge.source_gap)
            drift = abs(alignment - edge.source_alignment)
            before, _ = _primary(a, axis)
            after, _ = _primary(b, axis)
            fidelity = math.exp(-growth - drift * 3) if after >= before - 0.5 else 0.0
            lost = drift > 0.2 or growth > 0.8 or fidelity == 0
            message = (
                "The inferred brand/product association no longer stays together; "
                "review this source hypothesis."
            )
        elif edge.code in {"peer_order", "peer_cohesion"}:
            if flow == "wrap":
                same_row = (
                    abs(a.y + a.height / 2 - b.y - b.height / 2) <= max(a.height, b.height) * 0.5
                )
                ordered = b.x >= a.x - 0.5 if same_row else b.y >= a.y + a.height * 0.5
                gap, alignment = _pair(a, b, "horizontal" if same_row else "vertical", "center")
            else:
                before, _ = _primary(a, axis)
                after, _ = _primary(b, axis)
                ordered = after >= before - 0.5
            if edge.code == "peer_order":
                fidelity = 1.0 if ordered else 0.0
                lost = not ordered
                message = "Peer controls no longer follow their source membership order."
            else:
                growth = max(0.0, gap - edge.source_gap)
                drift = 0.0 if flow == "wrap" else abs(alignment - edge.source_alignment)
                fidelity = math.exp(-growth - drift)
                lost = growth > 0.6 or drift > 0.5
                message = "A peer control became detached from its source row."
        elif edge.code == "peer_prominence":
            source_area_ratio = (
                first.box.width * first.box.height / max(second.box.width * second.box.height, 1e-9)
            )
            area_ratio = a.width * a.height / max(b.width * b.height, 1e-9)
            area_drift = abs(math.log(max(area_ratio, 1e-9) / max(source_area_ratio, 1e-9))) / 2
            source_fonts = [
                max((member.font_size or 0 for member in unit.members), default=0)
                for unit in (first, second)
            ]
            candidate_fonts = [
                max((by_id[eid].get("font_size", 0) for eid in unit.element_ids), default=0)
                for unit in (first, second)
            ]
            font_drift = 0.0
            if min(source_fonts + candidate_fonts) > 0:
                font_drift = abs(
                    math.log(
                        (candidate_fonts[0] / candidate_fonts[1])
                        / (source_fonts[0] / source_fonts[1])
                    )
                )
            fidelity = math.exp(-max(area_drift, font_drift))
            lost = font_drift > math.log(1.4) or area_drift > math.log(1.6)
            message = "Peer controls lost their relative label or visual prominence."
        elif edge.code == "row_attachment":
            row = cluster_map[edge.cluster_id]
            owner = cluster_map[edge.target_cluster_id]
            row_box = _box_union([boxes[index] for index in row.unit_indices])
            body_index, action_index = owner.unit_indices[-2:]
            body, action = boxes[body_index], boxes[action_index]
            body_start, body_size = _primary(body, axis)
            row_start, row_size = _primary(row_box, axis)
            action_start, _ = _primary(action, axis)
            ordered = (
                row_start >= body_start + body_size - 0.5
                and row_start + row_size <= action_start + 0.5
            )
            gap, _ = _pair(body, row_box, axis, "center")
            after_gap, _ = _pair(row_box, action, axis, "center")
            source_row = _box_union([graph.units[index].box for index in row.unit_indices])
            source_after, _ = _pair(source_row, graph.units[action_index].box, edge.axis, "center")
            # A heading may occupy a separate region while its supporting
            # body, badge row and action still form one local column. Compare
            # that local spine in BOTH source and candidate, rather than the
            # bounding rectangle of the entire message (which includes the
            # heading's unrelated horizontal extent).
            local_spine = _box_union([body, action])
            source_spine = _box_union([graph.units[body_index].box, graph.units[action_index].box])
            _, alignment = _pair(row_box, local_spine, axis, "center")
            _, source_alignment = _pair(source_row, source_spine, edge.axis, "center")
            before_growth = max(0.0, gap - edge.source_gap)
            after_growth = max(0.0, after_gap - source_after)
            drift = abs(alignment - source_alignment)
            own_cost = _ownership_cost(
                row_box, b, candidate["canvas"]["width"], candidate["canvas"]["height"]
            )[0]
            competing = [
                _ownership_cost(
                    row_box, box, candidate["canvas"]["width"], candidate["canvas"]["height"]
                )[0]
                for name, box in message_boxes.items()
                if name != edge.target_cluster_id
            ]
            crossed = bool(competing and min(competing) + 0.12 < own_cost)
            fidelity = (
                math.exp(-before_growth - after_growth - drift * 2)
                if ordered and not crossed
                else 0.0
            )
            lost = not ordered or crossed or drift > 0.3 or max(before_growth, after_growth) > 0.6
            message = (
                "A peer row no longer stays between its supporting copy "
                "and locally associated action."
            )
        else:  # campaign_ownership
            source_message = cluster_map[edge.target_cluster_id]
            source_box = _box_union(
                [graph.units[index].box for index in source_message.unit_indices]
            )
            source_start, source_size = _primary(first.box, edge.axis)
            source_after, source_after_size = _primary(source_box, edge.axis)
            source_direction = source_after + source_after_size / 2 - source_start - source_size / 2
            start, size = _primary(a, axis)
            after, after_size = _primary(b, axis)
            direction = after + after_size / 2 - start - size / 2
            order_lost = source_direction * direction < -0.5
            growth = max(0.0, gap - edge.source_gap)
            drift = abs(alignment - edge.source_alignment)
            own_cost = _ownership_cost(
                a, b, candidate["canvas"]["width"], candidate["canvas"]["height"]
            )[0]
            competing = [
                _ownership_cost(
                    a, box, candidate["canvas"]["width"], candidate["canvas"]["height"]
                )[0]
                for name, box in message_boxes.items()
                if name != edge.target_cluster_id
            ] + [
                _ownership_cost(
                    boxes[index], b, candidate["canvas"]["width"], candidate["canvas"]["height"]
                )[0]
                for index in products
                if index != first.index
            ]
            crossed = bool(competing and min(competing) + 0.12 < own_cost)
            fidelity = math.exp(-growth - drift * 2) if not (order_lost or crossed) else 0.0
            lost = order_lost or crossed or growth > 0.9 or drift > 0.35
            message = (
                "A product lost its locally associated message or became closer to another offer."
            )
        values[edge.code].append(fidelity)
        target_ids = (
            cluster_map[edge.target_cluster_id].element_ids
            if edge.target_cluster_id
            else second.element_ids
        )
        source_ids = (
            cluster_map[edge.cluster_id].element_ids
            if edge.code == "row_attachment"
            else first.element_ids
        )
        affected = tuple(dict.fromkeys(source_ids + target_ids))
        relations.append(
            {
                "code": edge.code,
                "cluster_id": edge.cluster_id,
                "element_ids": list(affected),
                "axis": flow,
                "gap": round(gap, 6),
                "alignment": round(alignment, 6),
                "score": round(fidelity, 6),
                "confidence": edge.confidence,
            }
        )
        if lost:
            warnings.append(
                {
                    "code": edge.code,
                    "message": message,
                    "element_ids": list(affected),
                    "cluster_id": edge.cluster_id,
                    "confidence": edge.confidence,
                    "provenance": list(edge.provenance),
                }
            )
    metrics = {
        code: round(sum(measured) / len(measured), 6) if measured else 1.0
        for code, measured in values.items()
    }
    weighted = [(row["score"], row["confidence"]) for row in relations]
    total = (
        sum(value * confidence for value, confidence in weighted)
        / sum(confidence for _, confidence in weighted)
        if weighted
        else 1.0
    )
    return {
        "score": round(total, 6),
        "warnings": warnings,
        "metrics": metrics,
        "relations": relations,
        "ambiguities": [asdict(ambiguity) for ambiguity in graph.ambiguities],
    }
