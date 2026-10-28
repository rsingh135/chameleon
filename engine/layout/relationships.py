"""Immutable source relationships for experiments in coherent graphic reflow.

Geometry is evidence for a relationship, not proof of human meaning. Explicit
groups are strong; inferred message and brand clusters retain their confidence
and provenance. Candidates are measured against the original graph only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .appearance import unit_visible_box
from .geometry import Box, bounds, union
from .model import build_context

_CODES = (
    "reading_order",
    "message_attachment",
    "shared_axis",
    "brand_association",
    "explicit_group",
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


@dataclass(frozen=True)
class SourceRelationshipGraph:
    width: float
    height: float
    units: tuple[RelationshipUnit, ...]
    edges: tuple[RelationshipEdge, ...]
    clusters: tuple[RelationshipCluster, ...]
    source_ids: tuple[str, ...]
    assets_dir: str | None


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
    assets = str(Path(assets_dir).resolve()) if assets_dir is not None else None
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
    clusters, edges = [], []

    def connect(code, first, second, cluster, axis, alignment, confidence, provenance):
        gap, offset = _pair(units[first].box, units[second].box, axis, alignment)
        edges.append(
            RelationshipEdge(
                code, first, second, cluster, axis, alignment, gap, offset, confidence, provenance
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
        for action in (
            unit for unit in units if unit.role == "action" and unit.index not in claimed
        ):
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

    for logo in (unit for unit in units if unit.role == "logo"):
        hypotheses = []
        for hero in (unit for unit in units if unit.role == "hero"):
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

    return SourceRelationshipGraph(
        context.width, context.height, units, tuple(edges), tuple(clusters), ids, assets
    )


def analyze_relationships(
    graph: SourceRelationshipGraph, candidate: dict, state: dict[str, str] | None = None
) -> dict:
    """Measure original relationships under an explicitly declared cluster flow.

    ``state`` optionally maps a cluster ID to ``vertical`` or ``horizontal``.
    A complete block may move/scale without penalty; changing its flow is an
    explicit state change, never an interpretation inferred from the candidate.
    Structural and immutable appearance validation remains the caller's job.
    """
    by_id = {element["id"]: element for element in candidate["elements"]}
    if set(by_id) != set(graph.source_ids) or len(by_id) != len(candidate["elements"]):
        raise ValueError("Relationship analysis requires every original element exactly once")
    state = state or {}
    cluster_ids = {cluster.id for cluster in graph.clusters}
    if any(
        key not in cluster_ids or value not in {"vertical", "horizontal"}
        for key, value in state.items()
    ):
        raise ValueError("Relationship states must map known cluster IDs to vertical or horizontal")
    boxes = {
        unit.index: unit_visible_box([by_id[eid] for eid in unit.element_ids], graph.assets_dir)
        for unit in graph.units
    }
    values = {code: [] for code in _CODES}
    warnings, relations = [], []
    for edge in graph.edges:
        first, second = graph.units[edge.source_unit], graph.units[edge.target_unit]
        axis = state.get(edge.cluster_id, edge.axis)
        a, b = boxes[first.index], boxes[second.index]
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
        else:  # Brand ownership is a soft hypothesis, not an identity constraint.
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
        values[edge.code].append(fidelity)
        affected = tuple(dict.fromkeys(first.element_ids + second.element_ids))
        relations.append(
            {
                "code": edge.code,
                "cluster_id": edge.cluster_id,
                "element_ids": list(affected),
                "axis": axis,
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
    }
