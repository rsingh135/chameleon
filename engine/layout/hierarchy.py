"""Immutable source blocks for similarity-preserving layout alternatives.

Existing relationship clusters remain hypotheses with their original evidence.
Only explicit units are hard groups. Conservative complete-link agglomeration
can suggest local blocks among otherwise unowned foreground units; similarity
across different campaigns never establishes a shared spatial parent.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .appearance import unit_visible_box
from .geometry import Box, union
from .model import build_context
from .relationships import SourceRelationshipGraph, build_relationship_graph


@dataclass(frozen=True)
class LocalMember:
    """Element frame in its cluster's normalized, unrotated local coordinates."""

    element_id: str
    x: float
    y: float
    width: float
    height: float
    font_size: float | None


@dataclass(frozen=True)
class LayoutCluster:
    id: str
    kind: str
    unit_indices: tuple[int, ...]
    element_ids: tuple[str, ...]
    frame_box: Box
    visible_box: Box
    children: tuple[str, ...]
    confidence: float
    provenance: tuple[str, ...]
    members: tuple[LocalMember, ...]
    elements_json: str

    @property
    def elements(self) -> list[dict]:
        """A fresh copy of original elements, never a previous fitted result."""
        return json.loads(self.elements_json)


@dataclass(frozen=True)
class SourceHierarchy:
    source_digest: str
    width: float
    height: float
    clusters: tuple[LayoutCluster, ...]
    root_ids: tuple[str, ...]
    background_ids: tuple[str, ...]
    decoration_ids: tuple[str, ...]

    def cluster(self, cluster_id: str) -> LayoutCluster:
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        raise KeyError(cluster_id)

    @property
    def roots(self) -> tuple[LayoutCluster, ...]:
        return tuple(self.cluster(cluster_id) for cluster_id in self.root_ids)


@dataclass(frozen=True)
class ClusterPlacement:
    scale: float
    elements_json: str

    @property
    def elements(self) -> list[dict]:
        return json.loads(self.elements_json)


def _cluster(source, graph, name, kind, indices, children, confidence, provenance):
    indices = tuple(sorted(indices))
    ids = {eid for index in indices for eid in graph.units[index].element_ids}
    elements = [element for element in source["elements"] if element["id"] in ids]
    frame = union(elements)
    visible = unit_visible_box(elements, graph.assets_dir)
    members = tuple(
        LocalMember(
            element["id"],
            (element["x"] - frame.x) / frame.width,
            (element["y"] - frame.y) / frame.height,
            element["width"] / frame.width,
            element["height"] / frame.height,
            element["font_size"] / frame.height if element["type"] == "text" else None,
        )
        for element in elements
    )
    return LayoutCluster(
        name,
        kind,
        indices,
        tuple(element["id"] for element in elements),
        frame,
        visible,
        tuple(children),
        confidence,
        tuple(provenance),
        members,
        json.dumps(elements, sort_keys=True),
    )


def _residual_pair(first, second, short):
    """Dimensionless local adjacency; deliberately no semantic/name matching."""
    if "legal" in {first.role, second.role}:
        return None
    # Repeated products are an editing family, not automatically one rigid block.
    if all(member.element_type == "image" for member in first.members + second.members):
        return None
    a, b = first.box, second.box
    if min(a.width, a.height, b.width, b.height) <= 0:
        return None
    candidates = []
    for axis in ("vertical", "horizontal"):
        if axis == "vertical":
            p, q, sa, sb = a.y, b.y, a.height, b.height
            cross_a, cross_b, ca, cb = a.x, b.x, a.width, b.width
        else:
            p, q, sa, sb = a.x, b.x, a.width, b.width
            cross_a, cross_b, ca, cb = a.y, b.y, a.height, b.height
        gap = max(0.0, q - p - sa, p - q - sb)
        overlap = max(0.0, min(cross_a + ca, cross_b + cb) - max(cross_a, cross_b))
        alignment = min(
            abs((cross_a + ca * fraction) - (cross_b + cb * fraction))
            for fraction in (0.0, 0.5, 1.0)
        ) / max(ca, cb)
        if overlap / min(ca, cb) < 0.5 or alignment > 0.08:
            continue
        normalized_gap = gap / max(min(sa, sb), short * 0.03)
        if normalized_gap <= 0.75:
            candidates.append(normalized_gap + alignment)
    return min(candidates) if candidates else None


def _residual_groups(indices, graph):
    """Complete-link prevents a chain of nearby objects joining separate offers."""
    groups = [{index} for index in indices]
    costs = {}
    for index, first in enumerate(indices):
        for second in indices[index + 1 :]:
            costs[frozenset((first, second))] = _residual_pair(
                graph.units[first], graph.units[second], min(graph.width, graph.height)
            )
    while True:
        choices = []
        for i, first in enumerate(groups):
            for j in range(i + 1, len(groups)):
                second = groups[j]
                pair_costs = [costs[frozenset((a, b))] for a in first for b in second]
                if any(cost is None for cost in pair_costs):
                    continue
                units = [graph.units[index] for index in first | second]
                left, top = min(unit.box.x for unit in units), min(unit.box.y for unit in units)
                right = max(unit.box.x + unit.box.width for unit in units)
                bottom = max(unit.box.y + unit.box.height for unit in units)
                area = sum(unit.box.width * unit.box.height for unit in units)
                if (right - left) * (bottom - top) > area * 2.25:
                    continue
                choices.append((max(pair_costs), tuple(sorted(first | second)), i, j))
        if not choices:
            break
        _, _, i, j = min(choices)
        groups[i] |= groups.pop(j)
    return groups


def infer_hierarchy(
    source: dict,
    graph: SourceRelationshipGraph | None = None,
    assets_dir: str | Path | None = None,
) -> SourceHierarchy:
    """Build a laminar source hierarchy without changing supplied graph evidence.

    Campaign membership prevents any cross-campaign spatial merge. Partial
    overlaps between inferred clusters are declined rather than joining their
    complete connected component. Inference confidence is heuristic, not a
    calibrated probability or proof of the source's meaning.
    """
    graph = graph or build_relationship_graph(source, assets_dir)
    if tuple(element["id"] for element in source["elements"]) != graph.source_ids or (
        source["canvas"]["width"],
        source["canvas"]["height"],
    ) != (graph.width, graph.height):
        raise ValueError("Hierarchy graph must belong to the original source")
    context = build_context(source)
    nodes = {}
    for unit in graph.units:
        explicit = any(
            edge.code == "explicit_group" and edge.source_unit == unit.index for edge in graph.edges
        )
        name = f"unit:{unit.index}"
        nodes[name] = _cluster(
            source,
            graph,
            name,
            "explicit" if explicit else "unit",
            (unit.index,),
            (),
            1.0,
            ("explicit source group_id",) if explicit else ("atomic source unit",),
        )

    accepted = []
    priority = {"campaign": 0, "message": 1, "badge_row": 2, "brand": 3}
    for candidate in sorted(graph.clusters, key=lambda item: (priority.get(item.kind, 4), item.id)):
        if candidate.kind not in priority or candidate.confidence < (
            0.7 if candidate.kind == "brand" else 0.8
        ):
            continue
        members = set(candidate.unit_indices)
        if len(members) < 2:
            continue
        if any(
            members & set(other.unit_indices)
            and not members < set(other.unit_indices)
            and not set(other.unit_indices) < members
            for other in accepted
        ):
            continue
        accepted.append(candidate)

    for candidate in sorted(accepted, key=lambda item: (len(item.unit_indices), item.id)):
        members = set(candidate.unit_indices)
        candidates = [node for node in nodes.values() if set(node.unit_indices) < members]
        children, covered = [], set()
        for node in sorted(candidates, key=lambda item: (-len(item.unit_indices), item.id)):
            if not set(node.unit_indices) & covered:
                children.append(node.id)
                covered.update(node.unit_indices)
        children.sort(key=lambda key: (nodes[key].frame_box.y, nodes[key].frame_box.x))
        nodes[candidate.id] = _cluster(
            source,
            graph,
            candidate.id,
            candidate.kind,
            candidate.unit_indices,
            children,
            candidate.confidence,
            candidate.provenance,
        )

    owned = {index for candidate in accepted for index in candidate.unit_indices}
    residuals = [unit.index for unit in graph.units if unit.index not in owned]
    for index, group in enumerate(_residual_groups(residuals, graph)):
        if len(group) < 2:
            continue
        name = f"spatial:{index}"
        nodes[name] = _cluster(
            source,
            graph,
            name,
            "spatial",
            group,
            tuple(f"unit:{unit}" for unit in sorted(group)),
            0.55,
            (
                "unowned foreground units only",
                "complete-link local gap and common alignment",
                "compactness bounded; geometric hypothesis only",
            ),
        )

    child_ids = {child for node in nodes.values() for child in node.children}
    roots = sorted(
        (node for name, node in nodes.items() if name not in child_ids),
        key=lambda item: (item.frame_box.y, item.frame_box.x, item.id),
    )
    return SourceHierarchy(
        hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        graph.width,
        graph.height,
        tuple(nodes.values()),
        tuple(node.id for node in roots),
        tuple(element["id"] for element in context.backgrounds),
        tuple(element["id"] for element in context.decorations),
    )


def transform_cluster(
    cluster: LayoutCluster,
    scale: float,
    x: float,
    y: float,
    *,
    anchor: str = "frame",
) -> list[dict]:
    """Place one original block with a single similarity transform.

    ``x,y`` are the destination top-left of the chosen frame/visible anchor.
    Font sizes, control radii, strokes, frame ratios and all internal gaps share
    exactly one scale; there is no text refit or minimum-size clamp. The caller
    must reject unreadable/subminimum candidates rather than distort the block.
    """
    if anchor not in {"frame", "visible"}:
        raise ValueError("Cluster anchor must be 'frame' or 'visible'")
    if not all(math.isfinite(value) for value in (scale, x, y)) or scale <= 0:
        raise ValueError("Cluster placement requires a positive finite scale and finite position")
    origin = cluster.frame_box if anchor == "frame" else cluster.visible_box
    if origin.width <= 0 or origin.height <= 0:
        raise ValueError("An empty visible cluster cannot establish a placement anchor")
    elements = cluster.elements
    for element in elements:
        element["x"] = x + (element["x"] - origin.x) * scale
        element["y"] = y + (element["y"] - origin.y) * scale
        for key in ("width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in element:
                element[key] *= scale
    return elements


def fit_cluster(
    cluster: LayoutCluster,
    target: Box,
    *,
    alignment: tuple[str, str] = ("center", "center"),
    anchor: str = "frame",
    allow_upscale: bool = True,
) -> ClusterPlacement | None:
    """Uniformly contain an original block; preserve empty space and all ratios.

    Using visible bounds can place transparent frame padding outside ``target``;
    a realizer must still check full-frame canvas bounds. No independent member
    containment or clamping occurs here.
    """
    if anchor not in {"frame", "visible"}:
        raise ValueError("Cluster anchor must be 'frame' or 'visible'")
    fractions = {"start": 0.0, "center": 0.5, "end": 1.0}
    if len(alignment) != 2 or any(value not in fractions for value in alignment):
        raise ValueError("Cluster alignment needs horizontal/vertical start, center or end")
    if not all(math.isfinite(value) for value in (target.x, target.y, target.width, target.height)):
        raise ValueError("Cluster target must have finite geometry")
    box = cluster.frame_box if anchor == "frame" else cluster.visible_box
    if min(box.width, box.height, target.width, target.height) <= 0:
        return None
    scale = min(target.width / box.width, target.height / box.height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    x = target.x + (target.width - box.width * scale) * fractions[alignment[0]]
    y = target.y + (target.height - box.height * scale) * fractions[alignment[1]]
    elements = transform_cluster(cluster, scale, x, y, anchor=anchor)
    return ClusterPlacement(scale, json.dumps(elements, sort_keys=True))
