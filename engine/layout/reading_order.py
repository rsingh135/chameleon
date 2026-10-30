"""Source-first spatial reading evidence, without copy or role heuristics.

Proven source containers establish the outer order of repeated cards; each
card's immutable text members are ordered inside that card. Other occupied text
rectangles support recursive whitespace cuts. A vertical gutter
with vertically overlapping text on both sides reads as columns; a full-width
heading/footer can instead split above/below that column block. Competing cuts
and overlapping/rotated/cropped text are reported as uncertain. These geometric
conventions are evidence about reading order, not a claim to infer semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import median

from engine.resources import get_resources, using_resources

from .appearance import visible_box
from .geometry import Box, bounds, intersection
from .source_containers import SourceContainerProfile, infer_source_containers

MAX_TEXT_LEAVES = 256
MAX_REPORTED_PAIRS = 32


@dataclass(frozen=True)
class ReadingLeaf:
    element_id: str
    box: Box


@dataclass(frozen=True)
class ReadingRelation:
    before: str
    after: str
    provenance: str


@dataclass(frozen=True)
class ReadingOrderProfile:
    text_ids: tuple[str, ...]
    leaves: tuple[ReadingLeaf, ...]
    sequence: tuple[str, ...]
    relations: tuple[ReadingRelation, ...]
    ambiguous_pairs: tuple[tuple[str, str], ...]
    excluded: tuple[tuple[str, str], ...]
    row_groups: tuple[tuple[str, ...], ...]
    assets_dir: str
    fonts_dir: str
    containers: SourceContainerProfile | None = None


@dataclass(frozen=True)
class _Order:
    sequence: tuple[int, ...]
    relations: tuple[tuple[int, int, str], ...]
    ambiguous: tuple[tuple[int, int], ...]
    rows: tuple[tuple[int, ...], ...]


def _text_elements(design):
    elements = [element for element in design["elements"] if element.get("type") == "text"]
    ids = [element["id"] for element in elements]
    if len(ids) != len(set(ids)):
        raise ValueError("Reading-order analysis requires unique text element IDs.")
    return elements


def _leaves(design, allowed=None):
    width, height = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    leaves, excluded = [], []
    elements = [
        element for element in _text_elements(design) if allowed is None or element["id"] in allowed
    ]
    if len(elements) > MAX_TEXT_LEAVES:
        return (), tuple(
            (element["id"], "bounded reading-order text budget exceeded")
            for element in elements[:MAX_REPORTED_PAIRS]
        )
    for element in elements:
        eid = element["id"]
        if not str(element.get("content", "")).strip() or float(element.get("opacity", 1)) <= 0:
            excluded.append((eid, "no visible text"))
            continue
        rotation = (float(element.get("rotation", 0) or 0) + 180) % 360 - 180
        if abs(rotation) > 1:
            excluded.append((eid, "rotated text has ambiguous spatial reading order"))
            continue
        try:
            box = visible_box(element)
        except (LookupError, ValueError, OSError):
            excluded.append((eid, "exact occupied text measurement unavailable"))
            continue
        if not all(math.isfinite(value) for value in (box.x, box.y, box.width, box.height)):
            excluded.append((eid, "nonfinite text geometry"))
            continue
        area = box.width * box.height
        inside = max(0, min(width, box.x + box.width) - max(0, box.x)) * max(
            0, min(height, box.y + box.height) - max(0, box.y)
        )
        if area <= 0 or inside < area * 0.95:
            excluded.append((eid, "text is substantially outside the source/candidate viewport"))
            continue
        leaves.append(ReadingLeaf(eid, box))
    return tuple(leaves), tuple(excluded)


def _cut(indices, boxes, axis, epsilon):
    """Largest genuine whitespace separator; equal gaps prefer balanced cuts."""
    start = (lambda i: boxes[i].x) if axis == "x" else (lambda i: boxes[i].y)
    end = (
        (lambda i: boxes[i].x + boxes[i].width)
        if axis == "x"
        else (lambda i: boxes[i].y + boxes[i].height)
    )
    ordered = sorted(indices, key=lambda i: (start(i), end(i)))
    reach, options = end(ordered[0]), []
    for position, index in enumerate(ordered[1:], 1):
        gap = start(index) - reach
        if gap > epsilon:
            options.append((gap, min(position, len(ordered) - position), position))
        reach = max(reach, end(index))
    if not options:
        return None
    gap, _, position = max(options)
    return tuple(ordered[:position]), tuple(ordered[position:]), gap


def _span(indices, boxes):
    return min(boxes[i].y for i in indices), max(boxes[i].y + boxes[i].height for i in indices)


def _column_evidence(cut, boxes):
    a, b, _ = cut
    ay0, ay1 = _span(a, boxes)
    by0, by1 = _span(b, boxes)
    overlap = min(ay1, by1) - max(ay0, by0)
    return len(a) >= 2 and len(b) >= 2 and overlap >= 0.3 * min(ay1 - ay0, by1 - by0)


def _fallback(indices, boxes):
    """Source rows use common vertical support and disjoint horizontal ink."""
    rows = []
    for index in sorted(indices, key=lambda i: (boxes[i].y, boxes[i].x)):
        box = boxes[index]
        matched = None
        for row in rows:
            if all(
                min(box.y + box.height, boxes[j].y + boxes[j].height) - max(box.y, boxes[j].y)
                >= 0.5 * min(box.height, boxes[j].height)
                and (
                    box.x + box.width <= boxes[j].x + 0.02 * min(box.width, boxes[j].width)
                    or boxes[j].x + boxes[j].width <= box.x + 0.02 * min(box.width, boxes[j].width)
                )
                for j in row
            ):
                matched = row
                break
        if matched is None:
            rows.append([index])
        else:
            matched.append(index)
    rows.sort(key=lambda row: min(boxes[i].y for i in row))
    rows = [tuple(sorted(row, key=lambda i: boxes[i].x)) for row in rows]
    sequence = tuple(i for row in rows for i in row)
    relations, ambiguous = [], []
    row_of = {index: row_index for row_index, row in enumerate(rows) for index in row}
    for position, first in enumerate(sequence):
        for second in sequence[position + 1 :]:
            a, b = boxes[first], boxes[second]
            if row_of[first] == row_of[second]:
                relations.append((first, second, "common occupied source row, left to right"))
                continue
            overlap_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
            if b.y >= a.y + a.height or (
                overlap_x >= 0.25 * min(a.width, b.width)
                and b.y - a.y >= 0.3 * min(a.height, b.height)
            ):
                relations.append((first, second, "distinct occupied text tops in a vertical flow"))
            else:
                ambiguous.append((first, second))
    return _Order(sequence, tuple(relations), tuple(ambiguous), tuple(rows))


def _infer(leaves):
    boxes = tuple(leaf.box for leaf in leaves)
    if not boxes:
        return _Order((), (), (), ())
    epsilon = median(box.height for box in boxes) * 0.12

    def walk(indices, depth=0):
        if len(indices) == 1:
            return _Order(indices, (), (), (indices,))
        if depth >= 48:
            return _fallback(indices, boxes)
        xcut = _cut(indices, boxes, "x", epsilon)
        ycut = _cut(indices, boxes, "y", epsilon)
        if not xcut and not ycut:
            return _fallback(indices, boxes)
        axis = "x" if xcut and (not ycut or _column_evidence(xcut, boxes)) else "y"
        chosen, alternate = (xcut, ycut) if axis == "x" else (ycut, xcut)
        a, b, _ = chosen
        left, right = walk(a, depth + 1), walk(b, depth + 1)
        relations = list(left.relations + right.relations)
        ambiguous = list(left.ambiguous + right.ambiguous)
        alternate_first = set(alternate[0]) if alternate else set()
        alternate_last = set(alternate[1]) if alternate else set()
        for first in a:
            for second in b:
                if first in alternate_last and second in alternate_first:
                    ambiguous.append((first, second))
                else:
                    relations.append(
                        (
                            first,
                            second,
                            (
                                "full vertical whitespace gutter, column/row order"
                                if axis == "x"
                                else "full horizontal whitespace separator, top to bottom"
                            ),
                        )
                    )
        rows = left.rows + right.rows
        if axis == "x" and (
            min(boxes[i].y + boxes[i].height for i in indices) - max(boxes[i].y for i in indices)
            >= 0.5 * min(boxes[i].height for i in indices)
        ):
            rows = (left.sequence + right.sequence,)
        return _Order(
            left.sequence + right.sequence,
            tuple(relations),
            tuple(ambiguous),
            rows,
        )

    return walk(tuple(range(len(leaves))))


def _container_spatial_order(boxes):
    """Read complete frames by source-style x lanes, then y within each lane."""
    lanes = []
    for index in sorted(range(len(boxes)), key=lambda i: boxes[i].x):
        if (
            lanes
            and abs(boxes[index].x - boxes[lanes[-1][0]].x)
            < min(boxes[index].width, boxes[lanes[-1][0]].width) * 0.2
        ):
            lanes[-1].append(index)
        else:
            lanes.append([index])
    lanes = [sorted(lane, key=lambda i: boxes[i].y) for lane in lanes]
    sequence = tuple(index for lane in lanes for index in lane)
    lane_of = {index: number for number, lane in enumerate(lanes) for index in lane}
    relations, ambiguous = [], []
    for position, first in enumerate(sequence):
        for second in sequence[position + 1 :]:
            a, b = boxes[first], boxes[second]
            clear = (
                a.y + a.height <= b.y + 1e-6
                if lane_of[first] == lane_of[second]
                else a.x + a.width <= b.x + 1e-6
            )
            if clear:
                relations.append(
                    (first, second, "whole source-owned containers, lane then vertical order")
                )
            else:
                ambiguous.append((first, second))
    return _Order(sequence, tuple(relations), tuple(ambiguous), ())


def _hierarchical_order(design, leaves, containers):
    """Use immutable source ownership, never rediscover target card membership."""
    by_id = {element["id"]: element for element in design["elements"]}
    frame_ids = {container.frame_id for container in containers.containers}
    if sum(element["id"] in frame_ids for element in design["elements"]) != len(frame_ids):
        raise ValueError("Reading-order source container frames must occur exactly once.")
    groups, boxes, claimed = [], [], set()
    for container in containers.containers:
        if container.frame_id not in by_id:
            raise ValueError("Reading-order source container frame is missing from candidate.")
        members = tuple(
            index for index, leaf in enumerate(leaves) if leaf.element_id in container.member_ids
        )
        if claimed.intersection(members):
            raise ValueError("Reading-order source containers have ambiguous text ownership.")
        claimed.update(members)
        groups.append(members)
        boxes.append(bounds(by_id[container.frame_id]))
    count = len(groups)
    if count < 2:
        return _infer(leaves)
    groups.extend((index,) for index in range(len(leaves)) if index not in claimed)
    boxes.extend(leaves[group[0]].box for group in groups[count:])
    outer = _infer(tuple(ReadingLeaf(str(index), box) for index, box in enumerate(boxes)))
    parents = _container_spatial_order(boxes[:count])
    parent_iter = iter(parents.sequence)
    block_sequence = tuple(
        next(parent_iter) if index < count else index for index in outer.sequence
    )
    positions = {index: position for position, index in enumerate(block_sequence)}
    inner = [_infer(tuple(leaves[index] for index in group)) for group in groups]
    sequence = tuple(
        groups[block][index] for block in block_sequence for index in inner[block].sequence
    )
    relations, ambiguous, rows = [], [], []
    for group, order in zip(groups, inner, strict=True):
        relations.extend((group[a], group[b], evidence) for a, b, evidence in order.relations)
        ambiguous.extend((group[a], group[b]) for a, b in order.ambiguous)
        rows.extend(tuple(group[index] for index in row) for row in order.rows)
    outer_relations = list(parents.relations) + [
        relation for relation in outer.relations if relation[0] >= count or relation[1] >= count
    ]
    outer_ambiguous = list(parents.ambiguous) + [
        pair for pair in outer.ambiguous if pair[0] >= count or pair[1] >= count
    ]
    for a, b, evidence in outer_relations:
        if positions[a] > positions[b]:
            # Competing loose-text and container flow evidence stays uncertain.
            outer_ambiguous.append((a, b))
            continue
        relations.extend((first, second, evidence) for first in groups[a] for second in groups[b])
    ambiguous.extend(
        (first, second) for a, b in outer_ambiguous for first in groups[a] for second in groups[b]
    )
    rows.extend(
        tuple(groups[node][0] for node in row)
        for row in outer.rows
        if len(row) > 1 and all(node >= count for node in row)
    )
    return _Order(sequence, tuple(relations), tuple(ambiguous), tuple(rows))


def _container_detachments(design, leaves, profile):
    if profile.containers is None:
        return []
    originals = {leaf.element_id: leaf.box for leaf in profile.leaves}
    changed = {leaf.element_id: leaf.box for leaf in leaves}
    by_id = {element["id"]: element for element in design["elements"]}
    result = []
    for container in profile.containers.containers:
        frame = bounds(by_id[container.frame_id])
        for eid in container.member_ids:
            if eid not in originals or eid not in changed:
                continue
            before, after = originals[eid], changed[eid]
            source_coverage = intersection(before, container.box) / max(
                before.width * before.height, 1e-9
            )
            coverage = intersection(after, frame) / max(after.width * after.height, 1e-9)
            if coverage < min(0.9, source_coverage - 0.02):
                result.append(
                    {
                        "frame_id": container.frame_id,
                        "element_id": eid,
                        "source_coverage": source_coverage,
                        "candidate_coverage": coverage,
                    }
                )
    return result


def infer_reading_order(source: dict) -> ReadingOrderProfile:
    """Freeze occupied geometry and proven container ownership from the source."""
    resources = get_resources()
    elements = _text_elements(source)
    leaves, excluded = _leaves(source)
    # The same text budget also bounds hierarchy discovery and font work.
    containers = (
        infer_source_containers(source) if leaves and len(elements) <= MAX_TEXT_LEAVES else None
    )
    eligible = {leaf.element_id for leaf in leaves}
    records = tuple(
        container
        for container in (containers.containers if containers is not None else ())
        if eligible.intersection(container.member_ids)
    )
    containers = replace(containers, containers=records) if len(records) >= 2 else None
    order = _hierarchical_order(source, leaves, containers) if containers else _infer(leaves)

    def eid(index):
        return leaves[index].element_id

    return ReadingOrderProfile(
        tuple(element["id"] for element in elements),
        leaves,
        tuple(map(eid, order.sequence)),
        tuple(ReadingRelation(eid(a), eid(b), evidence) for a, b, evidence in order.relations),
        tuple((eid(a), eid(b)) for a, b in order.ambiguous),
        excluded,
        tuple(tuple(map(eid, row)) for row in order.rows if len(row) > 1),
        str(resources.assets_dir),
        str(resources.fonts_dir),
        containers,
    )


def analyze_reading_order(
    source: dict, candidate: dict, profile: ReadingOrderProfile | None = None
):
    """Count reversals supported by both source and candidate spatial evidence.

    ``high_confidence_inversions`` is an integer; ``inversion_pairs`` contains
    bounded explanatory records. Uncertain pairs do not become hard inversions.
    All text leaf IDs must still occur exactly once before exclusions are applied.
    """
    profile = infer_reading_order(source) if profile is None else profile
    ids = tuple(element["id"] for element in _text_elements(candidate))
    if set(ids) != set(profile.text_ids):
        raise ValueError("Reading-order comparison requires the same text leaf membership.")
    with using_resources(profile.assets_dir, profile.fonts_dir):
        leaves, excluded = _leaves(candidate, {leaf.element_id for leaf in profile.leaves})
        order = (
            _hierarchical_order(candidate, leaves, profile.containers)
            if profile.containers
            else _infer(leaves)
        )

    detachments = _container_detachments(candidate, leaves, profile)

    def eid(index):
        return leaves[index].element_id

    candidate_sequence = tuple(map(eid, order.sequence))
    reverse = {(eid(a), eid(b)): evidence for a, b, evidence in order.relations}
    available = set(candidate_sequence)
    compared = [
        relation
        for relation in profile.relations
        if relation.before in available and relation.after in available
    ]
    inversions = [
        {
            "before": relation.before,
            "after": relation.after,
            "source_evidence": relation.provenance,
            "candidate_evidence": reverse[(relation.after, relation.before)],
        }
        for relation in compared
        if (relation.after, relation.before) in reverse
    ]
    resolved = sum(
        (relation.before, relation.after) in reverse or (relation.after, relation.before) in reverse
        for relation in compared
    )
    uncertain = tuple(profile.ambiguous_pairs) + tuple((eid(a), eid(b)) for a, b in order.ambiguous)
    warnings = ["source_reading_order_inversion"] if inversions else []
    if uncertain or profile.excluded or excluded:
        warnings.append("reading_order_uncertain")
    if detachments:
        warnings.append("source_container_detachment")
    return {
        "active": bool(compared),
        "score": 1 - len(inversions) / max(1, len(compared)),
        "high_confidence_inversions": len(inversions),
        "inversion_pairs": inversions[:MAX_REPORTED_PAIRS],
        "compared_pairs": len(compared),
        "source_pairs": len(profile.relations),
        "resolved_pairs": resolved,
        "comparison_coverage": resolved / max(1, len(profile.relations)),
        "source_sequence": list(profile.sequence),
        "candidate_sequence": list(candidate_sequence),
        "source_row_groups": [list(row) for row in profile.row_groups],
        "container_order_active": bool(profile.containers),
        "container_detachments": detachments[:MAX_REPORTED_PAIRS],
        "source_containers": [
            {
                "frame_id": container.frame_id,
                "member_ids": list(container.member_ids),
                "confidence": container.confidence,
                "provenance": list(container.provenance),
            }
            for container in profile.containers.containers
        ]
        if profile.containers
        else [],
        "ambiguity_count": len(uncertain),
        "ambiguous_pairs": [list(pair) for pair in uncertain[:MAX_REPORTED_PAIRS]],
        "excluded_elements": [
            {"element_id": eid, "reason": reason, "stage": stage}
            for stage, exclusions in (("source", profile.excluded), ("candidate", excluded))
            for eid, reason in exclusions
        ][:MAX_REPORTED_PAIRS],
        "warnings": warnings,
    }
