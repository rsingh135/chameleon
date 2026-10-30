"""Source-bound element purposes and relationships for semantic reflow.

Measured affinity is evidence, not visual understanding. Rule-derived purposes
are explicitly uncalibrated; optional model proposals remain a separate layer.
This module grants no crop, deletion, content, or group-breaking authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from statistics import median

POLICY_VERSION = "design-semantics-v1"
SOURCE_DIGEST_PROTOCOL = "canonical-compact-v1"
MAX_ELEMENTS = 256
MAX_RELATIONS = 1024
PURPOSES = frozenset(
    {
        "background",
        "brand_mark",
        "corner_marker",
        "headline",
        "primary_visual",
        "body",
        "supporting_copy",
        "footer_destination",
        "footer_legal",
        "action",
        "decoration",
        "unknown",
    }
)
PROPOSED_RELATIONS = frozenset(
    {"attached_to", "caption_for", "supports", "reading_before", "aligned_with", "same_group"}
)
PURPOSE_DESCRIPTIONS = {
    "background": "Establishes the canvas treatment behind the foreground content.",
    "brand_mark": "Identifies the source brand; retain recognition and authored affiliation.",
    "corner_marker": "Independent supporting artwork associated with an authored canvas corner.",
    "headline": "Carries the primary message and its prominence relative to supporting copy.",
    "primary_visual": "Carries the main artwork and any lettering contained inside it.",
    "body": "Communicates information associated with the source message.",
    "supporting_copy": "Provides supporting context; its specific purpose remains unconfirmed.",
    "footer_destination": "Provides a website or contact destination in the source footer.",
    "footer_legal": "Provides intact qualification or attribution associated with the design.",
    "action": "Provides an action or destination that must remain connected to its message.",
    "decoration": "Supports the composition as secondary artwork or ornament.",
    "unknown": "Purpose is unconfirmed; preserve the original element and request review.",
}
_DESTINATION = re.compile(r"(?:https?://|www\.)\S+|\b[^\s@]+@[^\s@]+\.[^\s@]+\b", re.I)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def semantic_source_digest(source):
    """Compact protocol, deliberately distinct from visual_plan.source_digest."""
    return _digest(source)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_source(source):
    _canonical(source)
    if not isinstance(source, dict) or not isinstance(source.get("canvas"), dict):
        raise ValueError("Semantic descriptions require a native source canvas")
    if any(
        not _finite(source["canvas"].get(k)) or source["canvas"][k] <= 0
        for k in ("width", "height")
    ):
        raise ValueError("Semantic source dimensions must be finite and positive")
    elements = source.get("elements")
    if not isinstance(elements, list) or not 1 <= len(elements) <= MAX_ELEMENTS:
        raise ValueError("Semantic source exceeds its element budget")
    ids = set()
    for element in elements:
        if (
            not isinstance(element, dict)
            or not isinstance(element.get("type"), str)
            or element["type"] not in {"text", "image", "shape"}
        ):
            raise ValueError("Semantic source elements must use known native types")
        eid = element.get("id")
        if not isinstance(eid, str) or not eid or eid in ids:
            raise ValueError("Semantic source IDs must be unique nonempty strings")
        ids.add(eid)
        if any(not _finite(element.get(k)) for k in ("x", "y", "width", "height")):
            raise ValueError("Semantic source geometry must be finite")
        if min(element["width"], element["height"]) <= 0:
            raise ValueError("Semantic source frames must have positive area")
        if element["type"] == "text" and (
            not _finite(element.get("font_size")) or element["font_size"] <= 0
        ):
            raise ValueError("Semantic text must have a finite positive font size")


def _box(element):
    return [float(element[k]) for k in ("x", "y", "width", "height")]


def _insets(box, width, height):
    x, y, w, h = box
    short = min(width, height)
    return {
        "left": x / short,
        "top": y / short,
        "right": (width - x - w) / short,
        "bottom": (height - y - h) / short,
    }


def _background(element, width, height):
    x, y, w, h = _box(element)
    tolerance = min(width, height) * 0.015
    return (
        element["type"] in {"shape", "image"}
        and w * h >= width * height * 0.85
        and x <= tolerance
        and y <= tolerance
        and x + w >= width - tolerance
        and y + h >= height - tolerance
    )


def _corner_edges(element, width, height):
    box = _box(element)
    if (
        element.get("group_id")
        or box[2] > width * 0.36
        or box[3] > height * 0.3
        or box[2] * box[3] > width * height * 0.1
    ):
        return []
    gaps = _insets(box, width, height)
    horizontal = min(("left", "right"), key=gaps.get)
    vertical = min(("top", "bottom"), key=gaps.get)
    if all(-0.01 <= gaps[edge] <= 0.15 for edge in (horizontal, vertical)):
        return [vertical, horizontal]
    return []


def _purpose(element, width, height, largest_font):
    declared = str(element.get("role", element.get("semantic_role", ""))).lower()
    aliases = {
        "logo": "brand_mark",
        "hero": "primary_visual",
        "product": "primary_visual",
        "legal": "footer_legal",
        "footer": "footer_destination",
        "ornament": "decoration",
    }
    declared = aliases.get(declared, declared)
    if declared in PURPOSES and declared != "unknown":
        return declared, 1.0, "source_declared_role"
    if _background(element, width, height):
        return "background", 0.95, "source_frame_covers_canvas"
    if element["type"] in {"image", "shape"} and _corner_edges(element, width, height):
        return "corner_marker", 0.8, "small_independent_artwork_near_two_source_edges"
    if element["type"] == "text":
        footer = element["y"] + element["height"] / 2 >= height * 0.82
        if _DESTINATION.search(str(element.get("content", ""))):
            return ("footer_destination" if footer else "action"), 0.9, "destination_copy"
        if footer and not element.get("group_id"):
            return "supporting_copy", 0.55, "bottom_position_only_meaning_unconfirmed"
        if element.get("font_size", 0) >= largest_font * 0.85:
            return "headline", 0.75, "source_font_rank"
        return "body", 0.5, "text_role_unconfirmed"
    if element["type"] == "image":
        return "primary_visual", 0.55, "independent_image_content_unconfirmed"
    return "unknown", 0.0, "purpose_requires_review"


def _ink(element, assets_dir, fonts_dir):
    if assets_dir is not None or fonts_dir is not None:
        try:
            from engine.semantic_ink import element_ink_bounds

            return element_ink_bounds(element, assets_dir=assets_dir, fonts_dir=fonts_dir)
        except (ImportError, ValueError, OSError, RuntimeError):
            pass
    return {
        "box": _box(element),
        "provenance": "frame_fallback",
        "confidence": 0.0,
        "raster_scale": 1.0,
    }


def describe_design(source, *, assets_dir=None, fonts_dir=None):
    """Describe every original element without modifying it or loading a model.

    Known author groups stay hard. Corner and footer roles inferred from source
    evidence are preferences, and retain their provenance and uncertainty.
    Resource paths opt into bounded native ink measurement: callers must own
    the engine's renderer lock or run this in an isolated single-worker process.
    """
    _validate_source(source)
    width, height = (float(source["canvas"][k]) for k in ("width", "height"))
    largest = max(
        (e.get("font_size", 0) for e in source["elements"] if e["type"] == "text"), default=1
    )
    rows, relations, groups = [], [], defaultdict(list)
    for element in source["elements"]:
        eid = element["id"]
        purpose, confidence, reason = _purpose(element, width, height, largest)
        corner = (
            _corner_edges(element, width, height)
            if purpose in {"corner_marker", "brand_mark", "decoration"}
            else []
        )
        footer = purpose in {"footer_destination", "footer_legal"}
        measured = (
            _ink(element, assets_dir, fonts_dir)
            if corner or footer
            else {
                "box": _box(element),
                "provenance": "frame",
                "confidence": 1.0,
                "raster_scale": 1.0,
            }
        )
        ink = list(measured["box"])
        gaps = _insets(ink, width, height)
        anchors = []
        if corner:
            anchors.append(
                {
                    "edges": corner,
                    "insets_short_axis": {e: gaps[e] for e in corner},
                    "confidence": 0.85,
                    "provenance": "source_corner_affinity",
                    "hard": False,
                }
            )
        if footer:
            center = (ink[0] + ink[2] / 2) / width
            horizontal = (
                "center"
                if abs(center - 0.5) <= 0.08
                else ("left" if gaps["left"] < gaps["right"] else "right")
            )
            anchors.append(
                {
                    "edges": ["bottom"],
                    "insets_short_axis": {"bottom": gaps["bottom"]},
                    "horizontal": horizontal,
                    "source_center_x": center,
                    "confidence": 0.9,
                    "provenance": "source_footer_affinity",
                    "hard": False,
                }
            )
        group = element.get("group_id")
        if group:
            groups[str(group)].append(eid)
        rows.append(
            {
                "element_id": eid,
                "type": element["type"],
                "purpose": purpose,
                "design_purpose": PURPOSE_DESCRIPTIONS[purpose],
                "purpose_confidence": confidence,
                "confidence_kind": "heuristic_uncalibrated",
                "provenance": reason,
                "source_box": _box(element),
                "visible_box": ink,
                "visible_box_provenance": measured["provenance"],
                "normalized_box": [
                    element["x"] / width,
                    element["y"] / height,
                    element["width"] / width,
                    element["height"] / height,
                ],
                "rigid_group": str(group) if group else None,
                "anchors": anchors,
                "atomic_asset": element["type"] == "image",
                "preservation_rules": ["preserve_original_identity"]
                + (
                    ["preserve_atomic_asset_contents_and_proportions"]
                    if element["type"] == "image"
                    else []
                )
                + (["transform_authored_group_together"] if group else [])
                + (["retain_source_corner_affinity"] if corner else [])
                + (["retain_footer_alignment_and_readability"] if footer else []),
                "relationships": [],
            }
        )
    for members in groups.values():
        for other in members[1:]:
            relations.append(
                {
                    "source_id": members[0],
                    "target_id": other,
                    "type": "same_group",
                    "confidence": 1.0,
                    "origin": "source_group_id",
                    "hard": True,
                }
            )
    ordered = sorted(
        (
            r
            for r in rows
            if r["purpose"] not in {"background", "corner_marker", "brand_mark", "decoration"}
        ),
        key=lambda r: (r["source_box"][1], r["source_box"][0]),
    )
    for first, second in zip(ordered, ordered[1:], strict=False):
        if (
            first["source_box"][1] + first["source_box"][3]
            <= second["source_box"][1] + height * 0.01
        ):
            relations.append(
                {
                    "source_id": first["element_id"],
                    "target_id": second["element_id"],
                    "type": "reading_before",
                    "axis": "vertical",
                    "confidence": 0.7,
                    "origin": "source_vertical_separation",
                    "hard": False,
                }
            )
    for row in rows:
        row["relationships"] = [
            deepcopy(r) for r in relations if row["element_id"] in (r["source_id"], r["target_id"])
        ]
    body = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "source_digest_protocol": SOURCE_DIGEST_PROTOCOL,
        "source_digest": semantic_source_digest(source),
        "source_canvas": deepcopy(source["canvas"]),
        "authority": "source_evidence_and_uncalibrated_rules",
        "model_executed": False,
        "elements": rows,
        "relations": relations,
        "limitations": [
            "Geometry alone does not establish visual design intent.",
            "No annotation grants crop, deletion, or content-edit authority.",
        ],
    }
    return {**body, "semantic_digest": _digest(body)}


def validate_semantic_annotations(source, proposal):
    """Validate a separate all-element model/reviewer layer; never trust geometry."""
    _validate_source(source)
    _canonical(proposal)
    required = {
        "schema_version",
        "source_digest_protocol",
        "source_digest",
        "elements",
        "relations",
    }
    if not isinstance(proposal, dict) or set(proposal) != required:
        raise ValueError("Semantic proposals have missing or unknown fields")
    if (
        type(proposal["schema_version"]) is not int
        or proposal["schema_version"] != 1
        or proposal["source_digest_protocol"] != SOURCE_DIGEST_PROTOCOL
        or proposal["source_digest"] != semantic_source_digest(source)
    ):
        raise ValueError("Semantic proposal belongs to another source or protocol")
    ids = {e["id"] for e in source["elements"]}
    rows = proposal["elements"]
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError("A semantic proposal must describe every source element exactly once")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "element_id",
            "purpose",
            "confidence",
            "evidence",
        }:
            raise ValueError("Semantic purpose fields are closed")
        eid = row["element_id"]
        if not isinstance(eid, str) or eid not in ids or eid in seen:
            raise ValueError("Semantic proposal contains duplicate or unknown source IDs")
        seen.add(eid)
        if not isinstance(row["purpose"], str) or row["purpose"] not in PURPOSES:
            raise ValueError("Semantic purpose is outside the closed vocabulary")
        _validate_confidence_evidence(row)
    relations = proposal["relations"]
    if not isinstance(relations, list) or len(relations) > MAX_RELATIONS:
        raise ValueError("Semantic proposal exceeds its sparse relationship budget")
    edge_keys, order = set(), defaultdict(list)
    for relation in relations:
        if not isinstance(relation, dict) or set(relation) != {
            "source_id",
            "target_id",
            "type",
            "confidence",
            "evidence",
        }:
            raise ValueError("Semantic relationship fields are closed")
        first, second, kind = (relation[k] for k in ("source_id", "target_id", "type"))
        if (
            not isinstance(first, str)
            or not isinstance(second, str)
            or first not in ids
            or second not in ids
            or first == second
            or not isinstance(kind, str)
            or kind not in PROPOSED_RELATIONS
        ):
            raise ValueError("Semantic relation must connect two known different elements")
        key = first, second, kind
        if key in edge_keys:
            raise ValueError("Duplicate semantic relation")
        edge_keys.add(key)
        _validate_confidence_evidence(relation)
        if kind == "reading_before":
            order[first].append(second)
    visiting, visited = set(), set()

    def visit(eid):
        if eid in visiting:
            raise ValueError("Semantic reading order contains a cycle")
        if eid in visited:
            return
        visiting.add(eid)
        for other in order[eid]:
            visit(other)
        visiting.remove(eid)
        visited.add(eid)

    for eid in ids:
        visit(eid)
    body = deepcopy(proposal)
    return {**body, "proposal_digest": _digest(body), "authority": "soft_proposal_only"}


def _validate_confidence_evidence(row):
    if not _finite(row["confidence"]) or not 0 <= row["confidence"] <= 1:
        raise ValueError("Semantic confidence must be finite and between zero and one")
    if not isinstance(row["evidence"], str) or not 1 <= len(row["evidence"]) <= 1000:
        raise ValueError("Semantic evidence must be nonempty bounded text")


def audit_semantic_preservation(source, design, semantics=None, *, assets_dir=None, fonts_dir=None):
    """Measure lost source anchors/groups/order without claiming visual approval."""
    _validate_source(source)
    _validate_source(design)
    if semantics is None:
        semantics = describe_design(source, assets_dir=assets_dir, fonts_dir=fonts_dir)
    if (
        not isinstance(semantics, dict)
        or not {"schema_version", "policy_version", "source_canvas", "elements", "relations"}
        <= semantics.keys()
    ):
        raise ValueError("Semantic evidence must be an original-source sidecar")
    if (
        type(semantics["schema_version"]) is not int
        or semantics["schema_version"] != 1
        or semantics["policy_version"] != POLICY_VERSION
        or semantics["source_canvas"] != source["canvas"]
        or semantics.get("source_digest_protocol") != SOURCE_DIGEST_PROTOCOL
        or semantics.get("source_digest") != semantic_source_digest(source)
        or semantics.get("semantic_digest")
        != _digest({k: v for k, v in semantics.items() if k != "semantic_digest"})
    ):
        raise ValueError("Semantic evidence is stale or changed")
    originals = {e["id"]: e for e in source["elements"]}
    outputs = {e["id"]: e for e in design["elements"]}
    if set(originals) != set(outputs):
        raise ValueError("Semantic audit requires exactly the original element IDs")
    if (
        not isinstance(semantics["elements"], list)
        or len(semantics["elements"]) != len(originals)
        or any(
            not isinstance(row, dict) or not isinstance(row.get("element_id"), str)
            for row in semantics["elements"]
        )
        or {r["element_id"] for r in semantics["elements"]} != set(originals)
    ):
        raise ValueError("Semantic descriptor coverage does not match the source")
    width, height = (float(design["canvas"][k]) for k in ("width", "height"))
    corners, footers, group_scores, order_scores, violations = [], [], [], [], []
    for row in semantics["elements"]:
        if not row["anchors"]:
            continue
        element = outputs[row["element_id"]]
        box = (
            _ink(element, assets_dir, fonts_dir)["box"]
            if row["visible_box_provenance"] != "frame"
            else _box(element)
        )
        insets = _insets(box, width, height)
        for anchor in row["anchors"]:
            error = sum(
                abs(insets[edge] - value) for edge, value in anchor["insets_short_axis"].items()
            ) / len(anchor["edges"])
            footer = anchor["provenance"] == "source_footer_affinity"
            if footer and anchor.get("horizontal") == "center":
                error += abs((box[0] + box[2] / 2) / width - anchor["source_center_x"])
            retention = math.exp(-error / 0.08)
            (footers if footer else corners).append(retention)
            if error > 0.08:
                violations.append(
                    {
                        "code": "footer_affinity" if footer else "corner_affinity",
                        "element_id": row["element_id"],
                        "normalized_error": error,
                        "severity": "review",
                        "origin": anchor["provenance"],
                    }
                )
    groups = defaultdict(list)
    for row in semantics["elements"]:
        if row["rigid_group"]:
            groups[row["rigid_group"]].append(row["element_id"])
    for group, ids in groups.items():
        if len(ids) < 2:
            continue
        ratios = [
            outputs[eid][key] / originals[eid][key] for eid in ids for key in ("width", "height")
        ]
        ratios.extend(
            outputs[eid]["font_size"] / originals[eid]["font_size"]
            for eid in ids
            if originals[eid]["type"] == "text"
        )
        scale = median(ratios)
        pivot = ids[0]
        errors = [abs(r / scale - 1) for r in ratios]
        for eid in ids:
            for key in ("x", "y"):
                expected = (
                    outputs[pivot][key] + (originals[eid][key] - originals[pivot][key]) * scale
                )
                errors.append(abs(outputs[eid][key] - expected) / min(width, height))
        error = max(errors, default=0.0)
        group_scores.append(math.exp(-error / 0.03))
        if error > 0.03:
            violations.append(
                {
                    "code": "explicit_group_geometry",
                    "group_id": group,
                    "severity": "hard",
                    "normalized_error": error,
                }
            )
    for edge in semantics["relations"]:
        if edge["type"] != "reading_before":
            continue
        first, second = outputs[edge["source_id"]], outputs[edge["target_id"]]
        # Realized rows read horizontally; vertically separated content reads down.
        overlap = min(first["y"] + first["height"], second["y"] + second["height"]) - max(
            first["y"], second["y"]
        )
        horizontal = overlap > min(first["height"], second["height"]) * 0.3
        key, dimension = ("x", "width") if horizontal else ("y", "height")
        precedes = first[key] + first[dimension] / 2 < second[key] + second[dimension] / 2
        order_scores.append(float(precedes))
        if not precedes:
            violations.append(
                {
                    "code": "reading_order",
                    "source_id": edge["source_id"],
                    "target_id": edge["target_id"],
                    "severity": "review",
                }
            )

    def average(values):
        return sum(values) / len(values) if values else 1.0

    return {
        "schema_version": 1,
        "semantic_digest": semantics["semantic_digest"],
        "metrics": {
            "corner_anchor_retention": average(corners),
            "footer_anchor_retention": average(footers),
            "explicit_group_retention": average(group_scores),
            "reading_order_retention": average(order_scores),
        },
        "coverage": {
            "corner_anchors": len(corners),
            "footer_anchors": len(footers),
            "explicit_groups": len(group_scores),
            "reading_edges": len(order_scores),
        },
        "violations": violations,
        "visual_quality": "not_reviewed",
    }
