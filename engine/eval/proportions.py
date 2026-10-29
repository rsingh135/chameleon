"""Measure scale consistency inside source groups and related campaigns.

These are deformation diagnostics, not semantic or aesthetic scores. Standalone
text may reflow, so its font scale represents visual size; its frame aspect is
not treated as a rigid constraint. Explicit groups are measured as rigid units.
"""

from __future__ import annotations

import math
from statistics import median

from engine.layout.relationships import SourceRelationshipGraph, build_relationship_graph


def _spread(scales: list[float]) -> tuple[float, float]:
    if not scales:
        return 0.0, 1.0
    logs = [math.log(max(value, 1e-12)) for value in scales]
    center = median(logs)
    rms = math.sqrt(sum((value - center) ** 2 for value in logs) / len(logs))
    return rms, math.exp(max(logs) - min(logs))


def _size_scale(original: dict, output: dict) -> float:
    if original["type"] == "text":
        return output["font_size"] / original["font_size"]
    return math.sqrt(output["width"] * output["height"] / (original["width"] * original["height"]))


def _rigid_error(originals: list[dict], outputs: dict) -> tuple[float, float]:
    scales = []
    for original in originals:
        output = outputs[original["id"]]
        for field in ("width", "height", "font_size", "corner_radius", "stroke_width"):
            if original.get(field, 0) > 0:
                scales.append(output.get(field, 0.0) / original[field])
    scale = median(scales)
    deviations = [abs(math.log(max(value, 1e-12) / scale)) for value in scales]
    # Fit one translation after the common scalar. Rotation is unchanged by the
    # contract, so source-frame origins are sufficient for this residual.
    dx = median(outputs[e["id"]]["x"] - e["x"] * scale for e in originals)
    dy = median(outputs[e["id"]]["y"] - e["y"] * scale for e in originals)
    extent = (
        max(
            max(e["x"] + e["width"] for e in originals) - min(e["x"] for e in originals),
            max(e["y"] + e["height"] for e in originals) - min(e["y"] for e in originals),
            1e-6,
        )
        * scale
    )
    residual = math.sqrt(
        sum(
            (outputs[e["id"]]["x"] - (e["x"] * scale + dx)) ** 2
            + (outputs[e["id"]]["y"] - (e["y"] * scale + dy)) ** 2
            for e in originals
        )
        / len(originals)
    ) / max(extent, 1e-6)
    return max(deviations, default=0.0), residual


def analyze_proportions(
    source: dict, candidate: dict, graph: SourceRelationshipGraph | None = None
) -> dict:
    """Compare complete source units, campaign scales and repeated products."""
    graph = graph or build_relationship_graph(source)
    originals = {element["id"]: element for element in source["elements"]}
    outputs = {element["id"]: element for element in candidate["elements"]}
    if (
        set(originals) != set(outputs)
        or len(outputs) != len(candidate["elements"])
        or len(originals) != len(source["elements"])
    ):
        return {"score": 0.0, "warnings": [], "metrics": {}, "clusters": []}
    warnings, rows = [], []
    rigid_errors = []
    for unit in graph.units:
        if len(unit.element_ids) < 2 and originals[unit.element_ids[0]]["type"] == "text":
            continue
        members = [originals[eid] for eid in unit.element_ids]
        dimension_error, residual = _rigid_error(members, outputs)
        error = max(dimension_error, residual)
        rigid_errors.append(error)
        if error > math.log(1.05):
            warnings.append(
                {
                    "code": "rigid_group_deformation",
                    "element_ids": list(unit.element_ids),
                    "message": ("A fixed block changed its internal size or spacing ratios."),
                    "value": round(error, 6),
                }
            )

    campaign_errors = []
    product_scales = []
    for cluster in graph.clusters:
        if cluster.kind not in {"brand", "message", "badge_row", "campaign"}:
            continue
        scales = [_size_scale(originals[eid], outputs[eid]) for eid in cluster.element_ids]
        rms, ratio = _spread(scales)
        rows.append(
            {
                "id": cluster.id,
                "kind": cluster.kind,
                "confidence": cluster.confidence,
                "element_ids": list(cluster.element_ids),
                "scale_rms_log": round(rms, 6),
                "largest_to_smallest_scale": round(ratio, 6),
            }
        )
        if cluster.kind == "campaign":
            campaign_errors.append(rms)
            if cluster.primary_unit is not None:
                primary = graph.units[cluster.primary_unit]
                product_scales.append(
                    median(_size_scale(originals[eid], outputs[eid]) for eid in primary.element_ids)
                )
            if ratio > 1.15:
                warnings.append(
                    {
                        "code": "campaign_scale_drift",
                        "element_ids": list(cluster.element_ids),
                        "message": (
                            f"Related product/message members use scales differing by {ratio:.2f}x."
                        ),
                        "value": round(ratio, 6),
                    }
                )
    sibling_rms, sibling_ratio = _spread(product_scales)
    if sibling_ratio > 1.15:
        warnings.append(
            {
                "code": "sibling_product_scale_drift",
                "element_ids": [
                    eid
                    for cluster in graph.clusters
                    if cluster.kind == "campaign" and cluster.primary_unit is not None
                    for eid in graph.units[cluster.primary_unit].element_ids
                ],
                "message": f"Products' relative source sizes changed by {sibling_ratio:.2f}x.",
                "value": round(sibling_ratio, 6),
            }
        )
    rigid = max(rigid_errors, default=0.0)
    campaign = sum(campaign_errors) / max(1, len(campaign_errors))
    # Equal exact scales score one. The exponential is only a ranking utility;
    # retain raw deformation values so callers can assess the actual tradeoff.
    return {
        "score": round(math.exp(-(rigid + campaign + sibling_rms)), 6),
        "warnings": warnings,
        "metrics": {
            "rigid_group_error": round(rigid, 6),
            "campaign_scale_rms_log": round(campaign, 6),
            "sibling_product_scale_rms_log": round(sibling_rms, 6),
            "sibling_product_scale_ratio": round(sibling_ratio, 6),
        },
        "clusters": rows,
    }
