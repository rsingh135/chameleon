"""Measure scale consistency inside source groups and related campaigns.

These are deformation diagnostics, not semantic or aesthetic scores. Standalone
text may reflow, so its font scale represents visual size; its frame aspect is
not treated as a rigid constraint. Explicit groups are measured as rigid units.
"""

from __future__ import annotations

import math
from statistics import median

from engine.layout.appearance import unit_visible_box
from engine.layout.relationships import SourceRelationshipGraph, build_relationship_graph
from engine.layout.source_containers import analyze_source_containers
from engine.resources import using_resources


def _composition_members(source, graph):
    campaigns = {
        eid
        for cluster in graph.clusters
        if cluster.kind == "campaign"
        for eid in cluster.element_ids
    }
    if campaigns:
        return campaigns, "campaign"
    # Campaign inference is intentionally narrow. Typography-led posters and
    # diagrams still have immutable foreground units/bands whose overall span
    # should survive an aspect change. Select them from the SOURCE only.
    originals = {element["id"]: element for element in source["elements"]}
    sizes = [element["font_size"] for element in source["elements"] if element["type"] == "text"]
    typical = median(sizes) if sizes else 0
    overlays = {overlay.overlay_id for overlay in graph.overlays}
    members = set()
    for unit in graph.units:
        # A proven full-bleed photo supplies the background for its caption.
        # Its large frame cannot establish foreground composition retention;
        # photo visibility and geometry have their own independent guards.
        if unit.role == "image_underlay":
            continue
        elements = [originals[eid] for eid in unit.element_ids]
        texts = [element for element in elements if element["type"] == "text"]
        small_type = texts and max(element["font_size"] for element in texts) <= typical
        footer = (
            small_type
            and len(texts) == len(elements)
            and unit.box.y >= graph.height * 0.85
            and unit.box.height <= graph.height * 0.1
        )
        corner = (
            unit.role != "action"
            and unit.box.width * unit.box.height <= graph.width * graph.height * 0.02
            and (
                unit.box.x <= graph.width * 0.05
                or unit.box.x + unit.box.width >= graph.width * 0.95
            )
            and (
                unit.box.y <= graph.height * 0.05
                or unit.box.y + unit.box.height >= graph.height * 0.95
            )
        )
        # A small corner support or genuine fine footer cannot make an otherwise
        # compressed central composition appear to occupy the full canvas.
        if footer or corner:
            continue
        members.update(eid for eid in unit.element_ids if eid not in overlays)
    return members, "foreground-bands" if members else "unavailable"


def _composition_span(source, candidate, graph, originals, outputs):
    primary_ids, kind = _composition_members(source, graph)
    diagnostic = {
        "active": False,
        "kind": kind,
        "element_ids": sorted(primary_ids),
        "source_fraction": [],
        "candidate_fraction": [],
    }
    if not primary_ids:
        return 1.0, diagnostic
    # Graphs can be analyzed after their resource context has ended. Exact
    # document faces must still determine occupied line extents here.
    with using_resources(assets_dir=graph.assets_dir, fonts_dir=graph.fonts_dir):
        before = unit_visible_box([originals[eid] for eid in primary_ids], graph.assets_dir)
        after = unit_visible_box([outputs[eid] for eid in primary_ids], graph.assets_dir)
    if min(before.width, before.height) <= 0:
        return 1.0, diagnostic
    source_fraction = [before.width / graph.width, before.height / graph.height]
    candidate_fraction = [
        after.width / candidate["canvas"]["width"],
        after.height / candidate["canvas"]["height"],
    ]
    diagnostic.update(
        active=True, source_fraction=source_fraction, candidate_fraction=candidate_fraction
    )
    ratios = [
        after / before for after, before in zip(candidate_fraction, source_fraction, strict=True)
    ]
    # A typography/diagram composition can exchange width for height when its
    # source bands become columns. Retain its authored visual weight without
    # imposing the old orientation. Product campaigns retain both-axis guards.
    area_retention = max(0.0, min(1.0, ratios[0] * ratios[1]))
    axis_retention = max(0.0, min(1.0, *ratios))
    diagnostic.update(
        basis="occupied-area" if kind == "foreground-bands" else "occupied-axes",
        area_retention=area_retention,
        axis_retention=axis_retention,
    )
    return (area_retention if kind == "foreground-bands" else axis_retention), diagnostic


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
    with using_resources(assets_dir=graph.assets_dir, fonts_dir=graph.fonts_dir):
        containers = analyze_source_containers(source, candidate)
    adaptive_frames = set(containers["accepted_frame_ids"])
    warnings, rows = list(containers["warnings"]), []
    rigid_errors = []
    for unit in graph.units:
        member_ids = [eid for eid in unit.element_ids if eid not in adaptive_frames]
        if not member_ids or (len(member_ids) < 2 and originals[member_ids[0]]["type"] == "text"):
            continue
        members = [originals[eid] for eid in member_ids]
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
    span_retention, span_diagnostic = _composition_span(
        source, candidate, graph, originals, outputs
    )
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
            "composition_span_retention": round(span_retention, 6),
            "adaptive_frame_count": len(adaptive_frames),
            "source_container_detached_count": containers["metrics"]["detached_count"],
            "source_container_unsupported_deformation_count": containers["metrics"][
                "unsupported_deformation_count"
            ],
        },
        "clusters": rows,
        "composition_span": span_diagnostic,
        "containers": containers,
    }
