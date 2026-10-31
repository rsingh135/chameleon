"""Source evidence for a requested single-product horizontal focal anchor.

This profile does not declare that every design should be centered. It applies
only to a source with one unambiguous image campaign and a corroborating center
axis. Geometry supplies the evidence; the user's focal requirement supplies the
priority. Candidate y positions remain free to adapt to a new aspect ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from engine.resources import get_resources, using_resources

from .appearance import unit_visible_box
from .geometry import Box
from .relationships import SourceRelationshipGraph, build_relationship_graph

SOURCE_CENTER_BAND = 0.03
SUPPORT_CENTER_BAND = 0.06
TARGET_CENTER_BAND = 0.025


@dataclass(frozen=True)
class FocalProfile:
    primary_unit: int
    primary_element_ids: tuple[str, ...]
    supporting_units: tuple[int, ...]
    anchor_x: float
    source_center_x: float
    source_visible_box: Box
    source_visible_height_fraction: float
    source_occupied_centroid_x: float
    foreground_units: tuple[tuple[str, ...], ...]
    headline_font_sizes: tuple[tuple[str, float], ...]
    cta_heights: tuple[tuple[tuple[str, ...], float], ...]
    source_ids: tuple[str, ...]
    source_canvas: tuple[float, float]
    assets_dir: str | None
    fonts_dir: str | None
    provenance: tuple[str, ...]


def _actual_image(element: dict, assets: Path) -> bool:
    """Missing placeholders and fully transparent resources are not image evidence."""
    if element["type"] != "image" or element.get("opacity", 1.0) <= 0:
        return False
    try:
        from PIL import Image

        path = (assets / element.get("asset_id", "")).resolve()
        if not path.is_relative_to(assets.resolve()) or not path.is_file():
            return False
        with Image.open(path) as image:
            return (
                image.width > 0
                and image.height > 0
                and bool(image.convert("RGBA").getchannel("A").getbbox())
            )
    except (ImportError, OSError, ValueError):
        return False


def _center_x(box: Box, width: float) -> float:
    return (box.x + box.width / 2) / width


def _matching_graph(source: dict, graph: SourceRelationshipGraph) -> bool:
    ids = [element["id"] for element in source["elements"]]
    return (
        len(ids) == len(set(ids))
        and set(ids) == set(graph.source_ids)
        and source["canvas"]["width"] == graph.width
        and source["canvas"]["height"] == graph.height
    )


def _occupied_centroid_x(
    units: tuple[tuple[str, ...], ...], by_id: dict, assets: Path, width: float
) -> float:
    """Approximate occupied-box mass, not optical/luminance balance.

    Each foreground graph unit contributes one visible bounding box. A native
    button's background and label therefore contribute once as a complete unit.
    The reference is the author's source centroid, not universal canvas center.
    """
    weighted, mass = 0.0, 0.0
    for ids in units:
        elements = [by_id[eid] for eid in ids if by_id[eid].get("opacity", 1.0) > 0]
        if not elements:
            continue
        box = unit_visible_box(elements, assets)
        area = max(0.0, box.width) * max(0.0, box.height)
        weighted += _center_x(box, width) * area
        mass += area
    return weighted / mass if mass else 0.5


def infer_focal_profile(
    source: dict, graph: SourceRelationshipGraph | None = None
) -> FocalProfile | None:
    """Describe a supported x-center axis, never impose one on asymmetric sources."""
    graph = graph or build_relationship_graph(source)
    if not _matching_graph(source, graph) or graph.ambiguities:
        return None
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    heroes = [unit for unit in graph.units if unit.role == "hero"]
    if len(campaigns) != 1 or len(heroes) != 1:
        return None
    campaign = campaigns[0]
    primary_index = campaign.primary_unit
    if primary_index is None or primary_index != heroes[0].index:
        return None
    primary = graph.units[primary_index]
    originals = {element["id"]: element for element in source["elements"]}
    width, height = graph.width, graph.height
    if min(width, height) <= 0:
        return None
    with using_resources(graph.assets_dir, graph.fonts_dir):
        assets = get_resources().assets_dir
        fonts_dir = str(get_resources().fonts_dir)
        primary_elements = [originals[eid] for eid in primary.element_ids]
        if not any(_actual_image(element, assets) for element in primary_elements):
            return None
        visible = unit_visible_box(primary_elements, assets)
        if min(visible.width, visible.height) <= 0:
            return None
        center = _center_x(visible, width)
        if abs(center - 0.5) > SOURCE_CENTER_BAND + 1e-9:
            return None
        supports = []
        for index in campaign.unit_indices:
            unit = graph.units[index]
            if unit.role not in {"headline", "action", "logo"}:
                continue
            box = unit_visible_box([originals[eid] for eid in unit.element_ids], assets)
            companion_center = _center_x(box, width)
            if (
                min(box.width, box.height) > 0
                and abs(companion_center - 0.5) <= SUPPORT_CENTER_BAND + 1e-9
                and abs(companion_center - center) <= SUPPORT_CENTER_BAND + 1e-9
            ):
                supports.append(index)
        if len(supports) < 2:
            return None
        # Scale diagnostics cover the campaign's source typography/actions even
        # if a particular peer was not one of the corroborating center anchors.
        headlines = tuple(
            (eid, float(originals[eid]["font_size"]))
            for index in campaign.unit_indices
            if graph.units[index].role == "headline"
            for eid in graph.units[index].element_ids
            if originals[eid]["type"] == "text" and originals[eid].get("font_size", 0) > 0
        )
        actions = tuple(
            (
                graph.units[index].element_ids,
                unit_visible_box(
                    [originals[eid] for eid in graph.units[index].element_ids], assets
                ).height,
            )
            for index in campaign.unit_indices
            if graph.units[index].role == "action"
        )
        foreground_units = tuple(unit.element_ids for unit in graph.units)
        source_centroid = _occupied_centroid_x(foreground_units, originals, assets, width)
    return FocalProfile(
        primary_unit=primary_index,
        primary_element_ids=primary.element_ids,
        supporting_units=tuple(supports),
        anchor_x=0.5,
        source_center_x=center,
        source_visible_box=visible,
        source_visible_height_fraction=visible.height / min(width, height),
        source_occupied_centroid_x=source_centroid,
        foreground_units=foreground_units,
        headline_font_sizes=headlines,
        cta_heights=actions,
        source_ids=graph.source_ids,
        source_canvas=(width, height),
        assets_dir=str(assets),
        fonts_dir=fonts_dir,
        provenance=(
            "One unambiguous campaign has one primary hero unit with a readable image asset.",
            "The source alpha-visible hero center lies within 3% of horizontal canvas center.",
            "At least two source logo/headline/action units corroborate "
            "this horizontal center axis.",
            "This is source evidence for the requested x anchor; "
            "no y-center constraint is inferred.",
        ),
    )


def _inactive() -> dict:
    return {
        "score": 1.0,
        "warnings": [],
        "metrics": {
            "active": False,
            "center_error": 0.0,
            "center_retained": True,
            "visible_span_retention": 1.0,
            "visible_span_ratio": 1.0,
            "occupied_balance_error": 0.0,
            "balance_score": 1.0,
            "focal_product_scale_vs_headline": None,
            "focal_product_scale_vs_cta": None,
        },
    }


def analyze_focal(
    source: dict,
    candidate: dict,
    graph: SourceRelationshipGraph | None = None,
    profile: FocalProfile | None = None,
) -> dict:
    """Report x-anchor drift separately from source-relative image/text scale.

    The center score is exp(-8 * normalized x error), a diagnostic proxy. Span
    and hierarchy ratios are soft diagnostics, not claims of visual perfection.
    A source without this evidence gets a neutral, explicitly inactive result.
    """
    profile = profile or infer_focal_profile(source, graph)
    if profile is None:
        return _inactive()
    originals = {element["id"]: element for element in source["elements"]}
    outputs = {element["id"]: element for element in candidate["elements"]}
    width, height = candidate["canvas"]["width"], candidate["canvas"]["height"]
    if (
        set(originals) != set(profile.source_ids)
        or tuple(source["canvas"][key] for key in ("width", "height")) != profile.source_canvas
        or len(outputs) != len(candidate["elements"])
        or set(outputs) != set(originals)
        or min(width, height) <= 0
    ):
        result = _inactive()
        result.update(
            score=0.0,
            warnings=[
                {
                    "code": "focal_membership",
                    "element_ids": list(profile.primary_element_ids),
                    "message": (
                        "Focal comparison requires the original source "
                        "and complete candidate membership."
                    ),
                }
            ],
        )
        result["metrics"].update(
            active=True, center_retained=False, occupied_balance_error=1.0, balance_score=0.0
        )
        return result
    with using_resources(profile.assets_dir, profile.fonts_dir):
        visible = unit_visible_box(
            [outputs[eid] for eid in profile.primary_element_ids], get_resources().assets_dir
        )
        center_error = abs(_center_x(visible, width) - profile.anchor_x)
        balance_error = abs(
            _occupied_centroid_x(
                profile.foreground_units, outputs, get_resources().assets_dir, width
            )
            - profile.source_occupied_centroid_x
        )
        span_ratio = visible.height / min(width, height) / profile.source_visible_height_fraction
        product_scale = visible.height / profile.source_visible_box.height
        headline_ratios = [
            product_scale / (outputs[eid]["font_size"] / size)
            for eid, size in profile.headline_font_sizes
            if outputs[eid].get("font_size", 0) > 0
        ]
        cta_ratios = [
            product_scale / (box.height / source_height)
            for ids, source_height in profile.cta_heights
            if source_height > 0
            and (
                box := unit_visible_box([outputs[eid] for eid in ids], get_resources().assets_dir)
            ).height
            > 0
        ]
    retained = center_error <= TARGET_CENTER_BAND + 1e-9
    headline_ratio = median(headline_ratios) if headline_ratios else None
    cta_ratio = median(cta_ratios) if cta_ratios else None
    warnings = []
    if not retained:
        warnings.append(
            {
                "code": "focal_center_drift",
                "element_ids": list(profile.primary_element_ids),
                "message": (
                    f"The source-centered product moved {center_error:.1%} of canvas "
                    "width from its horizontal focal anchor."
                ),
            }
        )
    if span_ratio < 0.8:
        warnings.append(
            {
                "code": "focal_visible_span",
                "element_ids": list(profile.primary_element_ids),
                "message": (
                    f"The product retains only {span_ratio:.0%} of its source "
                    "visible-height share of the short canvas side."
                ),
            }
        )
    for name, value in (("headline", headline_ratio), ("cta", cta_ratio)):
        if value is not None and abs(math.log(max(value, 1e-12))) > math.log(1.15):
            warnings.append(
                {
                    "code": f"focal_product_scale_vs_{name}",
                    "element_ids": list(profile.primary_element_ids),
                    "message": (
                        f"Product scale relative to source {name} scale changed by {value:.2f}x."
                    ),
                }
            )
    return {
        "score": round(math.exp(-8 * center_error), 6),
        "warnings": warnings,
        "metrics": {
            "active": True,
            "center_error": round(center_error, 6),
            "center_retained": retained,
            "visible_span_retention": round(min(1.0, span_ratio), 6),
            "visible_span_ratio": round(span_ratio, 6),
            "occupied_balance_error": round(balance_error, 6),
            "balance_score": round(math.exp(-4 * balance_error), 6),
            "focal_product_scale_vs_headline": round(headline_ratio, 6)
            if headline_ratio is not None
            else None,
            "focal_product_scale_vs_cta": round(cta_ratio, 6) if cta_ratio is not None else None,
        },
    }
