"""Bounded multi-campaign states using ownership supplied by the source graph.

Local product/message blocks change flow together; their membership is never
re-inferred from a candidate. Global packing preserves campaign order. This is
a small supported family, not an optimizer for arbitrary page structures.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass

from .appearance import unit_visible_box
from .blocks import _brand, _message, _translate, _uniform
from .candidates import _base
from .geometry import Box, union
from .model import Unit, build_context
from .relationships import SourceRelationshipGraph, analyze_relationships
from .text import get_measurer

MAX_STATES = 8
MAX_CAMPAIGNS = 6


@dataclass(frozen=True)
class CampaignState:
    name: str
    topology: str
    cluster_axes: tuple[tuple[str, str], ...]
    design_json: str

    @property
    def design(self) -> dict:
        """Return a copy, keeping original source and state snapshots immutable."""
        return json.loads(self.design_json)


@dataclass
class _PreparedCampaign:
    campaign: object
    cell: Box
    source_elements: list[dict]
    identity: list[dict]
    message_units: list[Unit]
    rows: tuple[tuple[Unit, ...], ...]
    copy_width: float
    alignment: str
    short: float


def _packing(target: dict, count: int, columns: int) -> tuple[Box, ...]:
    width, height = target["width"], target["height"]
    short = min(width, height)
    pad, gap = short * 0.04, short * 0.06
    rows = math.ceil(count / columns)
    cell_width = (width - 2 * pad - gap * (columns - 1)) / columns
    cell_height = (height - 2 * pad - gap * (rows - 1)) / rows
    if min(cell_width, cell_height) <= 0:
        return ()
    cells = []
    for index in range(count):
        row, column = divmod(index, columns)
        row_count = min(columns, count - row * columns)
        row_width = row_count * cell_width + (row_count - 1) * gap
        left = (width - row_width) / 2
        cells.append(
            Box(
                left + column * (cell_width + gap),
                pad + row * (cell_height + gap),
                cell_width,
                cell_height,
            )
        )
    return tuple(cells)


def _parts(campaign, by_cluster, graph, context):
    nested = [by_cluster[cluster_id] for cluster_id in campaign.member_cluster_ids]
    messages = [cluster for cluster in nested if cluster.kind == "message"]
    brands = [cluster for cluster in nested if cluster.kind == "brand"]
    rows = [cluster for cluster in nested if cluster.kind == "badge_row"]
    if len(messages) != 1 or len(brands) > 1:
        return None
    message = messages[0]
    message_edges = [edge for edge in graph.edges if edge.cluster_id == message.id]
    if any(edge.axis != "vertical" for edge in message_edges):
        return None
    identity_indices = brands[0].unit_indices if brands else (campaign.primary_unit,)
    used = list(identity_indices) + list(message.unit_indices)
    used.extend(index for row in rows for index in row.unit_indices)
    if len(used) != len(set(used)) or set(used) != set(campaign.unit_indices):
        return None
    alignment = next(
        (edge.alignment for edge in message_edges if edge.code == "shared_axis"), "center"
    )
    identity_units = [context.units[index] for index in identity_indices]
    return (
        identity_units,
        [context.units[index] for index in message.unit_indices],
        tuple(tuple(context.units[index] for index in row.unit_indices) for row in rows),
        alignment,
    )


def _prepare(campaigns, cells, source, graph, context, mode):
    by_id = {element["id"]: element for element in source["elements"]}
    by_cluster = {cluster.id: cluster for cluster in graph.clusters}
    prepared = []
    for campaign, cell in zip(campaigns, cells, strict=True):
        elements = [by_id[eid] for eid in campaign.element_ids]
        short = min(cell.width, cell.height)
        if mode == "source":
            prepared.append(
                _PreparedCampaign(campaign, cell, elements, [], [], (), 0, "center", short)
            )
            continue
        parts = _parts(campaign, by_cluster, graph, context)
        if parts is None:
            return None
        identity_units, message_units, rows, alignment = parts
        brand_edges = [
            edge
            for edge in graph.edges
            if edge.code == "brand_association"
            and edge.source_unit in campaign.unit_indices
            and edge.target_unit in campaign.unit_indices
        ]
        identity_width = cell.width * (0.35 if mode == "columns" else 0.72)
        identity_height = cell.height * (0.96 if mode == "columns" else 0.44)
        if any(edge.axis != "vertical" for edge in brand_edges):
            group = Unit(
                [e for unit in identity_units for e in unit.elements],
                union([e for unit in identity_units for e in unit.elements]),
            )
            scale = min(identity_width / group.box.width, identity_height / group.box.height)
            identity = _uniform(group, scale, 0, 0, graph.assets_dir)
        else:
            identity = _brand(
                identity_units, identity_width, identity_height, short * 0.035, graph.assets_dir
            )
        prepared.append(
            _PreparedCampaign(
                campaign,
                cell,
                elements,
                identity,
                message_units,
                rows,
                cell.width * (0.62 if mode == "columns" else 0.92),
                alignment,
                short,
            )
        )
    return prepared


def _local(prepared: _PreparedCampaign, scale: float, mode: str, assets) -> list[dict] | None:
    cell = prepared.cell
    if mode == "source":
        group = Unit(prepared.source_elements, union(prepared.source_elements))
        scale = min(scale, cell.width / group.box.width, cell.height / group.box.height)
        result = _uniform(group, scale, 0, 0, assets)
    else:
        content = _message(
            prepared.message_units,
            prepared.copy_width,
            scale,
            prepared.short,
            assets,
            rows=prepared.rows,
            alignment=prepared.alignment,
        )
        if not content:
            return None
        identity = deepcopy(prepared.identity)
        identity_box, copy_box = (
            unit_visible_box(identity, assets),
            unit_visible_box(content, assets),
        )
        gap = prepared.short * 0.05
        if mode == "columns":
            _translate(identity, -identity_box.x, -identity_box.y - identity_box.height / 2)
            _translate(
                content, identity_box.width + gap - copy_box.x, -copy_box.y - copy_box.height / 2
            )
        else:
            _translate(identity, -identity_box.x - identity_box.width / 2, -identity_box.y)
            _translate(
                content, -copy_box.x - copy_box.width / 2, identity_box.height + gap - copy_box.y
            )
        result = identity + content
    frame = union(result)
    if frame.width > cell.width + 1e-6 or frame.height > cell.height + 1e-6:
        return None
    measurer = get_measurer()
    for element in result:
        if element["type"] == "text":
            measured = measurer.measure(element, element["font_size"], element["width"])
            if (
                measured.width > element["width"] + 0.01
                or measured.height > element["height"] + 0.01
            ):
                return None
    _translate(
        result,
        cell.x + (cell.width - frame.width) / 2 - frame.x,
        cell.y + (cell.height - frame.height) / 2 - frame.y,
    )
    return result


def _realize(prepared, preferred_scale, mode, graph):
    if mode == "source":
        preferred_scale = min(
            preferred_scale,
            *(
                min(
                    item.cell.width / union(item.source_elements).width,
                    item.cell.height / union(item.source_elements).height,
                )
                for item in prepared
            ),
        )

    def arrange(scale):
        placed = [_local(campaign, scale, mode, graph.assets_dir) for campaign in prepared]
        return (
            None
            if any(elements is None for elements in placed)
            else [element for elements in placed for element in elements]
        )

    result = arrange(preferred_scale)
    if result is None:
        lower, upper = 0.0, preferred_scale
        best = None
        for _ in range(18):
            scale = (lower + upper) / 2
            candidate = arrange(scale)
            if candidate is None:
                upper = scale
            else:
                lower, best = scale, candidate
        result = best
    if result is None or any(element.get("font_size", 1) < 1 for element in result):
        return None
    return result


def _axes(graph, campaigns, mode):
    axes = {edge.cluster_id: edge.axis for edge in graph.edges if edge.cluster_id}
    if mode != "source":
        nested = {
            cluster_id for campaign in campaigns for cluster_id in campaign.member_cluster_ids
        }
        for campaign in campaigns:
            axes[campaign.id] = "horizontal" if mode == "columns" else "vertical"
        for cluster in graph.clusters:
            if cluster.id in nested and cluster.kind == "badge_row":
                axes[cluster.id] = "wrap"
    return tuple(sorted(axes.items()))


def _comparison_row(campaigns, graph, context) -> bool:
    """Recognize source row anchors, not a universal comparison-card assumption."""
    by_cluster = {cluster.id: cluster for cluster in graph.clusters}
    headings, actions, products, frames = [], [], [], []
    for campaign in campaigns:
        messages = [
            by_cluster[key]
            for key in campaign.member_cluster_ids
            if by_cluster[key].kind == "message"
        ]
        if len(messages) != 1 or len(messages[0].unit_indices) < 3:
            return False
        message = messages[0]
        heading = context.units[message.unit_indices[0]]
        action = context.units[message.unit_indices[-1]]
        if graph.units[message.unit_indices[0]].role != "headline" or not (
            action.text and any(element["type"] == "shape" for element in action.elements)
        ):
            return False
        headings.append(unit_visible_box(heading.elements, graph.assets_dir).y)
        action_box = unit_visible_box(action.elements, graph.assets_dir)
        actions.append(action_box.y + action_box.height)
        product = graph.units[campaign.primary_unit].box
        products.append(product.y + product.height)
        frames.append(
            union(
                [
                    element
                    for index in campaign.unit_indices
                    for element in context.units[index].elements
                ]
            )
        )
    tolerance = graph.height * 0.015
    if any(max(values) - min(values) > tolerance for values in (headings, actions, products)):
        return False
    return all(
        first.x + first.width <= second.x + min(first.width, second.width) * 0.1
        for first, second in zip(frames, frames[1:], strict=False)
    )


def _aligned_trial(prepared, scale, graph):
    """Allocate shared source-comparison rows; surplus stays before peers/actions."""
    sections, source_sections, identity_boxes, product_boxes = [], [], [], []
    for item in prepared:
        units = item.message_units
        source_groups = (
            units[:1],
            units[1:-1],
            [unit for row in item.rows for unit in row],
            units[-1:],
        )
        source_sections.append(
            [
                unit_visible_box(
                    [element for unit in group for element in unit.elements], graph.assets_dir
                )
                if group
                else None
                for group in source_groups
            ]
        )
        parts = [
            _message(
                units[:1],
                item.copy_width,
                scale,
                item.short,
                graph.assets_dir,
                alignment=item.alignment,
            ),
            _message(
                units[1:-1],
                item.copy_width,
                scale,
                item.short,
                graph.assets_dir,
                alignment=item.alignment,
            ),
            _message(
                [],
                item.copy_width,
                scale,
                item.short,
                graph.assets_dir,
                rows=item.rows,
                alignment=item.alignment,
            )
            if item.rows
            else [],
            _message(
                units[-1:],
                item.copy_width,
                scale,
                item.short,
                graph.assets_dir,
                alignment=item.alignment,
            ),
        ]
        if any(group and not placed for group, placed in zip(source_groups, parts, strict=True)):
            return None
        sections.append(parts)
        identity_boxes.append(union(item.identity))
        primary_ids = set(graph.units[item.campaign.primary_unit].element_ids)
        product_boxes.append(
            unit_visible_box(
                [element for element in item.identity if element["id"] in primary_ids],
                graph.assets_dir,
            )
        )
    active = [index for index in range(4) if any(parts[index] for parts in sections)]
    heights = {
        index: max(
            unit_visible_box(parts[index], graph.assets_dir).height if parts[index] else 0
            for parts in sections
        )
        for index in active
    }
    starts, y, previous = {}, 0.0, None
    for index in active:
        if previous is not None:
            gaps = [
                max(0.0, source[index].y - source[previous].y - source[previous].height) * scale
                for source in source_sections
                if source[index] is not None and source[previous] is not None
            ]
            short = prepared[0].short
            y += max(short * 0.02, min(short * 0.06, max(gaps, default=0)))
        starts[index] = y
        y += heights[index]
        previous = index
    baseline = max(
        product.y + product.height - frame.y
        for product, frame in zip(product_boxes, identity_boxes, strict=True)
    )
    groups = []
    for item, parts, product in zip(prepared, sections, product_boxes, strict=True):
        identity = deepcopy(item.identity)
        identity_visible = unit_visible_box(identity, graph.assets_dir)
        _translate(identity, -identity_visible.x, baseline - product.y - product.height)
        content = []
        for index in active:
            part = parts[index]
            if not part:
                continue
            box = unit_visible_box(part, graph.assets_dir)
            top = starts[index] + (heights[index] - box.height if index == 3 else 0)
            _translate(part, 0, top - box.y)
            content.extend(part)
        copy_box = unit_visible_box(content, graph.assets_dir)
        _translate(content, identity_visible.width + item.short * 0.05 - copy_box.x, 0)
        combined = identity + content
        frame = union(combined)
        if frame.width > item.cell.width + 1e-6:
            return None
        for element in combined:
            if element["type"] == "text":
                measured = get_measurer().measure(element, element["font_size"], element["width"])
                if (
                    measured.width > element["width"] + 0.01
                    or measured.height > element["height"] + 0.01
                ):
                    return None
        groups.append(combined)
    frames = [union(group) for group in groups]
    top = min(frame.y for frame in frames)
    bottom = max(frame.y + frame.height for frame in frames)
    height = bottom - top
    if height > prepared[0].cell.height + 1e-6:
        return None
    dy = prepared[0].cell.y + (prepared[0].cell.height - height) / 2 - top
    for item, group, frame in zip(prepared, groups, frames, strict=True):
        _translate(group, item.cell.x + (item.cell.width - frame.width) / 2 - frame.x, dy)
    return [element for group in groups for element in group]


def _aligned(prepared, preferred_scale, graph):
    result = _aligned_trial(prepared, preferred_scale, graph)
    if result is None:
        lower, upper, best = 0.0, preferred_scale, None
        for _ in range(18):
            scale = (lower + upper) / 2
            trial = _aligned_trial(prepared, scale, graph)
            if trial is None:
                upper = scale
            else:
                lower, best = scale, trial
        result = best
    return (
        None
        if result is None or any(element.get("font_size", 1) < 1 for element in result)
        else result
    )


def _fingerprint(design):
    return tuple(
        tuple(
            round(element.get(field, 0), 3)
            for field in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width")
        )
        for element in design["elements"]
    )


def generate_campaign_states(
    source: dict, target: dict, graph: SourceRelationshipGraph
) -> tuple[CampaignState, ...]:
    """Realize two to six unambiguous complete source campaigns in at most eight states.

    Caller supplies the original relationship graph. Unsupported ownership,
    unassigned foreground or source/graph mismatch declines this family. Each
    adaptive state uses one shared message scale across campaigns; long copy
    cannot silently shrink only its own offer and break global typography.
    """
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    if not 2 <= len(campaigns) <= MAX_CAMPAIGNS or graph.ambiguities:
        return ()
    if (
        tuple(element["id"] for element in source["elements"]) != graph.source_ids
        or source["canvas"] != {"width": graph.width, "height": graph.height}
        or any(
            not isinstance(target.get(key), int) or isinstance(target[key], bool) or target[key] < 1
            for key in ("width", "height")
        )
    ):
        return ()
    context = build_context(source)
    owned = [index for campaign in campaigns for index in campaign.unit_indices]
    cluster_ids = {cluster.id for cluster in graph.clusters}
    if (
        len(owned) != len(set(owned))
        or set(owned) != set(range(len(context.units)))
        or any(
            campaign.primary_unit not in campaign.unit_indices
            or not any(
                member.element_type == "image"
                for member in graph.units[campaign.primary_unit].members
            )
            or len(campaign.element_ids) != len(set(campaign.element_ids))
            or not set(campaign.member_cluster_ids) <= cluster_ids
            or set(campaign.element_ids)
            != {eid for index in campaign.unit_indices for eid in graph.units[index].element_ids}
            for campaign in campaigns
        )
    ):
        return ()
    count = len(campaigns)
    packings = [("columns", count), ("stacked", 1)]
    if count >= 3:
        packings.insert(1, ("wrapped", math.ceil(math.sqrt(count))))
    if target["width"] < target["height"]:
        packings.reverse()
    scale = max(1.0, math.sqrt(target["width"] * target["height"] / (graph.width * graph.height)))
    states, seen, prepared_cache = [], set(), {}
    if target["width"] >= target["height"] and _comparison_row(campaigns, graph, context):
        cells = _packing(target, count, count)
        prepared = _prepare(campaigns, cells, source, graph, context, "columns") if cells else None
        prepared_cache[("columns", count)] = prepared
        elements = _aligned(prepared, scale, graph) if prepared is not None else None
        if elements is not None:
            by_id = {element["id"]: element for element in _base(context, target) + elements}
            if set(by_id) == set(graph.source_ids):
                design = {
                    "canvas": dict(target),
                    "elements": [by_id[element["id"]] for element in source["elements"]],
                }
                axes = _axes(graph, campaigns, "columns")
                # Common rows are useful only while their extra whitespace
                # keeps each offer connected. Narrow cells can require a
                # different topology rather than an arbitrarily distant CTA.
                if not analyze_relationships(graph, design, dict(axes))["warnings"]:
                    states.append(
                        CampaignState(
                            "campaigns-columns-aligned",
                            "campaigns-columns/aligned",
                            axes,
                            json.dumps(design, sort_keys=True),
                        )
                    )
                    seen.add(_fingerprint(design))
    for mode in ("columns", "stacked", "source"):
        for packing, columns in packings:
            cells = _packing(target, count, columns)
            if not cells:
                continue
            key = (mode, columns)
            if key not in prepared_cache:
                prepared_cache[key] = _prepare(campaigns, cells, source, graph, context, mode)
            prepared = prepared_cache[key]
            if prepared is None:
                continue
            for typography in (1.0, 0.85):
                elements = _realize(prepared, scale * typography, mode, graph)
                if elements is None:
                    continue
                by_id = {element["id"]: element for element in _base(context, target) + elements}
                if set(by_id) != set(graph.source_ids):
                    continue
                design = {
                    "canvas": dict(target),
                    "elements": [by_id[element["id"]] for element in source["elements"]],
                }
                # Independent bounded fits can differ only in subpixel noise.
                # Retain one representative without changing exported geometry.
                fingerprint = _fingerprint(design)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                encoded = json.dumps(design, sort_keys=True)
                states.append(
                    CampaignState(
                        f"campaigns-{packing}-{mode}-{typography:g}",
                        f"campaigns-{packing}/{mode}",
                        _axes(graph, campaigns, mode),
                        encoded,
                    )
                )
                if len(states) == MAX_STATES:
                    return tuple(states)
    return tuple(states)
