"""Parent-level alternatives with one shared foreground similarity scale.

Atomic source groups never stretch or receive independent font-size floors.
Containers may change flow and compress whitespace; text may rewrap at its
shared scale. The immutable hierarchy supplies membership, not candidate boxes.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass

from .appearance import unit_visible_box, visible_box
from .blocks import _peer_rows, _translate
from .candidates import _base
from .geometry import Box, bounds, intersection, union
from .hierarchy import infer_hierarchy, transform_cluster
from .model import build_context
from .relationships import build_relationship_graph
from .sibling_rows import aligned_sibling_rows
from .text import get_measurer

MAX_STATES = 12


@dataclass(frozen=True)
class GroupedState:
    name: str
    topology: str
    cluster_axes: tuple[tuple[str, str], ...]
    design_json: str
    # A standalone footer is an independent hierarchy root. Any readable-font
    # policy applied there is explicit; primary campaign leaves never get floors.
    exceptions: tuple[str, ...] = ()

    @property
    def design(self):
        return json.loads(self.design_json)


@dataclass(frozen=True)
class _Offer:
    campaign: object
    brand: object | None
    message: object
    rows: tuple
    identity_indices: tuple[int, ...]


def _primary_support(elements, assets):
    """Painted image support and complete text/control frames need breathing room."""
    boxes = [
        visible_box(element, assets) if element["type"] == "image" else bounds(element)
        for element in elements
        if element.get("opacity", 1) > 0
    ]
    boxes = [box for box in boxes if min(box.width, box.height) > 0]
    if not boxes:
        return None
    left, top = min(box.x for box in boxes), min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return Box(left, top, right - left, bottom - top)


def _primary_inset(source, graph, claimed):
    ids = {eid for index in claimed for eid in graph.units[index].element_ids}
    support = _primary_support([e for e in source["elements"] if e["id"] in ids], graph.assets_dir)
    source_short = min(graph.width, graph.height)
    fraction = (
        min(
            support.x,
            support.y,
            graph.width - support.x - support.width,
            graph.height - support.y - support.height,
        )
        / source_short
        if support
        else 0
    )
    # The source supplies a conservative container-padding preference. Extreme
    # source whitespace remains a soft span constraint rather than a giant inset.
    return max(0.04, min(0.08, fraction))


def _prepare(graph):
    campaigns = [cluster for cluster in graph.clusters if cluster.kind == "campaign"]
    if not campaigns or len(campaigns) > 6 or graph.ambiguities:
        return None
    by_id = {cluster.id: cluster for cluster in graph.clusters}
    offers, claimed = [], set()
    for campaign in campaigns:
        children = [by_id[name] for name in campaign.member_cluster_ids]
        messages = [cluster for cluster in children if cluster.kind == "message"]
        brands = [cluster for cluster in children if cluster.kind == "brand"]
        rows = tuple(cluster for cluster in children if cluster.kind == "badge_row")
        if len(messages) != 1 or len(brands) > 1:
            return None
        message, brand = messages[0], brands[0] if brands else None
        if any(
            edge.axis != "vertical"
            for edge in graph.edges
            if edge.cluster_id == message.id and edge.code == "reading_order"
        ):
            return None
        identity = brand.unit_indices if brand else (campaign.primary_unit,)
        indices = list(identity) + list(message.unit_indices)
        indices.extend(index for row in rows for index in row.unit_indices)
        if (
            None in identity
            or len(indices) != len(set(indices))
            or set(indices) != set(campaign.unit_indices)
            or claimed.intersection(indices)
        ):
            return None
        claimed.update(indices)
        offers.append(_Offer(campaign, brand, message, rows, identity))
    return tuple(offers), claimed


def _identity(offer, hierarchy, scale, axis, assets):
    nodes = [hierarchy.cluster(f"unit:{index}") for index in offer.identity_indices]
    elements, cursor = [], 0.0
    for position, node in enumerate(nodes):
        box = node.visible_box
        if min(box.width, box.height) <= 0:
            return None
        if position:
            previous = nodes[position - 1].visible_box
            source_gap = max(0, box.y - previous.y - previous.height)
            # Source whitespace is soft; element sizes still share exactly s.
            reference = min(previous.height, box.height)
            cursor += min(source_gap * 0.35, reference * 0.28) * scale
        x, y = (
            (-box.width * scale / 2, cursor)
            if axis == "vertical"
            else (cursor, -box.height * scale / 2)
        )
        elements.extend(transform_cluster(node, scale, x, y, anchor="visible"))
        cursor += (box.height if axis == "vertical" else box.width) * scale
    return elements


def _copy(offer, context, graph, hierarchy, scale, width, spacing=1.0):
    assets, measurer = graph.assets_dir, get_measurer()
    unit_index = {e["id"]: index for index, unit in enumerate(context.units) for e in unit.elements}
    units = [context.units[index] for index in offer.message.unit_indices]
    rows = [tuple(context.units[index] for index in row.unit_indices) for row in offer.rows]
    entries = [(unit_visible_box(unit.elements, assets), unit) for unit in units]
    entries.extend(
        (unit_visible_box([e for u in row for e in u.elements], assets), row) for row in rows
    )
    entries.sort(key=lambda item: item[0].y)
    alignment = next(
        (
            edge.alignment
            for edge in graph.edges
            if edge.cluster_id == offer.message.id and edge.code == "shared_axis"
        ),
        "center",
    )
    minimum_font = min(e["font_size"] for unit in units for e in unit.text)
    result, y, previous = [], 0.0, None
    for source_box, entry in entries:
        if previous is not None:
            gap = max(0, source_box.y - previous.y - previous.height)
            y += max(minimum_font * 0.35, min(minimum_font * 1.5, gap * 0.65)) * scale * spacing
        if isinstance(entry, tuple):
            source_gap = min(
                (
                    max(0, second.box.x - first.box.x - first.box.width)
                    for first, second in zip(entry, entry[1:], strict=False)
                ),
                default=minimum_font,
            )
            placed = _peer_rows(
                entry, width, scale, max(source_gap, minimum_font * 0.5) * scale, assets, alignment
            )
            if not placed:
                return None
            _translate(placed, 0, y)
            # The wrapping helper decides leaf positions only. Reconstruct each
            # entire leaf from its hierarchy snapshot with one exact transform.
            by_id = {element["id"]: element for element in placed}
            rebuilt = []
            row_indices = tuple(unit_index[unit.elements[0]["id"]] for unit in entry)
            for index in row_indices:
                node = hierarchy.cluster(f"unit:{index}")
                first = node.elements[0]
                position = by_id[first["id"]]
                x = position["x"] - (first["x"] - node.frame_box.x) * scale
                top = position["y"] - (first["y"] - node.frame_box.y) * scale
                rebuilt.extend(transform_cluster(node, scale, x, top))
            placed = rebuilt
        elif len(entry.elements) == 1 and entry.elements[0]["type"] == "text":
            element = deepcopy(entry.elements[0])
            element["font_size"] *= scale
            element.update(x=-width / 2, y=y, width=width)
            measured = measurer.measure(element, element["font_size"], width)
            if element.get("max_lines") is not None and measured.lines > element["max_lines"]:
                return None
            element["height"] = measured.height + 0.01
            placed = [element]
        else:
            # The source unit's entire local geometry gets one transform.
            index = unit_index[entry.elements[0]["id"]]
            node = hierarchy.cluster(f"unit:{index}")
            placed = transform_cluster(node, scale, -node.frame_box.width * scale / 2, y)
            box = unit_visible_box(placed, assets)
            if alignment != "center":
                left = -width / 2 if alignment == "start" else width / 2 - box.width
                _translate(placed, left - box.x, 0)
        result.extend(placed)
        visible = unit_visible_box(placed, assets)
        y, previous = visible.y + visible.height, source_box
    return result


def _cells(target, count, columns, top, bottom):
    pad = min(target.values()) * 0.04
    gap = min(target.values()) * 0.05
    rows = math.ceil(count / columns)
    width = (target["width"] - 2 * pad - gap * (columns - 1)) / columns
    height = (bottom - top - gap * (rows - 1)) / rows
    if min(width, height) <= 0:
        return ()
    cells = []
    for index in range(count):
        row, column = divmod(index, columns)
        row_count = min(columns, count - row * columns)
        left = (target["width"] - row_count * width - (row_count - 1) * gap) / 2
        cells.append(Box(left + column * (width + gap), top + row * (height + gap), width, height))
    return tuple(cells)


def _accessories(source, context, hierarchy, claimed, target, scale):
    width, height = target["width"], target["height"]
    pad, gap = min(width, height) * 0.04, min(width, height) * 0.025
    legal, supports, corners = [], [], []
    for index, unit in enumerate(context.units):
        if index in claimed:
            continue
        cx, cy = (
            (unit.box.x + unit.box.width / 2) / context.width,
            (unit.box.y + unit.box.height / 2) / context.height,
        )
        if unit.role == "legal" and len(unit.elements) == 1 and cy >= 0.85:
            legal.append(unit)
        elif unit.role in {"support", "logo"} and cy >= 0.75:
            supports.append((index, cx))
        elif unit.role == "badge" and cy <= 0.25 and (cx <= 0.25 or cx >= 0.75):
            corners.append((index, cx))
        else:
            return None
    elements, groups, exceptions, left, right = [], [], [], pad, width - pad
    for index, cx in supports:
        node = hierarchy.cluster(f"unit:{index}")
        placed = transform_cluster(node, scale, 0, 0)
        frame = union(placed)
        x = left if cx < 0.5 else right - frame.width
        _translate(placed, x - frame.x, height - pad - frame.height - frame.y)
        if cx < 0.5:
            left += frame.width + gap
        else:
            right -= frame.width + gap
        elements.extend(placed)
        groups.append(placed)
    y = height - pad
    for unit in reversed(legal):
        element = deepcopy(unit.elements[0])
        if element["type"] != "text" or right <= left:
            return None
        natural_size = element["font_size"] * scale
        floor = min(10.0, element["font_size"])
        element.update(x=left, width=right - left, font_size=max(natural_size, floor))
        if element["font_size"] > natural_size + 1e-9:
            exceptions.append(
                f"footer-text:{element['id']}: independent readable size "
                f"{element['font_size']:g}px instead of campaign-scaled {natural_size:.3f}px"
            )
        measured = get_measurer().measure(element, element["font_size"], element["width"])
        element.update(y=y - measured.height, height=measured.height + 0.01)
        y = element["y"] - gap
        elements.append(element)
        groups.append([element])
    bottom = y if legal else height - pad
    for index, cx in corners:
        placed = transform_cluster(hierarchy.cluster(f"unit:{index}"), scale, 0, 0)
        frame = union(placed)
        _translate(
            placed, (pad if cx < 0.5 else width - pad - frame.width) - frame.x, pad - frame.y
        )
        elements.extend(placed)
        groups.append(placed)
    return elements, groups, bottom, tuple(exceptions)


def _realize(
    source, target, graph, hierarchy, offers, claimed, columns, mode, brand_axis, share, scale
):
    context, assets = build_context(source), graph.assets_dir
    accessories = _accessories(source, context, hierarchy, claimed, target, scale)
    if accessories is None:
        return None
    fixed, fixed_groups, bottom, exceptions = accessories
    pad = min(target.values()) * 0.04
    cells = _cells(target, len(offers), columns, pad, bottom)
    if not cells:
        return None
    placed, axes, optical_blocks = [], {}, []
    optical_packing = len(offers) > 1
    spacing = 0.85 if optical_packing else 1.0
    identities = [_identity(offer, hierarchy, scale, brand_axis, assets) for offer in offers]
    if any(not identity for identity in identities):
        return None
    shared_lane = (
        max(unit_visible_box(identity, assets).width for identity in identities)
        if optical_packing and columns == 1 and mode == "columns"
        else None
    )
    for offer, cell, identity in zip(offers, cells, identities, strict=True):
        content = _copy(offer, context, graph, hierarchy, scale, cell.width * share, spacing)
        if not identity or not content:
            return None
        ib, mb = unit_visible_box(identity, assets), unit_visible_box(content, assets)
        font = max(
            e["font_size"]
            for index in offer.message.unit_indices
            for e in context.units[index].text
        )
        gap = font * scale * 0.8 * spacing
        if mode == "columns":
            if shared_lane is None:
                _translate(identity, -ib.x, -ib.y - ib.height / 2)
                _translate(content, ib.width + gap - mb.x, -mb.y - mb.height / 2)
            else:
                # Repeated offers retain one column system even when their
                # copy lengths differ. This changes containers, never sizes.
                _translate(identity, (shared_lane - ib.width) / 2 - ib.x, -ib.y - ib.height / 2)
                _translate(
                    content, shared_lane + gap + cell.width * share / 2, -mb.y - mb.height / 2
                )
        else:
            _translate(identity, -ib.x - ib.width / 2, -ib.y)
            _translate(content, -mb.x - mb.width / 2, ib.height + gap - mb.y)
        local = identity + content
        # Parent cells describe painted content, not each image's transparent
        # padding. Frame containment remains a whole-canvas hard requirement.
        frame = unit_visible_box(local, assets) if optical_packing else union(local)
        if shared_lane is not None:
            frame = Box(
                0,
                -max(ib.height, mb.height) / 2,
                shared_lane + gap + cell.width * share,
                max(ib.height, mb.height),
            )
        if frame.width > cell.width + 1e-6 or frame.height > cell.height + 1e-6:
            return None
        _translate(
            local,
            cell.x + (cell.width - frame.width) / 2 - frame.x,
            cell.y + (cell.height - frame.height) / 2 - frame.y,
        )
        # Use available parent whitespace before shrinking the children. A
        # source corner promotion must not force unrelated text frames beneath
        # its rotated AABB simply because the offer was centered by default.
        for fixed_group in fixed_groups:
            fixed_frame, local_frame = union(fixed_group), union(local)
            if (
                fixed_frame.y < target["height"] * 0.25
                and fixed_frame.x > target["width"] / 2
                and intersection(fixed_frame, local_frame) > 0.5
            ):
                shift = fixed_frame.x - pad * 0.5 - local_frame.x - local_frame.width
                if shift < 0 and local_frame.x + shift >= cell.x - 1e-6:
                    _translate(local, shift, 0)
        placed.extend(local)
        optical_blocks.append(local)
        axes[offer.message.id] = "vertical"
        if offer.brand:
            axes[offer.brand.id] = brand_axis
        axes[offer.campaign.id] = "horizontal" if mode == "columns" else "vertical"
        axes.update((row.id, "wrap") for row in offer.rows)
    if optical_packing:
        overall = union(placed)
        if overall.width > target["width"] or overall.height > target["height"]:
            return None
        support = _primary_support(placed, assets)
        inset = min(target.values()) * _primary_inset(source, graph, claimed)
        if support is None:
            return None
        # One parent translation must satisfy BOTH constraints: padded frames
        # remain on canvas, and painted/control content retains a real inset.
        # If no shared interval exists, the outer scale search must refit all
        # offers together; a CTA is never moved or shrunk independently.
        lower_x = max(-overall.x, inset - support.x)
        upper_x = min(
            target["width"] - overall.x - overall.width,
            target["width"] - inset - support.x - support.width,
        )
        lower_y = max(-overall.y, inset - support.y)
        upper_y = min(
            target["height"] - overall.y - overall.height,
            target["height"] - inset - support.y - support.height,
        )
        if lower_x > upper_x + 1e-6 or lower_y > upper_y + 1e-6:
            return None
        dx = max(lower_x, min(0, upper_x))
        dy = max(lower_y, min(0, upper_y))
        _translate(placed, dx, dy)
        for index, first in enumerate(optical_blocks):
            for second in optical_blocks[index + 1 :]:
                if (
                    intersection(unit_visible_box(first, assets), unit_visible_box(second, assets))
                    > 0.5
                ):
                    return None
    by_id = {element["id"]: element for element in placed}
    for fixed_group in fixed_groups:
        for index in claimed:
            members = [by_id[e["id"]] for e in context.units[index].elements]
            # Pixel/alpha bounds matter for image padding; ordinary controls
            # and rotated text also need the evaluator's conservative frames.
            image_pair = any(e["type"] == "image" for e in fixed_group + members)
            accessory = unit_visible_box(fixed_group, assets) if image_pair else union(fixed_group)
            block = unit_visible_box(members, assets) if image_pair else union(members)
            if intersection(accessory, block) > 0.5:
                return None
    elements = _base(context, target) + placed + fixed
    frame = union(elements)
    if (
        frame.x < -1e-6
        or frame.y < -1e-6
        or frame.x + frame.width > target["width"] + 1e-6
        or frame.y + frame.height > target["height"] + 1e-6
    ):
        return None
    if any(e.get("font_size", 1) < 1 or min(e["width"], e["height"]) <= 0.001 for e in elements):
        return None
    by_id = {element["id"]: element for element in elements}
    if set(by_id) != set(graph.source_ids):
        return None
    return (
        {
            "canvas": dict(target),
            "elements": [by_id[e["id"]] for e in source["elements"]],
        },
        tuple(axes.items()),
        exceptions,
    )


def _root_states(source, target, graph, hierarchy):
    """Pack explicit/spatial blocks without opening their internal coordinates.

    Loose connectors, solitary image/text units and unresolved ownership decline
    this path. A diagram with free arrows cannot be treated as independent cards.
    """
    roots = hierarchy.roots
    if (
        graph.ambiguities
        or not roots
        or len(roots) > 8
        or any(root.kind not in {"explicit", "spatial"} for root in roots)
    ):
        return ()
    measurer = get_measurer()
    for root in roots:
        for element in root.elements:
            if element["type"] == "text":
                measured = measurer.measure(element, element["font_size"], element["width"])
                if (
                    measured.width > element["width"] + 0.01
                    or measured.height > element["height"] + 0.01
                ):
                    return ()
    pad = min(target.values()) * 0.04
    preferred = len(roots) if target["width"] >= target["height"] else 1
    states = []
    for columns in dict.fromkeys((preferred, 1, math.ceil(math.sqrt(len(roots))))):
        cells = _cells(target, len(roots), columns, pad, target["height"] - pad)
        if not cells:
            continue
        scale = min(
            min(cell.width / root.frame_box.width, cell.height / root.frame_box.height)
            for root, cell in zip(roots, cells, strict=True)
        )
        elements = _base(build_context(source), target)
        for root, cell in zip(roots, cells, strict=True):
            elements.extend(
                transform_cluster(
                    root,
                    scale,
                    cell.x + (cell.width - root.frame_box.width * scale) / 2,
                    cell.y + (cell.height - root.frame_box.height * scale) / 2,
                )
            )
        if any(element.get("font_size", 1) < 1 for element in elements):
            continue
        by_id = {element["id"]: element for element in elements}
        if set(by_id) != set(graph.source_ids):
            continue
        design = {"canvas": dict(target), "elements": [by_id[e["id"]] for e in source["elements"]]}
        states.append(
            GroupedState(
                f"grouped-roots-{columns}",
                f"grouped-roots/{columns}",
                (),
                json.dumps(design, sort_keys=True),
            )
        )
    return tuple(states)


def generate_grouped_states(source, target, graph=None):
    """Offer up to twelve deterministic coupled-scale parent-layout proposals."""
    graph = graph or build_relationship_graph(source)
    prepared = _prepare(graph)
    hierarchy = infer_hierarchy(source, graph)
    if prepared is None:
        return _root_states(source, target, graph, hierarchy)
    offers, claimed = prepared
    count = len(offers)
    preferred = count if target["width"] / target["height"] >= 1.4 else 1
    packings = tuple(dict.fromkeys((preferred, 1, count)))
    by_packing = []
    for columns in packings:
        local_specs = []
        cell_ratio = target["width"] / columns / (target["height"] / math.ceil(count / columns))
        modes = ("columns", "stacked") if cell_ratio >= 1.2 else ("stacked", "columns")
        for mode in modes:
            shares = (
                (0.62, 0.72)
                if count == 1 and mode == "columns"
                else (0.92 if mode == "stacked" else 0.65,)
            )
            for axis in ("vertical", "horizontal"):
                for share in shares:
                    local_specs.append((columns, mode, axis, share))
        by_packing.append(local_specs)
    # Round-robin topology coverage prevents a wide source's first six local
    # variants from excluding a globally stacked solution on a custom canvas.
    specs = [
        options[index]
        for index in range(max(map(len, by_packing)))
        for options in by_packing
        if index < len(options)
    ]
    fonts = [e["font_size"] for unit in build_context(source).units for e in unit.text]
    minimum = 1 / min(fonts, default=1)
    upper = max(
        minimum, math.sqrt(target["width"] * target["height"] / (graph.width * graph.height)) * 2.25
    )
    states, seen = [], set()
    for spec_index, (columns, mode, axis, share) in enumerate(specs[: MAX_STATES // 2]):

        def fit(scale, columns=columns, mode=mode, axis=axis, share=share):
            return _realize(
                source, target, graph, hierarchy, offers, claimed, columns, mode, axis, share, scale
            )

        if fit(minimum) is None:
            continue
        low, high = minimum, upper
        for _ in range(18):
            middle = (low + high) / 2
            if fit(middle) is None:
                high = middle
            else:
                low = middle
        # Preserve six different cuts while reserving two slots for authored
        # sibling row constraints. Lower-scale duplicates are secondary.
        for scale in (low * 0.94, low) if spec_index < 4 else (low,):
            candidate = fit(scale)
            if candidate is None:
                continue
            design, axes, exceptions = candidate
            payload = json.dumps(design, sort_keys=True)
            if payload in seen:
                continue
            seen.add(payload)
            name = f"grouped-{columns}-{mode}-brand-{axis}-{share:g}-{len(states)}"
            states.append(
                GroupedState(name, f"grouped/{columns}/{mode}/{axis}", axes, payload, exceptions)
            )
    aligned_topologies = set()
    for state in sorted(
        states,
        key=lambda item: max(
            (
                element["font_size"]
                for element in item.design["elements"]
                if element["type"] == "text"
            ),
            default=0,
        ),
        reverse=True,
    ):
        if len(states) >= MAX_STATES or len(aligned_topologies) >= 2:
            break
        if (
            count < 2
            or int(state.topology.split("/")[1]) < count
            or "/columns/" not in state.topology
            or state.topology in aligned_topologies
        ):
            continue
        aligned = aligned_sibling_rows(
            source, target, graph, state.design, dict(state.cluster_axes)
        )
        if aligned is None:
            continue
        owned_ids = {eid for index in claimed for eid in graph.units[index].element_ids}
        support = _primary_support(
            [e for e in aligned["elements"] if e["id"] in owned_ids], graph.assets_dir
        )
        inset = min(target.values()) * _primary_inset(source, graph, claimed)
        if support is None or (
            support.x < inset - 1e-6
            or support.y < inset - 1e-6
            or support.x + support.width > target["width"] - inset + 1e-6
            or support.y + support.height > target["height"] - inset + 1e-6
        ):
            continue
        payload = json.dumps(aligned, sort_keys=True)
        if payload in seen:
            continue
        seen.add(payload)
        aligned_topologies.add(state.topology)
        states.append(
            GroupedState(
                f"{state.name}-rows",
                f"{state.topology}/aligned",
                state.cluster_axes,
                payload,
                state.exceptions,
            )
        )
    return tuple(states)
