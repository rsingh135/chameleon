"""Compile known visual-plan operations using source-owned deterministic fitters.

The menu is data, never executable instructions. A model chooses a known plan;
it cannot supply element JSON, replace artwork, or create crop authority.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy

from engine.eval import check_contract
from engine.eval.geometry import canvas_box, contains, rotated_aabb
from engine.resources import using_resources

from .backgrounds import generate_background_states
from .blocks import _message, _translate, _uniform
from .candidates import _base
from .geometry import Box, union
from .grouped import GroupedState
from .model import build_context
from .source_guides import _panel_place, _source_panels, _transform

MUTABLE = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


def _brands(graph):
    brands = [c for c in graph.clusters if c.kind == "brand"]
    messages = [c for c in graph.clusters if c.kind == "message"]
    if len(brands) != 1 or len(messages) != 1:
        return None
    brand, message = brands[0], messages[0]
    owned = set(brand.unit_indices) | set(message.unit_indices)
    if set(brand.unit_indices) & set(message.unit_indices):
        return None
    accents = []
    for unit in graph.units:
        if unit.index in owned:
            continue
        cx = (unit.box.x + unit.box.width / 2) / graph.width
        cy = (unit.box.y + unit.box.height / 2) / graph.height
        if unit.role != "badge" or cy >= 0.3 or not (cx < 0.25 or cx > 0.75):
            return None
        accents.append(unit.index)
    return brand, message, accents


def candidate_specs(source, target, graph):
    """Closed source-conditioned operations; no sample IDs or literal-copy rules."""
    from engine.visual_plan import descriptor

    families = []
    backgrounds = generate_background_states(source, target)
    background_specs = []
    for index in range(min(3, len(backgrounds) - 1)):
        before, after = backgrounds[index : index + 2]
        background_specs.append(
            descriptor(
                "background_caption",
                {"lower": before.name, "upper": after.name, "mix": 0.5},
                f"Full background photo with intermediate caption region {index + 1}",
            )
        )
    families.append(background_specs)
    brand = _brands(graph)
    brands = []
    if brand:
        ids = {"brand": brand[0].id, "message": brand[1].id}
        for topology, share, gap in (
            ("columns", 0.44, 0.06),
            ("stacked", 0.52, 0.04),
            ("columns", 0.52, 0.035),
            ("stacked", 0.62, 0.075),
        ):
            brands.append(
                descriptor(
                    "brand_message",
                    {**ids, "topology": topology, "share": share, "gap": gap},
                    (
                        "Brand and product beside related message"
                        if topology == "columns"
                        else "Centered brand and product above related message"
                    )
                    + f"; share {share:g}",
                )
            )
    families.append(brands)
    context = build_context(source)
    panels, loose = _source_panels(context)
    panel_specs = []
    if 2 <= len(panels) <= 6 and not loose and not context.decorations:
        tokens = [
            next(
                f"u{u.index}"
                for u in graph.units
                if panel.container.elements[0]["id"] in u.element_ids
            )
            for panel in panels
        ]
        for columns, gap in ((1, 0.025), (2, 0.035), (len(panels), 0.05), (1, 0.065)):
            panel_specs.append(
                descriptor(
                    "source_panels",
                    {"panels": tokens, "columns": columns, "gap": gap},
                    f"Keep ordered panel contents together in {columns} column(s), gap {gap:g}",
                )
            )
    families.append(panel_specs)
    # Give every supported family a place before secondary variants.
    return [
        family[i]
        for i in range(max(map(len, families), default=0))
        for family in families
        if i < len(family)
    ]


def _brand_message(source, target, graph, parameters):
    prepared = _brands(graph)
    if not prepared:
        return None
    brand, message, accents = prepared
    if parameters["brand"] != brand.id or parameters["message"] != message.id:
        return None
    context = build_context(source)
    width, height = target["width"], target["height"]
    short = min(width, height)
    pad, gap = short * 0.04, short * parameters["gap"]
    share, topology = parameters["share"], parameters["topology"]
    upper = math.sqrt(width * height / (graph.width * graph.height)) * 1.15
    header = max((context.units[i].box.height * upper for i in accents), default=0)
    top = pad + (header + gap if accents else 0)
    brand_units = [context.units[i] for i in brand.unit_indices]
    message_units = [context.units[i] for i in message.unit_indices]
    if topology == "columns":
        available = width - 2 * pad - gap
        image_cell = Box(pad, top, available * share, height - top - pad)
        copy_cell = Box(
            pad + image_cell.width + gap, top, available * (1 - share), height - top - pad
        )
    else:
        available = height - top - pad - gap
        image_cell = Box(pad, top, width - 2 * pad, available * share)
        copy_cell = Box(
            pad, top + image_cell.height + gap, width - 2 * pad, available * (1 - share)
        )
    originals = [e for unit in brand_units for e in unit.elements]
    source_box = union(originals)
    low, high, content, identity = 0.0, upper, None, None
    # The brand artwork and ALL text use one common source scale. A plan
    # changes placement/flow, never independently inflates a headline or product.
    for _ in range(22):
        scale = (low + high) / 2
        rigid = _transform(
            originals,
            scale,
            image_cell.x + image_cell.width / 2 - (source_box.x + source_box.width / 2) * scale,
            image_cell.y + image_cell.height / 2 - (source_box.y + source_box.height / 2) * scale,
        )
        candidate = _message(message_units, copy_cell.width, scale, short, graph.assets_dir)
        frame = union(rigid)
        fits = frame.width <= image_cell.width + 1e-6 and frame.height <= image_cell.height + 1e-6
        if candidate:
            box = union(candidate)
            fits = (
                fits
                and box.width <= copy_cell.width + 1e-6
                and box.height <= copy_cell.height + 1e-6
            )
            fits = fits and all(e.get("font_size", 1) >= 1 for e in candidate)
        else:
            fits = False
        if fits:
            low, content, identity = scale, candidate, rigid
        else:
            high = scale
    if content is None or identity is None:
        return None
    frame = union(content)
    _translate(
        content,
        copy_cell.x + copy_cell.width / 2,
        copy_cell.y + (copy_cell.height - frame.height) / 2 - frame.y,
    )
    elements = _base(context, target) + identity + content
    for index in accents:
        unit = context.units[index]
        placed = _uniform(unit, low, 0, 0, graph.assets_dir)
        frame = union(placed)
        left = unit.box.x + unit.box.width / 2 < graph.width / 2
        x = pad if left else width - pad - frame.width
        _translate(placed, x - frame.x, pad - frame.y)
        elements.extend(placed)
    axes = tuple((c.id, "vertical") for c in (brand, message))
    return elements, axes, ()


def _source_panel_grid(source, target, graph, parameters):
    context = build_context(source)
    panels, loose = _source_panels(context)
    if not panels or loose:
        return None
    token_panels = {
        next(
            f"u{u.index}" for u in graph.units if p.container.elements[0]["id"] in u.element_ids
        ): p
        for p in panels
    }
    if set(parameters["panels"]) != set(token_panels) or len(parameters["panels"]) != len(panels):
        return None
    ordered = [token_panels[token] for token in parameters["panels"]]
    columns = parameters["columns"]
    rows = math.ceil(len(ordered) / columns)
    width, height = target["width"], target["height"]
    padding, gap = min(width, height) * 0.025, min(width, height) * parameters["gap"]
    cell_width = (width - 2 * padding - gap * (columns - 1)) / columns
    cell_height = (height - 2 * padding - gap * (rows - 1)) / rows
    if min(cell_width, cell_height) <= 0:
        return None
    cells = [
        Box(
            padding + i % columns * (cell_width + gap),
            padding + i // columns * (cell_height + gap),
            cell_width,
            cell_height,
        )
        for i in range(len(ordered))
    ]
    low, high, selected = 0.0, math.sqrt(width * height / (graph.width * graph.height)) * 2, None
    for _ in range(24):
        scale = (low + high) / 2
        results = [
            _panel_place(panel, cell, scale) for panel, cell in zip(ordered, cells, strict=True)
        ]
        if all(results):
            low, selected = scale, results
        else:
            high = scale
    if not selected:
        return None
    elements = _base(context, target)
    exceptions = []
    for placed, notes in selected:
        elements.extend(placed)
        exceptions.extend(notes)
    return elements, (), tuple(exceptions)


def _background_midpoint(source, target, parameters):
    states = {s.name: s for s in generate_background_states(source, target)}
    if parameters["lower"] not in states or parameters["upper"] not in states:
        return None
    a, b = states[parameters["lower"]], states[parameters["upper"]]
    elements, t = [], parameters["mix"]
    for before, after in zip(a.design["elements"], b.design["elements"], strict=True):
        if {k: v for k, v in before.items() if k not in MUTABLE} != {
            k: v for k, v in after.items() if k not in MUTABLE
        }:
            return None
        element = deepcopy(before)
        for key in MUTABLE & before.keys():
            element[key] = before[key] + t * (after[key] - before[key])
        elements.append(element)
    return elements, a.cluster_axes, a.exceptions


def compile_spec(source, target, graph, spec):
    """Internal finite operation compiler; menu validation precedes public use."""
    family, parameters = spec["family"], spec["parameters"]
    try:
        if family == "brand_message":
            compiled = _brand_message(source, target, graph, parameters)
        elif family == "source_panels":
            compiled = _source_panel_grid(source, target, graph, parameters)
        elif family == "background_caption":
            compiled = _background_midpoint(source, target, parameters)
        else:
            return None
        if compiled is None:
            return None
        elements, axes, exceptions = compiled
        by_id = {e["id"]: e for e in elements}
        if len(by_id) != len(elements) or set(by_id) != set(graph.source_ids):
            return None
        design = {"canvas": dict(target), "elements": [by_id[e["id"]] for e in source["elements"]]}
        if any(
            not contains(canvas_box(target), rotated_aabb(e), slack=1e-6)
            for e in design["elements"]
        ):
            return None
        if check_contract(source, design, target_canvas=target):
            return None
        return GroupedState(
            "visual/" + spec["plan_id"],
            "visual-" + family,
            tuple(axes),
            json.dumps(design, sort_keys=True),
            tuple(exceptions),
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError):
        return None


def realize_plan(source, target, menu, decision, *, assets_dir=None, fonts_dir=None):
    """Return one newly compiled state or explicit keep; reject untrusted plans."""
    from engine.visual_plan import validate_bound_menu, validate_decision

    before = deepcopy(source)
    with using_resources(assets_dir, fonts_dir):
        graph = validate_bound_menu(source, target, menu)
        decision = validate_decision(decision, menu)
        if decision["plan_id"] == "keep":
            return ()
        spec = next(p for p in menu["plans"] if p["plan_id"] == decision["plan_id"])
        state = compile_spec(source, target, graph, spec)
        if state is None:
            raise ValueError("The selected visual plan no longer realizes a valid layout")
    if source != before:
        raise RuntimeError("Visual planning mutated the original source")
    return (state,)
