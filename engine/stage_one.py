"""Strict stage-one containment, including originally cropped decorations."""

from __future__ import annotations

import json
import math
from copy import deepcopy

from engine.layout.geometry import bounds, contain


def strictly_in_bounds(design):
    try:
        width, height = (design["canvas"][key] for key in ("width", "height"))
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (width, height)):
            return False
        for element in design["elements"]:
            values = [element.get(k, 0) for k in ("x", "y", "width", "height", "rotation")]
            if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
                return False
            box = bounds(element)
            if (
                min(box.width, box.height) <= 0
                or box.x < -1e-6
                or box.y < -1e-6
                or box.x + box.width > width + 1e-6
                or box.y + box.height > height + 1e-6
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def contain_resolution_decorations(source, target, resolution):
    """Keep one foreground scale; only contain source-classified background shapes.

    Images, text and controls never receive an authored-overflow exemption in
    this profile. If those cannot retain the uniform source arrangement, let
    ordinary layout search handle the request instead.
    """
    from engine.eval import check_contract
    from engine.layout import build_context
    from engine.layout.grouped import GroupedState

    if resolution is None:
        return None
    design = resolution.design
    if strictly_in_bounds(design):
        return resolution
    context = build_context(source)
    decorative = {
        e["id"] for e in context.backgrounds + context.decorations if e["type"] == "shape"
    }
    changed = []
    for element in design["elements"]:
        if strictly_in_bounds({"canvas": target, "elements": [element]}):
            continue
        if element["id"] not in decorative:
            return None
        before = deepcopy(element)
        contain(element, target["width"], target["height"])
        ratio = element["width"] / before["width"]
        for field in ("corner_radius", "stroke_width"):
            if field in element:
                element[field] *= ratio
        changed.append(element["id"])
    if not strictly_in_bounds(design) or check_contract(source, design, target):
        return None
    return GroupedState(
        "resolution-scale/contained-decorations",
        "source-similarity",
        (),
        json.dumps(design, sort_keys=True, separators=(",", ":")),
        tuple("contained-source-decoration:" + identity for identity in changed),
    )
