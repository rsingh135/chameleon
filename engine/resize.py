"""Offline, deterministic design reflow with source-aware candidate selection."""

import json
import math
from copy import deepcopy
from functools import lru_cache
from typing import Any

from engine.layout import build_context, generate_candidates, get_measurer
from engine.layout.geometry import bounds, intersection


def _quality(source: dict, candidate: dict, context) -> float:
    """Local ranking until the evaluation track is integrated."""
    original = {e["id"]: e for e in source["elements"]}
    source_scale = math.sqrt(
        candidate["canvas"]["width"]
        * candidate["canvas"]["height"]
        / (source["canvas"]["width"] * source["canvas"]["height"])
    )
    background_ids = {e["id"] for e in context.backgrounds + context.decorations}
    unit_ids = {e["id"]: i for i, u in enumerate(context.units) for e in u.elements}
    roles = {e["id"]: u.role for u in context.units for e in u.elements}
    elements = [e for e in candidate["elements"] if e["id"] not in background_ids]
    score = 0.0
    for i, element in enumerate(elements):
        box = bounds(element)
        if element["type"] == "text":
            measured = get_measurer().measure(element, element["font_size"], element["width"])
            overflow = max(0.0, measured.width - element["width"])
            overflow += max(0.0, measured.height - element["height"])
            score -= overflow * 100
            relative = element["font_size"] / max(
                1.0, original[element["id"]]["font_size"] * source_scale
            )
            weight = {"headline": 3.0, "body": 2.0, "legal": 0.5}.get(roles[element["id"]], 1.0)
            score += weight * math.log(max(0.01, relative))
        for other in elements[i + 1 :]:
            if unit_ids[element["id"]] == unit_ids[other["id"]]:
                continue
            overlap = intersection(box, bounds(other))
            if overlap <= 0:
                continue
            source_overlap = intersection(
                bounds(original[element["id"]]), bounds(original[other["id"]])
            )
            intentional = source_overlap > 0 and not (element["type"] == other["type"] == "text")
            if not intentional:
                fraction = overlap / max(
                    1.0, min(box.width * box.height, other["width"] * other["height"])
                )
                score -= fraction * 20
    return score


@lru_cache(maxsize=64)
def _resize_cached(source_json: str, width: int, height: int) -> dict:
    source = json.loads(source_json)
    target_canvas = {"width": width, "height": height}
    context = build_context(source)
    candidates = generate_candidates(source, target_canvas)
    return max(candidates, key=lambda candidate: _quality(source, candidate, context))


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Adapt composition without changing the supplied assets, fonts or copy.

    Deterministic, offline, and does not mutate either argument. A bounded task
    cache speeds up repeated viewer requests; each caller receives its own copy.
    """
    return deepcopy(
        _resize_cached(
            json.dumps(source, sort_keys=True), target_canvas["width"], target_canvas["height"]
        )
    )
