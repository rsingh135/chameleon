"""Source-derived checks for fine dividers, independent of element roles."""

from __future__ import annotations

import math


def analyze_rules(source: dict, candidate: dict) -> dict:
    """A divider may grow uniformly; it must not become a heavy rectangle."""
    outputs = {element["id"]: element for element in candidate["elements"]}
    errors, distorted = [], []
    for original in source["elements"]:
        if original["type"] != "shape" or original.get("shape_type") not in {"rectangle", "line"}:
            continue
        width, height = original["width"], original["height"]
        if min(width, height) <= 0:
            continue
        if max(width, height) / min(width, height) < 20:
            continue
        if min(width, height) > min(source["canvas"].values()) * 0.01:
            continue
        output = outputs.get(original["id"], {})
        new_width, new_height = output.get("width", 0), output.get("height", 0)
        error = (
            abs(math.log((new_width / width) / (new_height / height)))
            if min(new_width, new_height) > 0
            else math.inf
        )
        errors.append(error)
        if error > math.log(1.05):
            distorted.append({"element_id": original["id"], "aspect_log_error": error})
    return {
        "active": bool(errors),
        "rule_count": len(errors),
        "distorted_rules": distorted,
        "max_log_error": max(errors, default=0.0),
        "score": math.exp(-sum(errors) / len(errors)) if errors else 1.0,
    }
