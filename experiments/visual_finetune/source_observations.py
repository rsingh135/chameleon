"""Raw source typography facts, with no labels or inferred design purpose.

Population: text nodes with authored opacity in (0, 1] (missing means 1)
and finite font size in [1, 8192]. This is source eligibility, not proof of
pixel visibility, clipping, or occlusion. Rank is descending competition rank:
1 + the number of eligible text nodes with a strictly larger size. Exact numeric
ties share rank; [40, 20, 20, 10] ranks [1, 2, 2, 4]. Median uses the arithmetic
mean of the middle two sizes for an even population. IDs only locate a node.
"""

from __future__ import annotations

import math
from statistics import median

FIELDS = (
    "font_size",
    "font_size_over_source_short_axis",
    "font_size_over_scene_max",
    "font_size_over_scene_median",
    "font_size_rank",
    "visible_text_count",
    "font_weight",
    "font_style",
    "text_align",
)
_STYLES = {
    "font_weight": {"regular", "bold"},
    "font_style": {"normal", "italic"},
    "text_align": {"left", "center", "right"},
}


def _number(value, low, high):
    return type(value) in {int, float} and low <= value <= high and math.isfinite(value)


def _eligible_text(element):
    opacity = element.get("opacity", 1)
    return element.get("type") == "text" and _number(opacity, 0, 1) and opacity > 0


def observed_typography(source, element_id):
    """Return nine JSON-safe facts for a known source node; never mutate source.

    Images, shapes and opacity-ineligible text return explicit None fields.
    Missing/invalid sizes produce None size/ratio/rank fields; missing/invalid
    styles produce None, without inventing authored defaults. Source canvas axes
    must lie in [1, 1_000_000], with at most 4096 uniquely grounded nodes. Only
    size, opacity and schema-valid style fields are observed; no copy, geometry
    positions, IDs, groups, resource paths, roles or target metadata are features.
    """
    if not isinstance(source, dict) or not isinstance(source.get("canvas"), dict):
        raise ValueError("Typography observations require an editable source canvas")
    canvas, elements = source["canvas"], source.get("elements")
    if (
        any(not _number(canvas.get(k), 1, 1_000_000) for k in ("width", "height"))
        or not isinstance(elements, list)
        or not 1 <= len(elements) <= 4096
    ):
        raise ValueError("Typography source dimensions and node count must be finite and bounded")
    grounded = {}
    for element in elements:
        key = element.get("id") if isinstance(element, dict) else None
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 200
            or any(ord(c) < 32 for c in key)
            or key in grounded
            or element.get("type") not in ("text", "image", "shape")
        ):
            raise ValueError("Typography source must contain unique known nodes")
        grounded[key] = element
    if not isinstance(element_id, str) or element_id not in grounded:
        raise ValueError("Typography element grounding is not in the supplied source")
    selected = grounded[element_id]
    result = dict.fromkeys(FIELDS)
    if not _eligible_text(selected):
        return result
    for field, allowed in _STYLES.items():
        value = selected.get(field)
        result[field] = value if isinstance(value, str) and value in allowed else None
    sizes = [
        e["font_size"]
        for e in elements
        if _eligible_text(e) and _number(e.get("font_size"), 1, 8192)
    ]
    result["visible_text_count"] = len(sizes)
    size = selected.get("font_size")
    if _number(size, 1, 8192):
        result.update(
            font_size=size,
            font_size_over_source_short_axis=size / min(canvas["width"], canvas["height"]),
            font_size_over_scene_max=size / max(sizes),
            font_size_over_scene_median=size / median(sizes),
            font_size_rank=1 + sum(value > size for value in sizes),
        )
    return result
