"""Source-bound corner/footer placement with atomic similarity transforms.

Descriptors express source purpose and edge affinity; trusted geometry applies
those constraints. This generator does not change copy, assets or group topology.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy

from engine.eval import check_contract
from engine.stage_one import strictly_in_bounds

from .candidates import _base
from .geometry import Box, intersection, union
from .grouped import GroupedState
from .model import build_context

_FOOTER_DESTINATION = re.compile(
    r"(?:https?://|www\.)\S+|\b[^\s@]+@[^\s@]+\.[^\s@]+\b|"
    r"\b(?:[a-z0-9][a-z0-9-]*\.)+[a-z]{2,63}\b",
    re.I,
)
_FOOTER_LEGAL = re.compile(r"©|\bcopyright\b|\ball\s+rights\s+reserved\b", re.I)


def _strong_footer_evidence(source):
    height = source["canvas"]["height"]
    return any(
        element["type"] == "text"
        and element["y"] + element["height"] / 2 >= height * 0.8
        and (
            _FOOTER_DESTINATION.search(str(element.get("content", "")))
            or _FOOTER_LEGAL.search(str(element.get("content", "")))
        )
        for element in source["elements"]
    )


def _digest(value, *, compact=True):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            allow_nan=False,
            **({"separators": (",", ":")} if compact else {}),
        ).encode()
    ).hexdigest()


def _descriptors(source, semantics):
    if semantics is None:
        from engine.design_semantics import describe_design
        from engine.resources import get_resources

        resources = get_resources()
        semantics = describe_design(
            source, assets_dir=resources.assets_dir, fonts_dir=resources.fonts_dir
        )
    if not isinstance(semantics, dict) or semantics.get("source_digest") != _digest(source):
        raise ValueError("Semantic anchors must be bound to the unchanged source")
    if semantics.get("semantic_digest") != _digest(
        {key: value for key, value in semantics.items() if key != "semantic_digest"}
    ):
        raise ValueError("Semantic anchor descriptor digest changed after binding")
    raw = semantics.get("elements")
    rows = list(raw.values()) if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or len(rows) != len(source["elements"]):
        raise ValueError("Semantic anchors require one descriptor per source element")
    by_id = {row.get("id", row.get("element_id")): row for row in rows if isinstance(row, dict)}
    if len(by_id) != len(rows) or set(by_id) != {e["id"] for e in source["elements"]}:
        raise ValueError("Semantic anchors contain missing, duplicate or unknown identities")
    return semantics, by_id


def _anchor(row):
    values = row.get("anchors", [])
    if isinstance(values, list):
        value = {"edges": [], "insets_short_axis": {}}
        for item in values:
            if not isinstance(item, dict):
                raise ValueError("Semantic anchors must be bounded descriptor objects")
            value["edges"].extend(item.get("edges", []))
            value["insets_short_axis"].update(item.get("insets_short_axis", {}))
            for key in ("horizontal", "vertical"):
                if key in item:
                    value[key] = item[key]
    else:
        value = values
    if not isinstance(value, dict):
        value = {"edges": value}
    edges = value.get("edges", [])
    if isinstance(edges, dict):
        edges = list(edges.values())
    if isinstance(edges, str):
        edges = edges.replace("-", " ").split()
    edges = set(edges) if isinstance(edges, (list, tuple, set)) else set()
    horizontal = value.get("horizontal", value.get("x", "none"))
    vertical = value.get("vertical", value.get("y", "none"))
    if horizontal not in {"left", "right", "center"}:
        horizontal = next((edge for edge in ("left", "right") if edge in edges), "none")
    if vertical not in {"top", "bottom", "center"}:
        vertical = next((edge for edge in ("top", "bottom") if edge in edges), "none")
    insets = value.get("insets_short_axis", row.get("insets_short_axis", {}))
    return horizontal, vertical, insets if isinstance(insets, dict) else {}


def _units(source, descriptors, background_ids):
    ids = [e["id"] for e in source["elements"] if e["id"] not in background_ids]
    parents = {key: key for key in ids}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    memberships = {}
    for element in source["elements"]:
        if element["id"] not in parents:
            continue
        for group in (element.get("group_id"), descriptors[element["id"]].get("rigid_group")):
            if not isinstance(group, str) or not group:
                continue
            previous = memberships.setdefault(group, element["id"])
            parents[root(element["id"])] = root(previous)
    groups = {}
    for key in ids:
        groups.setdefault(root(key), []).append(key)
    anchored, primary = [], []
    originals = {e["id"]: e for e in source["elements"]}
    for members in groups.values():
        roles = {descriptors[key].get("purpose", "unknown") for key in members}
        candidates = []
        for key in members:
            row = descriptors[key]
            horizontal, vertical, insets = _anchor(row)
            purpose = row.get("purpose", "unknown")
            if purpose in {
                "brand_mark",
                "corner_marker",
                "footer_destination",
                "footer_legal",
                "action",
                "badge",
            } and (horizontal in {"left", "right"} or vertical in {"top", "bottom"}):
                candidates.append(
                    (
                        purpose in {"footer_destination", "footer_legal"},
                        horizontal,
                        vertical,
                        insets,
                        key,
                    )
                )
        # A footer inside a broad hero group cannot pin that complete campaign
        # to the page edge. Preserve the atomic group and its central purpose.
        if candidates and not roles.intersection(
            {"primary_visual", "hero_visual", "headline", "body"}
        ):
            verticals = {item[2] for item in candidates} - {"none", "center"}
            horizontals = {item[1] for item in candidates} - {"none", "center"}
            if len(verticals) > 1 or len(horizontals) > 1:
                primary.extend(members)
                continue
            _footer, horizontal, vertical, insets, reference = max(
                candidates, key=lambda item: item[0]
            )
            anchored.append(
                {
                    "ids": members,
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "insets": insets,
                    "reference_id": reference,
                }
            )
        else:
            primary.extend(members)
    if primary:
        anchored.append(
            {"ids": primary, "horizontal": "center", "vertical": "center", "insets": {}}
        )
    for index, unit in enumerate(anchored):
        elements = [originals[key] for key in unit["ids"]]
        source_visible = [Box(*descriptors[key]["visible_box"]) for key in unit["ids"]]
        left, top = min(box.x for box in source_visible), min(box.y for box in source_visible)
        right = max(box.x + box.width for box in source_visible)
        bottom = max(box.y + box.height for box in source_visible)
        unit.update(
            name=f"semantic-unit:{index}",
            frame=union(elements),
            visible=Box(left, top, right - left, bottom - top),
        )
        if unit.get("reference_id"):
            unit["reference"] = Box(*descriptors[unit["reference_id"]]["visible_box"])
    return anchored


def _insets(unit, source, target):
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    short, target_short = min(sw, sh), min(target["width"], target["height"])
    box = unit["visible"]
    measured = {
        "left": box.x / short,
        "right": (sw - box.x - box.width) / short,
        "top": box.y / short,
        "bottom": (sh - box.y - box.height) / short,
    }
    result = {}
    for edge, fallback in measured.items():
        value = unit["insets"].get(edge, fallback)
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("Semantic anchor insets must be finite source fractions")
        result[edge] = max(0, min(0.3, value)) * target_short
    return result


def _transform(elements, unit, scale, source, target):
    box, frame = unit["visible"], unit["frame"]
    reference = unit.get("reference", box)
    tw, th = target["width"], target["height"]
    insets = _insets(unit, source, target)
    horizontal, vertical = unit["horizontal"], unit["vertical"]
    x = (
        insets["left"]
        if horizontal == "left"
        else tw - insets["right"] - box.width * scale
        if horizontal == "right"
        else (tw - box.width * scale) / 2
    )
    y = (
        insets["top"]
        if vertical == "top"
        else th - insets["bottom"] - box.height * scale
        if vertical == "bottom"
        else (th - box.height * scale) / 2
    )
    # Pin the described leaf's ink, including when a larger plate belongs to
    # that same rigid group. Every member still shares one complete transform.
    if horizontal == "left":
        x -= (reference.x - box.x) * scale
    elif horizontal == "right":
        x += (box.x + box.width - reference.x - reference.width) * scale
    elif unit.get("reference_id"):
        x += (box.x + box.width / 2 - reference.x - reference.width / 2) * scale
    if vertical == "top":
        y -= (reference.y - box.y) * scale
    elif vertical == "bottom":
        y += (box.y + box.height - reference.y - reference.height) * scale
    if "target_origin" in unit:
        x, y = unit["target_origin"]
    # One complete-unit translation can contain transparent/unused frame padding.
    # Never independently clamp members, which would alter a rigid source group.
    fx, fy = x + (frame.x - box.x) * scale, y + (frame.y - box.y) * scale
    if frame.width * scale > tw or frame.height * scale > th:
        return None
    dx = max(0, -fx) - max(0, fx + frame.width * scale - tw)
    dy = max(0, -fy) - max(0, fy + frame.height * scale - th)
    if (
        horizontal in {"left", "right"}
        and abs(dx) > 1e-6
        or vertical in {"top", "bottom"}
        and abs(dy) > 1e-6
    ):
        return None
    x += dx
    y += dy
    placed = []
    for original in elements:
        element = deepcopy(original)
        element["x"] = x + (element["x"] - box.x) * scale
        element["y"] = y + (element["y"] - box.y) * scale
        for key in ("width", "height", "font_size", "corner_radius", "stroke_width"):
            if key in element:
                element[key] *= scale
        placed.append(element)
    return placed


def _support(unit, placed, originals, scale):
    first = placed[0]
    original = originals[first["id"]]
    box = unit["visible"]
    return Box(
        first["x"] + (box.x - original["x"]) * scale,
        first["y"] + (box.y - original["y"]) * scale,
        box.width * scale,
        box.height * scale,
    )


def _layout_at_scale(source, target, units, originals, scale):
    """Fit the primary support into unoccupied vertical intervals at its x span.

    Source ink is projected through an exact similarity transform. No native
    rasterization or full-frame marketing-score loop occurs during this search.
    """
    tw, th = target["width"], target["height"]
    placed_units = []
    for unit in (u for u in units if u.get("reference_id")):
        placed = _transform([originals[key] for key in unit["ids"]], unit, scale, source, target)
        if placed is None:
            return None
        placed_units.append((unit, placed, _support(unit, placed, originals, scale)))
    for unit in (u for u in units if not u.get("reference_id")):
        box = unit["visible"]
        x = (tw - box.width * scale) / 2
        padding = min(tw, th) * 0.02
        intervals = [(padding, th - padding)]
        for anchored, _placed, support in placed_units:
            overlap_x = min(x + box.width * scale, support.x + support.width) - max(x, support.x)
            if overlap_x <= min(box.width * scale, support.width) * 0.05:
                continue
            fonts = [
                originals[key]["font_size"] * scale
                for key in unit["ids"] + anchored["ids"]
                if originals[key]["type"] == "text"
            ]
            gutter = max(padding, min(fonts, default=padding) * 0.6)
            low, high = support.y - gutter, support.y + support.height + gutter
            remaining = []
            for top, bottom in intervals:
                if high <= top or low >= bottom:
                    remaining.append((top, bottom))
                else:
                    if low > top:
                        remaining.append((top, low))
                    if high < bottom:
                        remaining.append((high, bottom))
            intervals = remaining
        viable = [
            (top, bottom) for top, bottom in intervals if bottom - top + 1e-6 >= box.height * scale
        ]
        if not viable:
            return None
        top, bottom = max(viable, key=lambda band: band[1] - band[0])
        positioned = {**unit, "target_origin": (x, (top + bottom - box.height * scale) / 2)}
        placed = _transform(
            [originals[key] for key in unit["ids"]], positioned, scale, source, target
        )
        if placed is None:
            return None
        placed_units.append((unit, placed, _support(unit, placed, originals, scale)))
    for i, (first, _elements, a) in enumerate(placed_units):
        for second, _other, b in placed_units[i + 1 :]:
            old = intersection(first["visible"], second["visible"])
            if (
                old <= 1e-6
                and intersection(a, b) / max(1e-9, min(a.width * a.height, b.width * b.height))
                > 0.01
            ):
                return None
    return [element for _unit, elements, _box in placed_units for element in elements]


def generate_semantic_anchor_states(source, target, semantics=None):
    """Generate bounded alternatives from pinned source purposes and groups."""
    if source["canvas"] == target:
        return []
    original_aspect = source["canvas"]["width"] / source["canvas"]["height"]
    target_aspect = target["width"] / target["height"]
    if max(original_aspect / target_aspect, target_aspect / original_aspect) < 1.12:
        return []
    strong_footer = _strong_footer_evidence(source)
    # A corner badge and ordinary bottom CTA do not justify replacing normal
    # composition search. This metadata-only check precedes native ink work.
    if semantics is None and not strong_footer:
        return []
    semantics, descriptors = _descriptors(source, semantics)
    if not strong_footer and not any(
        row.get("purpose") in {"footer_destination", "footer_legal"} for row in descriptors.values()
    ):
        return []
    context = build_context(source)
    background_ids = {e["id"] for e in context.backgrounds}
    background_ids.update(
        e["id"] for e in context.decorations if not descriptors[e["id"]].get("anchors")
    )
    # A background paint cannot be stretched apart from a grouped foreground.
    grouped = {}
    for e in source["elements"]:
        if e.get("group_id"):
            grouped.setdefault(e["group_id"], set()).add(e["id"])
    if any(members & background_ids and members - background_ids for members in grouped.values()):
        return []
    units = _units(source, descriptors, background_ids)
    if len(units) < 2 or not any(
        u["horizontal"] != "center" or u["vertical"] != "center" for u in units
    ):
        return []
    originals = {e["id"]: e for e in source["elements"]}
    short = min(target["width"], target["height"])
    upper = short / min(source["canvas"].values())
    for unit in units:
        box = unit["frame"]
        upper = min(upper, target["width"] / box.width, target["height"] / box.height)
    low, high = 0.0, upper
    for _ in range(28):
        middle = (low + high) / 2
        if _layout_at_scale(source, target, units, originals, middle) is None:
            high = middle
        else:
            low = middle
    if low <= 1e-8:
        return []
    result, seen = [], set()
    for fraction in (1.0, 0.92):
        actual_scale = low * fraction * (1 - 1e-6)
        for _attempt in range(3):
            foreground = _layout_at_scale(source, target, units, originals, actual_scale)
            if foreground is not None:
                elements = _base(context, target) + foreground
                by_id = {e["id"]: e for e in elements}
                if set(by_id) != set(originals):
                    return []
                design = {
                    "canvas": dict(target),
                    "elements": [by_id[e["id"]] for e in source["elements"]],
                }
                if strictly_in_bounds(design) and not check_contract(source, design, target):
                    encoded = json.dumps(
                        design, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    if encoded not in seen:
                        axes = tuple(
                            (
                                u["name"],
                                "vertical"
                                if u["frame"].height >= u["frame"].width
                                else "horizontal",
                            )
                            for u in units
                        )
                        result.append(
                            GroupedState(
                                f"semantic-anchors/usable-band-{fraction:.2f}",
                                "source-semantic-anchors",
                                axes,
                                encoded,
                                (
                                    f"shared-foreground-scale:{actual_scale:.9g}",
                                    f"semantic-source:{semantics['source_digest']}",
                                    "source-short-axis-insets;atomic-groups;visible-usable-band",
                                ),
                            )
                        )
                        seen.add(encoded)
                    break
            actual_scale *= 0.98
    return result
