"""Immutable, source-only evidence for repeated design containers.

Only opaque native rectangles and pixel-proven flat rectangular vector plates
may adapt their frames. Membership uses original stacking and occupied support;
no sample IDs, content labels or target geometry enter this inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageColor

from engine.resources import get_resources, using_resources

from .appearance import unit_visible_box
from .geometry import Box, intersection
from .model import build_context


@dataclass(frozen=True)
class SourceContainer:
    frame_id: str
    frame_ids: tuple[str, ...]
    member_ids: tuple[str, ...]
    box: Box
    adaptive: bool
    confidence: float
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class SourceContainerProfile:
    containers: tuple[SourceContainer, ...]
    loose_ids: tuple[str, ...]
    source_digest: str
    assets_dir: str
    fonts_dir: str


@lru_cache(maxsize=64)
def _rectangular_plate_pixels(path, fingerprint):
    """Prove a vector raster is a flat rectangular container, not illustration.

    The entire inset is one opaque color. Every edge strip is constant along
    its long axis; only corner antialiasing may differ. Photographs, lettering,
    diagrams and decorative strokes therefore stay rigid.
    """
    del fingerprint
    try:
        with Image.open(path) as original:
            if min(original.size) < 12 or max(original.size) > 4096:
                return False
            image = original.convert("RGBA")
        width, height = image.size
        inset = max(2, math.ceil(min(width, height) * 0.025))
        interior = image.crop((inset, inset, width - inset, height - inset))
        colors = interior.getcolors(1)
        if not colors or colors[0][1][3] != 255:
            return False
        for x in [*range(inset), *range(width - inset, width)]:
            if image.crop((x, inset, x + 1, height - inset)).getcolors(1) is None:
                return False
        for y in [*range(inset), *range(height - inset, height)]:
            if image.crop((inset, y, width - inset, y + 1)).getcolors(1) is None:
                return False
        return True
    except (OSError, ValueError):
        return False


def _flat_vector_container(element):
    if (
        element["type"] != "image"
        or element.get("source_art", {}).get("kind") != "svg-nontext"
        or element.get("rotation", 0)
        or element.get("opacity", 1) < 0.99
        or element.get("fit", "contain") != "fill"
    ):
        return False
    directory = Path(get_resources().assets_dir).resolve()
    path = (directory / element.get("asset_id", "")).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        return False
    stat = path.stat()
    return _rectangular_plate_pixels(str(path), (stat.st_mtime_ns, stat.st_size))


def _native_frame(element):
    if (
        element["type"] != "shape"
        or element.get("shape_type") != "rectangle"
        or element.get("rotation", 0)
        or element.get("opacity", 1) < 0.99
    ):
        return False
    fill = str(element.get("fill_color", element.get("fill", "")))
    if fill.lower() in {"", "none", "transparent"}:
        return False
    rgba = re.fullmatch(
        r"rgba\(\s*([.\d]+)\s*,\s*([.\d]+)\s*,\s*([.\d]+)\s*,\s*([.\d]+)\s*\)",
        fill,
    )
    if rgba is not None:
        red, green, blue, alpha = map(float, rgba.groups())
        return all(0 <= value <= 255 for value in (red, green, blue)) and 0.99 <= alpha <= 1
    try:
        return ImageColor.getcolor(fill, "RGBA")[3] == 255
    except ValueError:
        return False


def infer_source_containers(source=None, *, context=None):
    """Freeze source enclosure/flat-frame evidence under current resources.

    Pass the original source for a fingerprinted reusable diagnostic profile.
    A prebuilt context is accepted by the proposal factory to avoid repeated
    grouping. Neither input nor resource pixels are changed.
    """
    if context is None:
        if source is None:
            raise ValueError("Source design or source context is required.")
        context = build_context(source)
    resources = get_resources()
    digest = (
        hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if source is not None
        else ""
    )
    units = context.units
    support = {id(u): unit_visible_box(u.elements) for u in units}

    def covered(unit, container):
        box = support[id(unit)]
        return (
            box.width * box.height > 0
            and intersection(box, container.box) >= box.width * box.height * 0.9
        )

    proof = {}
    for unit in units:
        if len(unit.elements) != 1:
            continue
        element = unit.elements[0]
        native = _native_frame(element)
        vector = not native and _flat_vector_container(element)
        box = unit.box
        if (
            not (native or vector)
            or not context.width * 0.1 <= box.width <= context.width * 0.65
            or box.height < context.height * 0.15
        ):
            continue
        texts = [
            e
            for child in units
            if child is not unit
            and covered(child, unit)
            and min(e.get("z_index", 0) for e in child.elements) > element.get("z_index", 0)
            for e in child.text
        ]
        if len(texts) >= 4:
            proof[id(unit)] = (
                "native-opaque-rectangle"
                if native
                else "pixel-proven-flat-rectangular-vector-plate"
            )
    candidates = [u for u in units if id(u) in proof]
    roots = [
        u
        for u in candidates
        if proof[id(u)] == "native-opaque-rectangle" and u.box.height >= context.height * 0.8
    ]
    if len(roots) < 2:
        roots = [
            u
            for u in candidates
            if not any(
                o is not u
                and o.box.width * o.box.height > u.box.width * u.box.height * 1.05
                and covered(u, o)
                for o in candidates
            )
        ]
    if not 2 <= len(roots) <= 12:
        return SourceContainerProfile(
            (), (), digest, str(resources.assets_dir), str(resources.fonts_dir)
        )
    assigned = {id(root): [] for root in roots}
    loose = []
    for unit in units:
        if any(unit is root for root in roots):
            continue
        containers = [
            root
            for root in roots
            if covered(unit, root)
            and max(e.get("z_index", 0) for e in root.elements)
            < min(e.get("z_index", 0) for e in unit.elements)
        ]
        if containers:
            owner = min(containers, key=lambda r: r.box.width * r.box.height)
            assigned[id(owner)].append(unit)
        else:
            loose.append(unit)
    if any(
        u.text and context.height * 0.2 < u.box.y + u.box.height / 2 < context.height * 0.8
        for u in loose
    ):
        return SourceContainerProfile(
            (), (), digest, str(resources.assets_dir), str(resources.fonts_dir)
        )
    lanes = []
    for root in sorted(roots, key=lambda u: u.box.x):
        if (
            lanes
            and abs(root.box.x - lanes[-1][0].box.x)
            < min(root.box.width, lanes[-1][0].box.width) * 0.2
        ):
            lanes[-1].append(root)
        else:
            lanes.append([root])
    ordered = [u for lane in lanes for u in sorted(lane, key=lambda u: u.box.y)]
    records = []
    for root in ordered:
        children = assigned[id(root)]
        if not children:
            return SourceContainerProfile(
                (), (), digest, str(resources.assets_dir), str(resources.fonts_dir)
            )
        child_texts = [e for u in children for e in u.text]
        nested_frames = tuple(
            e["id"]
            for u in children
            for e in u.elements
            if _native_frame(e)
            and not e.get("group_id")
            and e["width"] >= root.box.width * 0.88
            and e["height"] >= root.box.height * 0.06
            and any(
                text.get("z_index", 0) > e.get("z_index", 0)
                and (
                    (ink := unit_visible_box([text])).width * ink.height > 0
                    and intersection(ink, unit_visible_box([e])) >= ink.width * ink.height * 0.9
                )
                for text in child_texts
            )
        )
        root_id = root.elements[0]["id"]
        records.append(
            SourceContainer(
                root_id,
                (root_id, *nested_frames),
                tuple(e["id"] for u in children for e in u.elements),
                root.box,
                True,
                0.99,
                (proof[id(root)], "later-layer-occupied-enclosure", "repeated-source-panels"),
            )
        )
    return SourceContainerProfile(
        tuple(records),
        tuple(e["id"] for u in loose for e in u.elements),
        digest,
        str(resources.assets_dir),
        str(resources.fonts_dir),
    )


_GEOMETRY_FIELDS = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


@lru_cache(maxsize=32)
def _captured_source(source_json, assets, fonts):
    source = json.loads(source_json)
    with using_resources(assets, fonts):
        profile = infer_source_containers(source)
        frame_ids = {eid for c in profile.containers for eid in c.frame_ids}
        units = tuple(
            (
                tuple(e["id"] for e in unit.elements if e["id"] not in frame_ids),
                unit_visible_box([e for e in unit.elements if e["id"] not in frame_ids]),
            )
            for unit in build_context(source).units
            if any(e["id"] not in frame_ids for e in unit.elements)
        )
        leaves = tuple(
            ((element["id"],), unit_visible_box([element]))
            for element in source["elements"]
            if element["id"] not in frame_ids
        )
    return profile, units, leaves


def _coverage(inner, outer):
    return intersection(inner, outer) / max(inner.width * inner.height, 1e-9)


def _joined(boxes):
    left, top = min(b.x for b in boxes), min(b.y for b in boxes)
    right, bottom = max(b.x + b.width for b in boxes), max(b.y + b.height for b in boxes)
    return Box(left, top, right - left, bottom - top)


def _padding(frame, content):
    return (
        content.x - frame.x,
        content.y - frame.y,
        frame.x + frame.width - content.x - content.width,
        frame.y + frame.height - content.y - content.height,
    )


def _element_scale(before, after):
    if before["type"] == "text":
        return after["font_size"] / before["font_size"]
    return math.sqrt(after["width"] * after["height"] / (before["width"] * before["height"]))


def analyze_source_containers(source, candidate, profile=None):
    """Permit only source-proven frame adaptation that retains physical support.

    Membership and paint identity remain immutable. Child occupied bounds must
    retain their source enclosure, and at least a quarter of positive authored
    padding after the common child scale. Extra frame space may adapt to the
    target; it cannot introduce new overlap between independent containers.
    Native radii/strokes follow child scale, rather than a stretched frame axis.
    Child rigidity is deliberately left to the ordinary proportion diagnostic.
    """
    from statistics import median

    from engine.eval.contract import ASPECT_TOLERANCE

    from .geometry import bounds

    source_json = json.dumps(source, sort_keys=True)
    resources = get_resources()
    assets = str(resources.assets_dir) if profile is None else profile.assets_dir
    fonts = str(resources.fonts_dir) if profile is None else profile.fonts_dir
    captured, units, leaves = _captured_source(source_json, assets, fonts)
    profile = captured if profile is None else profile
    metrics = {
        "container_count": len(profile.containers),
        "accepted_frame_count": 0,
        "detached_count": 0,
        "unsupported_deformation_count": 0,
        "max_containment_loss": 0.0,
        "max_new_container_overlap": 0.0,
        "max_frame_style_log_error": 0.0,
    }
    result = {
        "active": bool(profile.containers),
        "accepted": True,
        "accepted_frame_ids": [],
        "warnings": [],
        "containers": [],
        "metrics": metrics,
    }
    if not profile.containers:
        return result

    def warn(code, ids, message):
        result["warnings"].append(
            {"code": code, "element_ids": list(ids), "message": message, "confidence": 0.99}
        )

    unsupported = "source_container_unsupported_deformation"
    detached = "source_container_detached"
    originals = {e["id"]: e for e in source["elements"]}
    outputs = {e["id"]: e for e in candidate["elements"]}
    frame_ids = {eid for c in profile.containers for eid in c.frame_ids}
    if (
        profile.source_digest != hashlib.sha256(source_json.encode()).hexdigest()
        or profile != captured
        or len(originals) != len(source["elements"])
        or len(outputs) != len(candidate["elements"])
        or set(originals) != set(outputs)
        or not frame_ids.issubset(originals)
    ):
        warn(
            unsupported,
            sorted(frame_ids),
            "Container comparison requires its immutable source profile "
            "and complete original membership.",
        )
    else:
        relevant = frame_ids | {eid for c in profile.containers for eid in c.member_ids}
        for eid in sorted(relevant):
            before, after = originals[eid], outputs[eid]
            if {k: v for k, v in before.items() if k not in _GEOMETRY_FIELDS} != {
                k: v for k, v in after.items() if k not in _GEOMETRY_FIELDS
            } or any(
                not isinstance(after.get(k), (int, float))
                or isinstance(after[k], bool)
                or not math.isfinite(after[k])
                or (k in {"width", "height", "font_size"} and after[k] <= 0)
                or (k in {"corner_radius", "stroke_width"} and after[k] < 0)
                for k in (
                    "x",
                    "y",
                    "width",
                    "height",
                    *(["font_size"] if before["type"] == "text" else []),
                    *(
                        field
                        for field in ("corner_radius", "stroke_width")
                        if field in before or field in after
                    ),
                )
            ):
                warn(
                    unsupported,
                    [eid],
                    "Container members must retain identity, grouping, paint, copy "
                    "and valid geometry.",
                )
    if result["warnings"]:
        result["accepted"] = False
        metrics["unsupported_deformation_count"] = len(result["warnings"])
        return result

    source_boxes = dict((*units, *leaves))
    with using_resources(assets, fonts):
        target_boxes = {
            ids: unit_visible_box([outputs[eid] for eid in ids]) for ids in source_boxes
        }
    accepted = []
    for container in profile.containers:
        membership = set(container.member_ids) - frame_ids
        children = [
            ids
            for ids, box in units
            if set(ids).issubset(membership) and box.width * box.height > 0
        ]
        record = {
            "frame_id": container.frame_id,
            "frame_ids": list(container.frame_ids),
            "member_ids": list(container.member_ids),
            "frames": [],
        }
        result["containers"].append(record)
        for frame_id in container.frame_ids:
            before, after = originals[frame_id], outputs[frame_id]
            source_frame, target_frame = bounds(before), bounds(after)
            owned = (
                children
                if frame_id == container.frame_id
                else [
                    ids
                    for ids, box in leaves
                    if ids[0] in membership
                    and box.width * box.height > 0
                    and _coverage(source_boxes[ids], source_frame) > 0.02
                    and min(originals[eid].get("z_index", 0) for eid in ids)
                    > before.get("z_index", 0)
                ]
            )
            if not container.adaptive:
                continue
            if not owned:
                # A decorative section without occupied child evidence gains
                # no exception. Its ordinary rigid measurement remains active.
                deformation = abs(
                    math.log(
                        (after["width"] / before["width"]) / (after["height"] / before["height"])
                    )
                )
                if deformation > math.log(1.05):
                    warn(
                        unsupported,
                        [frame_id],
                        "Frame deformation lacks corroborating original child support.",
                    )
                continue
            scales = [_element_scale(originals[eid], outputs[eid]) for ids in owned for eid in ids]
            child_scale = median(scales)
            containment_loss = max(
                max(
                    0.0,
                    _coverage(source_boxes[ids], source_frame)
                    - _coverage(target_boxes[ids], target_frame),
                )
                for ids in owned
            )
            if frame_id != container.frame_id:
                containment_loss = max(
                    containment_loss,
                    _coverage(source_frame, bounds(originals[container.frame_id]))
                    - _coverage(target_frame, bounds(outputs[container.frame_id])),
                )
            new_support = max(
                (
                    max(
                        0.0,
                        _coverage(target_boxes[ids], target_frame)
                        - _coverage(source_boxes[ids], source_frame),
                    )
                    for ids, box in leaves
                    if frame_id != container.frame_id
                    and ids[0] in membership
                    and box.width * box.height > 0
                    and _coverage(source_boxes[ids], source_frame) <= 0.02
                    and originals[ids[0]].get("z_index", 0) > before.get("z_index", 0)
                ),
                default=0.0,
            )
            before_padding = _padding(source_frame, _joined([source_boxes[ids] for ids in owned]))
            after_padding = _padding(target_frame, _joined([target_boxes[ids] for ids in owned]))
            padding_lost = any(
                original > 0 and changed + 0.5 < original * child_scale * 0.25
                for original, changed in zip(before_padding, after_padding, strict=True)
            )
            style_error = 0.0
            aspect_drift = (
                abs((after["width"] / after["height"]) / (before["width"] / before["height"]) - 1)
                if before["type"] == "image"
                else 0.0
            )
            for field in ("corner_radius", "stroke_width"):
                original, changed = before.get(field, 0), after.get(field, 0)
                if original > 0:
                    style_error = max(
                        style_error, abs(math.log(max(changed, 1e-12) / (original * child_scale)))
                    )
                elif changed != 0:
                    style_error = math.inf
            detail = {
                "frame_id": frame_id,
                "child_scale": child_scale,
                "containment_loss": containment_loss,
                "source_padding": list(before_padding),
                "candidate_padding": list(after_padding),
                "padding_retained": not padding_lost,
                "style_log_error": style_error if math.isfinite(style_error) else None,
                "image_aspect_drift": aspect_drift,
                "new_unowned_support": new_support,
            }
            record["frames"].append(detail)
            metrics["max_containment_loss"] = max(metrics["max_containment_loss"], containment_loss)
            metrics["max_frame_style_log_error"] = max(
                metrics["max_frame_style_log_error"],
                style_error if math.isfinite(style_error) else 100.0,
            )
            if containment_loss > 0.02 or padding_lost or new_support > 0.05:
                warn(
                    detached,
                    [frame_id, *(eid for ids in owned for eid in ids)],
                    "A source container changed its child support, enclosure or scaled padding.",
                )
            elif aspect_drift > ASPECT_TOLERANCE:
                warn(
                    unsupported,
                    [frame_id],
                    "A flat raster container still must retain "
                    "the immutable image aspect contract.",
                )
            elif style_error > math.log(1.05):
                warn(
                    unsupported,
                    [frame_id],
                    "Container stroke or corner geometry no longer follows its source child scale.",
                )
            else:
                accepted.append(frame_id)
    for index, first in enumerate(profile.containers):
        for second in profile.containers[index + 1 :]:
            before_a, before_b = (
                bounds(originals[first.frame_id]),
                bounds(originals[second.frame_id]),
            )
            after_a, after_b = bounds(outputs[first.frame_id]), bounds(outputs[second.frame_id])
            before = intersection(before_a, before_b) / max(
                1e-9, min(before_a.width * before_a.height, before_b.width * before_b.height)
            )
            after = intersection(after_a, after_b) / max(
                1e-9, min(after_a.width * after_a.height, after_b.width * after_b.height)
            )
            growth = max(0.0, after - before)
            metrics["max_new_container_overlap"] = max(metrics["max_new_container_overlap"], growth)
            if growth > 0.02:
                warn(
                    detached,
                    [first.frame_id, second.frame_id],
                    "Independent source containers acquired a new overlap.",
                )
    result["accepted"] = not result["warnings"]
    result["accepted_frame_ids"] = sorted(accepted) if result["accepted"] else []
    metrics["accepted_frame_count"] = len(result["accepted_frame_ids"])
    metrics["detached_count"] = sum(w["code"] == detached for w in result["warnings"])
    metrics["unsupported_deformation_count"] = sum(
        w["code"] == unsupported for w in result["warnings"]
    )
    return result
