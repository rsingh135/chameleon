"""Conservative source-supported alpha occlusion, separate from raw collisions.

Only a plate proven to support source foreground ink may explain a new image
intersection. Unknown pairs and every text/text collision remain counted. The
native renderer supplies bounded transparent masks, without changing assets,
font managers, renderer functions or the official raw-collision metric.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from engine.resources import get_renderer, native_render_example, render_input, using_resources
from renderer.converter.convert import convert_chameleon_to_crello

from .context import build_context
from .geometry import overlap_fraction, rotated_aabb
from .metrics import _decorative_underlays, _line_overlap, _occupied_boxes, _text_line_boxes
from .text import get_measurer

MAX_MASK_SIZE = 512
DEFAULT_MASK_SIZE = 384
MAX_MASKS = 128
MAX_PAIRS = 512
MAX_ELEMENTS = 256
MAX_FITTED_PIXELS = 8_000_000
MIN_VISIBILITY_RETENTION = 0.85
_SUPPORT = 0.985


@dataclass(frozen=True)
class _Mask:
    width: int
    height: int
    alpha: bytes
    error: str = ""

    def image(self):
        return Image.frombytes("L", (self.width, self.height), self.alpha)


@dataclass(frozen=True)
class PlateSupport:
    plate_id: str
    supported_ids: tuple[str, ...]
    lower_image_ids: tuple[str, ...]


@dataclass(frozen=True)
class OcclusionProfile:
    source_digest: str
    assets_dir: str
    fonts_dir: str
    resource_digest: str
    max_size: int
    plates: tuple[PlateSupport, ...]
    photo_ids: tuple[str, ...]
    source_visibility: tuple[tuple[str, float], ...]
    source_masks: tuple[tuple[str, _Mask], ...]
    issues: tuple[tuple[str, str], ...]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _safe_asset(root, asset_id):
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError("Occlusion assets require a nonempty local asset_id.")
    path = (Path(root) / asset_id).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Occlusion asset paths must remain inside the document assets directory.")
    return path


def _resource_digest(source, assets, fonts):
    files = {
        _safe_asset(assets, e.get("asset_id")) for e in source["elements"] if e["type"] == "image"
    }
    font_root = Path(fonts).resolve()
    for path in font_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".ttf", ".otf"}:
            if not path.resolve().is_relative_to(font_root):
                raise ValueError("Occlusion fonts must remain inside the document fonts directory.")
            files.add(path.resolve())
    fingerprint = hashlib.sha256()
    for path in sorted(files):
        fingerprint.update(str(path).encode())
        if not path.is_file():
            fingerprint.update(b"missing")
            continue
        stat = path.stat()
        fingerprint.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        # Imports already bound asset sizes; this diagnostic still avoids
        # allocating arbitrary files or trusting mtime alone for normal inputs.
        if stat.st_size > 32_000_000:
            fingerprint.update(b"oversized")
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1_048_576), b""):
                fingerprint.update(chunk)
    return fingerprint.hexdigest()


def _mass(mask):
    return ImageStat.Stat(mask).sum[0] / 255.0


def _ratio(first, second, denominator=None):
    denominator = min(_mass(first), _mass(second)) if denominator is None else denominator
    return _mass(ImageChops.multiply(first, second)) / max(1e-9, denominator)


def _opaque(mask):
    return mask.point(lambda alpha: 255 if alpha >= 254 else 0)


@lru_cache(maxsize=128)
def _render_mask(element_json, canvas_json, assets, fonts, resource_digest, max_size):
    del resource_digest  # Included in the immutable cache key.
    element, canvas = json.loads(element_json), json.loads(canvas_json)
    if element["width"] * element["height"] > MAX_FITTED_PIXELS:
        return _Mask(0, 0, b"", "element exceeds bounded fitted-image budget")
    if element["type"] == "image":
        path = _safe_asset(assets, element.get("asset_id"))
        if not path.is_file() or path.stat().st_size > 32_000_000:
            return _Mask(0, 0, b"", "missing or oversized local asset")
    try:
        import skia

        document = {"canvas": canvas, "elements": [element]}
        with using_resources(assets, fonts):
            renderer = get_renderer()
            example, warnings = convert_chameleon_to_crello(
                render_input(document), renderer.font_manager, assets
            )
            if warnings:
                return _Mask(0, 0, b"", "; ".join(warnings))
            example = native_render_example(example, document)
            png = renderer.render(
                example, max_size=max_size, format="png", canvas_color=skia.ColorTRANSPARENT
            )
        with Image.open(io.BytesIO(png)) as image:
            alpha = image.convert("RGBA").getchannel("A")
            return _Mask(alpha.width, alpha.height, alpha.tobytes())
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        return _Mask(0, 0, b"", f"mask unavailable: {type(error).__name__}: {error}")


class _Masks:
    def __init__(self, canvas, assets, fonts, resource_digest, max_size, initial=()):
        self.canvas, self.assets, self.fonts = _canonical(canvas), assets, fonts
        self.resource_digest, self.max_size = resource_digest, max_size
        self.masks, self.errors = dict(initial), {}

    def get(self, element):
        eid = element["id"]
        if eid not in self.masks:
            if len(self.masks) >= MAX_MASKS:
                self.errors[eid] = "mask budget exceeded"
                return None
            self.masks[eid] = _render_mask(
                _canonical(element),
                self.canvas,
                self.assets,
                self.fonts,
                self.resource_digest,
                self.max_size,
            )
        mask = self.masks[eid]
        if mask.error:
            self.errors[eid] = mask.error
            return None
        return mask.image()


def _ordered(design):
    return sorted(design["elements"], key=lambda e: e.get("z_index", 0))


def _visible(element, ordered, masks):
    base = masks.get(element)
    if base is None:
        return None, None
    visible = base.copy()
    later = False
    for other in ordered:
        if other["id"] == element["id"]:
            later = True
            continue
        if (
            not later
            or other["type"] == "text"
            or overlap_fraction(rotated_aabb(element), rotated_aabb(other)) <= 0
        ):
            continue
        alpha = masks.get(other)
        if alpha is None:
            return None, None
        visible = ImageChops.multiply(visible, ImageChops.invert(alpha))
    return visible, _mass(visible) / max(_mass(base), 1e-9)


def _known_plate(element):
    return element["type"] == "shape" or (
        element["type"] == "image" and element.get("source_art", {}).get("kind") == "svg-nontext"
    )


@lru_cache(maxsize=4)
def _infer(source_json, assets, fonts, resource_digest, max_size):
    source = json.loads(source_json)
    ordered = _ordered(source)
    masks = _Masks(source["canvas"], assets, fonts, resource_digest, max_size)
    plates = []
    for index, plate in enumerate(ordered):
        if not _known_plate(plate):
            continue
        lower = [
            e
            for e in ordered[:index]
            if e["type"] == "image"
            and overlap_fraction(rotated_aabb(plate), rotated_aabb(e)) > 0.005
        ]
        possible = [
            e
            for e in ordered[index + 1 :]
            if overlap_fraction(rotated_aabb(plate), rotated_aabb(e)) >= 0.9
        ]
        if not lower or not any(e["type"] == "text" for e in possible):
            continue
        alpha = masks.get(plate)
        if alpha is None:
            continue
        opaque, supported = _opaque(alpha), []
        for child in possible:
            ink = masks.get(child)
            if ink is not None and _mass(ink) >= 1 and _ratio(ink, opaque, _mass(ink)) >= _SUPPORT:
                supported.append(child["id"])
        if not any(e["type"] == "text" and e["id"] in supported for e in possible):
            continue
        occluded = []
        for image in lower:
            ink = masks.get(image)
            if ink is not None and _ratio(ink, opaque) > 0.005:
                occluded.append(image["id"])
        if occluded:
            plates.append(PlateSupport(plate["id"], tuple(supported), tuple(occluded)))
    photos = tuple(
        e["id"]
        for e in ordered
        if e["type"] == "image" and e.get("source_art", {}).get("kind") != "svg-nontext"
    )
    by_id = {e["id"]: e for e in ordered}
    visibility = []
    for eid in photos:
        _, fraction = _visible(by_id[eid], ordered, masks)
        if fraction is not None:
            visibility.append((eid, fraction))
    return OcclusionProfile(
        _digest(source_json),
        assets,
        fonts,
        resource_digest,
        max_size,
        tuple(plates),
        photos,
        tuple(visibility),
        tuple(masks.masks.items()),
        tuple(sorted(masks.errors.items())),
    )


def infer_occlusion_profile(source, assets_dir=None, fonts_dir=None, *, max_size=DEFAULT_MASK_SIZE):
    """Capture immutable source plate evidence and source photo visibility once."""
    if len(source["elements"]) > MAX_ELEMENTS:
        raise ValueError("Occlusion diagnostic element budget exceeded.")
    size = max(128, min(MAX_MASK_SIZE, int(max_size)))
    source_json = _canonical(source)
    with using_resources(assets_dir, fonts_dir) as resources:
        assets, fonts = str(resources.assets_dir), str(resources.fonts_dir)
        resource_digest = _resource_digest(source, assets, fonts)
    return _infer(source_json, assets, fonts, resource_digest, size)


def _raw_pairs(source, candidate, context):
    """Mirror raw metric pairs without changing that metric or its diagnostics."""
    originals, changed = context.by_id, {e["id"]: e for e in candidate["elements"]}
    measurer = get_measurer(context.fonts_dir)
    texts = [eid for eid in context.foreground_ids if originals[eid]["type"] == "text"]
    source_lines = {
        eid: _text_line_boxes(originals[eid], context.text_layouts[eid]) for eid in texts
    }
    lines = {eid: _text_line_boxes(changed[eid], measurer.layout(changed[eid])) for eid in texts}
    source_boxes = {
        eid: _occupied_boxes(originals[eid], source_lines, context.assets_dir)
        for eid in context.foreground_ids
    }
    boxes = {
        eid: _occupied_boxes(changed[eid], lines, context.assets_dir)
        for eid in context.foreground_ids
    }
    underlays = _decorative_underlays(source, context)
    pairs = []
    for a, b in combinations(context.foreground_ids, 2):
        if any(
            decor in underlays
            and other not in underlays
            and originals[decor].get("z_index", 0) < originals[other].get("z_index", 0)
            and changed[decor].get("z_index", 0) < changed[other].get("z_index", 0)
            for decor, other in ((a, b), (b, a))
        ):
            continue
        kinds = (changed[a]["type"], changed[b]["type"])
        plain_text = kinds == ("text", "text") and not any(
            e.get("rotation", 0) for e in (originals[a], originals[b], changed[a], changed[b])
        )
        if "image" in kinds:
            baseline = _line_overlap(source_boxes[a], source_boxes[b])
            fraction = _line_overlap(boxes[a], boxes[b]) - baseline
            if fraction <= 0.02:
                continue
        elif plain_text:
            baseline = _line_overlap(source_lines[a], source_lines[b])
            fraction = _line_overlap(lines[a], lines[b]) - baseline
            if fraction <= 0.02:
                continue
        else:
            if context.same_group(a, b) or frozenset((a, b)) in context.intentional_overlaps:
                continue
            baseline = 0.0
            fraction = overlap_fraction(rotated_aabb(changed[a]), rotated_aabb(changed[b]))
            if fraction <= 0:
                continue
        pairs.append((a, b, baseline, fraction))
    return pairs


def analyze_occlusion(source, candidate, profile=None, *, assets_dir=None, fonts_dir=None):
    """Explain source-supported image intersections; retain all uncertain pairs."""
    profile = profile or infer_occlusion_profile(source, assets_dir, fonts_dir)
    if _digest(_canonical(source)) != profile.source_digest:
        raise ValueError("Occlusion profile does not match the immutable source.")
    if _resource_digest(source, profile.assets_dir, profile.fonts_dir) != profile.resource_digest:
        raise ValueError("Occlusion resources changed after the source profile was captured.")
    if {e["id"] for e in source["elements"]} != {e["id"] for e in candidate["elements"]}:
        raise ValueError("Occlusion candidate element identity differs from source.")
    for element in candidate["elements"]:
        if element["type"] == "image":
            _safe_asset(profile.assets_dir, element.get("asset_id"))
    context = build_context(source, profile.assets_dir, profile.fonts_dir)
    raw = _raw_pairs(source, candidate, context)
    ordered = _ordered(candidate)
    by_id = {e["id"]: e for e in ordered}
    rank = {e["id"]: index for index, e in enumerate(ordered)}
    masks = _Masks(
        candidate["canvas"],
        profile.assets_dir,
        profile.fonts_dir,
        profile.resource_digest,
        profile.max_size,
    )
    warnings = [
        {"code": "source_mask_unavailable", "element_ids": [eid], "message": reason}
        for eid, reason in profile.issues
    ]
    photos = []
    retention = 1.0
    visible = {}
    source_visibility = dict(profile.source_visibility)
    for eid in profile.photo_ids:
        image, fraction = _visible(by_id[eid], ordered, masks)
        visible[eid] = image
        before = source_visibility.get(eid)
        ratio = fraction / before if fraction is not None and before and before > 1e-6 else None
        retention = 0.0 if ratio is None else min(retention, ratio)
        photos.append(
            {
                "element_id": eid,
                "source_fraction": before,
                "candidate_fraction": fraction,
                "retention": ratio,
            }
        )
        if ratio is None or ratio < MIN_VISIBILITY_RETENTION:
            warnings.append(
                {
                    "code": "photo_visibility_lost",
                    "element_ids": [eid],
                    "message": "Photo visibility must retain at least 85% of its source fraction.",
                }
            )
    discounts = []
    bounded = []
    guard = retention >= MIN_VISIBILITY_RETENTION and not profile.issues
    for a, b, baseline, fraction in raw[:MAX_PAIRS]:
        detail = {
            "element_ids": [a, b],
            "source_overlap": baseline,
            "raw_excess_overlap": fraction,
            "status": "retained_unknown_pair",
        }
        bounded.append(detail)
        if "image" not in (by_id[a]["type"], by_id[b]["type"]):
            detail["status"] = "retained_nonimage_pair"
            continue
        if not guard:
            detail["status"] = "retained_visibility_guard"
            continue
        for image_id, foreground_id in ((a, b), (b, a)):
            if by_id[image_id]["type"] != "image":
                continue
            for support in profile.plates:
                plate_id = support.plate_id
                if (
                    image_id not in support.lower_image_ids
                    or (foreground_id != plate_id and foreground_id not in support.supported_ids)
                    or not rank[image_id] < rank[plate_id] <= rank[foreground_id]
                ):
                    continue
                plate = masks.get(by_id[plate_id])
                ink = masks.get(by_id[foreground_id])
                if plate is None or ink is None:
                    continue
                coverage = _ratio(ink, _opaque(plate), _mass(ink))
                if coverage < _SUPPORT:
                    detail["status"] = "retained_foreground_outside_source_plate"
                    continue
                if image_id not in visible:
                    visible[image_id], _ = _visible(by_id[image_id], ordered, masks)
                image = visible[image_id]
                if image is None:
                    continue
                residual = _ratio(image, ink)
                detail.update(
                    plate_id=plate_id,
                    opaque_support_coverage=coverage,
                    visible_alpha_overlap=residual,
                )
                if residual > 0.02:
                    detail["status"] = "retained_visible_overlap"
                    continue
                detail["status"] = "discounted_source_plate"
                discounts.append(dict(detail))
                break
            if detail["status"] == "discounted_source_plate":
                break
    warnings.extend(
        {"code": "candidate_mask_unavailable", "element_ids": [eid], "message": reason}
        for eid, reason in sorted(masks.errors.items())
    )
    complete = not profile.issues and not masks.errors and len(raw) <= MAX_PAIRS
    if len(raw) > MAX_PAIRS:
        warnings.append(
            {
                "code": "pair_budget_exceeded",
                "element_ids": [],
                "message": "Unexamined collision pairs remain counted.",
            }
        )
    return {
        "raw_pair_count": len(raw),
        "visible_pair_count": len(raw) - len(discounts),
        "source_supported_discounts": discounts,
        "bounded_pairs": bounded,
        "visibility_retention": retention,
        "photo_visibility": photos,
        "warnings": warnings,
        "coverage": {
            "complete": complete,
            "max_mask_size": profile.max_size,
            "examined_pairs": len(bounded),
            "raw_pairs": len(raw),
            "candidate_masks": len(masks.masks),
            "source_masks": len(profile.source_masks),
        },
    }


def conservative_occlusion(source, candidate, profile=None):
    """Keep diagnostics optional: unavailable evidence never waives a collision."""
    try:
        return analyze_occlusion(source, candidate, profile)
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            "raw_pair_count": None,
            "visible_pair_count": None,
            "source_supported_discounts": [],
            "bounded_pairs": [],
            "visibility_retention": None,
            "photo_visibility": [],
            "warnings": [{"code": "occlusion_unavailable", "message": str(exc), "element_ids": []}],
            "coverage": {"complete": False},
        }


def collision_safety(report, raw_collisions):
    """Use visible pairs only with complete evidence matching the official count.

    The raw metric remains public and unchanged. Visibility is an independent
    guard, including when no raw pair would otherwise expose a hidden photo.
    """
    complete = report["coverage"]["complete"]
    verified = complete and report["raw_pair_count"] == raw_collisions
    count = report["visible_pair_count"] if verified else raw_collisions
    visible = not complete or report["visibility_retention"] >= MIN_VISIBILITY_RETENTION
    return count, visible
