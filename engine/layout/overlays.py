"""Conservative source-pixel evidence for translucent image support layers.

This infers physical attachment to an opaque underlay, not photo semantics.
Opaque illustrations remain ordinary foreground images. Ambiguous occlusion,
colored translucent artwork, rotations and missing resources decline inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageStat

from engine.resources import get_resources


@dataclass(frozen=True)
class SourceOverlay:
    overlay_id: str
    underlay_id: str
    confidence: float
    provenance: tuple[str, ...]
    opaque_visible_ratio: float
    underlay_opaque_ratio: float
    relative_frame: tuple[float, float, float, float]


@dataclass(frozen=True)
class _Pixels:
    width: int
    height: int
    visible: int
    opaque: int
    near_opaque: int
    channel_span: int
    alpha_span: int

    @property
    def opaque_visible_ratio(self):
        return self.opaque / max(self.visible, 1)


def _asset(element, directory):
    value = element.get("asset_id")
    if not isinstance(value, str) or not value:
        return None
    path = (directory / value).resolve()
    return path if path.is_relative_to(directory) and path.is_file() else None


@lru_cache(maxsize=128)
def _pixels(path, modified_ns, size_bytes):
    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            histogram = alpha.histogram()
            visible = sum(histogram[1:])
            if not visible:
                return None
            # Tiny alpha values amplify unpremultiplication quantization into
            # arbitrary RGB fringes. Judge paint color where alpha is stable.
            mask = alpha.point(lambda value: 255 if value >= 64 else 0)
            if mask.getbbox() is None:
                mask = alpha.point(lambda value: 255 if value else 0)
            extrema = ImageStat.Stat(rgba.convert("RGB"), mask).extrema
            occupied = [index for index, count in enumerate(histogram) if index and count]
            return _Pixels(
                rgba.width,
                rgba.height,
                visible,
                histogram[255],
                sum(histogram[250:]),
                max(high - low for low, high in extrema),
                max(occupied) - min(occupied),
            )
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def _evidence(path):
    if path is None:
        return None
    try:
        stat = path.stat()
        return _pixels(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _frame(element):
    values = tuple(element.get(key) for key in ("x", "y", "width", "height"))
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values
    ):
        return None
    if min(values[2:]) <= 0 or element.get("rotation", 0) != 0:
        return None
    return values


def _intersection(a, b):
    return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    )


def _inside(inner, outer):
    return (
        inner[0] >= outer[0] - 1e-6
        and inner[1] >= outer[1] - 1e-6
        and inner[0] + inner[2] <= outer[0] + outer[2] + 1e-6
        and inner[1] + inner[3] <= outer[1] + outer[3] + 1e-6
    )


def _fade(element, evidence):
    return (
        evidence is not None
        and element.get("opacity", 1) > 0
        and evidence.opaque_visible_ratio <= 0.05
        and evidence.channel_span <= 8
        and evidence.alpha_span >= 64
    )


def _opaque_support(element, path, frame, overlay):
    """Measure actual opaque underlay pixels beneath the whole overlay frame."""
    if element.get("opacity", 1) < 0.995:
        return 0.0
    try:
        with Image.open(path) as image:
            alpha = image.convert("RGBA").getchannel("A")
            sx, sy = frame[2] / image.width, frame[3] / image.height
            fit = element.get("fit", "contain")
            if fit != "fill":
                sx = sy = max(sx, sy) if fit == "cover" else min(sx, sy)
            left = frame[0] + (frame[2] - image.width * sx) / 2
            top = frame[1] + (frame[3] - image.height * sy) / 2
            box = (
                (overlay[0] - left) / sx,
                (overlay[1] - top) / sy,
                (overlay[0] + overlay[2] - left) / sx,
                (overlay[1] + overlay[3] - top) / sy,
            )
            if (
                box[0] < -1e-6
                or box[1] < -1e-6
                or box[2] > image.width + 1e-6
                or box[3] > image.height + 1e-6
            ):
                return 0.0
            crop = alpha.crop(
                (math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3]))
            )
            return sum(crop.histogram()[250:]) / max(crop.width * crop.height, 1)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return 0.0


def infer_source_overlays(
    source: dict, assets_dir: str | Path | None = None
) -> tuple[SourceOverlay, ...]:
    """Return immutable source attachments without altering source/group data."""
    directory = get_resources().assets_dir if assets_dir is None else Path(assets_dir).resolve()
    elements = sorted(source["elements"], key=lambda element: element.get("z_index", 0))
    images, texts = [], []
    for order, element in enumerate(elements):
        frame = _frame(element)
        if frame is None or element.get("opacity", 1) <= 0:
            continue
        if element.get("type") == "image":
            path = _asset(element, directory)
            images.append((order, element, frame, path, _evidence(path)))
        elif element.get("type") == "text" and str(element.get("content", "")).strip():
            texts.append((order, frame))
    result = []
    for order, element, frame, _path, evidence in images:
        if not _fade(element, evidence) or not any(
            following > order and _intersection(frame, text) >= text[2] * text[3] * 0.5
            for following, text in texts
        ):
            continue
        eligible = []
        for before, underlay, base, base_path, base_pixels in images:
            if (
                before >= order
                or base_pixels is None
                or not _inside(frame, base)
                or base[2] * base[3] <= frame[2] * frame[3] * 1.01
                or base_pixels.near_opaque / max(base_pixels.visible, 1) < 0.95
            ):
                continue
            opaque = _opaque_support(underlay, base_path, base, frame)
            if opaque >= 0.98:
                eligible.append((before, underlay, base, opaque))
        if not eligible:
            continue
        before, underlay, base, opaque = max(eligible, key=lambda item: item[0])
        # A partially covering image means the overlay belongs to a composition
        # of underlays, which this narrow one-owner inference cannot establish.
        if any(
            before < other_order < order
            and _intersection(frame, other_frame) > 1
            and not _fade(other, pixels)
            for other_order, other, other_frame, _, pixels in images
        ):
            continue
        result.append(
            SourceOverlay(
                element["id"],
                underlay["id"],
                0.95,
                (
                    "near-monochrome translucent source pixels",
                    "whole frame over opaque source image",
                    "source paint order places live text above overlay",
                    "physical attachment; no semantic photo inference",
                ),
                evidence.opaque_visible_ratio,
                opaque,
                (
                    (frame[0] - base[0]) / base[2],
                    (frame[1] - base[1]) / base[3],
                    frame[2] / base[2],
                    frame[3] / base[3],
                ),
            )
        )
    return tuple(result)
