"""Approximate occupied bounds for source composition, separate from frames.

Images use their alpha extent after fitting; text uses occupied line advances
and line height. These are bounding rectangles, not pixel masks or proof of
collision safety. Measurements retain overflow and rotate about the original
frame center. Asset pixels and element dictionaries are never modified.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from engine.resources import DEFAULT_ASSETS_DIR, get_resources

from .geometry import Box
from .text import get_measurer

DEFAULT_ASSETS = DEFAULT_ASSETS_DIR


@lru_cache(maxsize=256)
def _alpha_extent(path: str, modified_ns: int, size_bytes: int):
    """Stat-keyed image metadata avoids decoding an unchanged asset repeatedly."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height, image.convert("RGBA").getchannel("A").getbbox()
    except (ImportError, OSError, ValueError):
        return None


def _asset_extent(element: dict, assets_dir: str | Path | None):
    asset_id = element.get("asset_id")
    if not asset_id:
        return None
    root = (Path(assets_dir) if assets_dir is not None else get_resources().assets_dir).resolve()
    path = (root / asset_id).resolve()
    if not path.is_relative_to(root):
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return _alpha_extent(str(path), stat.st_mtime_ns, stat.st_size)


def _union(boxes: list[Box]) -> Box:
    occupied = [box for box in boxes if box.width > 0 and box.height > 0]
    if not occupied:
        return Box(0.0, 0.0, 0.0, 0.0)
    left, top = min(box.x for box in occupied), min(box.y for box in occupied)
    right = max(box.x + box.width for box in occupied)
    bottom = max(box.y + box.height for box in occupied)
    return Box(left, top, right - left, bottom - top)


def _rotate(box: Box, frame: Box, angle: float) -> Box:
    if not angle or box.width <= 0 or box.height <= 0:
        return box
    cx, cy = frame.x + frame.width / 2, frame.y + frame.height / 2
    sine, cosine = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    points = [
        (cx + (x - cx) * cosine - (y - cy) * sine, cy + (x - cx) * sine + (y - cy) * cosine)
        for x, y in (
            (box.x, box.y),
            (box.x + box.width, box.y),
            (box.x + box.width, box.y + box.height),
            (box.x, box.y + box.height),
        )
    ]
    left, top = min(x for x, _ in points), min(y for _, y in points)
    right, bottom = max(x for x, _ in points), max(y for _, y in points)
    return Box(left, top, right - left, bottom - top)


def _image_box(element: dict, frame: Box, assets_dir: str | Path | None) -> Box:
    extent = _asset_extent(element, assets_dir)
    if extent is None:
        return frame
    image_width, image_height, alpha = extent
    if alpha is None:
        return Box(frame.x, frame.y, 0.0, 0.0)
    fit = element.get("fit", "contain")
    sx, sy = frame.width / image_width, frame.height / image_height
    if fit in {"contain", "cover"}:
        sx = sy = (min if fit == "contain" else max)(sx, sy)
    from engine.image_viewport import image_position

    px, py = image_position(element.get("image_position")) if fit == "cover" else (0.5, 0.5)
    dx, dy = (frame.width - image_width * sx) * px, (frame.height - image_height * sy) * py
    left, top = max(0.0, dx + alpha[0] * sx), max(0.0, dy + alpha[1] * sy)
    right = min(frame.width, dx + alpha[2] * sx)
    bottom = min(frame.height, dy + alpha[3] * sy)
    return Box(frame.x + left, frame.y + top, max(0.0, right - left), max(0.0, bottom - top))


def _text_box(element: dict, frame: Box) -> Box:
    measurer = get_measurer()
    size = float(element.get("font_size", 24))
    measured = measurer.measure(element, size, frame.width)
    line_height = size * float(element.get("line_height", 1.2))
    align = {"left": 0.0, "center": 0.5, "right": 1.0}.get(element.get("text_align"), 0.0)
    lines = []
    for index, content in enumerate(measured.wrapped.split("\n")):
        if not content.strip():
            continue
        # A measured wrapped line must not wrap a second time. Width is an
        # advance extent and can exceed the frame for an unbreakable word.
        width = measurer.measure({**element, "content": content}, size, math.inf).width
        lines.append(
            Box(
                frame.x + (frame.width - width) * align,
                frame.y + index * line_height,
                width,
                line_height,
            )
        )
    return _union(lines)


def visible_box(element: dict, assets_dir: str | Path | None = None) -> Box:
    """Return approximate occupied bounds, retaining overflow beyond the frame.

    Missing/unreadable images fall back to frame geometry, which also matches
    the renderer's full-frame missing-asset placeholder. Fully transparent
    assets, zero opacity, and empty text have zero occupied area. Image fitting
    is continuous and can differ from raster rounding and resampling halos.
    """
    frame = Box(*(float(element[key]) for key in ("x", "y", "width", "height")))
    if element.get("opacity", 1) <= 0 or frame.width <= 0 or frame.height <= 0:
        return Box(frame.x, frame.y, 0.0, 0.0)
    if element["type"] == "image":
        occupied = _image_box(element, frame, assets_dir)
    elif element["type"] == "text":
        occupied = _text_box(element, frame)
    else:
        occupied = frame
    return _rotate(occupied, frame, float(element.get("rotation", 0) or 0))


def unit_visible_box(elements: list[dict], assets_dir: str | Path | None = None) -> Box:
    """Union visible element bounds, ignoring empty members; empty units are zero."""
    return _union([visible_box(element, assets_dir) for element in elements])
