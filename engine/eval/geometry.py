"""Axis-aligned box helpers. All coordinates are canvas pixels."""

from __future__ import annotations

import math
from typing import NamedTuple


class Box(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def element_box(el: dict) -> Box:
    """Unrotated box, as the schema stores it and `engine.main.self_check` reads it."""
    return Box(el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])


def rotated_aabb(el: dict) -> Box:
    """Axis-aligned bounds of the element after rotation about its centre.
    Equal to `element_box` when rotation is 0."""
    rot = float(el.get("rotation") or 0.0)
    box = element_box(el)
    if rot == 0.0:
        return box
    theta = math.radians(rot)
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    half_w = (box.width * c + box.height * s) / 2
    half_h = (box.width * s + box.height * c) / 2
    return Box(box.cx - half_w, box.cy - half_h, box.cx + half_w, box.cy + half_h)


def canvas_box(canvas: dict) -> Box:
    return Box(0.0, 0.0, float(canvas["width"]), float(canvas["height"]))


def intersection(a: Box, b: Box) -> Box | None:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1, y1)


def overlap_area(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    return inter.area if inter else 0.0


def overlap_fraction(a: Box, b: Box) -> float:
    """Overlap as a fraction of the smaller box: 1.0 means one box is fully
    inside the other, 0.0 means they are disjoint."""
    smaller = min(a.area, b.area)
    if smaller <= 0:
        return 0.0
    return overlap_area(a, b) / smaller


def contains(outer: Box, inner: Box, slack: float = 0.5) -> bool:
    return (
        inner.x0 >= outer.x0 - slack
        and inner.y0 >= outer.y0 - slack
        and inner.x1 <= outer.x1 + slack
        and inner.y1 <= outer.y1 + slack
    )


def union(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return Box(
        min(b.x0 for b in boxes),
        min(b.y0 for b in boxes),
        max(b.x1 for b in boxes),
        max(b.y1 for b in boxes),
    )


def coverage(box: Box, canvas: Box) -> float:
    """Fraction of the canvas this box covers (clipped to the canvas)."""
    if canvas.area <= 0:
        return 0.0
    return overlap_area(box, canvas) / canvas.area
