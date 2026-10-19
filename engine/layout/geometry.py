"""Geometry shared by reflow and the final canvas containment pass."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float


def bounds(element: dict) -> Box:
    """Return the axis-aligned bounds of the rotated element."""
    width, height = float(element["width"]), float(element["height"])
    angle = math.radians(float(element.get("rotation", 0)))
    rw = abs(width * math.cos(angle)) + abs(height * math.sin(angle))
    rh = abs(width * math.sin(angle)) + abs(height * math.cos(angle))
    return Box(element["x"] + (width - rw) / 2, element["y"] + (height - rh) / 2, rw, rh)


def union(elements: list[dict]) -> Box:
    boxes = [bounds(element) for element in elements]
    left, top = min(b.x for b in boxes), min(b.y for b in boxes)
    right = max(b.x + b.width for b in boxes)
    bottom = max(b.y + b.height for b in boxes)
    return Box(left, top, max(1.0, right - left), max(1.0, bottom - top))


def intersection(a: Box, b: Box) -> float:
    return max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)) * max(
        0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    )


def contain(element: dict, width: float, height: float) -> None:
    """Keep rotated bounds on canvas, preserving proportions and rotation."""
    box = bounds(element)
    scale = min(1.0, width / max(box.width, 1e-9), height / max(box.height, 1e-9))
    if scale < 1:
        element["width"] *= scale
        element["height"] *= scale
        if element["type"] == "text":
            element["font_size"] = max(1.0, element["font_size"] * scale)
    box = bounds(element)
    element["x"] += max(0.0, -box.x) - max(0.0, box.x + box.width - width)
    element["y"] += max(0.0, -box.y) - max(0.0, box.y + box.height - height)
