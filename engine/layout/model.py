"""Infer design units from geometry, grouping and typography, never sample IDs."""

from dataclasses import dataclass

from .geometry import Box, bounds, intersection, union


@dataclass
class Unit:
    elements: list[dict]
    box: Box
    role: str = "body"

    @property
    def text(self) -> list[dict]:
        return [e for e in self.elements if e["type"] == "text"]


@dataclass
class DesignContext:
    width: float
    height: float
    backgrounds: list[dict]
    decorations: list[dict]
    units: list[Unit]


def build_context(source: dict) -> DesignContext:
    width, height = source["canvas"]["width"], source["canvas"]["height"]
    backgrounds, decorations, foreground = [], [], []
    for element in source["elements"]:
        full = (
            element["width"] >= width * 0.9
            and element["height"] >= height * 0.9
            and element["x"] <= width * 0.05
            and element["y"] <= height * 0.05
        )
        if full and (
            element["type"] == "shape"
            or (element["type"] == "image" and element.get("fit") == "cover")
        ):
            backgrounds.append(element)
        elif (
            element["type"] == "shape"
            and not element.get("group_id")
            and element.get("opacity", 1) < 0.5
        ):
            decorations.append(element)
        else:
            foreground.append(element)

    grouped: dict[str, list[dict]] = {}
    for index, element in enumerate(foreground):
        key = f"group:{element['group_id']}" if element.get("group_id") else f"single:{index}"
        grouped.setdefault(key, []).append(element)

    # Infer unlabelled button/text containment when group_id is absent.
    singles = [k for k, v in grouped.items() if len(v) == 1 and not v[0].get("group_id")]
    for key in singles:
        if key not in grouped or grouped[key][0]["type"] != "text":
            continue
        text = grouped[key][0]
        tb = bounds(text)
        containers = []
        for other in singles:
            if other not in grouped or grouped[other][0]["type"] != "shape":
                continue
            shape = grouped[other][0]
            sb = bounds(shape)
            if (
                shape.get("z_index", 0) <= text.get("z_index", 0)
                and intersection(tb, sb) >= tb.width * tb.height * 0.95
                and sb.width * sb.height < width * height * 0.2
            ):
                containers.append((sb.width * sb.height, other))
        if containers:
            _, other = min(containers)
            grouped[other].extend(grouped.pop(key))

    units = [Unit(elements, union(elements)) for elements in grouped.values()]
    max_font = max((e.get("font_size", 0) for e in foreground), default=1)
    images = [u for u in units if not u.text and any(e["type"] == "image" for e in u.elements)]
    hero = max(images, key=lambda u: u.box.width * u.box.height, default=None)
    for unit in units:
        if unit.text:
            font = max(e["font_size"] for e in unit.text)
            content = " ".join(e["content"] for e in unit.text)
            if any(e["type"] == "shape" for e in unit.elements):
                unit.role = (
                    "action" if unit.box.y > height * 0.7 and font >= max_font * 0.4 else "badge"
                )
            elif font >= max_font * 0.8:
                unit.role = "headline"
            elif font <= max_font * 0.35 and len(content) > 50:
                unit.role = "legal"
            else:
                unit.role = "body"
        elif any(e["type"] == "image" for e in unit.elements):
            if unit.box.width / unit.box.height > 2 and unit.box.y < height * 0.3:
                unit.role = "logo"
            else:
                unit.role = "hero" if unit is hero else "support"
        else:
            unit.role = "badge"
    return DesignContext(width, height, backgrounds, decorations, units)
