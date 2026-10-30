"""Source-derived style relationships, separate from structural correctness.

These checks describe lost hierarchy, proximity, anchoring and prominence.
They are deterministic proxies for particular source relationships, not a
guarantee of subjective design quality. Canvas shapes and reading directions
may change; positions of ordinary content are deliberately not pinned.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from engine.layout import build_context, get_measurer
from engine.layout.geometry import Box, intersection, union
from engine.layout.relationships import SourceRelationshipGraph, build_relationship_graph
from engine.resources import DEFAULT_ASSETS_DIR, get_resources, using_resources

MEANINGFUL_FONT_RATIO = 1.15
MEANINGFUL_CONTROL_RATIO = 1.25
MAX_RATIO_DRIFT = 1.6
MIN_FONT_CONTRAST = 1.05
MIN_CONTROL_CONTRAST = 1.1
CORNER_BAND = 0.25
CORNER_DRIFT = 0.18
MIN_IMAGE_RETENTION = 0.65
COHESION_ALLOWANCE = 0.5
DEFAULT_ASSETS = DEFAULT_ASSETS_DIR
_METRICS = (
    "text_hierarchy",
    "control_hierarchy",
    "copy_cohesion",
    "corner_affinity",
    "image_prominence",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def _fidelity(source_ratio: float, candidate_ratio: float) -> float:
    if min(source_ratio, candidate_ratio) <= 0:
        return 0.0
    return math.exp(-abs(math.log(candidate_ratio / source_ratio)))


def _ids(unit) -> list[str]:
    return [element["id"] for element in unit.elements]


def _box(unit, by_id: dict) -> Box:
    return union([by_id[eid] for eid in _ids(unit)])


def _gap(first: Box, second: Box) -> float:
    dx = max(0.0, first.x - second.x - second.width, second.x - first.x - first.width)
    dy = max(0.0, first.y - second.y - second.height, second.y - first.y - first.height)
    return math.hypot(dx, dy)


def _line_size(unit, by_id: dict) -> float:
    return max(
        (
            float(by_id[element["id"]]["font_size"])
            * float(by_id[element["id"]].get("line_height", 1.2))
            for element in unit.text
        ),
        default=1.0,
    )


def _occupied_text_box(element: dict) -> Box:
    """Measure drawn lines inside a frame, rotating about the frame's center."""
    measured = get_measurer().measure(element, element["font_size"], element["width"])
    width = min(element["width"], measured.width)
    height = min(element["height"], measured.height)
    align = {"left": 0.0, "center": 0.5, "right": 1.0}.get(element.get("text_align"), 0.0)
    left = element["x"] + (element["width"] - width) * align
    top = element["y"]
    angle = math.radians(element.get("rotation", 0))
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = element["x"] + element["width"] / 2, element["y"] + element["height"] / 2
    corners = [
        (cx + (x - cx) * cosine - (y - cy) * sine, cy + (x - cx) * sine + (y - cy) * cosine)
        for x, y in (
            (left, top),
            (left + width, top),
            (left, top + height),
            (left + width, top + height),
        )
    ]
    x0, y0 = min(x for x, _ in corners), min(y for _, y in corners)
    x1, y1 = max(x for x, _ in corners), max(y for _, y in corners)
    return Box(x0, y0, x1 - x0, y1 - y0)


def _occupied_box(unit, by_id: dict) -> Box:
    boxes = [_occupied_text_box(by_id[element["id"]]) for element in unit.text]
    x0, y0 = min(box.x for box in boxes), min(box.y for box in boxes)
    x1 = max(box.x + box.width for box in boxes)
    y1 = max(box.y + box.height for box in boxes)
    return Box(x0, y0, x1 - x0, y1 - y0)


@lru_cache(maxsize=256)
def _read_alpha_extent(path: str, modified_ns: int, size_bytes: int):
    """Cache decoded dimensions/alpha bounds; stat keys invalidate changed assets."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            alpha = image.convert("RGBA").getchannel("A")
            return image.width, image.height, alpha.getbbox()
    except (ImportError, OSError, ValueError):
        return None


def _asset_extent(asset_id: str, assets_dir: Path):
    path = (assets_dir / asset_id).resolve()
    if not path.is_relative_to(assets_dir.resolve()):
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return _read_alpha_extent(str(path), stat.st_mtime_ns, stat.st_size)


def _visible_area(element: dict, extent) -> float:
    width, height = float(element["width"]), float(element["height"])
    if extent is None:
        return width * height
    image_width, image_height, alpha_box = extent
    if alpha_box is None:
        return 0.0
    visible_width = alpha_box[2] - alpha_box[0]
    visible_height = alpha_box[3] - alpha_box[1]
    if element.get("fit", "contain") == "fill":
        return width * height * visible_width * visible_height / (image_width * image_height)
    scale_fn = max if element.get("fit", "contain") == "cover" else min
    scale = scale_fn(width / image_width, height / image_height)
    from engine.image_viewport import image_position

    px, py = (
        image_position(element.get("image_position"))
        if element.get("fit") == "cover"
        else (0.5, 0.5)
    )
    offset_x, offset_y = (width - image_width * scale) * px, (height - image_height * scale) * py
    left, right = (
        max(0.0, offset_x + alpha_box[0] * scale),
        min(width, offset_x + alpha_box[2] * scale),
    )
    top, bottom = (
        max(0.0, offset_y + alpha_box[1] * scale),
        min(height, offset_y + alpha_box[3] * scale),
    )
    return max(0.0, right - left) * max(0.0, bottom - top)


def _analyze_style(
    source: dict,
    candidate: dict,
    assets_dir: str | Path | None = None,
    *,
    relationship_graph: SourceRelationshipGraph | None = None,
) -> dict:
    """Compare a candidate with style relationships inferred from its source.

    Returns ``{score, warnings, metrics}``. Each warning contains ``code``,
    ``message`` and the affected ``element_ids``. The five warning codes and
    metric names are text_hierarchy, control_hierarchy, copy_cohesion,
    corner_affinity and image_prominence. Structural validation remains the
    caller's responsibility; candidate IDs must match the source exactly.

    Missing or unreadable assets use element geometry instead of alpha bounds.
    No source or candidate fields are modified, and no assets are rewritten.
    Pass the immutable source ``relationship_graph`` to reuse its peer-row
    inference. Row members may reflow with their message rather than being
    mistaken for independent corner anchors.
    """
    original = {element["id"]: element for element in source["elements"]}
    resized = {element["id"]: element for element in candidate["elements"]}
    if original.keys() != resized.keys() or len(resized) != len(candidate["elements"]):
        raise ValueError("Style analysis requires every source element exactly once")
    if relationship_graph is not None and set(relationship_graph.source_ids) != original.keys():
        raise ValueError("Style relationship graph must describe the original source elements")
    context = build_context(source)
    values = {key: [] for key in _METRICS}
    warnings = []

    def warn(code: str, message: str, element_ids: list[str]) -> None:
        warnings.append({"code": code, "message": message, "element_ids": element_ids})

    # Compare meaningful font contrasts, without treating microscopic ties as hierarchy.
    texts = [element for element in source["elements"] if element["type"] == "text"]
    for first, second in combinations(texts, 2):
        larger, smaller = sorted(
            (first, second), key=lambda element: element["font_size"], reverse=True
        )
        source_ratio = larger["font_size"] / smaller["font_size"]
        if source_ratio < MEANINGFUL_FONT_RATIO:
            continue
        candidate_ratio = resized[larger["id"]]["font_size"] / resized[smaller["id"]]["font_size"]
        fidelity = _fidelity(source_ratio, candidate_ratio) if candidate_ratio > 1 else 0.0
        values["text_hierarchy"].append(fidelity)
        if candidate_ratio < MIN_FONT_CONTRAST or fidelity < 1 / MAX_RATIO_DRIFT:
            warn(
                "text_hierarchy",
                f"Text '{larger['id']}' was {source_ratio:.2f}x the size of '{smaller['id']}', "
                f"but is now {candidate_ratio:.2f}x; its original font hierarchy was lost.",
                [larger["id"], smaller["id"]],
            )

    controls = [
        unit
        for unit in context.units
        if unit.text and any(e["type"] == "shape" for e in unit.elements)
    ]
    for first, second in combinations(controls, 2):
        source_first, source_second = first.box, second.box
        cand_first, cand_second = _box(first, resized), _box(second, resized)
        lost = []
        for name, sa, sb, ca, cb in (
            ("width", source_first.width, source_second.width, cand_first.width, cand_second.width),
            (
                "area",
                source_first.width * source_first.height,
                source_second.width * source_second.height,
                cand_first.width * cand_first.height,
                cand_second.width * cand_second.height,
            ),
        ):
            if sa < sb:
                sa, sb, ca, cb = sb, sa, cb, ca
            source_ratio, candidate_ratio = sa / sb, ca / cb
            if source_ratio < MEANINGFUL_CONTROL_RATIO:
                continue
            fidelity = _fidelity(source_ratio, candidate_ratio)
            values["control_hierarchy"].append(fidelity if candidate_ratio >= 1 else 0.0)
            if candidate_ratio < MIN_CONTROL_CONTRAST or fidelity < 1 / MAX_RATIO_DRIFT:
                lost.append(f"{name} contrast {source_ratio:.2f}x became {candidate_ratio:.2f}x")
        if lost:
            warn(
                "control_hierarchy",
                "Grouped controls lost their original proportions: " + "; ".join(lost) + ".",
                _ids(first) + _ids(second),
            )

    # Adjacent source headline/body ink should remain neighbors after reflow.
    # Empty frame height must not hide a gap between the actual drawn lines.
    bodies = [unit for unit in context.units if unit.role == "body"]
    ink_units = [unit for unit in context.units if unit.role in {"headline", "body"}]
    source_ink = {tuple(_ids(unit)): _occupied_box(unit, original) for unit in ink_units}
    candidate_ink = {tuple(_ids(unit)): _occupied_box(unit, resized) for unit in ink_units}
    for headline in (unit for unit in context.units if unit.role == "headline"):
        heading_box = source_ink[tuple(_ids(headline))]
        neighbors = [
            body
            for body in bodies
            if source_ink[tuple(_ids(body))].y >= heading_box.y
            and intersection(
                Box(heading_box.x, 0, heading_box.width, 1),
                Box(source_ink[tuple(_ids(body))].x, 0, source_ink[tuple(_ids(body))].width, 1),
            )
            >= min(heading_box.width, source_ink[tuple(_ids(body))].width) * 0.6
        ]
        if not neighbors:
            continue
        body = min(neighbors, key=lambda unit: _gap(heading_box, source_ink[tuple(_ids(unit))]))
        source_line_size = _line_size(headline, original) + _line_size(body, original)
        source_gap = _gap(heading_box, source_ink[tuple(_ids(body))]) / source_line_size
        if source_gap > 1:
            continue
        candidate_line_size = _line_size(headline, resized) + _line_size(body, resized)
        candidate_gap = (
            _gap(candidate_ink[tuple(_ids(headline))], candidate_ink[tuple(_ids(body))])
            / candidate_line_size
        )
        growth = max(0.0, candidate_gap - source_gap)
        values["copy_cohesion"].append(math.exp(-growth))
        if growth > COHESION_ALLOWANCE:
            warn(
                "copy_cohesion",
                "Adjacent headline and body copy became separated: their gap grew from "
                f"{source_gap:.2f} to {candidate_gap:.2f} combined line heights.",
                _ids(headline) + _ids(body),
            )

    # Preserve actual corner anchors, not the source position of every unit.
    source_width, source_height = source["canvas"]["width"], source["canvas"]["height"]
    width, height = candidate["canvas"]["width"], candidate["canvas"]["height"]
    anchors = []
    for unit in context.units:
        if unit.role not in {"badge", "logo", "support"}:
            continue
        sx = (unit.box.x + unit.box.width / 2) / source_width
        sy = (unit.box.y + unit.box.height / 2) / source_height
        if CORNER_BAND < sx < 1 - CORNER_BAND or CORNER_BAND < sy < 1 - CORNER_BAND:
            continue
        anchors.append((unit, sx, sy))
    row_members = set()
    if anchors:
        graph = relationship_graph or build_relationship_graph(source, assets_dir)
        row_members = {
            eid
            for cluster in graph.clusters
            if cluster.kind == "badge_row"
            for eid in cluster.element_ids
        }
    for unit, sx, sy in anchors:
        if set(_ids(unit)) <= row_members:
            continue
        candidate_box = _box(unit, resized)
        cx = (candidate_box.x + candidate_box.width / 2) / width
        cy = (candidate_box.y + candidate_box.height / 2) / height
        source_dx, target_dx = (sx, cx) if sx <= CORNER_BAND else (1 - sx, 1 - cx)
        source_dy, target_dy = (sy, cy) if sy <= CORNER_BAND else (1 - sy, 1 - cy)
        drift = max(0.0, target_dx - source_dx, target_dy - source_dy)
        values["corner_affinity"].append(max(0.0, 1 - drift / 0.5))
        if drift > CORNER_DRIFT or target_dx > 0.5 or target_dy > 0.5:
            corner = ("top" if sy <= CORNER_BAND else "bottom") + " "
            corner += "left" if sx <= CORNER_BAND else "right"
            warn(
                "corner_affinity",
                f"A source {unit.role} anchored in the {corner} corner "
                "moved away from that corner.",
                _ids(unit),
            )

    # Compare visible image mass, allowing transparent padding and contain-fit letterboxing.
    assets = Path(assets_dir) if assets_dir is not None else get_resources().assets_dir
    source_canvas_area, target_canvas_area = source_width * source_height, width * height
    background_ids = {element["id"] for element in context.backgrounds}
    for element in source["elements"]:
        if element["type"] != "image" or element["id"] in background_ids:
            continue
        extent = _asset_extent(element.get("asset_id", ""), assets)
        source_area = _visible_area(element, extent)
        if source_area <= 0:
            continue
        target_area = _visible_area(resized[element["id"]], extent)
        retention = (target_area / target_canvas_area) / (source_area / source_canvas_area)
        values["image_prominence"].append(min(1.0, retention))
        if retention < MIN_IMAGE_RETENTION:
            warn(
                "image_prominence",
                f"Image '{element['id']}' retains only {retention:.0%} "
                "of its source canvas prominence"
                + (
                    " based on visible alpha bounds."
                    if extent is not None
                    else " based on geometry."
                ),
                [element["id"]],
            )

    metrics = {name: round(_mean(measured), 6) for name, measured in values.items()}
    applicable = [metrics[name] for name, measured in values.items() if measured]
    return {"score": round(_mean(applicable), 6), "warnings": warnings, "metrics": metrics}


def analyze_style(
    source: dict,
    candidate: dict,
    assets_dir: str | Path | None = None,
    *,
    relationship_graph: SourceRelationshipGraph | None = None,
) -> dict:
    """Analyze source style with current resources or the supplied source graph's fonts."""
    assets = assets_dir
    if assets is None and relationship_graph is not None:
        assets = relationship_graph.assets_dir
    fonts = relationship_graph.fonts_dir if relationship_graph is not None else None
    with using_resources(assets, fonts):
        return _analyze_style(source, candidate, assets, relationship_graph=relationship_graph)
