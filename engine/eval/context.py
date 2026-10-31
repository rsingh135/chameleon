"""One-time analysis of the SOURCE design, shared by contract checks and
metrics. Everything the engine is told about intent lives here: which
overlaps the designer made on purpose, which elements sit inside others,
which elements travel together, how the text actually wraps, and what the
asset pixels look like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from engine.eval.geometry import (
    Box,
    canvas_box,
    contains,
    coverage,
    element_box,
    overlap_fraction,
    rotated_aabb,
)
from engine.eval.text import DEFAULT_FONTS_DIR as DEFAULT_FONTS_DIR
from engine.eval.text import TextLayout, get_measurer
from engine.resources import DEFAULT_ASSETS_DIR as DEFAULT_ASSETS_DIR
from engine.resources import using_resources

# An element covering at least this much of the canvas is treated as a
# background plate: it is expected to sit under everything else.
BACKGROUND_COVERAGE = 0.90
# Overlap (as a fraction of the smaller box) below this is treated as touching,
# not overlapping, when recording intentional overlaps.
INTENTIONAL_OVERLAP_MIN = 0.02
EDGE_SLACK = 0.5


@dataclass(frozen=True)
class AssetStats:
    width: int
    height: int
    alpha_coverage: float  # mean alpha, 0..1 (1 = fully opaque image)
    busyness: float  # mean edge magnitude of the luminance, 0..1
    # Fraction of the opaque mass inside the central band when the asset is
    # centre-cropped to a given aspect: computed lazily by metrics.


@dataclass
class Context:
    source: dict
    canvas: Box
    by_id: dict[str, dict]
    in_bounds_ids: frozenset[str]
    background_ids: frozenset[str]
    foreground_ids: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]
    intentional_overlaps: frozenset[frozenset[str]]
    containment: frozenset[tuple[str, str]]  # (inner, outer)
    text_layouts: dict[str, TextLayout]
    assets: dict[str, AssetStats | None]
    assets_dir: str
    fonts_dir: str
    notes: list[str] = field(default_factory=list)

    def group_of(self, el_id: str) -> str | None:
        return self.by_id[el_id].get("group_id") or None

    def same_group(self, a: str, b: str) -> bool:
        ga, gb = self.group_of(a), self.group_of(b)
        return ga is not None and ga == gb

    def is_background(self, el_id: str) -> bool:
        return el_id in self.background_ids


def _in_bounds(el: dict, canvas: Box) -> bool:
    return contains(canvas, element_box(el), slack=EDGE_SLACK)


def _asset_stats(assets_dir: Path, asset_id: str) -> AssetStats | None:
    path = assets_dir / asset_id
    if not path.is_file():
        return None
    from PIL import Image, ImageFilter, ImageStat

    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        alpha = ImageStat.Stat(rgba.getchannel("A")).mean[0] / 255.0
        lum = rgba.convert("L")
        # Downsample before edge detection so busyness is about visible
        # structure, not pixel noise, and is comparable across asset sizes.
        lum.thumbnail((256, 256))
        edges = lum.filter(ImageFilter.FIND_EDGES)
        busyness = ImageStat.Stat(edges).mean[0] / 255.0
        return AssetStats(rgba.width, rgba.height, alpha, busyness)


def _build_context(
    source: dict,
    assets_dir: Path,
    fonts_dir: Path,
) -> Context:
    canvas = canvas_box(source["canvas"])
    elements = source["elements"]
    by_id = {el["id"]: el for el in elements}
    notes: list[str] = []

    in_bounds = frozenset(el["id"] for el in elements if _in_bounds(el, canvas))

    background = frozenset(
        el["id"]
        for el in elements
        if el["type"] != "text" and coverage(rotated_aabb(el), canvas) >= BACKGROUND_COVERAGE
    )
    foreground = tuple(el["id"] for el in elements if el["id"] not in background)

    groups: dict[str, list[str]] = {}
    for el in elements:
        gid = el.get("group_id")
        if gid:
            groups.setdefault(gid, []).append(el["id"])

    overlaps: set[frozenset[str]] = set()
    containment: set[tuple[str, str]] = set()
    boxes = {el["id"]: rotated_aabb(el) for el in elements}
    for a, b in combinations(foreground, 2):
        ba, bb = boxes[a], boxes[b]
        if overlap_fraction(ba, bb) >= INTENTIONAL_OVERLAP_MIN:
            overlaps.add(frozenset((a, b)))
        if contains(bb, ba):
            containment.add((a, b))
        elif contains(ba, bb):
            containment.add((b, a))

    measurer = get_measurer(fonts_dir)
    text_layouts: dict[str, TextLayout] = {}
    for el in elements:
        if el["type"] == "text":
            lay = measurer.layout(el)
            text_layouts[el["id"]] = lay
            if lay.widest_unbreakable > el["width"] + EDGE_SLACK:
                notes.append(f"{el['id']}: a word is wider than its box in the SOURCE")
            if lay.height > el["height"] + EDGE_SLACK:
                notes.append(f"{el['id']}: text is taller than its box in the SOURCE")

    assets: dict[str, AssetStats | None] = {}
    assets_path = Path(assets_dir)
    for el in elements:
        if el["type"] == "image":
            aid = el.get("asset_id", "")
            if aid not in assets:
                assets[aid] = _asset_stats(assets_path, aid)
                if assets[aid] is None:
                    notes.append(f"{el['id']}: asset '{aid}' not found (renders as placeholder)")

    return Context(
        source=source,
        canvas=canvas,
        by_id=by_id,
        in_bounds_ids=in_bounds,
        background_ids=background,
        foreground_ids=foreground,
        groups={k: tuple(v) for k, v in groups.items()},
        intentional_overlaps=frozenset(overlaps),
        containment=frozenset(containment),
        text_layouts=text_layouts,
        assets=assets,
        assets_dir=str(assets_dir),
        fonts_dir=str(fonts_dir),
        notes=notes,
    )


def build_context(
    source: dict,
    assets_dir: str | Path | None = None,
    fonts_dir: str | Path | None = None,
) -> Context:
    """Capture absolute document resources along with the original design analysis."""
    with using_resources(assets_dir, fonts_dir) as resources:
        return _build_context(source, resources.assets_dir, resources.fonts_dir)
