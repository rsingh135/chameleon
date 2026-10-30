"""Background-cover candidates with source-owned header and caption regions."""

from __future__ import annotations

import json
import math
from copy import deepcopy

from engine.scene import (
    analyze_scene,
    approved_cover,
    approved_support,
    cover_position,
    support_retained,
)

from .geometry import Box, bounds, intersection, union
from .grouped import GroupedState
from .media_text import pack_information_region
from .model import build_context
from .source_guides import _spread, _unit_rows


def generate_background_states(source, target, scene=None):
    """Keep the asset, crop its viewport, and fit complete live-copy regions.

    The present policy is deliberately narrow: one source-proven full-bleed
    photo, one pixel-proven lower fade, an upper header and lower captions.
    Unknown imagery and product campaigns continue through their existing path.
    """
    scene = scene or analyze_scene(source)
    photos = [r for r in scene["images"] if r["role"] == "background" and r["allow_cover"]]
    source_aspect = source["canvas"]["width"] / source["canvas"]["height"]
    target_aspect = target["width"] / target["height"]
    if len(photos) != 1 or abs(source_aspect / target_aspect - 1) <= 1e-9:
        return []
    photo_row = photos[0]
    supports = [r for r in scene["images"] if r["element_id"] in photo_row["support_ids"]]
    if len(supports) != 1:
        return []
    fade_row = supports[0]
    by_id = {e["id"]: e for e in source["elements"]}
    context = build_context(source)
    excluded = {photo_row["element_id"], fade_row["element_id"]}
    caption_ids = set(fade_row["support_ids"])
    caption_units, header_units = [], []
    fade = by_id[fade_row["element_id"]]
    for unit in context.units:
        ids = {e["id"] for e in unit.elements}
        if ids <= excluded:
            continue
        if ids & excluded or any(e["type"] == "image" for e in unit.elements):
            return []
        if any(e["id"] in caption_ids for e in unit.text):
            if any(e["id"] not in caption_ids for e in unit.text):
                return []
            caption_units.append(unit)
        elif unit.text and unit.box.y + unit.box.height <= fade["y"]:
            header_units.append(unit)
        else:
            return []
    if not caption_units or not header_units:
        return []
    captions = _unit_rows(context, caption_units)
    headers = _unit_rows(context, header_units)
    caption_box = union([e for u in caption_units for e in u.elements])
    header_box = union([e for u in header_units for e in u.elements])
    width, height = float(target["width"]), float(target["height"])
    padding = min(width, height) * 0.035
    area_scale = math.sqrt(width * height / (context.width * context.height))
    header_scale = min(area_scale, (width - 2 * padding) / header_box.width)
    position = cover_position(photo_row, target)
    header_cell = Box(
        padding,
        (padding if position[1] < 0.5 else height * header_box.y / context.height),
        width - 2 * padding,
        header_box.height * header_scale,
    )
    packed_header = _spread(headers, header_scale, header_cell, context.width)
    if packed_header is None:
        return []
    from engine.eval.metrics import MIN_FONT_FRACTION, MIN_FONT_PX

    states = []
    for fraction in (0.28, 0.32, 0.36):
        scale = min(
            area_scale * 1.15,
            (width - 2 * padding) / caption_box.width,
            height * fraction / caption_box.height,
        )
        cell = Box(
            (width - caption_box.width * scale) / 2,
            height - padding - caption_box.height * scale,
            caption_box.width * scale,
            caption_box.height * scale,
        )
        packed = pack_information_region(
            source,
            [row.elements for row in captions],
            cell,
            preferred_scale=scale,
            min_font_size=max(MIN_FONT_PX, MIN_FONT_FRACTION * min(width, height)),
        )
        if packed is None:
            continue
        changed = {e["id"]: e for e in [*packed_header[0], *packed.elements]}
        photo = deepcopy(by_id[photo_row["element_id"]])
        photo.update(x=0.0, y=0.0, width=width, height=height, fit="cover")
        if position != [0.5, 0.5]:
            photo["image_position"] = position
        changed[photo["id"]] = photo
        # Resize the original native gradient rectangle's full paint extent;
        # never crop away its transparent edge or lose caption support.
        moved_fade = deepcopy(fade)
        moved_fade.update(x=0.0, width=width)
        design = {"canvas": dict(target), "elements": []}
        for step in range(41):
            fade_height = height * (0.20 + 0.80 * step / 40)
            moved_fade.update(y=height - fade_height, height=fade_height)
            changed[moved_fade["id"]] = deepcopy(moved_fade)
            design["elements"] = [deepcopy(changed.get(e["id"], e)) for e in source["elements"]]
            for e in design["elements"]:
                if e["id"] in {b["id"] for b in context.backgrounds}:
                    e.update(x=0.0, y=0.0, width=width, height=height)
            if support_retained(source, design, scene, fade["id"]):
                break
        else:
            continue
        if not approved_cover(scene, photo, target) or not approved_support(
            scene, moved_fade, target
        ):
            continue
        # The upper dark labels must not be covered by the lower fade.
        if any(intersection(bounds(e), bounds(moved_fade)) > 0.01 for e in packed_header[0]):
            continue
        states.append(
            GroupedState(
                f"background/cover-caption-{fraction:g}",
                "source-background-cover",
                (),
                json.dumps(design, sort_keys=True, separators=(",", ":")),
                (
                    "source-proven background: proportional cover crop",
                    "source-proven fade: caption alpha support retained",
                    *packed.exceptions,
                ),
            )
        )
    return states
