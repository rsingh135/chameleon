"""Source-bound physical scene roles and proportional background viewports.

This is an offline evidence layer, not a claim to recognize people or products.
Only a full-bleed opaque photo beneath an independently verified caption fade
earns a cover policy. A model may suggest richer roles in the local harness;
it cannot manufacture this permission or change the asset bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from engine.layout.overlays import infer_source_overlays
from engine.resources import get_resources, using_resources


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _frame(element):
    values = tuple(element.get(k) for k in ("x", "y", "width", "height"))
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        return None
    return values if min(values[2:]) > 0 else None


def _inside(inner, outer):
    if inner is None or outer is None:
        return False
    return (
        inner[0] >= outer[0] - 1e-6
        and inner[1] >= outer[1] - 1e-6
        and inner[0] + inner[2] <= outer[0] + outer[2] + 1e-6
        and inner[1] + inner[3] <= outer[1] + outer[3] + 1e-6
    )


@lru_cache(maxsize=128)
def _pixels(path, modified_ns, byte_size):
    del modified_ns
    if not 0 < byte_size <= 32_000_000:
        return None
    try:
        with Image.open(path) as image:
            if image.width * image.height > 32_000_000:
                return None
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            column = rgba.crop((rgba.width // 2, 0, rgba.width // 2 + 1, rgba.height))
            row_field = ImageChops.difference(
                rgba, column.resize(rgba.size, Image.Resampling.NEAREST)
            )
            invariant = max(high for low, high in row_field.getextrema()) <= 2
            values = list(column.getchannel("A").tobytes())
            lower_fade = (
                invariant
                and values[-1] - values[0] >= 64
                and all(b >= a - 2 for a, b in zip(values, values[1:], strict=False))
            )
            # A bright, quiet authored upper field supplies physical headroom
            # evidence. This is a foreground-onset proxy, not face detection.
            rgb = rgba.convert("RGB")
            sample = rgb.resize((96, max(8, min(512, round(rgb.height * 96 / rgb.width)))))
            band = sample.crop((19, 0, 77, sample.height))
            upper = ImageStat.Stat(band.crop((0, 0, band.width, max(1, band.height // 5))))
            content_top = None
            if min(upper.mean) >= 225 and max(upper.stddev) <= 12:
                occupied = []
                for y in range(band.height):
                    pixels = band.crop((0, y, band.width, y + 1)).tobytes()
                    dark = sum(
                        min(pixels[i : i + 3]) < min(upper.mean) - 40
                        for i in range(0, len(pixels), 3)
                    )
                    occupied.append(dark / band.width >= 0.08)
                first = next(
                    (i for i in range(len(occupied) - 2) if all(occupied[i : i + 3])), None
                )
                if first is not None and 0.2 <= first / band.height <= 0.7:
                    content_top = first / band.height
            return {
                "asset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "image_size": [rgba.width, rgba.height],
                "opaque": alpha.getextrema()[0] >= 250,
                "lower_fade": lower_fade,
                "alpha_column": values if lower_fade else [],
                "content_top_fraction": content_top,
            }
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def _resource_key(source, root):
    resources = []
    for asset in sorted(
        {e.get("asset_id", "") for e in source["elements"] if e.get("type") == "image"}
    ):
        path = (root / asset).resolve()
        if not asset or not path.is_relative_to(root) or not path.is_file():
            resources.append((asset, 0, 0))
            continue
        stat = path.stat()
        resources.append((asset, stat.st_mtime_ns, stat.st_size))
    return tuple(resources)


@lru_cache(maxsize=64)
def _analyze(source_json, assets_dir, resource_key, native):
    source = json.loads(source_json)
    by_id = {e["id"]: e for e in source["elements"]}
    metadata = {}
    for asset, modified, size in resource_key:
        if size:
            metadata[asset] = _pixels(str(Path(assets_dir) / asset), modified, size)
    images = []
    for e in source["elements"]:
        if e.get("type") != "image":
            continue
        pixels = metadata.get(e.get("asset_id")) or {}
        images.append(
            {
                "element_id": e["id"],
                "role": "foreground",
                "confidence": 0.0,
                "evidence": ["no source proof authorizes background cropping"],
                "asset_sha256": pixels.get("asset_sha256"),
                "allow_cover": False,
                "allow_support_reflow": False,
                "allow_crop_position": False,
                "support_ids": [],
                "image_size": pixels.get("image_size"),
                "content_top_fraction": pixels.get("content_top_fraction"),
            }
        )
    rows = {row["element_id"]: row for row in images}
    canvas = (0, 0, source["canvas"]["width"], source["canvas"]["height"])
    overlays = infer_source_overlays(source, assets_dir)
    for attachment in overlays:
        photo, fade = by_id[attachment.underlay_id], by_id[attachment.overlay_id]
        pframe, fframe = _frame(photo), _frame(fade)
        pp = metadata.get(photo.get("asset_id")) or {}
        fp = metadata.get(fade.get("asset_id")) or {}
        if (
            not pframe
            or not fframe
            or not _inside(canvas, pframe)
            or photo.get("source_art", {}).get("kind") != "masked-photo"
            or photo.get("role") in {"product", "hero", "logo"}
            or photo.get("group_id")
            or fade.get("group_id")
            or photo.get("rotation", 0)
            or fade.get("rotation", 0)
            or photo.get("opacity", 1) < 0.995
            or not pp.get("opaque")
            or not fp.get("lower_fade")
            or fade.get("source_art", {}).get("native_type") != "rect"
        ):
            continue
        # Switching fill to cover must be an appearance-equivalent operation
        # on the source; do not legitimize a previously stretched photo.
        if any(
            e.get("fit", "contain") not in {"fill", "cover"}
            or abs(frame[2] / frame[3] / (p["image_size"][0] / p["image_size"][1]) - 1) > 1e-6
            for e, frame, p in ((photo, pframe, pp), (fade, fframe, fp))
        ):
            continue
        supported = [
            e["id"]
            for e in source["elements"]
            if e.get("type") == "text"
            and str(e.get("content", "")).strip()
            and e.get("z_index", 0) > fade.get("z_index", 0)
            and (frame := _frame(e))
            and _inside(frame, fframe)
        ]
        if len(supported) < 2:
            continue
        rows[photo["id"]].update(
            role="background",
            confidence=0.95,
            allow_cover=True,
            allow_crop_position=native,
            support_ids=[fade["id"]],
            evidence=[
                "opaque source photo fills the authored canvas",
                "source aspect matches original asset pixels",
                "verified fade beneath multiple live captions",
                "no explicit foreground product/hero/logo role",
            ],
        )
        rows[fade["id"]].update(
            role="background_support",
            confidence=0.95,
            allow_cover=False,
            allow_support_reflow=True,
            support_ids=supported,
            alpha_column=fp["alpha_column"],
            evidence=[
                "independent source overlay attachment",
                "horizontally invariant, monotone lower fade pixels",
                "source aspect matches original asset pixels",
            ],
        )
    result = {
        "schema_version": 1,
        "source_digest": _digest(source),
        "images": images,
        "method": "source-geometry-and-local-pixels",
        "model_applied": False,
    }
    result["scene_digest"] = _digest(result)
    return result


def analyze_scene(source, *, assets_dir=None, fonts_dir=None):
    """Return a fresh finite-JSON profile, cached by source and local resources."""
    if len(source["elements"]) > 256:
        raise ValueError("Scene element budget exceeded.")
    with using_resources(assets_dir, fonts_dir):
        from engine.native_typography import uses_native_typography

        root = get_resources().assets_dir.resolve()
        return deepcopy(
            _analyze(
                _canonical(source),
                str(root),
                _resource_key(source, root),
                uses_native_typography(get_resources().fonts_dir),
            )
        )


def approved_cover(profile, element, canvas):
    """Narrow source-derived exception to immutable fit / frame-aspect checks.

    Cover preserves pixel proportions while changing the crop viewport. The
    frame remains in bounds. It never grants permission to stretch image pixels.
    """
    row = next((r for r in profile["images"] if r["element_id"] == element.get("id")), None)
    frame = _frame(element)
    if not row or not row["allow_cover"] or element.get("fit") != "cover" or not frame:
        return False
    x, y, w, h = frame
    width, height = canvas["width"], canvas["height"]
    if abs(x) > 1e-6 or abs(w - width) > 1e-6:
        return False
    return row["role"] == "background" and abs(y) <= 1e-6 and abs(h - height) <= 1e-6


def approved_support(profile, element, canvas):
    """Resize a pixel-proven native gradient rectangle's complete paint field.

    Every horizontal source row contains the same pixels. Adapting the field
    retains its transparent edge and gradient, without stretching a pictured
    object. Arbitrary artwork and photographs never receive this permission.
    """
    row = next((r for r in profile["images"] if r["element_id"] == element.get("id")), None)
    frame = _frame(element)
    if not row or not row.get("allow_support_reflow") or element.get("fit") != "fill" or not frame:
        return False
    x, y, w, h = frame
    return (
        abs(x) <= 1e-6
        and abs(w - canvas["width"]) <= 1e-6
        and 0 <= y < canvas["height"]
        and abs(y + h - canvas["height"]) <= 1e-6
    )


def cover_position(row, canvas):
    """Keep source-evidenced upper headroom when a center crop would cut it.

    Sources without a bright quiet upper field retain the established center
    crop. A future reviewed model ROI can express richer intent separately.
    """
    onset, size = row.get("content_top_fraction"), row.get("image_size")
    if onset is None or not size or not row.get("allow_crop_position"):
        return [0.5, 0.5]
    scale = max(canvas["width"] / size[0], canvas["height"] / size[1])
    viewport_height = canvas["height"] / scale
    slack = size[1] - viewport_height
    if slack <= 1e-6:
        return [0.5, 0.5]
    desired_top = max(0, onset * size[1] - viewport_height * 0.08)
    return [0.5, min(0.5, desired_top / slack)]


def _alpha_at(row, element, y):
    size = row["image_size"]
    scale = (
        element["height"] / size[1]
        if element.get("fit") == "fill"
        else max(element["width"] / size[0], element["height"] / size[1])
    )
    pixel_y = (y - element["y"] - (element["height"] - size[1] * scale) / 2) / scale
    if pixel_y < 0 or pixel_y >= size[1]:
        return 0
    return row["alpha_column"][min(size[1] - 1, max(0, int(pixel_y)))]


def support_retained(source, candidate, profile, support_id):
    """A resized fade must cover its captions at least as strongly as before."""
    row = next((r for r in profile["images"] if r["element_id"] == support_id), None)
    if not row or row["role"] != "background_support" or not row.get("alpha_column"):
        return False
    original = {e["id"]: e for e in source["elements"]}
    changed = {e["id"]: e for e in candidate["elements"]}
    layer = changed.get(support_id)
    if layer is None or not approved_support(profile, layer, candidate["canvas"]):
        return False
    before = original[support_id]
    for eid in row["support_ids"]:
        if eid not in changed or not _inside(_frame(changed[eid]), _frame(layer)):
            return False
        if _alpha_at(row, layer, changed[eid]["y"]) + 3 < _alpha_at(
            row, before, original[eid]["y"]
        ):
            return False
    return True


def background_coverage(source, candidate, profile=None):
    """Viewport coverage only; this does not claim semantic subject retention."""
    profile = profile or analyze_scene(source)
    changed = {e["id"]: e for e in candidate["elements"]}
    width, height = candidate["canvas"]["width"], candidate["canvas"]["height"]
    coverages = []
    for row in profile["images"]:
        if row["role"] == "background":
            e = changed[row["element_id"]]
            visible_w = max(0, min(width, e["x"] + e["width"]) - max(0, e["x"]))
            visible_h = max(0, min(height, e["y"] + e["height"]) - max(0, e["y"]))
            coverages.append(visible_w * visible_h / (width * height))
    return min(coverages, default=1.0)
