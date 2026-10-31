"""Source-coordinate ink evidence from isolated, bounded native text painting.

Measurements are uncached. Call under the existing renderer lock; this module
does not start threads, share native Fonts, or change a document/renderer.
Image bounds remain an explicitly approximate alpha/frame diagnostic.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from io import BytesIO

from PIL import Image

from engine.layout.appearance import visible_box
from engine.resources import get_renderer, native_render_example, render_input, using_resources
from renderer.converter.convert import convert_chameleon_to_crello

MAX_RASTER_PIXELS = 4_000_000
MAX_RASTER_AXIS = 4096
MAX_LOGICAL_EXTENT = 1_000_000


def _fallback(element, reason, box=None):
    element = element if isinstance(element, dict) else {}
    if box is None:
        box = []
        for key in ("x", "y", "width", "height"):
            value = element.get(key, 0)
            box.append(
                float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0
            )
    return {
        "box": box,
        "provenance": "frame_fallback",
        "confidence": 0.0 if box[2] <= 0 or box[3] <= 0 else 0.25,
        "raster_scale": 0.0,
        "reason": reason,
    }


def _style(weight, italic):
    return (
        "bolditalic"
        if weight == "bold" and italic
        else "italic"
        if italic
        else "bold"
        if weight == "bold"
        else "regular"
    )


def _require_exact_faces(element, manager, fonts, manifest):
    """Reject the vendor's silent fallback and stale same-directory font bytes."""
    from cr_renderer.fonts import get_font_style, get_font_weight, normalize_family

    family = element["font_family"]
    weight = element.get("font_weight", "regular")
    italic = element.get("font_style", "normal") == "italic"
    faces = {(family, weight, _style(weight, italic))}
    for run in element.get("style_runs", []):
        run_weight = run.get("font_weight", weight)
        run_italic = run.get("font_style", "italic" if italic else "normal") == "italic"
        faces.add((family, run_weight, _style(run_weight, run_italic)))
    descriptor = element.get("native_text_effects") or {}
    for run in descriptor.get("runs", []):
        faces.add((run["font_family"], "regular", "regular"))
    if manifest:
        entries = json.loads((fonts / "font-manifest.json").read_text(encoding="utf-8"))
        entries = entries.get("fonts") if isinstance(entries, dict) else entries
        current = {}
        for entry in entries:
            path = (fonts / entry["file"]).resolve()
            if not path.is_relative_to(fonts):
                raise ValueError("Font path leaves the document directory")
            key = (
                normalize_family(entry["font_family"]),
                entry["weight"],
                _style(entry["weight"], entry["style"] == "italic"),
            )
            current[key] = path.read_bytes()
    else:
        from renderer.converter.brand_fonts import build_font_dict

        current = {
            (name, face["fontWeight"], face["fontStyle"]): face["bytes"]
            for name, values in build_font_dict(str(fonts)).items()
            for face in values
        }
    for face_family, face_weight, face_style in faces:
        if not manifest:
            face_weight = get_font_weight(face_family, face_weight)
            face_style = get_font_style(face_family, face_style)
        key = normalize_family(face_family), face_weight, face_style
        if key not in current:
            raise LookupError(
                f"Exact font face unavailable: {face_family}/{face_weight}/{face_style}"
            )
        if manager.lookup(face_family, face_weight, face_style) != current[key]:
            raise LookupError(
                "Font bytes changed after renderer loading; use immutable resources or restart"
            )


def _rotated_frame(element):
    x, y, width, height = (float(element[key]) for key in ("x", "y", "width", "height"))
    angle = math.radians(float(element.get("rotation", 0)))
    rotated_width = abs(math.cos(angle)) * width + abs(math.sin(angle)) * height
    rotated_height = abs(math.sin(angle)) * width + abs(math.cos(angle)) * height
    return (
        x + (width - rotated_width) / 2,
        y + (height - rotated_height) / 2,
        rotated_width,
        rotated_height,
    )


def _raster_limit(width, height):
    longest = max(width, height)
    maximum = max(
        1,
        min(
            MAX_RASTER_AXIS,
            longest,
            int(longest * min(1.0, math.sqrt(MAX_RASTER_PIXELS / (width * height)))),
        ),
    )
    while maximum > 0:
        scale = min(1.0, maximum / longest)
        size = round(width * scale), round(height * scale)
        if min(size) < 1:
            raise ValueError("Frame is too thin for a bounded uniform raster")
        if max(size) <= MAX_RASTER_AXIS and size[0] * size[1] <= MAX_RASTER_PIXELS:
            return maximum, scale, size
        maximum -= 1
    raise ValueError("No bounded raster size is available")


def element_ink_bounds(element, *, assets_dir=None, fonts_dir=None) -> dict:
    """Return source-space `[x,y,width,height]` plus measurement provenance.

    Native alpha includes exact live text styles, fonts, opacity and supported
    effects. Its uncertainty is at least one raster pixel (`1/raster_scale` in
    source coordinates). Empty paint returns zero size. A clipped, unsupported
    or failed measurement returns a frame fallback with an explicit reason.
    """
    try:
        frozen = deepcopy(element)
        values = [frozen[key] for key in ("x", "y", "width", "height")]
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
            or min(values[2:]) <= 0
        ):
            raise ValueError("Element geometry must be finite with positive size")
        if frozen.get("type") != "text":
            if frozen.get("type") == "image":
                with using_resources(assets_dir, fonts_dir):
                    box = visible_box(frozen)
                result = _fallback(
                    frozen,
                    "Approximate fitted image alpha/frame bounds; not a native paint measurement",
                    [box.x, box.y, box.width, box.height],
                )
                result["confidence"] = 0.5
                return result
            return _fallback(frozen, "Non-text frame geometry; native text alpha was not measured")
        size = frozen.get("font_size")
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, float))
            or not math.isfinite(size)
            or not 1 <= size <= 8192
        ):
            raise ValueError("Text font size exceeds the supported measurement budget")
        if not isinstance(frozen.get("content"), str) or len(frozen["content"]) > 20000:
            raise ValueError("Text content is missing or exceeds the measurement budget")
        with using_resources(assets_dir, fonts_dir) as resources:
            renderer = get_renderer(resources.fonts_dir)
            manifest = (resources.fonts_dir / "font-manifest.json").is_file()
            descriptor = frozen.get("native_text_effects")
            if descriptor is not None:
                if not manifest:
                    raise ValueError("Native text effects require a manifest-backed exact renderer")
                from engine.native_text_effects import validate_descriptor

                validate_descriptor(descriptor, frozen["content"])
            if not manifest and frozen.get("opacity", 1) != 1:
                raise ValueError("Bundled text renderer cannot prove nondefault text opacity")
            _require_exact_faces(frozen, renderer.font_manager, resources.fonts_dir, manifest)
            frame = _rotated_frame(frozen)
            occupied = visible_box(frozen)
            left, top = min(frame[0], occupied.x), min(frame[1], occupied.y)
            right = max(frame[0] + frame[2], occupied.x + occupied.width)
            bottom = max(frame[1] + frame[3], occupied.y + occupied.height)
            outline = (descriptor or {}).get("outline") or {}
            padding = max(4.0, 2 * size + size * float(outline.get("width_em", 0)) / 2)
            origin_x, origin_y = math.floor(left - padding), math.floor(top - padding)
            width, height = (
                math.ceil(right + padding) - origin_x,
                math.ceil(bottom + padding) - origin_y,
            )
            if max(width, height) > MAX_LOGICAL_EXTENT:
                raise ValueError(
                    "Text/effect extent exceeds the bounded logical measurement budget"
                )
            maximum, scale, expected_size = _raster_limit(width, height)
            local = deepcopy(frozen)
            local["x"] -= origin_x
            local["y"] -= origin_y
            design = {"canvas": {"width": width, "height": height}, "elements": [local]}
            example, warnings = convert_chameleon_to_crello(
                render_input(design), renderer.font_manager, str(resources.assets_dir)
            )
            if warnings:
                raise ValueError("Native conversion warnings: " + "; ".join(map(str, warnings)))
            example = native_render_example(example, design)
            encoded = renderer.render(example, max_size=maximum, canvas_color=None, format="png")
            with Image.open(BytesIO(encoded)) as image:
                if (
                    image.size != expected_size
                    or image.width * image.height > MAX_RASTER_PIXELS
                    or max(image.size) > MAX_RASTER_AXIS
                ):
                    raise ValueError("Native renderer returned an unexpected or oversized raster")
                alpha = image.convert("RGBA").getchannel("A")
                bounds = alpha.getbbox()
            if bounds is None:
                return {
                    "box": [float(frozen["x"]), float(frozen["y"]), 0.0, 0.0],
                    "provenance": "native_alpha",
                    "confidence": 1.0,
                    "raster_scale": scale,
                    "reason": "Native paint is fully transparent or empty",
                }
            if (
                min(bounds[:2]) <= 0
                or bounds[2] >= expected_size[0]
                or bounds[3] >= expected_size[1]
            ):
                raise ValueError("Ink reaches the measurement boundary; extent may be clipped")
            return {
                "box": [
                    origin_x + bounds[0] / scale,
                    origin_y + bounds[1] / scale,
                    (bounds[2] - bounds[0]) / scale,
                    (bounds[3] - bounds[1]) / scale,
                ],
                "provenance": "native_alpha",
                "confidence": 1.0 if scale == 1 else 0.8,
                "raster_scale": scale,
            }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        LookupError,
        RuntimeError,
        OverflowError,
        AttributeError,
        AssertionError,
    ) as error:
        return _fallback(
            element, f"Native ink unavailable: {type(error).__name__}: {str(error)[:240]}"
        )
