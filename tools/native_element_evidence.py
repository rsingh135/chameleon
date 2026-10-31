"""Bounded isolated native paint, without placement or design-quality authority.

Use one existing native worker on one thread. In an engine process the caller
must already own its renderer lock; this module never imports the viewer lock.
Its RLock serializes helper calls, not other renderer users. Native renderer
caches assume immutable resource paths. Metadata pins detect changed bytes on
later calls; use a fresh process/package after a change, never clear shared caches.
Only encoded PNG bytes and JSON receipts leave this synchronous CPU helper.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import sys
import threading
from copy import deepcopy
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from engine.image_viewport import image_position
from engine.layout.appearance import visible_box
from engine.native_typography import uses_native_typography
from engine.resources import get_renderer, native_render_example, render_input, using_resources
from engine.semantic_ink import _require_exact_faces, _rotated_frame
from experiments.vision_critic.contracts import safe_artifact_path
from renderer.converter.colorutil import parse_rgba
from renderer.converter.convert import convert_chameleon_to_crello
from viewer.resize_cache import ResourceBindings

ISOLATION_V1 = "native-element-isolation-v1"
_POLICY = {
    "version": ISOLATION_V1,
    "max_native_axis": 1024,
    "max_native_pixels": 4_000_000,
    "max_logical_extent": 1_000_000,
    "max_thumbnail_axis": 512,
    "alpha_padding": 8,
    "checker": {"colors_rgba": [[224, 224, 224, 255], [192, 192, 192, 255]], "cell_pixels": 8},
}
_ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.RLock()
_BINDINGS = ResourceBindings()
_WORKER_THREAD = None
_CODE_PIN = None
_COMMON = {
    "id",
    "type",
    "x",
    "y",
    "width",
    "height",
    "rotation",
    "opacity",
    "z_index",
    "group_id",
    "role",
    "semantic_role",
    "priority",
    "name",
    "source_art",
}
_FIELDS = {
    "text": {
        "content",
        "font_family",
        "font_size",
        "font_weight",
        "font_style",
        "color",
        "text_align",
        "line_height",
        "letter_spacing",
        "max_lines",
        "style_runs",
        "native_text_effects",
        "doory_font",
        "svg_font_family",
        "svg_font_style",
        "svg_font_weight",
    },
    "shape": {"shape_type", "fill_color", "stroke_color", "stroke_width", "corner_radius"},
    "image": {"asset_id", "fit", "image_position"},
}
_SHAPE_COLOR = re.compile(
    r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)"
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _number(value, low, high):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError("Native input has nonfinite, unsupported or excessive numeric values")
    return value


def _shape_color(value):
    if not isinstance(value, str) or _SHAPE_COLOR.fullmatch(value.strip()) is None:
        raise ValueError("Unsupported native shape color syntax; exact rgb/rgba is required")
    color = parse_rgba(value)
    for channel in color[:3]:
        _number(channel, 0, 255)
    _number(color[3], 0, 1)
    return color


def _validate(element):
    if not isinstance(element, dict) or len(_canonical(element).encode("utf-8")) > 256 * 1024:
        raise ValueError("Native element must be a bounded finite JSON object")
    kind = element.get("type")
    if kind not in _FIELDS or any(key not in _COMMON | _FIELDS[kind] for key in element):
        raise ValueError("Unsupported native element type or fields")
    for key in ("x", "y", "width", "height"):
        _number(element.get(key), -1_000_000 if key in {"x", "y"} else 0.000001, 1_000_000)
    for key, low, high in (
        ("rotation", -3600, 3600),
        ("opacity", 0, 1),
        ("stroke_width", 0, 1_000_000),
        ("corner_radius", 0, 1_000_000),
        ("line_height", 0.01, 10),
        ("letter_spacing", -1000, 1000),
    ):
        if key in element:
            _number(element[key], low, high)
    if kind == "text":
        _number(element.get("font_size"), 1, 8192)
        if not isinstance(element.get("content"), str) or len(element["content"]) > 20000:
            raise ValueError("Text exceeds the native content budget")
        for key, allowed in (
            ("font_weight", {"regular", "bold"}),
            ("font_style", {"normal", "italic"}),
            ("text_align", {"left", "center", "right"}),
        ):
            if key in element and element[key] not in allowed:
                raise ValueError("Unsupported native text face or alignment")
        runs = element.get("style_runs", [])
        if not isinstance(runs, list) or len(runs) > len(element["content"]) + 1:
            raise ValueError("Unsupported native text style runs")
        for run in runs:
            if (
                not isinstance(run, dict)
                or set(run) - {"start", "end", "color", "font_weight", "font_style"}
                or type(run.get("start")) is not int
                or type(run.get("end")) is not int
                or not 0 <= run["start"] <= run["end"] <= len(element["content"])
            ):
                raise ValueError("Unsupported native text style run range or fields")
            if run.get("font_weight", "regular") not in {"regular", "bold"} or run.get(
                "font_style", "normal"
            ) not in {"normal", "italic"}:
                raise ValueError("Unsupported native text style run face")
    elif kind == "shape":
        if element.get("shape_type") not in {"rectangle", "ellipse", "line"}:
            raise ValueError("Unsupported native shape type")
        fill = _shape_color(element.get("fill_color"))
        if element["shape_type"] != "line" and element.get("stroke_width", 0) > 0:
            raise ValueError("Native rectangle/ellipse converter does not preserve authored stroke")
        if element["shape_type"] == "line":
            stroke = _shape_color(element.get("stroke_color") or element.get("fill_color"))
            if stroke[3] != fill[3]:
                raise ValueError("Native line converter does not preserve separate stroke alpha")
    elif element.get("fit", "contain") not in {"contain", "cover", "fill"}:
        raise ValueError("Unsupported original image fit")
    if "image_position" in element:
        image_position(element["image_position"])
        if element.get("fit") != "cover":
            raise ValueError("Native image_position requires the authored cover viewport")


def _resources(element, assets, fonts):
    if not assets.is_dir() or not fonts.is_dir():
        raise ValueError("Explicit native asset and font directories must exist")
    font_files = [
        p for p in fonts.rglob("*") if p.is_file() and p.suffix.lower() in {".ttf", ".otf", ".json"}
    ]
    if len(font_files) > 200:
        raise ValueError("Native font inventory exceeds 200 files")
    total_font_bytes = 0
    for path in font_files:
        safe_artifact_path(fonts, path.relative_to(fonts).as_posix())
        size = path.stat().st_size
        total_font_bytes += size
        if path.suffix.lower() == ".json" and size > 1024 * 1024:
            raise ValueError("Native font manifest exceeds its one-MiB byte budget")
        if total_font_bytes > 64 * 1024 * 1024:
            raise ValueError("Native font inventory exceeds its aggregate 64-MiB byte budget")
    width, height = max(1, round(element["width"])), max(1, round(element["height"]))
    if element["type"] != "text" and width * height > 4_000_000:
        raise ValueError("Native converter intermediate exceeds its pixel budget")
    if element["type"] == "image":
        path = safe_artifact_path(assets, element.get("asset_id"))
        if not path.is_file() or path.stat().st_size > 32_000_000:
            raise ValueError("Native original asset is missing or exceeds its byte budget")
        with Image.open(path) as original:
            if (
                original.width * original.height > 20_000_000
                or getattr(original, "n_frames", 1) != 1
            ):
                raise ValueError("Native original image is oversized or animated")
            if element.get("fit", "contain") in {"contain", "cover"}:
                scale = (min if element.get("fit", "contain") == "contain" else max)(
                    width / original.width, height / original.height
                )
                if (
                    max(1, round(original.width * scale)) * max(1, round(original.height * scale))
                    > 4_000_000
                ):
                    raise ValueError("Native image fit intermediate exceeds its pixel budget")
            original.verify()
    return _BINDINGS.bind({"elements": [element]}, assets, fonts)


def _implementation():
    import cr_renderer
    import skia

    paths = {
        *(_ROOT / "engine").rglob("*.py"),
        *(_ROOT / "renderer").rglob("*.py"),
        Path(__file__).resolve(),
        _ROOT / "experiments/vision_critic/contracts.py",
        _ROOT / "viewer/resize_cache.py",
        _ROOT / "tools/scene_harness.py",
    }
    paths.update(Path(cr_renderer.__file__).parent.rglob("*.py"))
    paths.update(Path(module.__file__).resolve() for module in (skia, Image, Image.core))
    hashes = {}
    for path in sorted(paths):
        name = path.relative_to(_ROOT).as_posix() if path.is_relative_to(_ROOT) else str(path)
        hashes[name] = _sha(path)
    runtime = {"python": sys.version}
    for name in ("Pillow", "skia-python", "cr-renderer"):
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = "not-installed-as-distribution"
    return hashes, runtime


def _checker(image):
    colors, cell = _POLICY["checker"]["colors_rgba"], 8
    background = Image.new("RGBA", image.size, tuple(colors[0]))
    draw = ImageDraw.Draw(background)
    for y in range(0, image.height, cell):
        for x in range(0, image.width, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=tuple(colors[1]))
    return Image.alpha_composite(background, image).convert("RGB")


def _paint(element, assets, fonts, receipt):
    with using_resources(assets, fonts):
        manifest = uses_native_typography(fonts)
        receipt["manifest_native_preprocess"] = manifest
        if not manifest and image_position(element.get("image_position")) != (0.5, 0.5):
            raise ValueError(
                "Noncenter native image viewport requires manifest-backed preprocessing"
            )
        if not manifest and element.get("style_runs"):
            raise ValueError(
                "End-exclusive native style runs require manifest-backed preprocessing"
            )
        renderer = get_renderer(fonts)
        receipt["renderer_class"] = type(renderer).__module__ + "." + type(renderer).__qualname__
        if element["type"] == "text":
            descriptor = element.get("native_text_effects")
            if descriptor is not None:
                if not manifest:
                    raise ValueError(
                        "Native text effects require an exact manifest-backed renderer"
                    )
                from engine.native_text_effects import validate_descriptor

                validate_descriptor(descriptor, element["content"])
            if not manifest and element.get("opacity", 1) != 1:
                raise ValueError("Bundled text renderer cannot prove nondefault text opacity")
            _require_exact_faces(element, renderer.font_manager, fonts, manifest)
        frame, visible = _rotated_frame(element), visible_box(element)
        left, top = min(frame[0], visible.x), min(frame[1], visible.y)
        right = max(frame[0] + frame[2], visible.x + visible.width)
        bottom = max(frame[1] + frame[3], visible.y + visible.height)
        outline = (element.get("native_text_effects") or {}).get("outline") or {}
        font = float(element.get("font_size", 0))
        padding = max(
            16.0,
            font * (2 + float(outline.get("width_em", 0)) / 2),
            float(element.get("stroke_width", 0)) * 2,
        )
        _number(padding, 0, 1_000_000)
        origin = [math.floor(left - padding), math.floor(top - padding)]
        width, height = (
            math.ceil(right + padding) - origin[0],
            math.ceil(bottom + padding) - origin[1],
        )
        if max(width, height) > 1_000_000:
            raise ValueError("Logical isolation extent exceeds one-million-coordinate budget")
        maximum = max(1, min(1024, max(width, height)))
        scale = min(1.0, maximum / max(width, height))
        expected = (round(width * scale), round(height * scale))
        if min(expected) < 1 or expected[0] * expected[1] > 4_000_000:
            raise ValueError("Uniform native raster is empty or exceeds its pixel budget")
        local = deepcopy(element)
        local["x"] -= origin[0]
        local["y"] -= origin[1]
        design = {"canvas": {"width": width, "height": height}, "elements": [local]}
        example, warnings = convert_chameleon_to_crello(
            render_input(design), renderer.font_manager, str(assets)
        )
        if warnings:
            raise ValueError("Native conversion warnings: " + "; ".join(map(str, warnings)))
        example = native_render_example(example, design)
        encoded = renderer.render(example, max_size=maximum, canvas_color=None, format="png")
        with Image.open(BytesIO(encoded)) as image:
            if image.size != expected or image.mode != "RGBA":
                raise ValueError(
                    "Native isolated raster differs from its planned size or RGBA mode"
                )
            rgba = image.copy()
    bounds = rgba.getchannel("A").getbbox()
    receipt.update(
        original_frame_xywh=[element[k] for k in ("x", "y", "width", "height")],
        original_rotated_frame_xywh=list(frame),
        logical_canvas={"width": width, "height": height},
        source_origin_xy=origin,
        translation_xy=[-origin[0], -origin[1]],
        raster_scale=scale,
        native_raster_size=list(expected),
        logical_outer_padding=padding,
        native_alpha_bbox_xyxy=list(bounds) if bounds else None,
        outer_border_clipping_checked=True,
        clipped=False,
        original_opacity=element.get("opacity", 1),
        original_rotation=element.get("rotation", 0),
    )
    if bounds is not None and (
        min(bounds[:2]) <= 0 or bounds[2] >= rgba.width or bounds[3] >= rgba.height
    ):
        receipt["clipped"] = True
        raise ValueError("Native alpha touches outer guard canvas; paint may be clipped")
    if bounds is None:
        cropped, visible_source = Image.new("RGBA", (32, 32), (0, 0, 0, 0)), None
    else:
        ink = rgba.crop(bounds)
        cropped = Image.new("RGBA", (ink.width + 16, ink.height + 16), (0, 0, 0, 0))
        cropped.paste(ink, (8, 8))
        visible_source = [
            origin[0] + bounds[0] / scale,
            origin[1] + bounds[1] / scale,
            (bounds[2] - bounds[0]) / scale,
            (bounds[3] - bounds[1]) / scale,
        ]
    unscaled = list(cropped.size)
    thumb_scale = min(1.0, 512 / max(cropped.size))
    cropped.thumbnail((512, 512), Image.Resampling.LANCZOS)
    alpha = cropped.getchannel("A")
    histogram, pixels = alpha.histogram(), cropped.width * cropped.height
    receipt.update(
        paint_empty=bounds is None,
        source_visible_alpha_box_xywh=visible_source,
        padded_alpha_size_before_thumbnail=unscaled,
        thumbnail_scale=thumb_scale,
        thumbnail_size=list(cropped.size),
        alpha_extrema=list(alpha.getextrema()),
        alpha_nonzero_fraction=1 - histogram[0] / pixels,
        alpha_mean=sum(i * n for i, n in enumerate(histogram)) / (pixels * 255),
        combined_source_to_thumbnail_scale=scale * thumb_scale,
    )
    alpha_output, checker_output = BytesIO(), BytesIO()
    cropped.save(alpha_output, format="PNG")
    _checker(cropped).save(checker_output, format="PNG")
    return alpha_output.getvalue(), checker_output.getvalue()


def render_isolated_element(element, *, assets_dir, fonts_dir, policy=ISOLATION_V1):
    """Return diagnostic native PNG bytes or explicit unavailable, never a fallback.

    Resource paths must be immutable, and callers must own one existing native
    worker/thread and any engine renderer lock. This helper's lock alone does
    not establish native-object ownership outside this module.
    """
    global _WORKER_THREAD, _CODE_PIN
    receipt = {
        "protocol": ISOLATION_V1,
        "policy": deepcopy(_POLICY),
        "geometry_authority": False,
        "model_executed": False,
        "frame_or_asset_fallback_used": False,
        "native_single_original_layer": True,
        "checker_is_not_original_design_context": True,
        "single_native_worker_required": True,
        "module_lock_scope": "helper_calls_only",
        "resource_paths_immutable_required": True,
        "resources_unchanged": False,
    }
    alpha_png = checker_png = None
    with _LOCK:
        try:
            if policy != ISOLATION_V1:
                raise ValueError("Unsupported native isolation policy")
            frozen = deepcopy(element)
            _validate(frozen)
            receipt["source_element_sha256"] = _digest(frozen)
            assets, fonts = Path(assets_dir).resolve(), Path(fonts_dir).resolve()
            bound = _resources(frozen, assets, fonts)
            receipt["resource_identity"] = {
                "assets_dir": str(assets),
                "fonts_dir": str(fonts),
                "resource_digest": bound.digest,
                "hashes": bound.hashes(),
            }
            worker = threading.current_thread()
            if _WORKER_THREAD is not None and _WORKER_THREAD is not worker:
                raise ValueError("Native helper requires its original single worker thread")
            _WORKER_THREAD = worker
            implementation, runtime = _implementation()
            implementation_digest = _digest({"files": implementation, "runtime": runtime})
            if _CODE_PIN is not None and implementation_digest != _CODE_PIN:
                raise ValueError("Native implementation changed; use a fresh isolated worker")
            _CODE_PIN = implementation_digest
            receipt.update(
                implementation_hashes=implementation,
                runtime=runtime,
                implementation_sha256=implementation_digest,
            )
            alpha_png, checker_png = _paint(frozen, assets, fonts, receipt)
            current = _resources(frozen, assets, fonts)
            if current.digest != bound.digest or _implementation() != (implementation, runtime):
                raise ValueError("Native resources or implementation changed during paint")
            if element != frozen:
                raise ValueError("Original element changed during isolated native paint")
            receipt.update(
                resources_unchanged=True,
                status="available",
                reason=None,
                alpha_png_sha256=hashlib.sha256(alpha_png).hexdigest(),
                checker_png_sha256=hashlib.sha256(checker_png).hexdigest(),
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            LookupError,
            OSError,
            RuntimeError,
            OverflowError,
            AttributeError,
            AssertionError,
            ImportError,
            RecursionError,
            MemoryError,
            Image.DecompressionBombError,
        ) as exc:
            receipt.update(status="unavailable", reason=f"{type(exc).__name__}: {str(exc)[:400]}")
            alpha_png = checker_png = None
    receipt["receipt_digest"] = _digest(receipt)
    return {
        "status": receipt["status"],
        "reason": receipt["reason"],
        "alpha_png": alpha_png,
        "checker_png": checker_png,
        "receipt": receipt,
    }
