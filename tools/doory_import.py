"""Static, disclosed import of pinned Doory authoring JSON; never runs upstream code.

Author thumbnails remain independent, low-resolution references. Pinned named
font faces are preserved, but historical byte identity with upstream builds is
unknown. Editable text is never silently replaced by a screenshot.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import shutil
from copy import deepcopy
from pathlib import Path
from xml.sax.saxutils import quoteattr

from PIL import Image, ImageColor

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
# Fabric7.4.0 from the original app's integrity-pinned package-lock: Text.ts
# _renderTextCommon/_renderChars and Text/constants.ts use these frame metrics.
FABRIC_LINE_HEIGHT_EM = 1.13
FABRIC_BASELINE_EM = FABRIC_LINE_HEIGHT_EM * (1 - 0.222)


class ImportBlock(ValueError):
    def __init__(self, code, message, node_id=None):
        super().__init__(message)
        self.code, self.node_id = code, node_id

    def record(self):
        return {"code": self.code, "node_id": self.node_id, "message": str(self)}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ImportBlock("path_escape", "A poster resource leaves its corpus directory")
    return path


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ImportBlock("invalid_geometry", f"{name} must be a finite number")
    return float(value)


def _mul(a, b):
    return (
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    )


def _matrix(node):
    sx, sy = (_number(node.get(k, 1), k) for k in ("scaleX", "scaleY"))
    if (
        min(sx, sy) <= 0
        or abs(sx - sy) > 1e-6
        or any(node.get(k) for k in ("flipX", "flipY", "skewX", "skewY"))
    ):
        raise ImportBlock(
            "unsupported_transform",
            "Skew, reflection and nonuniform transforms cannot preserve editable type",
            node.get("id"),
        )
    angle = math.radians(_number(node.get("angle", 0), "angle"))
    c, s = math.cos(angle) * sx, math.sin(angle) * sx
    origins = {"left": 0.0, "top": 0.0, "center": 0.5, "right": 1.0, "bottom": 1.0}
    try:
        ox, oy = origins[node.get("originX", "left")], origins[node.get("originY", "top")]
    except KeyError as exc:
        raise ImportBlock("unsupported_origin", "Unknown object origin", node.get("id")) from exc
    dx = (0.5 - ox) * _number(node.get("width", 0), "width")
    dy = (0.5 - oy) * _number(node.get("height", 0), "height")
    return (
        c,
        s,
        -s,
        c,
        _number(node.get("left", 0), "left") + c * dx - s * dy,
        _number(node.get("top", 0), "top") + s * dx + c * dy,
    )


def _rgba(value):
    if value is None or value == "":
        return "rgba(0,0,0,0)"
    if not isinstance(value, str):
        raise ImportBlock("unsupported_paint", "This editable element requires a solid paint")
    try:
        r, g, b, a = ImageColor.getcolor(value, "RGBA")
    except ValueError as exc:
        raise ImportBlock("unsupported_paint", f"Unsupported paint: {value!r}") from exc
    return f"rgba({r},{g},{b},{a / 255:g})"


def _family(value):
    if not isinstance(value, str) or not value:
        raise ImportBlock("missing_font", "Text has no named font family")
    return value.split(",", 1)[0].strip().strip("\"'")


def _path(data):
    """Serialize author path data and compute tight M/L/C/Q/Z curve bounds."""
    lengths = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}
    if isinstance(data, str):
        tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", data)
        parts, offset = [], 0
        while offset < len(tokens):
            op = tokens[offset]
            if op not in lengths:
                raise ImportBlock(
                    "unsupported_path", "Only absolute M/L/C/Q/Z author paths are supported"
                )
            count = lengths[op]
            parts.append([op, *[float(v) for v in tokens[offset + 1 : offset + 1 + count]]])
            offset += count + 1
        data = parts
    if not isinstance(data, list) or not data:
        raise ImportBlock("unsupported_path", "A vector path requires command data")
    points, current, start, parts = [], (0.0, 0.0), (0.0, 0.0), []
    for command in data:
        if (
            not isinstance(command, list)
            or not command
            or command[0] not in lengths
            or len(command) != lengths[command[0]] + 1
        ):
            raise ImportBlock("unsupported_path", "Unsupported or malformed vector command")
        op, values = command[0], [_number(v, "path coordinate") for v in command[1:]]
        parts.append(op + " " + " ".join(f"{v:.12g}" for v in values))
        if op == "Z":
            current = start
            continue
        end = tuple(values[-2:])
        points.extend((current, end))
        if op == "M":
            start = end
        elif op in {"C", "Q"}:
            controls = [current] + [tuple(values[i : i + 2]) for i in range(0, len(values), 2)]
            for axis in (0, 1):
                p = [v[axis] for v in controls]
                if op == "C":
                    a, b, c = (
                        -p[0] + 3 * p[1] - 3 * p[2] + p[3],
                        p[0] - 2 * p[1] + p[2],
                        p[1] - p[0],
                    )
                    if abs(a) < 1e-12:
                        roots = [-c / (2 * b)] if abs(b) > 1e-12 else []
                    else:
                        discriminant = b * b - a * c
                        roots = (
                            [(-b - sign * math.sqrt(discriminant)) / a for sign in (-1, 1)]
                            if discriminant >= 0
                            else []
                        )
                else:
                    denominator = p[0] - 2 * p[1] + p[2]
                    roots = [(p[0] - p[1]) / denominator] if abs(denominator) > 1e-12 else []
                for t in roots:
                    if 0 < t < 1:
                        degree = len(controls) - 1
                        points.append(
                            tuple(
                                sum(
                                    math.comb(degree, i) * (1 - t) ** (degree - i) * t**i * point[k]
                                    for i, point in enumerate(controls)
                                )
                                for k in (0, 1)
                            )
                        )
        current = end
    # A leading M does not paint a segment from (0,0).
    if data[0][0] == "M":
        points = points[1:]
    left, top = min(p[0] for p in points), min(p[1] for p in points)
    right, bottom = max(p[0] for p in points), max(p[1] for p in points)
    return " ".join(parts), (left, top, right - left, bottom - top)


def _svg_matrix(world):
    return "matrix(" + " ".join(f"{v:.12g}" for v in world) + ")"


def _paint(value):
    if not isinstance(value, dict):
        return _rgba(value), ""
    if (
        value.get("type") != "linear"
        or value.get("gradientUnits", "percentage") != "percentage"
        or value.get("offsetX", 0)
        or value.get("offsetY", 0)
    ):
        raise ImportBlock(
            "unsupported_paint", "Only percentage linear gradients have an isolated artwork mapping"
        )
    coords, stops = value.get("coords", {}), value.get("colorStops", [])
    if not stops:
        raise ImportBlock("unsupported_paint", "Gradient has no stops")
    axes = " ".join(
        f'{axis}="{_number(coords.get(axis), axis):.12g}"' for axis in ("x1", "y1", "x2", "y2")
    )
    paint = []
    for stop in stops:
        offset = _number(stop.get("offset"), "gradient offset")
        if not 0 <= offset <= 1:
            raise ImportBlock("unsupported_paint", "Gradient offsets must remain inside 0..1")
        paint.append(
            f'<stop offset="{offset:g}" stop-color={quoteattr(_rgba(stop.get("color")))}/>'
        )
    return (
        "url(#paint)",
        f'<defs><linearGradient id="paint" {axes}>{"".join(paint)}</linearGradient></defs>',
    )


class _Importer:
    def __init__(self, entry, root, out, original, page_index):
        self.entry, self.root, self.out, self.original = entry, root, out, original
        self.page_index = page_index
        pages = original.get("pages")
        if (
            original.get("schemaVersion") != 1
            or not isinstance(pages, list)
            or not 0 <= page_index < len(pages)
        ):
            raise ImportBlock(
                "unsupported_document",
                "Expected a Doory schemaVersion 1 document and an existing page",
            )
        self.page = pages[page_index]
        self.width, self.height = original["width"], original["height"]
        if (
            any(type(v) is not int or not 1 <= v <= 8192 for v in (self.width, self.height))
            or self.width * self.height > 20_000_000
        ):
            raise ImportBlock(
                "unsupported_canvas", "Canvas exceeds the local engine's pixel limits"
            )
        self.elements, self.mapping, self.fonts, self.assets, self.warnings = [], [], {}, [], []
        self.seen, self.emitted_ids = set(), set()
        self.text_count = self.art_count = 0
        (out / "resources/assets").mkdir(parents=True, exist_ok=True)
        (out / "resources/fonts").mkdir(parents=True, exist_ok=True)

    def _font(self, node):
        from fontTools.ttLib import TTFont

        family = _family(node.get("fontFamily"))
        weight = str(node.get("fontWeight", 400))
        style = node.get("fontStyle", "normal")
        candidates = [
            f
            for f in self.entry.get("font_requirements", [])
            if f.get("family") == family
            and str(f.get("weight")) == weight
            and f.get("style", "normal") == style
        ]
        if len(candidates) != 1:
            raise ImportBlock(
                "missing_font",
                f"Supply the exact pinned {family} {weight} {style} face",
                node["id"],
            )
        face = candidates[0]
        data = _safe(self.root, face["path"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != face.get("sha256"):
            raise ImportBlock("font_hash", f"Pinned font bytes changed for {family}", node["id"])
        with TTFont(io.BytesIO(data)) as font:
            aliases = {record.toUnicode() for record in font["name"].names if record.nameID == 6}
            if face["postscript_name"] not in aliases:
                raise ImportBlock(
                    "font_alias", "Font PostScript name differs from pinned metadata", node["id"]
                )
            missing = sorted(
                {ord(c) for c in node["text"] if not c.isspace()} - set(font.getBestCmap())
            )
            if missing:
                raise ImportBlock(
                    "font_glyphs", f"Exact font lacks code points {missing}", node["id"]
                )
        alias = face["postscript_name"]
        if alias not in self.fonts:
            filename = digest + ".ttf"
            (self.out / "resources/fonts" / filename).write_bytes(data)
            self.fonts[alias] = {
                "font_family": alias,
                "file": filename,
                "weight": "regular",
                "style": "normal",
                "face": face,
                "data": data,
            }
        return alias

    def _emit(self, element, node, mode, world, group=None, **extra):
        if len(self.elements) >= 200:
            raise ImportBlock(
                "too_many_elements", "The expanded source exceeds 200 elements", node["id"]
            )
        element.update(id=element.get("id", node["id"]), z_index=len(self.elements))
        if element["id"] in self.emitted_ids:
            raise ImportBlock(
                "duplicate_element", "Generated and native element IDs collide", node["id"]
            )
        self.emitted_ids.add(element["id"])
        if group:
            element["group_id"] = group
        self.elements.append(element)
        self.mapping.append(
            {
                "node_id": node["id"],
                "name": node.get("name"),
                "element_id": element["id"],
                "mode": mode,
                "world_transform": list(world),
                **extra,
            }
        )

    def _frame(self, node, world, width=None, height=None):
        scale = math.hypot(world[0], world[1])
        w = _number(node["width"], "width") * scale if width is None else width
        h = _number(node["height"], "height") * scale if height is None else height
        return {
            "x": world[4] - w / 2,
            "y": world[5] - h / 2,
            "width": w,
            "height": h,
            "rotation": math.degrees(math.atan2(world[1], world[0])),
            "opacity": _number(node.get("opacity", 1), "opacity")
            if node.get("visible", True)
            else 0.0,
        }

    def _text(self, node, world, group):
        if any(
            node.get(k)
            for k in (
                "underline",
                "overline",
                "linethrough",
                "textBackgroundColor",
                "path",
                "splitByGrapheme",
            )
        ):
            raise ImportBlock(
                "unsupported_text_effect",
                "Decorated or path text has no live glyph mapping",
                node["id"],
            )
        if node.get("direction", "ltr") != "ltr" or node.get("textAlign", "left") not in {
            "left",
            "center",
            "right",
        }:
            raise ImportBlock(
                "unsupported_text_layout",
                "Only left-to-right plain text alignment is supported",
                node["id"],
            )
        alias = self._font(node)
        tracking_em = _number(node.get("charSpacing", 0), "charSpacing") / 1000
        descriptor = {"version": 1, "baseline_em": FABRIC_BASELINE_EM, "tracking_em": tracking_em}
        runs = []
        for run in node.get("styles") or []:
            style = run.get("style", {})
            start, end = run.get("start"), run.get("end")
            if (
                set(style) - {"fontFamily", "fontWeight", "fontStyle"}
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(node["text"])
            ):
                raise ImportBlock(
                    "mixed_text_faces", "Unsupported or invalid editable font run", node["id"]
                )
            exact = self._font({**node, **style, "text": node["text"][start:end]})
            runs.append({"start": start, "end": end, "font_family": exact})
        if runs:
            descriptor["runs"] = runs
        scale = math.hypot(world[0], world[1])
        size = _number(node["fontSize"], "fontSize") * scale
        if size < 1:
            raise ImportBlock(
                "unsupported_font_size",
                "Text smaller than 1 px is outside the engine contract",
                node["id"],
            )
        element = {
            **self._frame(node, world),
            "type": "text",
            "content": node["text"],
            "font_family": alias,
            "font_size": size,
            "font_weight": "regular",
            "font_style": "normal",
            "color": _rgba(node.get("fill"))
            if not isinstance(node.get("fill"), dict)
            else _rgba((node["fill"].get("colorStops") or [{}])[0].get("color")),
            "text_align": node.get("textAlign", "left"),
            "line_height": FABRIC_LINE_HEIGHT_EM
            * _number(node.get("lineHeight", 1.2), "lineHeight"),
            "letter_spacing": 0,
            "doory_font": {
                "family": node["fontFamily"],
                "weight": node.get("fontWeight", 400),
                "style": node.get("fontStyle", "normal"),
                "char_spacing_per_thousand_em": node.get("charSpacing", 0),
            },
        }
        # Preserve authored glyph size/baseline spacing. A final line's empty
        # leading is included in engine frame measurement, unlike Fabric's box.
        native_height = element["height"]
        element["height"] += max(0, element["line_height"] - FABRIC_LINE_HEIGHT_EM) * size
        padding = element["height"] - native_height
        angle = math.radians(element["rotation"])
        element["x"] -= math.sin(angle) * padding / 2
        element["y"] += (math.cos(angle) - 1) * padding / 2
        effects = node.get("effects") or {}
        shadow = effects.get("shadow") or node.get("shadow")
        if set(effects) - {"shadow", "outline"} or (
            shadow and (not isinstance(shadow, dict) or shadow.get("blur", 0) != 0)
        ):
            raise ImportBlock(
                "unsupported_text_effect",
                "Unsupported active text effect or blurred shadow",
                node["id"],
            )
        outline = effects.get("outline")
        if node.get("stroke") or outline:
            color = node.get("stroke") or outline.get("color")
            stroke = _number(
                node.get("strokeWidth", outline.get("width", 1) if outline else 1), "strokeWidth"
            )
            if outline and (
                outline.get("color") != color or abs(outline.get("width", stroke) - stroke) > 1e-6
            ):
                raise ImportBlock(
                    "unsupported_text_effect",
                    "Outline metadata and native stroke disagree",
                    node["id"],
                )
            solid_fill = (
                not isinstance(node.get("fill"), dict)
                and ImageColor.getcolor(node.get("fill") or "#00000000", "RGBA")[3] > 0
            )
            if node.get("paintFirst", "fill") == "stroke" and solid_fill:
                raise ImportBlock(
                    "unsupported_text_effect",
                    "Filled stroke-first text requires explicit paint ordering",
                    node["id"],
                )
            descriptor["outline"] = {
                "color": _rgba(color),
                "width_em": stroke * (1 if node.get("strokeUniform", False) else scale) / size,
                "fill": solid_fill or isinstance(node.get("fill"), dict),
            }
        if isinstance(node.get("fill"), dict):
            fill = node["fill"]
            _paint(fill)  # validates the exact supported percentage-gradient subset
            descriptor["gradient"] = {
                **fill["coords"],
                "stops": [
                    {"offset": stop["offset"], "color": _rgba(stop["color"])}
                    for stop in fill["colorStops"]
                ],
                "height_factor": native_height / element["height"],
            }
        if len(descriptor) > 1:
            from engine.native_text_effects import validate_descriptor

            try:
                validate_descriptor(descriptor, node["text"])
            except ValueError as exc:
                raise ImportBlock("unsupported_text_effect", str(exc), node["id"]) from exc
            element["native_text_effects"] = descriptor
        owner = group or (f"doory-shadow:{node['id']}" if shadow else None)
        if shadow:
            shade = deepcopy(element)
            shade.update(id=node["id"] + "--shadow", color=_rgba(shadow.get("color")))
            # A hard shadow is a solid copy of the same live glyphs.
            if "native_text_effects" in shade:
                shade["native_text_effects"].pop("gradient", None)
                shade["native_text_effects"].pop("outline", None)
            shade["x"] += _number(shadow.get("offsetX", 0), "shadow.offsetX") * scale
            shade["y"] += _number(shadow.get("offsetY", 0), "shadow.offsetY") * scale
            self._emit(
                shade,
                node,
                "editable_hard_shadow",
                world,
                owner,
                generated_role="shadow of the same native text; no new message",
            )
        self._emit(
            element,
            node,
            "editable_text",
            world,
            owner,
            native_frame_height=native_height,
            schema_frame_height=element["height"],
            font_historical_bytes_verified=False,
            live_text_effects=deepcopy(element.get("native_text_effects")),
        )
        self.text_count += 1

    def _photo_markup(self, node, world, parent):
        candidates = [
            asset
            for asset in self.entry.get("asset_requirements", [])
            if asset.get("source_reference") == node.get("src")
        ]
        if len(candidates) != 1:
            raise ImportBlock(
                "missing_image", "Photo requires one pinned local source asset", node["id"]
            )
        asset = candidates[0]
        path = _safe(self.root, asset["path"])
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        if digest != asset.get("sha256"):
            raise ImportBlock(
                "image_hash", "Photo bytes differ from the pinned original", node["id"]
            )
        retained = self.out / "original-resources" / (digest + path.suffix.lower())
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes(original)
        with Image.open(io.BytesIO(original)) as image:
            image.load()
            crop_x, crop_y = node.get("cropX", 0), node.get("cropY", 0)
            w, h = node["width"], node["height"]
            if crop_x < 0 or crop_y < 0 or crop_x + w > image.width or crop_y + h > image.height:
                raise ImportBlock("image_crop", "Photo crop leaves the original bitmap", node["id"])
            image = image.convert("RGBA").crop((crop_x, crop_y, crop_x + w, crop_y + h))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        href = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        bitmap = (
            f'<image x="{-w / 2:g}" y="{-h / 2:g}" width="{w:g}" height="{h:g}" '
            f'preserveAspectRatio="none" transform={quoteattr(_svg_matrix(world))} '
            f"xlink:href={quoteattr(href)}/>"
        )
        slot = node.get("slot")
        if not slot:
            if node.get("maskShape"):
                raise ImportBlock(
                    "unsupported_mask",
                    "Only the authored slot mask has a verified mapping",
                    node["id"],
                )
            return bitmap
        if parent != IDENTITY:
            raise ImportBlock(
                "unsupported_slot_group",
                "Absolute photo slots inside transformed groups need independent verification",
                node["id"],
            )
        sw, sh = (
            _number(slot.get("width"), "slot.width"),
            _number(slot.get("height"), "slot.height"),
        )
        if min(sw, sh) <= 0:
            raise ImportBlock("invalid_slot", "Slot dimensions must be positive", node["id"])
        mask = slot.get("mask")
        if mask == "circle":
            shape = f'<ellipse cx="0" cy="0" rx="{sw / 2:g}" ry="{sh / 2:g}"/>'
        elif mask is None or mask == "rounded":
            radius = min(sw, sh) * min(0.5, max(0, slot.get("radius", 0.18))) if mask else 0
            shape = (
                f'<rect x="{-sw / 2:g}" y="{-sh / 2:g}" width="{sw:g}" '
                f'height="{sh:g}" rx="{radius:g}"/>'
            )
        elif isinstance(mask, dict) and isinstance(mask.get("path"), str):
            data, (left, top, width, height) = _path(mask["path"])
            if min(width, height) <= 0:
                raise ImportBlock("invalid_mask", "Photo mask path has no area", node["id"])
            transform = (
                f"translate({-sw / 2:g} {-sh / 2:g}) "
                f"scale({sw / width:g} {sh / height:g}) translate({-left:g} {-top:g})"
            )
            shape = f"<path d={quoteattr(data)} transform={quoteattr(transform)}/>"
        else:
            raise ImportBlock("unsupported_mask", f"Unknown native slot mask {mask!r}", node["id"])
        cx, cy = _number(slot.get("cx"), "slot.cx"), _number(slot.get("cy"), "slot.cy")
        turned = _number(slot.get("angle", 0), "slot.angle")
        transform = f"translate({cx:g} {cy:g}) rotate({turned:g})"
        return (
            f'<defs><clipPath id="slot" clipPathUnits="userSpaceOnUse" '
            f"transform={quoteattr(transform)}>{shape}</clipPath></defs>"
            f'<g clip-path="url(#slot)">{bitmap}</g>'
        )

    def _art(self, node, world, group, parent):
        import resvg_py

        kind = node["type"].lower()
        if kind == "image":
            body = self._photo_markup(node, world, parent)
            mode = "isolated_masked_photo" if node.get("slot") else "isolated_photo"
        else:
            fill, defs = _paint(node.get("fill"))
            scale = math.hypot(world[0], world[1])
            stroke_width = _number(node.get("strokeWidth", 0), "strokeWidth")
            if node.get("strokeUniform", False):
                stroke_width /= scale
            dash = ""
            if node.get("strokeDashArray"):
                values = [_number(value, "stroke dash") for value in node["strokeDashArray"]]
                if any(value < 0 for value in values) or not any(values):
                    raise ImportBlock(
                        "invalid_dash", "Dash lengths must be nonnegative and nonempty", node["id"]
                    )
                divisor = scale if node.get("strokeUniform", False) else 1
                pattern = " ".join(f"{value / divisor:g}" for value in values)
                offset = _number(node.get("strokeDashOffset", 0), "strokeDashOffset") / divisor
                dash = f' stroke-dasharray="{pattern}" stroke-dashoffset="{offset:g}"'
            paint = (
                f"fill={quoteattr(fill)} stroke={quoteattr(_rgba(node.get('stroke')))} "
                f'stroke-width="{stroke_width:g}" '
                f"stroke-linecap={quoteattr(node.get('strokeLineCap', 'butt'))} "
                f"stroke-linejoin={quoteattr(node.get('strokeLineJoin', 'miter'))} "
                f"fill-rule={quoteattr(node.get('fillRule', 'nonzero'))}"
            ) + dash
            if kind == "path":
                data, (left, top, width, height) = _path(node.get("path"))
                offset = node.get("pathOffset", {"x": left + width / 2, "y": top + height / 2})
                shape = (
                    f"<path d={quoteattr(data)} "
                    f'transform="translate({-offset["x"]:g} {-offset["y"]:g})" {paint}/>'
                )
            elif kind == "rect":
                w, h = node["width"], node["height"]
                shape = (
                    f'<rect x="{-w / 2:g}" y="{-h / 2:g}" width="{w:g}" '
                    f'height="{h:g}" rx="{node.get("rx", 0):g}" '
                    f'ry="{node.get("ry", 0):g}" {paint}/>'
                )
            elif kind in {"ellipse", "circle"}:
                shape = f'<ellipse rx="{node["width"] / 2:g}" ry="{node["height"] / 2:g}" {paint}/>'
            else:
                raise ImportBlock(
                    "unsupported_art", f"Unknown non-text native type: {kind}", node["id"]
                )
            body = defs + f"<g transform={quoteattr(_svg_matrix(world))}>{shape}</g>"
            mode = "isolated_vector_art"
            self.art_count += 1
        opacity = node.get("opacity", 1) if node.get("visible", True) else 0
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}"><g opacity="{opacity:g}">{body}</g></svg>'
        )
        data = resvg_py.svg_to_bytes(
            svg_string=svg, width=self.width, height=self.height, skip_system_fonts=True
        )
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGBA")
            box = image.getchannel("A").getbbox()
            if box:
                image = image.crop(box)
            else:
                if kind == "image" and opacity > 0:
                    raise ImportBlock(
                        "empty_photo",
                        "A visible photo produced no pixels; its crop/mask must be inspected",
                        node["id"],
                    )
                box, image = (0, 0, 1, 1), Image.new("RGBA", (1, 1))
            filename = "doory-" + hashlib.sha256(node["id"].encode()).hexdigest()[:20] + ".png"
            image.save(self.out / "resources/assets" / filename)
        data = (self.out / "resources/assets" / filename).read_bytes()
        self.assets.append(
            {
                "asset_id": filename,
                "mime_type": "image/png",
                "data_base64": base64.b64encode(data).decode(),
            }
        )
        self._emit(
            {
                "type": "image",
                "x": box[0],
                "y": box[1],
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "asset_id": filename,
                "fit": "fill",
                "rotation": 0,
                "opacity": 1,
                "source_art": {
                    "kind": "masked-photo"
                    if mode == "isolated_masked_photo"
                    else "photo"
                    if mode == "isolated_photo"
                    else "svg-nontext",
                    "source_format": "doory-json",
                    "node_id": node["id"],
                    "original_sha256": self.entry["sha256"],
                    "native_type": node["type"],
                    "curve_editable": False,
                    "original_object_retained": True,
                },
            },
            node,
            mode,
            world,
            group,
            original_object_retained=True,
            curve_editable_in_engine=False,
            source_opacity_baked=True,
        )

    def _shape(self, node, world, group, parent):
        kind = node["type"].lower()
        if kind == "doryline":
            if (
                node.get("arrowStart", "none") != "none"
                or node.get("arrowEnd", "none") != "none"
                or node.get("strokeDashArray")
                or node.get("height", 0) != 0
            ):
                raise ImportBlock(
                    "unsupported_line",
                    "Only a plain horizontal-local DoryLine is supported",
                    node["id"],
                )
            scale = math.hypot(world[0], world[1])
            stroke = _number(node.get("strokeWidth", 1), "strokeWidth") * (
                1 if node.get("strokeUniform", False) else scale
            )
            if stroke <= 0:
                raise ImportBlock(
                    "unsupported_line", "DoryLine requires a visible stroke", node["id"]
                )
            rounded = node.get("strokeLineCap", "round") == "round"
            element = {
                **self._frame(
                    node, world, node["width"] * scale + (stroke if rounded else 0), stroke
                ),
                "type": "shape",
                "shape_type": "rectangle",
                "fill_color": _rgba(node.get("stroke")),
                "corner_radius": stroke / 2 if rounded else 0,
            }
            self._emit(element, node, "editable_line_capsule", world, group)
        elif (
            kind in {"rect", "ellipse", "circle"}
            and not isinstance(node.get("fill"), dict)
            and not node.get("stroke")
        ):
            rx, ry = node.get("rx", 0), node.get("ry", 0)
            if kind == "rect" and rx != ry:
                return self._art(node, world, group, parent)
            element = {
                **self._frame(node, world),
                "type": "shape",
                "shape_type": "rectangle" if kind == "rect" else "ellipse",
                "fill_color": _rgba(node.get("fill")),
            }
            if kind == "rect":
                element["corner_radius"] = rx * math.hypot(world[0], world[1])
            self._emit(element, node, "editable_shape", world, group)
        else:
            self._art(node, world, group, parent)

    def walk(self, nodes, parent=IDENTITY, group=None):
        if not isinstance(nodes, list):
            raise ImportBlock("invalid_tree", "Page objects must be a list")
        for node in nodes:
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("id"), str)
                or node["id"] in self.seen
            ):
                raise ImportBlock(
                    "invalid_identity", "Native object IDs must be present and unique"
                )
            self.seen.add(node["id"])
            if len(self.seen) > 200:
                raise ImportBlock(
                    "too_many_nodes", "The local engine supports at most 200 native leaves"
                )
            if (
                node.get("globalCompositeOperation", "source-over") != "source-over"
                or node.get("clipPath")
                or node.get("backgroundColor")
                or node.get("filters")
                or node.get("resizeFilter")
                or node.get("transformMatrix")
            ):
                raise ImportBlock(
                    "unsupported_compositing",
                    "Blends, arbitrary clips and object background paints require explicit support",
                    node["id"],
                )
            world = _mul(parent, _matrix(node))
            kind = str(node.get("type", "")).lower()
            if kind == "group":
                if node.get("opacity", 1) != 1 or node.get("shadow"):
                    raise ImportBlock(
                        "unsupported_group",
                        "Group opacity/shadows cannot be flattened onto separate children",
                        node["id"],
                    )
                self.walk(node.get("objects"), world, group or f"doory:{node['id']}")
            elif kind in {"textbox", "text", "i-text"}:
                self._text(node, world, group)
            else:
                if node.get("shadow") or node.get("effects"):
                    raise ImportBlock(
                        "unsupported_art_effect",
                        "Non-text effects need explicit artwork support",
                        node["id"],
                    )
                self._shape(node, world, group, parent)

    def source(self):
        background = self.page.get("background", {})
        fake = {"id": f"page-background:{self.page.get('id', self.page_index)}"}
        if background.get("type") == "solid":
            self._emit(
                {
                    "type": "shape",
                    "shape_type": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": self.width,
                    "height": self.height,
                    "fill_color": _rgba(background.get("color")),
                },
                fake,
                "editable_page_background",
                IDENTITY,
            )
        elif background.get("type") == "gradient" and background.get("kind") == "linear":
            # Directly follows the pinned author's buildFill percentage axes.
            angle = math.radians(_number(background.get("angle", 90), "background angle"))
            dx, dy = math.cos(angle) / 2, math.sin(angle) / 2
            native = {
                **fake,
                "type": "rect",
                "left": 0,
                "top": 0,
                "originX": "left",
                "originY": "top",
                "width": self.width,
                "height": self.height,
                "fill": {
                    "type": "linear",
                    "gradientUnits": "percentage",
                    "coords": {"x1": 0.5 - dx, "y1": 0.5 - dy, "x2": 0.5 + dx, "y2": 0.5 + dy},
                    "colorStops": sorted(
                        background.get("stops", []), key=lambda stop: stop["offset"]
                    ),
                },
            }
            self._art(native, _matrix(native), None, IDENTITY)
            self.mapping[-1]["mode"] = "isolated_gradient_page_background"
            self.mapping[-1]["original_page_background"] = deepcopy(background)
        else:
            raise ImportBlock("unsupported_background", "Unsupported page background paint")
        self.walk(self.page.get("objects"))
        return {"canvas": {"width": self.width, "height": self.height}, "elements": self.elements}


def _licenses(entry, root, out, faces):
    if (
        not entry.get("license_path")
        or not entry.get("notice_path")
        or not entry.get("attribution_text")
    ):
        raise ImportBlock(
            "missing_attribution",
            "Doory sources require their AAL license, NOTICE and author credit",
        )
    paths = [entry.get(key) for key in ("license_path", "notice_path", "third_party_path")]
    paths.extend(face["face"].get("license_path") for face in faces.values())
    result = []
    for index, value in enumerate(dict.fromkeys(path for path in paths if path)):
        source = _safe(root, value)
        name = f"{index}-{source.name}"
        destination = out / "licenses" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        result.append(
            {
                "path": f"licenses/{name}",
                "source_path": value,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "text": source.read_text(encoding="utf-8-sig"),
            }
        )
    return result


def _native_reference(entry, root, out, page_index):
    manifest_path = root / "originals/doory_reference/manifest.json"
    references = _read(manifest_path).get("references", []) if manifest_path.is_file() else []
    reference = entry.get("native_reference") or next(
        (r for r in references if r["id"] == entry["id"]), None
    )
    if not reference or page_index != reference.get("page_index", 0):
        return {
            "kind": "unavailable",
            "status": "pending",
            "reason": "No independent reference for this selected page",
        }
    path = _safe(root, reference["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ImportBlock("native_reference_hash", "Author thumbnail differs from its pinned hash")
    filename = "native-reference" + path.suffix.lower()
    (out / filename).write_bytes(data)
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGBA").save(out / "native.png")
        dimensions = list(image.size)
    return {
        **reference,
        "kind": "author_thumbnail",
        "status": "needs_review",
        "reference_file": filename,
        "dimensions": dimensions,
        "png_conversion": "Lossless decode of the original reference pixels; no enlargement",
        "independent_of_importer": True,
        "full_resolution": False,
    }


def prepare_doory(entry, corpus_root, outdir, page_index=None):
    root, out = Path(corpus_root).resolve(), Path(outdir).resolve()
    if out == root or out in root.parents:
        raise ValueError("Prepared output cannot overwrite the corpus root")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("source.json", "bundle.json", "imported.png", "native.png"):
        (out / name).unlink(missing_ok=True)
    for reference in out.glob("native-reference.*"):
        reference.unlink()
    report = {
        "status": "failed",
        "failures": [],
        "node_mapping": [],
        "editable_text_count": 0,
        "vector_art_count": 0,
        "original_filename": "original.json",
        "source_parity_status": "pending",
        "attribution": deepcopy(
            {
                key: entry.get(key)
                for key in (
                    "title",
                    "author",
                    "source_url",
                    "license_id",
                    "license_url",
                    "attribution_text",
                    "source_commit",
                )
            }
        ),
    }
    try:
        path = _safe(root, entry["original_path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ImportBlock("source_hash", "Original Doory JSON differs from the curated hash")
        (out / "original.json").write_bytes(raw)
        original = json.loads(raw)
        _json(out / "original-data.json", original)
        _licenses(entry, root, out, {})
        selected = entry.get("page_index", 0) if page_index is None else page_index
        importer = _Importer(entry, root, out, original, selected)
        report["native_reference"] = _native_reference(entry, root, out, selected)
        source = importer.source()
        manifest = [
            {k: face[k] for k in ("font_family", "file", "weight", "style")}
            for face in importer.fonts.values()
        ]
        _json(out / "resources/fonts/font-manifest.json", manifest)
        licenses = _licenses(entry, root, out, importer.fonts)
        attribution = {
            "text": entry.get("attribution_text"),
            "author": entry.get("author"),
            "source_url": entry.get("source_url"),
            "license_id": entry.get("license_id"),
            "license_url": entry.get("license_url"),
            "original_sha256": entry["sha256"],
            "license_paths": [item["path"] for item in licenses],
            "licenses": licenses,
        }
        bundle = {
            "task_id": entry["id"],
            "poster_id": entry["id"],
            "source": source,
            "target_canvas": {"width": 1080, "height": 1080},
            "attribution": attribution,
            "assets": importer.assets,
            "fonts": [
                {
                    "font_family": alias,
                    "postscript_name": alias,
                    "weight": "regular",
                    "style": "normal",
                    "data_base64": base64.b64encode(face["data"]).decode(),
                }
                for alias, face in importer.fonts.items()
            ],
            "document": {
                "snapshot_id": entry["sha256"],
                "page_id": str(importer.page.get("id", selected)),
                "artboard_id": str(importer.page.get("id", selected)),
                "source_format": "doory-json",
                "page_index": selected,
                "group_tree": importer.page["objects"],
                "original_document": original,
                "node_bindings": importer.mapping,
            },
        }
        from viewer.server import validate_request

        validate_request(bundle)
        _json(out / "source.json", source)
        _json(out / "bundle.json", bundle)
        report.update(
            status="ready",
            canvas=source["canvas"],
            node_mapping=importer.mapping,
            editable_text_count=importer.text_count,
            vector_art_count=importer.art_count,
            selected_page={
                "index": selected,
                "id": importer.page.get("id"),
                "name": importer.page.get("name"),
                "page_count": len(original["pages"]),
                "unselected_pages_retained_in_original": True,
            },
            font_provenance={
                "faces": [face["face"] for face in importer.fonts.values()],
                "historical_source_bytes_verified": False,
                "note": "Exact pinned named family/face preparation; "
                "upstream build font hashes are unavailable.",
            },
            attribution=attribution,
            original_tree_file="original-data.json",
            coordinate_policy="Serialized coordinates are pixels; "
            "the document unit field is retained as UI metadata",
            text_metrics_policy={
                "source_engine": "Fabric7.4.0 (original app package-lock)",
                "baseline_em": FABRIC_BASELINE_EM,
                "tracking_policy": "Native charSpacing / 1000 em at current font size; "
                "IR letter_spacing is zero to avoid double tracking",
                "line_height_em": FABRIC_LINE_HEIGHT_EM,
                "formula": "fontSize * 1.13 * (1 - 0.222)",
                "source_archive": "https://registry.npmjs.org/fabric/-/fabric-7.4.0.tgz",
                "source_methods": "Text.ts:_renderTextCommon,_renderChars,"
                "getHeightOfLineImpl; Text/constants.ts",
            },
            native_reference=report["native_reference"],
        )
    except ImportBlock as exc:
        report.update(status="unsupported", failures=[exc.record()])
    except Exception as exc:
        report.update(
            status="failed",
            failures=[{"code": "import_exception", "message": f"{type(exc).__name__}: {exc}"}],
        )
    _json(out / "import-report.json", report)
    return report


prepare_poster = prepare_doory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("fixtures/open_posters/catalogue.json")
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--page", type=int)
    args = parser.parse_args(argv)
    root = args.manifest.resolve().parent
    entries = _read(args.manifest)["examples"]
    selected = [
        entry
        for entry in entries
        if entry["original_path"].endswith(".json")
        and (not args.case_ids or entry["id"] in args.case_ids)
    ]
    if args.case_ids and set(args.case_ids) != {entry["id"] for entry in selected}:
        parser.error("Unknown Doory case ID")
    failed = False
    for entry in selected:
        report = prepare_doory(entry, root, _safe(root, entry["prepared_path"]), args.page)
        print(entry["id"], report["status"], report["failures"], flush=True)
        failed |= report["status"] != "ready"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
