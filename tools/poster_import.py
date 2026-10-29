"""Strict, disclosed import of licensed SVG originals into editable v0.9 sources.

Original bytes and native resvg pixels are retained. Text is never rasterized as
a fallback: supported runs use verified exact font files and remain text nodes.
Complex non-text artwork becomes separately disclosed rigid PNG assets, with
its original SVG and source tree retained. No whole-poster image fallback exists.

Development dependencies: resvg_py==0.5.0, fonttools==4.60.0.
API evidence: https://resvg-py.readthedocs.io/en/latest/api.html and
https://fonttools.readthedocs.io/en/latest/ttLib/ttFont.html.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageColor, ImageStat

SVG = "http://www.w3.org/2000/svg"
INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
KEY = "data-chameleon-import-key"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
DRAWABLE = {"rect", "circle", "ellipse", "line", "path", "polygon", "polyline", "image", "use"}
IGNORED = {"defs", "metadata", "title", "desc", "style", "namedview", "symbol"}
INHERITED = {
    "fill": "black",
    "fill-opacity": "1",
    "stroke": "none",
    "stroke-opacity": "1",
    "stroke-width": "1",
    "font-family": "serif",
    "font-size": "16",
    "font-weight": "400",
    "font-style": "normal",
    "text-anchor": "start",
    "letter-spacing": "0",
    "visibility": "visible",
    "white-space": "normal",
    "color": "black",
}
GENERIC = {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui"}


class ImportBlock(ValueError):
    def __init__(self, code: str, message: str, node_id: str | None = None):
        super().__init__(message)
        self.code, self.node_id = code, node_id

    def record(self) -> dict:
        return {"code": self.code, "node_id": self.node_id, "message": str(self)}


def _tag(node) -> str:
    return node.tag.rsplit("}", 1)[-1] if isinstance(node.tag, str) else ""


def _number(value: str | float | None, default=0.0) -> float:
    if value is None:
        return default
    match = re.fullmatch(r"\s*([-+\d.eE]+)(px|pt|pc|mm|cm|in)?\s*", str(value))
    if not match:
        raise ImportBlock("unsupported_length", f"Unsupported SVG length: {value!r}")
    number = (
        float(match[1])
        * {None: 1, "px": 1, "pt": 96 / 72, "pc": 16, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96}[
            match[2]
        ]
    )
    if not math.isfinite(number):
        raise ImportBlock("invalid_number", "SVG numbers must be finite.")
    return number


def _mul(a, b):
    return (
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    )


def _point(m, x, y):
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def parse_transform(value: str | None):
    result, consumed = IDENTITY, ""
    for match in re.finditer(r"([A-Za-z]+)\s*\(([^)]*)\)", value or ""):
        name = match[1]
        values = [float(part) for part in re.split(r"[\s,]+", match[2].strip()) if part]
        if not all(math.isfinite(v) for v in values):
            raise ImportBlock("invalid_transform", "Non-finite SVG transform.")
        if name == "matrix" and len(values) == 6:
            matrix = tuple(values)
        elif name == "translate" and len(values) in {1, 2}:
            matrix = (1, 0, 0, 1, values[0], values[1] if len(values) == 2 else 0)
        elif name == "scale" and len(values) in {1, 2}:
            matrix = (values[0], 0, 0, values[-1], 0, 0)
        elif name == "rotate" and len(values) in {1, 3}:
            angle = math.radians(values[0])
            matrix = (math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle), 0, 0)
            if len(values) == 3:
                x, y = values[1:]
                matrix = _mul((1, 0, 0, 1, x, y), _mul(matrix, (1, 0, 0, 1, -x, -y)))
        elif name in {"skewX", "skewY"} and len(values) == 1:
            tangent = math.tan(math.radians(values[0]))
            matrix = (1, 0, tangent, 1, 0, 0) if name == "skewX" else (1, tangent, 0, 1, 0, 0)
        else:
            raise ImportBlock("unsupported_transform", f"Unsupported transform {match[0]!r}")
        result = _mul(result, matrix)
        consumed += match[0]
    if re.sub(r"[\s,]", "", consumed) != re.sub(r"[\s,]", "", value or ""):
        raise ImportBlock("invalid_transform", f"Malformed SVG transform: {value!r}")
    return result


def _decls(value: str) -> dict:
    declarations = {}
    for item in value.split(";"):
        if item.strip():
            if ":" not in item or "!important" in item:
                raise ImportBlock(
                    "unsupported_css", "Malformed or !important CSS is not supported."
                )
            key, value = item.split(":", 1)
            declarations[key.strip()] = value.strip()
    return declarations


def _rgba(value: str, alpha=1.0) -> str:
    if value == "none":
        return "rgba(0,0,0,0)"
    try:
        red, green, blue, opacity = ImageColor.getcolor(value, "RGBA")
    except ValueError as exc:
        raise ImportBlock("unsupported_paint", f"Unsupported solid color: {value!r}") from exc
    return f"rgba({red},{green},{blue},{opacity / 255 * alpha:g})"


def _safe(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ImportBlock("path_escape", "Corpus resource path leaves its corpus root.")
    return path


def _licenses(entry, corpus, out):
    """Keep original source/font license evidence beside portable resources."""
    records = []
    declarations = [entry, *entry.get("font_requirements", [])]
    for declaration in declarations:
        relative = declaration.get("license_path")
        if not relative:
            continue
        original = _safe(corpus, relative)
        if not original.is_file():
            raise ImportBlock(
                "missing_license", f"Declared license evidence is missing: {relative}"
            )
        data = original.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target = Path("licenses") / (digest[:12] + "-" + original.name)
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        (out / target).write_bytes(data)
        record = {
            "path": target.as_posix(),
            "sha256": digest,
            "license_id": declaration.get("license_id"),
            "source_path": relative,
        }
        if record not in records:
            records.append(record)
    return records


@dataclass(frozen=True)
class FontFace:
    family: str
    postscript_name: str
    source_path: Path
    data: bytes
    weight: int
    italic: bool
    glyphs: frozenset[int]
    resolution: str
    requested_family: str


class _Importer:
    def __init__(self, entry, corpus_root, outdir, raw):
        self.entry, self.corpus, self.out = entry, corpus_root, outdir
        if b"<!ENTITY" in raw.upper() or re.search(rb"<!DOCTYPE[^>]*\[", raw, re.I):
            raise ImportBlock(
                "xml_entity", "Entity declarations and internal DTDs are unsupported."
            )
        self.root = ET.fromstring(raw)
        self.original_root = deepcopy(self.root)
        if _tag(self.root) != "svg":
            raise ImportBlock("not_svg", "The original is not an SVG document.")
        self.nodes, self.parents, self.styles, self.world = {}, {}, {}, {}
        self.text_runs, self.fonts, self.font_resolutions = [], {}, []
        self.elements, self.mapping, self.ignored, self.groups = [], [], [], []
        self.source_warnings = []
        self.assets, self.asset_payloads = [], []
        self.css = []
        self.generic_options = {}
        self._canvas()
        for node in self.root.iter():
            if _tag(node) == "style":
                css = re.sub(r"/\*.*?\*/", "", node.text or "", flags=re.S)
                for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
                    for selector in match[1].split(","):
                        selector = selector.strip()
                        if not re.fullmatch(r"[.#]?[\w-]+", selector):
                            raise ImportBlock(
                                "unsupported_css", f"Unsupported CSS selector: {selector}"
                            )
                        self.css.append((selector, _decls(match[2])))
                if "@" in css or re.sub(r"[^{}]+\{[^{}]*\}", "", css).strip():
                    raise ImportBlock("unsupported_css", "Unsupported stylesheet syntax.")
        self._index(self.root, None, INHERITED, self.viewport)
        self.original_root = deepcopy(self.root)
        self._expand_text(self.root)

    def _reference(self, value, key):
        match = re.fullmatch(r"url\(\s*['\"]?#([^'\")\s]+)['\"]?\s*\)", value)
        target_id = match[1] if match else value[1:] if value.startswith("#") else None
        target = next((node for node in self.nodes.values() if node.get("id") == target_id), None)
        if target is None:
            raise ImportBlock(
                "unsupported_reference", "Only existing internal SVG references are supported.", key
            )
        return target

    def _referenced_text(self, node, seen=()):
        key = node.get(KEY)
        if key in seen:
            raise ImportBlock("cyclic_reference", "Cyclic SVG references are unsupported.", key)
        if any(_tag(child) == "text" for child in node.iter()):
            return True
        for child in node.iter():
            if _tag(child) == "use":
                href = child.get("href", child.get("{http://www.w3.org/1999/xlink}href", ""))
                if self._referenced_text(self._reference(href, key), (*seen, key)):
                    return True
        return False

    def _has_clip(self, key):
        return any(
            self.nodes[k].get("clip-path", self.styles[k].get("clip-path", "none")) != "none"
            for k in self._ancestors(key)
        )

    def _canvas(self):
        view = [float(v) for v in re.split(r"[\s,]+", self.root.get("viewBox", "").strip()) if v]
        if view and (len(view) != 4 or min(view[2:]) <= 0):
            raise ImportBlock("invalid_viewbox", "Invalid SVG viewBox.")
        width, height = self.root.get("width"), self.root.get("height")
        # Percentage root dimensions require a host viewport; viewBox is the
        # explicitly recorded standalone viewport convention in that case.
        self.viewport_convention = "absolute SVG dimensions"
        if not width or not height or "%" in width + height:
            if not view:
                raise ImportBlock("missing_canvas", "Standalone SVG needs dimensions or a viewBox.")
            w, h = view[2:]
            self.viewport_convention = "viewBox dimensions for unspecified/percentage root viewport"
        else:
            w, h = _number(width), _number(height)
        self.width, self.height = round(w), round(h)
        if min(self.width, self.height) < 1 or self.width * self.height > 20_000_000:
            raise ImportBlock(
                "canvas_limit", "Native reference canvas must fit the 20 MP import limit."
            )
        self.viewport = IDENTITY
        if view:
            x, y, vw, vh = view
            aspect = self.root.get("preserveAspectRatio", "xMidYMid meet").split()
            if aspect == ["none"]:
                self.viewport = (
                    self.width / vw,
                    0,
                    0,
                    self.height / vh,
                    -x * self.width / vw,
                    -y * self.height / vh,
                )
            else:
                if aspect[0] not in {
                    "xMinYMin",
                    "xMidYMin",
                    "xMaxYMin",
                    "xMinYMid",
                    "xMidYMid",
                    "xMaxYMid",
                    "xMinYMax",
                    "xMidYMax",
                    "xMaxYMax",
                } or (len(aspect) > 1 and aspect[1] != "meet"):
                    raise ImportBlock(
                        "unsupported_viewport", "Only meet/none root viewBox mapping is supported."
                    )
                scale = min(self.width / vw, self.height / vh)
                ax = {"Min": 0, "Mid": 0.5, "Max": 1}[aspect[0][1:4]]
                ay = {"Min": 0, "Mid": 0.5, "Max": 1}[aspect[0][5:8]]
                self.viewport = (
                    scale,
                    0,
                    0,
                    scale,
                    (self.width - vw * scale) * ax - x * scale,
                    (self.height - vh * scale) * ay - y * scale,
                )

    def _index(self, node, parent, inherited, matrix):
        key = f"n{len(self.nodes):05d}"
        node.set(KEY, key)
        self.nodes[key], self.parents[key] = node, parent
        style = dict(inherited)
        style.update(
            {k: v for k, v in node.attrib.items() if k in INHERITED or k in {"opacity", "display"}}
        )
        ordered_css = sorted(
            enumerate(self.css),
            key=lambda item: (
                100 if item[1][0].startswith("#") else 10 if item[1][0].startswith(".") else 1,
                item[0],
            ),
        )
        for _, (selector, declarations) in ordered_css:
            if (
                selector == _tag(node)
                or selector == "#" + node.get("id", "")
                or (selector.startswith(".") and selector[1:] in node.get("class", "").split())
            ):
                style.update(declarations)
        style.update(_decls(node.get("style", "")))
        font_size = style.get("font-size", "16")
        if font_size.endswith("em"):
            font_size = str(float(font_size[:-2]) * float(inherited.get("font-size", 16)))
        style["font-size"] = str(_number(font_size))
        style.setdefault("opacity", "1")
        # Opacity and display do not inherit like fill/font properties.
        self.styles[key] = style
        self.world[key] = _mul(matrix, parse_transform(node.get("transform")))
        for child in node:
            child_style = {k: v for k, v in style.items() if k not in {"opacity", "display"}}
            self._index(child, key, child_style, self.world[key])

    def _expand_text(self, node, text_owner=None):
        if _tag(node) == "text":
            text_owner = node.get(KEY)
        positions = {
            name: re.split(r"[\s,]+", node.get(name, "").strip())
            for name in ("x", "y", "dx", "dy", "rotate")
            if node.get(name, "").strip()
        }
        if (
            text_owner
            and not len(node)
            and node.text
            and any(len(values) > 1 for values in positions.values())
        ):
            # An authored per-character location is representable as independent
            # live text, with complete source-object lineage retained. Do not
            # synthesize positions or convert the text into artwork.
            if any(_number(value) != 0 for value in positions.get("rotate", [])):
                raise ImportBlock(
                    "per_character_rotation",
                    "Rotated character lists need editable rotation support.",
                )
            raw = node.text
            node.text = None
            for name in positions:
                node.attrib.pop(name, None)
            for index, character in enumerate(raw):
                key = f"run{len(self.text_runs):05d}"
                wrapper = ET.SubElement(node, f"{{{SVG}}}tspan", {KEY: key})
                wrapper.text = character
                for name, values in positions.items():
                    if index < len(values):
                        wrapper.set(name, values[index])
                parent = node.get(KEY)
                self.nodes[key], self.parents[key] = wrapper, parent
                self.styles[key], self.world[key] = dict(self.styles[parent]), self.world[parent]
                self.text_runs.append(
                    {
                        "key": key,
                        "owner": text_owner,
                        "raw": character,
                        "content": character,
                        "offset": index,
                        "source_complete_text": raw,
                        "segmentation": "authored_character_positions",
                    }
                )
            return
        if text_owner:
            original_children = list(node)
            segments = [(None, node.text)] + [(child, child.tail) for child in original_children]
            node.text = None
            for child in original_children:
                child.tail = None
            offset = 0
            for after, value in segments:
                if value:
                    wrapper = ET.Element(f"{{{SVG}}}tspan", {KEY: f"run{len(self.text_runs):05d}"})
                    wrapper.text = value
                    index = 0 if after is None else list(node).index(after) + 1
                    node.insert(index, wrapper)
                    key = wrapper.get(KEY)
                    parent = node.get(KEY)
                    self.nodes[key], self.parents[key] = wrapper, parent
                    self.styles[key], self.world[key] = (
                        dict(self.styles[parent]),
                        self.world[parent],
                    )
                    wrapper.set(
                        "style",
                        ";".join(
                            f"{name}:{self.styles[parent][name]}"
                            for name in (
                                "fill",
                                "fill-opacity",
                                "stroke",
                                "stroke-opacity",
                                "font-family",
                                "font-size",
                                "font-weight",
                                "font-style",
                                "text-anchor",
                                "letter-spacing",
                                "color",
                            )
                        ),
                    )
                    normalized = re.sub(r"[\t\r\n]", " ", value)
                    preserve = any(
                        self.nodes[k].get("{http://www.w3.org/XML/1998/namespace}space")
                        == "preserve"
                        for k in self._ancestors(parent)
                    )
                    if not preserve and self.styles[key].get("white-space") != "pre":
                        normalized = re.sub(r" +", " ", normalized)
                    self.text_runs.append(
                        {
                            "key": key,
                            "owner": text_owner,
                            "raw": value,
                            "content": normalized,
                            "offset": offset,
                        }
                    )
                    offset += len(value)
        for child in list(node):
            if not child.get(KEY, "").startswith("run"):
                self._expand_text(child, text_owner)

    def _ancestors(self, key):
        while key is not None:
            yield key
            key = self.parents[key]

    def _visible(self, key):
        return all(
            _tag(self.nodes[k]) not in IGNORED
            and self.styles[k].get("display") != "none"
            and self.styles[k].get("visibility") != "hidden"
            for k in self._ancestors(key)
        )

    def audit(self):
        issues = []
        ids = [n.get("id") for n in self.root.iter() if n.get("id")]
        if len(ids) != len(set(ids)):
            issues.append(
                ImportBlock(
                    "duplicate_svg_id", "Duplicate original SVG IDs are unsupported."
                ).record()
            )
        for key, node in self.nodes.items():
            if not self._visible(key):
                continue
            style = self.styles[key]
            if _tag(node) not in DRAWABLE | IGNORED | {"svg", "g", "text", "tspan"}:
                issues.append(
                    ImportBlock(
                        "unsupported_feature",
                        f"Unmapped SVG element {_tag(node)} is not silently dropped.",
                        key,
                    ).record()
                )
            if (
                style.get("mix-blend-mode", "normal") != "normal"
                or style.get("transform", "none") != "none"
            ):
                issues.append(
                    ImportBlock(
                        "unsupported_css_compositing",
                        "CSS blending/transforms need explicit support.",
                        key,
                    ).record()
                )
            if _tag(node) == "use":
                try:
                    href = node.get("href", node.get("{http://www.w3.org/1999/xlink}href", ""))
                    if self._referenced_text(self._reference(href, key)):
                        raise ImportBlock(
                            "referenced_text",
                            "Text-bearing use instances need editable expansion.",
                            key,
                        )
                except ImportBlock as exc:
                    issues.append(exc.record())
            if _tag(node) in {
                "script",
                "foreignObject",
                "textPath",
                "switch",
                "animate",
                "animateTransform",
            }:
                issues.append(
                    ImportBlock(
                        "unsupported_feature", f"Unsupported {_tag(node)} element.", key
                    ).record()
                )
            clip = node.get("clip-path", style.get("clip-path", "none"))
            if clip != "none":
                try:
                    if _tag(self._reference(clip, key)) != "clipPath":
                        raise ImportBlock(
                            "unsupported_clip",
                            "Clip references must name an internal clipPath.",
                            key,
                        )
                except ImportBlock as exc:
                    issues.append(exc.record())
            for attr in ("filter", "mask"):
                if node.get(attr, style.get(attr, "none")) != "none":
                    issues.append(
                        ImportBlock(
                            "unsupported_feature",
                            f"Unsupported {attr}; no approximation was imported.",
                            key,
                        ).record()
                    )
            if _tag(node) == "image":
                href = node.get("href", node.get("{http://www.w3.org/1999/xlink}href", ""))
                if not href.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                    issues.append(
                        ImportBlock(
                            "external_image",
                            "Only embedded PNG/JPEG source images are supported.",
                            key,
                        ).record()
                    )
            if (
                _tag(node) in {"svg", "g"}
                and float(style.get("opacity", 1)) != 1
                and any(_tag(n) == "text" for n in node.iter())
            ):
                issues.append(
                    ImportBlock(
                        "group_compositing",
                        "Nontrivial group opacity containing editable text is unsupported.",
                        key,
                    ).record()
                )
        for run in self.text_runs:
            if not run["content"].strip() or not self._visible(run["key"]):
                continue
            try:
                self._text_checks(run)
                run["font"] = self._font(self.styles[run["key"]], run["content"])
            except ImportBlock as exc:
                exc.node_id = run["key"]
                issues.append(exc.record())
        if not issues:
            for run in self.text_runs:
                if "font" not in run or not self._has_clip(run["key"]):
                    continue
                clipped = self.isolated(run["key"], text=True)
                unclipped = self.isolated(run["key"], text=True, remove_clips=True)
                if ImageChops.difference(clipped, unclipped).getbbox():
                    issues.append(
                        ImportBlock(
                            "clipped_text",
                            "Source clipping changes text pixels; "
                            "editable text cannot silently lose that clip.",
                            run["key"],
                        ).record()
                    )
        return issues

    def _text_checks(self, run):
        key = run["key"]
        style = self.styles[key]
        for ancestor in self._ancestors(key):
            node = self.nodes[ancestor]
            if _tag(node) in {"text", "tspan"}:
                for attr in ("x", "y", "dx", "dy", "rotate"):
                    if len(re.split(r"[\s,]+", node.get(attr, "").strip())) > 1:
                        raise ImportBlock(
                            "per_character_position",
                            "Per-character position lists are not editable in v0.9.",
                        )
                if any(node.get(attr) is not None for attr in ("textLength", "lengthAdjust")):
                    raise ImportBlock(
                        "text_length", "SVG textLength/lengthAdjust cannot be preserved."
                    )
        for attr, neutral in (
            ("writing-mode", {"horizontal-tb", "lr", "lr-tb", "normal"}),
            ("text-decoration", {"none"}),
            ("font-variant", {"normal"}),
            ("font-feature-settings", {"normal"}),
            ("text-transform", {"none"}),
        ):
            if attr in style and style[attr] not in neutral:
                raise ImportBlock(
                    "unsupported_text_style", f"Unsupported text style {attr}: {style[attr]}"
                )
        if style.get("stroke", "none") != "none" or "url(" in style["fill"]:
            raise ImportBlock(
                "unsupported_text_paint", "Stroked or gradient text cannot remain editable in v0.9."
            )
        m = self.world[key]
        sx, sy = math.hypot(m[0], m[1]), math.hypot(m[2], m[3])
        if (
            sx <= 0
            or abs(sx - sy) > 1e-6 * max(sx, sy)
            or abs(m[0] * m[2] + m[1] * m[3]) > 1e-6
            or m[0] * m[3] - m[1] * m[2] <= 0
        ):
            raise ImportBlock(
                "unsupported_text_transform",
                "Skewed, reflected or nonuniform text cannot remain editable.",
            )

    def _font(self, style, content):
        from fontTools.ttLib import TTFont

        families = [v.strip().strip("\"'") for v in style["font-family"].split(",")]
        weight = {"normal": 400, "bold": 700}.get(style["font-weight"], style["font-weight"])
        try:
            weight = int(weight)
        except ValueError as exc:
            raise ImportBlock(
                "unsupported_font_weight", "Relative/unknown font weights need an explicit face."
            ) from exc
        italic = style.get("font-style", "normal") == "italic"
        if style.get("font-style", "normal") not in {"normal", "italic"}:
            raise ImportBlock(
                "unsupported_font_style", "Synthetic oblique text cannot be preserved."
            )
        requirements = self.entry.get("font_requirements", [])
        for family in families:
            for spec in requirements:
                if (
                    not isinstance(spec, dict)
                    or spec.get("family", spec.get("declared_family")) != family
                ):
                    continue
                sw = {"normal": 400, "regular": 400, "bold": 700}.get(
                    str(spec.get("weight", 400)), spec.get("weight", 400)
                )
                if int(sw) != weight or spec.get("style", "normal") != (
                    "italic" if italic else "normal"
                ):
                    continue
                path = _safe(self.corpus, spec.get("path", spec.get("file", "")))
                if not path.is_file() or path.suffix.lower() not in {".ttf", ".otf"}:
                    raise ImportBlock(
                        "missing_font", f"Exact original font file is missing: {path}"
                    )
                data = path.read_bytes()
                with TTFont(io.BytesIO(data), lazy=False) as font:
                    if "fvar" in font:
                        raise ImportBlock(
                            "variable_font",
                            "Supply the exact static font face; "
                            "variable-font instancing is not implicit.",
                        )
                    names = {n.toUnicode() for n in font["name"].names if n.nameID in {1, 16}}
                    ps = {n.toUnicode() for n in font["name"].names if n.nameID == 6}
                    postscript = spec.get("postscript_name")
                    if len(ps) != 1 or postscript not in ps:
                        raise ImportBlock(
                            "font_face_mismatch",
                            "Font bytes do not match the declared PostScript name.",
                        )
                    actual_weight = font["OS/2"].usWeightClass
                    actual_italic = bool(font["head"].macStyle & 2)
                    if actual_weight != weight or actual_italic != italic:
                        raise ImportBlock(
                            "font_face_mismatch",
                            "Font file weight/style does not match the SVG face.",
                        )
                    resolution = spec.get("resolution", "exact")
                    if family in GENERIC:
                        if (
                            resolution != "explicit_generic_policy"
                            or spec.get("resolved_family") not in names
                        ):
                            raise ImportBlock(
                                "generic_font_policy",
                                "A generic SVG family needs an explicit recorded resolution.",
                            )
                        resolved = spec["resolved_family"]
                        option = {
                            "sans-serif": "sans_serif_family",
                            "system-ui": "sans_serif_family",
                        }.get(family, family + "_family")
                        self.generic_options[option] = resolved
                    elif resolution not in {"exact", "exact_named_family_weight_style"}:
                        raise ImportBlock(
                            "font_family_mismatch",
                            "Named SVG families must match the font's actual family; "
                            "no substitute is allowed.",
                        )
                    else:
                        resolved = family
                        if family not in names:
                            provider_family = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")
                            evidence = spec.get("source_url", "")
                            if (
                                spec.get("source_family_alias") != family
                                or f"/@fontsource/{provider_family}@" not in evidence
                            ):
                                raise ImportBlock(
                                    "font_family_mismatch",
                                    "Named families need exact name-table "
                                    "or explicit provider alias evidence.",
                                )
                            resolved = sorted(names)[0]
                            resolution = "provider_evidenced_family_alias"
                    glyphs = frozenset((font.getBestCmap() or {}).keys())
                if any(ord(c) not in glyphs for c in content if not c.isspace()):
                    raise ImportBlock(
                        "missing_glyph", f"Exact face {postscript} lacks source characters."
                    )
                face = FontFace(
                    resolved, postscript, path, data, weight, italic, glyphs, resolution, family
                )
                self.fonts[postscript] = face
                record = {
                    "declared_families": families,
                    "selected_family": family,
                    "resolved_family": resolved,
                    "postscript_name": postscript,
                    "weight": weight,
                    "style": "italic" if italic else "normal",
                    "resolution": resolution,
                    "font_sha256": hashlib.sha256(data).hexdigest(),
                    "historical_source_bytes_verified": spec.get(
                        "historical_source_bytes_verified", False
                    ),
                }
                if record not in self.font_resolutions:
                    self.font_resolutions.append(record)
                return face
        raise ImportBlock(
            "missing_font",
            f"No exact supplied font face for {families}, weight {weight}, "
            f"style {style['font-style']}.",
        )

    def render(self, root=None, output_size=None):
        try:
            import resvg_py
        except ImportError as exc:
            raise ImportBlock(
                "missing_renderer", "Install development dependency resvg_py==0.5.0."
            ) from exc
        rendered_root = deepcopy(self.original_root if root is None else root)
        by_key = {node.get(KEY): node for node in rendered_root.iter()}
        for run in self.text_runs:
            face = run.get("font")
            if face is None or face.resolution != "provider_evidenced_family_alias":
                continue
            node = by_key.get(run["key"])
            if node is None:
                node = by_key.get(self.parents[run["key"]])
            if node is not None:
                node.set("style", node.get("style", "") + f";font-family:'{face.family}'")
        data = ET.tostring(rendered_root, encoding="unicode")
        png = resvg_py.svg_to_bytes(
            svg_string=data,
            width=self.width if output_size is None else output_size[0],
            height=self.height if output_size is None else output_size[1],
            dpi=96,
            skip_system_fonts=True,
            font_files=[str(f.source_path) for f in self.fonts.values()],
            **self.generic_options,
        )
        return Image.open(io.BytesIO(png)).convert("RGBA")

    def isolated(self, key, text=False, remove_clips=False, padding=0):
        clone = deepcopy(self.root)
        keep = set(self._ancestors(key))
        keep.update(n.get(KEY) for n in self.nodes[key].iter())
        owner = (
            next((r["owner"] for r in self.text_runs if r["key"] == key), None) if text else None
        )
        if owner:
            keep.update(n.get(KEY) for n in self.nodes[owner].iter())

        def prune(node):
            for child in list(node):
                child_key = child.get(KEY)
                if _tag(child) in IGNORED:
                    continue
                if child_key not in keep and node.get(KEY) != key:
                    node.remove(child)
                else:
                    prune(child)
            if text and node.get(KEY, "").startswith("run") and node.get(KEY) != key:
                node.set(
                    "style",
                    ";".join(filter(None, (node.get("style", ""), "fill:none", "stroke:none"))),
                )

        prune(clone)
        present_ids = {n.get("id") for n in clone.iter() if n.get("id")}
        definitions = next((n for n in clone if _tag(n) == "defs"), None)
        if definitions is None:
            definitions = ET.SubElement(clone, f"{{{SVG}}}defs")
        for node in list(clone.iter()):
            if _tag(node) == "use":
                href = node.get("href", node.get("{http://www.w3.org/1999/xlink}href", ""))
                target = self._reference(href, key)
                if target.get("id") not in present_ids:
                    definitions.append(deepcopy(target))
                    present_ids.update(n.get("id") for n in target.iter() if n.get("id"))
            if remove_clips:
                node.attrib.pop("clip-path", None)
                declarations = _decls(node.get("style", ""))
                declarations.pop("clip-path", None)
                node.set("style", ";".join(f"{k}:{v}" for k, v in declarations.items()))
        if padding:
            # Measure the complete original glyph shape across the root viewport
            # edge, then retain its original position so the engine canvas clips
            # it identically. Explicit authored clip paths stay in the subtree.
            left, top, right, bottom = padding
            size = self.width + left + right, self.height + top + bottom
            sx, _, _, sy, tx, ty = self.viewport
            clone.set("width", str(size[0]))
            clone.set("height", str(size[1]))
            clone.set("preserveAspectRatio", "none")
            clone.set(
                "viewBox", (f"{-(tx + left) / sx} {-(ty + top) / sy} {size[0] / sx} {size[1] / sy}")
            )
            return self.render(clone, size)
        return self.render(clone)

    def _emit(self, element, key, mode, bounds, **extra):
        element["id"] = f"svg-{key}"
        element["z_index"] = len(self.elements)
        self.elements.append(element)
        self.mapping.append(
            {
                "node_id": self.nodes[key].get("id"),
                "source_key": key,
                "element_id": element["id"],
                "mode": mode,
                "reference_bounds": list(bounds),
                "world_transform": list(self.world[key]),
                **extra,
            }
        )

    def _art(self, key):
        image = self.isolated(key)
        bounds = image.getchannel("A").getbbox()
        if not bounds:
            self.ignored.append({"source_key": key, "reason": "no painted pixels"})
            return
        cropped = image.crop(bounds)
        filename = f"svg-{key}.png"
        cropped.save(self.out / "resources/assets" / filename)
        data = (self.out / "resources/assets" / filename).read_bytes()
        self.asset_payloads.append(
            {
                "asset_id": filename,
                "mime_type": "image/png",
                "data_base64": base64.b64encode(data).decode(),
            }
        )
        self._emit(
            {
                "type": "image",
                "x": bounds[0],
                "y": bounds[1],
                "width": bounds[2] - bounds[0],
                "height": bounds[3] - bounds[1],
                "asset_id": filename,
                "fit": "fill",
                "rotation": 0,
                "opacity": 1,
                "source_art": {
                    "kind": "svg-nontext",
                    "source_key": key,
                    "source_node_id": self.nodes[key].get("id"),
                    "reference_bounds": list(bounds),
                    "curve_editable": False,
                },
            },
            key,
            "rigid_vector_art",
            bounds,
            curve_editable_in_engine=False,
            retained_original="original.svg",
            descendant_source_keys=[n.get(KEY) for n in self.nodes[key].iter()],
        )

    def _shape(self, key):
        node, style, m = self.nodes[key], self.styles[key], self.world[key]
        kind = _tag(node)
        if self._has_clip(key):
            return False
        if (
            kind not in {"rect", "circle", "ellipse"}
            or style.get("stroke", "none") != "none"
            or "url(" in style["fill"]
        ):
            return False
        sx, sy = math.hypot(m[0], m[1]), math.hypot(m[2], m[3])
        if min(sx, sy) <= 0 or abs(m[0] * m[2] + m[1] * m[3]) > 1e-6:
            return False
        if kind == "rect":
            x, y, w, h = (_number(node.get(k)) for k in ("x", "y", "width", "height"))
            rx = _number(node.get("rx", node.get("ry", "0"))) * sx
            ry = _number(node.get("ry", node.get("rx", "0"))) * sy
            if abs(rx - ry) > 1e-6:
                return False
        else:
            cx, cy = _number(node.get("cx")), _number(node.get("cy"))
            rx, ry = (
                (_number(node.get("r")),) * 2
                if kind == "circle"
                else (_number(node.get("rx")), _number(node.get("ry")))
            )
            x, y, w, h = cx - rx, cy - ry, rx * 2, ry * 2
            rx = 0
        if min(w, h) <= 0:
            self.ignored.append({"source_key": key, "reason": "zero-area SVG shape"})
            return True
        cx, cy = _point(m, x + w / 2, y + h / 2)
        fill = style["color"] if style["fill"] == "currentColor" else style["fill"]
        self._emit(
            {
                "type": "shape",
                "shape_type": "rectangle" if kind == "rect" else "ellipse",
                "x": cx - w * sx / 2,
                "y": cy - h * sy / 2,
                "width": w * sx,
                "height": h * sy,
                "rotation": math.degrees(math.atan2(m[1], m[0])),
                "opacity": float(style.get("opacity", 1)),
                "fill_color": _rgba(fill, float(style.get("fill-opacity", 1))),
                "corner_radius": rx,
            },
            key,
            "native_shape",
            (cx - w * sx / 2, cy - h * sy / 2, cx + w * sx / 2, cy + h * sy / 2),
        )
        return True

    def _text(self, run):
        from engine.eval.render import render_image
        from engine.layout.text import get_measurer

        key, face = run["key"], run["font"]
        style, m = self.styles[key], self.world[key]
        size = float(style["font-size"]) * math.hypot(m[0], m[1])
        native = self.isolated(key, text=True)
        bounds = native.getchannel("A").getbbox()
        if not bounds:
            self.ignored.append(
                {"source_key": key, "reason": "text run paints no pixels", "raw_text": run["raw"]}
            )
            return
        if bounds[0] <= 0 or bounds[1] <= 0 or bounds[2] >= self.width or bounds[3] >= self.height:
            measured_span = (
                get_measurer()
                .measure(
                    {
                        "content": run["content"],
                        "font_family": face.postscript_name,
                    },
                    size,
                    1e9,
                )
                .width
            )
            margin = max(32, math.ceil(max(size * 2, measured_span + size)))
            padding = (
                margin if bounds[0] <= 0 else 0,
                margin if bounds[1] <= 0 else 0,
                margin if bounds[2] >= self.width else 0,
                margin if bounds[3] >= self.height else 0,
            )
            expanded_image = self.isolated(key, text=True, padding=padding)
            expanded = expanded_image.getchannel("A").getbbox()
            if (
                not expanded
                or expanded[0] == 0
                or expanded[1] == 0
                or (expanded[2] >= expanded_image.width or expanded[3] >= expanded_image.height)
            ):
                raise ImportBlock(
                    "clipped_text", "Complete native glyph bounds exceed measurement extent.", key
                )
            bounds = tuple(value - padding[i % 2] for i, value in enumerate(expanded))
            self.source_warnings.append(
                {
                    "code": "root_viewport_clipped_text",
                    "source_key": key,
                    "message": (
                        f"Original author viewport {self.width}x{self.height} clips text with "
                        f"full native ink bounds {bounds}. Original cropping and the complete "
                        "editable text frame are retained; this is a source stress case."
                    ),
                    "full_ink_bounds": list(bounds),
                    "source_complete_text": run.get("source_complete_text", run["content"]),
                }
            )
        spacing = _number(
            style.get("letter-spacing", "0") if style.get("letter-spacing") != "normal" else "0"
        ) * math.hypot(m[0], m[1])
        align = {"start": "left", "middle": "center", "end": "right"}.get(style["text-anchor"])
        if align is None or size < 1:
            raise ImportBlock(
                "unsupported_text_metrics", "Unsupported text anchor or subpixel font size.", key
            )
        element = {
            "type": "text",
            "x": 0,
            "y": 0,
            "width": max(10, len(run["content"]) * size * 2),
            "height": size * 2,
            "content": run["content"],
            "font_family": face.postscript_name,
            "font_size": size,
            "font_weight": "regular",
            "font_style": "normal",
            "line_height": 1.0,
            "letter_spacing": spacing,
            "text_align": align,
            "color": "rgba(0,0,0,1)",
            "max_lines": 1,
            "rotation": math.degrees(math.atan2(m[1], m[0])),
            "opacity": 1,
        }
        measured = get_measurer().measure(element, size, element["width"])
        element["width"] = measured.width + 1
        angle = math.radians(element["rotation"])
        rw = abs(element["width"] * math.cos(angle)) + abs(element["height"] * math.sin(angle))
        rh = abs(element["width"] * math.sin(angle)) + abs(element["height"] * math.cos(angle))
        margin = size * 2
        element.update(
            x=(rw - element["width"]) / 2 + margin, y=(rh - element["height"]) / 2 + margin
        )
        canvas = {"width": math.ceil(rw + margin * 2), "height": math.ceil(rh + margin * 2)}
        prototype = {"canvas": canvas, "elements": [{**element, "id": "prototype"}]}
        image = render_image(prototype, max_size=max(canvas.values())).convert("RGB")
        ink = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        if not ink:
            raise ImportBlock(
                "text_render_failed", "The exact engine face rendered no glyphs.", key
            )
        if any(
            abs((ink[i + 2] - ink[i]) - (bounds[i + 2] - bounds[i]))
            > max(3, (bounds[i + 2] - bounds[i]) * 0.04)
            for i in (0, 1)
        ):
            raise ImportBlock(
                "text_shape_mismatch",
                f"Exact native/engine glyph extents differ: native={bounds}, engine={ink}, "
                f"font_size={size:.6g}, face={face.postscript_name}; "
                "no font scaling or substitution was applied.",
                key,
            )
        element["x"] += bounds[0] - ink[0]
        element["y"] += bounds[1] - ink[1]
        fill = style["color"] if style["fill"] == "currentColor" else style["fill"]
        element["color"] = _rgba(fill, float(style.get("fill-opacity", 1)))
        element["opacity"] = float(style.get("opacity", 1))
        element.update(
            svg_font_family=style["font-family"],
            svg_font_weight=style["font-weight"],
            svg_font_style=style["font-style"],
        )
        self._emit(
            element,
            key,
            "editable_text",
            bounds,
            raw_text=run["raw"],
            source_text_owner=run["owner"],
            postscript_name=face.postscript_name,
            font_resolution=face.resolution,
            source_complete_text=run.get("source_complete_text"),
            segmentation=run.get("segmentation"),
        )

    def convert(self):
        runs = {run["key"]: run for run in self.text_runs}
        drawable_count = sum(_tag(n) in DRAWABLE or _tag(n) == "text" for n in self.root.iter())

        def visit(node, in_defs=False):
            key, kind = node.get(KEY), _tag(node)
            if kind in IGNORED or in_defs or not self._visible(key):
                return
            if key in runs:
                if runs[key]["content"].strip():
                    self._text(runs[key])
                return
            if kind in {"svg", "g", "text", "tspan"}:
                leaves = sum(_tag(n) in DRAWABLE or _tag(n) == "text" for n in node.iter())
                layer = node.get(f"{{{INKSCAPE}}}groupmode") == "layer"
                text = any(_tag(n) == "text" for n in node.iter())
                if (
                    kind == "g"
                    and not text
                    and not layer
                    and leaves > 1
                    and leaves < drawable_count
                ):
                    self._art(key)
                else:
                    for child in node:
                        visit(child)
                if kind == "g":
                    self.groups.append(
                        {
                            "source_key": key,
                            "id": node.get("id"),
                            "layer": layer,
                            "whole_document": leaves == drawable_count,
                            "descendant_keys": [n.get(KEY) for n in node.iter()],
                        }
                    )
                return
            if kind in DRAWABLE:
                if not self._shape(key):
                    self._art(key)
                return
            raise ImportBlock(
                "unsupported_element", f"Unsupported rendered SVG element: {kind}", key
            )

        visit(self.root)
        if not self.elements:
            raise ImportBlock("empty_source", "The SVG has no imported painted elements.")
        if len(self.elements) == 1 and self.elements[0]["type"] == "image":
            raise ImportBlock(
                "whole_poster_raster", "A whole-poster raster fallback is not an editable import."
            )
        mappings = {m["element_id"]: m for m in self.mapping}
        for element in self.elements:
            key = mappings[element["id"]]["source_key"]
            eligible = []
            for group in self.groups:
                if key not in group["descendant_keys"] or group["layer"] or group["whole_document"]:
                    continue
                members = [m for m in self.mapping if m["source_key"] in group["descendant_keys"]]
                boxes = [m["reference_bounds"] for m in members]
                if len(boxes) < 2:
                    continue
                width = max(b[2] for b in boxes) - min(b[0] for b in boxes)
                height = max(b[3] for b in boxes) - min(b[1] for b in boxes)
                if (
                    width <= self.width * 0.9
                    and height <= self.height * 0.6
                    and width * height <= self.width * self.height * 0.35
                ):
                    eligible.append(group)
            if eligible:
                outer = max(eligible, key=lambda g: len(g["descendant_keys"]))
                element["group_id"] = "svg-group-" + outer["source_key"]
            elif owner := mappings[element["id"]].get("source_text_owner"):
                if sum(m.get("source_text_owner") == owner for m in self.mapping) > 1:
                    element["group_id"] = "svg-text-" + owner
        return {"canvas": {"width": self.width, "height": self.height}, "elements": self.elements}


def prepare_poster(entry: dict, corpus_root: str | Path, outdir: str | Path) -> dict:
    """Materialize one source; reserve entries are import-only, never resized here."""
    corpus, out = Path(corpus_root).resolve(), Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "id": entry.get("id"),
        "status": "failed",
        "failures": [],
        "node_mapping": [],
        "editable_text_count": 0,
        "vector_art_count": 0,
        "parity_review": "pending",
        "attribution": {
            k: entry.get(k)
            for k in (
                "title",
                "author",
                "source_url",
                "license_id",
                "license_url",
                "license_path",
                "source_commit",
                "split",
            )
        },
        "comparison_paper": (
            "Both native/imported renders composited on declared white paper; "
            "no authored background element is added."
        ),
    }
    try:
        original = _safe(corpus, entry["original_path"])
        raw = original.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        report["original_sha256"] = digest
        (out / "original.svg").write_bytes(raw)
        if entry.get("sha256") and entry["sha256"] != digest:
            raise ImportBlock(
                "source_hash_mismatch", "Original bytes differ from the catalogue SHA-256."
            )
        # Synthetic tests may omit attribution; genuine catalogue entries supply
        # pinned source/font license paths and those bytes travel with the case.
        licenses = _licenses(entry, corpus, out) if entry.get("license_id") != "test" else []
        report["licenses"] = licenses
        reference = (
            _safe(corpus, entry["prepared_reference_path"])
            if entry.get("prepared_reference_path")
            else original
        )
        reference_raw = reference.read_bytes()
        reference_digest = hashlib.sha256(reference_raw).hexdigest()
        if (
            entry.get("prepared_reference_sha256")
            and entry["prepared_reference_sha256"] != reference_digest
        ):
            raise ImportBlock(
                "reference_hash_mismatch", "Adapted reference differs from its declared SHA-256."
            )
        report.update(
            reference_sha256=reference_digest,
            baseline_kind=entry.get("baseline_kind", "author_original"),
            adaptations=entry.get("adaptations", []),
            adaptation_note=entry.get("adaptation_note"),
        )
        if reference != original:
            (out / "reference.svg").write_bytes(reference_raw)
        importer = _Importer(entry, corpus, out, reference_raw)
        report["canvas"] = {"width": importer.width, "height": importer.height}
        report["viewport_convention"] = importer.viewport_convention
        issues = importer.audit()
        report["font_resolutions"] = importer.font_resolutions
        report["source_tree"] = [
            {
                "source_key": k,
                "parent": importer.parents[k],
                "tag": _tag(n),
                "id": n.get("id"),
                "attributes": dict(n.attrib),
            }
            for k, n in importer.nodes.items()
        ]
        if not any(
            run["content"].strip() and importer._visible(run["key"]) and "font" not in run
            for run in importer.text_runs
        ):
            importer.render().save(out / "native.png")
        if issues:
            report.update(status="unsupported", failures=issues)
        else:
            for folder in (out / "resources/assets", out / "resources/fonts"):
                folder.mkdir(parents=True, exist_ok=True)
            font_manifest, font_payloads = [], []
            for ps, face in importer.fonts.items():
                filename = (
                    hashlib.sha256(face.data).hexdigest()[:16] + face.source_path.suffix.lower()
                )
                (out / "resources/fonts" / filename).write_bytes(face.data)
                font_manifest.append(
                    {
                        "font_family": ps,
                        "postscript_name": ps,
                        "weight": "regular",
                        "style": "normal",
                        "file": filename,
                    }
                )
                font_payloads.append(
                    {
                        "font_family": ps,
                        "postscript_name": ps,
                        "weight": "regular",
                        "style": "normal",
                        "data_base64": base64.b64encode(face.data).decode(),
                    }
                )
            (out / "resources/fonts/font-manifest.json").write_text(
                json.dumps({"fonts": font_manifest}, indent=2), encoding="utf-8"
            )
            importer.render().save(out / "native.png")
            from engine.resources import using_resources

            with using_resources(out / "resources/assets", out / "resources/fonts"):
                source = importer.convert()
                from engine.eval.render import render_image

                converted = render_image(source, max_size=max(source["canvas"].values())).convert(
                    "RGBA"
                )
            converted.save(out / "imported.png")
            native = Image.open(out / "native.png").convert("RGBA")
            paper = Image.new("RGBA", native.size, "white")
            native_paper = Image.alpha_composite(paper, native).convert("RGB")
            converted_paper = Image.alpha_composite(paper, converted).convert("RGB")
            mae = sum(ImageStat.Stat(ImageChops.difference(native_paper, converted_paper)).mean) / (
                3 * 255
            )
            report["native_imported_normalized_mae"] = mae
            (out / "source.json").write_text(
                json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            bundle = {
                "task_id": entry["id"],
                "poster_id": entry["id"],
                "attribution": {
                    "text": entry.get("attribution_text")
                    or (
                        f"{entry.get('title', entry['id'])} by {entry.get('author', 'unknown')}; "
                        f"{entry.get('license_id', 'license in source record')}"
                    ),
                    "author": entry.get("author"),
                    "source_url": entry.get("source_url"),
                    "license_id": entry.get("license_id"),
                    "license_url": entry.get("license_url"),
                    "original_sha256": digest,
                    "license_paths": [item["path"] for item in licenses],
                    "licenses": licenses,
                    "baseline_kind": report["baseline_kind"],
                    "reference_sha256": reference_digest,
                    "adaptation_note": "; ".join(
                        filter(
                            None,
                            (
                                report.get("adaptation_note"),
                                entry.get("source_warning"),
                                *(warning["message"] for warning in importer.source_warnings),
                            ),
                        )
                    )
                    or None,
                    "source_warnings": importer.source_warnings,
                },
                "source": source,
                "target_canvas": {"width": 1080, "height": 1080},
                "assets": importer.asset_payloads,
                "fonts": font_payloads,
                "document": {
                    "snapshot_id": reference_digest,
                    "page_id": entry["id"],
                    "artboard_id": entry["id"],
                    "source_kind": "licensed-original-svg",
                },
            }
            (out / "bundle.json").write_text(
                json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
            )
            report.update(
                status="ready",
                node_mapping=importer.mapping,
                editable_text_count=sum(m["mode"] == "editable_text" for m in importer.mapping),
                vector_art_count=sum(m["mode"] == "rigid_vector_art" for m in importer.mapping),
                native_shape_count=sum(m["mode"] == "native_shape" for m in importer.mapping),
                ignored_nodes=importer.ignored,
                source_warnings=importer.source_warnings,
                source_tree=[
                    {
                        "source_key": k,
                        "parent": importer.parents[k],
                        "tag": _tag(n),
                        "id": n.get("id"),
                        "attributes": dict(n.attrib),
                    }
                    for k, n in importer.nodes.items()
                ],
                authored_groups=importer.groups,
            )
    except ImportBlock as exc:
        report.update(status="unsupported", failures=[exc.record()])
    except Exception as exc:  # retain failures per original, never silently drop an example
        report.update(
            status="failed",
            failures=[{"code": type(exc).__name__, "node_id": None, "message": str(exc)}],
        )
    if report["status"] != "ready":
        # A stale previous bundle must never make a failed preparation runnable.
        for name in ("bundle.json", "source.json"):
            (out / name).unlink(missing_ok=True)
    (out / "import-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    catalogue = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [
        entry
        for entry in catalogue["examples"]
        if Path(entry["original_path"]).suffix.lower() == ".svg"
        and (not args.case or entry["id"] in args.case)
    ]
    if args.case and set(args.case) - {entry["id"] for entry in selected}:
        parser.error("Unknown catalogue case ID.")
    failed = False
    for entry in selected:
        out = _safe(args.manifest.parent, entry.get("prepared_path", "prepared/" + entry["id"]))
        report = prepare_poster(entry, args.manifest.parent, out)
        print(entry["id"], report["status"], json.dumps(report["failures"]), flush=True)
        failed |= report["status"] != "ready"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
