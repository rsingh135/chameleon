"""Serve the local preview UI and the offline resize engine on loopback.

Run ``python -m viewer.server --port 8765`` and open http://127.0.0.1:8765.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import sys
import threading
import time
from contextlib import suppress
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit

from engine.design_semantics import audit_semantic_preservation, describe_design
from engine.eval import build_context, score
from engine.eval.catalogue import CATALOGUE
from engine.eval.occlusion import conservative_occlusion
from engine.eval.proportions import analyze_proportions
from engine.eval.render import render_plate
from engine.eval.style import analyze_style
from engine.layout.focal import analyze_focal
from engine.layout.reading_order import analyze_reading_order
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.layout.rules import analyze_rules
from engine.resize import resize_with_state
from engine.resources import (
    ResourceContext,
    get_renderer,
    get_resources,
    native_render_example,
    render_input,
    using_resources,
)
from renderer.converter.convert import convert_chameleon_to_crello
from viewer import poster_library
from viewer.document_bridge import (
    DOCUMENT_ORIGIN,
    MAX_DOCUMENT_REQUEST_BYTES,
    prepare_document,
)
from viewer.resize_cache import (
    ResourceBindings,
    ResourceChangedError,
    SerializedCache,
    result_key,
    source_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = REPO_ROOT / "viewer"
ASSETS_DIR = REPO_ROOT / "brand_kit" / "assets"
FONTS_DIR = REPO_ROOT / "brand_kit" / "fonts"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_DIMENSION = 8192
MAX_PIXELS = 20_000_000
MAX_ELEMENTS = 200
BODY_READ_TIMEOUT = 10.0
MAX_IN_FLIGHT_RESIZES = 4
_ENGINE_LOCK = threading.Lock()
_RESOURCE_BINDINGS = ResourceBindings()
_SOURCE_PNG_CACHE = SerializedCache(kind="source")
_RESIZE_RESULT_CACHE = SerializedCache(kind="result")
_STATIC_SUFFIXES = {".html", ".css", ".js", ".json", ".svg", ".png", ".ico", ".woff", ".woff2"}


def _number(value, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}.")
    return value


def _canvas(value, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object with width and height.")
    for key in ("width", "height"):
        dimension = value.get(key)
        if type(dimension) is not int or not 1 <= dimension <= MAX_DIMENSION:
            raise ValueError(f"{name}.{key} must be an integer from 1 to {MAX_DIMENSION}.")
    if value["width"] * value["height"] > MAX_PIXELS:
        raise ValueError(f"{name} must contain at most {MAX_PIXELS:,} pixels.")


def _string(element: dict, key: str, name: str, *, maximum: int = 20000) -> str:
    value = element.get(key)
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{name}.{key} must be a string of at most {maximum} characters.")
    return value


def _finite_tree(value, depth: int = 0) -> None:
    if depth > 40:
        raise ValueError("Request nesting is too deep.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Request numbers must be finite.")
    children = (
        value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    )
    for child in children:
        _finite_tree(child, depth + 1)


def validate_request(payload) -> None:
    """Reject malformed or excessive work before it reaches the renderer."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object containing source and target_canvas.")
    _finite_tree(payload)
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be a design object.")
    _canvas(source.get("canvas"), "source.canvas")
    _canvas(payload.get("target_canvas"), "target_canvas")
    elements = source.get("elements")
    if not isinstance(elements, list) or not 1 <= len(elements) <= MAX_ELEMENTS:
        raise ValueError(f"source.elements must contain between 1 and {MAX_ELEMENTS} elements.")

    ids = set()
    for index, element in enumerate(elements):
        name = f"source.elements[{index}]"
        if not isinstance(element, dict):
            raise ValueError(f"{name} must be an object.")
        eid = _string(element, "id", name, maximum=200)
        if not eid or eid in ids:
            raise ValueError(f"{name}.id must be nonempty and unique.")
        ids.add(eid)
        kind = element.get("type")
        if kind not in ("text", "image", "shape"):
            raise ValueError(f"{name}.type must be text, image, or shape.")
        for key in ("x", "y"):
            _number(element.get(key), f"{name}.{key}", -32768, 32768)
        for key in ("width", "height"):
            _number(element.get(key), f"{name}.{key}", 0.001, MAX_DIMENSION)
        if element["width"] * element["height"] > MAX_PIXELS:
            raise ValueError(f"{name} exceeds the {MAX_PIXELS:,} pixel limit.")

        for key, lower, upper in (
            ("rotation", -3600, 3600),
            ("opacity", 0, 1),
            ("line_height", 0.01, 10),
            ("letter_spacing", -1000, 1000),
            ("stroke_width", 0, MAX_DIMENSION),
            ("corner_radius", 0, MAX_DIMENSION),
        ):
            if key in element:
                _number(element[key], f"{name}.{key}", lower, upper)
        if "z_index" in element and type(element["z_index"]) is not int:
            raise ValueError(f"{name}.z_index must be an integer.")
        if "group_id" in element and element["group_id"] is not None:
            _string(element, "group_id", name, maximum=200)

        if kind == "text":
            content = _string(element, "content", name)
            _string(element, "font_family", name, maximum=200)
            _string(element, "color", name, maximum=200)
            _number(element.get("font_size"), f"{name}.font_size", 1, MAX_DIMENSION)
            for key, choices in (
                ("font_weight", ("regular", "bold")),
                ("font_style", ("normal", "italic")),
                ("text_align", ("left", "center", "right")),
            ):
                if key in element and element[key] not in choices:
                    raise ValueError(f"{name}.{key} must be one of {', '.join(choices)}.")
            max_lines = element.get("max_lines")
            if max_lines is not None and (type(max_lines) is not int or max_lines < 1):
                raise ValueError(f"{name}.max_lines must be null or a positive integer.")
            runs = element.get("style_runs", [])
            if not isinstance(runs, list) or len(runs) > len(content) + 1:
                raise ValueError(f"{name}.style_runs must be a list of text ranges.")
            for run in runs:
                if not isinstance(run, dict):
                    raise ValueError(f"{name}.style_runs must contain objects.")
                start, end = run.get("start"), run.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start <= end:
                    raise ValueError(f"{name}.style_runs ranges must be nonnegative and ordered.")
                for key in ("color", "font_weight", "font_style"):
                    if key in run and not isinstance(run[key], str):
                        raise ValueError(f"{name}.style_runs.{key} must be a string.")
            if "native_text_effects" in element:
                from engine.native_text_effects import validate_descriptor

                try:
                    validate_descriptor(element["native_text_effects"], content)
                except (
                    AttributeError,
                    IndexError,
                    NotImplementedError,
                    OverflowError,
                    TypeError,
                ) as exc:
                    # The native color parser reports malformed user values
                    # with several exception types. Keep those validation
                    # failures at the input boundary instead of disconnecting.
                    detail = str(exc)[:300]
                    raise ValueError(f"{name}.native_text_effects is invalid: {detail}") from exc
        elif kind == "image":
            asset = _string(element, "asset_id", name, maximum=255)
            if (
                not asset
                or any(c in asset for c in "/\\:")
                or any(ord(c) < 32 for c in asset)
                or asset in (".", "..")
            ):
                raise ValueError(f"{name}.asset_id must be a filename inside brand_kit/assets.")
            try:
                (ASSETS_DIR / asset).resolve().relative_to(ASSETS_DIR.resolve())
            except ValueError as exc:
                raise ValueError(f"{name}.asset_id is outside brand_kit/assets.") from exc
            if element.get("fit", "contain") not in ("contain", "cover", "fill"):
                raise ValueError(f"{name}.fit must be contain, cover, or fill.")
        else:
            if element.get("shape_type") not in ("rectangle", "ellipse", "line"):
                raise ValueError(f"{name}.shape_type must be rectangle, ellipse, or line.")
            _string(element, "fill_color", name, maximum=200)
            if element.get("stroke_color") is not None:
                _string(element, "stroke_color", name, maximum=200)


def _render_png(design: dict, attribution: dict | None = None) -> tuple[str, list[str]]:
    resources = get_resources()
    renderer = get_renderer(resources.fonts_dir)
    example, warnings = convert_chameleon_to_crello(
        render_input(design), renderer.font_manager, str(resources.assets_dir)
    )
    example = native_render_example(example, design)
    canvas = design["canvas"]
    png = renderer.render(example, max_size=max(canvas["width"], canvas["height"]), format="png")
    if isinstance(attribution, dict):
        from PIL import Image, PngImagePlugin

        metadata = PngImagePlugin.PngInfo()
        for field, keyword in (
            ("text", "Attribution"),
            ("author", "Author"),
            ("source_url", "Source"),
            ("license_id", "License"),
            ("license_url", "License URL"),
            ("original_sha256", "Original SHA256"),
        ):
            value = attribution.get(field)
            if isinstance(value, str) and value.strip():
                metadata.add_text(keyword, value[:2000])
        with Image.open(BytesIO(png)) as image:
            exported = BytesIO()
            image.save(exported, format="PNG", pnginfo=metadata)
            png = exported.getvalue()
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii"), warnings


def _execution_identity() -> str:
    from viewer.benchmark import method_receipt

    return method_receipt()["digest"]


def _visual_model_enabled() -> bool:
    # Invalid configured paths also bypass result reuse: a cached deterministic
    # response must never conceal a requested model or its configuration error.
    if not os.environ.get("CHAMELEON_VISUAL_PLANNER_CONFIG"):
        return False
    from engine.visual_pipeline import visual_model_enabled

    return visual_model_enabled()


def _configured_resize(source, target, *, assets_dir, fonts_dir):
    from engine.visual_pipeline import resize_configured

    return resize_configured(
        source, target, assets_dir=assets_dir, fonts_dir=fonts_dir, strict_stage_one=True
    )


def _verify_resize_binding(source, resources, bound, execution, expected_source_key, attribution):
    current = _RESOURCE_BINDINGS.bind(
        source, resources.assets_dir, resources.fonts_dir, allow_missing=not bound.complete
    )
    if current.digest != bound.digest:
        raise ResourceChangedError("Resize resources changed. Restart the server.")
    if _execution_identity() != execution:
        raise RuntimeError("Engine files changed. Restart the server before resizing.")
    if (
        source_key(source, resources=current, execution_identity=execution, attribution=attribution)
        != expected_source_key
    ):
        raise RuntimeError("The processor changed the bound original source.")


def _resize_response(value, started, *, result_hit, source_hit):
    value["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    value["cache"] = {"result_hit": result_hit, "source_render_hit": source_hit}
    return value


def _source_semantics(source, state, *, assets_dir, fonts_dir):
    candidate = state.get("design_semantics")
    if candidate is not None:
        try:
            # The source-to-source audit validates digest and complete coverage.
            # Omit resources here so validation repeats no native ink analysis.
            audit_semantic_preservation(source, source, candidate)
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
            pass
        else:
            return deepcopy(candidate)
    return describe_design(source, assets_dir=assets_dir, fonts_dir=fonts_dir)


def perform_resize(payload: dict, *, resources: ResourceContext | None = None) -> dict:
    validate_request(payload)
    # Engine/renderer caches are reused, and one request owns them at a time.
    source, target = deepcopy(payload["source"]), deepcopy(payload["target_canvas"])
    requested_target = deepcopy(target)
    attribution = deepcopy(payload.get("attribution"))
    started = time.perf_counter()
    resources = resources or ResourceContext(ASSETS_DIR, FONTS_DIR)
    assets_dir, fonts_dir = resources.assets_dir, resources.fonts_dir
    execution = _execution_identity()
    model_enabled = _visual_model_enabled()
    bound = _RESOURCE_BINDINGS.bind(source, assets_dir, fonts_dir, allow_missing=not model_enabled)
    skey = source_key(
        source, resources=bound, execution_identity=execution, attribution=attribution
    )
    # Legacy placeholder renders remain available, but incomplete packages
    # cannot enter either cache. The full source key still guards mutation.
    source_cache_key = skey if bound.complete else None
    rkey = result_key(
        source,
        target,
        resources=bound,
        execution_identity=execution,
        attribution=attribution,
        model_enabled=model_enabled,
    )
    cached = _RESIZE_RESULT_CACHE.get(rkey)
    if cached is not None:
        _verify_resize_binding(source, resources, bound, execution, skey, attribution)
        return _resize_response(cached, started, result_hit=True, source_hit=True)
    with _ENGINE_LOCK, using_resources(assets_dir, fonts_dir):
        _verify_resize_binding(source, resources, bound, execution, skey, attribution)
        # Another request may have completed the same job while this one waited.
        cached = _RESIZE_RESULT_CACHE.get(rkey)
        if cached is not None:
            _verify_resize_binding(source, resources, bound, execution, skey, attribution)
            return _resize_response(cached, started, result_hit=True, source_hit=True)
        if model_enabled:
            plan = _configured_resize(
                deepcopy(source), deepcopy(target), assets_dir=assets_dir, fonts_dir=fonts_dir
            )
        else:
            plan = resize_with_state(deepcopy(source), deepcopy(target))
        design = plan["design"]
        if design.get("canvas") != requested_target:
            raise RuntimeError("The processor returned a mismatched target canvas.")
        design_semantics = _source_semantics(
            source, plan["state"], assets_dir=assets_dir, fonts_dir=fonts_dir
        )
        context = build_context(source, assets_dir, fonts_dir)
        plate = render_plate(design, assets_dir, fonts_dir, max_size=1024)
        result = score(source, design, context, plate, target)
        graph = build_relationship_graph(source, assets_dir)
        style = analyze_style(source, design, assets_dir, relationship_graph=graph)
        relationships = analyze_relationships(graph, design, plan["state"]["cluster_axes"])
        proportions = analyze_proportions(source, design, graph)
        focal = analyze_focal(source, design, graph)
        rules = analyze_rules(source, design)
        reading_order = analyze_reading_order(source, design)
        occlusion = conservative_occlusion(source, design)
        source_render = _SOURCE_PNG_CACHE.get(source_cache_key)
        source_hit = source_render is not None
        if source_render is None:
            source_png, source_warnings = _render_png(source, attribution)
            source_render = {"png": source_png, "warnings": source_warnings}
        else:
            source_png, source_warnings = source_render["png"], source_render["warnings"]
        output_png, output_warnings = _render_png(design, attribution)
        response = {
            "design": design,
            "source_png": source_png,
            "output_png": output_png,
            "report": {
                "violations": [str(violation) for violation in result["violations"]],
                "metrics": result["metrics"],
                "total": result["total"],
                "style": style,
                "relationships": relationships,
                "proportions": proportions,
                "focal": focal,
                "rules": rules,
                "reading_order": reading_order,
                "occlusion": occlusion,
                "design_semantics": design_semantics,
                "layout_state": plan["state"],
            },
            "warnings": list(dict.fromkeys(source_warnings + output_warnings)),
        }
        if target != requested_target:
            raise RuntimeError("The processor changed the requested target canvas.")
        _verify_resize_binding(source, resources, bound, execution, skey, attribution)
        _SOURCE_PNG_CACHE.put_success(source_cache_key, source_render)
        _RESIZE_RESULT_CACHE.put_success(rkey, response)
        return _resize_response(response, started, result_hit=False, source_hit=source_hit)


def sample_catalogue() -> dict:
    samples = []
    for path in sorted((REPO_ROOT / "input").glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        samples.append(
            {"id": path.stem, "label": task.get("difficulty_label", path.stem), "task": task}
        )
    targets = [
        {"id": name, "label": name.replace("_", " ").title(), **canvas}
        for name, canvas in CATALOGUE.items()
    ]
    return {"samples": samples, "targets": targets}


class ViewerHandler(BaseHTTPRequestHandler):
    # Also bound incomplete headers; the tighter total-body deadline below
    # prevents a trickling upload from extending its processing time forever.
    timeout = 15
    _rejection_drain_bytes = 64 * 1024
    _rejection_drain_seconds = 0.25

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        head: bool = False,
        extra_headers: dict | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status >= 400:
            self.close_connection = True
            self.send_header("Connection", "close")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        if (
            urlsplit(self.path).path == "/api/document/resize"
            and self.headers.get("Origin") == DOCUMENT_ORIGIN
            and self.headers.get("Host")
            in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        ):
            self.send_header("Access-Control-Allow-Origin", DOCUMENT_ORIGIN)
            self.send_header("Vary", "Origin")
            if self.command == "OPTIONS":
                self.send_header("Access-Control-Allow-Methods", "POST")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
                self.send_header("Access-Control-Allow-Private-Network", "true")
        try:
            self.end_headers()
            if not head:
                self.wfile.write(body)
            self.wfile.flush()
            if status >= 400:
                self._discard_rejected_body()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browser cancellation is normal. Never attempt a second error
            # response on the same disconnected socket or strand a work slot.
            self.close_connection = True

    def _json(self, status: int, payload: dict, *, head: bool = False) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
            head=head,
        )

    def _local_request(self, *, document: bool = False) -> bool:
        port = self.server.server_port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed_hosts:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Use the local server address."})
            return False
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://{host}" for host in allowed_hosts}
        if document:
            allowed_origins.add(DOCUMENT_ORIGIN)
        if origin and origin not in allowed_origins:
            self._json(
                HTTPStatus.FORBIDDEN, {"error": "Only same-origin local requests are allowed."}
            )
            return False
        return True

    def _require_current_method(self, *, action="Benchmark", head=False) -> bool:
        from viewer.benchmark import method_receipt

        try:
            current = method_receipt()
        except Exception as exc:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{action} failed: {exc}"},
                head=head,
            )
            return False
        if current["digest"] != self.server.benchmark_method["digest"]:
            self._json(
                HTTPStatus.CONFLICT,
                {"error": "Engine files changed. Restart the server before resizing."},
                head=head,
            )
            return False
        return True

    def _get(self, *, head: bool = False) -> None:
        if not self._local_request():
            return
        route = urlsplit(self.path).path
        if route == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok"}, head=head)
            return
        if route == "/api/benchmark/method":
            if self._require_current_method(head=head):
                self._json(HTTPStatus.OK, self.server.benchmark_method, head=head)
            return
        if route == "/api/resize/policy":
            if self._require_current_method(action="Resize", head=head):
                self._json(
                    HTTPStatus.OK,
                    self.server.benchmark_method["options"]["request_policy"],
                    head=head,
                )
            return
        if route == "/api/samples":
            try:
                self._json(HTTPStatus.OK, sample_catalogue(), head=head)
            except Exception as exc:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Samples unavailable: {exc}"}
                )
            return
        if route == "/api/posters":
            try:
                self._json(HTTPStatus.OK, poster_library.library_metadata(), head=head)
            except (ValueError, OSError, KeyError) as exc:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Poster library unavailable: {exc}"},
                )
            return
        if (
            route.startswith("/api/posters/")
            or route in {"/posters", "/posters/"}
            or route.startswith("/posters/")
        ):
            try:
                attachment = False
                if route.startswith("/api/posters/"):
                    parts = route.removeprefix("/api/posters/").split("/")
                    if len(parts) != 2:
                        raise FileNotFoundError("Poster endpoint not found.")
                    poster_id, kind = parts
                    if kind == "bundle":
                        payload = poster_library.read_bundle(poster_id)
                        validate_request(payload)
                        self._json(HTTPStatus.OK, payload, head=head)
                        return
                    if kind == "original":
                        path, attachment = poster_library.original_file(poster_id), True
                    elif kind == "preview":
                        path = poster_library.preview_file(poster_id)
                    else:
                        raise FileNotFoundError("Poster endpoint not found.")
                else:
                    relative = unquote(route.removeprefix("/posters").lstrip("/"))
                    path = poster_library.gallery_file(relative)
                if not path.is_file():
                    raise FileNotFoundError("Poster file is not available.")
                headers = {
                    "Content-Security-Policy": (
                        "default-src 'none'; img-src 'self' data:; "
                        "style-src 'self' 'unsafe-inline'; base-uri 'none'; form-action 'none'"
                    )
                }
                if attachment:
                    headers["Content-Disposition"] = "attachment"
                kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._send(HTTPStatus.OK, path.read_bytes(), kind, head=head, extra_headers=headers)
            except FileNotFoundError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)}, head=head)
            except (ValueError, OSError, KeyError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)}, head=head)
            return
        root = FONTS_DIR if route.startswith("/fonts/") else VIEWER_DIR
        relative = route.removeprefix("/fonts/") if root == FONTS_DIR else route.lstrip("/")
        relative = unquote(relative or "index.html")
        if "\\" in relative or "\0" in relative:
            self._json(HTTPStatus.NOT_FOUND, {"error": "File not found."}, head=head)
            return
        try:
            path = (root / relative).resolve()
            resolved_relative = path.relative_to(root.resolve()).as_posix().casefold()
        except (ValueError, OSError):
            self._json(HTTPStatus.NOT_FOUND, {"error": "File not found."}, head=head)
            return
        if root == VIEWER_DIR:
            if resolved_relative in {"benchmark", "benchmark.html"}:
                self._send(
                    HTTPStatus.SEE_OTHER,
                    b"",
                    "text/plain; charset=utf-8",
                    head=head,
                    extra_headers={"Location": "/"},
                )
                return
            if resolved_relative in {"benchmark.js", "benchmark.css"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": "File not found."}, head=head)
                return
        allowed = {".ttf", ".woff", ".woff2"} if root == FONTS_DIR else _STATIC_SUFFIXES
        if not path.is_file() or path.suffix.lower() not in allowed:
            self._json(HTTPStatus.NOT_FOUND, {"error": "File not found."}, head=head)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in {".html", ".css", ".js", ".json", ".svg"}:
            content_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type, head=head)

    def do_GET(self) -> None:
        self._get()

    def do_HEAD(self) -> None:
        self._get(head=True)

    def do_OPTIONS(self) -> None:
        document = urlsplit(self.path).path == "/api/document/resize"
        if not self._local_request(document=document):
            return
        if not document:
            self._json(HTTPStatus.NOT_FOUND, {"error": "API endpoint not found."})
            return
        headers = {
            name.strip().lower()
            for name in self.headers.get("Access-Control-Request-Headers", "").split(",")
            if name.strip()
        }
        if (
            self.headers.get("Origin") != DOCUMENT_ORIGIN
            or self.headers.get("Access-Control-Request-Method") != "POST"
            or not headers <= {"content-type"}
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "Unsupported document preflight."})
            return
        self._send(HTTPStatus.NO_CONTENT, b"", "application/json")

    def do_POST(self) -> None:
        self._body_read_attempted = False
        route = urlsplit(self.path).path
        document = route == "/api/document/resize"
        if not self._local_request(document=document):
            return
        benchmark = route == "/api/benchmark/run"
        if route not in {"/api/resize", "/api/document/resize", "/api/benchmark/run"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "API endpoint not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Use application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        limit = MAX_DOCUMENT_REQUEST_BYTES if document or benchmark else MAX_REQUEST_BYTES
        if length > limit:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"Request exceeds {limit // (1024 * 1024)} MiB."},
            )
            return
        if length <= 0:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "A JSON request body is required."})
            return
        try:
            raw = self._read_body(length)
        except TimeoutError:
            self._json(
                HTTPStatus.REQUEST_TIMEOUT,
                {"error": "Request body timed out; retry the upload."},
            )
            return
        except (ValueError, ConnectionError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        # Fully consume a bounded body before rejecting overload, avoiding a
        # Windows TCP reset that could conceal the useful 503 response.
        if not self.server.resize_slots.acquire(blocking=False):
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "The local engine is busy. Wait for a current run, then retry."},
            )
            return
        try:
            self._process_resize_request(raw, document=document, benchmark=benchmark)
        finally:
            self.server.resize_slots.release()

    def _discard_rejected_body(self) -> None:
        """Bounded transport cleanup after an early error was already flushed.

        Windows can reset a socket closed with ordinary POST bytes unread,
        concealing a complete 403/415 response. Never parse rejected data, wait
        before sending the rejection, follow ambiguous framing, or drain without
        independent byte and absolute-time limits.
        """
        if self.command != "POST" or getattr(self, "_body_read_attempted", False):
            return
        self._body_read_attempted = True
        lengths = self.headers.get_all("Content-Length", [])
        if (
            len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
            or len(lengths[0]) > 10
            or self.headers.get("Transfer-Encoding")
        ):
            return
        remaining = min(int(lengths[0]), self._rejection_drain_bytes)
        if remaining <= 0:
            return
        deadline = time.monotonic() + self._rejection_drain_seconds
        previous_timeout = self.connection.gettimeout()
        try:
            while remaining:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    break
                self.connection.settimeout(budget)
                chunk = self.rfile.read1(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, ValueError):
            pass  # Cleanup must never replace the error already sent.
        finally:
            with suppress(OSError):
                self.connection.settimeout(previous_timeout)

    def _read_body(self, length: int) -> bytes:
        self._body_read_attempted = True
        deadline = time.monotonic() + BODY_READ_TIMEOUT
        previous_timeout = self.connection.gettimeout()
        remaining = length
        chunks = []
        try:
            while remaining:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("Request body timed out.")
                self.connection.settimeout(budget)
                chunk = self.rfile.read1(min(remaining, 65536))
                if not chunk:
                    raise ValueError("Incomplete request body; retry the upload.")
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            self.connection.settimeout(previous_timeout)
        return b"".join(chunks)

    def _process_resize_request(self, raw: bytes, *, document: bool, benchmark: bool) -> None:
        try:
            payload = json.loads(raw.decode("utf-8"))
            if benchmark:
                from viewer.benchmark import run_benchmark

                if not self._require_current_method():
                    return
                try:
                    result = run_benchmark(payload)
                except (ValueError, UnicodeError, RecursionError):
                    raise
                except Exception as exc:
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"Benchmark failed: {exc}"},
                    )
                    return
                self._json(HTTPStatus.OK, result)
                return
            validate_request(payload)
            if not self._require_current_method(action="Resize"):
                return
            prepared = prepare_document(payload) if document else None
        except (ValueError, UnicodeError, RecursionError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except OSError as exc:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Document preparation failed: {exc}"},
            )
            return
        try:
            if not self._require_current_method(action="Resize"):
                return
            if prepared is None:
                result = perform_resize(payload)
            else:
                result = perform_resize(payload, resources=prepared.resources)
                result["document"] = prepared.receipt
                result["preview_url"] = result["output_png"]
            if not self._require_current_method(action="Resize"):
                return
            self._json(HTTPStatus.OK, result)
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Resize failed: {exc}"})


def create_server(port: int = 8765) -> ThreadingHTTPServer:
    from viewer.benchmark import method_receipt

    instance = ThreadingHTTPServer(("127.0.0.1", port), ViewerHandler)
    instance.resize_slots = threading.BoundedSemaphore(MAX_IN_FLIGHT_RESIZES)
    try:
        instance.benchmark_method = method_receipt()
    except Exception:
        instance.server_close()
        raise
    return instance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        server = create_server(args.port)
    except OSError as exc:
        parser.exit(1, f"Could not start local viewer: {exc}\n")
    print(f"Chameleon is running at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    # ``python -m viewer.server`` executes this module as __main__. The batch
    # runner imports viewer.server lazily; keep both names bound to this same
    # module so Studio and batch work share the renderer lock and request state.
    sys.modules["viewer.server"] = sys.modules[__name__]
    sys.modules["viewer"].server = sys.modules[__name__]
    raise SystemExit(main())
