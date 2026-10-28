"""Serve the local preview UI and the offline resize engine on loopback.

Run ``python -m viewer.server --port 8765`` and open http://127.0.0.1:8765.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import threading
import time
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from engine.eval import build_context, score
from engine.eval.catalogue import CATALOGUE
from engine.eval.render import render_plate
from engine.eval.style import analyze_style
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.resize import resize
from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = REPO_ROOT / "viewer"
ASSETS_DIR = REPO_ROOT / "brand_kit" / "assets"
FONTS_DIR = REPO_ROOT / "brand_kit" / "fonts"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_DIMENSION = 8192
MAX_PIXELS = 20_000_000
MAX_ELEMENTS = 200
_ENGINE_LOCK = threading.Lock()
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


def _render_png(design: dict) -> tuple[str, list[str]]:
    renderer = _get_renderer(str(FONTS_DIR))
    example, warnings = convert_chameleon_to_crello(design, renderer.font_manager, str(ASSETS_DIR))
    canvas = design["canvas"]
    png = renderer.render(example, max_size=max(canvas["width"], canvas["height"]), format="png")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii"), warnings


def perform_resize(payload: dict) -> dict:
    validate_request(payload)
    # Engine/renderer caches are reused, and one request owns them at a time.
    source, target = deepcopy(payload["source"]), deepcopy(payload["target_canvas"])
    started = time.perf_counter()
    with _ENGINE_LOCK:
        design = resize(source, target)
        context = build_context(source, ASSETS_DIR, FONTS_DIR)
        plate = render_plate(design, ASSETS_DIR, FONTS_DIR, max_size=1024)
        result = score(source, design, context, plate, target)
        style = analyze_style(source, design, ASSETS_DIR)
        relationships = analyze_relationships(build_relationship_graph(source, ASSETS_DIR), design)
        source_png, source_warnings = _render_png(source)
        output_png, output_warnings = _render_png(design)
    return {
        "design": design,
        "source_png": source_png,
        "output_png": output_png,
        "report": {
            "violations": [str(violation) for violation in result["violations"]],
            "metrics": result["metrics"],
            "total": result["total"],
            "style": style,
            "relationships": relationships,
        },
        "warnings": list(dict.fromkeys(source_warnings + output_warnings)),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


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
    def _send(self, status: int, body: bytes, content_type: str, *, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _json(self, status: int, payload: dict, *, head: bool = False) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
            head=head,
        )

    def _local_request(self) -> bool:
        port = self.server.server_port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed_hosts:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Use the local server address."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed_hosts}:
            self._json(
                HTTPStatus.FORBIDDEN, {"error": "Only same-origin local requests are allowed."}
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
        if route == "/api/samples":
            try:
                self._json(HTTPStatus.OK, sample_catalogue(), head=head)
            except Exception as exc:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Samples unavailable: {exc}"}
                )
            return
        root = FONTS_DIR if route.startswith("/fonts/") else VIEWER_DIR
        relative = route.removeprefix("/fonts/") if root == FONTS_DIR else route.lstrip("/")
        relative = unquote(relative or "index.html")
        if "\\" in relative or "\0" in relative:
            self._json(HTTPStatus.NOT_FOUND, {"error": "File not found."}, head=head)
            return
        try:
            path = (root / relative).resolve()
            path.relative_to(root.resolve())
        except (ValueError, OSError):
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

    def do_POST(self) -> None:
        if not self._local_request():
            return
        if urlsplit(self.path).path != "/api/resize":
            self._json(HTTPStatus.NOT_FOUND, {"error": "API endpoint not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Use application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length > MAX_REQUEST_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request exceeds 2 MiB."})
            return
        if length <= 0:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "A JSON request body is required."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            validate_request(payload)
        except (ValueError, UnicodeError, RecursionError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        try:
            self._json(HTTPStatus.OK, perform_resize(payload))
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Resize failed: {exc}"})


def create_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), ViewerHandler)


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
    raise SystemExit(main())
