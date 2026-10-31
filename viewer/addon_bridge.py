"""Stable Express viewer bridge over the same strict stage-one engine contract."""

from __future__ import annotations

import hashlib
import json
import os
import time
from copy import deepcopy
from urllib.parse import urlsplit

from engine.eval import check_contract
from engine.resources import ResourceContext, load_font_manager, using_resources
from engine.stage_one import strictly_in_bounds

DEVELOPMENT_ORIGIN = "https://localhost:5241"


def addon_origins(values=None) -> frozenset[str]:
    """Allow explicit HTTPS add-on origins, without opening the studio's other APIs."""
    values = values if values is not None else os.environ.get("CHAMELEON_ADDON_ORIGINS", "")
    candidates = values.split(",") if isinstance(values, str) else list(values)
    if len(candidates) > 16:
        raise ValueError("Configure at most 16 explicit add-on origins.")
    result = {DEVELOPMENT_ORIGIN}
    for candidate in candidates:
        value = candidate.strip()
        if not value:
            continue
        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Add-on origins require a valid HTTPS port.") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or "*" in value
            or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
        ):
            raise ValueError("Add-on origins must be explicit HTTPS origins without paths.")
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        result.add(f"https://{host}" + (f":{port}" if port and port != 443 else ""))
    return frozenset(result)


def source_digest(source: dict) -> str:
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_design_fonts(source: dict, fonts_dir=None) -> None:
    """Refuse substitutions instead of displaying an import with a different face."""
    from cr_renderer.fonts import normalize_family

    manager = load_font_manager(fonts_dir)
    for element in source["elements"]:
        if element["type"] != "text":
            continue
        name = element["font_family"]
        faces = manager._fonts.get(normalize_family(name), [])
        for text in [element, *element.get("style_runs", [])]:
            weight = text.get("font_weight", element.get("font_weight", "regular"))
            italic = text.get("font_style", element.get("font_style", "normal")) == "italic"
            style = (
                "bolditalic"
                if weight == "bold" and italic
                else ("bold" if weight == "bold" else "italic" if italic else "regular")
            )
            if not any(
                face["fontWeight"] == weight and face["fontStyle"] == style for face in faces
            ):
                raise ValueError(
                    "Font face is unavailable in the supplied brand kit: "
                    f"{name}, {weight}, {style}."
                )


def validate_request_id(payload: dict) -> str:
    request_id = payload.get("request_id")
    if (
        not isinstance(request_id, str)
        or not 1 <= len(request_id) <= 128
        or any(ord(char) < 32 for char in request_id)
    ):
        raise ValueError("request_id must be a nonempty string of at most 128 characters.")
    return request_id


def perform_addon_resize(payload: dict) -> dict:
    """Return real full-size previews; model/development caches cannot weaken bounds."""
    from viewer import server

    server.validate_request(payload)
    request_id = validate_request_id(payload)
    source, target = deepcopy(payload["source"]), deepcopy(payload["target_canvas"])
    resources = ResourceContext(server.ASSETS_DIR, server.FONTS_DIR)
    validate_design_fonts(source, resources.fonts_dir)
    execution = server._execution_identity()
    bound = server._RESOURCE_BINDINGS.bind(
        source, resources.assets_dir, resources.fonts_dir, allow_missing=True
    )
    skey = server.source_key(source, resources=bound, execution_identity=execution)
    started = time.perf_counter()
    with server._ENGINE_LOCK, using_resources(resources.assets_dir, resources.fonts_dir):
        server._verify_resize_binding(source, resources, bound, execution, skey, None)
        plan = server._configured_resize(
            deepcopy(source),
            deepcopy(target),
            assets_dir=resources.assets_dir,
            fonts_dir=resources.fonts_dir,
        )
        design = plan["design"]
        violations = check_contract(source, design, target)
        if violations or not strictly_in_bounds(design):
            raise RuntimeError(
                "Output failed the strict stage-one contract: " + "; ".join(map(str, violations))
            )
        source_png, source_warnings = server._render_png(source)
        output_png, output_warnings = server._render_png(design)
        server._verify_resize_binding(source, resources, bound, execution, skey, None)
    return {
        "schema_version": 1,
        "request_id": request_id,
        "source_digest": source_digest(source),
        "target_canvas": target,
        "design": design,
        "source_png": source_png,
        "output_png": output_png,
        "report": {"violations": [], "layout_state": plan["state"]},
        "warnings": list(dict.fromkeys(source_warnings + output_warnings)),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
