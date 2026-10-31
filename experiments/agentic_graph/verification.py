"""Deterministic evidence for the graph's verifier, without model authority.

``passed`` covers the hard contract and native render only. Local quality
proxies remain diagnostics; the caller must apply its explicit task criteria.
No source, asset, design or output file is written by this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from copy import deepcopy
from numbers import Real

from PIL import Image

from engine.eval import build_context, check_contract, score
from engine.eval.contract import FROZEN_FIELDS, REQUIRED_BY_TYPE, REQUIRED_COMMON
from engine.eval.metrics import MIN_FONT_FRACTION, MIN_FONT_PX
from engine.eval.occlusion import conservative_occlusion
from engine.eval.render import render_plate
from engine.resources import (
    get_renderer,
    load_font_manager,
    native_render_example,
    render_input,
    using_resources,
)
from engine.stage_one import strictly_in_bounds
from engine.visual_plan import _resources
from renderer.converter.convert import convert_chameleon_to_crello

MAX_ELEMENTS = 256
MAX_SIDE = 8192
MAX_PIXELS = 20_000_000
_DEFAULTS = {
    "font_weight": "regular",
    "font_style": "normal",
    "text_align": "left",
    "line_height": 1.2,
    "letter_spacing": 0,
    "style_runs": [],
    "fit": "contain",
    "group_id": None,
    "z_index": 0,
    "opacity": 1,
    "rotation": 0,
}


def _check(code, status, message, element_ids=(), evidence=None, *, scope="hard"):
    return {
        "code": code,
        "status": status,
        "scope": scope,
        "element_ids": sorted(set(element_ids)),
        "message": str(message),
        "evidence": evidence or {},
    }


def _finish(checks, render=None, diagnostics=None, quality=None):
    hard = [row for row in checks if row["scope"] == "hard"]
    failed = any(row["status"] == "failed" for row in hard)
    passed = bool(hard) and all(row["status"] == "passed" for row in hard)
    return {
        "schema_version": 1,
        "status": "failed" if failed else "passed" if passed else "unverified",
        "hard_pass": passed,
        "acceptance_scope": "hard_contract_and_native_render_only",
        "checks": checks,
        "failures": [row for row in hard if row["status"] in {"failed", "unverified"}],
        "render": render or {"status": "skipped"},
        "diagnostics": diagnostics or {"scope": "local_quality_proxies", "metrics": {}},
        "quality": quality or {"status": "unverified", "checks": []},
    }


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _canvas(canvas):
    return (
        isinstance(canvas, dict)
        and all(
            type(canvas.get(key)) is int and 0 < canvas[key] <= MAX_SIDE
            for key in ("width", "height")
        )
        and canvas["width"] * canvas["height"] <= MAX_PIXELS
    )


def _structure(design, label):
    if not isinstance(design, dict) or not _canvas(design.get("canvas")):
        return [_check("invalid_structure", "failed", f"{label} requires a bounded pixel canvas.")]
    elements = design.get("elements")
    if not isinstance(elements, list) or not 1 <= len(elements) <= MAX_ELEMENTS:
        return [_check("invalid_structure", "failed", f"{label} requires 1 to 256 elements.")]
    checks = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            checks.append(
                _check("invalid_structure", "failed", f"{label} element {index} is not an object.")
            )
            continue
        eid = element.get("id")
        ids = [eid] if isinstance(eid, str) else []
        kind = element.get("type")
        if not isinstance(kind, str):
            checks.append(
                _check(
                    "invalid_structure", "failed", f"{label} requires a known element type.", ids
                )
            )
            continue
        missing = [key for key in REQUIRED_COMMON if key not in element]
        missing += [key for key in REQUIRED_BY_TYPE.get(kind, ()) if key not in element]
        if missing:
            checks.append(
                _check("missing_field", "failed", f"{label} element is missing {missing}.", ids)
            )
            continue
        if not isinstance(eid, str) or not eid or kind not in REQUIRED_BY_TYPE:
            checks.append(
                _check(
                    "invalid_structure",
                    "failed",
                    f"{label} requires string IDs and known element types.",
                    ids,
                )
            )
            continue
        fields = {
            "x",
            "y",
            "width",
            "height",
            "font_size",
            "rotation",
            "corner_radius",
            "stroke_width",
        }
        invalid = [key for key in fields & element.keys() if not _finite(element[key])]
        if invalid:
            checks.append(
                _check(
                    "invalid_geometry",
                    "failed",
                    f"{label} has nonfinite/non-numeric geometry: {sorted(invalid)}.",
                    ids,
                )
            )
        elif min(element["width"], element["height"]) <= 0:
            checks.append(
                _check("degenerate_size", "failed", "Element dimensions must remain positive.", ids)
            )
        if kind == "text" and not all(
            isinstance(element[key], str) for key in ("content", "font_family", "color")
        ):
            checks.append(
                _check(
                    "invalid_structure",
                    "failed",
                    "Text content, font family and color must be strings.",
                    ids,
                )
            )
        if kind == "text" and (
            not isinstance(element.get("style_runs", []), list)
            or not all(isinstance(run, dict) for run in element.get("style_runs", []))
        ):
            checks.append(
                _check("invalid_structure", "failed", "Text style_runs must be object ranges.", ids)
            )
        if kind == "image" and not isinstance(element["asset_id"], str):
            checks.append(
                _check(
                    "invalid_structure",
                    "failed",
                    "Image asset_id must be a local resource name.",
                    ids,
                )
            )
    return checks


def _font_checks(design):
    from cr_renderer.fonts import normalize_family

    manager = load_font_manager()
    checks = []
    for element in design["elements"]:
        if element["type"] != "text":
            continue
        family = element["font_family"]
        faces = manager._fonts.get(normalize_family(family), [])
        for run in [element, *element.get("style_runs", [])]:
            weight = run.get("font_weight", element.get("font_weight", "regular")) or "regular"
            italic = run.get("font_style", element.get("font_style", "normal")) == "italic"
            style = (
                "bolditalic"
                if weight == "bold" and italic
                else ("bold" if weight == "bold" else "italic" if italic else "regular")
            )
            if not any(
                face["fontWeight"] == weight and face["fontStyle"] == style for face in faces
            ):
                checks.append(
                    _check(
                        "font_face_unavailable",
                        "failed",
                        "Supply the exact font resource; changing family or silently "
                        "substituting a face is not permitted.",
                        [element["id"]],
                        {"font_family": family, "weight": weight, "style": style},
                    )
                )
                break
    return checks


def _implicit_identity(source, design):
    """An omitted opacity/fit/etc. is a default, not permission to hide or crop."""
    old = {element["id"]: element for element in source["elements"]}
    checks = []
    for element in design["elements"]:
        original = old.get(element["id"])
        if original is None:
            continue
        for field in FROZEN_FIELDS:
            if field in original or field not in element:
                continue
            if field in _DEFAULTS and element[field] == _DEFAULTS[field]:
                continue
            checks.append(
                _check(
                    "frozen_field_added",
                    "failed",
                    f"Restore the source/default {field}; identity fields may not be invented.",
                    [element["id"]],
                    {"field": field},
                )
            )
    return checks


def _native_png(design):
    from engine.resources import get_resources

    resources = get_resources()
    renderer = get_renderer(resources.fonts_dir)
    example, warnings = convert_chameleon_to_crello(
        render_input(design), renderer.font_manager, str(resources.assets_dir)
    )
    example = native_render_example(example, design)
    png = renderer.render(example, max_size=max(design["canvas"].values()), format="png")
    with Image.open(io.BytesIO(png)) as decoded:
        decoded.load()
        if decoded.format != "PNG" or decoded.size != tuple(
            design["canvas"][key] for key in ("width", "height")
        ):
            raise ValueError("Native preview format or dimensions differ from its JSON canvas.")
    return png, warnings


def _quality_checks(source, design, metrics, occlusion):
    texts = [element for element in design["elements"] if element["type"] == "text"]
    ids = [element["id"] for element in texts]
    collisions = metrics.get("raw_collisions")
    pairs = occlusion.get("bounded_pairs", [])
    collision_ids = [eid for pair in pairs for eid in pair.get("element_ids", [])]
    rows = [
        _check(
            "unwanted_collisions",
            "passed" if collisions == 0 else "warning",
            "Source-relative occupied-content collisions are a local proxy, not a jury judgment.",
            collision_ids,
            {"raw_collisions": collisions, "occlusion": occlusion},
            scope="diagnostic",
        )
    ]
    minimum = max(MIN_FONT_PX, MIN_FONT_FRACTION * min(design["canvas"].values()))
    tiny = [element["id"] for element in texts if element["font_size"] < minimum]
    rows.append(
        _check(
            "text_readability",
            "skipped" if not texts else "warning" if tiny else "passed",
            "Absolute font-size diagnostic; compare source hierarchy and "
            "task-specific readability limits.",
            tiny,
            {"raw_min_font_px": metrics.get("raw_min_font_px"), "diagnostic_minimum_px": minimum},
            scope="diagnostic",
        )
    )
    contrast = metrics.get("raw_min_contrast")
    rows.append(
        _check(
            "rendered_contrast",
            "skipped"
            if not texts
            else "unverified"
            if contrast is None
            else "warning"
            if contrast < 4
            else "passed",
            "Rendered-plate contrast is a mean-region proxy; unsupported color syntax "
            "leaves it unverified.",
            ids,
            {"raw_min_contrast": contrast, "region_sampling": "mean luminance under text frames"},
            scope="diagnostic",
        )
    )
    busyness = metrics.get("raw_max_busyness")
    rows.append(
        _check(
            "rendered_busyness",
            "skipped"
            if not texts
            else "unverified"
            if busyness is None
            else "warning"
            if busyness >= 0.25
            else "passed",
            "Rendered background edges are a local busyness proxy; "
            "inspect actual source/result PNGs.",
            ids,
            {"raw_max_busyness": busyness},
            scope="diagnostic",
        )
    )
    rows.append(
        _check(
            "source_hierarchy",
            "passed" if metrics.get("hierarchy") == 1 else "warning",
            "Preserve source font-rank relationships while fitting the target.",
            ids,
            {
                "hierarchy": metrics.get("hierarchy"),
                "group_integrity": metrics.get("group_integrity"),
            },
            scope="diagnostic",
        )
    )
    return rows


def verify_design(source, design, target, *, assets_dir=None, fonts_dir=None):
    """Return JSON evidence and actionable failures, never a model approval.

    The graph applies its explicit fundamental-task acceptance policy to
    ``quality.checks``/``diagnostics.metrics``. Valid JSON/native rendering alone
    does not establish readability, faithful layout, or learned visual reasoning.
    """
    checks = []
    try:
        source, design, target = deepcopy(source), deepcopy(design), deepcopy(target)
        for value in (source, design, target):
            json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        return _finish([_check("invalid_json", "failed", str(exc))])
    if not _canvas(target):
        return _finish(
            [_check("invalid_target", "failed", "Target requires a bounded pixel canvas.")]
        )
    checks.extend(_structure(source, "Source"))
    checks.extend(_structure(design, "Candidate"))
    if checks:
        return _finish(checks)
    if len({element["id"] for element in source["elements"]}) != len(source["elements"]):
        return _finish([_check("invalid_source", "failed", "Source IDs must be unique.")])
    try:
        with using_resources(assets_dir, fonts_dir):
            bound = _resources(source)
            checks.extend(_font_checks(source))
            checks.extend(_font_checks(design))
            if checks:
                return _finish(checks)
            context = build_context(source)
            violations = check_contract(source, design, target, context)
            checks.extend(
                _check(row.code, "failed", row.message, [row.element_id] if row.element_id else [])
                for row in violations
            )
            checks.extend(_implicit_identity(source, design))
            for element in design["elements"]:
                if not strictly_in_bounds({"canvas": target, "elements": [element]}):
                    checks.append(
                        _check(
                            "strict_bounds",
                            "failed",
                            "Move or scale the complete rotated element inside the target canvas.",
                            [element["id"]],
                        )
                    )
            if checks:
                return _finish(checks)
            checks.extend(
                [
                    _check(
                        "contract",
                        "passed",
                        "Immutable elements, positive geometry, proportions and "
                        "native glyph fit passed.",
                    ),
                    _check(
                        "strict_bounds",
                        "passed",
                        "Every rotated element is inside the requested target.",
                    ),
                    _check(
                        "font_resources",
                        "passed",
                        "Requested font faces are available; no substitution was accepted.",
                    ),
                ]
            )
            try:
                source_png, source_warnings = _native_png(source)
                output_png, output_warnings = _native_png(design)
            except Exception as exc:
                checks.append(
                    _check(
                        "native_render",
                        "unverified",
                        f"Native preview could not be verified: {type(exc).__name__}: {exc}",
                    )
                )
                return _finish(checks, {"status": "unverified"})
            render = {
                "status": "passed",
                "width": target["width"],
                "height": target["height"],
                "png_sha256": hashlib.sha256(output_png).hexdigest(),
                "source_png_sha256": hashlib.sha256(source_png).hexdigest(),
                "warnings": list(dict.fromkeys(source_warnings + output_warnings)),
            }
            checks.append(
                _check(
                    "native_render",
                    "passed",
                    "Real source/result PNGs decode at exact canvas dimensions.",
                    evidence=render,
                )
            )
            try:
                plate = render_plate(design, max_size=max(target.values()))
                evaluation = score(source, design, context, plate, target)
                occlusion = conservative_occlusion(source, design)
                quality_checks = _quality_checks(source, design, evaluation["metrics"], occlusion)
                diagnostics = {
                    "scope": "local_quality_proxies",
                    "metrics": evaluation["metrics"],
                    "total": evaluation["total"],
                    "occlusion": occlusion,
                    "source_notes": context.notes,
                }
                quality = {"status": "diagnostic_only", "checks": quality_checks}
            except Exception as exc:
                diagnostics = {
                    "scope": "local_quality_proxies",
                    "metrics": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
                quality = {"status": "unverified", "checks": []}
            if _resources(source) != bound:
                checks.append(
                    _check(
                        "resource_mutation",
                        "failed",
                        "Source resource bytes/presence changed during verification; "
                        "bind fresh resources and retry.",
                    )
                )
            else:
                checks.append(
                    _check(
                        "resource_integrity",
                        "passed",
                        "Resource bytes and missing-asset presence remained unchanged.",
                    )
                )
            return _finish(checks, render, diagnostics, quality)
    except Exception as exc:
        checks.append(
            _check(
                "verification_unavailable",
                "unverified",
                f"Verification evidence is unavailable: {type(exc).__name__}: {exc}",
            )
        )
        return _finish(checks)
