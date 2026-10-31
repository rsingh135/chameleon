"""Opt-in strict framing for offline curriculum and research candidates.

This module is not imported by production selection and cannot admit candidates.
It adds restrictions after the real backend contract; it never relaxes it.
Rotated rectangle bounds are conservative geometry, not a visible-ink guarantee:
font overhang, shadows/effects, alpha pixels and renderer-specific strokes still
require native render inspection. No model or prompt decides these checks.
"""

from __future__ import annotations

import math
from numbers import Real

from engine.eval import Context, Violation, build_context, check_contract
from engine.eval.contract import NUMERIC_GEOMETRY, REQUIRED_BY_TYPE, REQUIRED_COMMON
from engine.eval.geometry import canvas_box, contains, rotated_aabb

from .contracts import ValidationError, canonical_json

POLICY_VERSION = "strict-keep-all-in-frame-v1"
MUTABLE_GEOMETRY = frozenset(
    {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}
)
FRAME_EPSILON = 1e-6


def _finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive_number(value) -> bool:
    return _finite_number(value) and value > 0


def _structure(design, *, source: bool) -> list[Violation]:
    name = "source" if source else "candidate"
    if not isinstance(design, dict):
        return [Violation("strict_structure", None, f"{name} must be an object")]
    canvas, elements = design.get("canvas"), design.get("elements")
    if not isinstance(canvas, dict) or any(
        not _positive_number(canvas.get(key)) for key in ("width", "height")
    ):
        return [
            Violation(
                "strict_structure", None, f"{name} canvas must have positive finite dimensions"
            )
        ]
    if not isinstance(elements, list) or not all(isinstance(element, dict) for element in elements):
        return [Violation("strict_structure", None, f"{name} elements must be a list of objects")]
    for element in elements:
        identity = element.get("id")
        if not isinstance(identity, str) or not identity.strip():
            return [
                Violation("strict_structure", None, f"{name} element IDs must be nonempty strings")
            ]
        if source:
            kind = element.get("type")
            if not isinstance(kind, str) or kind not in REQUIRED_BY_TYPE:
                return [
                    Violation("strict_source_invalid", identity, "Unsupported source element type")
                ]
            required = REQUIRED_COMMON + REQUIRED_BY_TYPE[kind]
            if any(key not in element for key in required):
                return [Violation("strict_source_invalid", identity, "Incomplete source element")]
            if any(
                not _finite_number(value)
                for key in NUMERIC_GEOMETRY
                if (value := element.get(key)) is not None
            ) or not all(_positive_number(element[key]) for key in ("width", "height")):
                return [Violation("strict_source_invalid", identity, "Invalid source geometry")]
    if source and len({element["id"] for element in elements}) != len(elements):
        return [Violation("strict_source_invalid", None, "Duplicate source element IDs")]
    return []


def strict_keep_all_in_frame(
    source: dict,
    candidate: dict,
    target_canvas: dict,
    *,
    background_bleed_ids: tuple[str, ...] = (),
    context: Context | None = None,
) -> list[Violation]:
    """Require backend validity, immutable identity/layering and every box in frame.

    All source elements are required, including authored crops. A caller may
    explicitly allow bleed only for named source shape/image backgrounds covering
    the source frame. That allowlist cannot override an existing backend violation.
    There is no policy dictionary accepting unrecognized configuration fields.
    Geometry/type-size changes remain allowed; every other element field is frozen,
    including unknown semantic metadata and the presence of optional fields.
    """
    if (
        not isinstance(target_canvas, dict)
        or set(target_canvas) != {"width", "height"}
        or any(not _positive_number(value) for value in target_canvas.values())
    ):
        return [
            Violation(
                "strict_target_invalid", None, "Target must contain only positive width/height"
            )
        ]
    for design, is_source in ((source, True), (candidate, False)):
        issues = _structure(design, source=is_source)
        if issues:
            return issues
    try:
        # Validate finite, serializable metadata too; equality must distinguish 0/False.
        canonical_json(source)
        canonical_json(candidate)
    except ValidationError:
        return [Violation("strict_structure", None, "Designs must be finite JSON values")]
    if (
        not isinstance(background_bleed_ids, (tuple, list))
        or not all(isinstance(identity, str) for identity in background_bleed_ids)
        or len(set(background_bleed_ids)) != len(background_bleed_ids)
    ):
        raise ValueError("Background bleed IDs must be an explicit list/tuple of unique IDs")
    source_by_id = {element["id"]: element for element in source["elements"]}
    source_frame = canvas_box(source["canvas"])
    for identity in background_bleed_ids:
        element = source_by_id.get(identity)
        if (
            element is None
            or element["type"] not in {"shape", "image"}
            or not contains(rotated_aabb(element), source_frame, slack=FRAME_EPSILON)
        ):
            raise ValueError(
                "Bleed allowlist must identify source full-frame shape/image backgrounds"
            )
    if context is not None and (
        context.source != source or context.by_id != source_by_id or context.canvas != source_frame
    ):
        raise ValueError("Evaluation context must belong to the supplied source")
    try:
        context = context or build_context(source)
        violations = check_contract(source, candidate, target_canvas, context)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return [
            Violation(
                "strict_validation_error",
                None,
                f"Backend validation failed: {type(error).__name__}",
            )
        ]
    if violations:
        return violations
    # The backend already established exact unique IDs; preserve list stacking too.
    if [element["id"] for element in source["elements"]] != [
        element["id"] for element in candidate["elements"]
    ]:
        violations.append(Violation("strict_layer_order", None, "Element list order changed"))
    frame = canvas_box(target_canvas)
    for element in candidate["elements"]:
        identity = element["id"]
        frozen_source = {
            key: value
            for key, value in source_by_id[identity].items()
            if key not in MUTABLE_GEOMETRY
        }
        frozen_candidate = {
            key: value for key, value in element.items() if key not in MUTABLE_GEOMETRY
        }
        if canonical_json(frozen_source) != canonical_json(frozen_candidate):
            violations.append(
                Violation("strict_identity_changed", identity, "Non-geometry fields changed")
            )
        if identity not in background_bleed_ids and not contains(
            frame, rotated_aabb(element), slack=FRAME_EPSILON
        ):
            violations.append(
                Violation(
                    "strict_out_of_frame", identity, "Rotated element bounds leave target frame"
                )
            )
    return violations
