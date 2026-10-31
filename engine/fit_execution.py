"""Native document constraints followed by the actual trained fit policy.

The caller supplies IDs produced by the trusted fitted-column generator. This
document profile measures native text frames, identity, groups and bounds; it
does not claim the marketing-photo pixel/occlusion profile was evaluated.
"""

from __future__ import annotations

import math
from copy import deepcopy

PROFILE = "measured_document_columns"
_EPS = 0.02


def _candidate_id(design):
    from engine.fit_policy import digest

    return "candidate-" + digest(design)


def _common_scale(source, design):
    original = {e["id"]: e for e in source["elements"]}
    scales = [
        e["font_size"] / original[e["id"]]["font_size"]
        for e in design["elements"]
        if e["type"] == "text"
    ]
    if not scales or not all(math.isfinite(s) and s > 0 for s in scales):
        return None
    scale = scales[0]
    return scale if all(math.isclose(s, scale, rel_tol=1e-7) for s in scales) else None


def _group_failures(source, design, scale):
    changed = {e["id"]: e for e in design["elements"]}
    groups = {}
    for element in source["elements"]:
        if element.get("group_id"):
            groups.setdefault(element["group_id"], []).append(element)
    failed = []
    for name, members in groups.items():
        first = members[0]
        current = changed[first["id"]]
        local = (
            scale
            if any(e["type"] == "text" for e in members)
            else current["width"] / first["width"]
        )
        dx, dy = current["x"] - first["x"] * local, current["y"] - first["y"] * local
        for old in members:
            new = changed[old["id"]]
            expected = {"x": old["x"] * local + dx, "y": old["y"] * local + dy}
            expected.update(
                {
                    key: old[key] * local
                    for key in ("width", "height", "font_size", "stroke_width", "corner_radius")
                    if key in old
                }
            )
            if any(
                not math.isclose(new[key], value, rel_tol=1e-7, abs_tol=_EPS)
                for key, value in expected.items()
            ):
                failed.append(str(name))
                break
    return failed


def _new_overlaps(source, design):
    from engine.eval.geometry import overlap_area, rotated_aabb

    originals = {e["id"]: e for e in source["elements"]}
    texts = [e for e in design["elements"] if e["type"] == "text"]
    pairs = []
    for index, first in enumerate(texts):
        for second in texts[index + 1 :]:
            old_a, old_b = originals[first["id"]], originals[second["id"]]
            if old_a.get("group_id") and old_a.get("group_id") == old_b.get("group_id"):
                continue
            if overlap_area(rotated_aabb(old_a), rotated_aabb(old_b)) > _EPS:
                continue
            if overlap_area(rotated_aabb(first), rotated_aabb(second)) > _EPS:
                pairs.append([first["id"], second["id"]])
    return pairs


def _assess(source, target, design, context):
    from engine.eval import check_contract
    from engine.layout.text import get_measurer
    from engine.stage_one import strictly_in_bounds

    violations = check_contract(source, design, target_canvas=target, context=context)
    failures = [str(v) for v in violations]
    if not strictly_in_bounds(design):
        failures.append("strict_document_bounds")
    scale, overlaps, group_failures, minimum = None, [], [], None
    if not failures:
        measurer = get_measurer()
        if measurer.backend != "skia":
            failures.append("native_text_measurer_unavailable")
        else:
            for element in design["elements"]:
                if element["type"] != "text":
                    continue
                measured = measurer.measure(element, element["font_size"], element["width"])
                if (
                    measured.width > element["width"] + _EPS
                    or measured.height > element["height"] + _EPS
                    or (
                        element.get("max_lines") is not None
                        and measured.lines > element["max_lines"]
                    )
                ):
                    failures.append("native_text_fit:" + element["id"])
        scale = _common_scale(source, design)
        if scale is None:
            failures.append("common_font_scale")
        else:
            group_failures = _group_failures(source, design, scale)
            failures.extend("explicit_group_similarity:" + name for name in group_failures)
        overlaps = _new_overlaps(source, design)
        failures.extend("new_text_frame_overlap:" + "/".join(pair) for pair in overlaps)
        minimum = min(
            (e["font_size"] for e in design["elements"] if e["type"] == "text"), default=None
        )
    return {
        "failures": failures,
        "minimum_font": minimum,
        "common_font_scale": scale,
        "new_text_frame_overlaps": overlaps,
        "explicit_group_failures": group_failures,
    }


def select_fitted_document(source, target, candidates, fit_candidate_ids, units, *, snapshot=None):
    """Select a known fitted document through native constraints and trained weights.

    ``fit_candidate_ids`` is an internal capability supplied by the trusted
    generator, never IDs from a user/model payload. Model absence or invalid
    decisions decline this route and let the caller use its ordinary fallback.
    """
    from engine.eval import build_context
    from engine.fit_policy import build_fit_menu, choose_fit_plan
    from engine.resources import get_resources

    if not fit_candidate_ids:
        return None
    resources = get_resources()
    context = build_context(source, resources.assets_dir, resources.fonts_dir)
    rows, seen = [], set()
    for design in candidates:
        identity = _candidate_id(design)
        if identity not in fit_candidate_ids or identity in seen:
            continue
        seen.add(identity)
        assessment = _assess(source, target, design, context)
        rows.append({"candidate_id": identity, "design": design, "assessment": assessment})
    valid = [row for row in rows if not row["assessment"]["failures"]]
    if not valid:
        return None
    best_font = max(row["assessment"]["minimum_font"] for row in valid)
    admitted = [row for row in valid if row["assessment"]["minimum_font"] >= best_font * 0.9]
    choices = [
        {"plan_id": row["candidate_id"], "design": row["design"], "hard_valid": True}
        for row in admitted
    ]
    try:
        menu = build_fit_menu(source, target, choices, units=units)
        result = choose_fit_plan(source, target, menu, choices)
        decision, receipt = result.get("decision"), result.get("receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("executed") is not True
            or not isinstance(decision, dict)
            or set(decision) - {"schema_version", "menu_digest", "plan_id", "rationale"}
            or type(decision.get("schema_version")) is not int
            or decision["schema_version"] != 1
            or decision.get("menu_digest") != menu["menu_digest"]
            or decision.get("plan_id") not in {row["plan_id"] for row in choices}
        ):
            return None
    except (ValueError, OSError, KeyError, TypeError, OverflowError):
        return None
    selected = next(row for row in admitted if row["candidate_id"] == decision["plan_id"])
    trace = {
        "profile": PROFILE,
        "admission_basis": (
            "native_contract_strict_bounds_measured_text_common_type_scale_"
            "groups_no_new_text_overlap"
        ),
        "policy_executed": True,
        "candidate_ids": [row["candidate_id"] for row in rows],
        "admitted_candidate_ids": [row["candidate_id"] for row in admitted],
        "selected_candidate_id": selected["candidate_id"],
        "minimum_font_floor": best_font * 0.9,
        "fit_policy": deepcopy(result),
        "quality_improved": None,
        "marketing_pixel_profile_evaluated": False,
    }
    if snapshot is not None:
        admitted_ids = {row["candidate_id"] for row in admitted}
        snapshot["candidates"] = [
            {
                "candidate_id": row["candidate_id"],
                "design": deepcopy(row["design"]),
                "contract_valid": not row["assessment"]["failures"],
                "admitted": row["candidate_id"] in admitted_ids,
                "diagnostics": {
                    "constraint_profile": PROFILE,
                    **deepcopy(row["assessment"]),
                    "evaluation": {
                        "total": None,
                        "metrics": {},
                        "violations": [
                            {"code": "document_constraint", "message": message}
                            for message in row["assessment"]["failures"]
                        ],
                    },
                },
                "selection": {
                    "safety": [not row["assessment"]["failures"]],
                    "fidelity": None,
                    "typography": [row["assessment"]["minimum_font"]],
                },
            }
            for row in rows
        ]
        snapshot["fit_execution"] = deepcopy(trace)
    return {"design": deepcopy(selected["design"]), "trace": trace}
