"""Admit source-purpose layouts, then rank their geometry with the fit policy.

Purpose/anchor authority comes from source evidence and deterministic constraints.
The small trained ranker judges geometry only. This profile does not claim a
full marketing pixel review or learned visual understanding of element purpose.
"""

from __future__ import annotations

import math
from copy import deepcopy

PROFILE = "source_semantic_anchors"


def _candidate_id(design):
    from engine.fit_policy import digest

    return "candidate-" + digest(design)


def _projected_support(source, design, semantics, scale):
    from engine.layout.geometry import Box

    originals = {e["id"]: e for e in source["elements"]}
    outputs = {e["id"]: e for e in design["elements"]}
    rows = []
    for descriptor in semantics["elements"]:
        if descriptor["purpose"] in {"background", "decoration"}:
            continue
        key = descriptor["element_id"]
        old, new = originals[key], outputs[key]
        x, y, width, height = descriptor["visible_box"]
        rows.append(
            (
                key,
                Box(x, y, width, height),
                Box(
                    new["x"] + (x - old["x"]) * scale,
                    new["y"] + (y - old["y"]) * scale,
                    width * scale,
                    height * scale,
                ),
            )
        )
    return rows


def _assess(source, target, design, semantics, context, resources):
    from engine.design_semantics import audit_semantic_preservation
    from engine.eval import check_contract
    from engine.fit_execution import _common_scale, _group_failures
    from engine.layout.geometry import intersection
    from engine.layout.text import get_measurer
    from engine.stage_one import strictly_in_bounds

    failures = [str(v) for v in check_contract(source, design, target, context=context)]
    if not strictly_in_bounds(design):
        failures.append("strict_semantic_bounds")
    scale = _common_scale(source, design) if not failures else None
    if scale is None:
        failures.append("common_foreground_font_scale")
    semantic_audit, overlaps, minimum = None, [], None
    if not failures:
        originals = {e["id"]: e for e in source["elements"]}
        backgrounds = {
            row["element_id"]
            for row in semantics["elements"]
            if row["purpose"] in {"background", "decoration"}
        }
        # Only the generator's complete similarity transforms may use projected
        # source ink. Arbitrary reflow/crop candidates take the ordinary profile.
        for element in design["elements"]:
            if element["id"] in backgrounds:
                continue
            for key in ("width", "height"):
                if not math.isclose(
                    element[key], originals[element["id"]][key] * scale, rel_tol=1e-7, abs_tol=0.02
                ):
                    failures.append("nonuniform_semantic_foreground:" + element["id"])
                    break
        failures.extend(
            "explicit_group_similarity:" + name for name in _group_failures(source, design, scale)
        )
        measurer = get_measurer()
        if measurer.backend != "skia":
            failures.append("native_text_measurer_unavailable")
        else:
            for element in design["elements"]:
                if element["type"] != "text":
                    continue
                measured = measurer.measure(element, element["font_size"], element["width"])
                if (
                    measured.width > element["width"] + 0.02
                    or measured.height > element["height"] + 0.02
                    or element.get("max_lines") is not None
                    and measured.lines > element["max_lines"]
                ):
                    failures.append("native_text_fit:" + element["id"])
        if not failures:
            semantic_audit = audit_semantic_preservation(
                source,
                design,
                semantics,
                assets_dir=resources.assets_dir,
                fonts_dir=resources.fonts_dir,
            )
            for key in ("corner_anchor_retention", "footer_anchor_retention"):
                if semantic_audit["metrics"][key] < 0.8:
                    failures.append(key)
            for key in ("explicit_group_retention", "reading_order_retention"):
                if semantic_audit["metrics"][key] < 1 - 1e-6:
                    failures.append(key)
            projected = _projected_support(source, design, semantics, scale)
            for index, (first, old_a, new_a) in enumerate(projected):
                for second, old_b, new_b in projected[index + 1 :]:
                    old = intersection(old_a, old_b) / max(
                        1e-9, min(old_a.width * old_a.height, old_b.width * old_b.height)
                    )
                    new = intersection(new_a, new_b) / max(
                        1e-9, min(new_a.width * new_a.height, new_b.width * new_b.height)
                    )
                    if new > old + 0.02:
                        overlaps.append([first, second])
            failures.extend("new_projected_ink_overlap:" + "/".join(pair) for pair in overlaps)
        minimum = min(
            (e["font_size"] for e in design["elements"] if e["type"] == "text"), default=0
        )
    return {
        "failures": failures,
        "minimum_font": minimum,
        "common_foreground_scale": scale,
        "semantic_audit": semantic_audit,
        "projected_ink_overlap_pairs": overlaps,
        "support_provenance": "native_source_support_projected_through_validated_similarity",
    }


def select_semantic_layout(
    source,
    target,
    candidates,
    semantic_candidate_ids,
    units=None,
    *,
    snapshot=None,
    semantics=None,
    model_path=None,
):
    """Use only generator-certified candidate IDs and preserve honest model roles."""
    from engine.design_semantics import describe_design
    from engine.eval import build_context
    from engine.fit_policy import build_fit_menu, choose_fit_plan
    from engine.layout.semantic_anchors import _descriptors
    from engine.resources import get_resources

    if not semantic_candidate_ids:
        return None
    resources = get_resources()
    if semantics is None:
        semantics = describe_design(
            source, assets_dir=resources.assets_dir, fonts_dir=resources.fonts_dir
        )
    _descriptors(source, semantics)
    if not any(row["anchors"] for row in semantics["elements"]):
        return None
    context = build_context(source, resources.assets_dir, resources.fonts_dir)
    rows, seen = [], set()
    for design in candidates:
        identity = _candidate_id(design)
        if identity not in semantic_candidate_ids or identity in seen:
            continue
        seen.add(identity)
        rows.append(
            {
                "candidate_id": identity,
                "design": design,
                "assessment": _assess(source, target, design, semantics, context, resources),
            }
        )
    valid = [row for row in rows if not row["assessment"]["failures"]]
    if not valid:
        return None
    best_font = max(row["assessment"]["minimum_font"] for row in valid)
    admitted = [row for row in valid if row["assessment"]["minimum_font"] >= best_font * 0.9]
    selected = max(admitted, key=lambda row: row["assessment"]["minimum_font"])
    receipt = {
        "executed": False,
        "kind": "trained_pairwise_fit_policy",
        "status": "no_choice",
        "choice_available": len(admitted) > 1,
    }
    if len(admitted) > 1:
        choices = [
            {"plan_id": row["candidate_id"], "design": row["design"], "hard_valid": True}
            for row in admitted
        ]
        try:
            menu = build_fit_menu(source, target, choices, units=units)
            result = choose_fit_plan(source, target, menu, choices, model_path)
            decision, receipt = result.get("decision"), result.get("receipt")
            if decision is not None:
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("executed") is not True
                    or not isinstance(decision, dict)
                    or decision.get("schema_version") != 1
                    or decision.get("menu_digest") != menu["menu_digest"]
                    or decision.get("plan_id") not in {row["plan_id"] for row in choices}
                    or set(decision) - {"schema_version", "menu_digest", "plan_id", "rationale"}
                ):
                    return None
                selected = next(
                    row for row in admitted if row["candidate_id"] == decision["plan_id"]
                )
            elif not isinstance(receipt, dict) or receipt.get("executed") is not False:
                return None
        except (ValueError, OSError, KeyError, TypeError, OverflowError):
            return None
    profile = {
        "profile": PROFILE,
        "semantic_digest": semantics["semantic_digest"],
        "purpose_authority": "source_evidence_and_deterministic_constraints",
        "learned_purpose_claim": False,
        "policy_executed": receipt.get("executed") is True,
        "choice_available": len(admitted) > 1,
        "candidate_ids": [row["candidate_id"] for row in rows],
        "admitted_candidate_ids": [row["candidate_id"] for row in admitted],
        "selected_candidate_id": selected["candidate_id"],
        "minimum_font_floor": best_font * 0.9,
        "native_text_fit_evaluated": True,
        "projected_source_ink_evaluated": True,
        "marketing_pixel_profile_evaluated": False,
        "quality_improved": None,
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
                    "evaluation": {"total": None, "metrics": {}, "violations": []},
                },
                "selection": {"profile": PROFILE},
                "provenance": [],
            }
            for row in rows
        ]
        snapshot["admitted_finalist_ids"] = [row["candidate_id"] for row in admitted]
        snapshot["baseline_candidate_id"] = selected["candidate_id"]
        snapshot["semantic_execution"] = deepcopy(profile)
        snapshot["fit_policy"] = deepcopy(receipt)
    return {
        "design": deepcopy(selected["design"]),
        "trace": {"semantic_execution": profile, "fit_policy": deepcopy(receipt)},
    }
