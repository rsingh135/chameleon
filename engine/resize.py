"""Offline, deterministic design reflow with source-aware candidate selection."""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.layout import build_context, generate_candidates, get_measurer
from engine.layout.backgrounds import generate_background_states
from engine.layout.composition import generate_composition_states
from engine.layout.fitted_columns import generate_fitted_column_states
from engine.layout.focal import analyze_focal, infer_focal_profile
from engine.layout.focal_states import generate_focal_states
from engine.layout.geometry import bounds, intersection
from engine.layout.grouped import GroupedState, generate_grouped_states
from engine.layout.reading_order import analyze_reading_order, infer_reading_order
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.layout.rules import analyze_rules
from engine.layout.semantic_anchors import generate_semantic_anchor_states
from engine.layout.source_guides import _transform, generate_source_guides
from engine.resources import get_resources, using_resources
from engine.scene import analyze_scene, background_coverage
from engine.scene_model import constrain_candidates, validate_model_response


@dataclass(frozen=True)
class _CandidateAssessment:
    design: dict
    evaluation: dict
    style: dict
    relationships: dict
    proportions: dict
    focal: dict
    rules: dict
    reading_order: dict
    occlusion: dict


def _quality(source: dict, candidate: dict, context) -> float:
    """Typography tie-breaker and a dependency-free development fallback."""
    original = {e["id"]: e for e in source["elements"]}
    source_scale = math.sqrt(
        candidate["canvas"]["width"]
        * candidate["canvas"]["height"]
        / (source["canvas"]["width"] * source["canvas"]["height"])
    )
    background_ids = {e["id"] for e in context.backgrounds + context.decorations}
    unit_ids = {e["id"]: i for i, u in enumerate(context.units) for e in u.elements}
    roles = {e["id"]: u.role for u in context.units for e in u.elements}
    elements = [e for e in candidate["elements"] if e["id"] not in background_ids]
    score = 0.0
    for i, element in enumerate(elements):
        box = bounds(element)
        if element["type"] == "text":
            measured = get_measurer().measure(element, element["font_size"], element["width"])
            overflow = max(0.0, measured.width - element["width"])
            overflow += max(0.0, measured.height - element["height"])
            score -= overflow * 100
            relative = element["font_size"] / max(
                1.0, original[element["id"]]["font_size"] * source_scale
            )
            weight = {"headline": 3.0, "body": 2.0, "legal": 0.5}.get(roles[element["id"]], 1.0)
            score += weight * math.log(max(0.01, relative))
        for other in elements[i + 1 :]:
            if unit_ids[element["id"]] == unit_ids[other["id"]]:
                continue
            overlap = intersection(box, bounds(other))
            if overlap <= 0:
                continue
            source_overlap = intersection(
                bounds(original[element["id"]]), bounds(original[other["id"]])
            )
            intentional = source_overlap > 0 and not (element["type"] == other["type"] == "text")
            if not intentional:
                fraction = overlap / max(
                    1.0, min(box.width * box.height, other["width"] * other["height"])
                )
                score -= fraction * 20
    return score


def _safe_identity(source: dict, target: dict) -> bool:
    """Keep an unchanged canvas only when its original content already fits."""
    if source["canvas"] != target:
        return False
    for element in source["elements"]:
        box = bounds(element)
        if (
            box.x < 0
            or box.y < 0
            or box.x + box.width > target["width"]
            or box.y + box.height > target["height"]
        ):
            return False
        if element["type"] == "text":
            measured = get_measurer().measure(element, element["font_size"], element["width"])
            if (
                element["font_size"] < 1
                or measured.width > element["width"] + 1e-6
                or measured.height > element["height"] + 1e-6
                or (element.get("max_lines") is not None and measured.lines > element["max_lines"])
            ):
                return False
    return True


def _resolution_state(source, target):
    """A valid proportional request changes resolution, not composition.

    Keep every permitted pixel geometry field at one scale, including authored
    decorative clipping. Cropped source images and invalid text fits decline
    this route; the ordinary constrained composition search remains available.
    Integer raster dimensions may leave at most half a pixel after uniform fit;
    center that remainder without stretching. This bounds rounding in pixels,
    not relative aspect error or an inferred requested scale at tiny resolutions.
    """
    dimensions = [source["canvas"][key] for key in ("width", "height")]
    dimensions += [target[key] for key in ("width", "height")]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in dimensions
    ):
        return None
    source_width, source_height, width, height = dimensions
    exact_aspect = source_width * height == source_height * width
    scale = width / source_width
    dx = dy = 0.0
    if not exact_aspect:
        scale = min(scale, height / source_height)
        remaining_width = max(0.0, width - source_width * scale)
        remaining_height = max(0.0, height - source_height * scale)
        if max(remaining_width, remaining_height) > 0.5 + 1e-9:
            return None
        dx, dy = remaining_width / 2, remaining_height / 2
    if not math.isfinite(scale) or scale <= 0:
        return None
    for element in source["elements"]:
        if any(
            isinstance(element[field], bool)
            or not isinstance(element[field], (int, float))
            or not math.isfinite(element[field])
            for field in ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width")
            if field in element
        ):
            return None
    try:
        from engine.eval import check_contract
        from engine.eval.geometry import canvas_box, contains, rotated_aabb

        canvas = canvas_box(source["canvas"])
        if any(
            element["type"] == "image" and not contains(canvas, rotated_aabb(element))
            for element in source["elements"]
        ):
            return None
        design = deepcopy(source)
        design["canvas"] = dict(target)
        if scale != 1.0 or dx or dy:
            design["elements"] = _transform(source["elements"], scale, dx, dy)
        if check_contract(source, design, target_canvas=target):
            return None
    except (ImportError, OSError, RuntimeError, ValueError):
        return None
    identity = source["canvas"] == target
    return GroupedState(
        "source" if identity else "resolution-scale",
        "source" if identity else "source-similarity",
        (),
        json.dumps(design, sort_keys=True, separators=(",", ":")),
        () if exact_aspect else (f"resolution-rounding-padding:{dx:.9g},{dy:.9g}",),
    )


def _readability(candidate: dict, context, metrics: dict) -> float:
    """Reach a readable floor, then compare hierarchy rather than inflate labels.

    Genuine source footer fine print has a separate, explicit floor. This does
    not change official metrics or UI warnings: all text under 10 px remains
    reported. Small body text elsewhere is still primary content.
    """
    from engine.eval.metrics import MIN_FONT_FRACTION, MIN_FONT_PX

    texts = [element for element in candidate["elements"] if element["type"] == "text"]
    if not texts:
        return metrics.get("text_size", 1.0)
    fine_print = {
        element["id"]
        for unit in context.units
        if unit.role == "legal" and unit.box.y + unit.box.height / 2 >= context.height * 0.85
        for element in unit.text
    }
    threshold = max(MIN_FONT_PX, MIN_FONT_FRACTION * min(candidate["canvas"].values()))
    return min(
        1.0,
        min(
            element["font_size"]
            / (max(8.0, threshold * 0.8) if element["id"] in fine_print else threshold)
            for element in texts
        ),
    )


def _evaluation_snapshot(result):
    return {
        "total": result["total"],
        "metrics": result["metrics"],
        "violations": [
            {"code": violation.code, "message": str(violation)}
            for violation in result["violations"]
        ],
    }


def _select_candidate(
    source: dict,
    target: dict,
    candidates: list[dict],
    state_axes: dict | None = None,
    *,
    snapshot: dict | None = None,
    fit_candidate_ids: set | None = None,
    policy_trace: dict | None = None,
    semantic_candidate_ids: set | None = None,
    semantic_trace: dict | None = None,
) -> dict:
    """Exclude disqualified layouts, then rank source-aware quality metrics.

    The evaluation package uses the real renderer's font manager. Minimal
    installations can still exercise the layout engine without that optional
    dependency, using the existing conservative typography ranking.
    """
    if snapshot is not None:
        snapshot["candidates"] = []
    if fit_candidate_ids:
        from engine.fit_execution import select_fitted_document

        fit_graph = build_relationship_graph(source)
        fitted = select_fitted_document(
            source,
            target,
            candidates,
            fit_candidate_ids,
            [{"element_ids": list(unit.element_ids)} for unit in fit_graph.units],
            snapshot=snapshot,
        )
        if fitted is not None:
            proposal = fitted["trace"]["fit_policy"]
            receipt = deepcopy(proposal["receipt"])
            receipt["decision"] = deepcopy(proposal["decision"])
            receipt["choice_available"] = len(receipt.get("scores", [])) > 1
            receipt["execution_profile"] = "measured_document_columns"
            if policy_trace is not None:
                policy_trace.update(receipt)
            if snapshot is not None:
                snapshot["fit_policy"] = receipt
            return fitted["design"]
    if semantic_candidate_ids:
        from engine.semantic_execution import select_semantic_layout

        semantic_graph = build_relationship_graph(source)
        anchored = select_semantic_layout(
            source,
            target,
            candidates,
            semantic_candidate_ids,
            [{"element_ids": list(unit.element_ids)} for unit in semantic_graph.units],
            snapshot=snapshot,
        )
        if anchored is not None:
            receipt = deepcopy(anchored["trace"]["fit_policy"])
            if "receipt" in receipt:
                receipt = {**receipt["receipt"], "decision": receipt.get("decision")}
            receipt["execution_profile"] = "source_semantic_anchors"
            if policy_trace is not None:
                policy_trace.update(receipt)
            if semantic_trace is not None:
                semantic_trace.update(anchored["trace"]["semantic_execution"])
            if snapshot is not None:
                snapshot["fit_policy"] = receipt
                snapshot["semantic_execution"] = deepcopy(anchored["trace"]["semantic_execution"])
            return anchored["design"]
    layout_context = build_context(source)
    try:
        from engine.eval import build_context as build_eval_context
        from engine.eval import score
        from engine.eval.occlusion import (
            collision_safety,
            conservative_occlusion,
            infer_occlusion_profile,
        )
        from engine.eval.proportions import analyze_proportions
        from engine.eval.style import analyze_style
    except ImportError as exc:
        if snapshot is not None:
            raise RuntimeError("Candidate snapshots require the full evaluator.") from exc
        if _safe_identity(source, target):
            return source
        if not candidates:
            raise ValueError("No layout candidates were generated") from None
        return max(
            candidates,
            key=lambda c: (
                -len(analyze_rules(source, c)["distorted_rules"]),
                _quality(source, c, layout_context),
            ),
        )

    resources = get_resources()
    context = build_eval_context(source, resources.assets_dir, resources.fonts_dir)
    if _safe_identity(source, target):
        identity = score(source, source, context, target_canvas=target)
        if not identity["violations"]:
            if snapshot is not None:
                snapshot["candidates"].append(
                    {
                        "design": source,
                        "contract_valid": True,
                        "admitted": True,
                        "diagnostics": {
                            "evaluation": _evaluation_snapshot(identity),
                            "identity": True,
                        },
                        "selection": {"safety": [], "fidelity": 1.0, "typography": []},
                    }
                )
            return source

    valid = []
    violations = set()
    graph = build_relationship_graph(source, resources.assets_dir)
    focal_profile = infer_focal_profile(source, graph)
    reading_profile = infer_reading_order(source)
    scene_profile = analyze_scene(source)
    try:
        occlusion_profile = infer_occlusion_profile(source)
    except (OSError, ValueError, RuntimeError):
        occlusion_profile = None
    for candidate in candidates:
        result = score(source, candidate, context, target_canvas=target)
        if result["violations"]:
            violations.update(v.code for v in result["violations"])
            if snapshot is not None:
                snapshot["candidates"].append(
                    {
                        "design": candidate,
                        "contract_valid": False,
                        "admitted": False,
                        "diagnostics": {"evaluation": _evaluation_snapshot(result)},
                        "selection": {"safety": [], "fidelity": None, "typography": []},
                    }
                )
            continue
        style = analyze_style(source, candidate, resources.assets_dir, relationship_graph=graph)
        axes = (state_axes or {}).get(json.dumps(candidate, sort_keys=True), {})
        relationships = analyze_relationships(graph, candidate, axes)
        proportions = analyze_proportions(source, candidate, graph)
        focal = analyze_focal(source, candidate, graph, focal_profile)
        rules = analyze_rules(source, candidate)
        reading_order = analyze_reading_order(source, candidate, reading_profile)
        occlusion = conservative_occlusion(source, candidate, occlusion_profile)
        valid.append(
            _CandidateAssessment(
                candidate,
                result,
                style,
                relationships,
                proportions,
                focal,
                rules,
                reading_order,
                occlusion,
            )
        )
    if not valid:
        detail = ", ".join(sorted(violations)) or "no candidates generated"
        raise ValueError(f"No valid layout for {target['width']}x{target['height']}: {detail}")

    def safety(item):
        candidate, result = item.design, item.evaluation
        relationships, proportions, focal = item.relationships, item.proportions, item.focal
        metrics = result["metrics"]
        # Reading order and explicit ownership are not compensated by a larger
        # minimum font. Softer association hypotheses remain below readability.
        essential = sum(
            w["code"]
            in {
                "reading_order",
                "explicit_group",
                "peer_order",
                "campaign_ownership",
                "row_attachment",
                "overlay_attachment",
                "ornament_attachment",
                "caption_hierarchy",
            }
            for w in relationships["warnings"]
        )
        essential += sum(
            warning["code"]
            in {"source_container_detached", "source_container_unsupported_deformation"}
            for warning in proportions.get("warnings", [])
        )
        action_ids = {
            eid for unit in graph.units if unit.role == "action" for eid in unit.element_ids
        }
        # Copy may continue into another column; an action retains its binding
        # to the message even when the larger composition changes orientation.
        detached = sum(
            w["confidence"] >= 0.8
            and (w["code"] != "shared_axis" or bool(action_ids.intersection(w["element_ids"])))
            for w in relationships["warnings"]
        )
        collisions, photo_visible = collision_safety(
            item.occlusion, metrics.get("raw_collisions", 0.0)
        )
        return (
            photo_visible,
            -collisions,
            -essential,
            -item.reading_order["high_confidence_inversions"],
            -len(item.rules["distorted_rules"]),
            background_coverage(source, candidate, scene_profile),
            _readability(candidate, layout_context, metrics),
            proportions["metrics"].get("rigid_group_error", 0.0) <= math.log(1.05),
            focal["metrics"].get("center_retained", True),
            focal["metrics"].get("visible_span_retention", 1.0),
            focal["metrics"].get("occupied_balance_error", 0.0) <= 0.08,
            -detached,
        )

    safest = max(map(safety, valid))
    feasible = [item for item in valid if safety(item) == safest]

    def fidelity(item):
        reading = item.reading_order
        # Ambiguous placement must not silently erase supported source order.
        # This stays soft: some aspect changes genuinely leave a pair unclear.
        order_retained = (
            reading["score"] * reading["comparison_coverage"] if reading["active"] else 1.0
        )
        return (
            item.style["score"]
            + item.relationships["score"]
            + item.proportions["score"]
            # Retaining authored occupancy matters more than preserving the
            # old global alignment axis when a composition changes orientation.
            + 3 * item.proportions["metrics"].get("composition_span_retention", 1.0)
            + order_retained
        ) / 7

    # Small changes to inferred gaps must not dominate typography or visual
    # hierarchy. Keep a near-best fidelity set before comparing source scale.
    best_fidelity = max(map(fidelity, feasible))
    finalists = [item for item in feasible if fidelity(item) >= best_fidelity - 0.01]
    originals = {element["id"]: element for element in source["elements"]}
    source_scale = math.sqrt(
        target["width"]
        * target["height"]
        / (source["canvas"]["width"] * source["canvas"]["height"])
    )

    def typography_preference(item):
        texts = [element for element in item.design["elements"] if element["type"] == "text"]
        drift = sum(
            abs(
                math.log(
                    element["font_size"] / (originals[element["id"]]["font_size"] * source_scale)
                )
            )
            for element in texts
        )
        # Equivalent similarity transforms differ at machine precision. Match
        # the diagnostic precision before comparing meaningful composition.
        return -round(drift / max(1, len(texts)), 6), item.evaluation["total"], fidelity(item)

    selected = max(finalists, key=typography_preference)
    if fit_candidate_ids:
        from engine.fit_policy import build_fit_menu, choose_fit_plan

        admitted_fit = [
            item for item in finalists if _candidate_id(item.design) in fit_candidate_ids
        ]
        if admitted_fit:
            rows = [
                {"plan_id": _candidate_id(item.design), "design": item.design, "hard_valid": True}
                for item in finalists
            ]
            units = [{"element_ids": list(unit.element_ids)} for unit in graph.units]
            menu = build_fit_menu(source, target, rows, units=units)
            proposal = choose_fit_plan(source, target, menu, rows)
            receipt = proposal["receipt"]
            if proposal["decision"] is not None:
                previous = _candidate_id(selected.design)
                choice = proposal["decision"]["plan_id"]
                selected = next(item for item in finalists if _candidate_id(item.design) == choice)
                receipt.update(
                    selected_candidate_id=choice,
                    comparison_candidate_id=previous,
                    selection_changed=choice != previous,
                    choice_available=len(finalists) > 1,
                    execution_scope="resize_computation",
                )
            if policy_trace is not None:
                policy_trace.update(receipt)
            if snapshot is not None:
                snapshot["fit_policy"] = deepcopy(receipt)
        elif policy_trace is not None:
            policy_trace.update(executed=False, status="fitted_candidates_not_admitted")
    if snapshot is not None:
        finalist_objects = {id(item) for item in finalists}
        for item in valid:
            snapshot["candidates"].append(
                {
                    "design": item.design,
                    "contract_valid": True,
                    "admitted": id(item) in finalist_objects,
                    "diagnostics": {
                        "evaluation": _evaluation_snapshot(item.evaluation),
                        "style": item.style,
                        "relationships": item.relationships,
                        "proportions": item.proportions,
                        "focal": item.focal,
                        "rules": item.rules,
                        "reading_order": item.reading_order,
                        "occlusion": item.occlusion,
                    },
                    "selection": {
                        "safety": list(safety(item)),
                        "fidelity": fidelity(item),
                        "typography": list(typography_preference(item)),
                    },
                }
            )
    return selected.design


@lru_cache(maxsize=64)
def _resize_cached(
    source_json: str,
    width: int,
    height: int,
    assets_dir: str,
    fonts_dir: str,
    scene_proposal_json: str = "",
    scene_digest: str = "",
    visual_plan_json: str = "",
    strict_bounds: bool = False,
    fit_model_digest: str = "",
) -> dict:
    with using_resources(assets_dir, fonts_dir):
        return _resize_in_context(
            source_json,
            width,
            height,
            scene_proposal=json.loads(scene_proposal_json) if scene_proposal_json else None,
            visual_plan=json.loads(visual_plan_json) if visual_plan_json else None,
            strict_bounds=strict_bounds,
            fit_model_digest=fit_model_digest,
        )


def _candidate_pool(
    source,
    target_canvas,
    *,
    capture_origins=False,
    scene_proposal=None,
    model_trace=None,
    visual_plan=None,
    strict_bounds=False,
):
    resolution = _resolution_state(source, target_canvas)
    if strict_bounds:
        from engine.stage_one import contain_resolution_decorations

        resolution = contain_resolution_decorations(source, target_canvas, resolution)
    if resolution is not None:
        candidates, legacy_count, states = [], 0, (resolution,)
    else:
        graph = build_relationship_graph(source)
        fitted = generate_fitted_column_states(source, target_canvas, graph)
        if fitted:
            # Authored document columns need measured text reflow. Arbitrary
            # marketing grids separate their section labels and live paragraphs.
            candidates, legacy_count = [], 0
            states = tuple(fitted)
        else:
            anchored = generate_semantic_anchor_states(source, target_canvas)
            if anchored:
                candidates, legacy_count, states = [], 0, tuple(anchored)
            else:
                candidates = generate_candidates(source, target_canvas)
                legacy_count = len(candidates)
                states = (
                    *generate_background_states(source, target_canvas),
                    *generate_composition_states(source, target_canvas, graph),
                    *generate_grouped_states(source, target_canvas, graph),
                    *generate_focal_states(source, target_canvas, graph),
                    *generate_source_guides(source, target_canvas, graph),
                )
    candidates.extend(state.design for state in states)
    if visual_plan is not None:
        from engine.layout.visual_plan import realize_plan
        from engine.visual_plan import build_plan_menu

        original_ids = {_candidate_id(design) for design in candidates}
        menu = build_plan_menu(source, target_canvas)
        proposed = realize_plan(source, target_canvas, menu, visual_plan)
        candidates.extend(state.design for state in proposed)
        states = (*states, *proposed)
        new_ids = sorted({_candidate_id(state.design) for state in proposed} - original_ids)
        if model_trace is not None:
            model_trace["visual_plan"] = {
                "menu_digest": menu["menu_digest"],
                "plan_id": visual_plan["plan_id"],
                "compiled_candidates": len(proposed),
                "new_candidates_generated": len(new_ids),
                "new_candidate_ids": new_ids,
            }
    by_design = {json.dumps(state.design, sort_keys=True): state for state in states}
    axes = {key: dict(state.cluster_axes) for key, state in by_design.items()}
    origins = {}
    if capture_origins:
        for index, candidate in enumerate(candidates[:legacy_count]):
            key = json.dumps(candidate, sort_keys=True)
            origins.setdefault(key, []).append(
                {
                    "name": "candidate",
                    "topology": "legacy",
                    "index": index,
                    "cluster_axes": {},
                    "sizing_exceptions": [],
                }
            )
        for state in states:
            key = json.dumps(state.design, sort_keys=True)
            origins.setdefault(key, []).append(
                {
                    "name": state.name,
                    "topology": state.topology,
                    "cluster_axes": dict(state.cluster_axes),
                    "sizing_exceptions": list(getattr(state, "exceptions", ())),
                }
            )
    if scene_proposal is not None:
        candidates, trace = constrain_candidates(
            source, candidates, scene_proposal, analyze_scene(source)
        )
        if model_trace is not None:
            model_trace.update(trace)
    if strict_bounds:
        from engine.stage_one import strictly_in_bounds

        candidates = [design for design in candidates if strictly_in_bounds(design)]
    return candidates, by_design, axes, origins


def _state_metadata(design, source, source_json, by_design):
    state = by_design.get(json.dumps(design, sort_keys=True))
    return {
        "name": state.name if state else "source" if design == source else "candidate",
        "topology": state.topology if state else "source" if design == source else "legacy",
        "source_digest": hashlib.sha256(source_json.encode()).hexdigest(),
        "cluster_axes": dict(state.cluster_axes) if state else {},
        "sizing_exceptions": list(getattr(state, "exceptions", ())),
    }


def _resize_in_context(
    source_json: str,
    width: int,
    height: int,
    *,
    scene_proposal=None,
    visual_plan=None,
    strict_bounds=False,
    fit_model_digest="",
) -> dict:
    source = json.loads(source_json)
    target_canvas = {"width": width, "height": height}
    model_trace = {}
    candidates, by_design, axes, _ = _candidate_pool(
        source,
        target_canvas,
        scene_proposal=scene_proposal,
        model_trace=model_trace,
        visual_plan=visual_plan,
        strict_bounds=strict_bounds,
    )
    fit_ids = {
        _candidate_id(state.design)
        for state in by_design.values()
        if state.topology == "fitted-source-columns"
    }
    fit_trace = {}
    semantic_trace = {}
    semantic_ids = {
        _candidate_id(state.design)
        for state in by_design.values()
        if state.topology == "source-semantic-anchors"
    }
    selected = _select_candidate(
        source,
        target_canvas,
        candidates,
        axes,
        fit_candidate_ids=fit_ids,
        policy_trace=fit_trace,
        semantic_candidate_ids=semantic_ids,
        semantic_trace=semantic_trace,
    )
    if fit_trace.get("executed") and fit_trace["model_sha256"] != fit_model_digest:
        raise ValueError("Fit policy changed during resize; retry with its current identity")
    state = _state_metadata(selected, source, source_json, by_design)
    if fit_trace:
        state["fit_policy"] = fit_trace
    if semantic_trace:
        state["semantic_execution"] = semantic_trace
    if scene_proposal is not None:
        state["scene_model"] = {k: v for k, v in model_trace.items() if k != "visual_plan"}
    if visual_plan is not None:
        state["visual_plan"] = model_trace["visual_plan"]
        state["visual_plan"]["model_candidate_selected"] = (
            _candidate_id(selected) in state["visual_plan"]["new_candidate_ids"]
        )
    if strict_bounds:
        state["constraint_profile"] = "stage-one-strict-bounds"
    return {
        "design": selected,
        "state": state,
    }


def _candidate_id(design):
    canonical = json.dumps(
        design, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return "candidate-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_candidate_snapshot(
    source: dict[str, Any],
    target_canvas: dict[str, Any],
    *,
    assets_dir: str | Path | None = None,
    fonts_dir: str | Path | None = None,
    visual_plan: dict | None = None,
    strict_bounds: bool = False,
) -> dict:
    """Capture actual selector decisions without changing selection or its cache.

    Broader contract-valid proposals can be inspected in shadow experiments.
    Only admitted finalists share the baseline's safety tier and fidelity window.
    No model is run here. An optional validated visual plan seeds bounded new
    candidates through the same pool as ordinary resizing. No result cache is used.
    """
    if type(strict_bounds) is not bool:
        raise ValueError("strict_bounds must be a boolean")
    source_json = json.dumps(source, sort_keys=True, allow_nan=False)
    original = json.loads(source_json)
    target = {"width": target_canvas["width"], "height": target_canvas["height"]}
    json.dumps(target, allow_nan=False)
    with using_resources(assets_dir, fonts_dir):
        model_trace = {}
        candidates, by_design, axes, origins = _candidate_pool(
            original,
            target,
            capture_origins=True,
            visual_plan=visual_plan,
            strict_bounds=strict_bounds,
            model_trace=model_trace,
        )
        trace = {}
        fit_ids = {
            _candidate_id(state.design)
            for state in by_design.values()
            if state.topology == "fitted-source-columns"
        }
        semantic_ids = {
            _candidate_id(state.design)
            for state in by_design.values()
            if state.topology == "source-semantic-anchors"
        }
        selected = _select_candidate(
            original,
            target,
            candidates,
            axes,
            snapshot=trace,
            fit_candidate_ids=fit_ids,
            semantic_candidate_ids=semantic_ids,
        )
        records = {_candidate_id(row["design"]): row for row in trace["candidates"]}
        ordered = {}
        for design in [*candidates, selected]:
            candidate_id = _candidate_id(design)
            if candidate_id not in records or candidate_id in ordered:
                continue
            row = records[candidate_id]
            key = json.dumps(row["design"], sort_keys=True)
            state = _state_metadata(row["design"], original, source_json, by_design)
            ordered[candidate_id] = {
                "candidate_id": candidate_id,
                **row,
                "state": state,
                "provenance": origins.get(key) or [state],
            }
        result = {
            "schema_version": 1,
            "source_digest": hashlib.sha256(source_json.encode()).hexdigest(),
            "target_canvas": target,
            "baseline_candidate_id": _candidate_id(selected),
            "admitted_finalist_ids": [key for key, row in ordered.items() if row["admitted"]],
            "candidates": list(ordered.values()),
        }
        if strict_bounds:
            result["strict_bounds"] = True
        if "fit_policy" in trace:
            result["fit_policy"] = trace["fit_policy"]
        if "semantic_execution" in trace:
            result["semantic_execution"] = trace["semantic_execution"]
        if visual_plan is not None:
            result["visual_plan"] = model_trace["visual_plan"]
            result["visual_plan"]["model_candidate_selected"] = (
                _candidate_id(selected) in model_trace["visual_plan"]["new_candidate_ids"]
            )
        json.dumps(result, allow_nan=False)
        return deepcopy(result)


def resize_with_state(
    source: dict[str, Any],
    target_canvas: dict[str, Any],
    *,
    assets_dir: str | Path | None = None,
    fonts_dir: str | Path | None = None,
    scene_proposal: dict | None = None,
    visual_plan: dict | None = None,
    strict_bounds: bool = False,
) -> dict:
    """Return a fresh design and the exact declared state used during selection."""
    with using_resources(assets_dir, fonts_dir) as resources:
        from engine.fit_policy import fit_policy_identity

        fit_identity = fit_policy_identity()
        if type(strict_bounds) is not bool:
            raise ValueError("strict_bounds must be a boolean")
        profile = analyze_scene(source)
        if scene_proposal is not None:
            # Rebind current asset bytes even on a result-cache hit.
            scene_proposal = validate_model_response(scene_proposal, profile)
        if visual_plan is not None:
            from engine.visual_plan import build_plan_menu, validate_plan_decision

            visual_plan = validate_plan_decision(
                build_plan_menu(source, target_canvas), visual_plan
            )
        return deepcopy(
            _resize_cached(
                json.dumps(source, sort_keys=True),
                target_canvas["width"],
                target_canvas["height"],
                str(resources.assets_dir),
                str(resources.fonts_dir),
                json.dumps(scene_proposal, sort_keys=True, allow_nan=False)
                if scene_proposal is not None
                else "",
                profile["scene_digest"],
                json.dumps(visual_plan, sort_keys=True, allow_nan=False)
                if visual_plan is not None
                else "",
                strict_bounds,
                fit_identity.get("model_sha256") or "",
            )
        )


def resize(
    source: dict[str, Any],
    target_canvas: dict[str, Any],
    *,
    assets_dir: str | Path | None = None,
    fonts_dir: str | Path | None = None,
    scene_proposal: dict | None = None,
    visual_plan: dict | None = None,
    strict_bounds: bool = False,
) -> dict[str, Any]:
    """Adapt composition without changing the supplied assets, fonts or copy.

    Deterministic, offline, and does not mutate either argument. A bounded task
    cache speeds up repeated viewer requests; each caller receives its own copy.
    """
    return resize_with_state(
        source,
        target_canvas,
        assets_dir=assets_dir,
        fonts_dir=fonts_dir,
        scene_proposal=scene_proposal,
        visual_plan=visual_plan,
        strict_bounds=strict_bounds,
    )["design"]
