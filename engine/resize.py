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
from engine.layout.focal import analyze_focal, infer_focal_profile
from engine.layout.focal_states import generate_focal_states
from engine.layout.geometry import bounds, intersection
from engine.layout.grouped import GroupedState, generate_grouped_states
from engine.layout.reading_order import analyze_reading_order, infer_reading_order
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.layout.rules import analyze_rules
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
) -> dict:
    """Exclude disqualified layouts, then rank source-aware quality metrics.

    The evaluation package uses the real renderer's font manager. Minimal
    installations can still exercise the layout engine without that optional
    dependency, using the existing conservative typography ranking.
    """
    if snapshot is not None:
        snapshot["candidates"] = []
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
) -> dict:
    with using_resources(assets_dir, fonts_dir):
        return _resize_in_context(
            source_json,
            width,
            height,
            scene_proposal=json.loads(scene_proposal_json) if scene_proposal_json else None,
        )


def _candidate_pool(
    source, target_canvas, *, capture_origins=False, scene_proposal=None, model_trace=None
):
    resolution = _resolution_state(source, target_canvas)
    if resolution is not None:
        candidates, legacy_count, states = [], 0, (resolution,)
    else:
        candidates = generate_candidates(source, target_canvas)
        legacy_count = len(candidates)
        graph = build_relationship_graph(source)
        states = (
            *generate_background_states(source, target_canvas),
            *generate_composition_states(source, target_canvas, graph),
            *generate_grouped_states(source, target_canvas, graph),
            *generate_focal_states(source, target_canvas, graph),
            *generate_source_guides(source, target_canvas, graph),
        )
    candidates.extend(state.design for state in states)
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


def _resize_in_context(source_json: str, width: int, height: int, *, scene_proposal=None) -> dict:
    source = json.loads(source_json)
    target_canvas = {"width": width, "height": height}
    model_trace = {}
    candidates, by_design, axes, _ = _candidate_pool(
        source, target_canvas, scene_proposal=scene_proposal, model_trace=model_trace
    )
    selected = _select_candidate(source, target_canvas, candidates, axes)
    state = _state_metadata(selected, source, source_json, by_design)
    if scene_proposal is not None:
        state["scene_model"] = model_trace
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
) -> dict:
    """Capture actual selector decisions without changing selection or its cache.

    Broader contract-valid proposals can be inspected in shadow experiments.
    Only admitted finalists share the baseline's safety tier and fidelity window.
    No model, experiment package, result-cache access or selection override is used.
    """
    source_json = json.dumps(source, sort_keys=True, allow_nan=False)
    original = json.loads(source_json)
    target = {"width": target_canvas["width"], "height": target_canvas["height"]}
    json.dumps(target, allow_nan=False)
    with using_resources(assets_dir, fonts_dir):
        candidates, by_design, axes, origins = _candidate_pool(
            original, target, capture_origins=True
        )
        trace = {}
        selected = _select_candidate(original, target, candidates, axes, snapshot=trace)
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
        json.dumps(result, allow_nan=False)
        return deepcopy(result)


def resize_with_state(
    source: dict[str, Any],
    target_canvas: dict[str, Any],
    *,
    assets_dir: str | Path | None = None,
    fonts_dir: str | Path | None = None,
    scene_proposal: dict | None = None,
) -> dict:
    """Return a fresh design and the exact declared state used during selection."""
    with using_resources(assets_dir, fonts_dir) as resources:
        profile = analyze_scene(source)
        if scene_proposal is not None:
            # Rebind current asset bytes even on a result-cache hit.
            scene_proposal = validate_model_response(scene_proposal, profile)
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
            )
        )


def resize(
    source: dict[str, Any],
    target_canvas: dict[str, Any],
    *,
    assets_dir: str | Path | None = None,
    fonts_dir: str | Path | None = None,
    scene_proposal: dict | None = None,
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
    )["design"]
