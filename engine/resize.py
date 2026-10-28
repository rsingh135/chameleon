"""Offline, deterministic design reflow with source-aware candidate selection."""

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.layout import build_context, generate_candidates, get_measurer
from engine.layout.composition import generate_composition_states
from engine.layout.geometry import bounds, intersection
from engine.layout.relationships import analyze_relationships, build_relationship_graph


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


def _select_candidate(source: dict, target: dict, candidates: list[dict]) -> dict:
    """Exclude disqualified layouts, then rank source-aware quality metrics.

    The evaluation package uses the real renderer's font manager. Minimal
    installations can still exercise the layout engine without that optional
    dependency, using the existing conservative typography ranking.
    """
    layout_context = build_context(source)
    try:
        from engine.eval import build_context as build_eval_context
        from engine.eval import score
        from engine.eval.style import analyze_style
    except ImportError:
        if _safe_identity(source, target):
            return source
        if not candidates:
            raise ValueError("No layout candidates were generated") from None
        return max(candidates, key=lambda c: _quality(source, c, layout_context))

    root = Path(__file__).resolve().parents[1]
    context = build_eval_context(source, root / "brand_kit/assets", root / "brand_kit/fonts")
    if _safe_identity(source, target):
        identity = score(source, source, context, target_canvas=target)
        if not identity["violations"]:
            return source

    valid = []
    violations = set()
    graph = build_relationship_graph(source, root / "brand_kit/assets")
    for candidate in candidates:
        result = score(source, candidate, context, target_canvas=target)
        if result["violations"]:
            violations.update(v.code for v in result["violations"])
            continue
        style = analyze_style(source, candidate, root / "brand_kit/assets")
        relationships = analyze_relationships(graph, candidate)
        valid.append((candidate, result, style, relationships))
    if not valid:
        detail = ", ".join(sorted(violations)) or "no candidates generated"
        raise ValueError(f"No valid layout for {target['width']}x{target['height']}: {detail}")

    def safety(item):
        candidate, result, style, relationships = item
        metrics = result["metrics"]
        # Reading order and explicit ownership are not compensated by a larger
        # minimum font. Softer association hypotheses remain below readability.
        essential = sum(
            w["code"] in {"reading_order", "explicit_group"} for w in relationships["warnings"]
        )
        detached = sum(w["confidence"] >= 0.8 for w in relationships["warnings"])
        return (
            -metrics.get("raw_collisions", 0.0),
            -essential,
            metrics.get("text_size", 1.0),
            -detached,
        )

    safest = max(map(safety, valid))
    feasible = [item for item in valid if safety(item) == safest]

    def fidelity(item):
        return (item[2]["score"] + item[3]["score"]) / 2

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
        texts = [element for element in item[0]["elements"] if element["type"] == "text"]
        drift = sum(
            abs(
                math.log(
                    element["font_size"] / (originals[element["id"]]["font_size"] * source_scale)
                )
            )
            for element in texts
        )
        return -drift / max(1, len(texts)), item[1]["total"], fidelity(item)

    return max(finalists, key=typography_preference)[0]


@lru_cache(maxsize=64)
def _resize_cached(source_json: str, width: int, height: int) -> dict:
    source = json.loads(source_json)
    target_canvas = {"width": width, "height": height}
    candidates = generate_candidates(source, target_canvas)
    candidates.extend(state.design for state in generate_composition_states(source, target_canvas))
    return _select_candidate(source, target_canvas, candidates)


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Adapt composition without changing the supplied assets, fonts or copy.

    Deterministic, offline, and does not mutate either argument. A bounded task
    cache speeds up repeated viewer requests; each caller receives its own copy.
    """
    return deepcopy(
        _resize_cached(
            json.dumps(source, sort_keys=True), target_canvas["width"], target_canvas["height"]
        )
    )
