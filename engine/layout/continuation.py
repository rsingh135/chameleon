"""Experimental discrete continuation search; the default resize engine is unchanged.

Every anchor is generated from the original design. Previous frames contribute
geometry only, so semantic roles and preferred typography cannot drift. A small
beam minimizes the worst unit-center jump before its quality/movement cost.
Endpoint-guided seeds supply gradual geometric alternatives. The final anchor
is pinned to the exact default endpoint, isolating transition quality.
This is neither a global optimum nor a collision-free continuous animation.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from engine.eval import build_context as build_eval_context
from engine.eval import score
from engine.resize import resize

from .candidates import _base, _finish, _place, _preferred_height, generate_candidates
from .geometry import Box, intersection, union
from .model import DesignContext, build_context

ROOT = Path(__file__).resolve().parents[2]
_PRIORITY = {"headline": 3.0, "logo": 3.0, "hero": 2.0, "action": 2.0, "legal": 0.5}
QUALITY_FLOOR_TOLERANCE = 0.06


def orientation_canvases(source_canvas: dict, target: dict, steps: int = 7) -> list[dict]:
    """Include both endpoints, interpolating each dimension in logarithmic space."""
    if type(steps) is not int or not 2 <= steps <= 25:
        raise ValueError("steps must be an integer between 2 and 25")
    for canvas in (source_canvas, target):
        if any(type(canvas.get(key)) is not int or canvas[key] < 1 for key in ("width", "height")):
            raise ValueError("canvas dimensions must be positive integers")
    canvases = []
    for index in range(steps):
        t = index / (steps - 1)
        canvases.append(
            {
                key: max(
                    1,
                    round(
                        math.exp((1 - t) * math.log(source_canvas[key]) + t * math.log(target[key]))
                    ),
                )
                for key in ("width", "height")
            }
        )
    canvases[0], canvases[-1] = dict(source_canvas), dict(target)
    return canvases


def _unit_boxes(design: dict, context: DesignContext) -> list[Box]:
    by_id = {element["id"]: element for element in design["elements"]}
    return [union([by_id[element["id"]] for element in unit.elements]) for unit in context.units]


def movement(previous: dict, current: dict, context: DesignContext) -> dict:
    """Measure discrete unit-center, normalized log-scale and font changes."""
    before, after = _unit_boxes(previous, context), _unit_boxes(current, context)
    pw, ph = previous["canvas"]["width"], previous["canvas"]["height"]
    cw, ch = current["canvas"]["width"], current["canvas"]["height"]
    ptext = {element["id"]: element for element in previous["elements"]}
    ctext = {element["id"]: element for element in current["elements"]}
    units = []
    for unit, a, b in zip(context.units, before, after, strict=True):
        center = math.hypot(
            (a.x + a.width / 2) / pw - (b.x + b.width / 2) / cw,
            (a.y + a.height / 2) / ph - (b.y + b.height / 2) / ch,
        )
        scale = (
            abs(math.log(max(1e-9, b.width / cw) / max(1e-9, a.width / pw)))
            + abs(math.log(max(1e-9, b.height / ch) / max(1e-9, a.height / ph)))
        ) / 2
        fonts = [
            abs(
                math.log(
                    (ctext[element["id"]]["font_size"] / math.sqrt(cw * ch))
                    / (ptext[element["id"]]["font_size"] / math.sqrt(pw * ph))
                )
            )
            for element in unit.text
        ]
        font = sum(fonts) / len(fonts) if fonts else 0.0
        units.append(
            {
                "ids": [element["id"] for element in unit.elements],
                "role": unit.role,
                "center_distance": center,
                "log_scale_change": scale,
                "font_log_change": font,
                "combined": center + 0.2 * scale + 0.15 * font,
                "weight": _PRIORITY.get(unit.role, 1.0),
            }
        )
    weight = sum(unit["weight"] for unit in units)
    return {
        "combined": sum(unit["combined"] * unit["weight"] for unit in units) / weight
        if weight
        else 0.0,
        "max_center_jump": max((unit["center_distance"] for unit in units), default=0.0),
        "max_font_log_change": max((unit["font_log_change"] for unit in units), default=0.0),
        "units": units,
    }


def _clamp_box(box: Box, width: float, height: float) -> Box:
    w, h = min(width, box.width), min(height, box.height)
    return Box(min(max(0.0, box.x), width - w), min(max(0.0, box.y), height - h), w, h)


def _repair_boxes(boxes: list[Box], context: DesignContext, target: dict) -> list[Box]:
    """Bounded local translation repair; intentional source overlaps are exempt."""
    width, height = target["width"], target["height"]
    boxes = [_clamp_box(box, width, height) for box in boxes]
    gap = min(width, height) * 0.012
    for _ in range(8):
        changed = False
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                b = boxes[j]
                if intersection(context.units[i].box, context.units[j].box) > 0:
                    continue
                dx = min(a.x + a.width, b.x + b.width) - max(a.x, b.x) + gap
                dy = min(a.y + a.height, b.y + b.height) - max(a.y, b.y) + gap
                if dx <= 0 or dy <= 0:
                    continue
                ai = _PRIORITY.get(context.units[i].role, 1.0)
                bj = _PRIORITY.get(context.units[j].role, 1.0)
                share = bj / (ai + bj)
                if dx / width < dy / height:
                    direction = -1 if a.x + a.width / 2 <= b.x + b.width / 2 else 1
                    a = Box(a.x + direction * dx * share, a.y, a.width, a.height)
                    b = Box(b.x - direction * dx * (1 - share), b.y, b.width, b.height)
                else:
                    direction = -1 if a.y + a.height / 2 <= b.y + b.height / 2 else 1
                    a = Box(a.x, a.y + direction * dy * share, a.width, a.height)
                    b = Box(b.x, b.y - direction * dy * (1 - share), b.width, b.height)
                boxes[i], boxes[j] = _clamp_box(a, width, height), _clamp_box(b, width, height)
                a = boxes[i]
                changed = True
        if not changed:
            break
    return boxes


def propagated_candidate(
    source: dict, context: DesignContext, previous: dict, target: dict, *, repair: bool = False
) -> dict:
    """Carry forward unit geometry, placing original content at original font preferences."""
    sx = target["width"] / previous["canvas"]["width"]
    sy = target["height"] / previous["canvas"]["height"]
    font_scale = math.sqrt(target["width"] * target["height"] / (context.width * context.height))
    boxes = [
        Box(box.x * sx, box.y * sy, box.width * sx, box.height * sy)
        for box in _unit_boxes(previous, context)
    ]
    if repair:
        restored = []
        for unit, box in zip(context.units, boxes, strict=True):
            if unit.text:
                desired_width = min(target["width"] * 0.85, unit.box.width * font_scale)
                w = max(box.width, desired_width)
                desired_height = _preferred_height(unit, w, target["height"], font_scale)
                h = max(box.height, min(target["height"] * 0.6, desired_height))
                box = Box(box.x + (box.width - w) / 2, box.y + (box.height - h) / 2, w, h)
            restored.append(box)
        boxes = _repair_boxes(restored, context, target)
    elements = _base(context, target)
    for unit, box in zip(context.units, boxes, strict=True):
        elements.extend(_place(unit, box, font_scale))
    return _finish(source, target, elements)


def guided_candidate(
    source: dict,
    context: DesignContext,
    endpoint: dict,
    target: dict,
    t: float,
    *,
    repair: bool = False,
) -> dict:
    """Interpolate normalized unit centers/log sizes, then place original content."""
    width, height = target["width"], target["height"]
    ew, eh = endpoint["canvas"]["width"], endpoint["canvas"]["height"]
    boxes = []
    for unit, end in zip(context.units, _unit_boxes(endpoint, context), strict=True):
        start = unit.box
        cx = (1 - t) * (start.x + start.width / 2) / context.width
        cx += t * (end.x + end.width / 2) / ew
        cy = (1 - t) * (start.y + start.height / 2) / context.height
        cy += t * (end.y + end.height / 2) / eh
        w = (
            math.exp((1 - t) * math.log(start.width / context.width) + t * math.log(end.width / ew))
            * width
        )
        h = (
            math.exp(
                (1 - t) * math.log(start.height / context.height) + t * math.log(end.height / eh)
            )
            * height
        )
        boxes.append(Box(cx * width - w / 2, cy * height - h / 2, w, h))
    if repair:
        boxes = _repair_boxes(boxes, context, target)
    font_scale = math.sqrt(width * height / (context.width * context.height))
    elements = _base(context, target)
    for unit, box in zip(context.units, boxes, strict=True):
        elements.extend(_place(unit, box, font_scale))
    return _finish(source, target, elements)


@dataclass
class _State:
    frames: tuple[dict, ...]
    choices: tuple[str, ...]
    cost: float
    max_center_jump: float


def _summary(result: dict, design: dict) -> dict:
    return {
        "quality": result["total"],
        "violations": [str(violation) for violation in result["violations"]],
        "metrics": result["metrics"],
        "min_font_px": min(
            (element["font_size"] for element in design["elements"] if element["type"] == "text"),
            default=None,
        ),
    }


def continue_layout(
    source: dict,
    target_canvas: dict,
    *,
    steps: int = 7,
    beam_width: int = 4,
    movement_weight: float = 0.65,
) -> dict:
    """Search discrete anchors, ending at the exact default-engine result."""
    if type(beam_width) is not int or not 1 <= beam_width <= 12:
        raise ValueError("beam_width must be an integer between 1 and 12")
    if not math.isfinite(movement_weight) or movement_weight < 0:
        raise ValueError("movement_weight must be finite and nonnegative")
    original = deepcopy(source)
    canvases = orientation_canvases(original["canvas"], target_canvas, steps)
    context = build_context(original)
    evaluation = build_eval_context(original, ROOT / "brand_kit/assets", ROOT / "brand_kit/fonts")
    cache = {}

    def assess(design):
        key = json.dumps(design, sort_keys=True)
        if key not in cache:
            cache[key] = score(original, design, evaluation, target_canvas=design["canvas"])
        return key, cache[key]

    baseline_frames = [deepcopy(original)] + [resize(original, canvas) for canvas in canvases[1:]]
    beam = [_State((deepcopy(original),), ("original",), 0.0, 0.0)]
    diagnostics = []
    best_endpoint_quality = 0.0
    for index, canvas in enumerate(canvases[1:], start=1):
        proposed = [("fresh", design) for design in generate_candidates(original, canvas)]
        proposed.append(("default-baseline", baseline_frames[index]))
        for repair in (False, True):
            proposed.append(
                (
                    "endpoint-guided-repaired" if repair else "endpoint-guided",
                    guided_candidate(
                        original,
                        context,
                        baseline_frames[-1],
                        canvas,
                        index / (steps - 1),
                        repair=repair,
                    ),
                )
            )
        for state in beam:
            for repair in (False, True):
                proposed.append(
                    (
                        "propagated-repaired" if repair else "propagated",
                        propagated_candidate(
                            original, context, state.frames[-1], canvas, repair=repair
                        ),
                    )
                )
        _, baseline_result = assess(baseline_frames[index])
        quality_floor = max(0.0, baseline_result["total"] - QUALITY_FLOOR_TOLERANCE)
        valid, rejected, rejected_quality = {}, [], []
        for origin, design in proposed:
            key, result = assess(design)
            if result["violations"]:
                rejected.append(
                    {"origin": origin, "violations": [str(v) for v in result["violations"]]}
                )
            elif result["total"] < quality_floor - 1e-9:
                rejected_quality.append({"origin": origin, "quality": result["total"]})
            elif key not in valid:
                valid[key] = (origin, design, result)
        if not valid:
            raise RuntimeError(f"No valid continuation candidate at anchor {index}: {rejected}")
        best_quality = max(result["total"] for _, _, result in valid.values())
        valid_before_endpoint_guard = len(valid)
        if index == steps - 1:
            best_endpoint_quality = best_quality
            baseline_key, baseline_result = assess(baseline_frames[-1])
            if baseline_result["violations"]:
                raise RuntimeError("The pinned default endpoint violates the local contract")
            valid = {
                baseline_key: ("pinned-default-endpoint", baseline_frames[-1], baseline_result)
            }
        alternatives = []
        for key, (origin, design, result) in valid.items():
            transitions = []
            for state in beam:
                jump = movement(state.frames[-1], design, context)
                cost = state.cost + (1 - result["total"]) + movement_weight * jump["combined"]
                transitions.append(
                    _State(
                        state.frames + (design,),
                        state.choices + (origin,),
                        cost,
                        max(state.max_center_jump, jump["max_center_jump"]),
                    )
                )
            best = min(transitions, key=lambda state: (state.max_center_jump, state.cost))
            remaining = steps - 1 - index
            terminal_distance = movement(design, baseline_frames[-1], context)["max_center_jump"]
            # Triangle inequality: the remaining displacement needs at least
            # this much worst-step motion. No additional tuning weight is used.
            lower_bound = (
                max(best.max_center_jump, terminal_distance / remaining)
                if remaining
                else best.max_center_jump
            )
            alternatives.append(
                (lower_bound, best.max_center_jump, best.cost, -result["total"], key, best)
            )
        alternatives.sort(key=lambda item: item[:5])
        beam = [item[5] for item in alternatives[:beam_width]]
        diagnostics.append(
            {
                "index": index,
                "canvas": canvas,
                "proposed_count": len(proposed),
                "valid_unique_count": valid_before_endpoint_guard,
                "eligible_after_endpoint_guard": len(valid),
                "rejected": rejected,
                "quality_floor": quality_floor,
                "rejected_quality": rejected_quality,
                "best_quality": best_quality,
            }
        )
    chosen = min(beam, key=lambda state: (state.max_center_jump, state.cost))
    anchors, baseline_anchors = [], []
    for index, (design, baseline) in enumerate(zip(chosen.frames, baseline_frames, strict=True)):
        _, result = assess(design)
        _, baseline_result = assess(baseline)
        anchor = {
            "index": index,
            "canvas": design["canvas"],
            "origin": chosen.choices[index],
            **_summary(result, design),
        }
        base = {"index": index, "canvas": baseline["canvas"], **_summary(baseline_result, baseline)}
        anchor["movement"] = movement(chosen.frames[max(0, index - 1)], design, context)
        base["movement"] = movement(baseline_frames[max(0, index - 1)], baseline, context)
        anchors.append(anchor)
        baseline_anchors.append(base)
    return {
        "prototype": True,
        "frames": deepcopy(list(chosen.frames)),
        "anchors": anchors,
        "baseline_frames": baseline_frames,
        "baseline_anchors": baseline_anchors,
        "search": {
            "method": "bounded beam over discrete log-interpolated canvases",
            "steps": steps,
            "beam_width": beam_width,
            "movement_weight": movement_weight,
            "objective": "lexicographic worst unit-center jump, then quality/movement cost",
            "beam_lookahead": "endpoint distance / remaining anchors bounds the worst jump",
            "quality_floor_tolerance": QUALITY_FLOOR_TOLERANCE,
            "endpoint_policy": "exact default baseline endpoint; controlled transition experiment",
            "quality_basis": "JSON-only local proxy score, not the official evaluator",
            "path_cost": chosen.cost,
            "diagnostics": diagnostics,
        },
        "comparison": {
            "prototype_endpoint_quality": anchors[-1]["quality"],
            "default_endpoint_quality": baseline_anchors[-1]["quality"],
            "best_explored_endpoint_quality": best_endpoint_quality,
            "endpoint_quality_gap": best_endpoint_quality - anchors[-1]["quality"],
            "endpoint_design_matches_baseline": chosen.frames[-1] == baseline_frames[-1],
            "prototype_total_movement": sum(
                anchor["movement"]["combined"] for anchor in anchors[1:]
            ),
            "default_anchor_total_movement": sum(
                anchor["movement"]["combined"] for anchor in baseline_anchors[1:]
            ),
            "prototype_max_center_jump": max(
                anchor["movement"]["max_center_jump"] for anchor in anchors
            ),
            "default_anchor_max_center_jump": max(
                anchor["movement"]["max_center_jump"] for anchor in baseline_anchors
            ),
            "selected_constraint_violations": [
                {"anchor": anchor["index"], "violations": anchor["violations"]}
                for anchor in anchors
                if anchor["violations"]
            ],
        },
        "limitations": [
            "Experimental bounded search; the default resize engine is unchanged.",
            "Only discrete anchors are checked; interpolating may cause overlaps or clipping.",
            "Proxy scores are not official scores and do not prove that a layout looks designed.",
            "Small text, collisions and abrupt moves may remain; inspect PNGs and diagnostics.",
            "The endpoint is pinned to the default; this experiment explores transitions only.",
        ],
    }
