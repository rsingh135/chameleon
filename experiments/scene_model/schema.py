"""Closed semantic predictions; identities and crop authority belong to the caller."""

from __future__ import annotations

import json
import math
from copy import deepcopy

ROLE_NAMES = ("background", "product", "logo", "foreground", "unknown")
REGION_KINDS = ("face", "person", "product", "logo", "other")
MAX_PREDICTION_BYTES = 16 * 1024


def _finite_probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0 <= value <= 1


def validate_prediction(value):
    required = {"proposed_role", "confidence"}
    optional = {"protected_regions", "rationale"}
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        raise ValueError(
            "Prediction must contain only semantic role, confidence and optional regions/rationale"
        )
    if value["proposed_role"] not in ROLE_NAMES or not _finite_probability(value["confidence"]):
        raise ValueError("Prediction needs a known role and finite confidence from 0 to 1")
    regions = value.get("protected_regions", [])
    if not isinstance(regions, list) or len(regions) > 32:
        raise ValueError("Prediction has too many protected regions")
    for region in regions:
        if not isinstance(region, dict) or set(region) != {"kind", "box", "confidence"}:
            raise ValueError("Protected regions require kind, box and confidence only")
        if region["kind"] not in REGION_KINDS or not _finite_probability(region["confidence"]):
            raise ValueError("Protected region has an unknown kind or invalid confidence")
        box = region["box"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in box
            )
        ):
            raise ValueError("Region box needs four finite normalized numbers")
        x, y, width, height = box
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("Region box must stay inside the exact input asset")
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > 1000:
        raise ValueError("Prediction rationale must be text of at most 1000 characters")
    if len(json.dumps(value, allow_nan=False).encode()) > MAX_PREDICTION_BYTES:
        raise ValueError("Prediction exceeds its byte limit")
    return deepcopy(value)


def parse_prediction(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_PREDICTION_BYTES:
        raise ValueError("Prediction text exceeds its byte limit")

    def pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError("Duplicate prediction key")
            value[key] = child
        return value

    def constant(_value):
        raise ValueError("Prediction numbers must be finite")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(
            "Model must return one strict JSON object without prose or code fences"
        ) from exc
    return validate_prediction(value)
