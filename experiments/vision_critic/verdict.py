"""Strict model-verdict parsing without training, model or GPU dependencies."""

from __future__ import annotations


def parse_model_verdict(text: str, candidate_element_ids: dict | None = None) -> dict:
    """Untrusted generated JSON must match the small choice/issues contract."""
    from .contracts import parse_json, validate_grounded_issues

    invalid = {"choice": None, "issues": [], "valid": False}
    try:
        value = parse_json(text)
    except ValueError:
        return invalid
    if not isinstance(value, dict) or set(value) != {"choice", "issues"}:
        return invalid
    choice, issues = value["choice"], value["issues"]
    if not isinstance(choice, str) or choice not in {"A", "B", "tie", "neither"}:
        return invalid
    try:
        issues = validate_grounded_issues(issues, candidate_element_ids or {"A": [], "B": []})
    except ValueError:
        return invalid
    return {"choice": choice, "issues": issues, "valid": True}
