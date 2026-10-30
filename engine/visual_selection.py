"""A visual critic can choose only the engine's already admitted alternatives.

Snapshots are internal artifacts from ``collect_candidate_snapshot``. Model
output contains a known ID, never candidate geometry or executable instructions.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _candidate_id(design):
    return "candidate-" + _digest(design)


def _records(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported candidate snapshot")
    if len(_canonical(snapshot).encode()) > 24 * 1024 * 1024:
        raise ValueError("Candidate snapshot exceeds the local byte limit")
    rows = snapshot.get("candidates")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2048:
        raise ValueError("Candidate snapshot needs a bounded candidate list")
    known = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("design"), dict):
            raise ValueError("Malformed candidate record")
        identity = row.get("candidate_id")
        if identity != _candidate_id(row["design"]) or identity in known:
            raise ValueError("Candidate identity does not bind its actual design")
        known[identity] = row
    admitted = snapshot.get("admitted_finalist_ids")
    if (
        not isinstance(admitted, list)
        or any(not isinstance(identity, str) for identity in admitted)
        or len(set(admitted)) != len(admitted)
    ):
        raise ValueError("Invalid admitted candidate set")
    for identity in admitted:
        row = known.get(identity)
        if not row or row.get("contract_valid") is not True or row.get("admitted") is not True:
            raise ValueError("Critic may see only actual admitted candidates")
    if snapshot.get("baseline_candidate_id") not in known:
        raise ValueError("Snapshot selection is missing")
    return known, admitted


def _visual(row):
    return row.get("state", {}).get("name", "").startswith("visual/") or any(
        origin.get("name", "").startswith("visual/") for origin in row.get("provenance", [])
    )


def build_visual_critic_menu(snapshot, candidate_ids=None):
    """Bind at most two distinct admitted designs; never invent a choice.

    Prefer the engine winner and a model-proposed alternative when both are
    admitted. Labels deliberately conceal which producer generated a design.
    ``candidate_ids`` permits a controlled order swap for visual-bias testing.
    """
    known, admitted = _records(snapshot)
    if candidate_ids is not None:
        ids = list(candidate_ids)
        if len(ids) > 2 or len(set(ids)) != len(ids) or any(i not in admitted for i in ids):
            raise ValueError("Critic comparison must use at most two distinct admitted IDs")
    else:
        first = snapshot["baseline_candidate_id"]
        ids = [first] if first in admitted else []
        rest = sorted(
            (i for i in admitted if i not in ids), key=lambda i: (not _visual(known[i]), i)
        )
        ids = (ids + rest)[:2]
    menu = {
        "schema_version": 1,
        "task": "critic",
        "source_digest": snapshot["source_digest"],
        "target_canvas": deepcopy(snapshot["target_canvas"]),
        "choice_available": len(ids) >= 2,
        "plans": [
            {
                "plan_id": identity,
                "label": f"Alternative {index + 1}",
                "description": f"Rendered alternative {index + 1}",
            }
            for index, identity in enumerate(ids)
        ],
    }
    menu["menu_digest"] = _digest(menu)
    return menu


def apply_visual_selection(
    source, target, snapshot, decision, *, assets_dir=None, fonts_dir=None, critic_menu=None
):
    """Apply one bounded visual choice without widening the selector's gates."""
    from engine.eval import check_contract
    from engine.resources import using_resources

    source_digest = hashlib.sha256(
        json.dumps(source, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if snapshot.get("source_digest") != source_digest or snapshot.get("target_canvas") != target:
        raise ValueError("Critic snapshot belongs to another source or target")
    known, _ = _records(snapshot)
    menu = build_visual_critic_menu(snapshot)
    if critic_menu is not None:
        if not isinstance(critic_menu, dict) or not isinstance(critic_menu.get("plans"), list):
            raise ValueError("Malformed critic comparison")
        try:
            expected = build_visual_critic_menu(
                snapshot, candidate_ids=[p["plan_id"] for p in critic_menu["plans"]]
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed critic comparison") from exc
        if critic_menu != expected:
            raise ValueError("Critic comparison is not bound to admitted snapshot designs")
        menu = expected
    required = {"schema_version", "menu_digest", "plan_id"}
    if (
        not isinstance(decision, dict)
        or not required <= decision.keys()
        or decision.keys() - required - {"rationale"}
        or type(decision["schema_version"]) is not int
        or decision["schema_version"] != 1
        or decision["menu_digest"] != menu["menu_digest"]
        or len(_canonical(decision).encode()) > 16_384
    ):
        raise ValueError("Invalid or stale visual selection")
    if "rationale" in decision and (
        not isinstance(decision["rationale"], str) or len(decision["rationale"]) > 1000
    ):
        raise ValueError("Visual rationale must be bounded text")
    choice = decision["plan_id"]
    if not isinstance(choice, str) or choice not in {p["plan_id"] for p in menu["plans"]}:
        raise ValueError("Visual choice was not in the admitted rendered comparison")
    row = known[choice]
    with using_resources(assets_dir, fonts_dir):
        if check_contract(source, row["design"], target_canvas=target):
            raise ValueError("Selected visual candidate no longer passes the hard contract")
        if snapshot.get("strict_bounds"):
            from engine.stage_one import strictly_in_bounds

            if not strictly_in_bounds(row["design"]):
                raise ValueError("Selected visual candidate violates strict stage-one bounds")
    return {
        "design": deepcopy(row["design"]),
        "state": deepcopy(row["state"]),
        "trace": {
            "critic_menu_digest": menu["menu_digest"],
            "selected_candidate_id": choice,
            "selection_changed": choice != snapshot["baseline_candidate_id"],
            "choice_available": menu["choice_available"],
            "quality_improved": None,
        },
    }
