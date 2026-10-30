"""Visual choices only select the engine's actual admitted, source-bound designs."""

import json
from copy import deepcopy

import pytest

from engine.resize import _candidate_id
from engine.visual_selection import apply_visual_selection, build_visual_critic_menu


def snapshot_case():
    import hashlib

    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "paint",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 20,
                "fill_color": "#cc2200",
                "opacity": 1,
            }
        ],
    }
    alternative = deepcopy(source)
    alternative["elements"][0]["x"] = 60
    designs = [source, alternative]
    identities = [_candidate_id(d) for d in designs]
    snapshot = {
        "schema_version": 1,
        "source_digest": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "target_canvas": source["canvas"],
        "baseline_candidate_id": identities[0],
        "admitted_finalist_ids": identities,
        "candidates": [
            {
                "candidate_id": identity,
                "design": design,
                "contract_valid": True,
                "admitted": True,
                "state": {"name": "baseline" if i == 0 else "visual/alternative"},
                "selection": {"safety": [1], "fidelity": 1.0},
                "diagnostics": {},
            }
            for i, (identity, design) in enumerate(zip(identities, designs, strict=True))
        ],
    }
    return source, snapshot


def test_critic_can_select_only_a_distinct_admitted_design_without_mutating_snapshot():
    source, snapshot = snapshot_case()
    original = deepcopy(snapshot)
    menu = build_visual_critic_menu(snapshot)
    assert len(menu["plans"]) == 2
    decision = {
        "schema_version": 1,
        "menu_digest": menu["menu_digest"],
        "plan_id": snapshot["admitted_finalist_ids"][1],
        "rationale": "More space.",
    }
    result = apply_visual_selection(source, source["canvas"], snapshot, decision)
    assert result["design"]["elements"][0]["x"] == 60
    assert result["trace"]["selection_changed"] is True
    result["design"]["elements"][0]["x"] = 3
    assert snapshot == original


@pytest.mark.parametrize("mutation", ["unknown", "stale", "extra", "unadmitted", "pixels"])
def test_visual_choice_cannot_forge_ids_menus_admission_or_candidate_geometry(mutation):
    source, snapshot = snapshot_case()
    menu = build_visual_critic_menu(snapshot)
    decision = {
        "schema_version": 1,
        "menu_digest": menu["menu_digest"],
        "plan_id": snapshot["admitted_finalist_ids"][1],
    }
    if mutation == "unknown":
        decision["plan_id"] = "invented"
    elif mutation == "stale":
        decision["menu_digest"] = "f" * 64
    elif mutation == "extra":
        decision["elements"] = []
    elif mutation == "unadmitted":
        snapshot["admitted_finalist_ids"] = snapshot["admitted_finalist_ids"][:1]
        snapshot["candidates"][1]["admitted"] = False
    else:
        snapshot["candidates"][1]["design"]["elements"][0]["x"] = -30
    with pytest.raises(ValueError):
        apply_visual_selection(source, source["canvas"], snapshot, decision)


def test_a_single_safe_option_reports_no_choice_instead_of_model_selection():
    source, snapshot = snapshot_case()
    snapshot["admitted_finalist_ids"] = snapshot["admitted_finalist_ids"][:1]
    snapshot["candidates"][1]["admitted"] = False
    menu = build_visual_critic_menu(snapshot)
    assert menu["choice_available"] is False
    assert len(menu["plans"]) == 1


def test_controlled_order_swap_preserves_the_selected_candidate_identity():
    source, snapshot = snapshot_case()
    ids = list(reversed(snapshot["admitted_finalist_ids"]))
    menu = build_visual_critic_menu(snapshot, candidate_ids=ids)
    decision = {"schema_version": 1, "menu_digest": menu["menu_digest"], "plan_id": ids[0]}
    result = apply_visual_selection(source, source["canvas"], snapshot, decision, critic_menu=menu)
    assert result["design"]["elements"][0]["x"] == 60


def test_strict_critic_rejects_authored_bleed_allowed_by_the_legacy_contract():
    source, snapshot = snapshot_case()
    source["elements"][0]["x"] = -5
    import hashlib

    snapshot["source_digest"] = hashlib.sha256(
        json.dumps(source, sort_keys=True).encode()
    ).hexdigest()
    snapshot["candidates"][0]["design"] = deepcopy(source)
    identity = _candidate_id(source)
    snapshot["candidates"][0]["candidate_id"] = identity
    snapshot["baseline_candidate_id"] = identity
    snapshot["admitted_finalist_ids"][0] = identity
    snapshot["strict_bounds"] = True
    menu = build_visual_critic_menu(snapshot)
    decision = {"schema_version": 1, "menu_digest": menu["menu_digest"], "plan_id": identity}
    with pytest.raises(ValueError, match="strict stage-one bounds"):
        apply_visual_selection(source, source["canvas"], snapshot, decision)
