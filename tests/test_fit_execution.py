"""The document route measures native constraints then executes trained weights."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.fit_execution import _candidate_id, select_fitted_document
from engine.layout.fitted_columns import generate_fitted_column_states
from engine.resources import using_resources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fitted_document():
    # Explicitly user-exposed reserve: acceptance inference only, no training.
    folder = ROOT / "fixtures/open_posters/prepared/doory_modern-resume"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    target = {"width": 1080, "height": 1080}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        states = generate_fitted_column_states(source, target)
        assert len(states) >= 2
        candidates = [state.design for state in states]
        yield source, target, candidates


def test_actual_trained_policy_selects_only_measured_known_choices(fitted_document):
    source, target, candidates = fitted_document
    original = deepcopy(source)
    ids = {_candidate_id(design) for design in candidates}
    snapshot = {}
    result = select_fitted_document(source, target, candidates, ids, [], snapshot=snapshot)
    assert result is not None
    trace = result["trace"]
    assert trace["profile"] == "measured_document_columns"
    assert trace["marketing_pixel_profile_evaluated"] is False
    receipt = trace["fit_policy"]["receipt"]
    assert receipt["executed"] is True
    assert receipt["kind"] == "trained_pairwise_fit_policy"
    assert receipt["model_digest"] and receipt["model_sha256"]
    assert len(receipt["scores"]) >= 2
    expected = max(receipt["scores"], key=lambda row: (row["score"], row["plan_id"]))
    assert trace["selected_candidate_id"] == expected["plan_id"]
    assert _candidate_id(result["design"]) == expected["plan_id"]
    assert all(
        row["diagnostics"]["minimum_font"] >= trace["minimum_font_floor"]
        for row in snapshot["candidates"]
        if row["admitted"]
    )
    assert all(row["candidate_id"] in ids for row in snapshot["candidates"])
    assert source == original


def test_new_text_overlap_and_copy_changes_cannot_enter_the_document_menu(fitted_document):
    source, target, candidates = fitted_document
    valid = candidates[0]
    overlap = deepcopy(valid)
    texts = [e for e in overlap["elements"] if e["type"] == "text"]
    education = next(e for e in texts if e["content"] == "EDUCATION")
    profile = next(e for e in texts if e["content"] == "PROFILE")
    education.update(x=profile["x"], y=profile["y"])
    rewritten = deepcopy(valid)
    next(e for e in rewritten["elements"] if e["type"] == "text")["content"] = "Changed source copy"
    choices = [valid, overlap, rewritten]
    ids = {_candidate_id(design) for design in choices}
    snapshot = {}
    result = select_fitted_document(source, target, choices, ids, [], snapshot=snapshot)
    assert result is not None and result["design"] == valid
    rejected = [row for row in snapshot["candidates"] if not row["admitted"]]
    assert len(rejected) == 2
    assert any(row["diagnostics"]["new_text_frame_overlaps"] for row in rejected)
    assert any(
        "frozen_field_changed" in failure
        for row in rejected
        for failure in row["diagnostics"]["failures"]
    )


def test_absent_or_invalid_trained_decision_declines_without_snapshot_mutation(
    fitted_document, monkeypatch
):
    from engine import fit_policy

    source, target, candidates = fitted_document
    ids = {_candidate_id(design) for design in candidates}
    snapshot = {"untouched": True}
    monkeypatch.setattr(
        fit_policy,
        "choose_fit_plan",
        lambda *_a, **_k: {"decision": None, "receipt": {"executed": False, "status": "no_model"}},
    )
    assert select_fitted_document(source, target, candidates, ids, [], snapshot=snapshot) is None
    assert snapshot == {"untouched": True}
    monkeypatch.setattr(
        fit_policy,
        "choose_fit_plan",
        lambda _s, _t, menu, _c: {
            "decision": {
                "schema_version": 1,
                "menu_digest": menu["menu_digest"],
                "plan_id": "invented",
            },
            "receipt": {"executed": True},
        },
    )
    assert select_fitted_document(source, target, candidates, ids, [], snapshot=snapshot) is None
    assert snapshot == {"untouched": True}
