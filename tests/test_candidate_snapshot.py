"""Research snapshots observe the real selector without changing resize behavior."""

import builtins
import hashlib
import json
import sys
from copy import deepcopy

import pytest

import engine.resize as engine
from engine.layout.grouped import GroupedState
from engine.layout.relationships import build_relationship_graph
from engine.resources import get_resources, using_resources


@pytest.fixture(autouse=True)
def isolated_pool(monkeypatch):
    for factory in (
        "generate_composition_states",
        "generate_grouped_states",
        "generate_focal_states",
        "generate_source_guides",
    ):
        monkeypatch.setattr(engine, factory, lambda *_args: ())
    engine._resize_cached.cache_clear()
    yield
    engine._resize_cached.cache_clear()


@pytest.fixture
def task():
    return {
        "canvas": {"width": 200, "height": 100},
        "elements": [
            {
                "id": "panel",
                "type": "shape",
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 80,
                "shape_type": "rectangle",
                "fill_color": "rgba(255,0,0,1)",
                "_note": "Café ✓",
            }
        ],
    }, {"width": 300, "height": 100}


def _id(design):
    encoded = json.dumps(design, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "candidate-" + hashlib.sha256(encoded).hexdigest()


def test_snapshot_matches_default_baseline_without_touching_result_cache_or_inputs(task):
    source, target = task
    before = deepcopy((source, target))
    expected = engine.resize_with_state(source, target)
    cache = engine._resize_cached.cache_info()
    imported = set(sys.modules)
    snapshot = engine.collect_candidate_snapshot(source, target)
    assert engine._resize_cached.cache_info() == cache
    assert not ({"torch", "transformers", "peft"} & (set(sys.modules) - imported))
    assert (source, target) == before
    assert snapshot["schema_version"] == 1
    assert snapshot["target_canvas"] == target
    rows = {row["candidate_id"]: row for row in snapshot["candidates"]}
    assert len(rows) == len(snapshot["candidates"])
    baseline = rows[snapshot["baseline_candidate_id"]]
    assert baseline["design"] == expected["design"]
    assert baseline["state"] == expected["state"]
    assert baseline["contract_valid"] and baseline["admitted"]
    assert snapshot["baseline_candidate_id"] in snapshot["admitted_finalist_ids"]
    assert all(row["candidate_id"] == _id(row["design"]) for row in rows.values())
    assert set(snapshot["admitted_finalist_ids"]) == {
        key for key, row in rows.items() if row["admitted"]
    }
    json.dumps(snapshot, allow_nan=False)
    again = engine.collect_candidate_snapshot(source, target)
    assert snapshot == again
    baseline["design"]["elements"][0]["x"] = 999
    baseline["state"]["cluster_axes"]["invented"] = "horizontal"
    assert engine.resize_with_state(source, target) == expected
    assert engine.collect_candidate_snapshot(source, target) == again


def test_invalid_proposals_are_explicitly_rejected_and_exact_duplicates_keep_origins(
    task, monkeypatch
):
    source, target = task
    valid = {**deepcopy(source), "canvas": target}
    invalid = deepcopy(valid)
    invalid["elements"][0]["fill_color"] = "rgba(0,0,0,1)"
    monkeypatch.setattr(engine, "generate_candidates", lambda *_: [invalid, valid, deepcopy(valid)])
    snapshot = engine.collect_candidate_snapshot(source, target)
    assert len(snapshot["candidates"]) == 2
    bad = next(row for row in snapshot["candidates"] if not row["contract_valid"])
    assert not bad["admitted"]
    assert bad["diagnostics"]["evaluation"]["violations"]
    good = next(row for row in snapshot["candidates"] if row["contract_valid"])
    assert len(good["provenance"]) == 2
    assert snapshot["baseline_candidate_id"] == good["candidate_id"]


def test_identity_snapshot_has_unchanged_baseline(task):
    source, _target = task
    snapshot = engine.collect_candidate_snapshot(source, source["canvas"])
    assert len(snapshot["candidates"]) == 1
    baseline = snapshot["candidates"][0]
    assert baseline["design"] == source
    assert baseline["diagnostics"]["identity"]
    assert snapshot["admitted_finalist_ids"] == [baseline["candidate_id"]]
    assert baseline["state"] == engine.resize_with_state(source, source["canvas"])["state"]


def test_generated_state_axes_and_compact_provenance_match_resize(task, monkeypatch):
    _source, target = task
    source = {
        "canvas": {"width": 200, "height": 100},
        "elements": [
            {
                "id": "heading",
                "type": "text",
                "x": 10,
                "y": 10,
                "width": 180,
                "height": 35,
                "content": "Heading",
                "font_family": "Solstice Sans",
                "font_size": 28,
                "color": "rgba(0,0,0,1)",
            },
            {
                "id": "body",
                "type": "text",
                "x": 20,
                "y": 55,
                "width": 160,
                "height": 25,
                "content": "Body",
                "font_family": "Solstice Sans",
                "font_size": 16,
                "color": "rgba(0,0,0,1)",
            },
        ],
    }
    cluster = build_relationship_graph(source).clusters[0].id
    candidate = {**deepcopy(source), "canvas": target}
    candidate["elements"][1].update(x=205, y=10, width=80)
    state = GroupedState(
        "research-proof",
        "grouped",
        ((cluster, "horizontal"),),
        json.dumps(candidate, separators=(",", ":")),
        ("preserved-source-pair",),
    )
    monkeypatch.setattr(engine, "generate_candidates", lambda *_: [deepcopy(candidate)])
    monkeypatch.setattr(engine, "generate_grouped_states", lambda *_: (state,))
    snapshot = engine.collect_candidate_snapshot(source, target)
    row = snapshot["candidates"][0]
    assert len(snapshot["candidates"]) == 1
    assert row["state"]["name"] == "research-proof"
    assert row["state"]["cluster_axes"] == {cluster: "horizontal"}
    assert row["state"]["sizing_exceptions"] == ["preserved-source-pair"]
    assert len(row["provenance"]) == 2
    assert row["state"] == engine.resize_with_state(source, target)["state"]


def test_active_and_explicit_resource_contexts_are_restored(task, tmp_path):
    source, target = task
    original = get_resources()
    with using_resources(tmp_path, original.fonts_dir):
        active = get_resources()
        first = engine.collect_candidate_snapshot(source, target)
        assert get_resources() == active
        second = engine.collect_candidate_snapshot(
            source, target, assets_dir=original.assets_dir, fonts_dir=original.fonts_dir
        )
        assert first["baseline_candidate_id"] == second["baseline_candidate_id"]
        assert get_resources() == active
    assert get_resources() == original


def test_no_valid_baseline_raises_instead_of_inventing_admission(task, monkeypatch):
    source, target = task
    monkeypatch.setattr(
        engine, "generate_candidates", lambda *_: [{"canvas": target, "elements": []}]
    )
    with pytest.raises(ValueError, match="id_missing"):
        engine.collect_candidate_snapshot(source, target)
    with pytest.raises(ValueError, match="id_missing"):
        engine.resize(source, target)


def test_snapshot_requires_evaluation_but_default_development_fallback_still_works(
    task, monkeypatch
):
    source, target = task
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "engine.eval":
            raise ImportError("Evaluator deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="full evaluator"):
        engine.collect_candidate_snapshot(source, target)
    assert engine.resize(source, target)["canvas"] == target


def test_nonfinite_proposal_cannot_publish_a_safe_json_snapshot(task, monkeypatch):
    source, target = task
    source["elements"][0].update(x=250, width=30, height=30)
    valid = {**deepcopy(source), "canvas": target}
    invalid = deepcopy(valid)
    invalid["elements"][0]["x"] = float("nan")
    monkeypatch.setattr(engine, "generate_candidates", lambda *_: [invalid, valid])
    # Abort a non-JSON research artifact instead of rewriting exact geometry or
    # claiming that an unserializable rejected proposal is safe evidence.
    with pytest.raises(ValueError, match="JSON"):
        engine.collect_candidate_snapshot(source, target)
    assert engine.resize(source, target) == valid


@pytest.mark.parametrize("part", ["source", "target"])
def test_nonfinite_input_is_rejected_before_collection_generation(task, monkeypatch, part):
    source, target = task
    if part == "source":
        source["elements"][0]["_note"] = float("inf")
    else:
        target["width"] = float("nan")

    def must_not_generate(*_args, **_kwargs):
        pytest.fail("Non-JSON input reached candidate generation")

    monkeypatch.setattr(engine, "generate_candidates", must_not_generate)
    with pytest.raises(ValueError, match="JSON"):
        engine.collect_candidate_snapshot(source, target)


def test_nonfinite_assessment_cannot_escape_in_snapshot_diagnostics(task, monkeypatch):
    import engine.eval as evaluation

    source, target = task
    monkeypatch.setattr(
        evaluation,
        "score",
        lambda *_args, **_kwargs: {
            "violations": [],
            "metrics": {"raw_collisions": 0, "text_size": 1},
            "total": float("nan"),
        },
    )
    with pytest.raises(ValueError, match="JSON"):
        engine.collect_candidate_snapshot(source, target)
