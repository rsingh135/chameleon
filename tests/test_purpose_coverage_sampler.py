"""Coverage and checkpoint regressions for label-policy-independent TRAIN sampling."""

import copy
import json
import random

import pytest

from tools.purpose_coverage_sampler import ClassCoverageSampler


def records():
    rows = []
    for family, count in {
        "long-body-family": 8,
        "event-family": 3,
        "lead-in-family": 2,
        "rare-lead-in-family": 1,
    }.items():
        rows.extend(
            {
                "record_id": f"{family}-{i}",
                "label": "support",
                "family_id": family,
                "split": "train",
            }
            for i in range(count)
        )
    rows.extend(
        {"record_id": f"body-{i}", "label": "body", "family_id": "body-family", "split": "train"}
        for i in range(20)
    )
    return rows


def test_all_four_support_families_early_and_all_fourteen_records_before_repeat():
    rows = records()
    family = {r["record_id"]: r["family_id"] for r in rows}
    expected = {r["record_id"] for r in rows if r["label"] == "support"}
    sampler = ClassCoverageSampler(rows, seed=1729)
    draws = [sampler.draw("support") for _ in range(4)]
    assert len({family[r] for r in draws[:4]}) == 4
    partial = sampler.coverage_report()["classes"]["support"]
    assert partial["unique_records_seen"] == partial["unique_families_seen"] == 4
    assert len(partial["missing_record_ids"]) == 10
    assert partial["minimum_additional_draws_for_first_coverage"] == 10
    draws.extend(sampler.draw("support") for _ in range(10))
    assert {"lead-in-family-0", "lead-in-family-1", "rare-lead-in-family-0"} <= set(draws[:7])
    assert len(set(draws)) == 14
    assert set(draws) == expected
    report = sampler.coverage_report()["classes"]["support"]
    assert report["missing_record_ids"] == []
    assert report["unique_records_seen"] == 14
    assert report["repeated_exposures"] == 0
    assert report["minimum_draws_for_full_class_coverage"] == 14
    assert sampler.draw("support") in expected
    assert sampler.coverage_report()["classes"]["support"]["repeated_exposures"] == 1


def test_interleaved_class_policy_keeps_minority_anchors_and_each_new_deck_unique():
    rows = records()
    by_label = {
        label: {r["record_id"] for r in rows if r["label"] == label}
        for label in ("support", "body")
    }
    sampler = ClassCoverageSampler(rows, seed=1729)
    drawn = {"support": [], "body": []}
    # The worker controls class selection; this uneven schedule does not reset either deck.
    for label in ["body", "body", "support"] * 40:
        drawn[label].append(sampler.draw(label))
    for label, ids in drawn.items():
        n = len(by_label[label])
        for start in range(0, len(ids), n):
            part = ids[start : start + n]
            assert len(set(part)) == len(part)
            if len(part) == n:
                assert set(part) == by_label[label]
    assert {"lead-in-family-0", "lead-in-family-1", "rare-lead-in-family-0"} <= set(
        drawn["support"][:14]
    )


@pytest.mark.parametrize("prefix_length", [0, 4, 14, 15, 47])
def test_json_checkpoint_resume_reproduces_next_draws_across_cycle_boundaries(prefix_length):
    rows = records()
    original = ClassCoverageSampler(rows, seed=1729)
    for label in (["support", "body", "support"] * 16)[:prefix_length]:
        original.draw(label)
    saved = json.loads(json.dumps(original.state_dict(), allow_nan=False))
    resumed = ClassCoverageSampler.from_state(list(reversed(rows)), saved)
    restored = ClassCoverageSampler(rows, seed=1729).restore(saved)
    future = ["body", "support", "support", "body"] * 30
    expected = [original.draw(label) for label in future]
    assert [resumed.draw(label) for label in future] == expected
    assert [restored.draw(label) for label in future] == expected
    assert original.state_dict() == resumed.state_dict() == restored.state_dict()


@pytest.mark.parametrize("change", ["record_id", "label", "family_id"])
def test_resume_rejects_changed_frozen_record_identity(change):
    rows = records()
    state = ClassCoverageSampler(rows, seed=1729).state_dict()
    altered = copy.deepcopy(rows)
    altered[0][change] += "-changed"
    with pytest.raises(ValueError, match="dataset identity"):
        ClassCoverageSampler.from_state(altered, state)


@pytest.mark.parametrize("split", ["validation", "dev", "test", "reserve", None])
def test_heldout_or_unspecified_split_is_never_admitted(split):
    rows = records()
    rows[0]["split"] = split
    with pytest.raises(ValueError, match="TRAIN"):
        ClassCoverageSampler(rows, seed=1729)


def test_duplicate_ids_are_rejected_instead_of_reweighting_a_record():
    rows = records()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="Duplicate record_id"):
        ClassCoverageSampler(rows, seed=1729)


@pytest.mark.parametrize("fault", ["deck", "cursor", "exposures", "rng", "total", "family_order"])
def test_invalid_restore_is_rejected_atomically(fault):
    sampler = ClassCoverageSampler(records(), seed=1729)
    sampler.draw("support")
    before = sampler.state_dict()
    damaged = copy.deepcopy(before)
    support = damaged["classes"]["support"]
    if fault == "deck":
        support["deck"][1] = support["deck"][0]
    elif fault == "cursor":
        support["cursor"] = 999
    elif fault == "exposures":
        support["exposures"][support["deck"][0]] += 1
    elif fault == "rng":
        damaged["rng_state"][1][0] = -1
    elif fault == "total":
        damaged["draw_count"] += 1
    else:
        # Still a permutation: corruption must not defeat the family-first invariant.
        support["deck"] = sorted(support["deck"])
    with pytest.raises(ValueError):
        sampler.restore(damaged)
    assert sampler.state_dict() == before


def test_input_order_private_rng_and_detached_state_do_not_change_sampling():
    rows = records()
    global_state = random.getstate()
    first = ClassCoverageSampler(rows, seed=1729)
    second = ClassCoverageSampler(list(reversed(rows)), seed=1729)
    state = first.state_dict()
    state["classes"]["support"]["deck"].clear()
    state["rng_state"][1].clear()
    assert [first.draw("support") for _ in range(30)] == [second.draw("support") for _ in range(30)]
    assert random.getstate() == global_state
    with pytest.raises(ValueError, match="Unknown"):
        first.draw("reserve")
