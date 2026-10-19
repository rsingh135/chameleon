"""Synthetic tasks must be valid designs (they pass the contract against
themselves) and deterministic for a seed."""

import json

from engine.eval import build_context, check_contract, score
from tools.synth import FAMILIES, generate


def test_deterministic():
    a = json.dumps(generate(6, seed=3), sort_keys=True)
    b = json.dumps(generate(6, seed=3), sort_keys=True)
    assert a == b


def test_sources_pass_contract_against_themselves():
    tasks = generate(len(FAMILIES) * 3, seed=0)
    for task in tasks:
        src = task["source"]
        ctx = build_context(src)
        violations = check_contract(src, src, src["canvas"], ctx)
        assert violations == [], (task["task_id"], [str(v) for v in violations])


def test_holdout_flag_matches_family_or_target():
    for task in generate(30, seed=1):
        if task["holdout"]:
            assert task["family"] in ("split_hero", "multi_image") or task["target_name"] in (
                "wide_skyscraper",
                "mobile_banner",
                "a4_landscape",
                "half_page",
            )


def test_sources_have_no_self_collisions():
    """A generated design is a stand-in for a designer's work: nothing in it
    may collide with anything else, or the harness would reward engines for
    preserving a defect."""
    for task in generate(40, seed=0):
        src = task["source"]
        m = score(src, src, build_context(src))["metrics"]
        assert m["raw_collisions"] == 0, (task["task_id"], task["family"])
