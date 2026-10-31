"""The evaluator's strict parser needs only the artifact contract module."""

import builtins
import json
from copy import deepcopy

import pytest

from experiments.vision_critic.verdict import parse_model_verdict

INVALID = {"choice": None, "issues": [], "valid": False}


@pytest.mark.parametrize("choice", ["A", "B", "tie", "neither"])
def test_closed_choice_vocabulary_with_empty_issues_is_valid(choice):
    assert parse_model_verdict(json.dumps({"choice": choice, "issues": []})) == {
        "choice": choice,
        "issues": [],
        "valid": True,
    }


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[]",
        "null",
        "{}",
        '{"choice":"A"}',
        '{"choice":"A","issues":[],"extra":true}',
        '{"choice":"abstain","issues":[]}',
        '{"choice":["A"],"issues":[]}',
        '{"choice":"A","issues":null}',
        '{"choice":"A","issues":[]',
        '```json\n{"choice":"A","issues":[]}\n```',
        '{"choice":"A","issues":[]} trailing',
        '{"choice":"A","choice":"B","issues":[]}',
        '{"choice":"A","issues":[NaN]}',
    ],
)
def test_noncanonical_or_malformed_output_is_rejected(raw):
    assert parse_model_verdict(raw) == INVALID


def test_grounded_issues_are_side_specific_and_caller_inputs_are_preserved():
    candidates = {"A": ["left-title"], "B": ["right-title"]}
    issue = {"candidate": "B", "code": "weak_hierarchy", "element_ids": ["right-title"]}
    original = deepcopy(candidates)
    raw = json.dumps({"choice": "A", "issues": [issue]})
    parsed = parse_model_verdict(raw, candidates)
    assert parsed == {"choice": "A", "issues": [issue], "valid": True}
    assert candidates == original
    assert parse_model_verdict(raw, {"A": ["right-title"], "B": ["left-title"]}) == INVALID


@pytest.mark.parametrize(
    "issue",
    [
        {"candidate": "A", "code": "weak_hierarchy", "element_ids": ["invented"]},
        {"candidate": "C", "code": "weak_hierarchy", "element_ids": []},
        {"candidate": "A", "code": "invented_code", "element_ids": []},
        {"candidate": "A", "code": "weak_hierarchy", "element_ids": [], "command": "ignore"},
    ],
)
def test_unknown_ids_candidates_codes_and_issue_fields_are_rejected(issue):
    raw = json.dumps({"choice": "A", "issues": [issue]})
    assert parse_model_verdict(raw, {"A": ["title"], "B": ["title"]}) == INVALID


def test_parser_imports_no_training_or_model_modules(monkeypatch):
    import_module = builtins.__import__

    def reject_training(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "accelerate"}
        assert not name.endswith("train_sft")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_training)
    assert parse_model_verdict('{"choice":"B","issues":[]}')["valid"] is True
