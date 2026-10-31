"""The graph's public seam: real verification and deliberately untrusted role outputs."""

import json
from copy import deepcopy

import pytest

from engine.resources import DEFAULT_ASSETS_DIR, DEFAULT_FONTS_DIR
from experiments.agentic_graph.fundamentals import fundamental_tasks
from experiments.agentic_graph.graph import GraphLimits, TaskCriteria, digest, run_graph


def source_and_design():
    source = fundamental_tasks()[0]["source"]
    bad = deepcopy(source)
    for element in bad["elements"]:
        element["font_size"] = 8
    return source, bad, deepcopy(source)


class Engine:
    def __init__(self, source, bad, good):
        self.source, self.bad, self.good = source, bad, good

    def row(self, design):
        return {
            "candidate_id": "candidate-" + digest(design),
            "design": deepcopy(design),
            "state": {"name": "visual/test"},
            "contract_valid": True,
            "admitted": True,
        }

    def snapshot(self, source, target, *, visual_plan=None, strict_bounds=False):
        assert strict_bounds is True
        chosen = self.good if visual_plan and visual_plan["plan_id"] == "repair" else self.bad
        rows = [self.row(self.bad)]
        if chosen != self.bad:
            rows.append(self.row(chosen))
        return {
            "baseline_candidate_id": rows[-1]["candidate_id"],
            "candidates": rows,
            "admitted_finalist_ids": [row["candidate_id"] for row in rows],
        }

    def menu(self, source, target):
        return {"status": "ready", "plans": [{"plan_id": "repair"}, {"plan_id": "alternate"}]}

    def critic_menu(self, snapshot):
        return {"plans": [{"plan_id": row["candidate_id"]} for row in snapshot["candidates"]]}

    def select(self, source, target, snapshot, decision, **kwargs):
        row = next(
            row for row in snapshot["candidates"] if row["candidate_id"] == decision["plan_id"]
        )
        return {"design": deepcopy(row["design"]), "state": row["state"]}

    def api(self):
        return {name: getattr(self, name) for name in ("snapshot", "menu", "critic_menu", "select")}


class Roles:
    def __init__(self, plans=("repair",), bad_verifier=False):
        self.plans, self.feedback, self.bad_verifier = iter(plans), [], bad_verifier

    def analyze(self, **kwargs):
        assert "target" not in kwargs and "feedback" not in kwargs
        return {"valid": True, "executed": True, "analysis": {"test": "source_only"}}

    def organize(self, **kwargs):
        self.feedback.append(deepcopy(kwargs["feedback"]))
        return {"valid": True, "executed": True, "decision": {"plan_id": next(self.plans)}}

    def verify(self, **kwargs):
        return {
            "valid": not self.bad_verifier,
            "executed": True,
            "decision": {"plan_id": kwargs["critic_menu"]["plans"][-1]["plan_id"]},
        }


def run(tmp_path, engine, roles, **kwargs):
    return run_graph(
        engine.source,
        engine.source["canvas"],
        tmp_path / "run",
        assets_dir=DEFAULT_ASSETS_DIR,
        fonts_dir=DEFAULT_FONTS_DIR,
        api=engine.api(),
        roles=roles,
        implementation_hashes=lambda: {"test": "fixed"},
        **kwargs,
    )


def test_three_roles_repair_failed_readability_and_persist_hash_linked_messages(tmp_path):
    source, bad, good = source_and_design()
    roles = Roles()
    result = run(tmp_path, Engine(source, bad, good), roles)
    assert result["status"] == "passed" and result["all_three_role_protocols_completed"]
    assert not result["all_three_roles_executed"]
    assert result["evidence_scope"] == "injected_protocol_test"
    assert result["design"] == good
    assert "minimum_font" in {row["code"] for row in roles.feedback[0]["failed_checks"]}
    messages = [
        json.loads(line) for line in (tmp_path / "run/messages.jsonl").read_text().splitlines()
    ]
    previous = None
    for message in messages:
        assert message["previous_message_sha256"] == previous
        previous = message.pop("message_sha256")
        assert digest(message) == previous
    assert (tmp_path / "run/final.png").is_file()
    assert json.loads((tmp_path / "run/state.json").read_text())["status"] == "passed"


def test_repeated_plan_stops_without_falsely_passing_task(tmp_path):
    source, bad, _ = source_and_design()
    result = run(tmp_path, Engine(source, bad, bad), Roles(("alternate", "alternate")))
    assert not result["task_passed"]
    assert result["stop_reason"] == "repeated_plan"
    assert result["design"] == bad


def test_no_model_calls_are_reported_as_trained_graph_success(tmp_path):
    source, _, good = source_and_design()
    result = run(tmp_path, Engine(source, good, good), None)
    assert result["status"] == "baseline_passed"
    assert not result["all_three_roles_executed"]


def test_model_cannot_pass_a_copy_changing_candidate(tmp_path):
    source, bad, good = source_and_design()
    good["elements"][0]["content"] = "Different copy"
    result = run(tmp_path, Engine(source, bad, good), Roles(("repair", "repair")))
    assert not result["task_passed"]
    assert result["design"] == bad
    assert all(row["content"] != "Different copy" for row in result["design"]["elements"])


def test_invalid_verifier_cannot_promote_an_unreviewed_good_alternative(tmp_path):
    source, bad, good = source_and_design()
    result = run(tmp_path, Engine(source, bad, good), Roles(bad_verifier=True))
    assert not result["task_passed"]
    assert result["stop_reason"] == "invalid_verifier"
    assert result["design"] == bad
    assert not result["all_three_roles_executed"]


def test_valid_flag_without_actual_execution_is_rejected(tmp_path):
    source, bad, good = source_and_design()

    class UnexecutedRoles(Roles):
        def analyze(self, **kwargs):
            return {"valid": True, "executed": False, "analysis": {"test": "forged"}}

    result = run(tmp_path, Engine(source, bad, good), UnexecutedRoles())
    assert not result["task_passed"] and not result["all_three_roles_executed"]
    assert result["stop_reason"] == "analyzer_abstained_or_invalid"


def test_existing_artifacts_are_preserved_instead_of_overwritten(tmp_path):
    source, bad, good = source_and_design()
    run(tmp_path, Engine(source, bad, good), None)
    before = (tmp_path / "run/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path, Engine(source, bad, good), None)
    assert (tmp_path / "run/report.json").read_bytes() == before


@pytest.mark.parametrize(
    "kwargs", [{"max_rounds": 0}, {"max_rounds": 99}, {"max_seconds": float("nan")}]
)
def test_unbounded_retry_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GraphLimits(**kwargs)


def test_nonfinite_metric_threshold_is_rejected():
    with pytest.raises(ValueError):
        TaskCriteria(minimum_metrics={"collision_free": float("nan")})
