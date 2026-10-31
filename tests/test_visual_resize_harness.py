"""Activation/control tests; fake decisions never constitute model-quality proof."""

import sys
from copy import deepcopy

import pytest
from PIL import Image

from engine.resources import DEFAULT_FONTS_DIR
from engine.visual_plan import source_digest
from tools import scene_harness as scene
from tools import visual_resize_harness as harness


@pytest.fixture
def document(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (20, 40), "#367ba5").save(assets / "photo.png")
    source = {
        "canvas": {"width": 100, "height": 200},
        "elements": [
            {
                "id": "image",
                "type": "image",
                "asset_id": "photo.png",
                "fit": "fill",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 200,
            }
        ],
    }
    target = {"width": 200, "height": 100}
    monkeypatch.setattr(harness, "_reserve_identities", lambda: (set(), set()))
    monkeypatch.setattr(harness, "_implementation_hashes", lambda: {"test": "a" * 64})
    return source, target, assets, DEFAULT_FONTS_DIR


class FakeEngine:
    def __init__(
        self, source, target, *, admitted=True, duplicate=False, sole_new=False, status="ready"
    ):
        self.source, self.target = deepcopy(source), target
        self.admitted, self.duplicate, self.sole_new = admitted, duplicate, sole_new
        self.status = status
        self.calls = []
        self.base = deepcopy(source)
        self.base["canvas"] = dict(target)
        self.base["elements"][0].update(width=50, height=100)
        self.changed = deepcopy(self.base)
        self.changed["elements"][0]["x"] = 150

    def menu(self, source, target):
        value = {
            "schema_version": 1,
            "source_digest": source_digest(source),
            "target_canvas": target,
            "status": self.status,
            "plans": [
                {"plan_id": "keep", "label": "Baseline"},
                {"plan_id": "new", "label": "Move semantic unit"},
            ],
        }
        value["menu_digest"] = scene._digest(value)
        return value

    def row(self, design, name):
        return {
            "candidate_id": harness._candidate_id(design),
            "design": deepcopy(design),
            "state": {"name": name},
            "contract_valid": True,
            "admitted": True,
        }

    def snapshot(self, source, target, *, visual_plan=None, strict_bounds=False):
        self.calls.append(deepcopy(visual_plan))
        base = self.row(self.base, "baseline")
        candidates = [base]
        admitted = [base["candidate_id"]]
        winner = base["candidate_id"]
        if visual_plan and visual_plan["plan_id"] != "keep":
            changed = self.row(self.base if self.duplicate else self.changed, "visual/new")
            candidates.append(changed)
            if self.admitted:
                admitted.append(changed["candidate_id"])
            if self.sole_new:
                admitted = [changed["candidate_id"]]
                winner = changed["candidate_id"]
        return {
            "schema_version": 1,
            "source_digest": source_digest(source),
            "target_canvas": target,
            "baseline_candidate_id": winner,
            "admitted_finalist_ids": admitted,
            "candidates": candidates,
        }

    def critic_menu(self, snapshot):
        ids = list(dict.fromkeys(snapshot["admitted_finalist_ids"]))
        value = {
            "schema_version": 1,
            "task": "critic",
            "source_digest": snapshot["source_digest"],
            "target_canvas": snapshot["target_canvas"],
            "plans": [{"plan_id": value, "label": f"Option{i}"} for i, value in enumerate(ids)],
        }
        value["menu_digest"] = scene._digest(value)
        return value

    def select(self, source, target, snapshot, decision):
        assert decision["plan_id"] in snapshot["admitted_finalist_ids"]
        row = next(r for r in snapshot["candidates"] if r["candidate_id"] == decision["plan_id"])
        return {
            "design": deepcopy(row["design"]),
            "state": row["state"],
            "trace": {"applied": True},
        }

    def api(self):
        return {
            "menu": self.menu,
            "snapshot": self.snapshot,
            "critic_menu": self.critic_menu,
            "select": self.select,
        }


def model_result(request, choice=None):
    return {
        "decision": {
            "schema_version": 1,
            "menu_digest": request["plan_menu"]["menu_digest"],
            "plan_id": choice or request["plans"][-1]["plan_id"],
            "rationale": "Synthetic activation fixture, not visual-quality evidence.",
        },
        "receipt": {
            "executed": True,
            "backend": "test_stub",
            "input_image_count": 3 if request["mode"] == "critique" else 1,
            "source_analysis_digest": request.get("source_analysis_digest"),
            "request_digest": scene._digest(request),
            "source_digest": request["source_digest"],
            "source_render_sha256": request["source_render"]["sha256"],
        },
    }


def execute(document, tmp_path, monkeypatch, engine=None, provider=None, **kwargs):
    source, target, assets, fonts = document
    engine = engine or FakeEngine(source, target)
    requests = []

    def fake_provider(command, request, timeout):
        requests.append(deepcopy(request))
        return provider(request) if provider else model_result(request)

    monkeypatch.setattr(harness, "_engine_api", engine.api)
    monkeypatch.setattr(harness, "run_provider", fake_provider)
    report = harness.inspect_visual_resize(
        source,
        assets,
        fonts,
        tmp_path / "evidence",
        targets=[target],
        command=[sys.executable, "-c", "pass"],
        **kwargs,
    )
    return report, requests, engine


def test_real_geometry_activation_and_three_critic_views_do_not_claim_quality(
    document,
    tmp_path,
    monkeypatch,
):
    original = deepcopy(document[0])
    report, requests, _ = execute(document, tmp_path, monkeypatch)
    row = report["targets"][0]
    assert [r["mode"] for r in requests] == ["plan", "critique"]
    critic = requests[1]
    assert len(critic["plans"]) == 2
    assert all("render" in choice for choice in critic["plans"])
    assert all("render" not in choice for choice in critic["plan_menu"]["plans"])
    assert critic["source_render"]["sha256"] == requests[0]["source_render"]["sha256"]
    assert row["model_executed"] and row["proposal_valid"]
    assert row["new_candidates_generated"] and row["model_candidate_selected"]
    assert row["layout_changed"] and not row["same_png"]
    assert row["quality_improved"] is None
    assert row["quality_review"]["status"] == "not_reviewed"
    assert row["fallback_reason"] is None
    assert row["status"] == "contract_valid"
    assert document[0] == original and report["resources_immutable"]


def test_valid_but_unadmitted_new_plan_is_an_honest_noop(document, tmp_path, monkeypatch):
    source, target, _, _ = document
    engine = FakeEngine(source, target, admitted=False)
    report, requests, _ = execute(document, tmp_path, monkeypatch, engine=engine)
    row = report["targets"][0]
    assert row["model_executed"] and row["proposal_valid"] and row["new_candidates_generated"]
    assert row["admitted_new_candidate_ids"] == []
    assert not row["model_candidate_selected"] and not row["layout_changed"]
    assert len(requests) == 1 and "no_choice" in row["critic"]["fallback_reason"]
    assert row["quality_improved"] is None


def test_duplicate_plan_geometry_does_not_count_as_new_candidate(document, tmp_path, monkeypatch):
    source, target, _, _ = document
    report, _, _ = execute(
        document, tmp_path, monkeypatch, engine=FakeEngine(source, target, duplicate=True)
    )
    row = report["targets"][0]
    assert not row["new_candidates_generated"] and not row["layout_changed"]


def test_planner_can_cause_sole_safe_new_winner_without_fake_critic_call(
    document, tmp_path, monkeypatch
):
    source, target, _, _ = document
    report, requests, _ = execute(
        document, tmp_path, monkeypatch, engine=FakeEngine(source, target, sole_new=True)
    )
    row = report["targets"][0]
    assert row["model_candidate_selected"] and row["layout_changed"]
    assert len(requests) == 1 and not row["critic"]["model_executed"]
    assert row["quality_improved"] is None


@pytest.mark.parametrize("choice", ["unknown", "__import__('os')"])
def test_unknown_model_choice_falls_back_without_compiling(document, tmp_path, monkeypatch, choice):
    report, _, engine = execute(
        document, tmp_path, monkeypatch, provider=lambda request: model_result(request, choice)
    )
    row = report["targets"][0]
    assert row["model_executed"] and not row["proposal_valid"]
    assert not row["layout_changed"] and "unknown" in row["fallback_reason"].lower()
    assert engine.calls == [None]


def test_resolution_route_does_not_pretend_a_model_reorganization(document, tmp_path, monkeypatch):
    source, target, _, _ = document
    report, requests, _ = execute(
        document, tmp_path, monkeypatch, engine=FakeEngine(source, target, status="resolution")
    )
    row = report["targets"][0]
    assert not requests and not row["model_executed"]
    assert not row["new_candidates_generated"] and not row["layout_changed"]
    assert "resolution" in row["fallback_reason"]


def test_reserved_source_rejected_before_any_render_or_model(document, tmp_path, monkeypatch):
    source, target, assets, fonts = document
    monkeypatch.setattr(harness, "_reserve_identities", lambda: (set(), {scene._digest(source)}))
    monkeypatch.setattr(scene, "_render", lambda *_a, **_kw: pytest.fail("Reserve rendered"))
    with pytest.raises(ValueError, match="reserve"):
        harness.inspect_visual_resize(
            source,
            assets,
            fonts,
            tmp_path / "out",
            targets=[target],
            command=[sys.executable, "-c", "pass"],
        )


def test_duplicate_targets_cannot_overwrite_evidence(document, tmp_path):
    source, target, assets, fonts = document
    with pytest.raises(ValueError, match="Duplicate"):
        harness.inspect_visual_resize(
            source,
            assets,
            fonts,
            tmp_path / "out",
            targets=[target, target],
            command=[sys.executable, "-c", "pass"],
        )


def test_resource_mutation_invalidates_entire_experiment(document, tmp_path, monkeypatch):
    def mutate(request):
        (document[2] / "photo.png").write_bytes(b"changed")
        return model_result(request, "keep")

    with pytest.raises(RuntimeError, match="resource bytes changed"):
        execute(document, tmp_path, monkeypatch, provider=mutate)


def test_changed_code_is_marked_incomplete(document, tmp_path, monkeypatch):
    values = iter([{"test": "a" * 64}, {"test": "b" * 64}])
    monkeypatch.setattr(harness, "_implementation_hashes", lambda: next(values))
    report, _, _ = execute(document, tmp_path, monkeypatch)
    assert not report["implementation_unchanged"]


def test_actual_operator_program_protocol_and_output_budget():
    command = [sys.executable, "-c", "import json,sys; print(json.dumps(json.load(sys.stdin)))"]
    assert harness.run_provider(command, {"probe": "finite source data"}, timeout=5) == {
        "probe": "finite source data"
    }
    command = [sys.executable, "-c", f"print('x'*{harness.MAX_PROVIDER_BYTES + 10000})"]
    with pytest.raises(ValueError, match="stdout budget"):
        harness.run_provider(command, {}, timeout=5)


def test_actual_operator_timeout_is_bounded():
    command = [sys.executable, "-c", "import time;time.sleep(5)"]
    with pytest.raises(ValueError, match="timeout"):
        harness.run_provider(command, {}, timeout=0.1)


def test_diagnostic_controls_bind_changed_inputs_without_mutating_primary(document):
    source, target, _, _ = document
    engine = FakeEngine(source, target)
    menu = engine.menu(source, target)
    original = harness.make_request(
        source, {"path": "original.png", "sha256": "a" * 64}, menu, mode="plan"
    )
    before = deepcopy(original)
    changed = harness.source_image_ablation(original, {"path": "blank.png", "sha256": "b" * 64})
    assert changed["request_digest"] != original["request_digest"]
    assert changed["diagnostic"]["not_production_source_evidence"]
    assert original == before
    swapped = harness.target_control(
        source, original["source_render"], engine.menu(source, {"width": 100, "height": 200})
    )
    assert swapped["target_canvas"] != target
    assert swapped["plan_menu"]["menu_digest"] != menu["menu_digest"]
    snapshot = engine.snapshot(source, target, visual_plan={"plan_id": "new"})
    critic = harness.make_request(
        source, original["source_render"], engine.critic_menu(snapshot), mode="critique"
    )
    reversed_request = harness.candidate_order_control(critic)
    assert [c["plan_id"] for c in reversed_request["plans"]] == [
        c["plan_id"] for c in reversed(critic["plans"])
    ]
    assert reversed_request["plan_menu"]["menu_digest"] != critic["plan_menu"]["menu_digest"]


def test_visual_improvement_requires_hash_bound_review(document, tmp_path, monkeypatch):
    report, _, _ = execute(document, tmp_path, monkeypatch)
    row = report["targets"][0]
    review = {
        "origin": "agent_visual_review",
        "reviewer": "test fixture",
        "verdict": "improved",
        "source_digest": row["source_digest"],
        "target_canvas": row["target_canvas"],
        "baseline_design_digest": row["baseline_design_digest"],
        "final_design_digest": row["final_design_digest"],
        "baseline_sha256": row["baseline_render"]["sha256"],
        "final_sha256": row["final_render"]["sha256"],
    }
    stale = {**review, "final_sha256": "f" * 64}
    with pytest.raises(ValueError, match="bind"):
        harness._attach_review(stale, row)
    assert row["quality_improved"] is None
    harness._attach_review(review, row)
    assert row["quality_improved"] is True


def analysis_result(request, *, abstain=False):
    from engine.visual_analysis import bind_analysis

    return {
        "analysis": bind_analysis(
            {
                "observations": []
                if abstain
                else [
                    {
                        "kind": "subject",
                        "element_ids": ["image"],
                        "evidence": "A tall blue subject dominates the source.",
                    }
                ],
                "abstain": abstain,
            },
            source_digest=request["source_digest"],
            source_render_sha256=request["source_render"]["sha256"],
            known_element_ids=[row["id"] for row in request["source_summary"]["elements"]],
        ),
        "receipt": {
            "executed": True,
            "backend": "test_stub",
            "input_image_count": 1,
            "request_digest": scene._digest(request),
            "source_digest": request["source_digest"],
            "source_render_sha256": request["source_render"]["sha256"],
        },
    }


def test_source_analysis_reaches_separate_organizer_and_critic(document, tmp_path, monkeypatch):
    def provider(request):
        return analysis_result(request) if request["mode"] == "analyze" else model_result(request)

    report, requests, _ = execute(
        document, tmp_path, monkeypatch, provider=provider, analyzer_command=["source-vlm"]
    )
    assert [request["mode"] for request in requests] == ["analyze", "plan", "critique"]
    assert "plan_menu" not in requests[0] and "target_canvas" not in requests[0]
    for request in requests[1:]:
        analysis = request["source_analysis"]
        assert analysis["source_render_sha256"] == request["source_render"]["sha256"]
        assert request["source_analysis_digest"] == scene._digest(analysis)
        assert "source_analysis" not in request["plan_menu"]
    assert report["analysis"]["analysis_valid"]
    assert report["targets"][0]["status"] == "contract_valid"
    assert report["targets"][0]["quality_improved"] is None
    assert [row["stage"] for row in report["model_receipts"]] == ["analysis", "planner", "critic"]


@pytest.mark.parametrize("failure", ["stale_pixels", "stale_receipt", "unknown_id", "abstain"])
def test_bad_source_analysis_retains_baseline_without_organizer(
    document, tmp_path, monkeypatch, failure
):
    def provider(request):
        result = analysis_result(request, abstain=failure == "abstain")
        if failure == "stale_pixels":
            result["analysis"]["source_render_sha256"] = "f" * 64
        elif failure == "stale_receipt":
            result["receipt"]["request_digest"] = "f" * 64
        elif failure == "unknown_id":
            result["analysis"]["observations"][0]["element_ids"] = ["invented"]
        return result

    report, requests, engine = execute(
        document, tmp_path, monkeypatch, provider=provider, analyzer_command=["source-vlm"]
    )
    row = report["targets"][0]
    assert [request["mode"] for request in requests] == ["analyze"]
    assert engine.calls == [None]
    assert not row["layout_changed"] and row["status"] == "contract_valid"
    assert row["analysis"]["model_executed"] and not row["planner"]["invoked"]
    assert row["fallback_reason"]


def test_source_analysis_executes_once_for_multiple_targets(document, tmp_path, monkeypatch):
    from engine import visual_pipeline

    source, target, assets, fonts = document
    engine = FakeEngine(source, target)
    calls = []

    def provider(command, request, timeout):
        calls.append((command, deepcopy(request)))
        return analysis_result(request) if request["mode"] == "analyze" else model_result(request)

    def snapshot(value, dimensions, **kwargs):
        current = FakeEngine(value, dimensions)
        return current.snapshot(value, dimensions, **kwargs)

    report = visual_pipeline.evaluate_visual_resize(
        source,
        assets,
        fonts,
        tmp_path / "multiple",
        targets=[target, {"width": 300, "height": 100}],
        command=["organizer"],
        analyzer_command=["source-vlm"],
        provider=provider,
        api={**engine.api(), "snapshot": snapshot},
        critic_enabled=False,
        implementation_hashes=lambda: {"test": "a" * 64},
    )
    assert [request["mode"] for _, request in calls] == ["analyze", "plan", "plan"]
    assert calls[0][0] == ["source-vlm"]
    assert all(command == ["organizer"] for command, _ in calls[1:])
    assert all(row["status"] == "contract_valid" for row in report["targets"])


def test_organizer_cannot_reuse_a_decision_from_different_analysis(document, tmp_path, monkeypatch):
    def provider(request):
        if request["mode"] == "analyze":
            return analysis_result(request)
        result = model_result(request)
        result["receipt"]["source_analysis_digest"] = "f" * 64
        return result

    report, requests, engine = execute(
        document, tmp_path, monkeypatch, provider=provider, analyzer_command=["source-vlm"]
    )
    row = report["targets"][0]
    assert [request["mode"] for request in requests] == ["analyze", "plan"]
    assert not row["proposal_valid"] and not row["layout_changed"]
    assert "supplied source analysis" in row["fallback_reason"]
    assert engine.calls == [None]


def test_failed_analyzer_rerun_clears_old_valid_sidecar(document, tmp_path, monkeypatch):
    def provider(request):
        return analysis_result(request) if request["mode"] == "analyze" else model_result(request)

    execute(document, tmp_path, monkeypatch, provider=provider, analyzer_command=["source-vlm"])
    sidecar = tmp_path / "evidence/source-analysis.json"
    assert sidecar.is_file()

    def unavailable(request):
        raise RuntimeError("Local analyzer weights unavailable")

    report, requests, _ = execute(
        document, tmp_path, monkeypatch, provider=unavailable, analyzer_command=["source-vlm"]
    )
    assert len(requests) == 1 and not sidecar.exists()
    assert not (tmp_path / "evidence/analysis-response.json").exists()
    assert "weights unavailable" in report["targets"][0]["fallback_reason"]
    assert report["targets"][0]["status"] == "contract_valid"
