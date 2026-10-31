"""Frozen independent scoring cannot confuse valid JSON with model quality."""

import builtins
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from experiments.vision_critic import independent_eval as evaluation
from experiments.vision_critic.contracts import canonical_json_sha256, sha256_file
from experiments.visual_planner.provider import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    files = {}
    images = {}
    for name, color in (("source", "red"), ("good", "red"), ("bad", "blue"), ("blank", "gray")):
        path = root / (name + ".png")
        Image.new("RGB", (16, 16), color).save(path)
        files[path.name] = sha256_file(path)
        images[name] = {"path": str(path), "sha256": files[path.name]}
    code_path = root / "code" / "provider.py"
    code_path.parent.mkdir()
    code_path.write_text("# frozen fixture implementation\n", encoding="utf-8")
    code_sha = sha256_file(code_path)
    files["code/provider.py"] = code_sha
    cases = [
        {
            "case_id": "validation-case",
            "family_id": "family-one",
            "split": "validation",
            "baseline_candidate_id": "candidate-good",
            "admitted_candidate_ids": ["candidate-good"],
        },
        {"case_id": "test-case", "family_id": "sealed-family", "split": "test"},
    ]
    requests, probes = {}, []
    conditions = ("original", "swapped", "blank_source", "replaced_source", "opaque_ids")
    for case in cases:
        for condition in conditions if case["split"] != "test" else ("original",):
            probe_id = case["case_id"] + "-" + condition
            known = (
                ["opaque-01", "opaque-02"] if condition == "opaque_ids" else ["hero", "headline"]
            )
            order = ["bad", "good"] if condition == "swapped" else ["good", "bad"]
            source = (
                "blank"
                if condition == "blank_source"
                else "bad"
                if (condition == "replaced_source")
                else "source"
            )
            menu = {
                "schema_version": 1,
                "source_digest": "a" * 64,
                "target_canvas": {"width": 16, "height": 16},
                "plans": [
                    {"plan_id": name, "label": "Layout " + str(i)} for i, name in enumerate(order)
                ],
            }
            menu["menu_digest"] = digest(menu)
            request = {
                "mode": "critique",
                "source_digest": "a" * 64,
                "plan_menu": menu,
                "target_canvas": menu["target_canvas"],
                "source_render": images[source],
                "source_summary": {"elements": [{"id": eid} for eid in known]},
                "plans": [{**plan, "render": images[plan["plan_id"]]} for plan in menu["plans"]],
            }
            relative = "requests/" + probe_id + ".json"
            write_json(root / relative, request)
            files[relative] = sha256_file(root / relative)
            requests[probe_id] = request
            probes.append(
                {
                    "probe_id": probe_id,
                    "case_id": case["case_id"],
                    "family_id": case["family_id"],
                    "split": case["split"],
                    "condition": condition,
                    "mode": "critique",
                    "request_path": relative,
                    "request_digest": digest(request),
                    "source_sha256": images[source]["sha256"],
                    "original_source_sha256": images["source"]["sha256"],
                    "menu_digest": menu["menu_digest"],
                    "known_element_ids": known,
                    "plan_map": {name: "candidate-" + name for name in order},
                    "element_map": dict(zip(known, ("hero", "headline"), strict=True)),
                }
            )
    bundle = {
        "schema_version": 1,
        "protocol": evaluation.PROTOCOL,
        "purpose": "synthetic",
        "snapshot": {"code_hashes": {"code/provider.py": code_sha}},
        "cases": cases,
        "probes": probes,
        "files": files,
    }
    bundle["bundle_sha256"] = canonical_json_sha256(bundle)
    bundle_path = root / "bundle.json"
    write_json(bundle_path, bundle)
    labels_path, test_labels_path = root / "labels.json", root / "test-labels.json"
    for path, case_id in ((labels_path, "validation-case"), (test_labels_path, "test-case")):
        write_json(
            path,
            {
                "schema_version": 1,
                "bundle_sha256": bundle["bundle_sha256"],
                "label_kind": "synthetic",
                "decisions": [
                    {
                        "case_id": case_id,
                        "preferred_candidate_id": "candidate-good",
                        "axes": {axis: "candidate-good" for axis in evaluation.AXES},
                    }
                ],
            },
        )
    return SimpleNamespace(
        root=root,
        bundle=bundle,
        path=bundle_path,
        requests=requests,
        labels=labels_path,
        test_labels=test_labels_path,
        code_sha=code_sha,
    )


def transcript(frozen, probe, *, choice=None, protocol="provider", fallback=False, executed=True):
    request = frozen.requests[probe["probe_id"]]
    plan_ids = [plan["plan_id"] for plan in request["plan_menu"]["plans"]]
    choice = choice or "AB"[plan_ids.index("good")]
    plan_id = plan_ids["AB".index(choice)]
    raw = json.dumps({"choice": choice, "issues": []})
    response = None
    if protocol == "provider":
        raw = json.dumps({"choice": choice, "observations": "Source colors", "reason": "Same art"})
        response = {
            "decision": {
                "schema_version": 1,
                "menu_digest": probe["menu_digest"],
                "plan_id": plan_id,
                "rationale": "Same art",
            },
            "receipt": {
                "executed": True,
                "request_digest": probe["request_digest"],
                "source_render_sha256": probe["source_sha256"],
                "source_digest": "a" * 64,
                "menu_digest": probe["menu_digest"],
                "model_id": "test/model",
                "model_revision": "b" * 40,
                "weights": {"model.gguf": "c" * 64},
                "implementation_unchanged": True,
                "implementation_hashes": {
                    "provider.py": frozen.code_sha,
                },
                "prompt_sha256": "d" * 64,
                "input_image_count": 3,
                "images": [request["source_render"], *[p["render"] for p in request["plans"]]],
                "raw_output": raw,
                "choice_letter": choice,
            },
        }
    identity = {"model": "fixed-test-model", "revision": "b" * 40}
    return {
        "probe_id": probe["probe_id"],
        "bundle_sha256": frozen.bundle["bundle_sha256"],
        "request_digest": probe["request_digest"],
        "method_name": "fixture-model",
        "method_identity": identity,
        "method_identity_sha256": canonical_json_sha256(identity),
        "protocol": protocol,
        "executed": executed,
        "fallback": fallback,
        "response": response,
        "raw_response": raw,
        "error": None,
    }


def score(frozen, rows, **kwargs):
    path = frozen.root / "transcripts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    labels = kwargs.pop("labels_path", frozen.labels)
    return evaluation.evaluate(frozen.path, [path], labels, **kwargs)


def validation_probes(frozen):
    return [p for p in frozen.bundle["probes"] if p["split"] == "validation"]


def test_equivalent_controls_use_canonical_identity_and_receipt_evidence(frozen):
    report = score(frozen, [transcript(frozen, probe) for probe in validation_probes(frozen)])
    method = report["methods"]["fixture-model"]
    assert method["counts"]["qualified_model_success"] == 5
    assert method["agreement"]["labeled_probe_count"] == 2
    assert method["agreement"]["strict_agreement"] == 1
    assert method["agreement"]["both_orders_correct_count"] == 1
    assert method["agreement"]["both_orders_correct_rate"] == 1
    assert all(axis["strict_agreement"] == 1 for axis in method["agreement"]["per_axis"].values())
    assert method["controls"]["swapped"]["invariance_rate"] == 1
    assert method["controls"]["opaque_ids"]["invariance_rate"] == 1
    assert method["controls"]["blank_source"]["change_rate"] == 0
    assert report["test_split_unblinded"] is False
    assert not report["inference_executed_by_evaluator"]


@pytest.mark.parametrize("letter", ["A", "B"])
def test_constant_letter_collapse_is_separate_from_json_validity(frozen, letter):
    rows = [
        transcript(frozen, p, choice=letter, protocol="critic") for p in validation_probes(frozen)
    ]
    method = score(frozen, rows)["methods"]["fixture-model"]
    assert method["counts"]["valid"] == 5
    assert method["counts"]["qualified_model_success"] == 0
    assert method["dominant_letter_fraction"] == 1
    assert method["constant_raw_response"]
    assert method["agreement"]["strict_agreement"] == 0.5
    assert method["agreement"]["conditions"] == ["original", "swapped"]
    assert method["agreement"]["both_orders_pair_count"] == 1
    assert method["agreement"]["both_orders_correct_count"] == 0
    assert method["agreement"]["both_orders_correct_rate"] == 0
    assert method["controls"]["swapped"]["invariance_rate"] == 0


@pytest.mark.parametrize("change", [{"fallback": True}, {"executed": False}])
def test_fallback_or_unexecuted_output_never_earns_a_model_match(frozen, change):
    row = transcript(frozen, validation_probes(frozen)[0])
    row.update(change)
    method = score(frozen, [row])["methods"]["fixture-model"]
    assert method["counts"]["valid"] == 1
    assert method["counts"]["qualified_model_success"] == 0
    assert method["counts"]["missing_transcripts"] == 4
    assert method["agreement"]["matches"] == 0
    assert method["agreement"]["labeled_probe_count"] == 2


def test_unknown_issue_ids_and_receipt_source_mismatch_are_reported_invalid(frozen):
    first, second = validation_probes(frozen)[:2]
    critic = transcript(frozen, first, protocol="critic")
    critic["raw_response"] = json.dumps(
        {
            "choice": "A",
            "issues": [
                {
                    "candidate": "A",
                    "code": "weak_hierarchy",
                    "element_ids": ["invented-id"],
                }
            ],
        }
    )
    provider = transcript(frozen, second)
    provider["response"]["receipt"]["source_render_sha256"] = "0" * 64
    method = score(frozen, [critic, provider])["methods"]["fixture-model"]
    assert method["counts"]["valid"] == 0
    assert method["counts"]["input_integrity_failures"] == 1
    assert method["agreement"]["matches"] == 0


def test_raw_provider_stdout_may_be_a_complete_response_envelope(frozen):
    row = transcript(frozen, validation_probes(frozen)[0])
    row["raw_response"] = json.dumps(row["response"])
    method = score(frozen, [row])["methods"]["fixture-model"]
    assert method["counts"]["valid"] == method["counts"]["qualified_model_success"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "method_identity", "request", "model"])
def test_duplicate_or_mixed_or_stale_transcripts_are_rejected(frozen, mutation):
    rows = [transcript(frozen, p) for p in validation_probes(frozen)[:2]]
    if mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "method_identity":
        rows[1]["method_identity"]["revision"] = "c" * 40
        rows[1]["method_identity_sha256"] = canonical_json_sha256(rows[1]["method_identity"])
    elif mutation == "request":
        rows[1]["request_digest"] = "0" * 64
    else:
        rows[1]["response"]["receipt"]["weights"]["model.gguf"] = "e" * 64
    with pytest.raises(ValueError):
        score(frozen, rows)


def test_changed_frozen_input_or_code_is_rejected(frozen):
    (frozen.root / "code" / "provider.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file changed"):
        score(frozen, [])


def test_test_label_path_is_not_opened_without_explicit_matching_freeze(frozen, monkeypatch):
    original = evaluation._json

    def reject_sealed(path):
        assert Path(path) != frozen.test_labels, "Sealed test label file was opened"
        return original(path)

    monkeypatch.setattr(evaluation, "_json", reject_sealed)
    with pytest.raises(ValueError, match="before opening"):
        score(frozen, [], test_labels_path=frozen.test_labels)
    freeze = frozen.root / "freeze.json"
    write_json(freeze, {"bundle_sha256": "0" * 64, "recipe_sha256": "a" * 64})
    with pytest.raises(ValueError, match="matching bundle"):
        score(
            frozen,
            [],
            test_labels_path=frozen.test_labels,
            recipe_freeze_path=freeze,
            include_test=True,
        )


def test_matching_freeze_allows_explicit_test_scoring(frozen):
    freeze = frozen.root / "freeze.json"
    write_json(freeze, {"bundle_sha256": frozen.bundle["bundle_sha256"], "recipe_sha256": "a" * 64})
    report = score(
        frozen,
        [transcript(frozen, p) for p in frozen.bundle["probes"]],
        test_labels_path=frozen.test_labels,
        recipe_freeze_path=freeze,
        include_test=True,
    )
    assert report["test_split_unblinded"]
    assert report["methods"]["fixture-model"]["agreement"]["labeled_probe_count"] == 3


def test_scoring_imports_no_model_or_gpu_libraries(frozen, monkeypatch):
    original = builtins.__import__

    def reject_model(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "accelerate"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_model)
    report = score(frozen, [transcript(frozen, validation_probes(frozen)[0], protocol="critic")])
    assert report["methods"]["fixture-model"]["counts"]["valid"] == 1


def rewrite_bundle(frozen):
    frozen.bundle["bundle_sha256"] = canonical_json_sha256(
        {key: value for key, value in frozen.bundle.items() if key != "bundle_sha256"}
    )
    write_json(frozen.path, frozen.bundle)


def test_optional_source_json_lineage_rejects_invented_original_ids(frozen):
    source = {"canvas": {"width": 16, "height": 16}, "elements": [{"id": "real-id"}]}
    write_json(frozen.root / "source.json", source)
    frozen.bundle["files"]["source.json"] = sha256_file(frozen.root / "source.json")
    case = frozen.bundle["cases"][0]
    case.update(
        source_json_path="source.json",
        source_json_sha256=canonical_json_sha256(source),
        source_element_ids=["real-id"],
        source_render_sha256=validation_probes(frozen)[0]["source_sha256"],
    )
    rewrite_bundle(frozen)
    with pytest.raises(ValueError, match="original JSON lineage"):
        score(frozen, [])


def test_physical_candidate_replacement_is_not_accepted_as_an_order_control(frozen):
    probe = validation_probes(frozen)[1]
    request = frozen.requests[probe["probe_id"]]
    request["plans"][0]["render"] = request["plans"][1]["render"]
    write_json(frozen.root / probe["request_path"], request)
    frozen.bundle["files"][probe["request_path"]] = sha256_file(frozen.root / probe["request_path"])
    probe["request_digest"] = digest(request)
    rewrite_bundle(frozen)
    with pytest.raises(ValueError, match="physical candidate"):
        score(frozen, [])


def test_opaque_critic_issue_ids_are_mapped_back_to_original_ids(frozen):
    probe = next(p for p in validation_probes(frozen) if p["condition"] == "opaque_ids")
    row = transcript(frozen, probe, protocol="critic")
    row["raw_response"] = json.dumps(
        {
            "choice": "A",
            "issues": [
                {
                    "candidate": "B",
                    "code": "weak_product_prominence",
                    "element_ids": ["opaque-01"],
                }
            ],
        }
    )
    prediction = next(
        r
        for r in score(frozen, [row])["methods"]["fixture-model"]["predictions"]
        if r["probe_id"] == probe["probe_id"]
    )
    assert prediction["valid"]
    assert prediction["issues"] == [
        {
            "candidate_id": "candidate-bad",
            "code": "weak_product_prominence",
            "element_ids": ["hero"],
        }
    ]


@pytest.mark.parametrize("choice", ["tie", "neither"])
def test_critic_abstention_is_recorded_without_awarding_baseline_fallback(frozen, choice):
    row = transcript(frozen, validation_probes(frozen)[0], protocol="critic")
    row["raw_response"] = json.dumps({"choice": choice, "issues": []})
    method = score(frozen, [row])["methods"]["fixture-model"]
    assert method["counts"]["valid"] == method["counts"]["abstain"] == 1
    assert method["admissibility"]["baseline_fallbacks"] == 5
    assert method["agreement"]["matches"] == 0


@pytest.fixture
def analyzer(frozen):
    frozen.bundle["probes"] = [p for p in frozen.bundle["probes"] if p["condition"] != "swapped"]
    for probe in frozen.bundle["probes"]:
        request = frozen.requests[probe["probe_id"]]
        request = {
            key: value
            for key, value in request.items()
            if key not in {"plan_menu", "plans", "target_canvas"}
        }
        request.update(schema_version=1, mode="analyze")
        frozen.requests[probe["probe_id"]] = request
        write_json(frozen.root / probe["request_path"], request)
        frozen.bundle["files"][probe["request_path"]] = sha256_file(
            frozen.root / probe["request_path"]
        )
        probe.update(mode="analyze", menu_digest=None, plan_map={}, request_digest=digest(request))
    rewrite_bundle(frozen)
    return frozen


def analysis_row(frozen, probe, *, unknown_id=False, abstain=False):
    observations = (
        []
        if abstain
        else [
            {
                "kind": "hierarchy",
                "element_ids": ["invented" if unknown_id else probe["known_element_ids"][0]],
                "evidence": "Dominant visible subject",
            }
        ]
    )
    identity = {"model": "frozen-analyzer", "revision": "b" * 40}
    analysis = {
        "schema_version": 1,
        "source_digest": "a" * 64,
        "source_render_sha256": probe["source_sha256"],
        "observations": observations,
        "abstain": abstain,
    }
    return {
        "probe_id": probe["probe_id"],
        "bundle_sha256": frozen.bundle["bundle_sha256"],
        "request_digest": probe["request_digest"],
        "method_name": "analyzer",
        "method_identity": identity,
        "method_identity_sha256": canonical_json_sha256(identity),
        "protocol": "provider",
        "executed": True,
        "fallback": False,
        "response": {"analysis": analysis},
        "raw_response": json.dumps({"analysis": analysis}),
        "error": None,
    }


def test_analyzer_known_ids_and_opaque_fingerprints_are_separate_from_choice_agreement(analyzer):
    rows = [analysis_row(analyzer, p) for p in validation_probes(analyzer)]
    method = score(analyzer, rows, labels_path=None)["methods"]["analyzer"]
    assert method["counts"]["valid"] == method["counts"]["known_ids_valid"] == 4
    assert method["agreement"]["labeled_probe_count"] == 0
    assert method["controls"]["opaque_ids"]["invariance_rate"] == 1
    assert method["letter_distribution"] == {}


@pytest.mark.parametrize("unknown_id,abstain", [(True, False), (False, True)])
def test_analyzer_rejects_unknown_ids_and_records_explicit_abstention(
    analyzer, unknown_id, abstain
):
    row = analysis_row(
        analyzer, validation_probes(analyzer)[0], unknown_id=unknown_id, abstain=abstain
    )
    method = score(analyzer, [row], labels_path=None)["methods"]["analyzer"]
    assert method["counts"]["valid"] == int(not unknown_id)
    assert method["counts"]["abstain"] == int(abstain)


def test_cli_writes_a_standalone_report_without_inference(frozen, capsys):
    row = transcript(frozen, validation_probes(frozen)[0], protocol="critic")
    transcripts = frozen.root / "supplied.jsonl"
    transcripts.write_text(json.dumps(row) + "\n", encoding="utf-8")
    destination = frozen.root / "report.json"
    assert (
        evaluation.main(
            [
                "--bundle",
                str(frozen.path),
                "--transcripts",
                str(transcripts),
                "--labels",
                str(frozen.labels),
                "--out",
                str(destination),
            ]
        )
        == 0
    )
    report = json.loads(destination.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["inference_executed_by_evaluator"] is False


def test_real_probe_builder_bundle_can_be_relocated_and_scored(tmp_path, monkeypatch):
    from test_independent_probes import collection as collection_fixture

    from experiments.vision_critic.independent_probes import prepare_collection

    collection, _, _ = collection_fixture.__wrapped__(tmp_path, monkeypatch)
    root = tmp_path / "generated"
    bundle = prepare_collection(collection, root, modes=("critique", "analyze"))
    relocated = tmp_path / "relocated"
    root.rename(relocated)
    identity = {"kind": "synthetic supplied transcript", "revision": "frozen"}
    rows = []
    for probe in bundle["probes"]:
        request = json.loads((relocated / probe["request_path"]).read_text(encoding="utf-8"))
        analyze = probe["mode"] == "analyze"
        response = None
        raw = '{"choice":"A","issues":[]}'
        if analyze:
            response = {
                "analysis": {
                    "schema_version": 1,
                    "source_digest": request["source_digest"],
                    "source_render_sha256": probe["source_sha256"],
                    "observations": [],
                    "abstain": True,
                }
            }
            raw = json.dumps(response)
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "bundle_sha256": bundle["bundle_sha256"],
                "request_digest": probe["request_digest"],
                "method_name": "fixture",
                "method_identity": identity,
                "method_identity_sha256": canonical_json_sha256(identity),
                "protocol": "provider" if analyze else "critic",
                "executed": True,
                "fallback": False,
                "response": response,
                "raw_response": raw,
                "error": None,
            }
        )
    transcripts = tmp_path / "generated-transcripts.jsonl"
    transcripts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = evaluation.evaluate(relocated / "bundle.json", [transcripts])
    assert report["methods"]["fixture"]["counts"]["valid"] == len(bundle["probes"])
    assert report["methods"]["fixture"]["counts"]["qualified_model_success"] == 0
    assert report["methods"]["fixture"]["agreement"]["labeled_probe_count"] == 0


def attach_source_analysis(frozen):
    for probe in frozen.bundle["probes"]:
        request = frozen.requests[probe["probe_id"]]
        request["source_analysis"] = {
            "schema_version": 1,
            "source_digest": request["source_digest"],
            "source_render_sha256": probe["source_sha256"],
            "observations": [
                {
                    "kind": "subject",
                    "element_ids": [probe["known_element_ids"][0]],
                    "evidence": "Visible subject",
                }
            ],
            "abstain": False,
        }
        request["source_analysis_digest"] = digest(request["source_analysis"])
        write_json(frozen.root / probe["request_path"], request)
        frozen.bundle["files"][probe["request_path"]] = sha256_file(
            frozen.root / probe["request_path"]
        )
        probe["request_digest"] = digest(request)
    rewrite_bundle(frozen)


@pytest.mark.parametrize("receipt_binding", ["matching", "missing", "stale"])
def test_analysis_handoff_must_be_bound_in_qualified_provider_receipt(frozen, receipt_binding):
    attach_source_analysis(frozen)
    probe = validation_probes(frozen)[0]
    row = transcript(frozen, probe)
    if receipt_binding != "missing":
        row["response"]["receipt"]["source_analysis_digest"] = (
            frozen.requests[probe["probe_id"]]["source_analysis_digest"]
            if receipt_binding == "matching"
            else "0" * 64
        )
    method = score(frozen, [row], labels_path=None)["methods"]["fixture-model"]
    assert method["counts"]["qualified_model_success"] == int(receipt_binding == "matching")
    assert method["counts"]["valid"] == int(receipt_binding != "stale")
    assert method["counts"]["input_integrity_failures"] == int(receipt_binding == "stale")


def test_source_analysis_for_different_pixels_is_rejected_before_scoring(frozen):
    attach_source_analysis(frozen)
    probe = validation_probes(frozen)[0]
    request = frozen.requests[probe["probe_id"]]
    request["source_analysis"]["source_render_sha256"] = "0" * 64
    request["source_analysis_digest"] = digest(request["source_analysis"])
    write_json(frozen.root / probe["request_path"], request)
    frozen.bundle["files"][probe["request_path"]] = sha256_file(frozen.root / probe["request_path"])
    probe["request_digest"] = digest(request)
    rewrite_bundle(frozen)
    with pytest.raises(ValueError, match="different source or render"):
        score(frozen, [], labels_path=None)


def test_abstain_labels_do_not_reduce_forced_choice_or_axis_accuracy(frozen):
    case = frozen.bundle["cases"][1]
    case["split"] = "validation"
    for probe in frozen.bundle["probes"]:
        if probe["case_id"] == case["case_id"]:
            probe["split"] = "validation"
    rewrite_bundle(frozen)
    write_json(
        frozen.labels,
        {
            "schema_version": 1,
            "bundle_sha256": frozen.bundle["bundle_sha256"],
            "label_kind": "human",
            "decisions": [
                {
                    "case_id": "validation-case",
                    "preferred_candidate_id": "candidate-good",
                    "axes": {axis: "candidate-good" for axis in evaluation.AXES},
                },
                {
                    "case_id": "test-case",
                    "preferred_candidate_id": "abstain",
                    "axes": {axis: "abstain" for axis in evaluation.AXES},
                },
            ],
        },
    )
    rows = [transcript(frozen, probe) for probe in validation_probes(frozen)]
    agreement = score(frozen, rows)["methods"]["fixture-model"]["agreement"]
    assert agreement["labeled_probe_count"] == 3
    assert agreement["scored_probe_count"] == 2
    assert agreement["abstain_label_count"] == 1
    assert agreement["strict_agreement"] == 1
    assert agreement["both_orders_correct_rate"] == 1
    assert all(
        axis["abstain_label_count"] == 1 and axis["strict_agreement"] == 1
        for axis in agreement["per_axis"].values()
    )


def test_verified_unadmitted_recommendation_cannot_earn_qualified_model_success(frozen):
    labels = json.loads(frozen.labels.read_text(encoding="utf-8"))
    labels["decisions"][0]["preferred_candidate_id"] = "candidate-bad"
    labels["decisions"][0]["axes"] = {axis: "candidate-bad" for axis in evaluation.AXES}
    write_json(frozen.labels, labels)
    probe = validation_probes(frozen)[0]
    method = score(frozen, [transcript(frozen, probe, choice="B")])["methods"]["fixture-model"]
    prediction = next(row for row in method["predictions"] if row["probe_id"] == probe["probe_id"])
    assert prediction["valid"] and prediction["receipt_verified"]
    assert prediction["recommendation_eligible"]
    assert prediction["selection"] == {
        "candidate_id": "candidate-good",
        "fallback": True,
        "applicable": True,
    }
    assert not prediction["qualified_model_success"]
    assert method["counts"]["qualified_model_success"] == 0
    assert method["agreement"]["matches"] == 1
    assert method["agreement"]["qualified_recommendations"] == 0


def test_verified_analyzer_abstention_is_valid_but_not_model_success(analyzer):
    probe = validation_probes(analyzer)[0]
    row = analysis_row(analyzer, probe, abstain=True)
    row["response"]["receipt"] = {
        "executed": True,
        "request_digest": probe["request_digest"],
        "source_render_sha256": probe["source_sha256"],
        "source_digest": "a" * 64,
        "menu_digest": None,
        "model_id": "test/model",
        "model_revision": "b" * 40,
        "weights": {"model.gguf": "c" * 64},
        "implementation_unchanged": True,
        "implementation_hashes": {"provider.py": analyzer.code_sha},
        "prompt_sha256": "d" * 64,
        "input_image_count": 1,
        "images": [analyzer.requests[probe["probe_id"]]["source_render"]],
        "raw_output": '{"observations":[],"abstain":true}',
    }
    method = score(analyzer, [row], labels_path=None)["methods"]["analyzer"]
    prediction = next(
        item for item in method["predictions"] if item["probe_id"] == probe["probe_id"]
    )
    assert prediction["valid"] and prediction["receipt_verified"] and prediction["abstain"]
    assert not prediction["qualified_model_success"]
    assert method["counts"]["abstain"] == 1
    assert method["counts"]["qualified_model_success"] == 0
