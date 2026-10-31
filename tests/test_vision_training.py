"""Real, tiny CPU training and offline SFT preflight; never a quality benchmark."""

import builtins
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from experiments.vision_critic.evaluate import (
    admissible_selection,
    evaluate_choices,
    evaluate_ranker,
    read_verdicts,
)
from experiments.vision_critic.features import (
    FEATURE_NAMES,
    candidate_features,
    image_signature,
    pair_features,
)
from experiments.vision_critic.train_ranker import fit, load_model, preference, save_model
from experiments.vision_critic.train_sft import (
    make_verdict_record,
    parse_model_verdict,
    prepare_plan,
    supervise_command,
    validate_resume,
)


@pytest.fixture
def records(tmp_path):
    rows = []
    for index in range(8):
        directory = tmp_path / f"source-{index}"
        directory.mkdir()
        image = Image.new("RGB", (64, 80), "white")
        ImageDraw.Draw(image).rectangle((12, 12, 52, 60), fill=(40 + index * 20, 60, 100))
        source, good, bad = [directory / name for name in ("source.png", "good.png", "bad.png")]
        image.save(source)
        image.save(good)
        Image.new("RGB", (64, 80), "black").save(bad)
        a, b = (good, bad) if index % 2 == 0 else (bad, good)
        candidates = [
            {
                "candidate_id": f"{index}-{letter}",
                "render": str(path),
                "admitted": True,
                "diagnostics": {},
            }
            for letter, path in (("a", a), ("b", b))
        ]
        rows.append(
            {
                "pair_id": f"pair-{index}",
                "case_id": f"case-{index}",
                "source_id": f"source-{index}",
                "family_id": f"family-{index}",
                "split": "train" if index < 4 else "validation",
                "decision": "A" if index % 2 == 0 else "B",
                "annotator": "synthetic-fixture",
                "corpus_sha256": "a" * 64,
                "purpose": "synthetic_infrastructure_only",
                "images": [str(source), str(a), str(b)],
                "source": {"render": str(source)},
                "candidate_a": candidates[0],
                "candidate_b": candidates[1],
                "baseline_candidate_id": candidates[0]["candidate_id"],
                "admitted_candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            }
        )
    return rows


def recipe():
    return {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_revision": "a" * 40,
        "max_steps": 10,
        "max_minutes": 5,
        "save_every": 2,
        "image_side": 448,
        "max_tokens": 4096,
        "learning_rate": 1e-4,
        "seed": 2026,
    }


def sft_rows(records):
    return [
        {
            **row,
            "prompt": "Compare source, A and B; return choice and issues as JSON.",
            "response": json.dumps({"choice": row["decision"], "issues": row.get("issues", [])}),
        }
        for row in records
    ]


def test_real_cpu_training_learns_then_reloads_and_scores_unseen_source_families(records, tmp_path):
    model = fit(records, steps=40, allow_synthetic=True)
    assert model["loss_history"][-1] < model["loss_history"][0] * 0.5
    assert any(abs(weight) > 1e-8 for weight in model["weights"])
    assert model["task_model_trained"] is False
    save_model(tmp_path / "model.json", model)
    restored = load_model(tmp_path / "model.json")
    for row in records:
        assert preference(restored, row) == preference(model, row)
    report = evaluate_ranker(restored, records)
    assert report["metrics"]["validation"]["binary_preference_accuracy"] == 1.0
    assert report["metrics"]["validation"]["family_macro_binary_accuracy"] == 1.0
    assert report["swap_consistency"] == 1.0
    assert "Synthetic" in report["quality_claim"]


def test_resumed_optimization_equals_uninterrupted_training(records, tmp_path):
    uninterrupted = fit(records, steps=40, allow_synthetic=True)
    halfway = fit(records, steps=20, allow_synthetic=True)
    save_model(tmp_path / "halfway.json", halfway)
    resumed = fit(
        records, steps=40, resume=load_model(tmp_path / "halfway.json"), allow_synthetic=True
    )
    np.testing.assert_array_equal(resumed["weights"], uninterrupted["weights"])
    assert resumed["loss_history"] == uninterrupted["loss_history"]


@pytest.mark.parametrize("change", ["label", "collection", "learning_rate"])
def test_resume_rejects_changed_dataset_or_recipe(records, change):
    previous = fit(records, steps=10, allow_synthetic=True)
    changed = deepcopy(records)
    learning_rate = 0.1
    if change == "label":
        changed[0]["decision"] = "B"
    elif change == "collection":
        changed[0]["collection_sha256"] = "b" * 64
    else:
        learning_rate = 0.2
    with pytest.raises(ValueError, match="Resume"):
        fit(changed, steps=20, resume=previous, learning_rate=learning_rate, allow_synthetic=True)


def test_checkpoint_corruption_cannot_be_reloaded(records, tmp_path):
    path = tmp_path / "model.json"
    save_model(path, fit(records, steps=5, allow_synthetic=True))
    data = json.loads(path.read_text())
    data["state"]["weights"][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_model(path)
    with pytest.raises(FileExistsError):
        save_model(path, data)


def test_ties_are_explicit_soft_targets_and_neither_abstain_are_not_forced_winners(records):
    changed = deepcopy(records)
    changed[0]["decision"] = "tie"
    changed[0]["images"][2] = changed[0]["images"][1]
    changed[1]["decision"] = "neither"
    changed[2]["decision"] = "abstain"
    result = fit(changed, steps=10, allow_synthetic=True)
    assert result["train_pair_count"] == 2
    assert result["excluded_train_decisions"] == 2
    assert np.isfinite(result["weights"]).all()


def test_feature_pair_reverses_exactly_under_candidate_swap(records):
    row = records[0]
    a, b = pair_features(row)
    swapped = {
        **row,
        "candidate_a": row["candidate_b"],
        "candidate_b": row["candidate_a"],
        "images": [row["images"][0], row["images"][2], row["images"][1]],
    }
    reversed_a, reversed_b = pair_features(swapped)
    np.testing.assert_array_equal(a, reversed_b)
    np.testing.assert_array_equal(b, reversed_a)
    assert len(a) == len(FEATURE_NAMES)


def test_family_and_source_leakage_cannot_be_trained(records):
    changed = deepcopy(records)
    changed[4]["family_id"] = changed[0]["family_id"]
    with pytest.raises(ValueError, match="families overlap"):
        fit(changed, steps=5, allow_synthetic=True)
    changed = deepcopy(records)
    changed[4]["source_id"] = changed[0]["source_id"]
    with pytest.raises(ValueError, match="sources overlap"):
        fit(changed, steps=5, allow_synthetic=True)
    with pytest.raises(ValueError, match="Synthetic"):
        fit(records, steps=5)


def test_unsafe_or_invalid_verdict_falls_back_to_exact_baseline_even_outside_pair(records):
    row = {
        **records[0],
        "baseline_candidate_id": "baseline-outside-pair",
        "admitted_candidate_ids": ["baseline-outside-pair"],
    }
    for choice in ("A", "B", "tie", "neither", None, "injected-id"):
        selection = admissible_selection(row, choice)
        assert selection["candidate_id"] == "baseline-outside-pair"
        assert selection["fallback"] is True
    row["admitted_candidate_ids"] = []
    assert admissible_selection(row, "A")["candidate_id"] is None


def test_preference_metrics_do_not_hide_unadmitted_human_winners(records):
    row = {**records[0], "split": "validation", "admitted_candidate_ids": [], "decision": "A"}
    metrics = evaluate_choices([row], {row["pair_id"]: "A"})["validation"]
    assert metrics["binary_preference_accuracy"] == 1.0
    assert metrics["fallback_count"] == 1
    assert metrics["no_selection_count"] == 1


def test_family_macro_metric_does_not_reward_many_copies_from_one_family(records):
    many = [{**records[0], "pair_id": f"copy-{i}", "split": "validation"} for i in range(9)]
    other = {**records[1], "pair_id": "other", "split": "validation"}
    choices = {row["pair_id"]: row["decision"] for row in many}
    choices["other"] = "A"
    metrics = evaluate_choices([*many, other], choices)["validation"]
    assert metrics["binary_preference_accuracy"] == 0.9
    assert metrics["family_macro_binary_accuracy"] == 0.5


def test_verdict_reader_marks_invalid_choices_and_rejects_duplicate_pairs(tmp_path, records):
    path = tmp_path / "verdicts.jsonl"
    for text in (
        '{"choice":"bad","issues":[]}',
        '{"choice":{"nested":"A"},"issues":[]}',
        '{"choice":["A"],"issues":[]}',
    ):
        verdict = make_verdict_record(records[0], text)
        path.write_text(json.dumps(verdict) + "\n")
        assert read_verdicts(path, records) == {records[0]["pair_id"]: None}
    valid = make_verdict_record(records[0], '{"choice":"A","issues":[]}')
    path.write_text((json.dumps(valid) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_verdicts(path, records)


def test_sft_dry_run_inspects_three_images_without_importing_gpu_libraries(records, monkeypatch):
    original = builtins.__import__

    def reject_gpu(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "accelerate"}:
            raise AssertionError("Dry-run attempted to import a GPU/model library")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_gpu)
    report = prepare_plan(records, sft_rows(records), recipe(), allow_synthetic=True)
    assert report["dry_run_passed"] is True
    assert report["task_model_trained"] is False
    assert report["gpu_compatibility_validated"] is False
    assert len(report["images_inspected"]) == len(records) * 3
    assert report["train_count"] == report["validation_count"] == 4
    assert "pending" in report["mask_format_plan"]["tokenizer_validation"]


@pytest.mark.parametrize(
    "bad_recipe",
    [
        {"base_revision": "main"},
        {"max_steps": 0},
        {"max_minutes": 241},
        {"max_tokens": 50},
        {"image_side": 4096},
        {"save_every": 0},
        {"model": "./local-base"},
    ],
)
def test_sft_dry_run_rejects_unbounded_or_unpinned_recipes(records, bad_recipe):
    with pytest.raises(ValueError):
        prepare_plan(records, sft_rows(records), {**recipe(), **bad_recipe}, allow_synthetic=True)


def test_sft_resume_identity_binds_base_data_and_recipe(records):
    plan = prepare_plan(records, sft_rows(records), recipe(), allow_synthetic=True)
    progress = {
        "base_model": plan["model"],
        "base_revision": plan["base_revision"],
        "corpus_sha256": plan["corpus_sha256"],
        "recipe_sha256": plan["recipe_sha256"],
        "completed_steps": 2,
        "packages": plan["packages"],
        "trainer_sha256": plan["trainer_sha256"],
    }
    validate_resume(progress, plan)
    for key in ("base_model", "base_revision", "corpus_sha256", "recipe_sha256"):
        with pytest.raises(ValueError, match="identity"):
            validate_resume({**progress, key: "changed"}, plan)


def test_sft_rejects_broken_image_or_response_and_missing_validation(records):
    rows = sft_rows(records)
    rows[0]["response"] = '{"choice":"maybe","issues":[]}'
    with pytest.raises(ValueError, match="response"):
        prepare_plan(records, rows, recipe(), allow_synthetic=True)
    with pytest.raises(ValueError, match="validation"):
        prepare_plan(records, sft_rows(records)[:4], recipe(), allow_synthetic=True)
    Path(records[0]["images"][0]).write_bytes(b"corrupt PNG")
    with pytest.raises(OSError):
        prepare_plan(records, sft_rows(records), recipe(), allow_synthetic=True)


@pytest.fixture
def shared_corpus(tmp_path):
    from test_vision_contracts import build_collection, make_label

    from experiments.vision_critic.contracts import sha256_file

    root = tmp_path / "collection"
    root.mkdir()
    collection = build_collection(root, purpose="synthetic", cases=5)
    for index, case in enumerate(collection["cases"]):
        case["split"] = "train" if index < 2 else "validation" if index < 4 else "test"
        color = (60 + index * 20, 80, 140)
        Image.new("RGB", (100, 100), color).save(root / case["source"]["render"])
        case["source"]["render_sha256"] = sha256_file(root / case["source"]["render"])
        for candidate_index, candidate in enumerate(case["candidates"]):
            candidate_color = color if candidate_index == index % 2 else "black"
            Image.new("RGB", (64, 64), candidate_color).save(root / candidate["render"])
            candidate["render_sha256"] = sha256_file(root / candidate["render"])
    (root / "collection.json").write_text(json.dumps(collection), encoding="utf-8")
    labels = root / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps(
                make_label(collection, case_index=index, decision="A" if index % 2 == 0 else "B")
            )
            + "\n"
            for index in range(4)
        ),
        encoding="utf-8",
    )
    return root, labels


def test_training_evaluation_and_sft_dry_run_clis_use_the_real_verified_dataset(
    shared_corpus, tmp_path
):
    collection, labels = shared_corpus
    run = tmp_path / "ranker"
    common = ["--collection", str(collection), "--labels", str(labels), "--allow-synthetic"]
    command = [
        sys.executable,
        "-m",
        "experiments.vision_critic.train_ranker",
        *common,
        "--out",
        str(run),
        "--steps",
        "20",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    model = load_model(run / "model.json")
    assert model["task_model_trained"] is False
    evaluation = json.loads((run / "evaluation.json").read_text())
    assert evaluation["metrics"]["validation"]["binary_preference_accuracy"] == 1.0
    sft_run = tmp_path / "sft-plan"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.vision_critic.train_sft",
            *common,
            "--out",
            str(sft_run),
            "--base-revision",
            "a" * 40,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((sft_run / "dry-run-report.json").read_text())
    assert report["train_count"] == report["validation_count"] == 2
    assert report["gpu_compatibility_validated"] is False
    assert report["task_model_trained"] is False
    assert not (sft_run / "published").exists()


def test_training_cli_rejects_changed_render_before_creating_model(shared_corpus, tmp_path):
    collection, labels = shared_corpus
    (collection / "artifacts/case-0/source.png").write_bytes(b"changed after review")
    output = tmp_path / "must-not-train"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.vision_critic.train_ranker",
            "--collection",
            str(collection),
            "--labels",
            str(labels),
            "--out",
            str(output),
            "--allow-synthetic",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        "[]",
        '{"choice":["A"],"issues":[]}',
        '{"choice":{"x":"A"},"issues":[]}',
        '{"choice":"maybe","issues":[]}',
        '{"choice":"A","issues":"bad_overlap"}',
        '{"choice":"A","issues":[{}]}',
        '{"choice":"A","issues":["invented_issue"]}',
        '{"choice":"A"}',
        '{"choice":"A","choice":"B","issues":[]}',
        '{"choice":"A","issues":[],"coordinates":[]}',
    ],
)
def test_untrusted_model_json_cannot_select_a_candidate_when_schema_is_invalid(text, records):
    verdict = parse_model_verdict(text)
    assert verdict == {"choice": None, "issues": [], "valid": False}
    row = {
        **records[0],
        "baseline_candidate_id": "baseline",
        "admitted_candidate_ids": ["baseline"],
    }
    assert admissible_selection(row, verdict["choice"])["candidate_id"] == "baseline"


def test_strict_model_json_accepts_a_grounded_choice():
    issue = {"candidate": "A", "code": "bad_overlap", "element_ids": ["title"]}
    assert parse_model_verdict(
        json.dumps({"choice": "B", "issues": [issue]}), {"A": ["title"], "B": []}
    ) == {
        "choice": "B",
        "issues": [issue],
        "valid": True,
    }


def test_sft_process_supervisor_stops_a_stuck_native_worker_without_touching_checkpoints(tmp_path):
    checkpoint = tmp_path / "published-checkpoint"
    checkpoint.write_bytes(b"previous complete checkpoint")
    marker = tmp_path / "must-not-finish"
    script = (
        "import time; from pathlib import Path; time.sleep(10); "
        f"Path({str(marker)!r}).write_text('late')"
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="wall-time"):
        supervise_command([sys.executable, "-c", script], 0.3)
    assert time.monotonic() - started < 6
    assert not marker.exists()
    assert checkpoint.read_bytes() == b"previous complete checkpoint"


def test_sft_process_supervisor_preserves_worker_failure_and_success():
    supervise_command([sys.executable, "-c", "pass"], 5)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        supervise_command([sys.executable, "-c", "raise SystemExit(7)"], 5)
    assert failure.value.returncode == 7


@pytest.mark.skipif(os.name != "posix", reason="GPU host process-group guarantee targets Linux")
def test_sft_process_supervisor_also_kills_delayed_descendants(tmp_path):
    marker = tmp_path / "descendant-must-not-finish"
    child = (
        "import time; from pathlib import Path; time.sleep(1); "
        f"Path({str(marker)!r}).write_text('late')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(10)"
    )
    with pytest.raises(TimeoutError):
        supervise_command([sys.executable, "-c", parent], 0.3)
    time.sleep(1.2)
    assert not marker.exists()


def test_real_nested_diagnostics_are_used_with_explicit_missingness_and_no_admission_leak(records):
    signature = image_signature(records[0]["images"][0])
    diagnostics = {
        "evaluation": {"total": 0.7, "metrics": {"raw_min_font_px": 14, "hierarchy": 0}},
        "style": {"score": 0.8},
        "occlusion": {"coverage": {"complete": False}},
    }
    features = candidate_features(signature, signature, diagnostics)
    assert features[FEATURE_NAMES.index("evaluation.metrics.raw_min_font_px")] == 14
    assert features[FEATURE_NAMES.index("evaluation.metrics.hierarchy_present")] == 1
    assert features[FEATURE_NAMES.index("evaluation.metrics.text_contrast_present")] == 0
    assert features[FEATURE_NAMES.index("occlusion.coverage.complete_present")] == 1
    leaked = {
        **diagnostics,
        "admitted": True,
        "baseline_candidate_id": "A",
        "selection": {"safety": [99]},
        "provenance": {"origins": ["preferred"]},
    }
    np.testing.assert_array_equal(features, candidate_features(signature, signature, leaked))


def test_test_labels_are_not_unblinded_by_iterative_training_or_default_evaluation(records):
    test_row = {**records[-1], "pair_id": "secret-test", "split": "test"}
    all_rows = [*records, test_row]
    model = fit(all_rows, steps=5, allow_synthetic=True)
    report = evaluate_ranker(model, all_rows)
    assert "test" not in report["metrics"]
    assert "secret-test" not in report["margins"]
    assert all("test" not in baseline for baseline in report["baselines"].values())
    assert "test" in evaluate_ranker(model, all_rows, include_test=True)["metrics"]
    assert "test" not in evaluate_choices(all_rows, {})
    assert "test" in evaluate_choices(all_rows, {}, include_test=True)


@pytest.mark.parametrize(
    "invalid",
    [
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"l2": float("nan")},
        {"max_seconds": float("inf")},
    ],
)
def test_cpu_training_rejects_nonfinite_budgets_before_decoding_images(records, invalid):
    with pytest.raises(ValueError, match="recipe"):
        fit(records, allow_synthetic=True, **invalid)


def test_cpu_training_budget_includes_image_feature_preparation(records, monkeypatch):
    from experiments.vision_critic import train_ranker

    elapsed = [0.0]
    original = train_ranker.pair_features

    def slow_features(record):
        features = original(record)
        elapsed[0] += 10
        return features

    monkeypatch.setattr(train_ranker, "pair_features", slow_features)
    monkeypatch.setattr(train_ranker.time, "monotonic", lambda: elapsed[0])
    with pytest.raises(TimeoutError, match="feature preparation"):
        fit(records, max_seconds=5, allow_synthetic=True)


def test_cpu_model_rejects_feature_or_trainer_source_drift(records, tmp_path, monkeypatch):
    from experiments.vision_critic import train_ranker

    state = fit(records, steps=5, allow_synthetic=True)
    path = tmp_path / "model.json"
    save_model(path, state)
    monkeypatch.setattr(train_ranker, "implementation_identity", lambda: {"features.py": "changed"})
    with pytest.raises(ValueError, match="implementation"):
        load_model(path)
    with pytest.raises(ValueError, match="implementation"):
        fit(records, steps=10, resume=state, allow_synthetic=True)
    with pytest.raises(ValueError, match="implementation"):
        evaluate_ranker(state, records)


def test_external_verdicts_are_bound_to_corpus_and_honor_invalid_flags(records, tmp_path):
    path = tmp_path / "verdicts.jsonl"
    valid = make_verdict_record(records[0], '{"choice":"A","issues":[]}')
    path.write_text(json.dumps(valid) + "\n")
    assert read_verdicts(path, records) == {records[0]["pair_id"]: "A"}
    path.write_text(json.dumps({**valid, "valid": False}) + "\n")
    assert read_verdicts(path, records) == {records[0]["pair_id"]: None}
    path.write_text(json.dumps({**valid, "input_sha256": "b" * 64}) + "\n")
    with pytest.raises(ValueError, match="corpus"):
        read_verdicts(path, records)
    path.write_text(json.dumps({**valid, "unexpected": True}) + "\n")
    with pytest.raises(ValueError, match="schema"):
        read_verdicts(path, records)


def test_sft_interrupt_terminates_the_owned_worker_and_preserves_interrupt(monkeypatch):
    from experiments.vision_critic import train_sft

    created = []
    real_popen = subprocess.Popen

    class InterruptOnce(real_popen):
        interrupted = False

        def wait(self, *args, **kwargs):
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return super().wait(*args, **kwargs)

    def launch(*args, **kwargs):
        worker = InterruptOnce(*args, **kwargs)
        created.append(worker)
        return worker

    monkeypatch.setattr(train_sft.subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt):
        supervise_command([sys.executable, "-c", "import time; time.sleep(10)"], 5)
    assert len(created) == 1 and created[0].poll() is not None


def test_appending_held_out_labels_preserves_frozen_training_and_allows_explicit_test_evaluation(
    shared_corpus, tmp_path
):
    from test_vision_contracts import make_label

    from experiments.vision_critic.dataset import load_preferences, sft_records

    collection, labels = shared_corpus
    before = load_preferences(collection, labels, allow_synthetic=True)
    model = fit(before, steps=10, allow_synthetic=True)
    model_path = tmp_path / "frozen.json"
    save_model(model_path, model)
    plan_before = prepare_plan(before, sft_records(before), recipe(), allow_synthetic=True)
    progress = {
        "base_model": plan_before["model"],
        "base_revision": plan_before["base_revision"],
        "corpus_sha256": plan_before["corpus_sha256"],
        "recipe_sha256": plan_before["recipe_sha256"],
        "completed_steps": 2,
        "packages": plan_before["packages"],
        "trainer_sha256": plan_before["trainer_sha256"],
    }
    frozen_prediction = make_verdict_record(sft_records(before)[0], '{"choice":"A","issues":[]}')
    verdict_file = tmp_path / "frozen-verdicts.jsonl"
    verdict_file.write_text(json.dumps(frozen_prediction) + "\n")
    manifest = json.loads((collection / "collection.json").read_text())
    test_label = make_label(manifest, case_index=4, decision="A")
    with labels.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(test_label) + "\n")
    after = load_preferences(collection, labels, allow_synthetic=True)
    assert before[0]["corpus_sha256"] != after[0]["corpus_sha256"]
    restored = load_model(model_path)
    default_report = evaluate_ranker(restored, after)
    assert "test" not in default_report["metrics"]
    assert default_report["training_corpus_sha256"] == model["corpus_sha256"]
    assert default_report["full_corpus_provenance"] != model["full_corpus_provenance"]
    unblinded = evaluate_ranker(restored, after, include_test=True)
    assert unblinded["metrics"]["test"]["pair_count"] == 1
    resumed = fit(after, steps=20, resume=restored, allow_synthetic=True)
    assert resumed["completed_steps"] == 20
    assert resumed["corpus_sha256"] == model["corpus_sha256"]
    plan_after = prepare_plan(after, sft_records(after), recipe(), allow_synthetic=True)
    assert plan_after["corpus_sha256"] == plan_before["corpus_sha256"]
    validate_resume(progress, plan_after)
    assert read_verdicts(verdict_file, after) == {frozen_prediction["pair_id"]: "A"}

    changed_labels = [json.loads(line) for line in labels.read_text().splitlines()]
    changed_labels[0]["decision"] = "B"
    labels.write_text("".join(json.dumps(row) + "\n" for row in changed_labels))
    changed = load_preferences(collection, labels, allow_synthetic=True)
    with pytest.raises(ValueError, match="Resume"):
        fit(changed, steps=20, resume=restored, allow_synthetic=True)
    with pytest.raises(ValueError, match="train/validation"):
        evaluate_ranker(restored, changed, include_test=True)
    changed_plan = prepare_plan(changed, sft_records(changed), recipe(), allow_synthetic=True)
    with pytest.raises(ValueError, match="corpus"):
        validate_resume(progress, changed_plan)


def test_changed_training_issue_or_image_invalidates_cpu_resume_and_sft_identity(records):
    state = fit(records, steps=5, allow_synthetic=True)
    initial_plan = prepare_plan(records, sft_rows(records), recipe(), allow_synthetic=True)
    changed = deepcopy(records)
    changed[0]["issues"] = [{"candidate": "B", "code": "bad_overlap", "element_ids": []}]
    with pytest.raises(ValueError, match="Resume"):
        fit(changed, steps=10, resume=state, allow_synthetic=True)
    issue_plan = prepare_plan(changed, sft_rows(changed), recipe(), allow_synthetic=True)
    assert issue_plan["corpus_sha256"] != initial_plan["corpus_sha256"]
    Image.new("RGB", (64, 80), "yellow").save(records[0]["images"][1])
    with pytest.raises(ValueError, match="Resume"):
        fit(records, steps=10, resume=state, allow_synthetic=True)
    changed_plan = prepare_plan(records, sft_rows(records), recipe(), allow_synthetic=True)
    assert changed_plan["corpus_sha256"] != initial_plan["corpus_sha256"]
