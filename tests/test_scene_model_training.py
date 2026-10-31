"""Actual CPU optimizer/checkpoint tests; toy vectors do not prove scene judgment."""

import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

from experiments.scene_model import train_clip as training


@pytest.fixture
def recipe():
    return {
        "max_steps": 30,
        "save_every": 10,
        "max_seconds": 10,
        "learning_rate": 0.1,
        "l2": 0.001,
        "seed": 2026,
    }


@pytest.fixture
def features():
    vectors, labels = [], []
    for role_index, role in enumerate(training.ROLES):
        for _ in range(2):
            vector = np.zeros(1024)
            vector[role_index], vector[512 + role_index] = 1, 1
            vectors.append(vector)
            labels.append(role)
    return np.asarray(vectors), labels


def _rows():
    return [
        {
            "record_id": f"r-{i}",
            "family_id": f"f-{i}",
            "source_id": f"s-{i}",
            "source_digest": str(i) * 64,
            "input_digest": str(i) * 64,
            "split": split,
            "supervision_origin": "agent_visual_review",
            "prediction": {"proposed_role": "background"},
        }
        for i, split in enumerate(("train", "validation"))
    ]


def test_real_optimizer_updates_weights_and_decreases_class_balanced_loss(features, recipe):
    vectors, labels = features
    result = training.fit(vectors, labels, recipe)
    assert result["completed_steps"] == recipe["max_steps"]
    assert result["loss_history"][-1] < result["loss_history"][0]
    scores = training.probabilities(vectors, result["weights"], result["bias"])
    assert [training.ROLES[int(i)] for i in scores.argmax(axis=1)] == labels


def test_duplicating_one_class_does_not_change_its_total_loss_mass(features, recipe):
    vectors, labels = features
    one_step = {**recipe, "max_steps": 1}
    reference = training.fit(vectors, labels, one_step)
    duplicated = training.fit(
        np.vstack([vectors, vectors[:2], vectors[:2]]),
        [*labels, *labels[:2], *labels[:2]],
        one_step,
    )
    assert np.allclose(reference["weights"], duplicated["weights"], atol=1e-14, rtol=0)
    assert np.allclose(reference["bias"], duplicated["bias"], atol=1e-14, rtol=0)


def test_checkpoint_reload_and_resume_match_an_uninterrupted_optimizer(features, recipe, tmp_path):
    vectors, labels = features
    first_recipe = {**recipe, "max_steps": 10}
    plan = training.prepare_plan(_rows(), first_recipe, "a" * 64)
    first = training.fit(vectors, labels, first_recipe)
    checkpoint = training.publish_checkpoint(tmp_path, plan, "b" * 64, first)
    training.verify_checkpoint(checkpoint)
    resumed_plan = training.prepare_plan(_rows(), recipe, "a" * 64)
    initial = training.load_resume(checkpoint, resumed_plan, "b" * 64)
    resumed = training.fit(vectors, labels, recipe, initial=initial)
    full = training.fit(vectors, labels, recipe)
    assert resumed == full


def test_corrupted_checkpoint_is_rejected(features, recipe, tmp_path):
    vectors, labels = features
    plan = training.prepare_plan(_rows(), recipe, "a" * 64)
    checkpoint = training.publish_checkpoint(
        tmp_path, plan, "b" * 64, training.fit(vectors, labels, recipe)
    )
    with (checkpoint / "head.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="published hashes"):
        training.verify_checkpoint(checkpoint)


def test_train_labels_change_resume_identity_but_later_test_labels_do_not(recipe):
    rows = _rows()
    plan = training.prepare_plan(rows, recipe, "a" * 64)
    updated = deepcopy(rows)
    updated[0]["input_digest"] = "c" * 64
    assert (
        training.prepare_plan(updated, recipe, "a" * 64)["corpus_sha256"] != plan["corpus_sha256"]
    )
    extra = {
        **rows[0],
        "record_id": "test",
        "source_id": "unseen-test",
        "source_digest": "d" * 64,
        "family_id": "unseen-test",
        "split": "test",
        "input_digest": "e" * 64,
    }
    assert (
        training.prepare_plan([*rows, extra], recipe, "a" * 64)["corpus_sha256"]
        == plan["corpus_sha256"]
    )


def test_missing_heldout_product_class_is_reported_without_a_recall_claim():
    report = training.evaluate(["background", "logo"], ["background", "logo"])
    assert report["accuracy"] == 1
    assert "product" in report["missing_roles"]
    assert "product" not in report["per_role_recall"]
    assert report["quality_validated"] is False


@pytest.mark.parametrize("mutation", ["not_normalized", "nan", "dimension"])
def test_incompatible_or_nonfinite_features_are_rejected(features, mutation):
    vectors, _ = features
    vectors = vectors.copy()
    if mutation == "not_normalized":
        vectors *= 0.5
    elif mutation == "nan":
        vectors[0, 0] = float("nan")
    else:
        vectors = vectors[:, :-1]
    with pytest.raises(ValueError):
        training.validate_features(vectors)


def test_ablated_view_is_explicitly_experimental(features, recipe):
    vectors, labels = features
    result = training.fit(vectors, labels, recipe)
    vectors = vectors.copy()
    vectors[:, :512] = 0
    with pytest.raises(ValueError, match="normalized"):
        training.probabilities(vectors, result["weights"], result["bias"])
    values = training.probabilities(vectors, result["weights"], result["bias"], ablation=True)
    assert np.allclose(values.sum(axis=1), 1)


def test_synthetic_records_do_not_pass_real_task_training_gate(recipe):
    rows = _rows()
    rows[0]["supervision_origin"] = "synthetic_rule"
    with pytest.raises(ValueError, match="synthetic infrastructure"):
        training.prepare_plan(rows, recipe, "a" * 64)


def test_worker_deadline_terminates_actual_local_process(monkeypatch):
    original = subprocess.Popen
    workers = []

    def capture(*args, **kwargs):
        worker = original(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(training.subprocess, "Popen", capture)
    with pytest.raises(subprocess.TimeoutExpired):
        training.supervise_command([sys.executable, "-c", "import time;time.sleep(10)"], 0.1)
    assert len(workers) == 1 and workers[0].poll() is not None
