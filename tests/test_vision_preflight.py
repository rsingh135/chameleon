import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.vision_preflight import review, review_artifacts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/vision_critic/pilot.json").read_text())


def manifest():
    return {
        "schema_version": 1,
        "sources": [
            {
                "source_id": split,
                "family_id": split,
                "split": split,
                "source_parity_verified": True,
                "training_rights_reviewed": True,
                "rights_reference": "review-record",
                "human_preference_pairs": 3,
            }
            for split in ("train", "validation", "test")
        ],
    }


def test_real_recipe_does_not_claim_training_readiness(config):
    sources = json.loads((ROOT / config["dataset_manifest"]).read_text())
    result = review(config, sources)
    assert result["recipe_valid"]
    assert not result["task_training_ready"]
    assert result["training_blockers"]


def test_family_leakage_rejected_even_with_different_source_ids(config):
    data = manifest()
    data["sources"][1]["family_id"] = "train"
    assert "Family leakage" in " ".join(review(config, data)["errors"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_parity_verified", False),
        ("training_rights_reviewed", False),
        ("rights_reference", ""),
        ("human_preference_pairs", 0),
    ],
)
def test_dataset_claims_do_not_bypass_missing_review(config, field, value):
    config["task_trainer_implemented"] = True
    data = manifest()
    data["sources"][0][field] = value
    assert not review(config, data)["task_training_ready"]


def test_no_geometry_or_content_output_allowed(config):
    config["output_contract"] = "new_layout_json"
    assert not review(config, manifest())["recipe_valid"]


def test_session_cannot_exceed_pilot_allowance(config):
    config["budget"]["session_minutes"] = 600
    assert not review(config, manifest())["recipe_valid"]


def test_preflight_is_a_static_gate_without_compute_side_effects(config):
    original = copy.deepcopy(config)
    config["task_trainer_implemented"] = True
    report = review(config, manifest())
    assert report["task_training_ready"]
    assert not report["starts_compute"]
    assert not report["quality_validated"]
    assert not report["gpu_model_compatibility_validated"]
    config["task_trainer_implemented"] = original["task_trainer_implemented"]
    assert config == original


def test_task_prerequisites_do_not_require_unblinding_test_labels(config):
    data = manifest()
    data["sources"][2]["human_preference_pairs"] = 0
    assert review(config, data)["task_training_ready"]


def _artifacts(tmp_path, purpose="research"):
    from test_vision_contracts import build_collection, make_label

    data = build_collection(tmp_path / "collection", purpose=purpose, cases=3)
    for case, split in zip(data["cases"], ("train", "validation", "test"), strict=True):
        case["split"] = split
    (tmp_path / "collection/collection.json").write_text(json.dumps(data), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(json.dumps(make_label(data, case_index=index)) for index in (0, 1)) + "\n",
        encoding="utf-8",
    )
    return data, labels


def test_actual_artifact_gate_counts_verified_labels_instead_of_claims(config, tmp_path):
    _, labels = _artifacts(tmp_path)
    report = review_artifacts(config, tmp_path / "collection", labels)
    assert report["validated_label_count"] == 2
    assert report["artifact_hashes_verified"]
    assert report["task_training_ready"]
    assert not report["gpu_model_compatibility_validated"]


def test_actual_artifact_gate_refuses_changed_bytes(config, tmp_path):
    data, labels = _artifacts(tmp_path)
    path = tmp_path / "collection" / data["cases"][0]["source"]["render"]
    path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="hash"):
        review_artifacts(config, tmp_path / "collection", labels)


def test_synthetic_opt_in_never_becomes_task_training_approval(config, tmp_path):
    _, labels = _artifacts(tmp_path, "synthetic")
    report = review_artifacts(config, tmp_path / "collection", labels, allow_synthetic=True)
    assert not report["task_training_ready"]
    assert "Synthetic practice" in " ".join(report["training_blockers"])


def test_file_entrypoint_validates_artifacts_outside_repo_cwd(config, tmp_path):
    _, labels = _artifacts(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/vision_preflight.py"),
            "--collection",
            str(tmp_path / "collection"),
            "--labels",
            str(labels),
            "--require-training-ready",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["validated_label_count"] == 2
