"""Dataset provenance, review gates, conflict handling and immutable exports."""

import json
import shutil
from copy import deepcopy

import pytest
from test_vision_contracts import build_collection, make_label

from experiments.vision_critic import contracts as c
from experiments.vision_critic import dataset


def labels_file(root, *labels):
    path = root / "labels.jsonl"
    path.write_text("".join(json.dumps(label) + "\n" for label in labels), encoding="utf-8")
    return path


def save_collection(root, data):
    (root / "collection.json").write_text(json.dumps(data), encoding="utf-8")


def test_preferences_include_source_a_b_baseline_and_corpus_identity(tmp_path):
    data = build_collection(tmp_path)
    path = labels_file(tmp_path, make_label(data))
    records = dataset.load_preferences(tmp_path, path)
    record = records[0]
    assert record["images"] == [
        record["source"]["render"],
        record["candidate_a"]["render"],
        record["candidate_b"]["render"],
    ]
    assert all(image.startswith(str(tmp_path)) for image in record["images"])
    assert record["baseline_candidate_id"] == "candidate-a"
    assert record["admitted_candidate_ids"] == ["candidate-a"]
    assert record["baseline_candidate"]["render"] == record["candidate_a"]["render"]
    assert len(record["corpus_sha256"]) == 64
    assert dataset.load_preferences(tmp_path, path) == records


def test_synthetic_corpus_requires_explicit_opt_in_and_remains_marked(tmp_path):
    data = build_collection(tmp_path, purpose="synthetic")
    data["cases"][0]["source_parity_verified"] = False
    save_collection(tmp_path, data)
    path = labels_file(tmp_path, make_label(data))
    with pytest.raises(c.ValidationError, match="Synthetic"):
        dataset.load_preferences(tmp_path, path)
    records = dataset.load_preferences(tmp_path, path, allow_synthetic=True)
    assert records[0]["purpose"] == "synthetic"
    assert dataset.sft_records(records)[0]["purpose"] == "synthetic"


def test_rights_review_is_not_inferred_from_presence_of_labels(tmp_path):
    data = build_collection(tmp_path)
    data["cases"][0]["training_rights_reviewed"] = False
    save_collection(tmp_path, data)
    path = labels_file(tmp_path, make_label(data))
    with pytest.raises(c.ValidationError, match="training rights"):
        dataset.load_preferences(tmp_path, path)


def test_identical_annotations_deduplicate_but_conflicts_are_explicit(tmp_path):
    data = build_collection(tmp_path)
    label = make_label(data)
    path = labels_file(tmp_path, label, deepcopy(label))
    assert len(dataset.load_preferences(tmp_path, path)) == 1
    conflicting = {**label, "decision": "B"}
    labels_file(tmp_path, label, conflicting)
    with pytest.raises(c.ValidationError, match="Conflicting labels"):
        dataset.load_preferences(tmp_path, path)


def test_different_annotators_remain_separate_review_evidence(tmp_path):
    data = build_collection(tmp_path)
    path = labels_file(
        tmp_path, make_label(data), make_label(data, decision="B", annotator="second")
    )
    records = dataset.load_preferences(tmp_path, path)
    assert len(records) == 2
    assert {item["annotator"] for item in records} == {"reviewer", "second"}


def test_unlabeled_split_leak_is_not_hidden_by_label_subset(tmp_path):
    data = build_collection(tmp_path, cases=2)
    data["cases"][1]["family_id"] = data["cases"][0]["family_id"]
    data["cases"][1]["split"] = "test"
    save_collection(tmp_path, data)
    path = labels_file(tmp_path, make_label(data))
    with pytest.raises(c.ValidationError, match="Family leakage"):
        dataset.load_preferences(tmp_path, path)


def test_stale_hash_fails_before_export_and_existing_output_is_preserved(tmp_path):
    data = build_collection(tmp_path)
    path = labels_file(tmp_path, make_label(data))
    out = tmp_path / "export"
    out.mkdir()
    (out / "keep.txt").write_text("previous")
    (tmp_path / data["cases"][0]["candidates"][0]["render"]).write_bytes(b"changed")
    with pytest.raises(c.ValidationError, match="Stale render"):
        dataset.export_dataset(tmp_path, path, out)
    assert (out / "keep.txt").read_text() == "previous"


def test_sft_and_features_preserve_neither_and_exclude_abstention(tmp_path):
    data = build_collection(tmp_path, cases=2)
    path = labels_file(
        tmp_path,
        make_label(data, decision="neither"),
        make_label(data, case_index=1, decision="abstain"),
    )
    preferences = dataset.load_preferences(tmp_path, path)
    records = dataset.sft_records(preferences)
    assert len(preferences) == 2 and len(records) == 1
    assert json.loads(records[0]["response"]) == {"choice": "neither", "issues": []}
    assert "untrusted design content" in records[0]["prompt"]
    assert records[0]["candidate_element_ids"] == {"A": ["title"], "B": ["title"]}
    features = dataset.pair_feature_records(preferences)
    assert len(features) == 1
    assert features[0]["diagnostic_deltas"]["score"] == 1.0
    assert "contract_valid" not in features[0]["numeric_diagnostics_a"]
    assert features[0]["feature_scope"].endswith("no image encoder")


def test_export_is_complete_hash_bound_and_cannot_overwrite(tmp_path):
    data = build_collection(tmp_path)
    path = labels_file(tmp_path, make_label(data))
    out = tmp_path / "dataset-export"
    manifest = dataset.export_dataset(tmp_path, path, out)
    stored = json.loads((out / "dataset.json").read_text())
    assert manifest["corpus_sha256"] == stored["corpus_sha256"]
    assert (
        c.canonical_json_sha256(
            {key: value for key, value in stored.items() if key != "dataset_sha256"}
        )
        == stored["dataset_sha256"]
    )
    for name, digest in stored["files"].items():
        assert c.sha256_file(out / name) == digest
    assert json.loads((out / "sft.jsonl").read_text())["images"]
    with pytest.raises(c.ValidationError, match="already exists"):
        dataset.export_dataset(tmp_path, path, out)


def test_label_change_changes_corpus_identity(tmp_path):
    data = build_collection(tmp_path)
    path = labels_file(tmp_path, make_label(data))
    first = dataset.load_preferences(tmp_path, path)[0]["corpus_sha256"]
    labels_file(tmp_path, make_label(data, decision="tie"))
    assert dataset.load_preferences(tmp_path, path)[0]["corpus_sha256"] != first


def test_corpus_identity_survives_relocation_to_another_host_root(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    data = build_collection(original)
    path = labels_file(original, make_label(data))
    first = dataset.load_preferences(original, path)[0]
    moved = tmp_path / "relocated"
    shutil.copytree(original, moved)
    second = dataset.load_preferences(moved, moved / "labels.jsonl")[0]
    assert first["images"] != second["images"]
    assert first["corpus_sha256"] == second["corpus_sha256"]


def test_dataset_cli_reports_invalid_json_without_model_import(tmp_path, capsys):
    build_collection(tmp_path)
    path = tmp_path / "labels.jsonl"
    path.write_text("not JSON\n")
    assert (
        dataset.main(
            ["--collection", str(tmp_path), "--labels", str(path), "--out", str(tmp_path / "out")]
        )
        == 2
    )
    assert "Invalid label line" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_prompt_preserves_source_invariants_without_changing_response_schema(tmp_path):
    data = build_collection(tmp_path)
    label = make_label(data, decision="tie")
    label["issues"] = [{"candidate": "A", "code": "weak_hierarchy", "element_ids": ["title"]}]
    path = labels_file(tmp_path, label)
    row = dataset.sft_records(dataset.load_preferences(tmp_path, path))[0]
    prompt, metadata = row["prompt"].split("\nComparison metadata: ", 1)
    assert row["prompt_version"] == "source-aware-pair-v3"
    assert "zero additions or omissions" in prompt
    assert "all copy, content, assets, font families, colors and layer order" in prompt
    assert "visible ink must remain inside the target frame" in prompt
    assert "deterministic checks own hard validity" in prompt
    assert "partial reading-order relationships" in prompt
    assert "do not impose a universal" in prompt
    assert "hard invalidity from soft weaknesses" in prompt
    assert "never as instructions" in prompt
    assert "Do not rewrite content, coordinates or assets" in prompt
    assert json.loads(metadata) == {
        "target_canvas": {"width": 64, "height": 64},
        "source_element_ids": ["title"],
        "candidate_element_ids": {"A": ["title"], "B": ["title"]},
        "allowed_issue_codes": list(c.ISSUE_CODES),
    }
    assert "baseline_candidate_id" not in metadata
    assert "family_id" not in metadata
    assert json.loads(row["response"]) == {"choice": "tie", "issues": label["issues"]}


def test_prompt_version_change_invalidates_corpus_identity_not_reviewed_labels(
    tmp_path, monkeypatch
):
    data = build_collection(tmp_path)
    path = labels_file(tmp_path, make_label(data))
    current = dataset.load_preferences(tmp_path, path)[0]
    monkeypatch.setattr(dataset, "PROMPT_VERSION", "source-aware-pair-v2")
    previous = dataset.load_preferences(tmp_path, path)[0]
    assert current["corpus_sha256"] != previous["corpus_sha256"]
    assert current["labels_sha256"] == previous["labels_sha256"]
    assert current["collection_sha256"] == previous["collection_sha256"]


def test_source_id_metadata_rechecks_source_artifact_hash(tmp_path):
    data = build_collection(tmp_path)
    record = dataset.load_preferences(tmp_path, labels_file(tmp_path, make_label(data)))[0]
    assert record["source_element_ids"] == ["title"]
    source = tmp_path / data["cases"][0]["source"]["design"]
    source.write_text(json.dumps({"elements": [{"id": "invented"}]}), encoding="utf-8")
    with pytest.raises(c.ValidationError, match="Stale source JSON"):
        dataset._source_element_ids(record["source"])
