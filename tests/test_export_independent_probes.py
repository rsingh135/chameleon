"""Neural-worker export preserves frozen identities and keeps gold separate."""

import builtins
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.vision_critic import independent_probes
from experiments.vision_critic.contracts import canonical_json_sha256, sha256_file
from tools import export_independent_probes as exporter


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    from test_independent_probes import collection as collection_fixture

    collection, _, _ = collection_fixture.__wrapped__(tmp_path, monkeypatch)
    root = tmp_path / "bundle"
    bundle = independent_probes.prepare_collection(
        collection, root, splits=("validation", "test"), modes=("critique", "analyze")
    )
    labels = tmp_path / "validation.json"
    case = next(row for row in bundle["cases"] if row["split"] == "validation")
    document = {
        "schema_version": 1,
        "bundle_sha256": bundle["bundle_sha256"],
        "label_kind": "synthetic",
        "decisions": [
            {
                "case_id": case["case_id"],
                "preferred_candidate_id": case["baseline_candidate_id"],
                "axes": {},
            }
        ],
    }
    labels.write_text(json.dumps(document), encoding="utf-8")
    return SimpleNamespace(
        root=root,
        bundle=bundle,
        path=root / "bundle.json",
        labels=labels,
        label_document=document,
        tmp=tmp_path,
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_exports_ordered_label_free_records_and_exact_identity_receipts(frozen, capsys):
    out = frozen.tmp / "export"
    assert exporter.main(["--bundle", str(frozen.path), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest == json.loads(capsys.readouterr().out)
    assert manifest["bundle_sha256"] == frozen.bundle["bundle_sha256"]
    assert manifest["exporter_sha256"] == sha256_file(Path(exporter.__file__))
    assert manifest["files"]["inputs.jsonl"] == sha256_file(out / "inputs.jsonl")
    assert not manifest["inference_executed"] and not manifest["test_labels_unblinded"]
    assert not (out / "validation-labels.jsonl").exists()
    rows = read_rows(out / "inputs.jsonl")
    assert [row["record_id"] for row in rows] == [p["probe_id"] for p in frozen.bundle["probes"]]
    for row, probe in zip(rows, frozen.bundle["probes"], strict=True):
        assert "label" not in row and row["label_origin"] == "withheld_evaluation"
        assert row["task"]["request_digest"] == probe["request_digest"]
        assert row["task"]["element_map"] == probe["element_map"]
        assert row["task"]["known_element_ids"] == probe["known_element_ids"]
        assert manifest["record_sha256"][row["record_id"]] == canonical_json_sha256(row)
        body = deepcopy(row)
        input_sha = body["review"].pop("record_input_sha256")
        assert input_sha == canonical_json_sha256(body)
        request = json.loads((frozen.root / probe["request_path"]).read_text(encoding="utf-8"))
        assert [image["role"] for image in row["images"]] == (
            ["source"] if probe["mode"] == "analyze" else ["source", "A", "B"]
        )
        refs = [request["source_render"], *[p["render"] for p in request.get("plans", [])]]
        assert [image["sha256"] for image in row["images"]] == [image["sha256"] for image in refs]
        for image in row["images"]:
            assert Path(image["path"]).is_absolute()
            assert sha256_file(image["path"]) == image["sha256"]


def test_validation_gold_is_separate_and_maps_reversed_and_opaque_plan_order(frozen):
    plain, labeled = frozen.tmp / "plain", frozen.tmp / "labeled"
    exporter.export_probes(frozen.path, plain)
    manifest = exporter.export_probes(frozen.path, labeled, labels_path=frozen.labels)
    assert (plain / "inputs.jsonl").read_bytes() == (labeled / "inputs.jsonl").read_bytes()
    gold = read_rows(labeled / "validation-labels.jsonl")
    assert {row["task"]["condition"]: row["label"] for row in gold} == {
        "original": "A",
        "swapped": "B",
        "opaque_ids": "A",
    }
    assert len(gold) == manifest["validation_gold_count"] == 3
    assert all(row["split"] == "validation" and row["label_origin"] == "synthetic" for row in gold)
    assert all(row["review"]["scope"] == "validation_only_not_sealed_test_truth" for row in gold)
    assert manifest["files"]["validation-labels.jsonl"] == sha256_file(
        labeled / "validation-labels.jsonl"
    )


@pytest.mark.parametrize("preferred", ["tie", "abstain"])
def test_explicit_nonwinner_gold_maps_to_N_without_fabricating_control_labels(frozen, preferred):
    frozen.label_document["decisions"][0]["preferred_candidate_id"] = preferred
    frozen.labels.write_text(json.dumps(frozen.label_document), encoding="utf-8")
    out = frozen.tmp / "export"
    exporter.export_probes(frozen.path, out, labels_path=frozen.labels)
    gold = read_rows(out / "validation-labels.jsonl")
    assert len(gold) == 3 and all(row["label"] == "N" for row in gold)
    assert all(row["task"]["condition"] not in {"blank_source", "replaced_source"} for row in gold)


@pytest.mark.parametrize("invalid", ["unknown_case", "unknown_candidate", "test_case"])
def test_unknown_or_test_gold_is_rejected_before_creating_export(frozen, invalid):
    row = frozen.label_document["decisions"][0]
    if invalid == "unknown_case":
        row["case_id"] = "unknown-case"
    elif invalid == "unknown_candidate":
        row["preferred_candidate_id"] = "invented-candidate"
    else:
        case = next(case for case in frozen.bundle["cases"] if case["split"] == "test")
        row.update(case_id=case["case_id"], preferred_candidate_id=case["baseline_candidate_id"])
    frozen.labels.write_text(json.dumps(frozen.label_document), encoding="utf-8")
    out = frozen.tmp / "export"
    with pytest.raises(ValueError, match="unknown|validation labels"):
        exporter.export_probes(frozen.path, out, labels_path=frozen.labels)
    assert not out.exists()


def test_default_export_never_opens_unrequested_test_labels_or_imports_model_libraries(
    frozen, monkeypatch
):
    secret = frozen.tmp / "test-labels.json"
    secret.write_text('{"secret": "sealed test judgments"}', encoding="utf-8")
    read_bytes, read_text, import_module = Path.read_bytes, Path.read_text, builtins.__import__

    def guarded_bytes(path, *args, **kwargs):
        assert path != secret, "Sealed test labels were opened"
        return read_bytes(path, *args, **kwargs)

    def guarded_text(path, *args, **kwargs):
        assert path != secret, "Sealed test labels were opened"
        return read_text(path, *args, **kwargs)

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "accelerate"}
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest = exporter.export_probes(frozen.path, frozen.tmp / "export")
    assert manifest["validation_labels_sha256"] is None
    assert not manifest["test_labels_unblinded"]


def test_existing_export_is_preserved_and_never_overwritten(frozen):
    out = frozen.tmp / "export"
    out.mkdir()
    sentinel = out / "inputs.jsonl"
    sentinel.write_bytes(b"existing export\n")
    with pytest.raises(FileExistsError):
        exporter.export_probes(frozen.path, out)
    assert sentinel.read_bytes() == b"existing export\n"


def test_record_budget_fails_explicitly_instead_of_silently_truncating(frozen):
    out = frozen.tmp / "export"
    with pytest.raises(ValueError, match="record budget"):
        exporter.export_probes(frozen.path, out, max_records=1)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["image", "element_id"])
def test_changed_frozen_pixels_or_unknown_original_ids_cannot_be_exported(frozen, mutation):
    if mutation == "image":
        image = next(path for path in frozen.bundle["files"] if path.startswith("images/"))
        (frozen.root / image).write_bytes(b"changed pixels")
    else:
        probe = frozen.bundle["probes"][0]
        presented = next(iter(probe["element_map"]))
        probe["element_map"][presented] = "invented-original-element"
        frozen.bundle["bundle_sha256"] = canonical_json_sha256(
            {key: value for key, value in frozen.bundle.items() if key != "bundle_sha256"}
        )
        frozen.path.write_text(json.dumps(frozen.bundle), encoding="utf-8")
    out = frozen.tmp / "export"
    with pytest.raises(ValueError, match="changed|unknown original"):
        exporter.export_probes(frozen.path, out)
    assert not out.exists()
