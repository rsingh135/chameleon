"""The semantic export adapter supplies exact, label-free source-only inputs."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.vision_critic import independent_probes
from experiments.vision_critic.contracts import canonical_json_sha256, load_collection
from tools import prepare_semantic_eval as adapter


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def exported(tmp_path):
    dataset = tmp_path / "semantic-export"
    dataset.mkdir()
    source_rows, frozen_rows, files, references = [], [], {}, {}
    for number in range(8):
        source_id = f"source-{number}"
        split = "train" if number < 5 else "validation"
        family = f"family-{number}"
        source = {
            "canvas": {"width": 40, "height": 32},
            "elements": [
                {
                    "id": f"message-{number}",
                    "type": "text",
                    "x": 2,
                    "y": 2,
                    "width": 30,
                    "height": 8,
                    "font_size": 8,
                    "content": f"Café {number}",
                },
                {
                    "id": f"art-{number}",
                    "type": "shape",
                    "x": 12,
                    "y": 14,
                    "width": 14,
                    "height": 12,
                },
            ],
        }
        compact, spaced = adapter._source_digests(source)
        prefix = compact[:20]
        json_relative = f"receipts/{prefix}.source.json"
        png_relative = f"images/{prefix}.png"
        receipt_relative = f"receipts/{prefix}.json"
        files[json_relative] = write_json(dataset / json_relative, source)
        (dataset / "images").mkdir(exist_ok=True)
        Image.new("RGB", (40, 32), (number * 20, 60, 140)).save(dataset / png_relative)
        files[png_relative] = hashlib.sha256((dataset / png_relative).read_bytes()).hexdigest()
        receipt = {
            "schema_version": 1,
            "protocol": adapter.PROTOCOL,
            "source_id": source_id,
            "family": family,
            "split": split,
            "source_digest_protocol": "canonical-compact-v1",
            "source_digest": compact,
            "analysis_source_digest_protocol": "sorted-spaced-json-v1",
            "analysis_source_digest": spaced,
            "source_json_sha256": files[json_relative],
            "source_render_sha256": files[png_relative],
            "source_canvas": source["canvas"],
            "original_element_ids": [element["id"] for element in source["elements"]],
            "annotation_path": f"receipts/{prefix}.annotation.json",
            "annotation_sha256": "f" * 64,
            "reviewer": "SECRET_WEAK_LABEL_METADATA_NOT_A_MODEL_INPUT",
        }
        files[receipt_relative] = write_json(dataset / receipt_relative, receipt)
        write_json(dataset / f"receipts/{prefix}.annotation.json", {"labels": "SECRET_LABELS"})
        source_rows.append(
            {
                "source_id": source_id,
                "split": split,
                "receipt": receipt_relative,
                "sha256": files[receipt_relative],
                "element_count": len(source["elements"]),
            }
        )
        frozen_rows.append({"source_id": source_id, "family": family, "split": split})
        references[source_id] = {
            "source": source,
            "receipt": receipt,
            "receipt_relative": receipt_relative,
            "json_relative": json_relative,
            "png_relative": png_relative,
        }
    for name in ("analysis.jsonl", "elements.jsonl", "relationships.jsonl"):
        (dataset / name).write_text("SECRET_TRAINING_AND_VALIDATION_LABELS\n")
        files[name] = hashlib.sha256((dataset / name).read_bytes()).hexdigest()
    frozen = {
        "schema_version": 1,
        "protocol": adapter.SPLIT_PROTOCOL,
        "status": "frozen_before_training_variants",
        "split_unit": "authored_source_work",
        "sources": frozen_rows,
        "excluded_source_ids": ["opensuse_banner_we_believe", "doory_modern-resume"],
        "excluded_splits": ["reserve", "manual_acceptance"],
    }
    files["frozen-splits.json"] = write_json(dataset / "frozen-splits.json", frozen)
    manifest = {
        "schema_version": 1,
        "protocol": adapter.PROTOCOL,
        "split_manifest_sha256": files["frozen-splits.json"],
        "frozen_split": frozen,
        "sources": source_rows,
        "files_sha256": files,
    }
    write_json(dataset / "manifest.json", manifest)
    return dataset, manifest, references


def test_default_bridge_copies_three_validation_sources_without_claiming_layout_validity(
    exported, tmp_path
):
    dataset, _, refs = exported
    inputs = tmp_path / "source-inputs"
    before = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
    result = adapter.build_source_collection(dataset, inputs)
    assert load_collection(inputs) == result
    assert result["purpose"] == "research"
    assert {case["source_id"] for case in result["cases"]} == {"source-5", "source-6", "source-7"}
    assert all(case["split"] == "validation" for case in result["cases"])
    assert {case["family_id"] for case in result["cases"]} == {"family-5", "family-6", "family-7"}
    for case in result["cases"]:
        original = refs[case["source_id"]]
        source, candidate = case["source"], case["candidates"][0]
        assert (inputs / source["design"]).read_bytes() == before[
            dataset / original["json_relative"]
        ]
        assert (inputs / source["render"]).read_bytes() == before[
            dataset / original["png_relative"]
        ]
        assert case["target_canvas"] == original["source"]["canvas"]
        assert case["baseline_candidate_id"] == original["receipt"]["source_digest"]
        assert not case["source_parity_verified"] and not case["training_rights_reviewed"]
        assert case["admitted_candidate_ids"] == [] and candidate["admitted"] is False
        assert candidate["design"] == source["design"] and candidate["render"] == source["render"]
        assert candidate["diagnostics"]["contract_valid"] is False
        assert candidate["diagnostics"]["contract_violations"] == ["not_evaluated_source_only"]
        assert source["json_sha256"] == canonical_json_sha256(original["source"])
    assert {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()} == before


def test_labels_and_unselected_receipts_are_never_opened_or_copied(exported, tmp_path, monkeypatch):
    dataset, _, refs = exported
    read_bytes, read_text = Path.read_bytes, Path.read_text
    unselected = {
        value[key]
        for source_id, value in refs.items()
        if int(source_id.split("-")[-1]) < 5
        for key in ("receipt_relative", "json_relative", "png_relative")
    }

    def guard(path):
        if path.is_relative_to(dataset):
            relative = path.relative_to(dataset).as_posix()
            assert relative not in unselected
            assert not relative.endswith(".jsonl") and ".annotation." not in relative

    def bytes_guard(path):
        guard(path)
        return read_bytes(path)

    def text_guard(path, *args, **kwargs):
        guard(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", bytes_guard)
    monkeypatch.setattr(Path, "read_text", text_guard)
    inputs = tmp_path / "source-inputs"
    adapter.build_source_collection(dataset, inputs)
    combined = b"".join(read_bytes(path) for path in inputs.rglob("*") if path.is_file())
    assert b"SECRET_" not in combined
    assert not any(
        path.suffix == ".jsonl" or ".annotation." in path.name for path in inputs.rglob("*")
    )


def test_source_only_builder_produces_twelve_label_free_analysis_probes(
    exported, tmp_path, monkeypatch
):
    code_root = tmp_path / "test-code"
    code_root.mkdir()
    (code_root / "adapter.py").write_text("# frozen test code\n")
    monkeypatch.setattr(independent_probes, "ROOT", code_root)
    monkeypatch.setattr(independent_probes, "CODE_FILES", ("adapter.py",))
    monkeypatch.setattr(
        independent_probes.subprocess, "check_output", lambda *_a, **_k: "fixture\n"
    )
    out = tmp_path / "analysis-bundle"
    result = adapter.prepare_semantic_eval(exported[0], out)
    assert independent_probes.load_bundle(out / "bundle.json") == result
    assert len(result["cases"]) == 3 and len(result["probes"]) == 12
    assert {row["condition"] for row in result["probes"]} == {
        "original",
        "blank_source",
        "replaced_source",
        "opaque_ids",
    }
    for probe in result["probes"]:
        request = json.loads((out / probe["request_path"]).read_text(encoding="utf-8"))
        assert request["mode"] == "analyze"
        assert not {"target_canvas", "plan_menu", "plans", "source_analysis"}.intersection(request)
        assert all("purpose" not in element for element in request["source_summary"]["elements"])
        assert probe["split"] == "validation"
        assert probe["plan_map"] == {} and probe["menu_digest"] is None


@pytest.mark.parametrize("artifact", ["receipt", "json", "png", "split"])
def test_changed_selected_export_bytes_are_rejected_before_outputs(exported, tmp_path, artifact):
    dataset, _, refs = exported
    relative = {
        "receipt": refs["source-5"]["receipt_relative"],
        "json": refs["source-5"]["json_relative"],
        "png": refs["source-5"]["png_relative"],
        "split": "frozen-splits.json",
    }[artifact]
    path = dataset / relative
    path.write_bytes(path.read_bytes() + b"changed")
    inputs = tmp_path / "source-inputs"
    with pytest.raises(ValueError, match="bytes differ"):
        adapter.build_source_collection(dataset, inputs)
    assert not inputs.exists()


def test_unselected_file_corruption_does_not_unseal_or_block_validation_sources(exported, tmp_path):
    dataset, _, refs = exported
    (dataset / refs["source-0"]["receipt_relative"]).write_text("invalid and intentionally unread")
    result = adapter.build_source_collection(dataset, tmp_path / "source-inputs")
    assert len(result["cases"]) == 3


@pytest.mark.parametrize("change", ["split", "family", "unknown", "excluded"])
def test_frozen_source_assignment_changes_are_rejected(exported, tmp_path, change):
    dataset, manifest, _ = exported
    if change == "split":
        manifest["sources"][5]["split"] = "train"
    elif change == "family":
        manifest["frozen_split"]["sources"][5]["family"] = "family-0"
    elif change == "unknown":
        manifest["sources"][5]["source_id"] = "invented-source"
    else:
        manifest["frozen_split"]["excluded_source_ids"].append("source-5")
    if change in {"family", "excluded"}:
        split_sha = write_json(dataset / "frozen-splits.json", manifest["frozen_split"])
        manifest["split_manifest_sha256"] = split_sha
        manifest["files_sha256"]["frozen-splits.json"] = split_sha
    write_json(dataset / "manifest.json", manifest)
    with pytest.raises(ValueError, match="frozen|Family|excluded"):
        adapter.build_source_collection(dataset, tmp_path / "source-inputs")


@pytest.mark.parametrize(
    "field", ["source_digest", "analysis_source_digest", "original_element_ids"]
)
def test_receipt_semantic_identity_cannot_override_verified_source_json(exported, tmp_path, field):
    dataset, manifest, refs = exported
    reference = refs["source-5"]
    receipt = reference["receipt"]
    receipt[field] = ["unknown"] if field == "original_element_ids" else "0" * 64
    receipt_sha = write_json(dataset / reference["receipt_relative"], receipt)
    manifest["sources"][5]["sha256"] = receipt_sha
    manifest["files_sha256"][reference["receipt_relative"]] = receipt_sha
    write_json(dataset / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identity|filename"):
        adapter.build_source_collection(dataset, tmp_path / "source-inputs")


def test_existing_outputs_and_frozen_dataset_are_never_overwritten(exported, tmp_path):
    dataset, _, _ = exported
    out = tmp_path / "existing-bundle"
    out.mkdir()
    (out / "keep.txt").write_text("earlier run")
    with pytest.raises(FileExistsError):
        adapter.prepare_semantic_eval(dataset, out)
    assert (out / "keep.txt").read_text() == "earlier run"
    with pytest.raises(ValueError, match="frozen"):
        adapter.build_source_collection(dataset, dataset / "new-output")
    with pytest.raises(ValueError, match="separate"):
        adapter.prepare_semantic_eval(dataset, tmp_path / "same", tmp_path / "same")


@pytest.mark.parametrize("options", [{"split": "test"}, {"max_sources": 0}, {"max_sources": True}])
def test_adapter_does_not_unseal_test_or_accept_unbounded_source_budgets(
    exported, tmp_path, options
):
    with pytest.raises(ValueError, match="source budget"):
        adapter.build_source_collection(exported[0], tmp_path / "source-inputs", **options)
