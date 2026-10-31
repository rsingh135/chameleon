"""Task masks preserve raw purpose evidence; fixture bytes never train a model."""

from __future__ import annotations

import base64
from copy import deepcopy

import pytest

from experiments.visual_finetune.data import purpose_rows
from tools import purpose_task_projection as tool

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j6QAAAABJRU5ErkJggg=="
)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path / "parent"
    root.mkdir()
    rows, sources, proposal_records = [], [], []
    for sid, split, count in (("train-scene", "train", 20), ("validation-scene", "validation", 3)):
        elements = [
            {
                "id": f"e-{index}",
                "type": "text",
                "content": f"copy {index}",
                "x": 0,
                "y": 0,
                "width": 1,
                "height": 1,
                "z_index": index,
            }
            for index in range(count)
        ]
        source_rel = f"sources/{sid}/source.json"
        tool._write(root / source_rel, {"canvas": {"width": 1, "height": 1}, "elements": elements})
        receipt_rel = f"receipts/{sid}.json"
        tool._write(
            root / receipt_rel,
            {
                "source_id": sid,
                "split": split,
                "source_json_sha256": tool._sha(root / source_rel),
                "resources": {},
            },
        )
        sources.append(
            {
                "source_id": sid,
                "split": split,
                "element_count": count,
                "source_json": source_rel,
                "receipt": receipt_rel,
            }
        )
        image_rel = f"images/{sid}.png"
        (root / image_rel).parent.mkdir(exist_ok=True)
        (root / image_rel).write_bytes(PNG)
        for index, element in enumerate(elements):
            identity = f"{sid}-{index}"
            reviewed = not (split == "train" and index >= 18)
            purpose = (
                "supporting_copy"
                if split == "train" and index < 14
                else ("body" if reviewed else "unknown")
            )
            isolated = f"isolated-checker/{identity}.png"
            alpha = f"isolated-alpha/{identity}.png"
            context = f"context-crops/{identity}.png"
            iso_receipt = f"isolated-receipts/{identity}.json"
            for relative in (isolated, alpha, context):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_bytes(PNG)
            tool._write(
                root / iso_receipt,
                {
                    "record_id": identity,
                    "source_element": element,
                    "status": "available",
                    "clipped": False,
                    "label_used_for_paint_choices": False,
                    "native_single_original_layer": True,
                    "no_replacement_asset_or_content": True,
                    "resource_hashes": {},
                    "isolated_alpha_image": {"path": alpha, "sha256": tool._sha(root / alpha)},
                },
            )
            row = {
                "record_id": identity,
                "split": split,
                "review_mask": reviewed,
                "training_eligible": split == "train" and reviewed,
                "supervision_origin": "agent_visual_review",
                "target": {"purpose": purpose, "confidence": 0.9},
                "receipt": receipt_rel,
                "input": {
                    "source_image": image_rel,
                    "context_image": context,
                    "isolated_image": {"path": isolated, "sha256": tool._sha(root / isolated)},
                    "bbox_normalized": [0, 0, 1, 1],
                    "type": "text",
                    "text_data": element["content"],
                },
                "provenance": {
                    "element_index": index,
                    "reviewed": reviewed,
                    "uncertain": not reviewed,
                },
                "isolated_evidence": {
                    "available": True,
                    "receipt": iso_receipt,
                    "receipt_sha256": tool._sha(root / iso_receipt),
                },
            }
            rows.append(row)
            if split == "train" and index < 14:
                paths = {
                    "source_json": source_rel,
                    "source_receipt": receipt_rel,
                    "full_native_png": image_rel,
                    "context_png": context,
                    "isolated_checker_png": isolated,
                    "isolated_alpha_png": alpha,
                    "isolated_receipt": iso_receipt,
                }
                proposal_records.append(
                    {
                        "record_id": identity,
                        "source_id": sid,
                        "family_id": sid,
                        "split": "train",
                        "element_id": element["id"],
                        "source_element_index": index,
                        "authored_paint_order_index": index,
                        "copy_exact": element["content"],
                        "canonical_purpose_unchanged": "supporting_copy",
                        "canonical_model_word_unchanged": "support",
                        "canonical_label_origin_unchanged": "agent_visual_review",
                        "wrong_canonical_label_established": False,
                        "automatic_relabel": False,
                        "human_reviewed": False,
                        "task_decision": "mask_ambiguous"
                        if index < 8
                        else "retain_supporting_copy",
                        "task_training_eligible": index >= 8,
                        "pixel_and_source_refs": {
                            role: {"path": rel, "sha256": tool._sha(root / rel)}
                            for role, rel in paths.items()
                        },
                    }
                )
    tool._jsonl(root / "elements.jsonl", rows)
    tool._jsonl(root / "train-elements.jsonl", [row for row in rows if row["training_eligible"]])
    tool._jsonl(
        root / "validation-elements.jsonl", [row for row in rows if row["split"] == "validation"]
    )
    tool._jsonl(root / "masked-elements.jsonl", [row for row in rows if not row["review_mask"]])
    tool._write(
        root / "parent-manifest.json",
        {
            "protocol": "semantic-analysis-training-data-v1",
            "human_reviewed": False,
            "sources": sources,
        },
    )
    files = {
        path.relative_to(root).as_posix(): tool._sha(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    parent = root / "manifest.json"
    tool._write(
        parent,
        {
            "schema_version": 1,
            "protocol": "semantic-purpose-isolated-layer-evidence-v1",
            "parent_manifest_sha256": tool._sha(root / "parent-manifest.json"),
            "record_count": len(rows),
            "files_sha256": files,
        },
    )
    proposal = tmp_path / "proposal.json"
    definition = {"version": "test-support-definition", "uncertainty_policy": "mask, never unknown"}
    decision = {
        "schema_version": 1,
        "protocol": "supporting-copy-task-projection-v1",
        "role_definition": definition,
        "role_definition_sha256": tool._digest(definition),
        "records": proposal_records,
        "task_mask_record_ids": [row["record_id"] for row in proposal_records[:8]],
        "task_retain_record_ids": [row["record_id"] for row in proposal_records[8:]],
    }
    decision.update(
        {
            key: False
            for key in (
                "canonical_labels_changed",
                "source_content_or_pixels_changed",
                "existing_exports_or_annotations_edited",
                "old_validation_or_sealed_gold_edited",
                "sealed_gold_inspected",
                "human_gold",
                "quality_approved",
            )
        }
    )
    tool._write(proposal, decision)
    monkeypatch.setattr(tool, "PARENT_SHA", tool._sha(parent))
    monkeypatch.setattr(tool, "PROPOSAL_SHA", tool._sha(proposal))
    monkeypatch.setattr(tool, "DEFINITION_SHA", tool._digest(definition))
    monkeypatch.setattr(tool, "EXPECTED_FILES", len(files))
    monkeypatch.setattr(tool, "MASK_IDS", frozenset(decision["task_mask_record_ids"]))
    return {"parent_manifest": parent, "proposal": proposal, "out": tmp_path / "projected"}


def test_projection_preserves_all_raw_fields_bytes_and_actual_loader(cohort):
    original = tool._rows(cohort["parent_manifest"].parent / "elements.jsonl")
    before = {
        path: tool._sha(path)
        for path in cohort["parent_manifest"].parent.rglob("*")
        if path.is_file()
    }
    projected = tool.project_tasks(**cohort)
    active = tool._rows(cohort["out"] / "elements.jsonl")
    assert [
        {key: value for key, value in row.items() if key not in tool.TASK_FIELDS} for row in active
    ] == original
    assert {path: tool._sha(path) for path in before} == before
    assert projected["task_counts"] == {
        "raw_rows": 23,
        "raw_reviewed": 21,
        "existing_raw_masked": 2,
        "new_task_masks": 8,
        "task_masked": 10,
        "task_eligible": 13,
        "train": 10,
        "validation": 3,
        "reviewed_support_unchanged": 14,
        "retained_support": 6,
    }
    assert sum(row["target"]["purpose"] == "unknown" for row in active) == 2
    assert all(
        row["review_mask"] and not row["task_mask"]
        for row in active
        if row["record_id"] in tool.MASK_IDS
    )
    assert all(row["task_mask"] for row in active if row["split"] == "validation")
    receipt = tool._read(cohort["out"] / "task-projection/receipt.json")
    assert receipt["wrong_label_established"] is receipt["masked_to_unknown"] is False
    for original_path, reference in receipt["original_artifacts"].items():
        assert tool._sha(cohort["out"] / reference["path"]) == reference["sha256"]
        assert reference["sha256"] == tool._sha(cohort["parent_manifest"].parent / original_path)
    loaded, _ = purpose_rows(cohort["out"] / "manifest.json", purpose_views=3)
    assert len(loaded) == 13 and all(row["record_id"] not in tool.MASK_IDS for row in loaded)
    assert all("task_mask" not in row["prompt"] for row in loaded)
    assert all(
        tool._sha(cohort["out"] / relative) == value
        for relative, value in projected["files_sha256"].items()
    )


@pytest.mark.parametrize("field", ["parent_manifest", "proposal"])
def test_stale_authority_hash_fails_before_output(cohort, field):
    cohort[field].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        tool.project_tasks(**cohort)
    assert not cohort["out"].exists()


@pytest.mark.parametrize(
    "kind", ["duplicate", "validation", "wrong_purpose", "stale_png", "definition"]
)
def test_invalid_task_decisions_fail_closed(cohort, monkeypatch, kind):
    decision = tool._read(cohort["proposal"])
    if kind == "duplicate":
        decision["records"][1] = deepcopy(decision["records"][0])
    elif kind == "validation":
        decision["records"][0]["split"] = "validation"
    elif kind == "wrong_purpose":
        decision["records"][0]["canonical_purpose_unchanged"] = "body"
    elif kind == "stale_png":
        decision["records"][0]["pixel_and_source_refs"]["context_png"]["sha256"] = "0" * 64
    else:
        decision["role_definition"]["uncertainty_policy"] = "wrong"
    tool._write(cohort["proposal"], decision)
    monkeypatch.setattr(tool, "PROPOSAL_SHA", tool._sha(cohort["proposal"]))
    with pytest.raises(ValueError):
        tool.project_tasks(**cohort)
    assert not cohort["out"].exists()


def test_known_id_that_is_validation_cannot_be_task_masked(cohort, monkeypatch):
    decision = tool._read(cohort["proposal"])
    decision["records"][0]["record_id"] = "validation-scene-0"
    decision["task_mask_record_ids"][0] = "validation-scene-0"
    tool._write(cohort["proposal"], decision)
    monkeypatch.setattr(tool, "PROPOSAL_SHA", tool._sha(cohort["proposal"]))
    monkeypatch.setattr(tool, "MASK_IDS", frozenset(decision["task_mask_record_ids"]))
    with pytest.raises(ValueError, match="never validation"):
        tool.project_tasks(**cohort)


def test_existing_output_and_parent_overlap_refuse(cohort):
    with pytest.raises(ValueError, match="fresh and separate"):
        tool.project_tasks(**{**cohort, "out": cohort["parent_manifest"].parent / "child"})
    cohort["out"].mkdir()
    with pytest.raises(ValueError, match="fresh and separate"):
        tool.project_tasks(**cohort)


def test_copy_mutation_never_publishes_ready(cohort, monkeypatch):
    copy = tool.shutil.copyfile

    def corrupt(source, destination):
        result = copy(source, destination)
        if source.name == "train-scene.png":
            destination.write_bytes(b"bad copy")
        return result

    monkeypatch.setattr(tool.shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="changed while copying"):
        tool.project_tasks(**cohort)
    assert (cohort["out"] / "INCOMPLETE.json").exists()
    assert not (cohort["out"] / "READY.json").exists()
