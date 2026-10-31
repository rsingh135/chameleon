"""Copy one frozen purpose cohort and add explicit TRAIN-only task masks.

Canonical labels, raw review flags, source pixels and isolated evidence never
change. A task mask is not an unknown label or a finding of a wrong annotation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

from experiments.vision_critic.contracts import parse_json, safe_artifact_path, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "001292824831719997a0f8c8ac7abf0cfedd4f00b38044efcc1f04473418fa61"
PROPOSAL_SHA = "eb33d3b00b1bc319bce9470512791019a7eb05128a162723b50cb83a96e4b352"
DEFINITION_SHA = "11582f221d2d9527ad3c8b0245a0728b236a00697badac680e7eff5109413b83"
PROTOCOL = "purpose-task-mask-projection-v1"
EXPECTED_FILES = 1512
MAX_FILE_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2
MASK_IDS = frozenset(
    {
        "9cf8ed03c4a502fbe0d5c94e4d4d803a516bd32153bf0e58b0dac090ecc2528c",
        "094e5cdd83cedbd205614433921b2a648bf81c4acd5a68b26c83cd7440399d94",
        "540b0839319164760f788908d96340edca5085555f249e2dae0fb99769ee09d2",
        "9b0ab39a8786e26b0de00aa6d74ea1ac7754b7adff13e320e15b0b41f1f3488b",
        "3f76f5cc0cd83791eb2c7bf33da3c7d013e75ce09dd6c7c834048b74ad0a3ca4",
        "982aa5cf6cd6417d9825ccc1556f68e6ac04bca5dfd398af6fd2b9d255488355",
        "06288334135eeb16d466662979b6ba7c6957e4f898897bf216def19c2bfb6be2",
        "b6a8c79299f229010c1739c09571044051b77601bc96ade85bd4b09b901cd817",
    }
)
MUTABLE_JSONL = frozenset(
    {"elements.jsonl", "train-elements.jsonl", "validation-elements.jsonl", "masked-elements.jsonl"}
)
TASK_FIELDS = frozenset(
    {"task_mask", "task_review_mask", "task_mask_reason", "task_projection_receipt"}
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    import hashlib

    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Projection input is missing or exceeds its byte budget")
    return sha256_file(path)


def _read(path):
    _sha(path)
    return parse_json(Path(path).read_text(encoding="utf-8-sig"))


def _rows(path):
    _sha(path)
    result = [
        parse_json(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if any(
        not isinstance(row, dict) or not isinstance(row.get("record_id"), str) for row in result
    ):
        raise ValueError("Projection rows need explicit record identities")
    if len({row["record_id"] for row in result}) != len(result):
        raise ValueError("Duplicate original purpose record identity")
    return result


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _jsonl(path, rows):
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _proposal(path):
    if _sha(path) != PROPOSAL_SHA:
        raise ValueError("Task projection proposal hash differs from the accepted proposal")
    proposal = _read(path)
    if (
        proposal.get("protocol") != "supporting-copy-task-projection-v1"
        or type(proposal.get("schema_version")) is not int
        or proposal["schema_version"] != 1
    ):
        raise ValueError("Unsupported support task projection proposal")
    if (
        _digest(proposal["role_definition"]) != DEFINITION_SHA
        or proposal.get("role_definition_sha256") != DEFINITION_SHA
    ):
        raise ValueError("Support task definition hash differs")
    for key in (
        "canonical_labels_changed",
        "source_content_or_pixels_changed",
        "existing_exports_or_annotations_edited",
        "old_validation_or_sealed_gold_edited",
        "sealed_gold_inspected",
        "human_gold",
        "quality_approved",
    ):
        if proposal.get(key) is not False:
            raise ValueError("Task projection must preserve canonical data and sealed evaluation")
    records = proposal.get("records")
    if not isinstance(records, list) or len(records) != 14:
        raise ValueError("Task proposal must bind the fourteen original support records")
    ids = [row["record_id"] for row in records]
    if len(set(ids)) != 14:
        raise ValueError("Duplicate proposed support record identity")
    masked, retained = proposal["task_mask_record_ids"], proposal["task_retain_record_ids"]
    if (
        len(masked) != 8
        or len(set(masked)) != 8
        or set(masked) != MASK_IDS
        or len(retained) != 6
        or len(set(retained)) != 6
        or set(masked) & set(retained)
        or set(masked) | set(retained) != set(ids)
    ):
        raise ValueError("Task proposal changes the exact accepted mask/retain membership")
    return proposal


def _bind_review(root, item, row, files):
    if (
        item.get("split") != "train"
        or row.get("split") != "train"
        or row.get("review_mask") is not True
        or row.get("training_eligible") is not True
        or row["target"].get("purpose") != "supporting_copy"
        or item.get("canonical_purpose_unchanged") != "supporting_copy"
        or item.get("canonical_model_word_unchanged") != "support"
        or item.get("canonical_label_origin_unchanged") != row.get("supervision_origin")
        or item.get("wrong_canonical_label_established") is not False
        or item.get("automatic_relabel") is not False
        or item.get("human_reviewed") is not False
    ):
        raise ValueError(
            "Masks may only affect reviewed original TRAIN support, never validation or labels"
        )
    receipt = _read(safe_artifact_path(root, row["receipt"]))
    source_id = item["source_id"]
    if receipt["source_id"] != source_id or item["family_id"] != receipt.get(
        "family_id", receipt.get("family", source_id)
    ):
        raise ValueError("Task review source/family identity differs")
    source_relative = "sources/" + source_id + "/source.json"
    source = _read(safe_artifact_path(root, source_relative))
    index = row["provenance"]["element_index"]
    if (
        type(index) is not int
        or index != item["source_element_index"]
        or not 0 <= index < len(source["elements"])
    ):
        raise ValueError("Task review source element index differs")
    element = source["elements"][index]
    paint_order = sorted(
        range(len(source["elements"])), key=lambda i: source["elements"][i].get("z_index", 0)
    )
    if paint_order.index(index) != item["authored_paint_order_index"]:
        raise ValueError("Task review authored paint order differs")
    if (
        element["id"] != item["element_id"]
        or element.get("content", "") != item["copy_exact"]
        or row["input"].get("text_data", "") != item["copy_exact"]
    ):
        raise ValueError("Task review element identity or exact copy differs")
    isolation = _read(safe_artifact_path(root, row["isolated_evidence"]["receipt"]))
    if (
        isolation["record_id"] != row["record_id"]
        or isolation["source_element"] != element
        or isolation.get("status") != "available"
        or isolation.get("clipped") is not False
        or isolation.get("label_used_for_paint_choices") is not False
        or isolation.get("native_single_original_layer") is not True
        or isolation.get("no_replacement_asset_or_content") is not True
    ):
        raise ValueError("Task review does not bind available original native isolated evidence")
    paths = {
        "source_json": source_relative,
        "source_receipt": row["receipt"],
        "full_native_png": row["input"]["source_image"],
        "context_png": row["input"]["context_image"],
        "isolated_checker_png": row["input"]["isolated_image"]["path"],
        "isolated_alpha_png": isolation["isolated_alpha_image"]["path"],
        "isolated_receipt": row["isolated_evidence"]["receipt"],
    }
    for role, relative in paths.items():
        if files.get(relative) != item["pixel_and_source_refs"][role]["sha256"]:
            raise ValueError("Stale proposal role-specific source/PNG/receipt hash: " + role)
    if isolation["resource_hashes"] != receipt["resources"]:
        raise ValueError("Task review source resource identity differs")
    masked = row["record_id"] in MASK_IDS
    if item.get("task_decision") != (
        "mask_ambiguous" if masked else "retain_supporting_copy"
    ) or item.get("task_training_eligible") is not (not masked):
        raise ValueError("Proposed task decision differs from accepted mask membership")
    return paths


def project_tasks(*, parent_manifest, proposal, out):
    parent_manifest, proposal, out = (
        Path(path).resolve() for path in (parent_manifest, proposal, out)
    )
    root = parent_manifest.parent
    if (
        out.exists()
        or out.is_relative_to(root)
        or root.is_relative_to(out)
        or proposal.is_relative_to(out)
    ):
        raise ValueError("Projection output must be fresh and separate from immutable inputs")
    if _sha(parent_manifest) != PARENT_SHA:
        raise ValueError("Projection parent manifest hash differs")
    parent = _read(parent_manifest)
    if parent.get("protocol") != "semantic-purpose-isolated-layer-evidence-v1":
        raise ValueError("Projection requires the frozen original three-view purpose cohort")
    decision = _proposal(proposal)
    files = parent.get("files_sha256")
    if (
        not isinstance(files, dict)
        or len(files) != EXPECTED_FILES
        or not set(files) >= MUTABLE_JSONL
    ):
        raise ValueError("Parent purpose artifact inventory differs")
    snapshots = {str(parent_manifest): PARENT_SHA, str(proposal): PROPOSAL_SHA}
    for relative, expected in files.items():
        path = safe_artifact_path(root, relative)
        if _sha(path) != expected:
            raise ValueError("Frozen parent artifact changed: " + relative)
        snapshots[str(path)] = expected
    if sum(Path(path).stat().st_size for path in snapshots) > MAX_TOTAL_BYTES:
        raise ValueError("Purpose projection exceeds its total byte budget")
    original = _rows(root / "elements.jsonl")
    if len(original) != parent["record_count"] or any(TASK_FIELDS & set(row) for row in original):
        raise ValueError("Parent row count differs or is already task-projected")
    by_id = {row["record_id"]: row for row in original}
    bindings = []
    for item in decision["records"]:
        row = by_id.get(item["record_id"])
        if row is None:
            raise ValueError("Proposed support record is absent from the frozen cohort")
        bindings.append(
            {
                "record_id": item["record_id"],
                "paths": _bind_review(root, item, row, files),
                "task_decision": item["task_decision"],
                "wrong_label_established": False,
            }
        )
    out.mkdir(parents=True)
    try:
        preserved = {}
        for relative, expected in files.items():
            location = "raw-parent/" + relative if relative in MUTABLE_JSONL else relative
            target = safe_artifact_path(out, location)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_artifact_path(root, relative), target)
            if _sha(target) != expected:
                raise ValueError("Parent bytes changed while copying")
            preserved[relative] = {"path": location, "sha256": expected}
        raw_manifest = out / "raw-parent/manifest.json"
        shutil.copyfile(parent_manifest, raw_manifest)
        _write(out / "task-projection/role-definition.json", decision["role_definition"])
        proposal_copy = out / "task-projection/proposal.json"
        proposal_copy.parent.mkdir(exist_ok=True)
        shutil.copyfile(proposal, proposal_copy)
        receipt_relative = "task-projection/receipt.json"
        projected = []
        for row in original:
            value = deepcopy(row)
            keep = row["review_mask"] is True and row["record_id"] not in MASK_IDS
            value.update(
                task_mask=keep,
                task_review_mask=keep,
                task_mask_reason="ambiguous_support_task_only"
                if row["record_id"] in MASK_IDS
                else (None if keep else "existing_raw_review_mask_false_preserved"),
                task_projection_receipt=receipt_relative,
            )
            projected.append(value)
        train = [
            row
            for row in projected
            if row["split"] == "train" and row["training_eligible"] and row["task_mask"]
        ]
        validation = [
            row
            for row in projected
            if row["split"] == "validation" and row["review_mask"] and row["task_mask"]
        ]
        masked = [row for row in projected if not row["task_mask"]]
        for relative, rows in (
            ("elements.jsonl", projected),
            ("train-elements.jsonl", train),
            ("validation-elements.jsonl", validation),
            ("masked-elements.jsonl", masked),
        ):
            _jsonl(out / relative, rows)
        counts = {
            "raw_rows": len(original),
            "raw_reviewed": sum(row["review_mask"] is True for row in original),
            "existing_raw_masked": sum(row["review_mask"] is not True for row in original),
            "new_task_masks": len(MASK_IDS),
            "task_masked": len(masked),
            "task_eligible": len(train) + len(validation),
            "train": len(train),
            "validation": len(validation),
            "reviewed_support_unchanged": 14,
            "retained_support": 6,
        }
        receipt = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "parent_manifest_sha256": PARENT_SHA,
            "proposal_sha256": PROPOSAL_SHA,
            "role_definition_sha256": DEFINITION_SHA,
            "original_artifacts": preserved,
            "raw_parent_manifest": {"path": "raw-parent/manifest.json", "sha256": PARENT_SHA},
            "counts": counts,
            "bindings": bindings,
            "canonical_labels_changed": False,
            "raw_review_masks_changed": False,
            "wrong_label_established": False,
            "masked_to_unknown": False,
            "source_pixels_or_receipts_changed": False,
            "validation_masks_added": False,
            "projection_adjudication_origin": "agent_visual_review_weak_task_eligibility_only",
        }
        _write(out / receipt_relative, receipt)
        artifacts = {
            path.relative_to(out).as_posix(): _sha(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        }
        result = {
            **parent,
            "status": "complete_task_mask_projection",
            "files_sha256": artifacts,
            "task_projection": {
                "protocol": PROTOCOL,
                "receipt": {"path": receipt_relative, "sha256": _sha(out / receipt_relative)},
                "parent_manifest_sha256": PARENT_SHA,
                "proposal_sha256": PROPOSAL_SHA,
                "role_definition_sha256": DEFINITION_SHA,
            },
            "task_counts": counts,
            "quality_promotion": False,
        }
        _write(out / "manifest.json", result)
        from experiments.visual_finetune.data import purpose_rows

        loaded, _ = purpose_rows(out / "manifest.json", purpose_views=3)
        actual = Counter(row["split"] for row in loaded)
        if (
            len(loaded) != counts["task_eligible"]
            or actual != {"train": len(train), "validation": len(validation)}
            or set(row["record_id"] for row in loaded)
            != {row["record_id"] for row in train + validation}
        ):
            raise ValueError("Current classifier loader did not honor the exact task masks")
        if any(_sha(path) != value for path, value in snapshots.items()):
            raise ValueError("Immutable parent/proposal changed during projection")
        _write(
            out / "READY.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "ready_train_task_projection",
                "manifest_sha256": _sha(out / "manifest.json"),
                "counts": counts,
                "wrong_label_established": False,
                "quality_approved": False,
                "model_executed": False,
                "native_rendering_executed": False,
            },
        )
        return result
    except BaseException:
        _write(
            out / "INCOMPLETE.json",
            {
                "training_eligible": False,
                "reason": "Task projection failed; choose a fresh output.",
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = project_tasks(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Task projection not admitted: {exc}\n")
    print(json.dumps({"counts": result["task_counts"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
