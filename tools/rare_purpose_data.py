"""Project one frozen, weak, authored train cohort without widening public admission.

The three model views are source, native context, and diagnostic isolated paint.
Intent/placement/relations remain sidecars; no label controls paint or cropping.
Run the CLI in one CPU worker. It never imports a model or invokes design reflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from engine.design_semantics import semantic_source_digest, validate_semantic_annotations
from experiments.vision_critic.contracts import canonical_json_sha256, safe_artifact_path
from tools.semantic_training_data import (
    MAX_CONTEXT_SIDE,
    PROTOCOL,
    _context_box,
    _digest,
    _jsonl,
    _read,
    _resource_receipts,
    _sha,
    _write,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SHA = "7d6bdc8a83eda55a76577b11da1eb091fefa2c67528596e761964daebd6c12fe"
AUDIT_SHA = "9ab699bc134b6fc1b77f9f5bcaf7d193be9568f738554ead719dafbf31f511f9"
HELPER_SHA = "02f4f6ed096dd13ed99c0c9f27e22206d2788dd070501cec22747b485ad527d1"
ISOLATED_PROTOCOL = "semantic-purpose-isolated-layer-evidence-v1"
ISOLATION_V1 = "native-element-isolation-v1"
ISOLATION_V2_24MP = "native-element-isolation-v2-24mp-source"
ISOLATION_POLICIES = (ISOLATION_V1, ISOLATION_V2_24MP)
SOURCE_FAMILIES = {
    "apple-contained-feature": "cc0-ftkelle-apple-red",
    "apple-fullbleed-feature": "cc0-ftkelle-apple-red",
    "apple-atmosphere-event": "cc0-ftkelle-apple-red",
    "moon-contained-feature": "cc0-rakeshk9548-moon-mountain",
    "moon-fullbleed-feature": "cc0-rakeshk9548-moon-mountain",
    "moon-atmosphere-event": "cc0-rakeshk9548-moon-mountain",
    "cards-contained-workshop": "cc0-kenney-playing-cards-pack",
    "cards-joker-evening": "cc0-kenney-playing-cards-pack",
}
TASK_PURPOSES = frozenset(
    {
        "background",
        "brand_mark",
        "headline",
        "decoration",
        "supporting_copy",
        "body",
        "footer_destination",
        "primary_visual",
        "action",
        "unknown",
    }
)
CODE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "tools/native_element_evidence.py",
    ROOT / "tools/semantic_training_data.py",
    ROOT / "engine/design_semantics.py",
    ROOT / "experiments/vision_critic/contracts.py",
)
LIMITATIONS = [
    "Agent-authored intent plus agent visual review is weak supervision, not human gold.",
    "Eight scenes share three original asset/work families; no independent validation is added.",
    "Identical source pixels can support different authored intent; purpose is not pixel truth.",
    "The fixed light checker can reduce white-glyph contrast, especially after downsampling.",
    "Isolated paint diagnoses layer appearance, not its role or original compositional intent.",
    "No training, model execution, resize quality, or generalization improvement is claimed.",
]


def _code_hashes():
    return {path.relative_to(ROOT).as_posix(): _sha(path) for path in CODE_PATHS}


def _linked(root, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Expected a closed path/sha256 source-package reference")
    path = safe_artifact_path(root, reference["path"])
    if _sha(path) != reference["sha256"]:
        raise ValueError("Frozen source-package linked artifact changed")
    return path


def _task_mask(label):
    """Keep raw review identity distinct from the frozen task's taxonomy mask."""
    if label.get("training_eligible") is not True or label.get("observable") != "clear":
        raise ValueError("This frozen cohort requires explicit clear authored review labels")
    purpose = label.get("purpose")
    if purpose == "footer_legal":
        return False, "unsupported_task_taxonomy:footer_legal"
    if purpose not in TASK_PURPOSES:
        raise ValueError("Unexpected unsupported purpose; do not silently relabel or mask it")
    return True, None


def _validate_scene(entry, source, annotation, native_size):
    if (
        entry.get("source_id") not in SOURCE_FAMILIES
        or entry.get("family_id") != SOURCE_FAMILIES[entry["source_id"]]
        or entry.get("split") != "train"
    ):
        raise ValueError("Only the frozen eight scenes in their three TRAIN families are admitted")
    if (
        annotation.get("source_id") != entry["source_id"]
        or annotation.get("human_reviewed") is not False
        or annotation.get("source_render_sha256") != entry["native_png"]["sha256"]
        or annotation.get("source_digest") != semantic_source_digest(source)
        or list(native_size) != [source["canvas"][key] for key in ("width", "height")]
    ):
        raise ValueError("Authored annotation/native source identity differs")
    proposal = {
        "schema_version": 1,
        "source_digest_protocol": annotation["source_digest_protocol"],
        "source_digest": annotation["source_digest"],
        "elements": [
            {key: row[key] for key in ("element_id", "purpose", "confidence", "evidence")}
            for row in annotation["elements"]
        ],
        "relations": [
            {key: row[key] for key in ("source_id", "target_id", "type", "confidence", "evidence")}
            for row in annotation["relations"]
        ],
    }
    validate_semantic_annotations(source, proposal)
    for label in annotation["elements"]:
        if label.get("label_origin") != "explicit_authored_synthetic_intent":
            raise ValueError("Source-authored labels must retain their actual weak provenance")
        _task_mask(label)


def _admit_package(package, audit_path):
    """Exact audited package authority, distinct from the public eight-source gate."""
    if _sha(audit_path) != AUDIT_SHA or _sha(package / "manifest.json") != PACKAGE_SHA:
        raise ValueError("This adapter only admits the exact accepted rare-purpose package")
    audit, manifest = _read(audit_path), _read(package / "manifest.json")
    if audit.get("blockers") != [] or audit.get("source_health_approved") is not True:
        raise ValueError("Independent source-package audit did not approve source health")
    inventory = audit["v2_file_sha256"]
    if len(inventory) != 147:
        raise ValueError("Unexpected accepted package file count")
    if sum(safe_artifact_path(package, rel).stat().st_size for rel in inventory) > 64 * 1024**2:
        raise ValueError("Accepted source package exceeds its byte budget")
    for relative, expected in inventory.items():
        if _sha(safe_artifact_path(package, relative)) != expected:
            raise ValueError(f"Accepted source package changed: {relative}")
    if manifest.get("training_only") is not True or manifest.get("human_gold_labels") is not False:
        raise ValueError("Source package must remain weak TRAIN-only evidence")
    freeze = _read(_linked(package, manifest["family_split_freeze"]))
    if freeze.get("all_new_sources_split") != "train":
        raise ValueError("The original family split must be frozen before variants")
    reviews = {}
    for filename in ("visual-review-root.json", "visual-review-peer.json"):
        review = _read(safe_artifact_path(package, filename))
        for row in review["source_rows"]:
            if row["source_id"] in reviews:
                raise ValueError("Duplicate visual review source")
            reviews[row["source_id"]] = {"reviewer": review["reviewer"], **row}
    entries = manifest["source_rows"]
    if len(entries) != 8 or {entry["source_id"] for entry in entries} != set(SOURCE_FAMILIES):
        raise ValueError("Frozen scene membership differs")
    admitted, elements, relations, masked = [], 0, 0, 0
    for entry in entries:
        source = _read(_linked(package, entry["source_json"]))
        annotation = _read(_linked(package, entry["annotation"]))
        native = _linked(package, entry["native_png"])
        render = _read(_linked(package, entry["render_receipt"]))
        rights = _read(_linked(package, entry["rights_receipt"]))
        _linked(package, entry["resource_receipt"])
        review = reviews[entry["source_id"]]
        if (
            review["native_png_sha256"] != entry["native_png"]["sha256"]
            or review["annotation_sha256"] != entry["annotation"]["sha256"]
            or review.get("required_annotation_masks") != []
            or review.get("review_origin") != "agent_visual_review"
            or render.get("source_json_sha256") != entry["source_json"]["sha256"]
            or render.get("repeatability_pass") is not True
            or rights.get("image_license") != "CC0-1.0"
            or rights.get("font_license") != "OFL-1.1"
        ):
            raise ValueError("Source review, native receipt or exact license binding differs")
        with Image.open(native) as image:
            image.verify()
            _validate_scene(entry, source, annotation, image.size)
        resources = _resource_receipts(source, package)
        elements += len(source["elements"])
        relations += len(annotation["relations"])
        masked += sum(not _task_mask(row)[0] for row in annotation["elements"])
        admitted.append((entry, source, annotation, review, resources))
    if (elements, relations, masked) != (82, 46, 8):
        raise ValueError("Expected all 82 raw labels, 46 relations and exactly eight legal masks")
    return inventory, admitted


def _isolate(element, assets_dir, fonts_dir, policy):
    from tools.native_element_evidence import render_isolated_element

    return render_isolated_element(
        element, assets_dir=assets_dir, fonts_dir=fonts_dir, policy=policy
    )


def _save_isolation(out, relative, result, metadata):
    receipt = {**deepcopy(result["receipt"]), **metadata}
    if "receipt_digest" in receipt:
        receipt["native_helper_receipt_digest"] = receipt.pop("receipt_digest")
    if result["status"] != "available" and str(result.get("reason", "")).split(":")[0] in {
        "OSError",
        "IOError",
        "FileNotFoundError",
        "PermissionError",
        "IsADirectoryError",
        "NotADirectoryError",
    }:
        raise OSError("Native isolation I/O failure must abort export: " + result["reason"])
    receipt.update(
        label_used_for_paint_choices=False,
        no_replacement_asset_or_content=True,
        status=result["status"],
    )
    reference = None
    if result["status"] == "available":
        if receipt.get("clipped") is not False:
            raise ValueError("Available isolated evidence must explicitly be unclipped")
        for name, key in (("alpha", "alpha_png"), ("checker", "checker_png")):
            path = out / "isolated" / name / f"{relative}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(result[key])
            receipt[f"{name}_image"] = {
                "path": path.relative_to(out).as_posix(),
                "sha256": _sha(path),
            }
        reference = receipt["checker_image"]
    receipt_path = out / "isolated/receipts" / f"{relative}.json"
    receipt["projection_receipt_digest"] = canonical_json_sha256(receipt)
    _write(receipt_path, receipt)
    return reference, {
        "available": result["status"] == "available",
        "status": result["status"],
        "reason": result.get("reason"),
        "receipt": receipt_path.relative_to(out).as_posix(),
        "receipt_sha256": _sha(receipt_path),
    }


def export_rare_purpose(package, audit_path, out, *, isolation_policy=ISOLATION_V1):
    """Write fresh portable weak TRAIN evidence; failed writes never produce READY."""
    package, audit_path, out = map(lambda p: Path(p).resolve(), (package, audit_path, out))
    if isolation_policy not in ISOLATION_POLICIES:
        raise ValueError("An explicitly supported native isolation policy is required")
    if out.exists() or out.is_relative_to(package) or package.is_relative_to(out):
        raise ValueError("Output must be fresh and independent of the frozen source package")
    if _sha(ROOT / "tools/native_element_evidence.py") != HELPER_SHA:
        raise ValueError("Native isolation helper differs from the accepted policy implementation")
    code_before = _code_hashes()
    inventory, admitted = _admit_package(package, audit_path)
    digests = [semantic_source_digest(row[1]) for row in admitted]
    if len(set(digests)) != len(digests):
        raise ValueError("Distinct authored sources must not alias the same source identity")
    out.mkdir(parents=True)
    for relative, expected in inventory.items():
        copied = out / "source-package" / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_artifact_path(package, relative), copied)
        if _sha(copied) != expected:
            raise ValueError("Frozen source copy changed")
    shutil.copyfile(audit_path, out / "source-package-independent-audit.json")
    rows, source_rows = [], []
    for entry, source, annotation, review, resources in admitted:
        sid, digest = entry["source_id"], semantic_source_digest(source)
        prefix, receipt_rel = digest[:20], f"receipts/{digest[:20]}.json"
        image_rel = f"images/{prefix}.png"
        (out / "images").mkdir(exist_ok=True)
        shutil.copyfile(package / entry["native_png"]["path"], out / image_rel)
        labels = {row["element_id"]: row for row in annotation["elements"]}
        with Image.open(out / image_rel) as opened:
            native = opened.copy()
        width, height = native.size
        source_rows.append(
            {
                "source_id": sid,
                "family_id": entry["family_id"],
                "split": "train",
                "receipt": receipt_rel,
                "source_json": "source-package/" + entry["source_json"]["path"],
                "element_count": len(source["elements"]),
            }
        )
        _write(
            out / receipt_rel,
            {
                "schema_version": 1,
                "protocol": "rare-purpose-projection-source-receipt-v1",
                "source_id": sid,
                "family_id": entry["family_id"],
                "split": "train",
                "source_json_sha256": entry["source_json"]["sha256"],
                "source_digest_protocol": "canonical-compact-v1",
                "source_digest": digest,
                "source_render_sha256": entry["native_png"]["sha256"],
                "resources": resources,
                "visual_review": review,
                "raw_annotation": "source-package/" + entry["annotation"]["path"],
                "placements_and_relations_are_model_inputs": False,
            },
        )
        for index, element in enumerate(source["elements"]):
            label = labels[element["id"]]
            eligible, mask_reason = _task_mask(label)
            relative = f"{prefix}/{index:03d}"
            context_path = out / "context-crops" / f"{relative}.png"
            context_path.parent.mkdir(parents=True, exist_ok=True)
            box = _context_box(element, native.size)
            crop = native.crop(box)
            crop.thumbnail((MAX_CONTEXT_SIDE, MAX_CONTEXT_SIDE), Image.Resampling.LANCZOS)
            crop.save(context_path, format="PNG")
            isolated, evidence = (
                None,
                {
                    "available": False,
                    "status": "excluded",
                    "reason": mask_reason,
                },
            )
            if eligible:
                result = _isolate(
                    deepcopy(element),
                    package / entry["assets_dir"],
                    package / entry["fonts_dir"],
                    isolation_policy,
                )
                isolated, evidence = _save_isolation(
                    out,
                    relative,
                    result,
                    {
                        "source_id": sid,
                        "family_id": entry["family_id"],
                        "source_element": deepcopy(element),
                        "source_json_sha256": entry["source_json"]["sha256"],
                        "source_render_sha256": entry["native_png"]["sha256"],
                        "source_element_canonical_sha256": canonical_json_sha256(element),
                        "resource_hashes": resources,
                        "producer_implementation_hashes": code_before,
                    },
                )
            rows.append(
                {
                    "record_id": _digest([digest, "element", index]),
                    "family_id": entry["family_id"],
                    "split": "train",
                    "training_eligible": eligible,
                    "review_mask": eligible,
                    "raw_review_mask": True,
                    "task_mask_reason": mask_reason,
                    "experimental_only": True,
                    "supervision_origin": "agent_visual_review",
                    "label_origin": "explicit_authored_synthetic_intent",
                    "receipt": receipt_rel,
                    "input": {
                        "source_image": image_rel,
                        "context_image": context_path.relative_to(out).as_posix(),
                        "isolated_image": isolated,
                        "bbox_normalized": [
                            element["x"] / width,
                            element["y"] / height,
                            element["width"] / width,
                            element["height"] / height,
                        ],
                        "type": element["type"],
                        "text_data": element.get("content", ""),
                        "text_is_untrusted_data": True,
                    },
                    "target": {"purpose": label["purpose"], "confidence": label["confidence"]},
                    "isolated_evidence": evidence,
                    "provenance": {
                        "element_index": index,
                        "reviewed": True,
                        "uncertain": False,
                        "visible_in_source_frame": True,
                        "crop_kind": "native_render_context_not_isolated_asset",
                        "context_box_pixels": box,
                        "context_size": list(crop.size),
                        "context_sha256": _sha(context_path),
                    },
                }
            )
        native.close()
    _jsonl(out / "elements.jsonl", rows)
    # Revalidate original bytes and every linked resource/code after the serial native worker.
    if _admit_package(package, audit_path)[0] != inventory or _code_hashes() != code_before:
        raise ValueError("Original package or producer changed during export")
    files = {
        path.relative_to(out).as_posix(): _sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    counts = {
        "sources": len(source_rows),
        "train_families": len({r["family_id"] for r in rows}),
        "raw_labels": len(rows),
        "raw_reviewed": sum(r["raw_review_mask"] for r in rows),
        "task_eligible_train": sum(r["training_eligible"] for r in rows),
        "unsupported_task_masked": sum(not r["review_mask"] for r in rows),
        "isolated_available": sum(r["isolated_evidence"]["available"] for r in rows),
        "eligible_unavailable": sum(
            r["training_eligible"] and not r["isolated_evidence"]["available"] for r in rows
        ),
        "raw_relations": sum(len(a[2]["relations"]) for a in admitted),
        "validation": 0,
        "test": 0,
    }
    parent = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "human_reviewed": False,
        "admission": "RARE_AUTHORED_INTENT_AGENT_REVIEW_WEAK_TRAIN_ONLY",
        "source_scope": "separate_frozen_three_family_rare_purpose_cohort",
        "source_package_sha256": PACKAGE_SHA,
        "independent_audit_sha256": AUDIT_SHA,
        "sources": source_rows,
        "counts": counts,
        "files_sha256": files,
        "task_purposes": sorted(TASK_PURPOSES),
        "unsupported_task_purposes": ["footer_legal"],
        "implementation_hashes": code_before,
        "implementation_after_hashes": _code_hashes(),
        "limitations": LIMITATIONS,
        "blank_or_mismatch_controls_included": False,
        "isolation_policy": isolation_policy,
    }
    _write(out / "parent-manifest.json", parent)
    manifest = {
        **parent,
        "protocol": ISOLATED_PROTOCOL,
        "parent_manifest_sha256": _sha(out / "parent-manifest.json"),
        "files_sha256": {**files, "parent-manifest.json": _sha(out / "parent-manifest.json")},
        "ordered_views": ["source", "context", "isolated_target_fixed_checker"],
        "isolation_helper_sha256": HELPER_SHA,
    }
    _write(out / "manifest.json", manifest)
    _write(
        out / "READY.json",
        {
            "status": "complete"
            if counts["eligible_unavailable"] == 0
            else "review_only_unavailable",
            "protocol": "rare-purpose-projection-ready-v1",
            "counts": counts,
            "manifest": {"path": "manifest.json", "sha256": _sha(out / "manifest.json")},
            "parent_manifest": {
                "path": "parent-manifest.json",
                "sha256": _sha(out / "parent-manifest.json"),
            },
            "human_gold": False,
            "model_executed": False,
            "quality_approved": False,
            "created_utc": datetime.now(UTC).isoformat(),
        },
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--isolation-policy", choices=ISOLATION_POLICIES, default=ISOLATION_V1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        result = export_rare_purpose(
            args.package, args.audit, args.out, isolation_policy=args.isolation_policy
        )
        print(json.dumps(result["counts"]))
        return
    if args.out.exists():
        raise ValueError("Fresh immutable output required")
    log = args.out.parent / (args.out.name + "-worker.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tools.rare_purpose_data",
        "--worker",
        "--package",
        str(args.package),
        "--audit",
        str(args.audit),
        "--out",
        str(args.out),
        "--isolation-policy",
        args.isolation_policy,
    ]
    with log.open("xb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"], check=False)
            else:
                process.kill()
            process.wait(timeout=30)
            raise TimeoutError("Bounded rare-purpose export exceeded 600 seconds") from None
    if code:
        raise RuntimeError(f"Rare-purpose worker failed; inspect {log}")
    print(json.dumps({"out": str(args.out.resolve()), "log": str(log.resolve())}))


if __name__ == "__main__":
    main()
