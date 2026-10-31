"""Optional synthetic blank-view abstention diagnostics, separate from source data.

These controls are not original-source observations or independent scenes. An
experiment may explicitly mix their synthetic abstention rule into training;
held-out parent families remain held out. No renderer or model is invoked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from engine.visual_analysis import analysis_json_schema, bind_analysis
from tools import semantic_training_data as data

PROTOCOL = "semantic-analysis-blank-controls-v1"
BACKGROUND = [255, 255, 255]


def _snapshot(root):
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths or len(paths) > 5000:
        raise ValueError("Parent dataset exceeds its bounded file inventory")
    if any(not path.resolve().is_relative_to(root) for path in paths):
        raise ValueError("Parent dataset files must remain inside the dataset root")
    if sum(path.stat().st_size for path in paths) > data.MAX_TOTAL_BYTES:
        raise ValueError("Parent dataset exceeds its total byte budget")
    return {path.relative_to(root).as_posix(): data._sha(path) for path in paths}


def build_controls(*, dataset, out):
    """Write fresh, explicitly synthetic controls after verifying parent identity."""
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Control output must be a fresh directory")
    if out.is_relative_to(dataset) or dataset.is_relative_to(out):
        raise ValueError("Control output must not overlap its immutable parent dataset")
    before = _snapshot(dataset)
    manifest = data._read(dataset / "manifest.json")
    if (
        manifest.get("protocol") != data.PROTOCOL
        or manifest.get("status") != "complete_reviewed_annotations"
        or manifest.get("human_reviewed") is not False
        or manifest.get("quality_promotion") is not False
        or (dataset / "INCOMPLETE.json").exists()
    ):
        raise ValueError("Controls require a complete, unpromoted reviewed parent dataset")
    declared = manifest.get("files_sha256")
    if not isinstance(declared, dict) or declared != {
        name: value for name, value in before.items() if name != "manifest.json"
    }:
        raise ValueError("Parent dataset files do not match their declared receipts")
    path = dataset / "analysis.jsonl"
    if path.stat().st_size > data.MAX_JSON_BYTES:
        raise ValueError("Parent analysis records exceed their byte budget")
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        data._canonical(row)
        if row.get("variant") == "original_ids":
            records.append(row)
    if len(records) != 8:
        raise ValueError("Controls require exactly eight original-ID parent rows")
    sources = {row["receipt"]: row for row in manifest["sources"]}
    seen, prepared, total_pixels = set(), [], 0
    for row in records:
        receipt_path = data._within(dataset, row["receipt"])
        receipt = data._read(receipt_path)
        identity = receipt.get("source_id")
        if identity not in data.ALLOWED_SOURCES or identity in seen:
            raise ValueError("Control parents must be unique reviewed sources")
        seen.add(identity)
        parent = sources.get(row["receipt"], {})
        if (
            parent.get("source_id") != identity
            or parent.get("split") != row.get("split")
            or receipt.get("split") != row.get("split")
            or row.get("split") not in {"train", "validation"}
            or type(row.get("training_eligible")) is not bool
            or row["training_eligible"] != (row["split"] == "train")
        ):
            raise ValueError("Parent source, split or eligibility differs from its receipt")
        image_path = data._within(dataset, row["input"]["source_image"])
        original_sha = data._sha(image_path)
        if (
            original_sha != receipt["source_render_sha256"]
            or original_sha != row["input"]["source_render_sha256"]
        ):
            raise ValueError("Parent image does not match its exact source-render receipt")
        with Image.open(image_path) as image:
            dimensions = image.size
            if (
                image.format != "PNG"
                or dimensions != tuple(receipt["source_canvas"][key] for key in ("width", "height"))
                or image.width * image.height > data.MAX_PIXELS
            ):
                raise ValueError("Parent image must be a bounded full source-canvas PNG")
            image.verify()
            total_pixels += image.width * image.height
        if total_pixels > data.MAX_TOTAL_PIXELS:
            raise ValueError("Controls exceed their total source-pixel budget")
        ids = [element["id"] for element in row["input"]["source_summary"]["elements"]]
        binding = bind_analysis(
            row["target"],
            source_digest=receipt["analysis_source_digest"],
            source_render_sha256=original_sha,
            known_element_ids=ids,
        )
        if binding != row["validated_binding"] or row["input"][
            "response_schema"
        ] != analysis_json_schema(ids):
            raise ValueError("Parent analysis schema or source binding is stale")
        prepared.append((row, receipt, dimensions, ids))
    out.mkdir(parents=True)
    controls, receipts = [], []
    try:
        for row, receipt, dimensions, ids in prepared:
            record_id = data._digest([PROTOCOL, row["record_id"]])
            image_relative = f"blank-views/{record_id[:20]}.png"
            (out / "blank-views").mkdir(exist_ok=True)
            Image.new("RGB", dimensions, tuple(BACKGROUND)).save(out / image_relative, format="PNG")
            blank_sha = data._sha(out / image_relative)
            raw = {"observations": [], "abstain": True}
            bound = bind_analysis(
                raw,
                source_digest=receipt["analysis_source_digest"],
                source_render_sha256=blank_sha,
                known_element_ids=ids,
            )
            receipt_relative = f"receipts/{record_id[:20]}.json"
            common = {
                "diagnostic_only": True,
                "view_kind": "synthetic_blank_view",
                "source_render_parity": False,
                "supervision_origin": "synthetic_abstention_rule",
                "human_reviewed": False,
                "experimental_only": True,
                "split": row["split"],
                "training_eligible": row["split"] == "train" and row["training_eligible"],
            }
            controls.append(
                {
                    **common,
                    "record_id": record_id,
                    "receipt": receipt_relative,
                    "variant": "synthetic_blank_view",
                    "input": {
                        **row["input"],
                        "source_image": image_relative,
                        "source_render_sha256": blank_sha,
                    },
                    "target": raw,
                    "validated_binding": bound,
                }
            )
            control_receipt = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                **common,
                "source_id": receipt["source_id"],
                "family": receipt["family"],
                "parent_record_id": row["record_id"],
                "parent_receipt": row["receipt"],
                "parent_manifest_sha256": before["manifest.json"],
                "original_source_render_sha256": receipt["source_render_sha256"],
                "blank_render_sha256": blank_sha,
                "source_canvas": receipt["source_canvas"],
                "source_digest": receipt["analysis_source_digest"],
                "source_digest_protocol": receipt["analysis_source_digest_protocol"],
                "blank_rgb": BACKGROUND,
                "rule": "Uniform synthetic view supplies no visible design relationship; abstain.",
                "limitation": "Metadata still describes the parent scene; these are diagnostic "
                "mismatched-view controls, not real-source quality data.",
            }
            data._write(out / receipt_relative, control_receipt)
            receipts.append(
                {
                    "source_id": receipt["source_id"],
                    "family": receipt["family"],
                    "split": row["split"],
                    "receipt": receipt_relative,
                    "sha256": data._sha(out / receipt_relative),
                }
            )
        data._jsonl(out / "controls.jsonl", controls)
        after = _snapshot(dataset)
        if before != after:
            raise ValueError("Immutable parent dataset changed while creating controls")
        result = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "ready_optional_diagnostics",
            "diagnostic_only": True,
            "source_render_parity": False,
            "supervision_origin": "synthetic_abstention_rule",
            "human_reviewed": False,
            "quality_promotion": False,
            "model_executed": False,
            "native_renderer_executed": False,
            "parent_dataset": str(dataset),
            "parent_manifest_sha256": before["manifest.json"],
            "parent_files_sha256_before": before,
            "parent_files_sha256_after": after,
            "parent_immutability_verified": True,
            "sources": receipts,
            "counts": {
                "controls": len(controls),
                "train": sum(row["training_eligible"] for row in controls),
                "validation": sum(row["split"] == "validation" for row in controls),
                "new_independent_scenes": 0,
            },
            "blank_rgb": BACKGROUND,
            "code_sha256": {"tools/semantic_analysis_controls.py": data._sha(Path(__file__))},
            "limitations": [
                "Optional synthetic abstention experiment; mix only by explicit training policy.",
                "Blank views are not the original pixels and do not establish "
                "real-source visual quality.",
                "Eight controls reuse eight parent source groups, adding zero independent scenes.",
                "Validation controls must never train or calibrate a model.",
            ],
        }
        result["files_sha256"] = {
            path.relative_to(out).as_posix(): data._sha(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        }
        data._write(out / "manifest.json", result)
        data._write(
            out / "READY.json",
            {
                "protocol": PROTOCOL,
                "manifest_sha256": data._sha(out / "manifest.json"),
                "diagnostic_only": True,
                "optional_training_policy_required": True,
            },
        )
        return result
    except BaseException:
        data._write(
            out / "INCOMPLETE.json",
            {"training_eligible": False, "reason": "Control build failed; choose a fresh output."},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_controls(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Controls not admitted: {exc}\n")
    print(
        json.dumps(
            {"status": result["status"], "counts": result["counts"], "out": str(args.out)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
