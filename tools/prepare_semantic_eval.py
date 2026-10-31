"""Prepare label-free source-analysis probes from a frozen semantic export.

Only the manifest, frozen split, selected receipts, source JSON and source PNG
are opened. Annotation and training JSONL files remain sealed. The intermediate
collection contains original-source placeholders, never resized candidates or
hard-validity/quality claims. No model, native rendering or training is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.vision_critic.contracts import (
    canonical_json,
    canonical_json_sha256,
    parse_json,
    safe_artifact_path,
    validate_collection,
)

PROTOCOL = "semantic-analysis-training-data-v1"
SPLIT_PROTOCOL = "semantic-analysis-family-split-v1"
MAX_TOTAL_BYTES = 128 * 1024 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    return parse_json(data.decode("utf-8-sig"))


def _object_file(path, maximum):
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Semantic evaluation metadata is missing or oversized")
    data = path.read_bytes()
    if len(data) > maximum:
        raise ValueError("Semantic evaluation metadata exceeds its byte budget")
    value = _json(data)
    if not isinstance(value, dict):
        raise ValueError("Semantic evaluation metadata must be a JSON object")
    return data, value


def _verified_file(root, relative, expected, files, maximum):
    if files.get(relative) != expected or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Selected semantic source is not bound by the export manifest")
    path = safe_artifact_path(root, relative)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Selected semantic source is missing or oversized")
    data = path.read_bytes()
    if len(data) > maximum or _sha(data) != expected:
        raise ValueError("Selected semantic source bytes differ from the frozen manifest")
    return data


def _source_digests(source):
    compact = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False)
    spaced = json.dumps(source, sort_keys=True, allow_nan=False)
    return _sha(compact.encode()), _sha(spaced.encode())


def _selected_sources(dataset, *, split, max_sources):
    if (
        split not in {"train", "validation"}
        or type(max_sources) is not int
        or not 1 <= max_sources <= 20
    ):
        raise ValueError("Use train/validation and a source budget from 1 to 20")
    manifest_path = safe_artifact_path(dataset, "manifest.json")
    manifest_bytes, manifest = _object_file(manifest_path, 32 * 1024 * 1024)
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported semantic export protocol")
    files = manifest.get("files_sha256")
    frozen = manifest.get("frozen_split")
    if not isinstance(files, dict) or not isinstance(frozen, dict):
        raise ValueError("Semantic export requires frozen files and family splits")
    split_bytes = _verified_file(
        dataset, "frozen-splits.json", manifest.get("split_manifest_sha256"), files, 1024 * 1024
    )
    if _json(split_bytes) != frozen or frozen.get("protocol") != SPLIT_PROTOCOL:
        raise ValueError("Frozen semantic split differs from the export manifest")
    if frozen.get("split_unit") != "authored_source_work":
        raise ValueError("Semantic evaluation must retain the authored-source split unit")
    frozen_rows = frozen.get("sources")
    exported = manifest.get("sources")
    if not isinstance(frozen_rows, list) or not isinstance(exported, list):
        raise ValueError("Semantic export requires source lists")
    by_id, families = {}, {}
    excluded = set(frozen.get("excluded_source_ids", []))
    for row in frozen_rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key] for key in ("source_id", "family", "split")
        ):
            raise ValueError("Frozen source identity is invalid")
        source_id, family, assigned = (row[key] for key in ("source_id", "family", "split"))
        if source_id in by_id or source_id in excluded or assigned not in {"train", "validation"}:
            raise ValueError("Duplicate, excluded or unsupported frozen source")
        if family in families and families[family] != assigned:
            raise ValueError("Family leakage in the frozen semantic split")
        by_id[source_id], families[family] = row, assigned
    seen, selected = set(), []
    for row in exported:
        if not isinstance(row, dict) or not isinstance(row.get("source_id"), str):
            raise ValueError("Exported source identity is invalid")
        source_id = row["source_id"]
        if (
            source_id in seen
            or source_id not in by_id
            or row.get("split") != by_id[source_id]["split"]
        ):
            raise ValueError("Exported sources differ from the frozen split")
        seen.add(source_id)
        if row["split"] == split and len(selected) < max_sources:
            selected.append((row, by_id[source_id]))
    if seen != set(by_id) or not selected:
        raise ValueError("Frozen source coverage is incomplete or no selected sources exist")
    records, total, prefixes = [], 0, set()
    for exported_row, frozen_row in selected:
        receipt_relative = exported_row.get("receipt")
        if not isinstance(receipt_relative, str):
            raise ValueError("Selected source receipt path is missing")
        receipt_bytes = _verified_file(
            dataset, receipt_relative, exported_row.get("sha256"), files, 1024 * 1024
        )
        receipt = _json(receipt_bytes)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("protocol") != PROTOCOL
            or any(receipt.get(key) != frozen_row[key] for key in ("source_id", "family", "split"))
            or receipt.get("source_digest_protocol") != "canonical-compact-v1"
            or receipt.get("analysis_source_digest_protocol") != "sorted-spaced-json-v1"
        ):
            raise ValueError("Selected source receipt identity differs from its frozen split")
        compact = receipt.get("source_digest")
        if not isinstance(compact, str) or len(compact) != 64:
            raise ValueError("Selected source requires a full compact design digest")
        prefix = compact[:20]
        if prefix in prefixes:
            raise ValueError("Selected source digests must identify separate original works")
        prefixes.add(prefix)
        if receipt_relative != f"receipts/{prefix}.json":
            raise ValueError("Source receipt filename differs from the bound source digest")
        json_relative, png_relative = f"receipts/{prefix}.source.json", f"images/{prefix}.png"
        source_bytes = _verified_file(
            dataset, json_relative, receipt.get("source_json_sha256"), files, 4 * 1024 * 1024
        )
        png_bytes = _verified_file(
            dataset, png_relative, receipt.get("source_render_sha256"), files, 32_000_000
        )
        source = _json(source_bytes)
        if not isinstance(source, dict):
            raise ValueError("Selected source JSON must be an object")
        actual_compact, spaced = _source_digests(source)
        elements = source.get("elements")
        if (
            actual_compact != compact
            or spaced != receipt.get("analysis_source_digest")
            or source.get("canvas") != receipt.get("source_canvas")
            or not isinstance(elements, list)
            or not elements
            or any(
                not isinstance(element, dict) or not isinstance(element.get("id"), str)
                for element in elements
            )
            or [element["id"] for element in elements] != receipt.get("original_element_ids")
            or len({element["id"] for element in elements}) != len(elements)
            or len(elements) != exported_row.get("element_count")
        ):
            raise ValueError("Selected source JSON differs from its receipt identity")
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(png_bytes)) as image:
            if (
                image.format != "PNG"
                or image.width * image.height > 20_000_000
                or image.size != tuple(source["canvas"].get(key) for key in ("width", "height"))
            ):
                raise ValueError("Source PNG dimensions differ from its frozen canvas")
            image.verify()
        total += len(receipt_bytes) + len(source_bytes) + len(png_bytes)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Selected semantic sources exceed the total byte budget")
        records.append(
            {
                "source_id": frozen_row["source_id"],
                "family": frozen_row["family"],
                "split": frozen_row["split"],
                "prefix": prefix,
                "source": source,
                "source_bytes": source_bytes,
                "png_bytes": png_bytes,
                "receipt_sha256": _sha(receipt_bytes),
                "source_json_sha256": _sha(source_bytes),
                "source_digest": compact,
                "analysis_source_digest": spaced,
                "source_render_sha256": _sha(png_bytes),
            }
        )
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Semantic export manifest changed during source selection")
    return manifest_bytes, manifest, records


def build_source_collection(dataset, inputs_out, *, split="validation", max_sources=3):
    """Verify selected exported inputs, then copy only their source JSON/PNG."""
    dataset, inputs_out = Path(dataset).resolve(), Path(inputs_out).resolve()
    if inputs_out.is_relative_to(dataset):
        raise ValueError("Evaluation outputs must not modify the frozen semantic export")
    manifest_bytes, manifest, records = _selected_sources(
        dataset, split=split, max_sources=max_sources
    )
    inputs_out.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in records:
        json_relative, png_relative = f"sources/{row['prefix']}.json", f"images/{row['prefix']}.png"
        for relative, data in (
            (json_relative, row["source_bytes"]),
            (png_relative, row["png_bytes"]),
        ):
            destination = inputs_out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
        source = {
            "design": json_relative,
            "json_sha256": canonical_json_sha256(row["source"]),
            "render": png_relative,
            "render_sha256": row["source_render_sha256"],
        }
        candidate_id = row["source_digest"]
        cases.append(
            {
                "case_id": row["source_id"],
                "source_id": row["source_id"],
                "family_id": row["family"],
                "split": row["split"],
                "source_parity_verified": False,
                "training_rights_reviewed": False,
                "rights_reference": "Evaluation-only source export; no training-rights claim.",
                "target_canvas": row["source"]["canvas"],
                "source": source,
                "baseline_candidate_id": candidate_id,
                "admitted_candidate_ids": [],
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "design": json_relative,
                        "design_sha256": source["json_sha256"],
                        "render": png_relative,
                        "render_sha256": source["render_sha256"],
                        "admitted": False,
                        "diagnostics": {
                            "contract_valid": False,
                            "contract_violations": ["not_evaluated_source_only"],
                            "evaluation_kind": "source_only_no_resized_candidate",
                        },
                        "provenance": {
                            "element_ids": [element["id"] for element in row["source"]["elements"]],
                            "original_source_only": True,
                        },
                    }
                ],
            }
        )
    collection = {
        "schema_version": 1,
        "collection_id": f"semantic-source-eval-{_sha(manifest_bytes)[:20]}-{split}",
        "purpose": "research",
        "provenance": {
            "engine_revision": "frozen-semantic-source-export",
            "working_tree_digest": _sha(manifest_bytes),
            "policy_version": "source-analysis-inputs-v1",
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "adapter_sha256": _sha(Path(__file__).read_bytes()),
            "labels_opened": False,
            "native_renders_generated": False,
            "model_executed": False,
            "quality_claim": "Unlabeled source inputs; model quality is untested.",
        },
        "cases": cases,
    }
    validate_collection(collection, root=inputs_out)
    with (inputs_out / "collection.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(collection) + "\n")
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": _sha(manifest_bytes),
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "split": split,
        "labels_opened": False,
        "copied_source_count": len(records),
        "sources": [
            {
                key: value
                for key, value in row.items()
                if key not in {"source", "source_bytes", "png_bytes"}
            }
            for row in records
        ],
    }
    with (inputs_out / "preparation.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(report) + "\n")
    return collection


def prepare_semantic_eval(dataset, out, inputs_out=None, *, split="validation", max_sources=3):
    """Bridge the verified source-only collection into the existing probe builder."""
    from experiments.vision_critic.independent_probes import prepare_collection

    out = Path(out).resolve()
    inputs_out = Path(inputs_out).resolve() if inputs_out else out.with_name(out.name + "-inputs")
    dataset = Path(dataset).resolve()
    if out.exists() or inputs_out.exists():
        raise FileExistsError("Semantic evaluation output directories must be fresh")
    if (
        out.is_relative_to(dataset)
        or out.is_relative_to(inputs_out)
        or inputs_out.is_relative_to(out)
    ):
        raise ValueError("Semantic evaluation output paths must be separate from frozen inputs")
    build_source_collection(dataset, inputs_out, split=split, max_sources=max_sources)
    return prepare_collection(
        inputs_out, out, splits=(split,), max_cases=max_sources, modes=("analyze",)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs-out", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-sources", type=int, default=3)
    args = parser.parse_args(argv)
    bundle = prepare_semantic_eval(
        args.dataset, args.out, args.inputs_out, split=args.split, max_sources=args.max_sources
    )
    print(json.dumps({"bundle_sha256": bundle["bundle_sha256"], "probes": len(bundle["probes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
