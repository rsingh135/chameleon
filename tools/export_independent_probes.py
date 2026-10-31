"""Export frozen probes as label-free neural-worker JSONL without inference.

Optional validation gold is a separate file, never merged into inputs. Image
paths point to verified copies in the original bundle; pixels are not changed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from experiments.vision_critic.contracts import (
    SHA256_PATTERN,
    canonical_json,
    canonical_json_sha256,
    parse_json,
    safe_artifact_path,
    sha256_file,
)
from experiments.vision_critic.independent_probes import load_bundle

PROTOCOL = "chameleon-neural-probe-export-v1"
GOLD_CONDITIONS = {"original", "swapped", "opaque_ids"}
AXES = {"hierarchy", "attachment", "reading_order", "prominence"}


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Export metadata exceeds its byte budget")
    return parse_json(path.read_text(encoding="utf-8-sig"))


def _image(root, files, reference, role):
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("Neural probe requires an explicit frozen image")
    path = Path(reference["path"])
    path = path if path.is_absolute() else root / path
    try:
        relative = path.absolute().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Neural probe image escaped the frozen bundle") from exc
    path = safe_artifact_path(root, relative)
    if files.get(relative) != reference.get("sha256"):
        raise ValueError("Neural probe image differs from its frozen identity")
    return {"role": role, "path": str(path.resolve()), "sha256": reference["sha256"]}


def _record(bundle, root, probe, case):
    request = _read_json(safe_artifact_path(root, probe["request_path"]))
    images = [_image(root, bundle["files"], request["source_render"], "source")]
    plans = request.get("plan_menu", {}).get("plans", [])
    if probe["mode"] == "analyze":
        if plans or request.get("plans"):
            raise ValueError("Source-only analysis cannot export candidate images")
    else:
        if len(plans) != 2:
            raise ValueError("Neural pair records require exactly two candidate images")
        rendered = {row["plan_id"]: row for row in request.get("plans", [])}
        if len(rendered) != 2 or set(rendered) != {row["plan_id"] for row in plans}:
            raise ValueError("Candidate images do not match the ordered plan menu")
        images.extend(
            _image(root, bundle["files"], rendered[plan["plan_id"]].get("render"), side)
            for side, plan in zip("AB", plans, strict=True)
        )
    if "source_element_ids" in case and not set(probe["element_map"].values()) <= set(
        case["source_element_ids"]
    ):
        raise ValueError("Probe maps unknown original source element IDs")
    if "candidate_ids" in case and not set(probe["plan_map"].values()) <= set(
        case["candidate_ids"]
    ):
        raise ValueError("Probe maps unknown original candidate IDs")
    source_digest = request.get("source_digest", request.get("plan_menu", {}).get("source_digest"))
    if not isinstance(source_digest, str) or not SHA256_PATTERN.fullmatch(source_digest):
        raise ValueError("Neural probe requires its bound source design digest")
    task = {
        "bundle_sha256": bundle["bundle_sha256"],
        "case_id": probe["case_id"],
        "mode": probe["mode"],
        "condition": probe["condition"],
        "request_digest": probe["request_digest"],
        "source_digest": source_digest,
        "menu_digest": probe["menu_digest"],
        "known_element_ids": deepcopy(probe["known_element_ids"]),
        "plan_map": deepcopy(probe["plan_map"]),
        "element_map": deepcopy(probe["element_map"]),
    }
    if "source_analysis_digest" in request:
        task["source_analysis_digest"] = request["source_analysis_digest"]
    record = {
        "record_id": probe["probe_id"],
        "family_id": probe["family_id"],
        "split": probe["split"],
        "label_origin": "withheld_evaluation",
        "images": images,
        "task": task,
        "review": {
            "scope": "independent_evaluation_only",
            "weak_or_unreviewed_inputs": True,
            "quality_claim": False,
        },
    }
    record["review"]["record_input_sha256"] = canonical_json_sha256(record)
    return record, [probe["plan_map"][plan["plan_id"]] for plan in plans]


def _validation_gold(path, bundle, cases, records, choices):
    labels = _read_json(path)
    if (
        not isinstance(labels, dict)
        or type(labels.get("schema_version")) is not int
        or labels["schema_version"] != 1
        or labels.get("bundle_sha256") != bundle["bundle_sha256"]
        or labels.get("label_kind") not in {"human", "synthetic"}
        or not isinstance(labels.get("decisions"), list)
    ):
        raise ValueError("Validation labels differ from the frozen bundle or label schema")
    decisions = {}
    for row in labels["decisions"]:
        case = cases.get(row.get("case_id")) if isinstance(row, dict) else None
        if case is None or row["case_id"] in decisions:
            raise ValueError("Validation label contains an unknown or duplicate case ID")
        if case["split"] != "validation":
            raise ValueError("Only validation labels may be exported; test labels remain sealed")
        known = set(case.get("candidate_ids", []))
        for record in records:
            if record["task"]["case_id"] == case["case_id"]:
                known.update(choices[record["record_id"]])
        outcomes = known | {"tie", "abstain"}
        axes = row.get("axes", {})
        if (
            row.get("preferred_candidate_id") not in outcomes
            or not isinstance(axes, dict)
            or axes.keys() - AXES
            or any(value not in outcomes for value in axes.values())
        ):
            raise ValueError("Validation label references an unknown candidate or outcome")
        decisions[row["case_id"]] = row["preferred_candidate_id"]
    gold = []
    for record in records:
        preferred = decisions.get(record["task"]["case_id"])
        if (
            preferred is None
            or record["task"]["condition"] not in GOLD_CONDITIONS
            or record["task"]["mode"] == "analyze"
        ):
            continue
        ordered = choices[record["record_id"]]
        if preferred in {"tie", "abstain"}:
            label = "N"
        elif preferred in ordered:
            label = "AB"[ordered.index(preferred)]
        else:
            raise ValueError("Preferred candidate is absent from the exported comparison")
        labeled = deepcopy(record)
        labeled.update(label=label, label_origin=labels["label_kind"])
        labeled["review"]["scope"] = "validation_only_not_sealed_test_truth"
        labeled["review"]["label_projection"] = "tie_and_abstain_map_to_N"
        gold.append(labeled)
    return gold


def _jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def export_probes(bundle_path, out, *, labels_path=None, max_records=200):
    """Verify and export a bounded artifact; never open labels unless requested."""
    if type(max_records) is not int or not 1 <= max_records <= 200:
        raise ValueError("Export record budget must be an integer from 1 to 200")
    bundle_path, out = Path(bundle_path).resolve(), Path(out).absolute()
    if out.exists():
        raise FileExistsError(f"Export directory already exists: {out}")
    root = bundle_path.parent
    if out.resolve().is_relative_to(root):
        raise ValueError("Export directory must be outside the immutable input bundle")
    exporter_sha = sha256_file(Path(__file__))
    bundle = load_bundle(bundle_path)
    if len(bundle["probes"]) > max_records:
        raise ValueError("Frozen probes exceed the requested export record budget")
    cases = {case["case_id"]: case for case in bundle["cases"]}
    records, choices = [], {}
    for probe in bundle["probes"]:
        record, ordered = _record(bundle, root, probe, cases[probe["case_id"]])
        records.append(record)
        choices[record["record_id"]] = ordered
    labels_sha = sha256_file(labels_path) if labels_path is not None else None
    gold = _validation_gold(labels_path, bundle, cases, records, choices) if labels_path else None
    if load_bundle(bundle_path)["bundle_sha256"] != bundle["bundle_sha256"]:
        raise ValueError("Frozen bundle changed during export")
    if sha256_file(Path(__file__)) != exporter_sha or (
        labels_path is not None and sha256_file(labels_path) != labels_sha
    ):
        raise ValueError("Exporter or requested validation labels changed during export")
    out.mkdir(parents=True, exist_ok=False)
    _jsonl(out / "inputs.jsonl", records)
    if gold is not None:
        _jsonl(out / "validation-labels.jsonl", gold)
    files = {"inputs.jsonl": sha256_file(out / "inputs.jsonl")}
    if gold is not None:
        files["validation-labels.jsonl"] = sha256_file(out / "validation-labels.jsonl")
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_file_sha256": sha256_file(bundle_path),
        "bundle_root": str(root),
        "bundle_path": str(bundle_path),
        "exporter_sha256": exporter_sha,
        "input_record_count": len(records),
        "validation_gold_count": len(gold) if gold is not None else 0,
        "validation_labels_sha256": labels_sha,
        "mode_counts": dict(Counter(row["task"]["mode"] for row in records)),
        "files": files,
        "record_sha256": {row["record_id"]: canonical_json_sha256(row) for row in records},
        "record_input_digest_protocol": "canonical_json_excluding_review.record_input_sha256",
        "image_path_policy": "absolute_paths_into_verified_frozen_bundle",
        "inference_executed": False,
        "test_labels_unblinded": False,
        "quality_claim": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    with (out / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(manifest) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Optional validation-only gold JSON")
    parser.add_argument("--max-records", type=int, default=200)
    args = parser.parse_args(argv)
    manifest = export_probes(
        args.bundle, args.out, labels_path=args.labels, max_records=args.max_records
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
