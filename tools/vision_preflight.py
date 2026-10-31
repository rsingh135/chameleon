"""Check experiment gates locally without importing models or starting paid compute.

A passing static recipe is not CUDA compatibility or design quality evidence.
Training readiness additionally requires reviewed, family-disjoint labelled data
and an implemented task trainer. This tool does not grant deployment approval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train", "validation", "test"}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def review(config: dict, manifest: dict) -> dict:
    errors = []
    blockers = []
    if config.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        errors.append("Unsupported recipe or source-manifest schema")
    if config.get("mode") not in {"off", "shadow", "rerank"}:
        errors.append("Unknown critic mode")
    if not config.get("include_baseline") or not config.get("deterministic_validation_required"):
        errors.append("Baseline fallback and deterministic validation are required")
    if config.get("output_contract") != "candidate_id_or_abstain":
        errors.append("The critic may only select a submitted candidate or abstain")
    limit = config.get("candidate_limit")
    if type(limit) is not int or not 2 <= limit <= 8:
        errors.append("Candidate limit must be an integer from 2 to 8")
    budget = config.get("budget", {})
    values = [
        budget.get(key)
        for key in (
            "session_minutes",
            "padded_hourly_usd",
            "pilot_compute_allowance_usd",
            "planning_reserve_usd",
        )
    ]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in values
    ):
        errors.append("Budget values must be finite positive numbers")
    elif values[0] / 60 * values[1] > values[2]:
        errors.append("One session exceeds the pilot compute allowance")

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        errors.append("sources must be a list")
        sources = []
    families = {}
    source_ids = set()
    counts = dict.fromkeys(sorted(SPLITS), 0)
    for source in sources:
        if not isinstance(source, dict):
            errors.append("Every source entry must be an object")
            continue
        identity = source.get("source_id")
        family = source.get("family_id")
        split = source.get("split")
        if not isinstance(identity, str) or not identity or identity in source_ids:
            errors.append("Source IDs must be nonempty and unique")
        source_ids.add(str(identity))
        if not isinstance(family, str) or not family or split not in SPLITS:
            errors.append(f"Invalid family/split for {identity}")
            continue
        if family in families and families[family] != split:
            errors.append(f"Family leakage across splits: {family}")
        families[family] = split
        counts[split] += 1
        if source.get("source_parity_verified") is not True:
            blockers.append(f"Source rendering needs review: {identity}")
        if source.get("training_rights_reviewed") is not True or not source.get("rights_reference"):
            blockers.append(f"Training rights need a recorded review: {identity}")
        if split != "test" and (
            type(source.get("human_preference_pairs")) is not int
            or source.get("human_preference_pairs", 0) < 1
        ):
            blockers.append(f"No reviewed preference pairs: {identity}")
    if not all(counts.values()):
        blockers.append("Need independently grouped train, validation, and untouched test sources")
    if not config.get("task_trainer_implemented"):
        blockers.append("Task-specific preference trainer and evaluator are not implemented")
    return {
        "recipe_valid": not errors,
        "task_training_ready": not errors and not blockers,
        "starts_compute": False,
        "gpu_model_compatibility_validated": False,
        "quality_validated": False,
        "source_counts": counts,
        "family_count": len(families),
        "errors": errors,
        "training_blockers": blockers,
        "scope": "Static prerequisite review only; does not verify manifest claims or labels",
    }


def review_artifacts(
    config: dict, collection_root: Path, labels_path: Path, *, allow_synthetic: bool = False
) -> dict:
    """Count validated on-disk evidence rather than trusting declared pair counts."""
    from experiments.vision_critic.contracts import load_collection
    from experiments.vision_critic.dataset import load_preferences

    collection = load_collection(collection_root)
    preferences = load_preferences(collection_root, labels_path, allow_synthetic=allow_synthetic)
    sources = {}
    for case in collection["cases"]:
        sources.setdefault(
            case["source_id"],
            {
                key: case[key]
                for key in (
                    "source_id",
                    "family_id",
                    "split",
                    "source_parity_verified",
                    "training_rights_reviewed",
                    "rights_reference",
                )
            },
        )["human_preference_pairs"] = 0
    for record in preferences:
        if record["decision"] != "abstain":
            sources[record["source_id"]]["human_preference_pairs"] += 1
    report = review(config, {"schema_version": 1, "sources": list(sources.values())})
    report.update(
        artifact_hashes_verified=True,
        collection_id=collection["collection_id"],
        purpose=collection["purpose"],
        validated_label_count=len(preferences),
        scope="Verified data prerequisites only; this tool does not inspect CUDA/runtime",
    )
    if collection["purpose"] == "synthetic":
        report["task_training_ready"] = False
        report["training_blockers"].append(
            "Synthetic practice is never reviewed task-training data"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vision_critic/pilot.json")
    parser.add_argument("--require-training-ready", action="store_true")
    parser.add_argument("--collection", type=Path, help="Verify actual collected artifacts")
    parser.add_argument("--labels", type=Path, help="Validate the matching JSONL ledger")
    parser.add_argument("--allow-synthetic", action="store_true", help="Inspect test fixtures only")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if bool(args.collection) != bool(args.labels):
        parser.error("--collection and --labels are required together")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.collection:
            report = review_artifacts(
                config, args.collection, args.labels, allow_synthetic=args.allow_synthetic
            )
        else:
            manifest_path = (ROOT / config["dataset_manifest"]).resolve()
            manifest_path.relative_to(ROOT)
            report = review(config, json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"recipe_valid": False, "error": str(error)}), file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return (
        0
        if report["recipe_valid"]
        and (not args.require_training_ready or report["task_training_ready"])
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
