"""Evaluate held-out human preferences separately from admissible selection.

All predictions are shadow recommendations. Selection falls back to the exact
case baseline when a requested candidate is not admitted; an unadmitted baseline
produces no selection. Metrics never manufacture quality labels from geometry.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def admissible_selection(record: dict, choice: str | None) -> dict:
    choice = choice if isinstance(choice, str) else None
    admitted = set(record.get("admitted_candidate_ids", []))
    requested = {
        "A": record["candidate_a"]["candidate_id"],
        "B": record["candidate_b"]["candidate_id"],
    }.get(choice)
    if requested in admitted:
        return {"candidate_id": requested, "fallback": False, "reason": "admitted recommendation"}
    baseline = record.get("baseline_candidate_id")
    reason = (
        "tie or neither"
        if choice in {"tie", "neither"}
        else "invalid or missing verdict"
        if choice not in {"A", "B"}
        else "recommended candidate is not admitted"
    )
    return {
        "candidate_id": baseline if baseline in admitted else None,
        "fallback": True,
        "reason": reason,
    }


def _metrics(rows: list[tuple[dict, str | None]]) -> dict:
    judgments = [
        (record, choice)
        for record, choice in rows
        if record["decision"] in {"A", "B", "tie", "neither"}
    ]
    binary = [(record, choice) for record, choice in judgments if record["decision"] in {"A", "B"}]
    families, sources = defaultdict(list), defaultdict(list)
    for record, choice in binary:
        correct = float(choice == record["decision"])
        families[record["family_id"]].append(correct)
        sources[record["source_id"]].append(correct)

    def mean(values):
        return sum(values) / len(values) if values else None

    selections = [admissible_selection(record, choice) for record, choice in rows]
    return {
        "pair_count": len(rows),
        "binary_pair_count": len(binary),
        "binary_preference_accuracy": mean([float(c == r["decision"]) for r, c in binary]),
        "categorical_preference_accuracy": mean([float(c == r["decision"]) for r, c in judgments]),
        "family_macro_binary_accuracy": mean([mean(values) for values in families.values()]),
        "source_macro_binary_accuracy": mean([mean(values) for values in sources.values()]),
        "per_family_binary_accuracy": {name: mean(values) for name, values in families.items()},
        "per_source_binary_accuracy": {name: mean(values) for name, values in sources.items()},
        "tie_label_count": sum(record["decision"] == "tie" for record, _ in rows),
        "neither_label_count": sum(record["decision"] == "neither" for record, _ in rows),
        "abstain_label_count": sum(record["decision"] == "abstain" for record, _ in rows),
        "invalid_or_missing_verdicts": sum(
            choice not in {"A", "B", "tie", "neither"} for _, choice in rows
        ),
        "fallback_count": sum(selection["fallback"] for selection in selections),
        "no_selection_count": sum(selection["candidate_id"] is None for selection in selections),
        "admitted_selection_count": sum(
            selection["candidate_id"] is not None for selection in selections
        ),
        "candidate_pair_coverage": mean([float(r["decision"] != "neither") for r, _ in judgments]),
    }


def evaluate_choices(
    records: list[dict], choices: dict[str, str | None], *, include_test: bool = False
) -> dict:
    records = [record for record in records if include_test or record["split"] != "test"]
    splits = sorted({record["split"] for record in records})
    return {
        split: _metrics(
            [
                (
                    record,
                    choices.get(record["pair_id"])
                    if isinstance(choices.get(record["pair_id"]), str)
                    else None,
                )
                for record in records
                if record["split"] == split
            ]
        )
        for split in splits
    }


def evaluate_ranker(
    state: dict, records: list[dict], tie_threshold: float = 0.1, *, include_test: bool = False
) -> dict:
    from .features import corpus_identity, full_corpus_provenance, pair_features
    from .train_ranker import implementation_identity, preference

    if state.get("implementation_sha256") != implementation_identity():
        raise ValueError("Evaluation feature/trainer implementation differs from the checkpoint")
    if corpus_identity(records) != state["corpus_sha256"]:
        raise ValueError(
            "Evaluation train/validation inputs or annotations differ from the checkpoint"
        )

    provenance = full_corpus_provenance(records)

    records = [record for record in records if include_test or record["split"] != "test"]
    choices, margins, swapped_consistent, reference = {}, {}, [], {}
    baseline = {}
    for record in records:
        choice, margin = preference(state, record, tie_threshold)
        choices[record["pair_id"]] = choice
        margins[record["pair_id"]] = margin
        swapped = {
            **record,
            "candidate_a": record["candidate_b"],
            "candidate_b": record["candidate_a"],
            "images": [record["images"][0], record["images"][2], record["images"][1]],
        }
        swap_choice, swap_margin = preference(state, swapped, tie_threshold)
        swapped_consistent.append(
            abs(margin + swap_margin) < 1e-9
            and {"A": "B", "B": "A", "tie": "tie"}[choice] == swap_choice
        )
        a, b = pair_features(record)
        delta = float(a[:75].sum() - b[:75].sum())
        reference[record["pair_id"]] = "tie" if abs(delta) < 1e-9 else "A" if delta > 0 else "B"
        baseline[record["pair_id"]] = next(
            (
                letter
                for letter, candidate in (
                    ("A", record["candidate_a"]),
                    ("B", record["candidate_b"]),
                )
                if candidate["candidate_id"] == record.get("baseline_candidate_id")
            ),
            None,
        )
    return {
        "schema_version": 1,
        "full_corpus_provenance": provenance,
        "training_corpus_sha256": state["corpus_sha256"],
        "model_kind": state["model_kind"],
        "purpose": state["purpose"],
        "task_model_trained": state["task_model_trained"],
        "quality_claim": "Synthetic metrics validate infrastructure only"
        if not state["task_model_trained"]
        else "Held-out labeled preferences; inspect family coverage",
        "test_split_unblinded": include_test,
        "metrics": evaluate_choices(records, choices, include_test=True),
        "baselines": {
            "always_A": evaluate_choices(
                records, {r["pair_id"]: "A" for r in records}, include_test=True
            ),
            "always_B": evaluate_choices(
                records, {r["pair_id"]: "B" for r in records}, include_test=True
            ),
            "case_baseline_in_pair": evaluate_choices(records, baseline, include_test=True),
            "untrained_reference_similarity": evaluate_choices(
                records, reference, include_test=True
            ),
        },
        "labeled_pair_oracle": {
            "human_preferred_option_present": sum(
                r["decision"] in {"A", "B", "tie"} for r in records
            ),
            "neither_preferred": sum(r["decision"] == "neither" for r in records),
            "meaning": "Labeled-pair coverage; no claim about the full candidate pool",
        },
        "swap_consistency": sum(swapped_consistent) / len(swapped_consistent) if records else None,
        "margins": margins,
    }


def read_verdicts(path: Path, records: list[dict]) -> dict[str, str | None]:
    from .contracts import parse_json
    from .features import pair_input_identity
    from .train_sft import parse_model_verdict

    known = {record["pair_id"]: record for record in records}
    verdicts = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = parse_json(line)
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "schema_version",
                "pair_id",
                "corpus_sha256",
                "input_sha256",
                "choice",
                "issues",
                "valid",
                "raw_response",
            }
            or row["schema_version"] != 1
        ):
            raise ValueError(f"Verdict line {line_number} must match the versioned verdict schema")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in verdicts:
            raise ValueError(f"Missing/duplicate pair_id at verdict line {line_number}")
        if pair_id not in known or row["input_sha256"] != pair_input_identity(known[pair_id]):
            raise ValueError("Verdict belongs to an unknown pair or stale input corpus")
        if not isinstance(row["raw_response"], str) or type(row["valid"]) is not bool:
            raise ValueError("Verdict raw response/validity has invalid types")
        candidate_ids = {
            side: known[pair_id][key].get("provenance", {}).get("element_ids", [])
            for side, key in (("A", "candidate_a"), ("B", "candidate_b"))
        }
        parsed = parse_model_verdict(row["raw_response"], candidate_ids)
        verdicts[pair_id] = (
            parsed["choice"]
            if (
                row["valid"] is True
                and parsed["valid"]
                and row["choice"] == parsed["choice"]
                and row["issues"] == parsed["issues"]
            )
            else None
        )
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unblind test labels after freezing the model/recipe",
    )
    args = parser.parse_args()
    if bool(args.model) == bool(args.verdicts):
        parser.error("Choose exactly one of --model or --verdicts")
    from .dataset import load_preferences
    from .train_ranker import atomic_json, load_model

    records = load_preferences(args.collection, args.labels, allow_synthetic=args.allow_synthetic)
    report = (
        evaluate_ranker(load_model(args.model), records, include_test=args.include_test)
        if args.model
        else {
            "schema_version": 1,
            "metrics": evaluate_choices(
                records, read_verdicts(args.verdicts, records), include_test=args.include_test
            ),
            "test_split_unblinded": args.include_test,
            "quality_claim": "Synthetic infrastructure only"
            if args.allow_synthetic
            else "Held-out labeled model verdicts; no GPU compatibility claim",
        }
    )
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
