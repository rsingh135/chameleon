"""Test-only fixture labels. Never turn algorithm scores into human preferences."""

import json
import math
import os
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import pytest

from experiments.vision_critic.contracts import load_collection, make_pair_id, validate_label


def label_synthetic_fixture(collection_root: Path, destination: Path) -> int:
    """Exercise the pipeline on synthetic metric labels; leave test cases blind."""
    collection = load_collection(collection_root)
    if collection["purpose"] != "synthetic":
        raise ValueError("Automated fixture labels are only permitted for synthetic test data")
    labels = []
    for case in collection["cases"]:
        if case["split"] == "test":
            continue
        for a, b in combinations(case["candidates"], 2):

            def metric(candidate):
                value = candidate["provenance"].get("selection", {}).get("fidelity", 0)
                return value if isinstance(value, (int, float)) and math.isfinite(value) else 0

            difference = metric(a) - metric(b)
            label = {
                "schema_version": 1,
                "collection_id": collection["collection_id"],
                "case_id": case["case_id"],
                "pair_id": make_pair_id(
                    collection["collection_id"],
                    case["case_id"],
                    a["candidate_id"],
                    b["candidate_id"],
                ),
                "candidate_a_id": a["candidate_id"],
                "candidate_b_id": b["candidate_id"],
                "decision": "tie" if abs(difference) < 1e-8 else "A" if difference > 0 else "B",
                "issues": [],
                "annotator": "automated-synthetic-metric-fixture",
                "created_at": datetime.now(UTC).isoformat(),
                "source_render_sha256": case["source"]["render_sha256"],
                "a_render_sha256": a["render_sha256"],
                "b_render_sha256": b["render_sha256"],
            }
            labels.append(validate_label(label, collection, allow_synthetic=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(label, sort_keys=True) + "\n" for label in labels))
        stream.flush()
        os.fsync(stream.fileno())
    return len(labels)


def test_synthetic_pipeline_fixture_never_labels_test_cases(tmp_path):
    from test_vision_contracts import build_collection

    root = tmp_path / "collection"
    collection = build_collection(root, purpose="synthetic", cases=3)
    for case, split in zip(collection["cases"], ("train", "validation", "test"), strict=True):
        case["split"] = split
        case["source_parity_verified"] = False
    (root / "collection.json").write_text(json.dumps(collection), encoding="utf-8")
    path = tmp_path / "synthetic-labels.jsonl"
    assert label_synthetic_fixture(root, path) == 2
    labels = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert collection["cases"][2]["case_id"] not in {row["case_id"] for row in labels}
    assert all(row["annotator"].startswith("automated-synthetic") for row in labels)
    with pytest.raises(FileExistsError):
        label_synthetic_fixture(root, path)


def test_real_research_corpus_cannot_receive_automated_fixture_labels(tmp_path):
    from test_vision_contracts import build_collection

    root = tmp_path / "collection"
    build_collection(root, purpose="research")
    destination = tmp_path / "labels.jsonl"
    with pytest.raises(ValueError, match="only permitted for synthetic"):
        label_synthetic_fixture(root, destination)
    assert not destination.exists()
