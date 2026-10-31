"""Train a real CPU pairwise logistic head over handcrafted rendered features.

Reviewed labels are required by default. --allow-synthetic is solely for
infrastructure fixtures and marks every resulting artifact accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    corpus_identity,
    full_corpus_provenance,
    pair_features,
)


def implementation_identity() -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("features.py", "train_ranker.py")
    }


def atomic_json(path: Path, value: dict, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _encoded(state: dict) -> bytes:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save_model(path: Path, state: dict) -> None:
    atomic_json(path, {"state": state, "state_sha256": hashlib.sha256(_encoded(state)).hexdigest()})


def load_model(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    state = artifact["state"]
    if hashlib.sha256(_encoded(state)).hexdigest() != artifact["state_sha256"]:
        raise ValueError("Ranker checkpoint hash mismatch")
    if (
        state.get("feature_version") != FEATURE_VERSION
        or state.get("feature_names") != FEATURE_NAMES
    ):
        raise ValueError("Ranker feature contract differs from this implementation")
    if state.get("implementation_sha256") != implementation_identity():
        raise ValueError(
            "Ranker feature/trainer implementation changed; revalidate before migration"
        )
    for name in ("weights", "mean", "scale"):
        vector = np.asarray(state[name], dtype=np.float64)
        if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
            raise ValueError("Invalid ranker checkpoint vector")
    if (np.asarray(state["scale"]) <= 0).any():
        raise ValueError("Ranker checkpoint contains invalid feature scale")
    return state


def score(state: dict, vector: np.ndarray) -> float:
    return float(
        ((vector - np.asarray(state["mean"])) / np.asarray(state["scale"]))
        @ np.asarray(state["weights"])
    )


def preference(state: dict, record: dict, tie_threshold: float = 0.1) -> tuple[str, float]:
    a, b = pair_features(record)
    margin = score(state, a) - score(state, b)
    return ("tie" if abs(margin) <= tie_threshold else "A" if margin > 0 else "B"), margin


def fit(
    records: list[dict],
    *,
    steps: int = 300,
    learning_rate: float = 0.1,
    l2: float = 0.01,
    max_seconds: float = 60,
    resume: dict | None = None,
    allow_synthetic: bool = False,
) -> dict:
    started = time.monotonic()
    if (
        type(steps) is not int
        or not 1 <= steps <= 10_000
        or not np.isfinite([learning_rate, l2, max_seconds]).all()
        or learning_rate <= 0
        or l2 < 0
        or max_seconds <= 0
    ):
        raise ValueError("Invalid or unbounded CPU training recipe")
    train = [
        record
        for record in records
        if record["split"] == "train" and record["decision"] in {"A", "B", "tie"}
    ]
    validation = [record for record in records if record["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Reviewed train and validation records are both required")
    if {record["family_id"] for record in train} & {record["family_id"] for record in validation}:
        raise ValueError("Training and validation families overlap")
    if {record["source_id"] for record in train} & {record["source_id"] for record in validation}:
        raise ValueError("Training and validation sources overlap")
    purpose = "synthetic_infrastructure_only" if allow_synthetic else "reviewed_preference_training"
    if not allow_synthetic and any(
        "synthetic" in str(record.get("purpose", "")).lower() for record in records
    ):
        raise ValueError("Synthetic records cannot establish a task-trained critic")
    identity = corpus_identity(records)
    vectors = []
    for record in train:
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("CPU feature preparation exceeded the training deadline")
        vectors.append(pair_features(record))
    all_vectors = np.concatenate([np.stack(pair) for pair in vectors], axis=0)
    mean, scale = all_vectors.mean(axis=0), np.maximum(all_vectors.std(axis=0), 1e-6)
    differences = np.stack([(a - b) / scale for a, b in vectors])
    targets = np.asarray([{"A": 1.0, "B": 0.0, "tie": 0.5}[r["decision"]] for r in train])
    weights = np.zeros(len(FEATURE_NAMES))
    completed = 0
    history = []
    if resume is not None:
        if resume.get("implementation_sha256") != implementation_identity():
            raise ValueError("Resume feature/trainer implementation changed")
        for name, expected in (
            ("corpus_sha256", identity),
            ("feature_version", FEATURE_VERSION),
            ("learning_rate", learning_rate),
            ("l2", l2),
            ("purpose", purpose),
        ):
            if resume.get(name) != expected:
                raise ValueError(f"Resume identity/recipe differs: {name}")
        if not np.array_equal(resume["mean"], mean) or not np.array_equal(resume["scale"], scale):
            raise ValueError("Resume feature normalization differs")
        weights = np.asarray(resume["weights"], dtype=np.float64).copy()
        completed = resume["completed_steps"]
        history = list(resume["loss_history"])
        if completed >= steps:
            raise ValueError("Resume target steps must exceed the saved completed steps")
    for step in range(completed, steps):
        if time.monotonic() - started > max_seconds:
            break
        logits = differences @ weights
        probability = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
        loss = float(
            np.mean(np.logaddexp(0, logits) - targets * logits) + l2 * np.dot(weights, weights) / 2
        )
        gradient = differences.T @ (probability - targets) / len(targets) + l2 * weights
        weights -= learning_rate * gradient
        if not np.isfinite(weights).all() or not np.isfinite(loss):
            raise ValueError("Non-finite CPU optimization; lower learning rate")
        history.append(loss)
        completed = step + 1
    if completed == 0:
        raise TimeoutError("CPU training deadline elapsed before the first completed step")
    return {
        "schema_version": 1,
        "model_kind": "handcrafted_rendered_feature_pairwise_logistic",
        "implementation_sha256": implementation_identity(),
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "completed_steps": completed,
        "requested_steps": steps,
        "learning_rate": learning_rate,
        "l2": l2,
        "loss_history": history,
        "corpus_sha256": identity,
        "identity_scope": "non-test input bytes and annotations; full collection remains frozen",
        "full_corpus_provenance": full_corpus_provenance(records),
        "purpose": purpose,
        "task_model_trained": not allow_synthetic,
        "train_pair_count": len(train),
        "excluded_train_decisions": len(
            [
                r
                for r in records
                if r["split"] == "train" and r["decision"] in {"neither", "abstain"}
            ]
        ),
        "tie_handling": "soft binary target 0.5; neither and abstain excluded",
        "train_families": sorted({r["family_id"] for r in train}),
        "validation_families": sorted({r["family_id"] for r in validation}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    from .dataset import load_preferences
    from .evaluate import evaluate_ranker

    records = load_preferences(args.collection, args.labels, allow_synthetic=args.allow_synthetic)
    state = fit(
        records,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_seconds=args.max_seconds,
        resume=load_model(args.resume) if args.resume else None,
        allow_synthetic=args.allow_synthetic,
    )
    save_model(args.out / "model.json", state)
    reloaded = load_model(args.out / "model.json")
    report = evaluate_ranker(reloaded, records)
    atomic_json(args.out / "evaluation.json", report)
    print(json.dumps({"model": str(args.out / "model.json"), "evaluation": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
