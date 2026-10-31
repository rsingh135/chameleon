"""Train a small scene-role head on frozen, pinned local CLIP two-view features.

This is classifier training, not VLM LoRA or learned crop-region prediction.
No download occurs. Real data must pass the scene manifest/receipt checks.
The CPU optimizer is class-balanced full-batch gradient descent; immutable
checkpoints contain everything needed to resume that optimizer exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .dataset import canonical, dataset_summary, digest, load_records, read_json, sha256

FEATURE_SCHEMA = "clip-two-view-letterbox-l2-source-asset-v2"
ROLES = ["background", "product", "logo", "foreground", "unknown"]
FEATURE_DIM = 1024
MAX_STEPS = 2000


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_recipe(recipe):
    for field, limit in (("max_steps", MAX_STEPS), ("save_every", 100), ("max_seconds", 1800)):
        if type(recipe.get(field)) is not int or not 1 <= recipe[field] <= limit:
            raise ValueError(f"Scene classifier {field} is outside its bounded range")
    if not _finite(recipe.get("learning_rate")) or not 0 < recipe["learning_rate"] <= 1:
        raise ValueError("Scene classifier learning rate must be in (0,1]")
    if not _finite(recipe.get("l2")) or not 0 <= recipe["l2"] <= 1:
        raise ValueError("Scene classifier L2 must be in [0,1]")
    if type(recipe.get("seed")) is not int or not 0 <= recipe["seed"] < 2**32:
        raise ValueError("Scene classifier seed must be a bounded integer")


def prepare_plan(rows, recipe, backbone_sha256):
    """Freeze non-test supervision, optimizer and local feature identities."""
    validate_recipe(recipe)
    if len(backbone_sha256) != 64 or any(c not in "0123456789abcdef" for c in backbone_sha256):
        raise ValueError("Scene classifier needs an exact local backbone SHA-256")
    active = [r for r in rows if r["split"] != "test"]
    for field in ("family_id", "source_id", "source_digest"):
        memberships = {}
        for row in rows:
            if memberships.setdefault(row[field], row["split"]) != row["split"]:
                raise ValueError(f"Scene {field} crosses experiment splits")
    if any(r["supervision_origin"] == "synthetic_rule" for r in active):
        raise ValueError(
            "Real scene classifier training cannot use synthetic infrastructure labels"
        )
    if not all(any(r["split"] == split for r in active) for split in ("train", "validation")):
        raise ValueError("Scene classifier needs reviewed train and validation families")
    summary = dataset_summary(rows)
    trainer_sha256 = sha256(__file__)
    stable = {
        k: v for k, v in recipe.items() if k not in {"max_steps", "save_every", "max_seconds"}
    }
    inputs = [
        {
            "record_id": r["record_id"],
            "split": r["split"],
            "family_id": r["family_id"],
            "source_id": r["source_id"],
            "input_digest": r["input_digest"],
        }
        for r in active
    ]
    return {
        "schema_version": 1,
        "model_kind": "frozen_clip_two_view_role_head",
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "roles": ROLES,
        "backbone_sha256": backbone_sha256,
        "recipe": dict(recipe),
        "trainer_sha256": trainer_sha256,
        "recipe_sha256": digest({"recipe": stable, "trainer_sha256": trainer_sha256}),
        "corpus_sha256": digest(inputs),
        "full_corpus_provenance_sha256": digest(
            [{"record_id": r["record_id"], "input_digest": r["input_digest"]} for r in rows]
        ),
        "dataset": summary,
        "task_model_trained": False,
        "quality_validated": False,
        "predicts_regions": False,
        "limitations": [
            "Agent-reviewed labels are not independent human quality evidence",
            "Missing held-out role classes cannot establish their recall",
            "Only the linear role head is optimized; CLIP remains frozen",
        ],
    }


def validate_features(features):
    import numpy as np

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM or not np.isfinite(array).all():
        raise ValueError("Scene features must be finite N x 1024 two-view vectors")
    for offset in (0, 512):
        norms = np.linalg.norm(array[:, offset : offset + 512], axis=1)
        if not np.allclose(norms, 1, atol=1e-4, rtol=0):
            raise ValueError("Each CLIP view must be independently L2 normalized")
    return array


def probabilities(features, weights, bias, *, ablation=False):
    import numpy as np

    if ablation:
        features = np.asarray(features, dtype=np.float64)
        if (
            features.ndim != 2
            or features.shape[1] != FEATURE_DIM
            or not np.isfinite(features).all()
        ):
            raise ValueError("Ablation features must remain finite two-view vectors")
    else:
        features = validate_features(features)
    weights, bias = np.asarray(weights, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    if (
        weights.shape != (len(ROLES), FEATURE_DIM)
        or bias.shape != (len(ROLES),)
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("Scene head parameters have an invalid shape or value")
    logits = features @ weights.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / exponential.sum(axis=1, keepdims=True)


def fit(features, labels, recipe, *, initial=None, callback=None, deadline=None):
    """Optimize actual role labels with equal total mass per represented class."""
    import numpy as np

    validate_recipe(recipe)
    features = validate_features(features)
    if len(labels) != len(features) or not labels or any(label not in ROLES for label in labels):
        raise ValueError("Scene labels do not match known role features")
    target = np.asarray([ROLES.index(label) for label in labels], dtype=np.int64)
    counts = Counter(labels)
    sample_weight = np.asarray([1 / (len(counts) * counts[label]) for label in labels])
    if initial is None:
        rng = np.random.default_rng(recipe["seed"])
        weights = rng.normal(0, 0.001, (len(ROLES), FEATURE_DIM))
        bias, completed, history = np.zeros(len(ROLES)), 0, []
    else:
        weights = np.asarray(initial["weights"], dtype=np.float64).copy()
        bias = np.asarray(initial["bias"], dtype=np.float64).copy()
        completed, history = initial["completed_steps"], list(initial["loss_history"])
        probabilities(features, weights, bias)
        if type(completed) is not int or not 0 < completed < recipe["max_steps"]:
            raise ValueError("Resume target must exceed saved completed steps")
    deadline = deadline or time.monotonic() + recipe["max_seconds"]
    start_steps = completed
    for index in range(completed, recipe["max_steps"]):
        if time.monotonic() >= deadline:
            break
        predicted = probabilities(features, weights, bias)
        loss = float(
            -(
                np.log(np.maximum(predicted[np.arange(len(target)), target], 1e-12)) * sample_weight
            ).sum()
            + 0.5 * recipe["l2"] * (weights * weights).sum()
        )
        residual = predicted.copy()
        residual[np.arange(len(target)), target] -= 1
        residual *= sample_weight[:, None]
        gradient_w = residual.T @ features + recipe["l2"] * weights
        gradient_b = residual.sum(axis=0)
        if not math.isfinite(loss) or not np.isfinite(gradient_w).all():
            raise RuntimeError("Scene role optimizer produced non-finite values")
        weights -= recipe["learning_rate"] * gradient_w
        bias -= recipe["learning_rate"] * gradient_b
        completed, history = index + 1, [*history, loss]
        if callback and (completed % recipe["save_every"] == 0 or completed == recipe["max_steps"]):
            callback(weights, bias, completed, history)
    if completed == start_steps:
        raise TimeoutError("Scene role optimizer did not complete a new step before its deadline")
    return {
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "completed_steps": completed,
        "loss_history": history,
        "optimizer": "class_balanced_full_batch_gradient_descent",
        "optimizer_has_momentum": False,
    }


def evaluate(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Scene evaluation requires aligned nonempty labels and predictions")
    confusion = {role: {other: 0 for other in ROLES} for role in ROLES}
    for label, predicted in zip(labels, predictions, strict=True):
        if label not in ROLES or predicted not in ROLES:
            raise ValueError("Unknown evaluation role")
        confusion[label][predicted] += 1
    support = Counter(labels)
    recalls = {role: confusion[role][role] / support[role] for role in support}
    return {
        "count": len(labels),
        "accuracy": sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels),
        "macro_recall": sum(recalls.values()) / len(recalls),
        "per_role_recall": recalls,
        "class_support": dict(support),
        "confusion": confusion,
        "missing_roles": [role for role in ROLES if role not in support],
        "prediction_counts": dict(Counter(predictions)),
        "quality_validated": False,
    }


def _checkpoint_files(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Scene checkpoint must be a real directory")
    files = {}
    for path in directory.iterdir():
        if path.name == "checkpoint-manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Scene checkpoint may contain regular files only")
        files[path.name] = {"sha256": sha256(path), "size": path.stat().st_size}
    if set(files) != {"head.json", "training-state.json"} or any(
        v["size"] <= 0 for v in files.values()
    ):
        raise ValueError("Scene checkpoint is incomplete")
    return files


def verify_checkpoint(directory):
    manifest = read_json(Path(directory) / "checkpoint-manifest.json")
    if manifest != {"schema_version": 1, "files": _checkpoint_files(directory)}:
        raise ValueError("Scene checkpoint files differ from their published hashes")
    return manifest


def publish_checkpoint(out, plan, processor_digest, result):
    import numpy as np

    out = Path(out)
    destination = out / "published" / f"checkpoint-{result['completed_steps']:06d}"
    if destination.exists():
        raise FileExistsError("Scene checkpoint generation already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = out / (".partial-" + destination.name)
    stage.mkdir()
    identity = {
        key: plan[key]
        for key in (
            "backbone_sha256",
            "corpus_sha256",
            "recipe_sha256",
            "trainer_sha256",
            "feature_schema",
        )
    }
    identity["processor_digest"] = processor_digest
    identity["feature_sha256"] = plan.get("feature_sha256")
    identity["numpy_version"] = np.__version__
    head = {
        "schema_version": 1,
        "model_kind": plan["model_kind"],
        "roles": ROLES,
        "feature_dim": FEATURE_DIM,
        **identity,
        "weights": result["weights"],
        "bias": result["bias"],
        "completed_steps": result["completed_steps"],
        "quality_validated": False,
        "predicts_regions": False,
        "supervision_origins": plan["dataset"]["supervision_origins"],
    }
    state = {
        "schema_version": 1,
        **identity,
        "completed_steps": result["completed_steps"],
        "loss_history": result["loss_history"],
        "recipe": plan["recipe"],
        "optimizer": "class_balanced_full_batch_gradient_descent",
        "optimizer_has_momentum": False,
    }
    _write(stage / "head.json", head)
    _write(stage / "training-state.json", state)
    _write(
        stage / "checkpoint-manifest.json", {"schema_version": 1, "files": _checkpoint_files(stage)}
    )
    verify_checkpoint(stage)
    stage.rename(destination)
    verify_checkpoint(destination)
    return destination


def load_resume(directory, plan, processor_digest):
    import numpy as np

    verify_checkpoint(directory)
    head = read_json(Path(directory) / "head.json")
    state = read_json(Path(directory) / "training-state.json")
    for key in (
        "backbone_sha256",
        "corpus_sha256",
        "recipe_sha256",
        "trainer_sha256",
        "feature_schema",
    ):
        if state.get(key) != plan[key] or head.get(key) != plan[key]:
            raise ValueError(f"Scene resume identity differs: {key}")
    if (
        state.get("processor_digest") != processor_digest
        or head.get("processor_digest") != processor_digest
    ):
        raise ValueError("Scene resume processor differs")
    if state.get("feature_sha256") != plan.get("feature_sha256"):
        raise ValueError("Scene resume frozen features differ")
    if state.get("numpy_version") != np.__version__:
        raise ValueError("Scene resume optimizer runtime differs")
    if head.get("roles") != ROLES or head.get("feature_dim") != FEATURE_DIM:
        raise ValueError("Scene resume role/feature schema differs")
    if (
        state["completed_steps"] != head["completed_steps"]
        or len(state["loss_history"]) != state["completed_steps"]
    ):
        raise ValueError("Scene resume progress is inconsistent")
    if state["completed_steps"] >= plan["recipe"]["max_steps"]:
        raise ValueError("Scene resume needs a larger target step count")
    return {"weights": head["weights"], "bias": head["bias"], **state}


def execute(plan, rows, checkpoint, out, *, resume=None):
    """Encode real local image pairs and train the head, with honest coverage."""
    import numpy as np

    from .provider import clip_role_probabilities, encode_clip_views, load_clip_encoder

    started = time.monotonic()
    deadline = started + plan["recipe"]["max_seconds"]
    out = Path(out)
    active = [row for row in rows if row["split"] != "test"]
    encoder = load_clip_encoder(checkpoint, expected_sha256=plan["backbone_sha256"], device="cpu")
    if encoder.processor_identity.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Scene training/provider feature schemas differ")
    model_load_seconds = time.monotonic() - started
    vectors, baseline = [], []
    pair_seconds = []
    for row in active:
        if time.monotonic() >= deadline:
            raise TimeoutError("Scene encoding exhausted the bounded run before training")
        pair_started = time.monotonic()
        features = encode_clip_views(encoder, *row["images"])
        vector = features.detach().cpu().numpy().reshape(-1).tolist()
        validate_features([vector])
        vectors.append(vector)
        baseline.append(ROLES[int(np.argmax(clip_role_probabilities(encoder, features)))])
        pair_seconds.append(time.monotonic() - pair_started)
    feature_digest = digest(vectors)
    plan["feature_sha256"] = feature_digest
    _write(out / "plan.json", plan)
    _write(
        out / "features.json",
        {
            "schema_version": 1,
            "feature_schema": FEATURE_SCHEMA,
            "backbone_sha256": plan["backbone_sha256"],
            "processor_digest": encoder.processor_digest,
            "feature_sha256": feature_digest,
            "record_ids": [r["record_id"] for r in active],
            "features": vectors,
        },
    )
    train_indices = [i for i, row in enumerate(active) if row["split"] == "train"]
    initial = load_resume(resume, plan, encoder.processor_digest) if resume else None
    last = None

    def save(weights, bias, completed, history):
        nonlocal last
        last = publish_checkpoint(
            out,
            plan,
            encoder.processor_digest,
            {
                "weights": weights.tolist(),
                "bias": bias.tolist(),
                "completed_steps": completed,
                "loss_history": history,
            },
        )

    result = fit(
        [vectors[i] for i in train_indices],
        [active[i]["prediction"]["proposed_role"] for i in train_indices],
        plan["recipe"],
        initial=initial,
        callback=save,
        deadline=deadline,
    )
    if last is None or last.name != f"checkpoint-{result['completed_steps']:06d}":
        last = publish_checkpoint(out, plan, encoder.processor_digest, result)
    verify_checkpoint(last)
    reloaded = read_json(last / "head.json")
    if reloaded["weights"] != result["weights"] or reloaded["bias"] != result["bias"]:
        raise RuntimeError("Reloaded scene parameters differ from the trained checkpoint")
    predicted = probabilities(vectors, reloaded["weights"], reloaded["bias"])
    outcomes = [ROLES[int(index)] for index in predicted.argmax(axis=1)]
    reports = {}
    for split in ("train", "validation"):
        indices = [i for i, row in enumerate(active) if row["split"] == split]
        labels = [active[i]["prediction"]["proposed_role"] for i in indices]
        reports[split] = {
            "learned_head": evaluate(labels, [outcomes[i] for i in indices]),
            "zero_shot_clip": evaluate(labels, [baseline[i] for i in indices]),
        }
        family_scores = {}
        for family in sorted({active[i]["family_id"] for i in indices}):
            members = [i for i in indices if active[i]["family_id"] == family]
            truth = [active[i]["prediction"]["proposed_role"] for i in members]
            family_scores[family] = {
                "learned_head": evaluate(truth, [outcomes[i] for i in members]),
                "zero_shot_clip": evaluate(truth, [baseline[i] for i in members]),
            }
        reports[split]["families"] = family_scores
        for name, offset in (("source_view_removed", 0), ("asset_view_removed", 512)):
            ablated = np.asarray([vectors[i] for i in indices]).copy()
            ablated[:, offset : offset + 512] = 0
            choices = probabilities(ablated, reloaded["weights"], reloaded["bias"], ablation=True)
            reports[split][name] = evaluate(labels, [ROLES[int(i)] for i in choices.argmax(axis=1)])
    report = {
        "schema_version": 1,
        "model_kind": plan["model_kind"],
        "task_model_trained": True,
        "quality_validated": False,
        "base_weights_updated": False,
        "predicts_regions": False,
        "completed_steps": result["completed_steps"],
        "loss_history": result["loss_history"],
        "checkpoint": str(last),
        "head_sha256": sha256(last / "head.json"),
        "adapter_reload_verified": True,
        "head_reload_verified": True,
        "vlm_lora_trained": False,
        "resumed_from": str(resume) if resume else None,
        "backbone_sha256": plan["backbone_sha256"],
        "processor_digest": encoder.processor_digest,
        "processor_identity": encoder.processor_identity,
        "corpus_sha256": plan["corpus_sha256"],
        "feature_sha256": feature_digest,
        "recipe_sha256": plan["recipe_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "latency_scope": (
            "local model load and two-view encode plus zero-shot comparison; "
            "no server or geometry work"
        ),
        "model_load_seconds": model_load_seconds,
        "cold_first_pair_seconds": model_load_seconds + pair_seconds[0],
        "warm_pair_mean_seconds": sum(pair_seconds[1:]) / max(1, len(pair_seconds) - 1),
        "pair_seconds": pair_seconds,
        "numpy_version": np.__version__,
        "splits": reports,
        "predictions": [
            {
                "record_id": row["record_id"],
                "family_id": row["family_id"],
                "split": row["split"],
                "label": row["prediction"]["proposed_role"],
                "head_role": outcomes[i],
                "head_probabilities": predicted[i].tolist(),
                "baseline_role": baseline[i],
            }
            for i, row in enumerate(active)
        ],
    }
    _write(out / "training-report.json", report)
    return report


def supervise_command(command, timeout_seconds):
    """Bound this local CPU worker; it does not launch or stop cloud services."""
    worker = subprocess.Popen(command, start_new_session=os.name == "posix")
    try:
        code = worker.wait(timeout=timeout_seconds)
    except BaseException:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGTERM)
        elif worker.poll() is None:
            worker.terminate()
        try:
            worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(worker.pid, signal.SIGKILL)
            else:
                worker.kill()
            worker.wait(timeout=3)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker-plan":
        task = read_json(sys.argv[2])
        execute(task["plan"], task["rows"], task["checkpoint"], task["out"], resume=task["resume"])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    rows = load_records(args.manifest)
    recipe = {
        key: getattr(args, key)
        for key in ("max_steps", "save_every", "max_seconds", "learning_rate", "l2", "seed")
    }
    plan = prepare_plan(rows, recipe, args.clip_sha256)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError("Scene run output must be a new empty directory")
    _write(args.out / "plan.json", plan)
    if not args.execute:
        return 0
    task = {
        "plan": plan,
        "rows": rows,
        "checkpoint": str(args.clip_checkpoint.resolve()),
        "out": str(args.out.resolve()),
        "resume": str(args.resume.resolve()) if args.resume else None,
    }
    _write(args.out / "worker-plan.json", task)
    supervise_command(
        [
            sys.executable,
            "-m",
            "experiments.scene_model.train_clip",
            "--_worker-plan",
            str((args.out / "worker-plan.json").resolve()),
        ],
        args.max_seconds + 10,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
