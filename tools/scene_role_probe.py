"""One fixed CPU hybrid role-head experiment; no engine/provider edits or downloads.

The frozen original CLIP head supplies five logits. Ten source-only geometric
features condition a new small head. Training uses the original family split;
validation never selects a recipe, normalizer, threshold or checkpoint.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

from experiments.scene_model.dataset import (
    ROOT,
    _reserves,
    canonical,
    dataset_summary,
    digest,
    load_records,
    read_json,
    sha256,
)
from experiments.scene_model.provider import CLIP_B32_SHA256, FEATURE_SCHEMA
from experiments.scene_model.schema import ROLE_NAMES
from experiments.scene_model.train_clip import evaluate, supervise_command, validate_features

SCHEMA = "clip-frozen-logits-source-geometry-v1"
BASE_HEAD_SHA = "cf0c9ce35d166a4e85b13664d361c1f13f911195afbb00c28b251151886ab988"
GEOMETRY_NAMES = (
    "frame_area_ratio",
    "center_x",
    "center_y",
    "log_frame_aspect",
    "log_asset_aspect",
    "visible_alpha_fraction",
    "opaque_fraction",
    "live_text_overlap",
    "paint_rank",
    "canvas_edge_contacts",
)
DIM = len(ROLE_NAMES) + len(GEOMETRY_NAMES)
RECIPE = {
    "seed": 2026,
    "steps": 200,
    "learning_rate": 0.05,
    "l2": 0.01,
    "checkpoint_every": 100,
    "max_seconds": 600,
    "protection_threshold": 0.8,
}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def geometry(source, element, asset):
    """Measured pixels/geometry only: no ID/name strings, labels or target data."""
    from PIL import Image

    width, height = source["canvas"]["width"], source["canvas"]["height"]
    x, y, w, h = (element[key] for key in ("x", "y", "width", "height"))
    if min(width, height, w, h) <= 0 or not all(
        math.isfinite(v) for v in (width, height, x, y, w, h)
    ):
        raise ValueError("Hybrid features need valid finite source geometry")
    with Image.open(asset) as image:
        histogram = image.convert("RGBA").getchannel("A").histogram()
        pixels = image.width * image.height
        aspect = image.width / image.height
    text_area = 0.0
    for other in source["elements"]:
        if other["type"] == "text":
            text_area += max(0, min(x + w, other["x"] + other["width"]) - max(x, other["x"])) * max(
                0, min(y + h, other["y"] + other["height"]) - max(y, other["y"])
            )
    order = sorted(
        enumerate(source["elements"]), key=lambda item: (item[1].get("z_index", 0), item[0])
    )
    rank = next(i for i, (_, item) in enumerate(order) if item["id"] == element["id"])
    contacts = sum(
        (
            abs(x) <= width * 0.01,
            abs(y) <= height * 0.01,
            abs(x + w - width) <= width * 0.01,
            abs(y + h - height) <= height * 0.01,
        )
    )
    return [
        min(4, w * h / (width * height)),
        max(-4, min(4, (x + w / 2) / width)),
        max(-4, min(4, (y + h / 2) / height)),
        max(-5, min(5, math.log(w / h))),
        max(-5, min(5, math.log(aspect))),
        (pixels - histogram[0]) / pixels,
        histogram[255] / pixels,
        min(1, text_area / (w * h)),
        rank / max(1, len(order) - 1),
        contacts / 4,
    ]


def hybrid_features(vectors, geometric, base_head):
    import numpy as np

    vectors = validate_features(vectors)
    weights, bias = np.asarray(base_head["weights"]), np.asarray(base_head["bias"])
    if (
        weights.shape != (5, 1024)
        or bias.shape != (5,)
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("Frozen original head parameters differ from their expected shape")
    measured = np.asarray(geometric, dtype=np.float64)
    if measured.shape != (len(vectors), len(GEOMETRY_NAMES)) or not np.isfinite(measured).all():
        raise ValueError("Hybrid geometry dimensions or finite values differ")
    return np.column_stack((vectors @ weights.T + bias, measured))


def normalize_training(features, train_indices):
    import numpy as np

    features = np.asarray(features, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] != DIM
        or not np.isfinite(features).all()
        or not train_indices
    ):
        raise ValueError("Hybrid normalizer needs finite features and a training split")
    training = features[train_indices]
    mean, scale = training.mean(axis=0), training.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (features - mean) / scale, {"mean": mean.tolist(), "scale": scale.tolist()}


def scores(features, weights, bias):
    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    weights, bias = np.asarray(weights), np.asarray(bias)
    if (
        values.ndim != 2
        or values.shape[1] != DIM
        or weights.shape != (5, DIM)
        or bias.shape != (5,)
    ):
        raise ValueError("Hybrid parameter dimensions differ")
    logits = values @ weights.T + bias
    if not np.isfinite(logits).all():
        raise ValueError("Hybrid prediction is not finite")
    result = np.exp(logits - logits.max(axis=1, keepdims=True))
    return result / result.sum(axis=1, keepdims=True)


def fit(features, labels, *, deadline, checkpoint=None):
    import numpy as np

    vectors = np.asarray(features)
    if (
        len(vectors) != len(labels)
        or not labels
        or any(label not in ROLE_NAMES for label in labels)
    ):
        raise ValueError("Hybrid labels and training features differ")
    target = np.asarray([ROLE_NAMES.index(label) for label in labels])
    counts = Counter(labels)
    sample_weight = np.asarray([1 / (len(counts) * counts[label]) for label in labels])
    weights = np.random.default_rng(RECIPE["seed"]).normal(0, 0.001, (5, DIM))
    bias, history = np.zeros(5), []
    for step in range(1, RECIPE["steps"] + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("Hybrid CPU deadline exceeded; no completed result published")
        predicted = scores(vectors, weights, bias)
        loss = float(
            -(
                np.log(np.maximum(predicted[np.arange(len(target)), target], 1e-12)) * sample_weight
            ).sum()
            + 0.5 * RECIPE["l2"] * (weights * weights).sum()
        )
        residual = predicted.copy()
        residual[np.arange(len(target)), target] -= 1
        residual *= sample_weight[:, None]
        gradient = residual.T @ vectors + RECIPE["l2"] * weights
        if not math.isfinite(loss) or not np.isfinite(gradient).all():
            raise RuntimeError("Hybrid optimizer produced a non-finite value")
        weights -= RECIPE["learning_rate"] * gradient
        bias -= RECIPE["learning_rate"] * residual.sum(axis=0)
        history.append(loss)
        if checkpoint and step % RECIPE["checkpoint_every"] == 0:
            checkpoint(step, weights.tolist(), bias.tolist(), list(history))
    return {
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "loss_history": history,
        "completed_steps": RECIPE["steps"],
    }


def validate_splits(rows):
    """Retain original family/asset boundaries, including explicit reserve exclusions."""
    reserve_ids, reserve_hashes = _reserves()
    for key in ("family_id", "source_id", "source_digest"):
        seen = {}
        for row in rows:
            if row["split"] not in {"train", "validation"}:
                raise ValueError("This fixed probe never consumes final-test records")
            if seen.setdefault(row[key], row["split"]) != row["split"]:
                raise ValueError(f"Hybrid {key} crosses splits")
    assets = {}
    for row in rows:
        if (
            row["source_id"] in reserve_ids
            or row["family_id"] in reserve_ids
            or row["source_digest"] in reserve_hashes
        ):
            raise ValueError("Hybrid probe cannot consume reserve sources")
        if row["supervision_origin"] not in {
            "agent_visual_review",
            "human_review",
            "user_explicit_and_agent_review",
        }:
            raise ValueError("Hybrid probe requires reviewed, nonsynthetic labels")
        if assets.setdefault(row["image_sha256"][1], row["split"]) != row["split"]:
            raise ValueError("Hybrid selected asset bytes cross splits")
    if not all(any(row["split"] == split for row in rows) for split in ("train", "validation")):
        raise ValueError("Hybrid needs both training and validation families")


def prepare_inputs(paths):
    """Bind cached features to the original labels and source/asset byte receipts."""
    rows = load_records(paths["manifest"])
    validate_splits(rows)
    original_report = read_json(paths["training_report"])
    base_head = read_json(paths["base_head"])
    cached = read_json(paths["features"])
    if (
        sha256(paths["base_head"]) != BASE_HEAD_SHA
        or original_report["head_sha256"] != BASE_HEAD_SHA
    ):
        raise ValueError("Frozen original head SHA-256 differs")
    if sha256(paths["base_checkpoint"]) != CLIP_B32_SHA256:
        raise ValueError("Frozen local CLIP backbone SHA-256 differs")
    for artifact in (cached, base_head, original_report):
        if (
            artifact.get("backbone_sha256") != CLIP_B32_SHA256
            or artifact.get("processor_digest") != base_head["processor_digest"]
        ):
            raise ValueError("Cached base/processor identities differ")
    if (
        cached.get("feature_schema") != FEATURE_SCHEMA
        or base_head.get("feature_schema") != FEATURE_SCHEMA
    ):
        raise ValueError("Cached full-view feature schema differs")
    vectors = cached["features"]
    if (
        digest(vectors) != cached.get("feature_sha256")
        or cached["feature_sha256"] != original_report["feature_sha256"]
    ):
        raise ValueError("Cached CLIP feature bytes differ")
    if cached["record_ids"] != [row["record_id"] for row in rows] or len(vectors) != len(rows):
        raise ValueError("Cached feature row identities/order differ")
    expected = [
        {
            "record_id": row["record_id"],
            "split": row["split"],
            "family_id": row["family_id"],
            "source_id": row["source_id"],
            "input_digest": row["input_digest"],
        }
        for row in rows
    ]
    if (
        digest(expected) != original_report["corpus_sha256"]
        or base_head["corpus_sha256"] != original_report["corpus_sha256"]
    ):
        raise ValueError("Original reviewed corpus or labels changed")
    prior = original_report["predictions"]
    if [(p["record_id"], p["label"], p["split"], p["family_id"]) for p in prior] != [
        (row["record_id"], row["prediction"]["proposed_role"], row["split"], row["family_id"])
        for row in rows
    ]:
        raise ValueError("Original baseline prediction identities differ")
    measured = []
    for row in rows:
        source = read_json(Path(row["harness_dir"]) / "source.json")
        asset = Path(row["images"][1])
        if digest(source) != row["source_digest"] or sha256(asset) != row["image_sha256"][1]:
            raise ValueError("Hybrid source/asset bytes changed")
        element = next(e for e in source["elements"] if e["id"] == row["element_id"])
        measured.append(geometry(source, element, asset))
    raw = hybrid_features(vectors, measured, base_head)
    normalized, normalizer = normalize_training(
        raw, [i for i, row in enumerate(rows) if row["split"] == "train"]
    )
    bindings = {
        key: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
        for key, path in paths.items()
    }
    bindings["probe_code"] = {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)}
    return {
        "rows": rows,
        "features": normalized,
        "raw_features": raw.tolist(),
        "normalizer": normalizer,
        "bindings": bindings,
        "base_head": base_head,
        "original_report": original_report,
    }


def publish(out, model, state):
    directory = Path(out) / "published" / f"checkpoint-{state['completed_steps']:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "model.json", model)
    write_new(directory / "training-state.json", state)
    files = {
        name: {"sha256": sha256(directory / name), "size": (directory / name).stat().st_size}
        for name in ("model.json", "training-state.json")
    }
    write_new(directory / "checkpoint-manifest.json", {"schema_version": 1, "files": files})
    return directory


def load_model(path, expected_sha):
    import numpy as np

    if sha256(path) != expected_sha:
        raise ValueError("Explicit hybrid model SHA-256 differs")
    model = read_json(path)
    if (
        model.get("feature_schema") != SCHEMA
        or model.get("roles") != list(ROLE_NAMES)
        or model.get("geometry_names") != list(GEOMETRY_NAMES)
    ):
        raise ValueError("Hybrid feature/role schema differs")
    if model["bindings"]["probe_code"]["sha256"] != sha256(__file__):
        raise ValueError("Hybrid preprocessing/optimizer code changed")
    for key in ("base_head", "base_checkpoint"):
        binding = model["bindings"][key]
        if sha256(binding["path"]) != binding["sha256"]:
            raise ValueError("Hybrid frozen dependency bytes differ")
    mean, scale = np.asarray(model["normalizer"]["mean"]), np.asarray(model["normalizer"]["scale"])
    if (
        mean.shape != (DIM,)
        or scale.shape != (DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Hybrid normalizer is invalid")
    scores(np.zeros((1, DIM)), model["weights"], model["bias"])
    return model


def execute(plan_path):
    import numpy as np

    started = time.monotonic()
    packet = read_json(plan_path)
    paths, out = packet["paths"], Path(packet["out"]).resolve()
    if (
        packet.get("recipe") != RECIPE
        or not out.is_relative_to(ROOT / "output")
        or Path(plan_path).resolve() != out / "plan.json"
    ):
        raise ValueError("Hybrid worker needs its fixed bounded recipe and output plan")
    data = prepare_inputs(paths)
    if (
        data["bindings"] != packet["bindings"]
        or digest(data["raw_features"]) != packet["raw_feature_sha256"]
        or data["normalizer"] != packet["normalizer"]
    ):
        raise ValueError("Hybrid execution inputs changed after the frozen plan")
    rows, features = data["rows"], data["features"]
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    fixed = {
        "schema_version": 1,
        "model_kind": "frozen_clip_logits_geometry_role_head",
        "feature_schema": SCHEMA,
        "roles": list(ROLE_NAMES),
        "geometry_names": list(GEOMETRY_NAMES),
        "normalizer": data["normalizer"],
        "bindings": data["bindings"],
        "recipe": RECIPE,
        "processor_digest": data["base_head"]["processor_digest"],
        "base_weights_updated": False,
        "confidence_calibrated": False,
        "predicts_regions": False,
        "quality_validated": False,
    }
    latest = None

    def checkpoint(step, weights, bias, history):
        nonlocal latest
        latest = publish(
            out,
            {**fixed, "weights": weights, "bias": bias},
            {
                "completed_steps": step,
                "loss_history": history,
                "optimizer": "class_balanced_full_batch_gradient_descent",
                "momentum": False,
            },
        )

    trained = fit(
        features[train],
        [rows[i]["prediction"]["proposed_role"] for i in train],
        deadline=started + RECIPE["max_seconds"] - 5,
        checkpoint=checkpoint,
    )
    final_data = prepare_inputs(paths)
    if (
        final_data["bindings"] != data["bindings"]
        or digest(final_data["raw_features"]) != packet["raw_feature_sha256"]
    ):
        raise RuntimeError("Hybrid inputs changed during the experiment")
    model_path = latest / "model.json"
    model = load_model(model_path, sha256(model_path))
    predicted = scores(features, model["weights"], model["bias"])
    outcomes = [ROLE_NAMES[int(i)] for i in predicted.argmax(axis=1)]
    reports = {}
    for split in ("train", "validation"):
        indices = [i for i, row in enumerate(rows) if row["split"] == split]
        truth = [rows[i]["prediction"]["proposed_role"] for i in indices]
        reports[split] = {
            "hybrid_head": evaluate(truth, [outcomes[i] for i in indices]),
            "previous_head": evaluate(
                truth, [data["original_report"]["predictions"][i]["head_role"] for i in indices]
            ),
            "zero_shot_clip": evaluate(
                truth, [data["original_report"]["predictions"][i]["baseline_role"] for i in indices]
            ),
            "always_background": evaluate(truth, ["background"] * len(indices)),
        }
        action = [
            i
            for i in indices
            if outcomes[i] in {"product", "logo"}
            and max(predicted[i]) >= RECIPE["protection_threshold"]
        ]
        reports[split]["eligible_protection_predictions"] = {
            "count": len(action),
            "correct_roles": sum(
                outcomes[i] == rows[i]["prediction"]["proposed_role"] for i in action
            ),
            "not_actual_candidate_vetoes": True,
            "threshold": 0.8,
        }
        reports[split]["families"] = {
            family: evaluate(
                [
                    rows[i]["prediction"]["proposed_role"]
                    for i in indices
                    if rows[i]["family_id"] == family
                ],
                [outcomes[i] for i in indices if rows[i]["family_id"] == family],
            )
            for family in sorted({rows[i]["family_id"] for i in indices})
        }
    report = {
        **fixed,
        "task_head_trained": True,
        "completed_steps": trained["completed_steps"],
        "loss_history": trained["loss_history"],
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "reload_predictions_identical": bool(
            np.array_equal(predicted, scores(features, trained["weights"], trained["bias"]))
        ),
        "elapsed_seconds": time.monotonic() - started,
        "numpy_version": np.__version__,
        "dataset": dataset_summary(rows),
        "splits": reports,
        "reserve_sources_used": [],
        "actual_resize_evaluation": (
            "pending; eligible predictions are not actual activations/vetoes"
        ),
        "promoted": False,
        "predictions": [
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "family_id": row["family_id"],
                "split": row["split"],
                "label": row["prediction"]["proposed_role"],
                "role": outcomes[i],
                "probabilities": predicted[i].tolist(),
            }
            for i, row in enumerate(rows)
        ],
    }
    write_new(out / "training-report.json", report)
    return report


def predict(request, model_path, expected_sha):
    import numpy as np

    from engine.scene_model import validate_model_response
    from experiments.scene_model.provider import (
        encode_clip_views,
        load_clip_encoder,
        prepare_request,
    )

    model = load_model(model_path, expected_sha)
    before = digest(request)
    source_path, selected = prepare_request(request)
    base_head = read_json(model["bindings"]["base_head"]["path"])
    encoder = None
    if selected:
        encoder = load_clip_encoder(
            model["bindings"]["base_checkpoint"]["path"], expected_sha256=CLIP_B32_SHA256, threads=4
        )
        if encoder.processor_digest != model["processor_digest"]:
            raise ValueError("Hybrid inference processor differs from training")
    images = []
    for element, evidence, asset in selected:
        vectors = encode_clip_views(encoder, source_path, asset).detach().cpu().numpy()
        raw = hybrid_features(vectors, [geometry(request["source"], element, asset)], base_head)
        normalized = (raw - np.asarray(model["normalizer"]["mean"])) / np.asarray(
            model["normalizer"]["scale"]
        )
        probabilities = scores(normalized, model["weights"], model["bias"])[0]
        best = int(probabilities.argmax())
        images.append(
            {
                "element_id": element["id"],
                "asset_sha256": evidence["asset_sha256"],
                "proposed_role": ROLE_NAMES[best],
                "confidence": float(probabilities[best]),
                "protected_regions": [],
                "rationale": (
                    "Experimental frozen-CLIP-logit/source-geometry task head. "
                    "Uncalibrated; no region detector or crop authority."
                ),
            }
        )
    prepare_request(request)
    if digest(request) != before:
        raise RuntimeError("Hybrid inference changed its bound source request")
    return validate_model_response(
        {
            "schema_version": 1,
            "source_digest": request["source_digest"],
            "scene_digest": request["scene_digest"],
            "images": images,
        },
        request["deterministic_scene"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    defaults = {
        "manifest": "output/scene-model-source-inspection/agent-reviewed-scene-manifest.json",
        "features": "output/scene-model-clip-stable-80-resume/features.json",
        "base_head": (
            "output/scene-model-clip-stable-80-resume/published/checkpoint-000080/head.json"
        ),
        "training_report": "output/scene-model-clip-stable-80-resume/training-report.json",
        "base_checkpoint": str(Path.home() / ".cache/clip/ViT-B-32.pt"),
    }
    for key, default in defaults.items():
        train.add_argument("--" + key.replace("_", "-"), type=Path, default=Path(default))
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--execute", action="store_true")
    worker = sub.add_parser("_worker")
    worker.add_argument("--plan", type=Path, required=True)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--model-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "_worker":
        report = execute(args.plan)
        print(canonical({"completed_steps": report["completed_steps"], "promoted": False}))
    elif args.action == "predict":
        from experiments.scene_model.provider import MAX_REQUEST_BYTES, _parse

        text = sys.stdin.read(MAX_REQUEST_BYTES + 1)
        if len(text.encode()) > MAX_REQUEST_BYTES:
            raise ValueError("Hybrid request exceeds its byte budget")
        print(canonical(predict(_parse(text), args.model, args.model_sha256)))
    else:
        out = args.out.resolve()
        if not out.is_relative_to(ROOT / "output"):
            raise ValueError("Hybrid experiment output must stay under repository output/")
        paths = {key: str(getattr(args, key).resolve()) for key in defaults}
        data = prepare_inputs(paths)
        out.mkdir(parents=True, exist_ok=False)
        packet = {
            "schema_version": 1,
            "paths": paths,
            "bindings": data["bindings"],
            "out": str(out),
            "recipe": RECIPE,
            "normalizer": data["normalizer"],
            "raw_feature_sha256": digest(data["raw_features"]),
            "dataset": dataset_summary(data["rows"]),
        }
        write_new(out / "plan.json", packet)
        if args.execute:
            supervise_command(
                [
                    sys.executable,
                    "-m",
                    "tools.scene_role_probe",
                    "_worker",
                    "--plan",
                    str(out / "plan.json"),
                ],
                RECIPE["max_seconds"],
            )
        else:
            print(
                canonical(
                    {"planned": True, "dataset": packet["dataset"], "training_executed": False}
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
