"""Score saved purpose logits against separate local gold without model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.visual_finetune.data import PURPOSE_CLASSES

SHA = re.compile(r"[0-9a-f]{64}\Z")
REQUEST_PROTOCOL = "visual-adapter-label-free-evaluation-v1"
GOLD_PROTOCOL = "visual-purpose-evaluation-gold-v1"
FREEZE_PROTOCOL = "visual-purpose-evaluation-freeze-v1"
SCORING = "visual_prefix_single_token_closed_class_logits_not_generated_prose"
MISSING, INVALID = "__missing__", "__invalid__"


def _hash(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4096


def _json(path, *, prediction=False, expected_sha=None):
    with Path(path).open("rb") as stream:
        data = stream.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Purpose evaluation JSON exceeds its byte budget")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Frozen evaluation file SHA-256 changed")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate evaluation JSON key")
            result[key] = value
        return result

    def constant(value):
        if prediction:
            return float(value)  # Nonfinite prediction vectors become invalid rows below.
        raise ValueError("Evaluation metadata must contain only finite JSON values")

    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("Purpose evaluation artifacts must be JSON objects")
    return value, digest


def _indexed(rows, name):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError(name + " requires bounded nonempty records")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not _text(row.get("record_id")):
            raise ValueError(name + " requires explicit record IDs")
        if row["record_id"] in result:
            raise ValueError(name + " contains duplicate record IDs")
        result[row["record_id"]] = row
    return result


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _valid_prediction(row, request):
    classes = list(PURPOSE_CLASSES)
    if row.get("family_id") != request.get("family_id", "independent") or (
        row.get("control") != request.get("control", "independent")
    ):
        return "Prediction family or control differs from its request"
    if row.get("input_image_sha256") != [image["sha256"] for image in request["images"]]:
        return "Prediction image identities or image order differ from its request"
    if row.get("label") is not None:
        return "Independent prediction unexpectedly contains a gold label"
    scores, probabilities = row.get("closed_scores"), row.get("closed_probabilities")
    if any(
        not isinstance(vector, dict) or set(vector) != set(classes)
        for vector in (scores, probabilities)
    ):
        return "Prediction must contain the complete fixed class vectors"
    if not all(_finite(value) for vector in (scores, probabilities) for value in vector.values()):
        return "Prediction class vectors must be finite numbers"
    if any(not 0 <= value <= 1 for value in probabilities.values()) or not math.isclose(
        sum(probabilities.values()), 1.0, abs_tol=1e-5, rel_tol=1e-5
    ):
        return "Prediction probabilities must be normalized"
    prediction = row.get("prediction")
    if (
        prediction not in classes
        or prediction != max(classes, key=lambda key: scores[key])
        or probabilities[prediction] != max(probabilities.values())
    ):
        return "Prediction is outside the class vocabulary or differs from its argmax"
    maximum = max(scores.values())
    weights = {key: math.exp(float(scores[key]) - float(maximum)) for key in classes}
    total = sum(weights.values())
    if any(
        not math.isclose(probabilities[key], weights[key] / total, abs_tol=1e-5, rel_tol=1e-5)
        for key in classes
    ):
        return "Prediction probabilities differ from its closed logits"
    if (
        type(row.get("raw_next_token_is_allowed")) is not bool
        or not isinstance(row.get("raw_next_token"), str)
        or type(row.get("raw_next_token_id")) is not int
        or row["raw_next_token_id"] < 0
    ):
        return "Prediction raw-token metadata is malformed"
    raw = row["raw_next_token"].strip()
    if row["raw_next_token_is_allowed"] and (raw not in classes or scores[raw] != maximum):
        return "Allowed global raw token must identify a maximal closed class"
    return None


def _metrics(gold, states):
    classes = list(PURPOSE_CLASSES)
    confusion = {label: dict.fromkeys([*classes, MISSING, INVALID], 0) for label in classes}
    histogram, valid, missing, invalid, abstain, raw_allowed, raw_correct = (
        Counter(),
        0,
        0,
        0,
        0,
        0,
        0,
    )
    for identity, annotation in gold.items():
        state = states[identity]
        outcome = (
            state["prediction"]
            if state["status"] == "valid"
            else (MISSING if state["status"] == "missing" else INVALID)
        )
        confusion[annotation["label"]][outcome] += 1
        if outcome == MISSING:
            missing += 1
        elif outcome == INVALID:
            invalid += 1
        else:
            valid += 1
            histogram[outcome] += 1
            abstain += outcome == "unknown"
            raw_allowed += state["raw_allowed"]
            raw_correct += state["raw_allowed"] and state["raw_word"] == annotation["label"]
    per_class = {}
    for label in classes:
        support = sum(confusion[label].values())
        predicted = sum(confusion[other][label] for other in classes)
        tp = confusion[label][label]
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "true_positive": tp,
            "false_positive": predicted - tp,
            "false_negative": support - tp,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + predicted) if support + predicted else None,
        }
    supported = [item for item in per_class.values() if item["support"]]
    n = len(gold)
    labels = Counter(row["label"] for row in gold.values())
    majority = max(classes, key=lambda key: labels[key]) if n else None
    return {
        "gold_count": n,
        "valid_count": valid,
        "invalid_count": invalid,
        "missing_count": missing,
        "abstain_count": abstain,
        "qualified_prediction_count": valid - abstain,
        "accuracy_all_gold": sum(confusion[label][label] for label in classes) / n if n else None,
        "macro_f1_supported": sum(item["f1"] for item in supported) / len(supported)
        if supported
        else None,
        "macro_recall_supported": sum(item["recall"] for item in supported) / len(supported)
        if supported
        else None,
        "supported_class_count": len(supported),
        "class_count": len(classes),
        "per_class": per_class,
        "confusion": confusion,
        "prediction_histogram": dict(histogram),
        "dominant_prediction_fraction": max(histogram.values()) / valid if valid else None,
        "raw_allowed_rate_valid": raw_allowed / valid if valid else None,
        "strict_raw_token_accuracy_all_gold": raw_correct / n if n else None,
        "majority_baseline": {
            "label": majority,
            "accuracy_all_gold": labels[majority] / n if n else None,
        },
    }


def score(request_path, predictions_path, receipt_path, gold_path, freeze_path):
    """Join local gold after verifying immutable request and saved-output bindings."""
    request, request_sha = _json(request_path)
    predictions, prediction_sha = _json(predictions_path, prediction=True)
    receipt, receipt_sha = _json(receipt_path)
    freeze, freeze_sha = _json(freeze_path)
    if request.get("protocol") != REQUEST_PROTOCOL or request.get("task") != "purpose":
        raise ValueError("Expected a label-free purpose request")
    purpose_views = request.get("purpose_views", 2)
    if type(purpose_views) is not int or purpose_views not in (2, 3):
        raise ValueError("Purpose request must explicitly declare two or three views")
    if freeze.get("protocol") != FREEZE_PROTOCOL or freeze.get("request_sha256") != request_sha:
        raise ValueError("Purpose evaluation freeze differs from the request")
    if (
        receipt.get("execution_success") is not True
        or receipt.get("task") != "purpose"
        or (
            receipt.get("request_sha256") != request_sha
            or receipt.get("prediction_file_sha256") != prediction_sha
        )
    ):
        raise ValueError("Purpose execution receipt differs from the request or prediction bytes")
    for key in ("adapter_sha256", "adapter_config_sha256"):
        if not _hash(freeze.get(key)) or receipt.get(key) != freeze[key]:
            raise ValueError("Purpose adapter or configuration differs from its local freeze")
    if not _hash(freeze.get("gold_sha256")):
        raise ValueError("Purpose evaluation freeze must bind local gold SHA-256")
    if predictions.get("scoring") != SCORING:
        raise ValueError("Expected saved closed-class visual-prefix logits, not generated prose")
    requests = _indexed(request.get("records"), "Label-free request")
    for row in requests.values():
        if any(key in row for key in ("label", "target", "expected", "answer")):
            raise ValueError("Worker request must not contain local gold answers")
        images = row.get("images")
        if (
            not isinstance(images, list)
            or len(images) != purpose_views
            or any(
                not isinstance(image, dict)
                or not _hash(image.get("sha256"))
                or not _text(image.get("path"))
                for image in images
            )
            or not _text(row.get("prompt"))
            or not _text(row.get("family_id", "independent"))
            or not _text(row.get("control", "independent"))
        ):
            raise ValueError(
                "Purpose request needs its exact declared ordered images and raw metadata"
            )
    gold, gold_sha = _json(gold_path, expected_sha=freeze["gold_sha256"])
    if (
        gold.get("protocol") != GOLD_PROTOCOL
        or gold.get("request_sha256") != request_sha
        or not _text(gold.get("label_scope"))
    ):
        raise ValueError("Local purpose gold differs from the frozen request or scope")
    annotations = _indexed(gold.get("records"), "Local gold")
    for identity, row in annotations.items():
        parent = row.get("parent_record_id")
        if (
            identity not in requests
            or row.get("label") not in PURPOSE_CLASSES
            or not _text(row.get("label_origin"))
            or parent is not None
            and (not _text(parent) or parent == identity or parent not in requests)
        ):
            raise ValueError("Local gold contains unknown IDs, labels, origins or control parents")
    rows = predictions.get("predictions")
    if not isinstance(rows, list) or len(rows) > 10000:
        raise ValueError("Saved predictions require a bounded row list")
    states = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not _text(row.get("record_id"))
            or row["record_id"] not in requests
        ):
            raise ValueError("Saved prediction contains an unknown record ID")
        identity = row["record_id"]
        if identity in states:
            raise ValueError("Saved predictions contain duplicate record IDs")
        error = _valid_prediction(row, requests[identity])
        states[identity] = {
            "record_id": identity,
            "status": "invalid" if error else "valid",
            "error": error,
            "prediction": row.get("prediction") if not error else None,
            "raw_allowed": row.get("raw_next_token_is_allowed") is True,
            "raw_word": row.get("raw_next_token", "").strip() if not error else None,
            "probabilities": row.get("closed_probabilities") if not error else None,
        }
    for identity in requests.keys() - states.keys():
        states[identity] = {
            "record_id": identity,
            "status": "missing",
            "error": "Missing prediction",
            "prediction": None,
            "raw_allowed": False,
            "raw_word": None,
            "probabilities": None,
        }
    controls = sorted({requests[key].get("control", "independent") for key in annotations})
    paired = defaultdict(list)
    for identity, annotation in annotations.items():
        parent = annotation.get("parent_record_id")
        if parent is not None:
            paired[requests[identity].get("control", "independent")].append((parent, identity))
    pair_metrics = {}
    for condition, pairs in paired.items():
        comparable = [
            (states[a], states[b])
            for a, b in pairs
            if states[a]["status"] == states[b]["status"] == "valid"
        ]
        pair_metrics[condition] = {
            "pair_count": len(pairs),
            "comparable_count": len(comparable),
            "prediction_consistency": sum(a["prediction"] == b["prediction"] for a, b in comparable)
            / len(comparable)
            if comparable
            else None,
            "mean_probability_l1": sum(
                sum(
                    abs(a["probabilities"][key] - b["probabilities"][key])
                    for key in PURPOSE_CLASSES
                )
                for a, b in comparable
            )
            / len(comparable)
            if comparable
            else None,
            "scope": "Control sensitivity only; no fabricated label or quality approval",
        }
    return {
        "schema_version": 1,
        "protocol": "visual-purpose-offline-score-v1",
        "classes": list(PURPOSE_CLASSES),
        "purpose_views": purpose_views,
        "label_scope": gold["label_scope"],
        "bindings": {
            "request_sha256": request_sha,
            "prediction_file_sha256": prediction_sha,
            "inference_receipt_sha256": receipt_sha,
            "gold_sha256": gold_sha,
            "freeze_sha256": freeze_sha,
            "adapter_sha256": freeze["adapter_sha256"],
            "adapter_config_sha256": freeze["adapter_config_sha256"],
        },
        "metrics": _metrics(annotations, states),
        "per_control": {
            condition: _metrics(
                {
                    key: row
                    for key, row in annotations.items()
                    if requests[key].get("control", "independent") == condition
                },
                states,
            )
            for condition in controls
        },
        "paired_controls": dict(pair_metrics),
        "row_status": [
            {key: value for key, value in state.items() if key != "probabilities"}
            for identity, state in sorted(states.items())
            if identity in annotations
        ],
        "qualification_scope": "Receipt-bound valid non-abstaining closed-class predictions only",
        "receipt_evidence_verified": True,
        "cold_load_independently_verified": False,
        "inference_executed": False,
        "quality_approved": False,
        "promotion_eligible": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "predictions", "receipt", "gold", "freeze", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = score(args.request, args.predictions, args.receipt, args.gold, args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
