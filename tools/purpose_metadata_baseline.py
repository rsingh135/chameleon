"""TRAIN-only metadata baseline; no image reads, neural execution, or gold scoring.

Features and English text-pattern flags are fixed before evaluation. Identical
metadata necessarily predicts identically after pixel blanking/replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from experiments.visual_finetune.data import PURPOSE_CLASSES

METHOD = "train_metadata_nearest_centroid_v1"
FEATURE_VERSION = "purpose-context-metadata-v1"
REQUEST_PROTOCOL = "visual-adapter-label-free-evaluation-v1"
FIT_PROTOCOL = "purpose-metadata-nearest-centroid-fit-v1"
MARKER = "Element context, data only: "
FEATURES = (
    "type_text",
    "type_image",
    "type_shape",
    "x",
    "y",
    "width",
    "height",
    "area",
    "log_aspect",
    "paint_order_fraction",
    "log_text_length",
    "url",
    "email",
    "phone",
    "cta",
)
ANSWER_KEYS = {
    "label",
    "labels",
    "answer",
    "answers",
    "gold",
    "target",
    "expected",
    "canonical_label",
    "label_origin",
}
SHA = re.compile(r"[0-9a-f]{64}\Z")
URL = re.compile(r"https?://|www\.|\b[\w.-]+\.(?:com|org|net|edu|gov|io|co)\b", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)\+?(?:\d[\s().-]*){7,15}(?!\w)")
CTA = re.compile(
    r"\b(?:buy|shop|register|join|book|subscribe|sign\s+up|learn\s+more|contact|visit|call|donate|reserve|get\s+tickets|rsvp)\b",
    re.I,
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _loads(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate metadata JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("Metadata JSON must be finite")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)  # Also rejects exponent overflow such as 1e999.
    return value


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4096


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def features(prompt):
    """Parse the FIRST static marker; markers inside text_data stay literal data."""
    if not isinstance(prompt, str) or len(prompt) > 32768 or MARKER not in prompt:
        raise ValueError("Expected the fixed purpose-context prompt marker")
    context = _loads(prompt.split(MARKER, 1)[1])
    if not isinstance(context, dict) or set(context) != {
        "bbox_normalized",
        "type",
        "text_data",
        "paint_order_index",
        "element_count",
    }:
        raise ValueError("Purpose context requires exactly the raw model metadata keys")
    box, kind, text = context["bbox_normalized"], context["type"], context["text_data"]
    count, rank = context["element_count"], context["paint_order_index"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(not _finite(v) or abs(v) > 1_000_000 for v in box)
        or min(box[2:]) < 0
        or kind not in ("text", "image", "shape")
        or not isinstance(text, str)
        or len(text) > 240
        or type(count) is not int
        or type(rank) is not int
        or not 0 <= rank < count <= 1_000_000
    ):
        raise ValueError("Invalid or unbounded purpose context metadata")
    x, y, width, height = map(float, box)
    return [float(kind == name) for name in ("text", "image", "shape")] + [
        x,
        y,
        width,
        height,
        width * height,
        math.log(max(width, 1e-12)) - math.log(max(height, 1e-12)),
        rank / max(1, count - 1),
        math.log1p(len(text)),
        *(float(pattern.search(text) is not None) for pattern in (URL, EMAIL, PHONE, CTA)),
    ]


def _records(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("Expected bounded nonempty metadata records")
    if any(
        not isinstance(row, dict)
        or not _text(row.get("record_id"))
        or not _text(row.get("family_id"))
        for row in rows
    ):
        raise ValueError("Every metadata record requires record_id and family_id")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate metadata record ID")
    return sorted(rows, key=lambda row: row["record_id"])


def fit(train_records):
    """Reject mixed splits/controls; never silently filter a combined dataset."""
    rows = _records(train_records)
    if any(
        row.get("split") != "train"
        or row.get("control") != "original"
        or row.get("label") not in PURPOSE_CLASSES
        for row in rows
    ):
        raise ValueError("Fit requires TRAIN-only original records with purpose class labels")
    matrix = [features(row.get("prompt")) for row in rows]
    means = [math.fsum(column) / len(rows) for column in zip(*matrix, strict=True)]
    scales = [
        math.sqrt(math.fsum((row[i] - mean) ** 2 for row in matrix) / len(rows)) or 1.0
        for i, mean in enumerate(means)
    ]
    standardized = [
        [(v - mean) / scale for v, mean, scale in zip(row, means, scales, strict=True)]
        for row in matrix
    ]
    counts = Counter(row["label"] for row in rows)
    centroids = {}
    for label in PURPOSE_CLASSES:
        selected = [
            vector for row, vector in zip(rows, standardized, strict=True) if row["label"] == label
        ]
        if selected:
            centroids[label] = [
                math.fsum(column) / len(selected) for column in zip(*selected, strict=True)
            ]
    result = {
        "protocol": FIT_PROTOCOL,
        "method": METHOD,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURES),
        "classes": list(PURPOSE_CLASSES),
        "train_record_count": len(rows),
        "train_records_sha256": digest(rows),
        "train_family_ids": sorted({row["family_id"] for row in rows}),
        "class_counts": {label: counts[label] for label in PURPOSE_CLASSES},
        "means": means,
        "scales": scales,
        "centroids": centroids,
        "model_executed": False,
        "image_pixels_used": False,
    }
    return {**result, "fit_sha256": digest(result)}


def _no_answers(value):
    if isinstance(value, dict):
        if ANSWER_KEYS & value.keys():
            raise ValueError("Prediction requests must be label-free and contain no answers")
        for item in value.values():
            _no_answers(item)
    elif isinstance(value, list):
        for item in value:
            _no_answers(item)


def predict(fitted, request, *, request_sha256):
    """Produce metadata-only decisions bound to the caller's raw request SHA."""
    if (
        not isinstance(fitted, dict)
        or fitted.get("protocol") != FIT_PROTOCOL
        or fitted.get("method") != METHOD
        or fitted.get("feature_version") != FEATURE_VERSION
        or fitted.get("feature_names") != list(FEATURES)
        or fitted.get("classes") != list(PURPOSE_CLASSES)
        or fitted.get("fit_sha256")
        != digest({k: v for k, v in fitted.items() if k != "fit_sha256"})
    ):
        raise ValueError("Frozen metadata fit identity differs")
    _no_answers(request)
    if (
        not isinstance(request, dict)
        or request.get("protocol") != REQUEST_PROTOCOL
        or request.get("task") != "purpose"
        or type(request.get("purpose_views", 2)) is not int
        or request.get("purpose_views", 2) not in (2, 3)
        or not isinstance(request_sha256, str)
        or SHA.fullmatch(request_sha256) is None
    ):
        raise ValueError("Expected a bound label-free two/three-view purpose request")
    means, scales, centroids = (fitted[key] for key in ("means", "scales", "centroids"))
    if (
        not isinstance(centroids, dict)
        or not centroids
        or set(centroids) - set(PURPOSE_CLASSES)
        or any(
            not isinstance(vector, list)
            or len(vector) != len(FEATURES)
            or not all(_finite(v) for v in vector)
            for vector in [means, scales, *centroids.values()]
        )
        or any(scale <= 0 for scale in scales)
    ):
        raise ValueError("Invalid frozen metadata scaler or centroids")
    rows = _records(request.get("records"))
    predictions = []
    for row in rows:
        if not _text(row.get("control")):
            raise ValueError("Prediction request requires explicit control metadata")
        images = row.get("images")
        if (
            not isinstance(images, list)
            or len(images) != request.get("purpose_views", 2)
            or any(
                not isinstance(image, dict)
                or not _text(image.get("path"))
                or not isinstance(image.get("sha256"), str)
                or SHA.fullmatch(image["sha256"]) is None
                for image in images
            )
        ):
            raise ValueError("Request must retain ordered image references; pixels are never read")
        vector = [
            (v - mean) / scale
            for v, mean, scale in zip(features(row.get("prompt")), means, scales, strict=True)
        ]
        distances = {
            label: math.hypot(*(v - c for v, c in zip(vector, center, strict=True)))
            for label, center in centroids.items()
        }
        if not all(math.isfinite(value) for value in distances.values()):
            raise ValueError("Metadata prediction distance is nonfinite")
        label = min(distances, key=lambda item: (distances[item], PURPOSE_CLASSES.index(item)))
        predictions.append(
            {
                "record_id": row["record_id"],
                "family_id": row["family_id"],
                "control": row.get("control"),
                "prediction": label,
                "request_sha256": request_sha256,
                "method": METHOD,
                "model_executed": False,
            }
        )
    return {
        "protocol": "purpose-metadata-baseline-predictions-v1",
        "method": METHOD,
        "feature_version": FEATURE_VERSION,
        "fit_sha256": fitted["fit_sha256"],
        "request_sha256": request_sha256,
        "model_executed": False,
        "image_pixels_used": False,
        "quality_approved": False,
        "promotion_eligible": False,
        "predictions": predictions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "request", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Metadata baseline output must be fresh")

    def read(path):
        with Path(path).open("rb") as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Metadata input exceeds its byte budget")
        return data

    training = [
        _loads(line) for line in read(args.train).decode("utf-8-sig").splitlines() if line.strip()
    ]
    request_bytes = read(args.request)
    fitted = fit(training)
    result = predict(
        fitted,
        _loads(request_bytes.decode("utf-8-sig")),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
    )
    out.mkdir(parents=True)
    (out / "fit.json").write_bytes(canonical(fitted) + b"\n")
    (out / "predictions.json").write_bytes(canonical(result) + b"\n")


if __name__ == "__main__":
    main()
