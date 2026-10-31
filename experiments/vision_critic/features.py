"""Deterministic rendered-image features for a small CPU ranking baseline.

These handcrafted spatial/color/edge comparisons are not a pretrained vision
encoder. They provide an inexpensive, reproducible training and evaluation
baseline. They cannot establish typography or semantic design quality.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

FEATURE_VERSION = "rendered-handcrafted-v2"
# These exact paths are produced by engine/eval/metrics.py, score.py,
# style/proportions/occlusion.py and engine/layout/focal.py/relationships.py.
# Selection/admission/baseline flags and generation provenance are excluded.
DIAGNOSTICS = (
    "evaluation.total",
    "evaluation.metrics.raw_collisions",
    "evaluation.metrics.collision_free",
    "evaluation.metrics.raw_min_font_px",
    "evaluation.metrics.text_size",
    "evaluation.metrics.raw_line_growth",
    "evaluation.metrics.text_lines_stable",
    "evaluation.metrics.hierarchy",
    "evaluation.metrics.containment",
    "evaluation.metrics.group_integrity",
    "evaluation.metrics.hero",
    "evaluation.metrics.position_affinity",
    "evaluation.metrics.scale_consistency",
    "evaluation.metrics.alignment",
    "evaluation.metrics.balance",
    "evaluation.metrics.margin",
    "evaluation.metrics.raw_coverage",
    "evaluation.metrics.coverage",
    "evaluation.metrics.text_contrast",
    "evaluation.metrics.text_busyness",
    "style.score",
    "relationships.score",
    "proportions.score",
    "proportions.metrics.rigid_group_error",
    "focal.metrics.center_retained",
    "focal.metrics.occupied_balance_error",
    "occlusion.visible_pair_count",
    "occlusion.visibility_retention",
    "occlusion.coverage.complete",
)
FEATURE_NAMES = (
    [f"spatial_color_similarity_{i}" for i in range(48)]
    + [f"color_histogram_similarity_{i}" for i in range(24)]
    + [f"mean_color_similarity_{i}" for i in range(3)]
    + ["edge_similarity", "aspect_ratio_similarity"]
    + [item for name in DIAGNOSTICS for item in (name, name + "_present")]
)


def image_signature(path: str | Path) -> dict:
    with Image.open(path) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds this baseline's 20 megapixel decode limit")
        aspect = image.width / image.height
        rgb = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float64) / 255.0
        spatial = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float64) / 255.0
    histograms = np.concatenate(
        [
            np.histogram(rgb[:, :, channel], bins=8, range=(0, 1))[0] / (32 * 32)
            for channel in range(3)
        ]
    )
    gray = rgb.mean(axis=2)
    edge = (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2
    return {
        "spatial": spatial.ravel(),
        "histogram": histograms,
        "mean": rgb.mean(axis=(0, 1)),
        "edge": float(edge),
        "aspect": aspect,
    }


def candidate_features(
    source: dict, candidate: dict, diagnostics: dict | None = None
) -> np.ndarray:
    diagnostics = diagnostics or {}
    values = [
        *(-np.abs(source["spatial"] - candidate["spatial"])),
        *(-np.abs(source["histogram"] - candidate["histogram"])),
        *(-np.abs(source["mean"] - candidate["mean"])),
        -abs(source["edge"] - candidate["edge"]),
        -abs(math.log(source["aspect"] / candidate["aspect"])),
    ]
    for name in DIAGNOSTICS:
        value = diagnostics
        for component in name.split("."):
            value = value.get(component) if isinstance(value, dict) else None
        present = isinstance(value, (int, float, bool)) and math.isfinite(value)
        values.extend((float(value) if present else 0.0, float(present)))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid rendered feature vector")
    return result


def pair_features(record: dict) -> tuple[np.ndarray, np.ndarray]:
    source = image_signature(record["images"][0])
    candidates = [record["candidate_a"], record["candidate_b"]]
    vectors = [
        candidate_features(
            source, image_signature(record["images"][index + 1]), candidate.get("diagnostics")
        )
        for index, candidate in enumerate(candidates)
    ]
    return vectors[0], vectors[1]


def pair_input_identity(record: dict) -> str:
    """Bind a prediction to verified inputs, independently of human label files."""
    images = []
    for path in record["images"]:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        images.append(digest.hexdigest())
    fields = ("pair_id", "case_id", "source_id", "family_id", "split", "collection_sha256")
    identity = {field: record.get(field) for field in fields}
    identity["candidate_a_id"] = (
        record.get("candidate_a_id") or record["candidate_a"]["candidate_id"]
    )
    identity["candidate_b_id"] = (
        record.get("candidate_b_id") or record["candidate_b"]["candidate_id"]
    )
    identity["image_sha256"] = images
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def full_corpus_provenance(records: list[dict]) -> dict:
    return {
        field: sorted({record[field] for record in records if record.get(field)})
        for field in ("collection_sha256", "corpus_sha256", "labels_sha256")
    }


def corpus_identity(records: list[dict]) -> str:
    """Bind non-test input bytes and annotations; permit later held-out labels.

    Global labels/corpus digests are deliberately excluded: appending a test
    judgment must not change a frozen train/validation recipe or checkpoint.
    The frozen collection hash still binds every design JSON and resource claim.
    """
    rows = [
        {
            "input_sha256": pair_input_identity(record),
            "decision": record["decision"],
            "issues": record.get("issues", []),
            "annotator": record.get("annotator"),
        }
        for record in records
        if record["split"] != "test"
    ]
    encoded = json.dumps(
        sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
