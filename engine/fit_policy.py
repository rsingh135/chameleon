"""A small trained layout policy, not a vision-language model or aesthetic judge.

Synthetic fit/readability comparisons supervise a pairwise logistic ranker. It
selects existing trusted candidate IDs; the executor owns geometry and all gates.
The implementation uses only the standard library and loads no model runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median

FEATURE_VERSION = "fit-features-v1"
LABEL_KIND = "synthetic_geometry_teacher_v1"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models/fit_policy/default.json"
FEATURE_NAMES = (
    "source_log_aspect",
    "target_log_aspect",
    "aspect_change",
    "foreground_width",
    "foreground_height",
    "foreground_bbox_area",
    "foreground_area",
    "foreground_density",
    "center_x_error",
    "center_y_error",
    "edge_margin",
    "overflow_fraction",
    "text_fraction",
    "text_count",
    "image_count",
    "column_count",
    "column_width_use",
    "wide_column_use",
    "minimum_font",
    "median_font",
    "font_scale",
    "font_scale_consistency",
    "hierarchy_retention",
    "reading_order_retention",
    "unit_scale_consistency",
    "unit_attachment",
    "image_proportion_retention",
    "text_frame_fill",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_digest(source):
    return hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value, fallback=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value) if math.isfinite(value) else fallback


def _box(element):
    return tuple(_finite(element.get(key)) for key in ("x", "y", "width", "height"))


def _foreground(source):
    width, height = source["canvas"]["width"], source["canvas"]["height"]
    foreground = []
    for element in source["elements"]:
        area = element.get("width", 0) * element.get("height", 0) / (width * height)
        role = str(element.get("role", element.get("semantic_role", ""))).lower()
        if role in {"background", "decoration", "ornament"}:
            continue
        if element["type"] in {"shape", "image"} and area >= 0.8:
            continue
        foreground.append(element["id"])
    return set(foreground or [e["id"] for e in source["elements"]])


def _columns(elements, canvas_width):
    groups = []
    for element in sorted(elements, key=lambda e: e.get("x", 0)):
        if not 0.1 <= element.get("width", 0) / canvas_width <= 0.72:
            continue
        x = element.get("x", 0) / canvas_width
        group = next((g for g in groups if abs(g[0] - x) <= 0.04), None)
        if group is None:
            groups.append([x, [element]])
        else:
            group[1].append(element)
    repeated = [rows for _x, rows in groups if len(rows) >= 2]
    return repeated or ([elements] if elements else [])


def extract_features(source, target, design, *, units=None):
    """Measure source relationships and target geometry without loading pixels.

    Bounding frames approximate visible occupancy; this policy cannot inspect
    alpha silhouettes or background subjects. Native visual QA remains required.
    """
    sw, sh = (float(source["canvas"][k]) for k in ("width", "height"))
    tw, th = (float(target[k]) for k in ("width", "height"))
    if min(sw, sh, tw, th) <= 0 or not all(math.isfinite(v) for v in (sw, sh, tw, th)):
        raise ValueError("Fit policy requires finite positive canvases")
    if len(source["elements"]) > 256 or len(design["elements"]) > 256:
        raise ValueError("Fit policy element budget exceeded")
    original = {e["id"]: e for e in source["elements"]}
    current = {e["id"]: e for e in design["elements"]}
    if original.keys() != current.keys():
        raise ValueError("Fit policy cannot score changed element identities")
    foreground_ids = _foreground(source)
    foreground = [e for key, e in current.items() if key in foreground_ids]
    boxes = [_box(e) for e in foreground]
    if not boxes:
        raise ValueError("Fit policy needs foreground content")
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    fw, fh = (x1 - x0) / tw, (y1 - y0) / th
    area = sum(max(0, b[2]) * max(0, b[3]) for b in boxes) / (tw * th)
    overflow = sum(
        max(0, w * h - max(0, min(tw, x + w) - max(0, x)) * max(0, min(th, y + h) - max(0, y)))
        for x, y, w, h in (_box(e) for e in current.values())
    ) / (tw * th)
    texts = [e for e in current.values() if e["type"] == "text"]
    source_texts = [e for e in source["elements"] if e["type"] == "text"]
    images = [e for e in foreground if e["type"] == "image"]
    fonts = [max(0.01, _finite(e.get("font_size"), 1)) for e in texts]
    scales = [
        math.log(
            max(0.001, _finite(e.get("font_size"), 1))
            / max(0.001, _finite(original[e["id"]].get("font_size"), 1))
        )
        for e in texts
    ]
    center_scale = median(scales) if scales else 0
    hierarchy = []
    for index, left in enumerate(source_texts):
        for right in source_texts[index + 1 :]:
            old = left.get("font_size", 1) - right.get("font_size", 1)
            if abs(old) < 0.1:
                continue
            new = current[left["id"]].get("font_size", 1) - current[right["id"]].get("font_size", 1)
            hierarchy.append(float(old * new > 0))
    reading = []
    for column in _columns(source_texts, sw):
        ordered = sorted(column, key=lambda e: e.get("y", 0))
        for first, second in zip(ordered, ordered[1:], strict=False):
            a, b = current[first["id"]], current[second["id"]]
            same_column = abs(a.get("x", 0) - b.get("x", 0)) / tw <= 0.08
            reading.append(
                float(
                    a.get("y", 0) <= b.get("y", 0)
                    if same_column
                    else a.get("x", 0) <= b.get("x", 0)
                )
            )
    column_rows = _columns(texts, tw)
    group_scales, attachment = [], []
    if units is None:
        groups = defaultdict(list)
        for e in source["elements"]:
            if e.get("group_id"):
                groups[str(e["group_id"])].append(e["id"])
        unit_ids = list(groups.values())
    else:
        unit_ids = [u.get("element_ids", []) for u in units]
    for ids in unit_ids:
        ids = [key for key in ids if key in original]
        if len(ids) < 2:
            continue
        ratios = [
            math.log(
                max(0.001, current[key].get("width", 1)) / max(0.001, original[key].get("width", 1))
            )
            for key in ids
        ]
        mid = median(ratios)
        group_scales.append(math.exp(-sum(abs(v - mid) for v in ratios) / len(ratios)))
        for a, b in zip(ids, ids[1:], strict=False):
            ox = (original[a].get("x", 0) - original[b].get("x", 0)) / sw
            oy = (original[a].get("y", 0) - original[b].get("y", 0)) / sh
            nx = (current[a].get("x", 0) - current[b].get("x", 0)) / tw
            ny = (current[a].get("y", 0) - current[b].get("y", 0)) / th
            attachment.append(math.exp(-abs(math.hypot(nx, ny) - math.hypot(ox, oy))))
    image_ratios = [
        math.exp(
            -abs(
                math.log(
                    max(0.001, e.get("width", 1))
                    / max(0.001, e.get("height", 1))
                    / (
                        max(0.001, original[e["id"]].get("width", 1))
                        / max(0.001, original[e["id"]].get("height", 1))
                    )
                )
            )
        )
        for e in images
    ]
    short = min(tw, th)
    text_area = sum(max(0, e.get("width", 0) * e.get("height", 0)) for e in texts)
    ink_estimate = sum(
        len(str(e.get("content", ""))) * e.get("font_size", 1) ** 2 * 0.45 for e in texts
    )

    def average(values):
        return sum(values) / len(values) if values else 1.0

    features = {
        "source_log_aspect": math.log(sw / sh),
        "target_log_aspect": math.log(tw / th),
        "aspect_change": abs(math.log(tw / th) - math.log(sw / sh)),
        "foreground_width": fw,
        "foreground_height": fh,
        "foreground_bbox_area": fw * fh,
        "foreground_area": area,
        "foreground_density": area / max(0.001, fw * fh),
        "center_x_error": abs((x0 + x1) / (2 * tw) - 0.5),
        "center_y_error": abs((y0 + y1) / (2 * th) - 0.5),
        "edge_margin": min(x0 / tw, y0 / th, 1 - x1 / tw, 1 - y1 / th),
        "overflow_fraction": overflow,
        "text_fraction": text_area / (tw * th),
        "text_count": len(texts) / 100,
        "image_count": len(images) / 20,
        "column_count": min(6, len(column_rows)) / 6,
        "column_width_use": sum(max(e.get("width", 0) for e in rows) for rows in column_rows) / tw,
        "wide_column_use": min(3, tw / th) * min(6, len(column_rows)) / 6,
        "minimum_font": min(fonts, default=0) / short * 100,
        "median_font": median(fonts) / short * 100 if fonts else 0,
        "font_scale": center_scale,
        "font_scale_consistency": math.exp(-average([abs(v - center_scale) for v in scales]))
        if scales
        else 1,
        "hierarchy_retention": average(hierarchy),
        "reading_order_retention": average(reading),
        "unit_scale_consistency": average(group_scales),
        "unit_attachment": average(attachment),
        "image_proportion_retention": average(image_ratios),
        "text_frame_fill": ink_estimate / max(1, text_area),
    }
    return {name: max(-20.0, min(20.0, _finite(features[name]))) for name in FEATURE_NAMES}


def _validate_features(features):
    if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
        raise ValueError("Fit feature schema differs from the trained policy")
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in features.values()):
        raise ValueError("Fit features must be finite numbers")
    return [float(features[name]) for name in FEATURE_NAMES]


def _groups(records):
    groups = defaultdict(list)
    for row in records:
        target = row["target_canvas"]
        key = (row["family"], row["source_id"], target["width"], target["height"])
        groups[key].append(row)
    return groups


def _vector(features, means, scales):
    return [
        (v - m) / s for v, m, s in zip(_validate_features(features), means, scales, strict=True)
    ]


def _score(vector, weights):
    return sum(v * w for v, w in zip(vector, weights, strict=True))


def _metrics(records, weights, means, scales):
    correct, pairs, regrets, winners = 0.0, 0, [], 0
    for rows in _groups(records).values():
        ranked = sorted(rows, key=lambda row: row["teacher_score"], reverse=True)
        scored = [(row, _score(_vector(row["features"], means, scales), weights)) for row in ranked]
        winner = max(scored, key=lambda pair: (pair[1], pair[0]["plan_id"]))[0]
        regrets.append(ranked[0]["teacher_score"] - winner["teacher_score"])
        winners += float(abs(regrets[-1]) <= 1e-8)
        for i, (left, left_score) in enumerate(scored):
            for right, right_score in scored[i + 1 :]:
                if left["teacher_score"] - right["teacher_score"] < 0.01:
                    continue
                correct += (
                    1.0 if left_score > right_score else 0.5 if left_score == right_score else 0
                )
                pairs += 1
    return {
        "records": len(records),
        "groups": len(regrets),
        "comparison_pairs": pairs,
        "pairwise_accuracy": correct / pairs if pairs else None,
        "teacher_winner_accuracy": winners / len(regrets) if regrets else None,
        "mean_teacher_regret": sum(regrets) / len(regrets) if regrets else None,
    }


def train_model(dataset, out, *, epochs=200, learning_rate=0.08, l2=0.02, seed=1729):
    """Actually fit pairwise logistic weights to development-only comparisons."""
    if (
        dataset.get("schema_version") != 1
        or dataset.get("feature_version") != FEATURE_VERSION
        or dataset.get("label_kind") != LABEL_KIND
    ):
        raise ValueError("Unsupported fit training dataset/label schema")
    records = dataset.get("records")
    if not isinstance(records, list) or not 4 <= len(records) <= 20_000:
        raise ValueError("Fit training requires a bounded dataset")
    if not 1 <= epochs <= 1000 or not 0 < learning_rate <= 1 or not 0 <= l2 <= 1:
        raise ValueError("Invalid bounded fit training hyperparameters")
    for row in records:
        _validate_features(row["features"])
        if row.get("split") not in {"train", "validation"} or not row.get("family"):
            raise ValueError("Explicit family-separated train/validation splits are required")
        if not math.isfinite(row["teacher_score"]) or row.get("hard_valid") is not True:
            raise ValueError("Only hard-valid finite-scored candidates may train the policy")
        if row.get("reserve") or row.get("split_role") in {"reserve", "manual_acceptance"}:
            raise ValueError("Reserve/manual acceptance sources cannot train the fit policy")
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    families = {
        split: sorted({row["family"] for row in rows})
        for split, rows in (("train", train), ("validation", validation))
    }
    if not train or not validation or set(families["train"]) & set(families["validation"]):
        raise ValueError("Train and validation source families must be nonempty and disjoint")
    if {r["source_digest"] for r in train} & {r["source_digest"] for r in validation}:
        raise ValueError("The same source cannot leak across fit training splits")
    values = [_validate_features(row["features"]) for row in train]
    means = [sum(column) / len(column) for column in zip(*values, strict=True)]
    scales = [
        max(0.1, math.sqrt(sum((row[i] - means[i]) ** 2 for row in values) / len(values)))
        for i in range(len(FEATURE_NAMES))
    ]
    rng, pairs = random.Random(seed), []
    for rows in _groups(train).values():
        ranked = sorted(rows, key=lambda row: row["teacher_score"], reverse=True)
        candidates = [
            (a, b)
            for i, a in enumerate(ranked)
            for b in ranked[i + 1 :]
            if a["teacher_score"] - b["teacher_score"] >= 0.01
        ]
        rng.shuffle(candidates)
        for left, right in candidates[:128]:
            a, b = (_vector(r["features"], means, scales) for r in (left, right))
            pairs.append([x - y for x, y in zip(a, b, strict=True)])
    if not pairs:
        raise ValueError("No non-tied supervised candidate comparisons exist")
    weights = [0.0] * len(FEATURE_NAMES)
    started, losses = time.monotonic(), []
    for epoch in range(epochs):
        rng.shuffle(pairs)
        rate = learning_rate / math.sqrt(1 + epoch / 20)
        loss = 0.0
        for pair in pairs:
            margin = max(-40, min(40, _score(pair, weights)))
            gradient = 1 / (1 + math.exp(margin))
            loss += math.log1p(math.exp(-margin))
            for i, value in enumerate(pair):
                weights[i] += rate * (gradient * value - l2 * weights[i] / len(pairs))
        if epoch in {0, epochs - 1}:
            losses.append({"epoch": epoch + 1, "mean_pairwise_log_loss": loss / len(pairs)})
    artifact = {
        "schema_version": 1,
        "kind": "trained_pairwise_fit_policy",
        "feature_version": FEATURE_VERSION,
        "label_kind": LABEL_KIND,
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "means": means,
        "scales": scales,
        "dataset_digest": digest(dataset),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "families": families,
        "source_digests": {
            s: sorted({r["source_digest"] for r in rows})
            for s, rows in (("train", train), ("validation", validation))
        },
        "training": {
            "optimizer": "pairwise_logistic_sgd",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
            "pairs": len(pairs),
            "seconds": time.monotonic() - started,
            "losses": losses,
        },
        "metrics": {
            "train": _metrics(train, weights, means, scales),
            "validation": _metrics(validation, weights, means, scales),
            "untrained_validation": _metrics(validation, [0.0] * len(weights), means, scales),
            "validation_by_source_kind": {
                kind: _metrics(
                    [row for row in validation if row.get("source_kind", "unspecified") == kind],
                    weights,
                    means,
                    scales,
                )
                for kind in sorted({row.get("source_kind", "unspecified") for row in validation})
            },
            "train_by_source_kind": {
                kind: _metrics(
                    [row for row in train if row.get("source_kind", "unspecified") == kind],
                    weights,
                    means,
                    scales,
                )
                for kind in sorted({row.get("source_kind", "unspecified") for row in train})
            },
        },
        "quality_claim": "Synthetic geometry-teacher agreement; no human aesthetic validation.",
        "vlm_fine_tuning": False,
    }
    artifact["model_digest"] = digest(artifact)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return artifact


def load_model(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("Fit policy artifact is missing or oversized")
    with path.open("rb") as stream:
        data = stream.read(1_000_001)
    if len(data) > 1_000_000:
        raise ValueError("Fit policy artifact is missing or oversized")
    artifact = json.loads(data.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("Fit policy artifact must be a versioned object")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("kind") != "trained_pairwise_fit_policy"
        or artifact.get("feature_version") != FEATURE_VERSION
        or artifact.get("feature_names") != list(FEATURE_NAMES)
        or artifact.get("model_digest")
        != digest({k: v for k, v in artifact.items() if k != "model_digest"})
    ):
        raise ValueError("Fit policy artifact identity/schema mismatch")
    for key in ("weights", "means", "scales"):
        values = artifact.get(key)
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
            raise ValueError("Fit policy parameter shape mismatch")
        if any(type(v) not in {int, float} or not math.isfinite(v) for v in values):
            raise ValueError("Fit policy parameters must be finite")
    if any(v <= 0 for v in artifact["scales"]):
        raise ValueError("Fit policy scales must be positive")
    return artifact


def fit_policy_identity(model_path=None):
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    identity = {
        "enabled": False,
        "path": str(path.resolve()),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if not path.is_file():
        return {**identity, "status": "no_model", "model_sha256": None}
    try:
        stat = path.stat()
        identity.update(
            model_file_size=stat.st_size,
            model_mtime_ns=stat.st_mtime_ns,
            model_sha256=None,
            model_bytes_verified=False,
        )
        if stat.st_size <= 1_000_000:
            with path.open("rb") as stream:
                data = stream.read(1_000_001)
            if len(data) <= 1_000_000:
                identity.update(
                    model_sha256=hashlib.sha256(data).hexdigest(), model_bytes_verified=True
                )
        artifact = load_model(path)
        return {
            **identity,
            "enabled": True,
            "status": "ready",
            "model_sha256": identity["model_sha256"],
            "model_digest": artifact["model_digest"],
            "feature_version": artifact["feature_version"],
            "label_kind": artifact["label_kind"],
            "dataset_digest": artifact["dataset_digest"],
            "families": artifact["families"],
            "metrics": artifact["metrics"],
            "vlm_fine_tuning": False,
        }
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return {**identity, "status": "invalid_model", "error": str(exc)[:500]}


def build_fit_menu(source, target, candidate_designs, *, units=None):
    menu = {
        "schema_version": 1,
        "task": "fitted_layout_policy",
        "source_digest": source_digest(source),
        "target_canvas": deepcopy(target),
        "plans": [
            {"plan_id": row["plan_id"], "design_digest": digest(row["design"])}
            for row in candidate_designs
            if row.get("hard_valid") is True
        ],
    }
    if not menu["plans"] or len({p["plan_id"] for p in menu["plans"]}) != len(menu["plans"]):
        raise ValueError("Fit menu requires distinct known hard-valid plans")
    if units is not None:
        menu["units"] = deepcopy(units)
    menu["menu_digest"] = digest(menu)
    return menu


def choose_fit_plan(source, target, menu, candidate_designs, model_path=None):
    """Rank actual known designs; returned decision can only select a bound ID."""
    if (
        menu.get("source_digest") != source_digest(source)
        or menu.get("target_canvas") != target
        or menu.get("menu_digest") != digest({k: v for k, v in menu.items() if k != "menu_digest"})
    ):
        raise ValueError("Fit menu belongs to another source/target or changed after binding")
    plans = {p["plan_id"]: p for p in menu["plans"]}
    rows = {r["plan_id"]: r for r in candidate_designs if r.get("hard_valid") is True}
    if not 1 <= len(rows) <= 2048 or set(rows) != set(plans) or len(plans) != len(menu["plans"]):
        raise ValueError("Fit choices must exactly match their known hard-valid menu")
    identity = fit_policy_identity(model_path)
    if not identity["enabled"]:
        return {
            "decision": None,
            "receipt": {
                "executed": False,
                "kind": "trained_pairwise_fit_policy",
                "status": identity["status"],
                "identity": identity,
                "quality_improved": None,
            },
        }
    model_path = Path(identity["path"])
    model = load_model(model_path)
    if (
        model["model_digest"] != identity["model_digest"]
        or hashlib.sha256(model_path.read_bytes()).hexdigest() != identity["model_sha256"]
    ):
        raise ValueError("Fit policy artifact changed while loading its pinned identity")
    scores = []
    for plan_id, row in rows.items():
        if row["design"].get("canvas") != target or plans[plan_id].get("design_digest") != digest(
            row["design"]
        ):
            raise ValueError("Fit candidate geometry differs from its bound menu")
        features = extract_features(source, target, row["design"], units=menu.get("units"))
        vector = _vector(features, model["means"], model["scales"])
        scores.append(
            {
                "plan_id": plan_id,
                "score": _score(vector, model["weights"]),
                "design_digest": digest(row["design"]),
                "features": features,
            }
        )
    selected = max(scores, key=lambda row: (row["score"], row["plan_id"]))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != identity["model_sha256"]:
        raise ValueError("Fit policy artifact changed during candidate ranking")
    request_digest = digest(
        {
            "menu_digest": menu["menu_digest"],
            "designs": {r["plan_id"]: r["design_digest"] for r in scores},
        }
    )
    return {
        "decision": {
            "schema_version": 1,
            "menu_digest": menu["menu_digest"],
            "plan_id": selected["plan_id"],
            "rationale": "Locally trained pairwise fit policy selected this admitted geometry.",
        },
        "receipt": {
            "executed": True,
            "kind": model["kind"],
            "feature_version": FEATURE_VERSION,
            "model_digest": model["model_digest"],
            "model_sha256": identity["model_sha256"],
            "dataset_digest": model["dataset_digest"],
            "request_digest": request_digest,
            "source_digest": menu["source_digest"],
            "target_canvas": deepcopy(target),
            "scores": scores,
            "quality_improved": None,
            "vlm_fine_tuning": False,
        },
    }
