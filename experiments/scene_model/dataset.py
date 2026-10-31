"""Reviewed, source-bound two-view scene supervision without opaque ID targets.

Records point at scene-harness receipts and a reviewed full harness response.
Only semantic labels enter the model target. Original identities remain audit
metadata, and reserved personal-test sources cannot enter training manifests.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from PIL import Image

MAX_RECORDS = 1024
MAX_JSON_BYTES = 24 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SPLITS = {"train", "validation", "test"}
ORIGINS = {
    "human_review",
    "agent_visual_review",
    "user_explicit_and_agent_review",
    "synthetic_rule",
}
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Scene JSON exceeds its byte budget")

    def constant(_value):
        raise ValueError("Scene JSON numbers must be finite")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate scene JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"), parse_constant=constant, object_pairs_hook=pairs
        )
    except RecursionError as exc:
        raise ValueError("Scene JSON exceeds its nesting budget") from exc


def _inside(root, value, relative_to=None):
    if not isinstance(value, str) or not value:
        raise ValueError("Scene artifact path must be nonempty text")
    path = Path(value)
    if not path.is_absolute():
        path = (relative_to or root) / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Scene artifact path leaves the allowed local root")
    return path


def _identifier(value, label):
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise ValueError(f"Invalid scene {label}")
    return value


@lru_cache(maxsize=1)
def _reserves():
    catalogue = ROOT / "fixtures/open_posters/catalogue.json"
    if not catalogue.is_file():
        raise ValueError("Reserve exclusion catalogue is missing")
    ids, source_digests = set(), set()
    for row in read_json(catalogue)["examples"]:
        if row.get("split") != "reserve":
            continue
        ids.add(row["id"])
        source = catalogue.parent / "prepared" / row["id"] / "source.json"
        if source.is_file():
            source_digests.add(digest(read_json(source)))
    return frozenset(ids), frozenset(source_digests)


def _image(path):
    with Image.open(path) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Scene image exceeds the decode budget")
        image.verify()


def _load_record(record, root, manifest_dir, allow_synthetic):
    from tools.scene_harness import validate_model_response

    from .prompt import build_prompt
    from .schema import validate_prediction

    required = {
        "record_id",
        "source_id",
        "family_id",
        "split",
        "harness_dir",
        "labels",
        "element_id",
        "supervision_origin",
        "reviewed_by",
        "rights_accepted",
        "source_parity_reviewed",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Scene record fields do not match the closed manifest schema")
    for key in ("record_id", "source_id", "family_id"):
        _identifier(record[key], key)
    if record["split"] not in SPLITS or record["supervision_origin"] not in ORIGINS:
        raise ValueError("Unknown scene split or supervision origin")
    if record["supervision_origin"] == "synthetic_rule" and not allow_synthetic:
        raise ValueError("Synthetic scene labels require an explicit infrastructure opt-in")
    if record["rights_accepted"] is not True or record["source_parity_reviewed"] is not True:
        raise ValueError("Scene sources require explicit rights and source-parity review")
    if not isinstance(record["reviewed_by"], str) or not record["reviewed_by"].strip():
        raise ValueError("Scene labels require reviewer provenance")
    reserve_ids, reserve_digests = _reserves()
    if record["source_id"] in reserve_ids or record["family_id"] in reserve_ids:
        raise ValueError("Personal reserve sources cannot enter scene training")
    directory = _inside(root, record["harness_dir"], manifest_dir)
    request = read_json(directory / "model-request.json")
    source, profile = read_json(directory / "source.json"), read_json(directory / "scene.json")
    report = read_json(directory / "report.json")
    source_digest = digest(source)
    if source_digest in reserve_digests:
        raise ValueError("Renamed personal reserve source cannot enter scene training")
    profile_body = {k: v for k, v in profile.items() if k != "scene_digest"}
    if (
        source_digest != profile.get("source_digest")
        or digest(profile_body) != profile.get("scene_digest")
        or request.get("source") != source
        or request.get("deterministic_scene") != profile
        or request.get("source_digest") != source_digest
        or request.get("scene_digest") != profile["scene_digest"]
    ):
        raise ValueError("Scene source/profile/request identities differ")
    if (
        report.get("source_digest") != source_digest
        or report.get("scene_digest") != profile["scene_digest"]
        or report.get("source_immutable") is not True
        or report.get("resources_immutable") is not True
    ):
        raise ValueError("Scene source immutability receipt is incomplete")
    for image in profile["images"]:
        original_asset = _inside(root, request["asset_files"][image["element_id"]], directory)
        if sha256(original_asset) != image["asset_sha256"]:
            raise ValueError("Scene source asset bytes changed after review")
    labels = read_json(_inside(root, record["labels"], manifest_dir))
    validate_model_response(labels, profile)
    selected = [row for row in labels["images"] if row["element_id"] == record["element_id"]]
    if len(selected) != 1:
        raise ValueError("Scene training record needs exactly one reviewed image label")
    label = selected[0]
    elements = [
        e for e in source["elements"] if e["id"] == record["element_id"] and e["type"] == "image"
    ]
    if len(elements) != 1:
        raise ValueError("Scene image label does not identify a source image")
    element = elements[0]
    render = _inside(root, request["source_render"]["path"], directory)
    asset = _inside(root, request["asset_files"][element["id"]], directory)
    source_render_sha256, asset_sha256 = sha256(render), sha256(asset)
    if (
        source_render_sha256 != request["source_render"].get("sha256")
        or source_render_sha256 != report["source_render"].get("sha256")
        or asset_sha256 != label["asset_sha256"]
    ):
        raise ValueError("Scene source render or original image bytes changed")
    for path in (render, asset):
        _image(path)
    prediction = validate_prediction(
        {k: v for k, v in label.items() if k not in {"element_id", "asset_sha256"}}
    )
    # No mutable inferred-role label is supplied to the model as input. The
    # image pair and original element/source geometry are the shared provider seam.
    prompt = build_prompt(element, source["canvas"])
    row = {
        **record,
        "source_digest": source_digest,
        "scene_digest": profile["scene_digest"],
        "images": [str(render), str(asset)],
        "image_sha256": [source_render_sha256, asset_sha256],
        "prompt": prompt,
        "response": canonical(prediction),
        "prediction": prediction,
        "source_resource_receipt": report.get("resource_hashes", {}),
    }
    row["input_digest"] = digest(
        {
            "source": source_digest,
            "images": row["image_sha256"],
            "prompt": prompt,
            "response": row["response"],
            "origin": row["supervision_origin"],
            "reviewer": row["reviewed_by"],
            "source_resources": row["source_resource_receipt"],
        }
    )
    return row


def load_records(manifest, *, allowed_root=None, allow_synthetic=False):
    """Validate all bytes and splits without loading a model or tokenizer."""
    manifest = Path(manifest).resolve()
    root = Path(allowed_root or ROOT).resolve()
    if not manifest.is_relative_to(root):
        raise ValueError("Scene manifest leaves the allowed local root")
    payload = read_json(manifest)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "records"}:
        raise ValueError("Scene manifest requires schema_version and records")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported scene dataset schema")
    records = payload["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError("Scene manifest record count is outside its budget")
    try:
        rows = [_load_record(record, root, manifest.parent, allow_synthetic) for record in records]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed scene record or source receipt") from exc
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate scene record ID")
    if len({(row["source_digest"], row["element_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate source image supervision record")
    for field in ("family_id", "source_id", "source_digest"):
        memberships = {}
        for row in rows:
            previous = memberships.setdefault(row[field], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Scene {field} crosses experiment splits")
    # An identical selected asset is not independent semantic test evidence.
    asset_splits = {}
    for row in rows:
        previous = asset_splits.setdefault(row["image_sha256"][1], row["split"])
        if previous != row["split"]:
            raise ValueError("Selected scene image bytes cross experiment splits")
    return rows


def dataset_summary(rows):
    return {
        "records": len(rows),
        "families": len({r["family_id"] for r in rows}),
        "splits": dict(Counter(r["split"] for r in rows)),
        "supervision_origins": dict(Counter(r["supervision_origin"] for r in rows)),
        "roles_by_split": {
            split: dict(
                Counter(r["prediction"]["proposed_role"] for r in rows if r["split"] == split)
            )
            for split in sorted(SPLITS)
        },
        "quality_validated": False,
    }
