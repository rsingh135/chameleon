"""Freeze small, train-only weak critic evidence without rendering or models.

Existing source/contract/resource modules own identity and validity. This adapter
only validates supplied evidence, copies immutable bytes and counterbalances A/B.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from engine.design_semantics import semantic_source_digest
from engine.eval import build_context, check_contract
from engine.resources import using_resources
from engine.stage_one import strictly_in_bounds
from experiments.vision_critic.contracts import (
    canonical_json_sha256,
    safe_artifact_path,
    sha256_file,
)
from tools import semantic_training_data as data

PROTOCOL = "critic-pair-training-data-v1"
MAX_RECORDS = 64
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "output/semantic-analysis-data-v1/dataset-smoke-v1/manifest.json"


def _snapshot_sha(path):
    """Code package markers may be empty; initial image/JSON admission may not."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > data.MAX_FILE_BYTES:
        raise ValueError("Snapshot file is missing or exceeds its byte budget")
    return sha256_file(path)


def _object(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= set(value)
        or set(value) - set(required) - set(optional)
    ):
        raise ValueError("Evidence object has missing or unknown fields")
    return value


def _text(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 2000:
        raise ValueError("Evidence identity/provenance needs bounded nonempty text")
    return value


def _canvas(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"width", "height"}
        or any(type(number) is not int or number <= 0 for number in value.values())
    ):
        raise ValueError("Canvas must have positive integer width and height")
    if value["width"] * value["height"] > data.MAX_PIXELS:
        raise ValueError("Canvas exceeds the native pixel budget")
    return value


def _authority(manifest_path, split_manifest):
    manifest = data._read(manifest_path)
    if (
        manifest.get("protocol") != data.PROTOCOL
        or manifest.get("status") != "complete_reviewed_annotations"
    ):
        raise ValueError("A complete frozen source authority is required")
    _, splits = data._split(split_manifest)
    if data._sha(split_manifest) != manifest.get("split_manifest_sha256"):
        raise ValueError("Family split differs from the frozen source authority")
    root, result = manifest_path.parent, {}
    for entry in manifest["sources"]:
        receipt_path = safe_artifact_path(root, entry["receipt"])
        if data._sha(receipt_path) != manifest["files_sha256"].get(entry["receipt"]):
            raise ValueError("Frozen source receipt hash differs")
        receipt = data._read(receipt_path)
        identity = receipt["source_id"]
        if (
            identity not in splits
            or receipt["split"] != splits[identity]["split"]
            or receipt["family"] != splits[identity]["family"]
        ):
            raise ValueError("Frozen source family identity differs from the split")
        relative = entry["receipt"].removesuffix(".json") + ".source.json"
        source_path = safe_artifact_path(root, relative)
        if data._sha(source_path) != manifest["files_sha256"].get(relative):
            raise ValueError("Frozen original source hash differs")
        source = data._read(source_path)
        if (
            semantic_source_digest(source) != receipt["source_digest"]
            or [e["id"] for e in source["elements"]] != receipt["original_element_ids"]
        ):
            raise ValueError("Frozen original IDs or compact source digest differ")
        render_relative = "images/" + receipt["source_digest"][:20] + ".png"
        render_path = safe_artifact_path(root, render_relative)
        if (
            data._sha(render_path) != receipt["source_render_sha256"]
            or manifest["files_sha256"].get(render_relative) != receipt["source_render_sha256"]
        ):
            raise ValueError("Frozen original PNG hash differs")
        result[identity] = {
            "receipt": receipt,
            "source": source,
            "split": splits[identity],
            "files": {
                str(receipt_path): data._sha(receipt_path),
                str(source_path): data._sha(source_path),
                str(render_path): receipt["source_render_sha256"],
            },
        }
    if set(result) != set(splits):
        raise ValueError("Frozen source authority coverage is incomplete")
    return manifest, result


def _artifact(root, value, source, canvas, snapshots, *, render_canvas=None):
    required = {"design", "json_sha256" if source else "design_sha256", "render", "render_sha256"}
    optional = () if source else ("contract_valid", "contract_violations", "provenance")
    _object(value, required, optional)
    design_path, image_path = (safe_artifact_path(root, value[key]) for key in ("design", "render"))
    for path in (design_path, image_path):
        actual = data._sha(path)
        if str(path) in snapshots and snapshots[str(path)] != actual:
            raise ValueError("Immutable evidence changed during validation")
        snapshots[str(path)] = actual
    design = data._read(design_path)
    digest_key = "json_sha256" if source else "design_sha256"
    if (
        canonical_json_sha256(design) != value[digest_key]
        or snapshots[str(image_path)] != value["render_sha256"]
    ):
        raise ValueError("Supplied JSON or PNG hash differs from its actual bytes")
    if not isinstance(design, dict) or _canvas(design.get("canvas")) != canvas:
        raise ValueError("Design canvas differs from its declared source/target")
    elements = design.get("elements")
    if (
        not isinstance(elements, list)
        or len(elements) > 256
        or any(
            not isinstance(element, dict) or not isinstance(element.get("id"), str)
            for element in elements
        )
    ):
        raise ValueError("Candidate elements must be a bounded native list with string IDs")
    with Image.open(image_path) as image:
        expected_render = canvas if render_canvas is None else render_canvas
        if image.format != "PNG" or image.size != tuple(
            expected_render[key] for key in ("width", "height")
        ):
            raise ValueError("PNG dimensions differ from the corresponding source/target canvas")
        image.verify()
    return {
        "design": design,
        "design_path": design_path,
        "image_path": image_path,
        "metadata": deepcopy(value),
    }


def _rights(value, catalogue_entry):
    _object(value, {"license_id", "license_url", "source_url", "acquired_utc", "attribution"})
    for field in value:
        _text(value[field])
    for field in ("license_url", "source_url"):
        if not value[field].startswith(("https://", "http://")):
            raise ValueError("License and source URLs must be explicit HTTP(S) references")
    when = datetime.fromisoformat(value["acquired_utc"].replace("Z", "+00:00"))
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("Acquisition timestamp needs an explicit UTC offset")
    expected = {
        "license_id": catalogue_entry["license_id"],
        "license_url": catalogue_entry["license_url"],
        "source_url": catalogue_entry["source_url"],
        "attribution": catalogue_entry["attribution_text"],
    }
    if any(value[key] != text for key, text in expected.items()):
        raise ValueError("License/source authority differs from the original catalogue")


def _generation(path, input_path, raw, split_manifest, source_manifest, snapshots):
    if path is None:
        raise ValueError("A bound generation receipt is required for a training corpus")
    path = Path(path).resolve()
    if not path.is_relative_to(input_path.parent):
        raise ValueError("Generation receipt must remain within the evidence root")
    snapshots[str(path)] = data._sha(path)
    receipt = data._read(path)
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "complete"
        or receipt.get("input_raw_sha256") != snapshots[str(input_path)]
        or receipt.get("input_canonical_sha256") != canonical_json_sha256(raw)
        or receipt.get("records") != len(raw["records"])
        or receipt.get("split_sha256") != snapshots[str(split_manifest)]
        or receipt.get("frozen_manifest_sha256") != snapshots[str(source_manifest)]
        or safe_artifact_path(input_path.parent, receipt.get("input")) != input_path
    ):
        raise ValueError("Generation receipt does not bind this exact input/source/split evidence")
    hashes = receipt.get("implementation_hashes")
    if (
        not isinstance(hashes, dict)
        or not 1 <= len(hashes) <= 512
        or hashes != receipt.get("implementation_after_hashes")
    ):
        raise ValueError("Generation producer/code identity is absent or changed")
    for relative, expected in hashes.items():
        code_path = safe_artifact_path(ROOT, relative)
        actual = _snapshot_sha(code_path)
        if actual != expected:
            raise ValueError("Generation producer/code bytes differ from their recorded identity")
        snapshots[str(code_path)] = actual
    expected_sources = sorted({row["source_id"] for row in raw["records"]})
    if receipt.get("train_sources") != expected_sources:
        raise ValueError("Generation receipt source-family coverage differs")
    return {"path": path, "receipt": receipt}


def freeze_pairs(
    *,
    input,
    split_manifest,
    out,
    source_manifest=DEFAULT_SOURCE_MANIFEST,
    swap=False,
    generation_receipt=None,
):
    """Verify source+A+B evidence, then publish a portable train-only weak corpus."""
    input, split_manifest, out = (
        Path(input).resolve(),
        Path(split_manifest).resolve(),
        Path(out).resolve(),
    )
    source_manifest = Path(source_manifest).resolve()
    if out.exists():
        raise ValueError("Pair dataset output must be fresh")
    for immutable in (input.parent, source_manifest.parent):
        if out.is_relative_to(immutable) or immutable.is_relative_to(out):
            raise ValueError("Output overlaps an immutable evidence/source root")
    snapshots = {str(path): data._sha(path) for path in (input, split_manifest, source_manifest)}
    frozen, authority = _authority(source_manifest, split_manifest)
    for owner in authority.values():
        snapshots.update(owner["files"])
    raw = data._read(input)
    _object(raw, {"schema_version", "protocol", "records", "resource_roots"}, {"provenance"})
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["protocol"] != "critic-pair-evidence-v1"
    ):
        raise ValueError("Unsupported critic evidence protocol")
    if not isinstance(raw["records"], list) or not 1 <= len(raw["records"]) <= MAX_RECORDS:
        raise ValueError("Supply a bounded nonempty critic evidence batch")
    if not isinstance(raw["resource_roots"], dict):
        raise ValueError("Source resource roots must be explicit")
    generation = _generation(
        generation_receipt, input, raw, split_manifest, source_manifest, snapshots
    )
    catalogue_root = ROOT / "fixtures/open_posters"
    catalogue_path = catalogue_root / "catalogue.json"
    snapshots[str(catalogue_path)] = data._sha(catalogue_path)
    catalogue = data._read(catalogue_path)
    forbidden_hashes = {
        semantic_source_digest(item["source"])
        for item in authority.values()
        if item["split"]["split"] != "train"
    }
    forbidden_render_hashes = {
        item["receipt"]["source_render_sha256"]
        for item in authority.values()
        if item["split"]["split"] != "train"
    }
    for entry in catalogue["examples"]:
        if (entry["split"] == "reserve" or entry["id"] in data.FORBIDDEN_SOURCES) and entry.get(
            "prepared_path"
        ):
            path = safe_artifact_path(catalogue_root, entry["prepared_path"] + "/source.json")
            if path.is_file():
                snapshots[str(path)] = data._sha(path)
                forbidden_hashes.add(semantic_source_digest(data._read(path)))
            for name in ("native.png", "imported.png"):
                image_path = safe_artifact_path(catalogue_root, entry["prepared_path"] + "/" + name)
                if image_path.is_file():
                    snapshots[str(image_path)] = data._sha(image_path)
                    forbidden_render_hashes.add(snapshots[str(image_path)])
    prepared, seen_ids, seen_pairs, families, source_resources = [], set(), set(), {}, {}
    for row in raw["records"]:
        _object(
            row,
            {
                "record_id",
                "pair_id",
                "source_id",
                "family_id",
                "split",
                "label",
                "label_origin",
                "source",
                "A",
                "B",
                "target_canvas",
                "rights",
                "provenance",
            },
            {"control"},
        )
        for key in ("record_id", "pair_id", "source_id", "family_id"):
            _text(row[key])
        if row["record_id"] in seen_ids or row["pair_id"] in seen_pairs:
            raise ValueError("Duplicate record or original pair identity")
        seen_ids.add(row["record_id"])
        seen_pairs.add(row["pair_id"])
        identity = row["source_id"]
        owner = authority.get(identity)
        if (
            owner is None
            or owner["split"]["split"] != "train"
            or row["split"] != "train"
            or row["family_id"] != owner["split"]["family"]
        ):
            raise ValueError("Only original frozen TRAIN source/family identities are eligible")
        if identity in families and families[identity] != row["family_id"]:
            raise ValueError("A source cannot change family identity")
        families[identity] = row["family_id"]
        if row["label_origin"] not in {"algorithmic_weak", "agent_weak"} or row["label"] not in {
            "A",
            "B",
            "N",
        }:
            raise ValueError("Critic labels must explicitly be weak A/B/N supervision")
        if not isinstance(row["provenance"], dict):
            raise ValueError("Weak-label derivation provenance must be explicit")
        data._canonical(row["provenance"])
        mismatch = row["provenance"].get("reason") == "source_mismatch"
        displayed = owner
        if mismatch:
            displayed = authority.get(row["provenance"].get("mismatched_source_id"))
            if (
                row["label"] != "N"
                or row.get("control") != "source_mismatch"
                or row["provenance"].get("displayed_source_parity") is not False
                or displayed is None
                or displayed["split"]["split"] != "train"
                or displayed["split"]["family"] == owner["split"]["family"]
            ):
                raise ValueError(
                    "Source mismatch requires N, explicit false parity, and another TRAIN family"
                )
        elif row.get("control") == "source_mismatch":
            raise ValueError("Source mismatch control needs explicit mismatch provenance")
        _rights(row["rights"], owner["receipt"]["catalogue_entry"])
        target = _canvas(row["target_canvas"])
        source = _artifact(
            input.parent,
            row["source"],
            True,
            _canvas(owner["source"]["canvas"]),
            snapshots,
            render_canvas=_canvas(displayed["source"]["canvas"]),
        )
        if source["metadata"]["render_sha256"] != displayed["receipt"]["source_render_sha256"]:
            raise ValueError("Displayed source PNG differs from the exact frozen source pixels")
        if (
            source["design"] != owner["source"]
            or semantic_source_digest(source["design"]) in forbidden_hashes
        ):
            raise ValueError("Source JSON differs from frozen original TRAIN authority")
        candidates = {
            key: _artifact(input.parent, row[key], False, target, snapshots) for key in ("A", "B")
        }
        for candidate in candidates.values():
            if semantic_source_digest(candidate["design"]) in forbidden_hashes:
                raise ValueError("A validation/reserved original entered candidate evidence")
            if candidate["metadata"]["render_sha256"] in forbidden_render_hashes:
                raise ValueError("A validation/reserved source PNG entered candidate evidence")
        if identity not in source_resources:
            roots = _object(raw["resource_roots"].get(identity), {"assets_dir", "fonts_dir"})
            assets_dir, fonts_dir = (
                safe_artifact_path(input.parent, roots[key]) for key in ("assets_dir", "fonts_dir")
            )
            resources = (
                data._resource_receipts(source["design"], assets_dir.parent.parent)
                if assets_dir.name == "assets" and fonts_dir == assets_dir.parent / "fonts"
                else None
            )
            if resources is None:
                raise ValueError(
                    "Resource roots must be sibling resources/assets and "
                    "resources/fonts directories"
                )
            if resources != owner["receipt"]["resources"]:
                raise ValueError(
                    "Source asset/font bytes differ from the frozen training authority"
                )
            legal = data._legal_receipts(owner["receipt"]["catalogue_entry"], catalogue_root)
            if legal != owner["receipt"]["legal_files_sha256"]:
                raise ValueError(
                    "Original source license/notice bytes differ from frozen authority"
                )
            for name, value in legal.items():
                snapshots[str(safe_artifact_path(catalogue_root, name))] = value
            for kind, folder in (("assets", assets_dir), ("fonts", fonts_dir)):
                for name, value in resources[kind].items():
                    snapshots[str(safe_artifact_path(folder, name))] = value
            source_resources[identity] = {
                "roots": {"assets": assets_dir, "fonts": fonts_dir},
                "hashes": resources,
                "legal": legal,
            }
        resource = source_resources[identity]
        with using_resources(
            assets_dir=resource["roots"]["assets"], fonts_dir=resource["roots"]["fonts"]
        ):
            context = build_context(source["design"])
            for candidate in candidates.values():
                violations = check_contract(source["design"], candidate["design"], target, context)
                bounds = strictly_in_bounds(candidate["design"])
                candidate["computed"] = {
                    "contract_valid": not violations,
                    "strictly_in_bounds": bounds,
                    "valid": not violations and bounds,
                    "contract_violations": [asdict(item) for item in violations],
                }
        if row["label"] in {"A", "B"} and not candidates[row["label"]]["computed"]["valid"]:
            raise ValueError("A/B weak label selects a contract-invalid or out-of-frame candidate")
        if (
            row["label"] == "N"
            and any(candidate["computed"]["valid"] for candidate in candidates.values())
            and row["provenance"].get("reason") not in {"ambiguity", "source_mismatch"}
        ):
            raise ValueError(
                "NEITHER with a valid candidate needs explicit weak ambiguity/source_mismatch"
            )
        if row["provenance"].get("reason") == "source_mismatch":
            other = authority.get(row["provenance"].get("mismatched_source_id"))
            if (
                row["label"] != "N"
                or other is None
                or other["split"]["split"] != "train"
                or other is owner
            ):
                raise ValueError(
                    "Mismatched-source diagnostics may use only another frozen TRAIN family"
                )
        displayed_legal = data._legal_receipts(
            displayed["receipt"]["catalogue_entry"], catalogue_root
        )
        if displayed_legal != displayed["receipt"]["legal_files_sha256"]:
            raise ValueError("Displayed source license/notice bytes differ from frozen authority")
        for name, value in displayed_legal.items():
            snapshots[str(safe_artifact_path(catalogue_root, name))] = value
        prepared.append(
            {
                "row": deepcopy(row),
                "source": source,
                "candidates": candidates,
                "displayed": displayed,
                "source_render_parity": not mismatch,
                "displayed_legal": displayed_legal,
            }
        )
    if sum(Path(path).stat().st_size for path in snapshots) > data.MAX_TOTAL_BYTES:
        raise ValueError("Evidence exceeds its total byte budget")
    if any(_snapshot_sha(Path(path)) != value for path, value in snapshots.items()):
        raise ValueError("Immutable evidence changed before freezing")
    out.mkdir(parents=True)
    records, receipts, copied = [], [], {}

    def copy(path, folder):
        value = snapshots[str(path)]
        relative = f"{folder}/{value}{path.suffix.lower()}"
        if relative not in copied:
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if data._sha(destination) != value:
                raise ValueError("Immutable evidence changed while copying")
            copied[relative] = value
        return {"path": relative, "sha256": value}

    try:
        generation_copy = copy(generation["path"], "producer-evidence") if generation else None
        input_copy = copy(input, "producer-evidence")
        for item in prepared:
            row = item["row"]
            images = [copy(item["source"]["image_path"], "images")]
            images.extend(
                copy(item["candidates"][key]["image_path"], "images") for key in ("A", "B")
            )
            designs = {
                "source": copy(item["source"]["design_path"], "designs"),
                **{
                    key: copy(item["candidates"][key]["design_path"], "designs")
                    for key in ("A", "B")
                },
            }
            receipt_relative = "receipts/" + data._digest(row["record_id"])[:24] + ".json"
            receipt = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "record_id": row["record_id"],
                "pair_id": row["pair_id"],
                "source_id": row["source_id"],
                "family_id": row["family_id"],
                "split": "train",
                "source_compact_digest": semantic_source_digest(item["source"]["design"]),
                "source_render_parity": item["source_render_parity"],
                "original_source_render_sha256": authority[row["source_id"]]["receipt"][
                    "source_render_sha256"
                ],
                "displayed_source_id": item["displayed"]["split"]["source_id"],
                "displayed_source_family": item["displayed"]["split"]["family"],
                "displayed_source_canvas": item["displayed"]["source"]["canvas"],
                "displayed_source_render_sha256": item["source"]["metadata"]["render_sha256"],
                "displayed_source_catalogue_entry": item["displayed"]["receipt"]["catalogue_entry"],
                "source_original_ids": [e["id"] for e in item["source"]["design"]["elements"]],
                "target_canvas": row["target_canvas"],
                "designs": designs,
                "declared_candidate_metadata": {
                    key: item["candidates"][key]["metadata"] for key in ("A", "B")
                },
                "computed_candidate_checks": {
                    key: item["candidates"][key]["computed"] for key in ("A", "B")
                },
                "rights": row["rights"],
                "provenance": row["provenance"],
                "resources": {
                    kind: {
                        name: copy(
                            safe_artifact_path(
                                source_resources[row["source_id"]]["roots"][kind], name
                            ),
                            "resources",
                        )
                        for name in values
                    }
                    for kind, values in source_resources[row["source_id"]]["hashes"].items()
                },
                "legal_files": {
                    name: copy(safe_artifact_path(catalogue_root, name), "licenses")
                    for name in source_resources[row["source_id"]]["legal"]
                },
                "human_preference": False,
                "quality_approved": False,
                "displayed_source_legal_files": {
                    name: copy(safe_artifact_path(catalogue_root, name), "licenses")
                    for name in item["displayed_legal"]
                },
            }
            data._write(out / receipt_relative, receipt)
            receipts.append({"path": receipt_relative, "sha256": data._sha(out / receipt_relative)})
            record = {
                "record_id": row["record_id"],
                "pair_id": row["pair_id"],
                "original_pair_id": row["pair_id"],
                "source_id": row["source_id"],
                "family_id": row["family_id"],
                "split": "train",
                "training_eligible": generation is not None,
                "experimental_only": True,
                "label": row["label"],
                "label_origin": row["label_origin"],
                "images": images,
                "target_canvas": row["target_canvas"],
                "receipt": receipt_relative,
                "control": row.get("control", row["provenance"].get("reason", "original")),
                "augmentation": "original",
                "human_preference": False,
                "source_render_parity": item["source_render_parity"],
                "displayed_source_id": item["displayed"]["split"]["source_id"],
                "displayed_source_family": item["displayed"]["split"]["family"],
            }
            records.append(record)
            if swap:
                swapped = deepcopy(record)
                swapped.update(
                    record_id=data._digest([row["record_id"], "swap-ab-v1"]),
                    pair_id=data._digest([row["pair_id"], "swap-ab-v1"]),
                    label={"A": "B", "B": "A", "N": "N"}[row["label"]],
                    images=[images[0], images[2], images[1]],
                    augmentation="swap-ab-v1",
                )
                records.append(swapped)
        if any(_snapshot_sha(Path(path)) != value for path, value in snapshots.items()):
            raise ValueError("Immutable evidence changed during freezing")
        if len({record["record_id"] for record in records}) != len(records):
            raise ValueError("Augmented record IDs collide")
        result = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "ready_weak_train_only" if generation else "review_only_unbound_generation",
            "records": records,
            "receipts": receipts,
            "source_manifest_sha256": snapshots[str(source_manifest)],
            "split_manifest_sha256": snapshots[str(split_manifest)],
            "frozen_split": frozen["frozen_split"],
            "input_sha256": snapshots[str(input)],
            "input_copy": input_copy,
            "generation_receipt": generation_copy,
            "generation_receipt_verified": generation is not None,
            "native_png_json_linkage": "traceable_producer_attestation_not_independent_pixel_proof"
            if generation
            else "unbound_self_hashed_artifacts_not_training_eligible",
            "input_files_sha256": snapshots,
            "immutable_inputs_verified": True,
            "human_preference": False,
            "quality_approved": False,
            "native_renders_generated": False,
            "model_executed": False,
            "counts": {
                "original_pairs": len(prepared),
                "records_including_swaps": len(records),
                "sources": len(families),
                "families": len(set(families.values())),
                "labels": dict(Counter(record["label"] for record in records)),
            },
            "limitations": [
                "Algorithmic/agent weak labels are not human aesthetic preferences.",
                "Deterministic checks establish contract/bounds, not visual quality.",
                "No validation or test records are supplied by this train-only experiment.",
                "Swapped views share original pair lineage and are not independent observations.",
                "Producer receipts bind claimed native generation; "
                "pixels are not independently rerendered.",
                "Easy corrupted layouts do not establish taste or resized fidelity; "
                "goods retain readability and spacing limitations.",
            ],
            "files_sha256": {**copied, **{row["path"]: row["sha256"] for row in receipts}},
        }
        data._write(out / "manifest.json", result)
        data._write(
            out / "READY.json",
            {
                "protocol": PROTOCOL,
                "manifest_sha256": data._sha(out / "manifest.json"),
                "training_eligible": generation is not None,
                "weak_only": True,
            },
        )
        return result
    except BaseException:
        data._write(
            out / "INCOMPLETE.json",
            {"training_eligible": False, "reason": "Freeze failed; use a fresh output."},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--generation-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = freeze_pairs(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Critic pairs not admitted: {exc}\n")
    print(
        json.dumps(
            {"status": result["status"], "counts": result["counts"], "out": str(args.out)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
