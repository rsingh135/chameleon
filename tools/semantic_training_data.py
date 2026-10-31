"""Admit explicitly reviewed source-analysis labels to a weak-supervision experiment.

Reads prepared native pixels only. It never renders, resizes a design, runs a
model, changes inspection artifacts, or treats agent review as human taste.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path, PureWindowsPath

from PIL import Image

from engine.design_semantics import (
    describe_design,
    semantic_source_digest,
    validate_semantic_annotations,
)
from engine.visual_analysis import analysis_json_schema, analysis_prompt, bind_analysis
from engine.visual_pipeline import _source_summary
from engine.visual_plan import source_digest as analysis_source_digest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "semantic-analysis-training-data-v1"
ALLOWED_SOURCES = frozenset(
    {
        "doory_bold-sale-announcement",
        "doory_botanical-celebration-invitation",
        "doory_grand-opening-flyer",
        "doory_night-bloom-concert-poster",
        "doory_rooftop-event-story",
        "opensuse_trifold_front",
        "opensuse_booth_poster",
        "opensuse_cheatsheet",
    }
)
FORBIDDEN_SOURCES = frozenset({"opensuse_banner_we_believe", "doory_modern-resume"})
MAX_JSON_BYTES = 24 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_TOTAL_PIXELS = 64_000_000
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_CONTEXT_SIDE = 256
CODE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "engine/design_semantics.py",
    ROOT / "engine/visual_analysis.py",
    ROOT / "engine/visual_pipeline.py",
    ROOT / "engine/visual_plan.py",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError(f"Missing or oversized input file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _read(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds its byte budget: {path}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("JSON values must be finite")

    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid or excessively nested UTF-8 JSON") from exc


def _within(root, relative):
    if (
        not isinstance(relative, str)
        or not relative
        or PureWindowsPath(relative).is_absolute()
        or PureWindowsPath(relative).drive
    ):
        raise ValueError("Input references must be relative paths")
    result = (Path(root) / relative).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Input reference escapes its immutable root")
    return result


def _split(path):
    manifest = _read(path)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("protocol") != "semantic-analysis-family-split-v1"
        or not isinstance(manifest.get("sources"), list)
    ):
        raise ValueError("A frozen semantic-analysis family split is required")
    rows, family_splits = {}, {}
    for row in manifest["sources"]:
        if not isinstance(row, dict) or set(row) != {"source_id", "family", "split"}:
            raise ValueError("Invalid frozen source split row")
        identity, family, split = row["source_id"], row["family"], row["split"]
        if (
            not isinstance(identity, str)
            or identity not in ALLOWED_SOURCES
            or identity in rows
            or not isinstance(family, str)
            or not family.strip()
            or not isinstance(split, str)
            or split not in {"train", "validation"}
        ):
            raise ValueError("Frozen split contains duplicate, forbidden or invalid sources")
        if family in family_splits and family_splits[family] != split:
            raise ValueError("A family cannot cross train and validation")
        family_splits[family] = split
        rows[identity] = deepcopy(row)
    if set(rows) != ALLOWED_SOURCES:
        raise ValueError("Frozen split must declare exactly the eight reviewed sources")
    return manifest, rows


def _resource_receipts(source, prepared):
    assets, fonts = {}, {}
    assets_root, fonts_root = prepared / "resources/assets", prepared / "resources/fonts"
    if not assets_root.is_dir() or not fonts_root.is_dir():
        raise ValueError("Prepared native source needs asset and font directories")
    for element in source["elements"]:
        if element["type"] == "image":
            name = element.get("asset_id")
            assets[name] = _sha(_within(assets_root, name))
    for path in sorted(fonts_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf", ".json"}:
            if not path.resolve().is_relative_to(fonts_root.resolve()) or len(fonts) >= 200:
                raise ValueError("Font references escape their root or exceed their file budget")
            fonts[path.relative_to(fonts_root).as_posix()] = _sha(path)
    return {"assets": assets, "fonts": fonts}


def _legal_receipts(entry, catalogue_root):
    files = {}

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"license_path", "notice_path", "third_party_path", "provenance_path"}:
                    files[child] = _sha(_within(catalogue_root, child))
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    if not entry.get("license_id") or not entry.get("attribution_text"):
        raise ValueError("Reviewed source needs explicit licensing and attribution")
    visit(entry)
    return files


def _annotation(path, source, identity, render_sha):
    annotation = _read(path)
    required = {
        "source_id",
        "source_digest",
        "source_render_sha256",
        "supervision_origin",
        "reviewer",
        "reviewed_element_ids",
        "uncertain_element_ids",
        "purpose_proposal",
        "analysis_raw",
        "limitations",
    }
    if (
        not isinstance(annotation, dict)
        or not required <= set(annotation)
        or set(annotation)
        - required
        - {"source_render_path", "source_render_kind", "reference_render_sha256"}
    ):
        raise ValueError("Annotation wrapper has missing or unknown fields")
    if (
        annotation["source_id"] != identity
        or annotation["source_digest"] != semantic_source_digest(source)
        or annotation["source_render_sha256"] != render_sha
        or annotation["supervision_origin"] != "agent_visual_review"
        or not isinstance(annotation["reviewer"], str)
        or not 1 <= len(annotation["reviewer"].strip()) <= 200
    ):
        raise ValueError("Annotation provenance does not bind the exact source and native pixels")
    ids = {element["id"] for element in source["elements"]}
    masks = []
    for key in ("reviewed_element_ids", "uncertain_element_ids"):
        values = annotation[key]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or value not in ids for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("Annotation review masks must contain unique original IDs")
        masks.append(set(values))
    if masks[0] | masks[1] != ids:
        raise ValueError("Every original element must be explicitly reviewed or uncertain")
    limitations = annotation["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or len(limitations) > 32
        or any(
            not isinstance(value, str) or not 1 <= len(value.strip()) <= 1000
            for value in limitations
        )
    ):
        raise ValueError("Agent review needs explicit bounded limitations")
    proposal = validate_semantic_annotations(source, annotation["purpose_proposal"])
    summary = _source_summary(source)
    bound = bind_analysis(
        annotation["analysis_raw"],
        source_digest=analysis_source_digest(source),
        source_render_sha256=render_sha,
        known_element_ids=[row["id"] for row in summary["elements"]],
    )
    analysis_prompt(summary)
    return annotation, proposal, summary, bound


def _context_box(element, size):
    width, height = size
    x, y, w, h = (element[key] for key in ("x", "y", "width", "height"))
    padding = max(8, min(width, height) * 0.025)
    left, top = max(0, math.floor(x - padding)), max(0, math.floor(y - padding))
    right, bottom = min(width, math.ceil(x + w + padding)), min(height, math.ceil(y + h + padding))
    if right <= left or bottom <= top:
        # Off-canvas objects have no visible isolated pixels. Keep nearest context,
        # disclose visibility, and mask their element training target.
        center_x, center_y = min(width - 1, max(0, x + w / 2)), min(height - 1, max(0, y + h / 2))
        left, top = max(0, math.floor(center_x - padding)), max(0, math.floor(center_y - padding))
        right, bottom = (
            min(width, math.ceil(center_x + padding)),
            min(height, math.ceil(center_y + padding)),
        )
    return [left, top, right, bottom]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _jsonl(path, rows):
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _opaque(summary, raw, seed):
    summary, raw = deepcopy(summary), deepcopy(raw)
    mapping = {
        row["id"]: "e_" + hashlib.sha256((seed + "\0" + row["id"]).encode()).hexdigest()[:16]
        for row in summary["elements"]
    }
    for row in summary["elements"]:
        row["id"] = mapping[row["id"]]
    for observation in raw["observations"]:
        observation["element_ids"] = [mapping[value] for value in observation["element_ids"]]
    return summary, raw, {value: key for key, value in mapping.items()}


def build_dataset(
    *, catalogue, annotations, split_manifest, out, allow_partial=False, opaque_ids=False
):
    """Validate all admitted rows before writing a fresh, source-bound experiment."""
    catalogue, annotations = Path(catalogue).resolve(), Path(annotations).resolve()
    split_manifest, out = Path(split_manifest).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Dataset output must be a fresh directory")
    immutable = [catalogue.parent, annotations, split_manifest]
    if any(out.is_relative_to(path) or path.is_relative_to(out) for path in immutable):
        raise ValueError("Dataset output must not overlap immutable input roots")
    snapshots = {str(path): _sha(path) for path in (catalogue, split_manifest, *CODE_PATHS)}
    frozen, splits = _split(split_manifest)
    catalogue_data = _read(catalogue)
    entries = catalogue_data.get("examples") if isinstance(catalogue_data, dict) else None
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise ValueError("Catalogue must contain source entries")
    identities = [row.get("id") for row in entries]
    if any(not isinstance(value, str) for value in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("Catalogue source IDs must be unique strings")
    entries = {row["id"]: row for row in entries}
    forbidden_hashes = set()
    for identity, entry in entries.items():
        if (entry.get("split") == "reserve" or identity in FORBIDDEN_SOURCES) and entry.get(
            "prepared_path"
        ):
            source_path = _within(catalogue.parent, entry["prepared_path"]) / "source.json"
            if source_path.is_file():
                snapshots[str(source_path)] = _sha(source_path)
                forbidden_hashes.add(semantic_source_digest(_read(source_path)))
    missing = [
        dict(splits[identity], reason="missing_annotation")
        for identity in sorted(splits)
        if not (annotations / f"{identity}.json").is_file()
    ]
    if missing and not allow_partial:
        raise ValueError(
            "Missing reviewed annotations: " + ", ".join(row["source_id"] for row in missing)
        )
    prepared_rows, total_pixels, checked_files = [], 0, {catalogue, split_manifest}
    for identity in sorted(splits):
        annotation_path = annotations / f"{identity}.json"
        if not annotation_path.is_file():
            continue
        entry = entries.get(identity, {})
        if entry.get("split") != "reviewed" or identity in FORBIDDEN_SOURCES:
            raise ValueError("Only the eight explicitly reviewed, nonreserved sources are admitted")
        prepared = _within(catalogue.parent, entry.get("prepared_path"))
        source_path = _within(prepared, "source.json")
        snapshots[str(source_path)] = _sha(source_path)
        snapshots[str(annotation_path)] = _sha(annotation_path)
        declared_annotation = _read(annotation_path)
        if not isinstance(declared_annotation, dict):
            raise ValueError("Annotation wrapper must be an object")
        render_relative = declared_annotation.get("source_render_path", "native.png")
        image_path = _within(prepared, render_relative)
        if image_path.suffix.lower() != ".png":
            raise ValueError("Declared source render must use a PNG path")
        render_kind = {
            "native.png": "prepared_native_reference",
            "imported.png": "imported_source_native_resolution",
        }.get(render_relative, "reviewer_declared_original_source_render")
        reference_path = _within(prepared, "native.png")
        reference_sha = _sha(reference_path) if reference_path.is_file() else None
        if reference_sha is not None:
            snapshots[str(reference_path)] = reference_sha
            checked_files.add(reference_path)
        if (
            declared_annotation.get("source_render_kind", render_kind) != render_kind
            or declared_annotation.get("reference_render_sha256", reference_sha) != reference_sha
        ):
            raise ValueError(
                "Declared source render kind or reference SHA does not match its pixels"
            )
        source = _read(source_path)
        descriptor = describe_design(
            source
        )  # Native frame rules only; no renderer/font measurements.
        compact = semantic_source_digest(source)
        if compact in forbidden_hashes:
            raise ValueError("A reserved source was aliased under a reviewed source ID")
        render_sha = _sha(image_path)
        snapshots[str(image_path)] = render_sha
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.width * image.height > MAX_PIXELS:
                raise ValueError("Native source must be a bounded original PNG")
            if tuple(source["canvas"][key] for key in ("width", "height")) != image.size:
                raise ValueError("Native PNG dimensions differ from the original source canvas")
            image.verify()
            total_pixels += image.width * image.height
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError("Dataset exceeds its native pixel budget")
        annotation, proposal, summary, bound = _annotation(
            annotation_path, source, identity, render_sha
        )
        resources = _resource_receipts(source, prepared)
        legal = _legal_receipts(entry, catalogue.parent)
        original = _within(catalogue.parent, entry.get("original_path"))
        snapshots[str(original)] = _sha(original)
        if snapshots[str(original)] != entry.get("sha256"):
            raise ValueError("Original authored source does not match its catalogue receipt")
        checked_files.update({source_path, image_path, annotation_path, original})
        checked_files.update(
            _within(prepared / "resources/assets", name) for name in resources["assets"]
        )
        checked_files.update(
            _within(prepared / "resources/fonts", name) for name in resources["fonts"]
        )
        checked_files.update(_within(catalogue.parent, name) for name in legal)
        for kind, folder in (("assets", "resources/assets"), ("fonts", "resources/fonts")):
            for name, value in resources[kind].items():
                snapshots[str(_within(prepared / folder, name))] = value
        for name, value in legal.items():
            snapshots[str(_within(catalogue.parent, name))] = value
        prepared_rows.append(
            dict(
                identity=identity,
                entry=entry,
                source=source,
                source_path=source_path,
                image_path=image_path,
                prepared=prepared,
                descriptor=descriptor,
                annotation=annotation,
                annotation_path=annotation_path,
                proposal=proposal,
                summary=summary,
                bound=bound,
                resources=resources,
                legal=legal,
                render_relative=render_relative,
                render_kind=render_kind,
                reference_sha=reference_sha,
            )
        )
    if not prepared_rows:
        raise ValueError("No annotated source is available for this experiment")
    if sum(path.stat().st_size for path in checked_files) > MAX_TOTAL_BYTES:
        raise ValueError("Dataset inputs exceed their total byte budget")
    if any(_sha(Path(path)) != value for path, value in snapshots.items()):
        raise ValueError("An immutable input changed during dataset admission")
    # Exact sources/renders cannot cross families/splits. Shared assets and fonts
    # are disclosed below, rather than disguised as independent unseen brands.
    for key in ("source_digest", "source_render_sha256"):
        owners = defaultdict(set)
        for row in prepared_rows:
            owners[row["bound"][key]].add(splits[row["identity"]]["split"])
        if any(len(values) > 1 for values in owners.values()):
            raise ValueError("Exact original source or native render leaks across splits")
    out.mkdir(parents=True)
    analyses, elements, relationships, receipts = [], [], [], []
    try:
        for row in prepared_rows:
            identity, source, annotation = row["identity"], row["source"], row["annotation"]
            split = splits[identity]["split"]
            key = semantic_source_digest(source)
            prefix = key[:20]
            image_relative = f"images/{prefix}.png"
            (out / "images").mkdir(exist_ok=True)
            shutil.copyfile(row["image_path"], out / image_relative)
            if _sha(out / image_relative) != row["bound"]["source_render_sha256"]:
                raise ValueError("Native source bytes changed while copying the training image")
            receipt_relative = f"receipts/{prefix}.json"
            (out / "receipts").mkdir(exist_ok=True)
            shutil.copyfile(row["source_path"], out / f"receipts/{prefix}.source.json")
            shutil.copyfile(row["annotation_path"], out / f"receipts/{prefix}.annotation.json")
            common = {
                "split": split,
                "receipt": receipt_relative,
                "supervision_origin": "agent_visual_review",
                "experimental_only": True,
                "training_eligible": split == "train",
            }
            variants = [("original_ids", row["summary"], annotation["analysis_raw"])]
            reverse_map = None
            if opaque_ids:
                summary, raw, reverse_map = _opaque(row["summary"], annotation["analysis_raw"], key)
                variants.append(("opaque_ids", summary, raw))
                _write(out / f"id-maps/{prefix}.json", reverse_map)
            for variant, summary, raw in variants:
                bound = bind_analysis(
                    raw,
                    source_digest=row["bound"]["source_digest"],
                    source_render_sha256=row["bound"]["source_render_sha256"],
                    known_element_ids=[value["id"] for value in summary["elements"]],
                )
                analyses.append(
                    {
                        **common,
                        "record_id": _digest([key, "analysis", variant]),
                        "variant": variant,
                        "input": {
                            "source_image": image_relative,
                            "source_render_sha256": row["bound"]["source_render_sha256"],
                            "source_summary": summary,
                            "prompt": analysis_prompt(summary),
                            "response_schema": analysis_json_schema(
                                [value["id"] for value in summary["elements"]]
                            ),
                        },
                        "target": raw,
                        "validated_binding": bound,
                    }
                )
            labels = {value["element_id"]: value for value in row["proposal"]["elements"]}
            reviewed, uncertain = (
                set(annotation["reviewed_element_ids"]),
                set(annotation["uncertain_element_ids"]),
            )
            with Image.open(row["image_path"]) as image:
                for index, element in enumerate(source["elements"]):
                    label = labels[element["id"]]
                    context_box = _context_box(element, image.size)
                    crop = image.crop(context_box)
                    crop.thumbnail((MAX_CONTEXT_SIDE, MAX_CONTEXT_SIDE), Image.Resampling.LANCZOS)
                    crop_relative = f"context-crops/{prefix}/{index:03d}.png"
                    (out / crop_relative).parent.mkdir(parents=True, exist_ok=True)
                    crop.save(out / crop_relative, format="PNG")
                    visible = (
                        element["x"] < image.width
                        and element["y"] < image.height
                        and element["x"] + element["width"] > 0
                        and element["y"] + element["height"] > 0
                    )
                    mask = element["id"] in reviewed and element["id"] not in uncertain and visible
                    elements.append(
                        {
                            **common,
                            "record_id": _digest([key, "element", index]),
                            "training_eligible": split == "train" and mask,
                            "input": {
                                "source_image": image_relative,
                                "context_image": crop_relative,
                                "bbox_normalized": [
                                    element[k] / source["canvas"][axis]
                                    for k, axis in (
                                        ("x", "width"),
                                        ("y", "height"),
                                        ("width", "width"),
                                        ("height", "height"),
                                    )
                                ],
                                "type": element["type"],
                                "text_data": element.get("content", ""),
                                "text_is_untrusted_data": True,
                            },
                            "target": {
                                "purpose": label["purpose"],
                                "confidence": label["confidence"],
                            },
                            "review_mask": mask,
                            "provenance": {
                                "element_index": index,
                                "reviewed": element["id"] in reviewed,
                                "uncertain": element["id"] in uncertain,
                                "visible_in_source_frame": visible,
                                "crop_kind": "native_render_context_not_isolated_asset",
                                "context_box_pixels": context_box,
                                "context_size": list(crop.size),
                                "context_sha256": _sha(out / crop_relative),
                            },
                        }
                    )
            relationships.append(
                {
                    **common,
                    "training_eligible": False,
                    "record_id": _digest([key, "relationships"]),
                    "authority": "soft_proposal_only_no_geometry_or_crop_permissions",
                    "purpose_relations": row["proposal"]["relations"],
                    "source_structure": {
                        "groups": [
                            {"element_id": element["id"], "group_id": element.get("group_id")}
                            for element in source["elements"]
                            if element.get("group_id") is not None
                        ],
                        "anchors": [
                            {"element_id": value["element_id"], "anchors": value["anchors"]}
                            for value in row["descriptor"]["elements"]
                        ],
                        "relations": row["descriptor"]["relations"],
                    },
                    "feature_use": "artifact_only_not_training_features",
                }
            )
            receipt = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                **splits[identity],
                "source_digest_protocol": "canonical-compact-v1",
                "source_digest": key,
                "analysis_source_digest_protocol": "sorted-spaced-json-v1",
                "analysis_source_digest": row["bound"]["source_digest"],
                "source_json_sha256": snapshots[str(row["source_path"])],
                "source_render_sha256": row["bound"]["source_render_sha256"],
                "source_canvas": source["canvas"],
                "original_element_ids": [value["id"] for value in source["elements"]],
                "source_path": str(row["source_path"]),
                "source_render_path": row["render_relative"],
                "source_render_absolute_path": str(row["image_path"]),
                "source_render_kind": row["render_kind"],
                "canonical_original_reference_sha256": row["reference_sha"],
                "reference_byte_identity_claimed": row["reference_sha"]
                == row["bound"]["source_render_sha256"],
                "annotation_path": str(row["annotation_path"]),
                "annotation_sha256": snapshots[str(row["annotation_path"])],
                "reviewer": annotation["reviewer"],
                "limitations": annotation["limitations"],
                "resources": row["resources"],
                "legal_files_sha256": row["legal"],
                "catalogue_entry": row["entry"],
                "purpose_proposal": row["proposal"],
                "original_inspection_training_eligibility_unchanged": True,
                "training_admission": "WEAKSUPERVISED_EXPERIMENT_ONLY",
                "human_reviewed": False,
                "quality_promotion": False,
            }
            _write(out / receipt_relative, receipt)
            receipts.append(
                {
                    "source_id": identity,
                    "split": split,
                    "receipt": receipt_relative,
                    "sha256": _sha(out / receipt_relative),
                    "element_count": len(source["elements"]),
                }
            )
        sharing = []
        for kind in ("assets", "fonts"):
            hashes = defaultdict(list)
            for row in prepared_rows:
                for name, value in row["resources"][kind].items():
                    hashes[value].append(
                        {
                            "source_id": row["identity"],
                            "split": splits[row["identity"]]["split"],
                            "resource": name,
                        }
                    )
            sharing.extend(
                {"kind": kind, "sha256": value, "uses": uses}
                for value, uses in sorted(hashes.items())
                if len({use["split"] for use in uses}) > 1
            )
        _jsonl(out / "analysis.jsonl", analyses)
        _jsonl(out / "elements.jsonl", elements)
        _jsonl(out / "relationships.jsonl", relationships)
        shutil.copyfile(split_manifest, out / "frozen-splits.json")
        code = {path.relative_to(ROOT).as_posix(): snapshots[str(path)] for path in CODE_PATHS}
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "partial_explicit_omissions" if missing else "complete_reviewed_annotations",
            "admission": "WEAKSUPERVISED_EXPERIMENT_ONLY",
            "human_reviewed": False,
            "quality_promotion": False,
            "model_executed": False,
            "native_renders_generated": False,
            "split_manifest_sha256": snapshots[str(split_manifest)],
            "frozen_split": frozen,
            "catalogue_sha256": snapshots[str(catalogue)],
            "sources": receipts,
            "omissions": missing,
            "counts": {
                "sources": len(receipts),
                "analysis_records": len(analyses),
                "element_records": len(elements),
                "train_elements": sum(row["training_eligible"] for row in elements),
                "validation_elements": sum(row["split"] == "validation" for row in elements),
            },
            "variants": ["original_ids", "opaque_ids"] if opaque_ids else ["original_ids"],
            "cross_split_shared_resources": sharing,
            "code_sha256": code,
            "limitations": [
                "Agent visual review is weak supervision, not human taste or quality acceptance.",
                "Validation labels must never train or calibrate a model.",
                "Shared brands, fonts and assets prevent claims of unseen-brand generalization.",
                "Context crops include other layers and are not isolated source assets.",
                "Rule-derived source anchors are diagnostic artifacts, "
                "not reviewed training labels.",
                "Original IDs occur only as response grounding and receipt artifacts; "
                "element features exclude IDs and campaign names.",
            ],
        }
        if any(_sha(Path(path)) != value for path, value in snapshots.items()):
            raise ValueError("An immutable input changed while building this dataset")
        manifest["files_sha256"] = {
            path.relative_to(out).as_posix(): _sha(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        }
        _write(out / "manifest.json", manifest)
        return manifest
    except BaseException:
        # Keep failed work visible and ineligible; never silently reuse it.
        _write(
            out / "INCOMPLETE.json",
            {
                "training_eligible": False,
                "reason": "Dataset build failed; inspect error and choose a fresh output.",
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue", type=Path, default=ROOT / "fixtures/open_posters/catalogue.json"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--opaque-ids", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = build_dataset(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Dataset not admitted: {exc}\n")
    print(
        json.dumps(
            {"status": manifest["status"], "counts": manifest["counts"], "out": str(args.out)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
