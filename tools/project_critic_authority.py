"""Narrow opt-in authority for three inherited project-synthetic TRAIN sources.

This is not an expansion of the curated public-source allowlist. No public
licence is invented for project artwork. Native pixels remain producer-attested
evidence; the only permitted source derivation is equivalent HEX-to-RGBA colour.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from engine.design_semantics import semantic_source_digest
from experiments.vision_critic.contracts import (
    canonical_json_sha256,
    safe_artifact_path,
    sha256_file,
)
from tools import semantic_training_data as data

PROTOCOL = "project-synthetic-source-authority-v1"
TRANSFORM_VERSION = "hex-to-rgba-equivalent-v1"
RIGHTS_ORIGIN = "project_synthetic_internal_training"
ROOT = Path(__file__).resolve().parents[1]
PARENT_DATASET_SHA256 = "fc2bf3c72b19d01bc00f9ef04ddce3e625c85440ae8adb83039bde87acb09230"
PRODUCER_SHA256 = "fb67e457a62ad5ed069776107047a47751b8fa4e38f63a1cb685208dad23b996"
FONT_SHA256 = "4500567b25fc1af2983cee01c1e57e01a0108b8b3844c182936d14c7dec7f69c"
OFL_SHA256 = "5c4a92c8f8ce0ab211f4c72889d82383c9644546115d9ba39a4f2d5b8f37014b"
PARENTS = {
    "synthetic_materials_brief": (
        "materials_brief",
        "1c6ad27b6ebd1194956b27ee385633e73dc2964ea1f5d899d65aa5fd207e72ec",
    ),
    "synthetic_guild_bulletin": (
        "guild_bulletin",
        "69386d355d7d04c8e3034e98be552d002160604dc35bd3e47f250387728c343f",
    ),
    "synthetic_garden_plan": (
        "garden_plan",
        "c947312d071c633efad3c0162450bd1319685fa19372894fb2f0bcdd01dfd455",
    ),
}
OPERATIONS_ID = "synthetic_operations_digest"
OPERATIONS_COMPACT = "d1c75ef16fb807ee196bb5edbc2c72d65b2b02e3e0be6599e1349b2b4417f9a4"
OPERATIONS_SPACED = "b48deb788fa73cf1702161dfaa89c5ab9a3d4e0719cc06c187d5b41bbd139ace"
COLOR_FIELDS = frozenset({"color", "fill_color", "stroke_color"})


def _object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Project authority has missing or unknown fields")


def _spaced(source):
    import hashlib
    import json

    return hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file(root, descriptor, snapshots):
    _object(descriptor, {"path", "sha256"})
    path = safe_artifact_path(root, descriptor["path"])
    actual = data._sha(path)
    if actual != descriptor["sha256"]:
        raise ValueError("Project authority artifact hash differs")
    snapshots[str(path)] = actual
    return path


def _source_file(root, descriptor, snapshots, *, parent=False):
    fields = {"path", "raw_sha256", "compact_sha256", "spaced_sha256"}
    if parent:
        fields |= {"parent_source_id", "parent_repo_path"}
    _object(descriptor, fields)
    path = _file(root, {"path": descriptor["path"], "sha256": descriptor["raw_sha256"]}, snapshots)
    source = data._read(path)
    if (
        semantic_source_digest(source) != descriptor["compact_sha256"]
        or _spaced(source) != descriptor["spaced_sha256"]
    ):
        raise ValueError("Project source compact/spaced identities differ")
    return path, source


def color_transform_audit(parent, derived):
    """Return exact changed paths; reject geometry, text, identity or style edits."""
    changes = []

    def walk(before, after, path):
        if data._canonical(before) == data._canonical(after):
            return
        if isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after):
            for key in before:
                walk(before[key], after[key], [*path, key])
            return
        if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
            for index, (old, new) in enumerate(zip(before, after, strict=True)):
                walk(old, new, [*path, index])
            return
        match = (
            re.fullmatch(r"#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?", before)
            if isinstance(before, str)
            else None
        )
        rgba = (
            re.fullmatch(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)", after)
            if isinstance(after, str)
            else None
        )
        if not path or path[-1] not in COLOR_FIELDS or not match or not rgba:
            raise ValueError("Project derivation changes a field other than equivalent HEX colour")
        expected = tuple(int(match[1][index : index + 2], 16) for index in (0, 2, 4))
        actual = tuple(int(rgba[index]) for index in (1, 2, 3))
        alpha = float(rgba[4])
        expected_alpha = int(match[2], 16) / 255 if match[2] else 1.0
        if (
            actual != expected
            or not math.isfinite(alpha)
            or not math.isclose(alpha, expected_alpha, abs_tol=1e-9)
        ):
            raise ValueError("Project RGBA normalization changes visible colour channels")
        changes.append({"path": path, "before": before, "after": after})

    walk(parent, derived, [])
    return {
        "version": TRANSFORM_VERSION,
        "parent_compact_sha256": semantic_source_digest(parent),
        "derived_compact_sha256": semantic_source_digest(derived),
        "changes": changes,
    }


def validate_rights(rights):
    _object(
        rights,
        {
            "origin",
            "public_license_granted",
            "authorization_basis",
            "authorization_reference",
            "parent_producer_reference",
            "acquired_utc",
            "acquired_utc_scope",
            "attribution",
        },
    )
    if (
        rights["origin"] != RIGHTS_ORIGIN
        or rights["public_license_granted"] is not False
        or rights["authorization_basis"]
        != "explicit user instruction for internal training/testing"
        or not isinstance(rights["authorization_reference"], str)
        or not rights["authorization_reference"].strip()
        or rights["parent_producer_reference"]
        != "experiments/fit_policy_dataset.py@" + PRODUCER_SHA256
        or rights["acquired_utc_scope"] != "local_evidence_package_acquisition"
        or rights["attribution"]
        != "Chameleon project-authored synthetic fixture; bundled Poppins font under OFL-1.1"
    ):
        raise ValueError(
            "Project rights must be explicit internal authorization, never a public licence"
        )
    when = datetime.fromisoformat(rights["acquired_utc"].replace("Z", "+00:00"))
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("Project acquisition time needs an explicit UTC offset")
    return rights


def _native_receipt(root, descriptor, snapshots):
    receipt_path = _file(root, descriptor, snapshots)
    receipt = data._read(receipt_path)
    code = receipt.get("implementation_before_hashes")
    if (
        receipt.get("protocol") != "native-critic-valid-fit-generation-v3"
        or receipt.get("status") != "complete"
    ):
        raise ValueError("Project native receipt has an unsupported producer protocol")
    if not isinstance(code, dict) or not code or code != receipt.get("implementation_after_hashes"):
        raise ValueError("Project native producer identity is missing or changed")
    for relative, expected in code.items():
        archived = safe_artifact_path(root, "repository-code/" + relative)
        path = archived if archived.is_file() else safe_artifact_path(ROOT, relative)
        if (
            not path.is_file()
            or path.stat().st_size > data.MAX_FILE_BYTES
            or sha256_file(path) != expected
        ):
            raise ValueError("Project native producer code hash differs")
        snapshots[str(path)] = expected
    native_input = safe_artifact_path(root, receipt["core_evidence"])
    if data._sha(native_input) != receipt["core_evidence_raw_sha256"]:
        raise ValueError("Project native receipt input bytes differ")
    snapshots[str(native_input)] = receipt["core_evidence_raw_sha256"]
    core = data._read(native_input)
    if core.get("protocol") != "critic-valid-fit-evidence-v3":
        raise ValueError("Project native source evidence differs from its receipt")
    if core.get("model_executed") is not False or core.get("gpu_used") is not False:
        raise ValueError("Project native authority must disclose model-free source generation")
    return core


def _candidate_bindings(root, row, family, parent_rows, snapshots):
    result = {}
    candidates = row.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {"narrow", "fit1", "fit2"}:
        raise ValueError("Project source needs its three explicitly native-rendered candidates")
    for candidate in candidates.values():
        descriptor = candidate["parent"]
        base = {
            key: descriptor[key]
            for key in ("path", "raw_sha256", "compact_sha256", "spaced_sha256")
        }
        _, original = _source_file(root, base, snapshots)
        matches = [
            item
            for item in parent_rows
            if item.get("family") == family
            and item.get("design_digest") == descriptor.get("dataset_design_digest")
        ]
        if (
            not matches
            or any(item.get("split") != "train" for item in matches)
            or descriptor["compact_sha256"] != descriptor["dataset_design_digest"]
        ):
            raise ValueError("Project candidate is not an exact inherited TRAIN candidate")
        derived_path, derived = _source_file(root, candidate["derived"], snapshots)
        audit = color_transform_audit(original, derived)
        declared = candidate["transform"]
        changes = [
            {key: change[key] for key in ("path", "before", "after")}
            for change in declared["changes"]
        ]
        if declared.get("version") != TRANSFORM_VERSION or changes != audit["changes"]:
            raise ValueError("Project candidate colour-only transform audit differs")
        if candidate.get("design") != derived:
            raise ValueError("Project native candidate metadata differs from its exact JSON")
        image = candidate["render"]
        image_path = _file(root, {"path": image["path"], "sha256": image["sha256"]}, snapshots)
        dimensions = [derived["canvas"][key] for key in ("width", "height")]
        if image.get("dimensions") != dimensions or math.prod(dimensions) > data.MAX_PIXELS:
            raise ValueError("Project candidate native dimensions differ")
        with Image.open(image_path) as picture:
            if picture.format != "PNG" or list(picture.size) != dimensions:
                raise ValueError("Project candidate PNG dimensions differ")
            picture.verify()
        identity = canonical_json_sha256(derived)
        if identity in result:
            raise ValueError("Project candidate geometry must be distinct")
        result[identity] = {
            "render_sha256": image["sha256"],
            "source_path": str(derived_path),
            "parent": descriptor,
            "transform_audit": audit,
        }
    return result


def load_project_authority(path):
    """Return a narrowly checked source registry; never render or edit inputs."""
    path = Path(path).resolve()
    root, document = path.parent, data._read(path)
    snapshots = {str(path): data._sha(path)}
    _object(
        document,
        {
            "schema_version",
            "protocol",
            "parent_dataset",
            "producer",
            "generation_receipt",
            "sources",
        },
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["protocol"] != PROTOCOL
    ):
        raise ValueError("Unsupported optional project source authority")
    parent_path = _file(root, document["parent_dataset"], snapshots)
    producer_path = _file(root, document["producer"], snapshots)
    if (
        snapshots[str(parent_path)] != PARENT_DATASET_SHA256
        or snapshots[str(producer_path)] != PRODUCER_SHA256
    ):
        raise ValueError(
            "Project authority must inherit the exact frozen dataset and authoring producer"
        )
    parent = data._read(parent_path)
    if parent.get("label_kind") != "synthetic_geometry_teacher_v1":
        raise ValueError("Project parents are weak synthetic geometry data")
    parent_rows = parent["records"]
    operations = [item for item in parent_rows if item.get("source_id") == OPERATIONS_ID]
    if not operations or any(
        item.get("split") != "validation" or item.get("source_digest") != OPERATIONS_SPACED
        for item in operations
    ):
        raise ValueError("Operations validation must remain excluded in the frozen parent split")
    native = _native_receipt(root, document["generation_receipt"], snapshots)
    if native.get("parent_dataset", {}).get("raw_sha256") != PARENT_DATASET_SHA256:
        raise ValueError("Project native evidence changes the frozen parent dataset identity")
    native_sources = {row["source_id"]: row for row in native["sources"]}
    if (
        set(native_sources) != {family + "__rgba_v1" for family in PARENTS}
        or len(native["sources"]) != 3
    ):
        raise ValueError(
            "Native source evidence must contain only the three project TRAIN families"
        )
    sources = document["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Project authority permits exactly three inherited TRAIN families")
    owners, seen = {}, set()
    for row in sources:
        _object(
            row,
            {
                "source_id",
                "family",
                "split",
                "parent_source",
                "source",
                "render",
                "resource_roots",
                "resources",
                "legal_files_sha256",
                "transform",
                "rights",
            },
        )
        family = row["family"]
        if (
            family not in PARENTS
            or family in seen
            or row["source_id"] != family + "__rgba_v1"
            or row["split"] != "train"
        ):
            raise ValueError(
                "Project source is not one of the three exact inherited TRAIN families"
            )
        seen.add(family)
        expected_stem, expected_raw = PARENTS[family]
        parent_descriptor = row["parent_source"]
        parent_file, original = _source_file(root, parent_descriptor, snapshots, parent=True)
        if (
            parent_descriptor["parent_source_id"] != family
            or parent_descriptor["parent_repo_path"]
            != f"output/fit-policy-training-expanded/synthetic-sources/{expected_stem}.json"
            or snapshots[str(parent_file)] != expected_raw
        ):
            raise ValueError("Project source does not bind the original authored parent bytes")
        inherited = [item for item in parent_rows if item.get("source_id") == family]
        if not inherited or any(
            item.get("family") != family
            or item.get("split") != "train"
            or item.get("source_digest") != _spaced(original)
            for item in inherited
        ):
            raise ValueError(
                "Project parent differs from its frozen TRAIN split or spaced identity"
            )
        source_path, source = _source_file(root, row["source"], snapshots)
        transform = row["transform"]
        _object(transform, {"version", "path", "sha256"})
        audit_path = _file(
            root, {"path": transform["path"], "sha256": transform["sha256"]}, snapshots
        )
        if transform["version"] != TRANSFORM_VERSION or data._read(
            audit_path
        ) != color_transform_audit(original, source):
            raise ValueError(
                "Project colour transform audit does not match independently checked changes"
            )
        image = row["render"]
        _object(image, {"path", "sha256", "width", "height"})
        image_path = _file(root, {"path": image["path"], "sha256": image["sha256"]}, snapshots)
        canvas = source["canvas"]
        if (
            [image["width"], image["height"]] != [canvas["width"], canvas["height"]]
            or type(image["width"]) is not int
            or type(image["height"]) is not int
            or image["width"] * image["height"] > data.MAX_PIXELS
        ):
            raise ValueError("Project native dimensions differ from the original source canvas")
        with Image.open(image_path) as picture:
            if picture.format != "PNG" or picture.size != (image["width"], image["height"]):
                raise ValueError("Project source is not its declared full native PNG")
            picture.verify()
        bound = native_sources.get(row["source_id"], {})
        if (
            bound.get("family_id") != family
            or bound.get("split") != "train"
            or bound.get("source", {}).get("raw_sha256") != row["source"]["raw_sha256"]
            or bound.get("source_render", {}).get("sha256") != image["sha256"]
            or bound.get("source_render", {}).get("dimensions") != [image["width"], image["height"]]
        ):
            raise ValueError("Project native generation receipt does not bind this source JSON/PNG")
        candidate_bindings = _candidate_bindings(root, bound, family, parent_rows, snapshots)
        _object(row["resource_roots"], {"assets_dir", "fonts_dir"})
        assets, fonts = (
            safe_artifact_path(root, row["resource_roots"][key])
            for key in ("assets_dir", "fonts_dir")
        )
        if assets.name != "assets" or fonts != assets.parent / "fonts":
            raise ValueError("Project resources must use sibling assets/fonts directories")
        resources = data._resource_receipts(source, assets.parent.parent)
        if resources != row["resources"] or resources.get("assets") != {}:
            raise ValueError("Project exact resource map differs or introduces image assets")
        expected_font_keys = {"SolsticeSans-Regular.ttf", "font-manifest.json"}
        if (
            set(resources["fonts"]) != expected_font_keys
            or resources["fonts"]["SolsticeSans-Regular.ttf"] != FONT_SHA256
        ):
            raise ValueError("Project font bytes differ from the inherited exact face")
        manifest = data._read(fonts / "font-manifest.json")
        if manifest != [
            {
                "font_family": "Solstice Sans",
                "file": "SolsticeSans-Regular.ttf",
                "weight": "regular",
                "style": "normal",
            }
        ]:
            raise ValueError("Project font manifest changes the inherited family/face mapping")
        for kind, directory in (("assets", assets), ("fonts", fonts)):
            for name, value in resources[kind].items():
                snapshots[str(safe_artifact_path(directory, name))] = value
        legal = row["legal_files_sha256"]
        if legal != {"licenses/Poppins-OFL.txt": OFL_SHA256}:
            raise ValueError("Project font licence must preserve the actual OFL notice")
        for name, value in legal.items():
            legal_path = safe_artifact_path(root, name)
            if data._sha(legal_path) != value:
                raise ValueError("Project OFL licence bytes differ")
            snapshots[str(legal_path)] = value
        rights = validate_rights(row["rights"])
        receipt = {
            "source_id": row["source_id"],
            "family": family,
            "split": "train",
            "source_digest": semantic_source_digest(source),
            "analysis_source_digest": _spaced(source),
            "source_canvas": canvas,
            "original_element_ids": [e["id"] for e in source["elements"]],
            "source_render_sha256": image["sha256"],
            "source_path": str(source_path),
            "source_json_sha256": snapshots[str(source_path)],
            "resources": resources,
            "legal_files_sha256": legal,
            "rights": rights,
            "parent_source": parent_descriptor,
            "transform": transform,
            "authority_kind": RIGHTS_ORIGIN,
        }
        owners[row["source_id"]] = {
            "receipt": receipt,
            "source": source,
            "split": {"source_id": row["source_id"], "family": family, "split": "train"},
            "files": {},
            "authority_kind": RIGHTS_ORIGIN,
            "legal_root": root,
            "candidate_bindings": candidate_bindings,
        }
    if seen != set(PARENTS):
        raise ValueError("Project source authority omitted an inherited TRAIN family")
    if sum(Path(name).stat().st_size for name in snapshots) > data.MAX_TOTAL_BYTES:
        raise ValueError("Project authority exceeds its total byte budget")
    for name, expected in snapshots.items():
        if sha256_file(name) != expected:
            raise ValueError("Project authority changed during validation")
    return {
        "document": document,
        "owners": owners,
        "snapshots": snapshots,
        "forbidden_source_hashes": {OPERATIONS_COMPACT},
        "forbidden_render_hashes": set(),
        "manifest_path": path,
    }
