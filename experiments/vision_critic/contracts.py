"""Versioned, standard-library contracts for source-aware preference artifacts.

Design hashes use sorted-key, compact UTF-8 JSON, not raw file formatting. Render
hashes cover actual file bytes. Artifact paths never escape the collection root.
These checks establish provenance/integrity, not aesthetic quality or rights.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
DECISIONS = ("A", "B", "tie", "neither", "abstain")
SPLITS = ("train", "validation", "test")
ISSUE_CODES = (
    "unreadable_text",
    "weak_hierarchy",
    "detached_action",
    "broken_reading_order",
    "source_style_loss",
    "poor_balance",
    "poor_spacing",
    "weak_product_prominence",
    "bad_overlap",
    "distorted_ornament",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
MAX_LABEL_BYTES = 64 * 1024


class ValidationError(ValueError):
    """Artifact metadata, identity, provenance or bytes failed validation."""


def canonical_json(value) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValidationError("Value is not finite, serializable JSON") from error


def canonical_json_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str):
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: _invalid(f"Nonfinite JSON: {value}"),
        )
    except json.JSONDecodeError as error:
        raise ValidationError("Invalid JSON") from error


def _invalid(message: str):
    raise ValidationError(message)


def _object(value, fields: set[str], where: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValidationError(f"{where} must contain exactly {sorted(fields)}")
    return value


def _text(value, where: str, *, empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 4096
        or (not empty and not value.strip())
        or any(ord(character) < 32 for character in value)
    ):
        raise ValidationError(f"Invalid {where}")
    return value


def _sha(value, where: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValidationError(f"{where} must be lowercase SHA-256")
    return value


def _bool(value, where: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{where} must be a boolean")
    return value


def _list(value, where: str) -> list:
    if not isinstance(value, list):
        raise ValidationError(f"{where} must be a list")
    return value


def _strings(value, where: str) -> list[str]:
    values = _list(value, where)
    for item in values:
        _text(item, where)
    if len(set(values)) != len(values):
        raise ValidationError(f"{where} must not contain duplicates")
    return values


def _relative(value) -> str:
    _text(value, "artifact path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValidationError("Artifact paths must be normalized relative POSIX paths")
    for part in path.parts:
        if (
            part.endswith((".", " "))
            or any(character in '<>"|?*' for character in part)
            or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)
        ):
            raise ValidationError("Artifact path is not portable")
    if not path.parts or value == ".":
        raise ValidationError("An artifact file path is required")
    return value


def safe_artifact_path(root: str | Path, relative: str) -> Path:
    """Resolve a file without allowing symlink/reparse traversal or root escape."""
    relative = _relative(relative)
    root = Path(root).absolute()
    target = root.joinpath(*PurePosixPath(relative).parts)
    for component in (*reversed(target.parents), target):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            raise ValidationError("Artifact paths must not contain links or reparse points")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValidationError("Artifact escaped its collection root")
    return target


def make_pair_id(collection_id: str, case_id: str, a: str, b: str) -> str:
    for value in (collection_id, case_id, a, b):
        _text(value, "pair identity")
    if a == b:
        raise ValidationError("A pair needs two different candidates")
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "collection_id": collection_id,
            "case_id": case_id,
            "candidate_a_id": a,
            "candidate_b_id": b,
        }
    )


def _artifact(metadata: dict, *, source: bool, root: Path | None, canvas=None) -> None:
    design_key = "json_sha256" if source else "design_sha256"
    for field in ("design", "render"):
        _relative(metadata[field])
    _sha(metadata[design_key], design_key)
    _sha(metadata["render_sha256"], "render_sha256")
    if root is None:
        return
    design_path = safe_artifact_path(root, metadata["design"])
    render_path = safe_artifact_path(root, metadata["render"])
    for path in (design_path, render_path):
        if not path.is_file():
            raise ValidationError(f"Missing artifact file: {path}")
    design = parse_json(design_path.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValidationError("Design artifact must be a JSON object")
    if canonical_json_sha256(design) != metadata[design_key]:
        raise ValidationError(f"Stale design hash: {metadata['design']}")
    if sha256_file(render_path) != metadata["render_sha256"]:
        raise ValidationError(f"Stale render hash: {metadata['render']}")
    if canvas is not None and (
        not isinstance(design.get("canvas"), dict)
        or any(design["canvas"].get(key) != value for key, value in canvas.items())
    ):
        raise ValidationError("Candidate canvas differs from the collection target")
    if not source and "element_ids" in metadata["provenance"]:
        elements = design.get("elements")
        if not isinstance(elements, list) or not all(isinstance(item, dict) for item in elements):
            raise ValidationError("Design elements must be a list of objects")
        actual = [element.get("id") for element in elements]
        _strings(actual, "artifact element IDs")
        if set(actual) != set(metadata["provenance"]["element_ids"]):
            raise ValidationError("Candidate element IDs differ from design artifact")


def validate_collection(data: dict, root: str | Path | None = None) -> dict:
    _object(
        data, {"schema_version", "collection_id", "purpose", "provenance", "cases"}, "collection"
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("Unsupported collection schema")
    _text(data["collection_id"], "collection_id")
    if not isinstance(data["purpose"], str) or data["purpose"] not in {"synthetic", "research"}:
        raise ValidationError("Collection purpose must be synthetic or research")
    provenance = data["provenance"]
    if not isinstance(provenance, dict) or not {
        "engine_revision",
        "working_tree_digest",
        "policy_version",
    }.issubset(provenance):
        raise ValidationError("Collection engine/policy provenance is required")
    _text(provenance["engine_revision"], "engine_revision")
    _sha(provenance["working_tree_digest"], "working_tree_digest")
    _text(provenance["policy_version"], "policy_version")
    canonical_json(provenance)
    cases = _list(data["cases"], "cases")
    case_ids, families, sources, source_hash_splits, source_artifacts = set(), {}, {}, {}, {}
    for case in cases:
        _object(
            case,
            {
                "case_id",
                "source_id",
                "family_id",
                "split",
                "source_parity_verified",
                "training_rights_reviewed",
                "rights_reference",
                "target_canvas",
                "source",
                "baseline_candidate_id",
                "admitted_candidate_ids",
                "candidates",
            },
            "case",
        )
        for field in ("case_id", "source_id", "family_id", "baseline_candidate_id"):
            _text(case[field], field)
        if case["case_id"] in case_ids:
            raise ValidationError("Duplicate case_id")
        case_ids.add(case["case_id"])
        if case["split"] not in SPLITS:
            raise ValidationError("Unknown case split")
        family, split, source_id = case["family_id"], case["split"], case["source_id"]
        if family in families and families[family] != split:
            raise ValidationError("Family leakage across splits")
        families[family] = split
        if source_id in sources and sources[source_id] != (family, split):
            raise ValidationError("Source identity changes family or split")
        sources[source_id] = family, split
        _bool(case["source_parity_verified"], "source_parity_verified")
        _bool(case["training_rights_reviewed"], "training_rights_reviewed")
        _text(case["rights_reference"], "rights_reference", empty=True)
        if case["training_rights_reviewed"] and not case["rights_reference"].strip():
            raise ValidationError("Reviewed training rights need a reference")
        canvas = _object(case["target_canvas"], {"width", "height"}, "target_canvas")
        if any(type(value) is not int or value <= 0 for value in canvas.values()):
            raise ValidationError("Target dimensions must be positive integers")
        source = _object(
            case["source"], {"design", "json_sha256", "render", "render_sha256"}, "source"
        )
        _artifact(source, source=True, root=Path(root) if root is not None else None)
        identity = source["json_sha256"], source["render_sha256"]
        if source_id in source_artifacts and source_artifacts[source_id] != identity:
            raise ValidationError("Source ID refers to inconsistent original artifacts")
        source_artifacts[source_id] = identity
        for digest in (source["json_sha256"], source["render_sha256"]):
            if digest in source_hash_splits and source_hash_splits[digest] != split:
                raise ValidationError("Identical source artifacts leak across splits")
            source_hash_splits[digest] = split
        admitted = _strings(case["admitted_candidate_ids"], "admitted_candidate_ids")
        ids, designs, flags = set(), set(), set()
        candidates = _list(case["candidates"], "candidates")
        if not candidates:
            raise ValidationError("A case needs at least one candidate")
        for candidate in candidates:
            _object(
                candidate,
                {
                    "candidate_id",
                    "design",
                    "design_sha256",
                    "render",
                    "render_sha256",
                    "admitted",
                    "diagnostics",
                    "provenance",
                },
                "candidate",
            )
            candidate_id = _text(candidate["candidate_id"], "candidate_id")
            _sha(candidate["design_sha256"], "design_sha256")
            if candidate_id in ids or candidate["design_sha256"] in designs:
                raise ValidationError("Duplicate candidate identity or design within a case")
            ids.add(candidate_id)
            designs.add(candidate["design_sha256"])
            _bool(candidate["admitted"], "candidate.admitted")
            if not isinstance(candidate["diagnostics"], dict) or not isinstance(
                candidate["provenance"], dict
            ):
                raise ValidationError("Candidate diagnostics and provenance must be objects")
            canonical_json(candidate["diagnostics"])
            canonical_json(candidate["provenance"])
            if "element_ids" in candidate["provenance"]:
                _strings(candidate["provenance"]["element_ids"], "candidate element_ids")
            if candidate["admitted"]:
                if (
                    candidate["diagnostics"].get("contract_valid") is not True
                    or candidate["diagnostics"].get("contract_violations") != []
                ):
                    raise ValidationError(
                        "Admitted candidate must explicitly pass the hard contract"
                    )
                flags.add(candidate_id)
            _artifact(
                candidate,
                source=False,
                root=Path(root) if root is not None else None,
                canvas=canvas,
            )
        if case["baseline_candidate_id"] not in ids:
            raise ValidationError("Baseline candidate is absent")
        if set(admitted) != flags:
            raise ValidationError("Admitted candidate IDs disagree with candidate flags")
    return deepcopy(data)


def load_collection(root: str | Path, verify_files: bool = True) -> dict:
    path = safe_artifact_path(root, "collection.json")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValidationError("Collection manifest exceeds 64 MiB")
    data = parse_json(path.read_text(encoding="utf-8"))
    return validate_collection(data, root=root if verify_files else None)


def validate_grounded_issues(issues, candidates: dict[str, list[str]]) -> list[dict]:
    """Validate the same bounded issue objects for human labels and model output."""
    _object(candidates, {"A", "B"}, "candidate element IDs")
    for side in ("A", "B"):
        _strings(candidates[side], f"candidate {side} element IDs")
    issues = _list(issues, "issues")
    if len(issues) > 32:
        raise ValidationError("At most 32 grounded issues are allowed")
    seen = set()
    for issue in issues:
        _object(issue, {"candidate", "code", "element_ids"}, "issue")
        if (
            not isinstance(issue["candidate"], str)
            or issue["candidate"] not in {"A", "B"}
            or issue["code"] not in ISSUE_CODES
        ):
            raise ValidationError("Unknown issue candidate or code")
        element_ids = _strings(issue["element_ids"], "issue element_ids")
        if len(element_ids) > 128:
            raise ValidationError("Too many issue element IDs")
        if not set(element_ids).issubset(candidates[issue["candidate"]]):
            raise ValidationError("Issue references unknown candidate element IDs")
        signature = canonical_json(issue)
        if signature in seen:
            raise ValidationError("Duplicate issue")
        seen.add(signature)
    if len(canonical_json(issues).encode("utf-8")) > MAX_LABEL_BYTES:
        raise ValidationError("Issues exceed 64 KiB")
    return deepcopy(issues)


def validate_label(label: dict, collection: dict, *, allow_synthetic: bool = False) -> dict:
    if type(allow_synthetic) is not bool:
        raise ValidationError("allow_synthetic must be an explicit boolean")
    _object(
        label,
        {
            "schema_version",
            "pair_id",
            "case_id",
            "collection_id",
            "candidate_a_id",
            "candidate_b_id",
            "decision",
            "issues",
            "annotator",
            "created_at",
            "source_render_sha256",
            "a_render_sha256",
            "b_render_sha256",
        },
        "label",
    )
    if type(label["schema_version"]) is not int or label["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("Unsupported label schema")
    if label["collection_id"] != collection["collection_id"]:
        raise ValidationError("Label belongs to another collection")
    synthetic = collection["purpose"] == "synthetic"
    if synthetic and not allow_synthetic:
        raise ValidationError("Synthetic labeling requires explicit allow_synthetic=True")
    cases = [case for case in collection["cases"] if case["case_id"] == label["case_id"]]
    if len(cases) != 1:
        raise ValidationError("Unknown label case_id")
    case = cases[0]
    if not case["source_parity_verified"] and not (synthetic and allow_synthetic):
        raise ValidationError("Source parity must be verified before labeling")
    candidates = {item["candidate_id"]: item for item in case["candidates"]}
    a, b = label["candidate_a_id"], label["candidate_b_id"]
    if (
        not isinstance(a, str)
        or not isinstance(b, str)
        or a not in candidates
        or b not in candidates
    ):
        raise ValidationError("Unknown label candidate ID")
    if label["pair_id"] != make_pair_id(collection["collection_id"], case["case_id"], a, b):
        raise ValidationError("Incorrect oriented pair_id")
    if label["decision"] not in DECISIONS:
        raise ValidationError("Unknown label decision")
    _text(label["annotator"], "annotator")
    _text(label["created_at"], "created_at")
    try:
        timestamp = datetime.fromisoformat(label["created_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationError("Invalid label timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValidationError("Label timestamp needs an explicit UTC offset")
    for candidate in (candidates[a], candidates[b]):
        if (
            candidate["diagnostics"].get("contract_valid") is not True
            or candidate["diagnostics"].get("contract_violations") != []
        ):
            raise ValidationError("Only explicitly contract-valid candidates can be labeled")
    expected_hashes = {
        "source_render_sha256": case["source"]["render_sha256"],
        "a_render_sha256": candidates[a]["render_sha256"],
        "b_render_sha256": candidates[b]["render_sha256"],
    }
    if any(label[field] != expected for field, expected in expected_hashes.items()):
        raise ValidationError("Stale label render hashes")
    validate_grounded_issues(
        label["issues"],
        {
            "A": candidates[a]["provenance"].get("element_ids", []),
            "B": candidates[b]["provenance"].get("element_ids", []),
        },
    )
    if len(canonical_json(label).encode("utf-8")) > MAX_LABEL_BYTES:
        raise ValidationError("Label exceeds 64 KiB")
    return deepcopy(label)
