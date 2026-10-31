"""Collect immutable exact-resource layout evidence without changing selection.

The admitted pool is the default. A broader contract-valid shadow collection
requires --pool shadow; it does not authorize those candidates for selection.
No labels, model downloads, cloud commands or production policy changes occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path

from experiments.vision_critic.contracts import canonical_json_sha256, parse_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
INPUT_KEYS = {"schema_version", "purpose", "sources"}
SOURCE_KEYS = {
    "source_id",
    "family_id",
    "split",
    "source",
    "assets_dir",
    "fonts_dir",
    "targets",
    "source_parity_verified",
    "training_rights_reviewed",
    "rights_reference",
}


def _json(path: Path):
    return _json_bytes(path)[0]


def _json_bytes(path: Path) -> tuple[dict, str]:
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("JSON input exceeds 16 MiB")
    payload = path.read_bytes()
    if len(payload) > 16 * 1024**2:
        raise ValueError("JSON input exceeds 16 MiB")
    return parse_json(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save_image(image, path: Path) -> None:
    with path.open("wb") as stream:
        image.save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())


def _render_verified(renderer, design: dict):
    fingerprint = canonical_json_sha256(design)
    supplied = deepcopy(design)
    image = renderer(supplied)
    if canonical_json_sha256(supplied) != fingerprint or (
        canonical_json_sha256(design) != fingerprint
    ):
        raise ValueError("Renderer mutated design evidence")
    return image


def _canvas(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError("Canvas must have exactly width and height")
    if any(type(number) is not int or not 1 <= number <= 8192 for number in value.values()):
        raise ValueError("Canvas dimensions must be integers from 1 to 8192")
    if value["width"] * value["height"] > 20_000_000:
        raise ValueError("Canvas exceeds the 20-megapixel training-image limit")
    return dict(value)


def code_provenance() -> dict:
    """Hash working-tree implementation bytes, including uncommitted modules."""
    files = {}
    for folder in ("engine", "renderer/converter"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path.is_symlink():
                raise ValueError("Implementation files must not be symlinks")
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name in ("collect.py", "contracts.py"):
        path = Path(__file__).with_name(name)
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        revision = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    return {
        "engine_revision": revision,
        "working_tree_digest": canonical_json_sha256(files),
        "code_files": files,
    }


def resource_provenance(assets_dir: Path, fonts_dir: Path) -> dict:
    files = {}
    total_bytes = 0
    for label, directory in (("assets", assets_dir), ("fonts", fonts_dir)):
        if not directory.is_dir():
            raise ValueError(f"Missing resource directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Resource trees must not contain symlinks")
            if not path.is_file():
                continue
            total_bytes += path.stat().st_size
            if len(files) >= 4096 or total_bytes > 1024**3:
                raise ValueError("Resource hashing exceeds the explicit collection limit")
            files[f"{label}/{path.relative_to(directory).as_posix()}"] = sha256_file(path)
    return {"resource_digest": canonical_json_sha256(files), "resource_files": files}


def validate_source_resources(source: dict, context) -> None:
    """Refuse missing pixels/faces instead of collecting renderer substitutions."""
    from experiments.vision_critic.contracts import safe_artifact_path

    texts = [element for element in source["elements"] if element["type"] == "text"]
    for element in source["elements"]:
        if element["type"] == "image":
            path = safe_artifact_path(context.assets_dir, element["asset_id"])
            if not path.is_file():
                raise ValueError(f"Missing source image bytes: {element['asset_id']}")
    if not texts:
        return
    from cr_renderer.fonts import get_font_style, get_font_weight, normalize_family

    from engine.resources import load_font_manager

    manager = load_font_manager(context.fonts_dir)
    for element in texts:
        family = element["font_family"]
        weight = element.get("font_weight", "regular")
        italic = element.get("font_style") == "italic"
        style = (
            "bolditalic"
            if weight == "bold" and italic
            else "bold"
            if weight == "bold"
            else "italic"
            if italic
            else "regular"
        )
        expected_weight = get_font_weight(family, weight)
        expected_style = get_font_style(family, style)
        faces = manager._fonts.get(normalize_family(family), [])
        if not any(
            face.get("fontWeight", "regular") == expected_weight
            and face.get("fontStyle", "regular") == expected_style
            for face in faces
        ):
            raise ValueError(f"Exact source font face is unavailable: {family}/{weight}/{style}")


def reject_reserve_source(source_path: Path, source: dict, source_id: str) -> None:
    """Guard the existing reserve pack by ID, path and exact design fingerprints."""
    catalogue_path = ROOT / "fixtures/open_posters/catalogue.json"
    catalogue = _json(catalogue_path)
    digest = canonical_json_sha256(source)
    for entry in catalogue["examples"]:
        if entry["split"] != "reserve":
            continue
        prepared = catalogue_path.parent / "prepared" / entry["id"]
        original = catalogue_path.parent / entry["original_path"]
        if (
            source_id == entry["id"]
            or source_path == original.resolve()
            or (source_path.is_relative_to(prepared.resolve()))
        ):
            raise ValueError("The existing ten reserve designs must remain untouched")
        prepared_source = prepared / "source.json"
        if prepared_source.is_file() and canonical_json_sha256(_json(prepared_source)) == digest:
            raise ValueError("Copied reserve design detected; use a separate development corpus")


def read_input(path: Path, max_cases: int = 24, *, fingerprints: dict | None = None) -> dict:
    data, input_digest = _json_bytes(path)
    if not isinstance(data, dict) or set(data) != INPUT_KEYS or data["schema_version"] != 1:
        raise ValueError("Collection input requires schema_version, purpose and sources only")
    if data["purpose"] not in {"research", "synthetic"}:
        raise ValueError("Collection purpose must be research or synthetic")
    if not isinstance(data["sources"], list) or not data["sources"]:
        raise ValueError("Supply at least one explicit source")
    seen = set()
    families = {}
    designs = {}
    cases = 0
    source_files, source_data = {}, {}
    for entry in data["sources"]:
        required = SOURCE_KEYS - {"assets_dir", "fonts_dir"}
        if not isinstance(entry, dict) or set(entry) - SOURCE_KEYS or required - set(entry):
            raise ValueError("Source entry has missing or unknown fields")
        for key in ("source_id", "family_id", "source"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"Source needs a nonempty {key}")
        identity, family, split = entry["source_id"], entry["family_id"], entry["split"]
        if identity in seen or split not in {"train", "validation", "test"}:
            raise ValueError("Source IDs must be unique and splits explicit")
        if family in families and families[family] != split:
            raise ValueError("Source family leaks across splits")
        for key in ("source_parity_verified", "training_rights_reviewed"):
            if type(entry[key]) is not bool:
                raise ValueError("Review flags must be explicit booleans")
        if not isinstance(entry["rights_reference"], str):
            raise ValueError("rights_reference must be a string, possibly empty pending review")
        source_path = (path.parent / entry["source"]).resolve()
        source, source_digest = _json_bytes(source_path)
        if not isinstance(source, dict) or "elements" not in source:
            raise ValueError("Source must be design JSON with canvas and elements")
        _canvas(source["canvas"])
        reject_reserve_source(source_path, source, identity)
        digest = canonical_json_sha256(source)
        if digest in designs and designs[digest] != split:
            raise ValueError("Identical source content leaks across splits under different IDs")
        if not isinstance(entry["targets"], list) or not entry["targets"]:
            raise ValueError("Provide explicit target canvases")
        targets = [canonical_json_sha256(_canvas(target)) for target in entry["targets"]]
        if len(targets) != len(set(targets)):
            raise ValueError("Repeated targets would duplicate a case")
        cases += len(targets)
        seen.add(identity)
        families[family] = split
        designs[digest] = split
        source_files[identity] = source_digest
        source_data[identity] = source
    if cases > max_cases:
        raise ValueError(f"Input exceeds the explicit {max_cases}-case limit")
    if fingerprints is not None:
        fingerprints.update(
            input_manifest_sha256=input_digest, source_files=source_files, source_data=source_data
        )
    return data


def normalize_snapshot(snapshot: dict) -> dict:
    """Adapt the public backend snapshot without reconstructing its safety tier."""
    admitted = snapshot["admitted_finalist_ids"]
    ids = [row["candidate_id"] for row in snapshot["candidates"]]
    if (
        not isinstance(admitted, list)
        or len(set(admitted)) != len(admitted)
        or (
            len(set(ids)) != len(ids)
            or not set(admitted) <= set(ids)
            or snapshot["baseline_candidate_id"] not in admitted
        )
    ):
        raise ValueError("Backend snapshot must have unique IDs and an admitted baseline")
    rows = []
    for original in snapshot["candidates"]:
        if type(original["contract_valid"]) is not bool or type(original["admitted"]) is not bool:
            raise ValueError("Backend snapshot must give explicit validity/admission")
        if original["admitted"] != (original["candidate_id"] in admitted):
            raise ValueError("Backend snapshot admission metadata is inconsistent")
        row = dict(original)
        row["diagnostics"] = {
            **original["diagnostics"],
            "contract_valid": original["contract_valid"],
            "contract_violations": []
            if original["contract_valid"]
            else [{"code": "backend_contract_rejected"}],
        }
        rows.append(row)
    return {
        "baseline_candidate_id": snapshot["baseline_candidate_id"],
        "admitted_candidate_ids": list(admitted),
        "candidates": rows,
    }


def choose_candidates(snapshot: dict, limit: int, pool: str) -> list[dict]:
    """Keep baseline and state diversity without changing backend admission."""
    rows = snapshot["candidates"]
    baseline = snapshot["baseline_candidate_id"]
    admitted = set(snapshot["admitted_candidate_ids"])
    valid = [
        row
        for row in rows
        if row["diagnostics"]["contract_valid"]
        and not row["diagnostics"]["contract_violations"]
        and (pool == "shadow" or row["candidate_id"] in admitted)
    ]
    by_id = {row["candidate_id"]: row for row in valid}
    if baseline not in by_id:
        raise ValueError("The authoritative baseline is not a valid member of this pool")
    selected = [by_id.pop(baseline)]
    while by_id and len(selected) < limit:
        topologies = {item.get("state", {}).get("topology") for item in selected}
        row = min(
            by_id.values(),
            key=lambda item: (
                item.get("state", {}).get("topology") in topologies,
                item["candidate_id"],
            ),
        )
        selected.append(by_id.pop(row["candidate_id"]))
    return selected


def collect(
    manifest: Path,
    destination: Path,
    *,
    pool: str = "admitted",
    candidate_limit: int = 4,
    max_cases: int = 24,
    snapshot_provider=None,
    renderer=None,
) -> dict:
    """Publish a new collection atomically; never overwrite old evidence."""
    from engine.resources import using_resources
    from experiments.vision_critic.contracts import load_collection

    if (
        pool not in {"admitted", "shadow"}
        or type(candidate_limit) is not int
        or not (2 <= candidate_limit <= 8)
    ):
        raise ValueError("Choose admitted/shadow and 2..8 candidates")
    if snapshot_provider is None:
        from engine.resize import collect_candidate_snapshot

        snapshot_provider = collect_candidate_snapshot
    if renderer is None:
        from engine.eval.render import render_image

        renderer = render_image
    manifest, destination = manifest.resolve(), destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Collection destination must not exist")
    reviewed = {}
    data = read_input(manifest, max_cases, fingerprints=reviewed)
    implementation = code_provenance()
    source_files = reviewed["source_files"]
    input_digest = reviewed["input_manifest_sha256"]
    if sha256_file(manifest) != input_digest:
        raise ValueError("Input manifest changed after prerequisite review")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (".vision-collection-" + uuid.uuid4().hex)
    stage.mkdir()
    collection = {
        "schema_version": 1,
        "collection_id": "",
        "purpose": data["purpose"],
        "provenance": {
            **implementation,
            "policy_version": "backend-admitted-v1",
            "pool": pool,
            "input_manifest_sha256": input_digest,
            "source_files": source_files,
        },
        "cases": [],
    }
    try:
        for entry in data["sources"]:
            source_path = (manifest.parent / entry["source"]).resolve()
            if sha256_file(source_path) != source_files[entry["source_id"]]:
                raise ValueError("Source file changed after prerequisite review")
            source = deepcopy(reviewed["source_data"][entry["source_id"]])
            resources = {
                key: (manifest.parent / entry[key]).resolve()
                for key in ("assets_dir", "fonts_dir")
                if key in entry
            }
            with using_resources(**resources) as context:
                resource_before = resource_provenance(context.assets_dir, context.fonts_dir)
                validate_source_resources(source, context)
                source_image = _render_verified(renderer, source)
                if source_image.size != tuple(source["canvas"][key] for key in ("width", "height")):
                    raise ValueError("Source render does not match its exact canvas")
                for target in entry["targets"]:
                    source_copy, target_copy = deepcopy(source), deepcopy(target)
                    supplied = snapshot_provider(source_copy, target_copy)
                    if canonical_json_sha256(source_copy) != canonical_json_sha256(source) or (
                        target_copy != target
                    ):
                        raise ValueError("Snapshot provider mutated collection input")
                    snapshot = normalize_snapshot(supplied)
                    rows = choose_candidates(snapshot, candidate_limit, pool)
                    from engine.eval import build_context, check_contract

                    evaluation_context = build_context(source)
                    for row in rows:
                        if check_contract(source, row["design"], target, evaluation_context):
                            raise ValueError(
                                "Backend supplied contract-invalid collection evidence"
                            )
                    case_id = canonical_json_sha256(
                        {
                            "source_id": entry["source_id"],
                            "source": canonical_json_sha256(source),
                            "target": target,
                        }
                    )
                    folder = stage / "cases" / case_id
                    _write(folder / "source.json", source)
                    _save_image(source_image, folder / "source.png")
                    case = {
                        key: entry[key]
                        for key in (
                            "source_id",
                            "family_id",
                            "split",
                            "source_parity_verified",
                            "training_rights_reviewed",
                            "rights_reference",
                        )
                    }
                    prefix = f"cases/{case_id}"
                    case.update(
                        {
                            "case_id": case_id,
                            "target_canvas": target,
                            "source": {
                                "design": f"{prefix}/source.json",
                                "json_sha256": canonical_json_sha256(source),
                                "render": f"{prefix}/source.png",
                                "render_sha256": sha256_file(folder / "source.png"),
                            },
                            "baseline_candidate_id": snapshot["baseline_candidate_id"],
                            "admitted_candidate_ids": [],
                            "candidates": [],
                        }
                    )
                    for index, row in enumerate(rows):
                        name = f"candidate-{index:02d}"
                        _write(folder / (name + ".json"), row["design"])
                        image = _render_verified(renderer, row["design"])
                        if image.size != (target["width"], target["height"]):
                            raise ValueError("Candidate render does not match target canvas")
                        _save_image(image, folder / (name + ".png"))
                        admitted = row["candidate_id"] in snapshot["admitted_candidate_ids"]
                        if admitted:
                            case["admitted_candidate_ids"].append(row["candidate_id"])
                        case["candidates"].append(
                            {
                                "candidate_id": row["candidate_id"],
                                "design": f"{prefix}/{name}.json",
                                "design_sha256": canonical_json_sha256(row["design"]),
                                "render": f"{prefix}/{name}.png",
                                "render_sha256": sha256_file(folder / (name + ".png")),
                                "admitted": admitted,
                                "diagnostics": row["diagnostics"],
                                "provenance": {
                                    "state": row.get("state", {}),
                                    "origins": row.get("provenance", []),
                                    "selection": row.get("selection", {}),
                                    **resource_before,
                                    "element_ids": [e["id"] for e in row["design"]["elements"]],
                                },
                            }
                        )
                    collection["cases"].append(case)
                if resource_provenance(context.assets_dir, context.fonts_dir) != resource_before:
                    raise ValueError(
                        "Resources changed while collecting; evidence was not published"
                    )
                if sha256_file(source_path) != source_files[entry["source_id"]]:
                    raise ValueError("Source file changed during collection")
        if sha256_file(manifest) != input_digest:
            raise ValueError("Input manifest changed during collection")
        if implementation != code_provenance():
            raise ValueError("Implementation changed during collection; retry a frozen revision")
        unhashed = {key: value for key, value in collection.items() if key != "collection_id"}
        collection["collection_id"] = canonical_json_sha256(unhashed)
        _write(stage / "collection.json", collection)
        load_collection(stage)
        if destination.exists():
            raise ValueError("Collection destination appeared during publication")
        stage.rename(destination)
        return {
            "collection_dir": str(destination),
            "collection_id": collection["collection_id"],
            "case_count": len(collection["cases"]),
            "purpose": data["purpose"],
            "pool": pool,
            "production_selection_changed": False,
        }
    finally:
        # Exclusively created staging stays in the destination parent. Never delete its siblings.
        if (
            stage.exists()
            and not stage.is_symlink()
            and (stage.resolve().parent == destination.parent.resolve())
            and (stage.name.startswith(".vision-collection-"))
        ):
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool", choices=("admitted", "shadow"), default="admitted")
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=24)
    args = parser.parse_args()
    try:
        report = collect(
            args.manifest,
            args.out,
            pool=args.pool,
            candidate_limit=args.candidate_limit,
            max_cases=args.max_cases,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, json.dumps({"error": str(error)}) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
