"""Rebuild the reviewed development pilot from tracked original source files.

Only source rendering occurs here. No model, resize target or reserve example
is executed. All generated receipts stay under the repository's output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.scene_harness import inspect_scene, load_input, validate_model_response

from .dataset import ROOT, _identifier, _reserves, canonical, digest, load_records, read_json


def _local(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Pilot paths must be nonempty repository-relative strings")
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError("Pilot inputs must stay inside the repository")
    return path


def prepare(labels, out):
    labels, out = Path(labels).resolve(), Path(out).resolve()
    if not out.is_relative_to(ROOT / "output"):
        raise ValueError("Generated pilot data must stay under repository output/")
    payload = read_json(labels)
    sources = payload.get("sources")
    if (
        payload.get("schema_version") != 1
        or not isinstance(sources, list)
        or not 1 <= len(sources) <= 20
    ):
        raise ValueError("Pilot labels need schema_version 1 and at most20 sources")
    reserve_ids, reserve_hashes = _reserves()
    seen, records = set(), []
    for entry in sources:
        identity = _identifier(entry["source_id"], "source_id")
        if identity in seen or identity in reserve_ids or entry["source_digest"] in reserve_hashes:
            raise ValueError("Duplicate or reserved pilot source")
        seen.add(identity)
        if entry.get("supervision_origin") != "agent_visual_review":
            raise ValueError("This pilot rebuild preserves its explicit agent-review provenance")
        if entry.get("source_parity_reviewed") is not True or not entry.get("license"):
            raise ValueError("Pilot source needs reviewed parity and rights provenance")
        directory = out / identity
        source, assets, fonts, _ = load_input(
            source_path=_local(entry["source_path"]),
            assets_dir=_local(entry["assets_dir"]),
            fonts_dir=_local(entry["fonts_dir"]),
            out=directory,
        )
        if digest(source) != entry["source_digest"]:
            raise ValueError("Pilot source changed since visual annotation")
        report = inspect_scene(source, assets, fonts, directory)
        if not report["implementation_unchanged"]:
            raise ValueError("Implementation changed during source preparation")
        profile = read_json(directory / "scene.json")
        response = {
            "schema_version": 1,
            "source_digest": profile["source_digest"],
            "scene_digest": profile["scene_digest"],
            "images": entry["images"],
        }
        response = validate_model_response(response, profile)
        if {r["element_id"] for r in response["images"]} != {
            r["element_id"] for r in profile["images"]
        }:
            raise ValueError("Every source image needs its reviewed pilot role")
        label_path = directory / "agent-labels.json"
        label_path.write_text(canonical(response) + "\n", encoding="utf-8")
        for image in response["images"]:
            records.append(
                {
                    "record_id": identity + "--" + image["element_id"],
                    "source_id": identity,
                    "family_id": entry["family_id"],
                    "split": entry["split"],
                    "harness_dir": str(directory.relative_to(out)),
                    "labels": str(label_path.relative_to(out)),
                    "element_id": image["element_id"],
                    "supervision_origin": "agent_visual_review",
                    "reviewed_by": entry["reviewed_by"],
                    "rights_accepted": True,
                    "source_parity_reviewed": True,
                }
            )
    manifest = out / "manifest.json"
    manifest.write_text(
        canonical({"schema_version": 1, "records": records}) + "\n", encoding="utf-8"
    )
    rows = load_records(manifest)
    return {
        "manifest": str(manifest),
        "sources": len(sources),
        "records": len(rows),
        "human_review_occurred": False,
        "quality_validated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "fixtures/scene_model/pilot-labels.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "output/scene-model-pilot-data")
    args = parser.parse_args(argv)
    try:
        print(canonical(prepare(args.labels, args.out)))
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Pilot preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
