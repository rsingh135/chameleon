"""Freeze label-free model inputs and controlled visual-reasoning probes.

Preparation copies original pixels and metadata; it never invokes a model or
opens preference labels. Blank/replacement images are explicitly separate
controls, never substituted into the original artifact. Test inputs are excluded
by default. This is evaluation plumbing, not evidence of visual quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from .contracts import (
    canonical_json,
    canonical_json_sha256,
    load_collection,
    parse_json,
    safe_artifact_path,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "chameleon-independent-eval-v1"
CODE_FILES = (
    "experiments/vision_critic/independent_probes.py",
    "experiments/vision_critic/independent_eval.py",
    "tools/independent_model_eval.py",
    "tools/prepare_semantic_eval.py",
    "experiments/vision_critic/contracts.py",
    "experiments/vision_critic/dataset.py",
    "experiments/vision_critic/train_sft.py",
    "experiments/visual_planner/provider.py",
    "experiments/visual_planner/native.py",
    "experiments/visual_planner/worker.py",
    "experiments/visual_planner/download_model.py",
    "engine/visual_analysis.py",
    "engine/visual_plan.py",
)
digest = canonical_json_sha256


def request_digest(value):
    # The existing provider protocol deliberately uses ASCII-escaped canonical
    # JSON; bundle/model identities use the shared UTF-8 artifact contract.
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _read_json(path, maximum=4 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("Evaluation JSON exceeds its byte budget")
    return parse_json(path.read_text(encoding="utf-8-sig"))


def load_bundle(path):
    """Verify the frozen artifact, including exact bytes and request bindings."""
    path = Path(path).resolve()
    root = path.parent
    bundle = _read_json(path, 32 * 1024 * 1024)
    if (
        not isinstance(bundle, dict)
        or type(bundle.get("schema_version")) is not int
        or bundle["schema_version"] != 1
        or bundle.get("protocol") != PROTOCOL
        or bundle.get("bundle_sha256")
        != digest({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    ):
        raise ValueError("Evaluation bundle identity/protocol mismatch")
    files, probes = bundle.get("files"), bundle.get("probes")
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise ValueError("Evaluation bundle has an invalid file budget")
    if not isinstance(probes, list) or not 1 <= len(probes) <= 200:
        raise ValueError("Evaluation bundle has an invalid probe budget")
    for relative, expected in files.items():
        artifact = safe_artifact_path(root, relative)
        if not artifact.is_file() or artifact.stat().st_size > 32_000_000:
            raise ValueError("Frozen evaluation artifact is missing or oversized")
        if sha256_file(artifact) != expected:
            raise ValueError("Frozen evaluation artifact bytes changed")
    cases, families = {}, {}
    for case in bundle.get("cases", []):
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("Evaluation cases require explicit identities")
        if case["case_id"] in cases or case.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Duplicate case or invalid frozen split")
        family, split = case.get("family_id"), case["split"]
        if not isinstance(family, str) or (family in families and families[family] != split):
            raise ValueError("Evaluation family leaks across frozen splits")
        families[family] = split
        cases[case["case_id"]] = case
        if "source_json_path" in case:
            if case["source_json_path"] not in files:
                raise ValueError("Original source JSON is not frozen")
            source = _read_json(safe_artifact_path(root, case["source_json_path"]))
            if digest(source) != case.get("source_json_sha256"):
                raise ValueError("Original source JSON identity changed")
    known = set()
    for probe in probes:
        if not isinstance(probe, dict) or not isinstance(probe.get("probe_id"), str):
            raise ValueError("Evaluation probes require explicit identities")
        if probe["probe_id"] in known or probe.get("request_path") not in files:
            raise ValueError("Duplicate probe or unbound request")
        known.add(probe["probe_id"])
        case = cases.get(probe.get("case_id"))
        if case is None or any(probe.get(key) != case.get(key) for key in ("family_id", "split")):
            raise ValueError("Probe case/family/split binding differs")
        request = _read_json(safe_artifact_path(root, probe["request_path"]))
        if request_digest(request) != probe.get("request_digest"):
            raise ValueError("Frozen request differs from the probe identity")
        images = [request.get("source_render")]
        images += [row.get("render") for row in request.get("plans", [])]
        for image in images:
            if not isinstance(image, dict) or not isinstance(image.get("path"), str):
                raise ValueError("Probe image reference is invalid")
            image_path = Path(image["path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.is_relative_to(root):
                raise ValueError("Probe image escaped the frozen bundle")
            relative = image_path.relative_to(root).as_posix()
            safe_artifact_path(root, relative)
            if relative not in files or files[relative] != image.get("sha256"):
                raise ValueError("Probe image differs from its frozen bytes")
        if probe.get("source_sha256") != request["source_render"]["sha256"]:
            raise ValueError("Probe source image identity mismatch")
        if request.get("mode") != probe.get("mode"):
            raise ValueError("Probe task mode differs from its request")
        rows = request.get("source_summary", {}).get("elements", [])
        ids = [row.get("id") for row in rows if isinstance(row, dict)]
        element_map = probe.get("element_map", {})
        if (
            len(ids) != len(rows)
            or len(set(ids)) != len(ids)
            or ids != probe.get("known_element_ids")
            or set(element_map) != set(ids)
            or len(set(element_map.values())) != len(ids)
        ):
            raise ValueError("Probe known-element identity map differs from its request")
        menu = request.get("plan_menu")
        if menu is not None and (
            request_digest({key: value for key, value in menu.items() if key != "menu_digest"})
            != menu.get("menu_digest")
            or probe.get("menu_digest") != menu.get("menu_digest")
        ):
            raise ValueError("Probe menu identity mismatch")
        plans = menu.get("plans", []) if menu else []
        plan_ids = [row.get("plan_id") for row in plans if isinstance(row, dict)]
        plan_map = probe.get("plan_map", {})
        if (
            len(plan_ids) != len(plans)
            or len(set(plan_ids)) != len(plan_ids)
            or set(plan_map) != set(plan_ids)
            or len(set(plan_map.values())) != len(plan_ids)
        ):
            raise ValueError("Probe candidate identity map differs from its menu")
    for relative, expected in bundle.get("snapshot", {}).get("code_hashes", {}).items():
        if files.get(relative) != expected:
            raise ValueError("Evaluation code snapshot differs from its frozen files")
    return bundle


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _copy_artifact(path, out, folder, *, expected_sha256=None, json_sha256=None):
    data = Path(path).read_bytes()
    if len(data) > 32_000_000:
        raise ValueError("Evaluation artifact exceeds the byte budget")
    identity = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and identity != expected_sha256:
        raise ValueError("Source/candidate image bytes changed after collection validation")
    if json_sha256 is not None and digest(parse_json(data.decode("utf-8-sig"))) != json_sha256:
        raise ValueError("Source/candidate design changed after collection validation")
    suffix = Path(path).suffix.lower()
    destination = out / folder / f"{identity}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(data)
    elif destination.read_bytes() != data:
        raise ValueError("Content-addressed evaluation copy differs")
    return {"path": str(destination.resolve()), "sha256": identity}


def _summary(source):
    rows = []
    summary = {"source_canvas": deepcopy(source["canvas"]), "elements": rows}
    for element in source["elements"]:
        row = {key: element[key] for key in ("id", "type", "x", "y", "width", "height")}
        if "font_size" in element:
            row["font_size"] = element["font_size"]
        if isinstance(element.get("content"), str):
            row["content"] = element["content"][:80]
        rows.append(row)
        if len(json.dumps(summary, ensure_ascii=True)) > 11000:
            rows.pop()
            break
    if not rows:
        raise ValueError("Source summary cannot fit a known element in its budget")
    summary["omitted_elements"] = len(source["elements"]) - len(rows)
    return summary


def _bind_menu(request):
    if "plan_menu" not in request:
        return
    menu = request["plan_menu"]
    menu["menu_digest"] = request_digest(
        {key: value for key, value in menu.items() if key != "menu_digest"}
    )


def _opaque_request(request, seed):
    result = deepcopy(request)
    element_map = {}
    for row in result["source_summary"]["elements"]:
        original = row["id"]
        opaque = "e_" + digest([seed, "element", original])[:18]
        row["id"] = opaque
        element_map[opaque] = original
    plan_map = {}
    for collection in (result.get("plan_menu", {}).get("plans", []), result.get("plans", [])):
        for row in collection:
            original = row["plan_id"]
            opaque = "p_" + digest([seed, "plan", original])[:18]
            row["plan_id"] = opaque
            row["label"] = "Native candidate"
            row["description"] = "Existing native layout candidate."
            plan_map[opaque] = original
    _bind_menu(result)
    return result, element_map, plan_map


def prepare_collection(
    collection_root, out, *, splits=("validation",), max_cases=4, modes=("critique",)
):
    """Create at most 200 copied-input probes without reading any label file."""
    from engine.visual_plan import source_digest

    if (
        type(max_cases) is not int
        or not 1 <= max_cases <= 20
        or not splits
        or set(splits) - {"train", "validation", "test"}
        or not modes
        or set(modes) - {"critique", "analyze"}
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("Invalid bounded evaluation preparation options")
    root, out = Path(collection_root).resolve(), Path(out).resolve()
    collection_path = root / "collection.json"
    collection_bytes = collection_path.read_bytes()
    if len(collection_bytes) > 32_000_000:
        raise ValueError("Evaluation collection exceeds its byte budget")
    collection = load_collection(root)
    selected = [case for case in collection["cases"] if case["split"] in splits][:max_cases]
    if not selected:
        raise ValueError("No cases match the requested frozen splits")
    if len(selected) * len(modes) * 5 > 200:
        raise ValueError("Evaluation matrix exceeds 200 probes")
    out.mkdir(parents=True, exist_ok=False)
    (out / "input").mkdir()
    (out / "input" / "collection.json").write_bytes(collection_bytes)
    code_hashes, originals = {}, {}
    for relative in CODE_FILES:
        path = ROOT / relative
        data = path.read_bytes()
        originals[relative] = hashlib.sha256(data).hexdigest()
        destination = out / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        code_hashes[destination.relative_to(out).as_posix()] = originals[relative]
    cases, probes, limitations = [], [], []
    for case in selected:
        source_path = safe_artifact_path(root, case["source"]["design"])
        source = _read_json(source_path)
        source_copy = _copy_artifact(
            source_path, out, "data", json_sha256=case["source"]["json_sha256"]
        )
        original_ref = _copy_artifact(
            safe_artifact_path(root, case["source"]["render"]),
            out,
            "images",
            expected_sha256=case["source"]["render_sha256"],
        )
        # Controls never modify the original. The blank has matching dimensions;
        # replacement pixels come unchanged from a different non-test source.
        from PIL import Image

        blank_path = out / "images" / f"blank-{digest(case['source'])[:16]}.png"
        Image.new(
            "RGB", tuple(source["canvas"][key] for key in ("width", "height")), (127, 127, 127)
        ).save(blank_path)
        blank_ref = {"path": str(blank_path), "sha256": sha256_file(blank_path)}
        replacement = next(
            (
                row
                for row in collection["cases"]
                if row["split"] != "test"
                and row["source"]["render_sha256"] != case["source"]["render_sha256"]
                and row["family_id"] != case["family_id"]
            ),
            None,
        )
        replacement_ref = (
            _copy_artifact(
                safe_artifact_path(root, replacement["source"]["render"]),
                out,
                "images",
                expected_sha256=replacement["source"]["render_sha256"],
            )
            if replacement
            else None
        )
        if replacement_ref is None:
            limitations.append(f"{case['case_id']}: no different non-test family replacement")
        baseline = next(
            row
            for row in case["candidates"]
            if row["candidate_id"] == case["baseline_candidate_id"]
        )
        alternative = next(
            (row for row in case["candidates"] if row["candidate_id"] != baseline["candidate_id"]),
            None,
        )
        cases.append(
            {
                "source_json_path": Path(source_copy["path"]).relative_to(out).as_posix(),
                "source_json_sha256": case["source"]["json_sha256"],
                "source_visual_digest": source_digest(source),
                "source_render_sha256": original_ref["sha256"],
                "source_element_ids": [row["id"] for row in source["elements"]],
                "candidate_ids": [row["candidate_id"] for row in case["candidates"]],
                **{
                    key: deepcopy(case[key])
                    for key in (
                        "case_id",
                        "family_id",
                        "split",
                        "baseline_candidate_id",
                        "admitted_candidate_ids",
                    )
                },
            }
        )
        for mode in modes:
            if mode == "critique" and alternative is None:
                limitations.append(f"{case['case_id']}: no second native candidate")
                continue
            request = {
                "schema_version": 1,
                "mode": mode,
                "source_digest": source_digest(source),
                "source_render": original_ref,
                "source_summary": _summary(source),
            }
            if mode == "critique":
                plans, rendered = [], []
                for number, candidate in enumerate((baseline, alternative), 1):
                    _copy_artifact(
                        safe_artifact_path(root, candidate["design"]),
                        out,
                        "data",
                        json_sha256=candidate["design_sha256"],
                    )
                    plan = {
                        "plan_id": candidate["candidate_id"],
                        "label": f"Native candidate {number}",
                        "description": "Existing native layout candidate.",
                    }
                    plans.append(plan)
                    rendered.append(
                        {
                            **plan,
                            "render": _copy_artifact(
                                safe_artifact_path(root, candidate["render"]),
                                out,
                                "images",
                                expected_sha256=candidate["render_sha256"],
                            ),
                        }
                    )
                request.update(
                    target_canvas=deepcopy(case["target_canvas"]),
                    plan_menu={
                        "schema_version": 1,
                        "source_digest": request["source_digest"],
                        "target_canvas": deepcopy(case["target_canvas"]),
                        "plans": plans,
                    },
                    plans=rendered,
                )
                _bind_menu(request)
            variants = [("original", deepcopy(request)), ("blank_source", deepcopy(request))]
            variants[1][1]["source_render"] = blank_ref
            if replacement_ref:
                replaced = deepcopy(request)
                replaced["source_render"] = replacement_ref
                variants.append(("replaced_source", replaced))
            if mode == "critique":
                swapped = deepcopy(request)
                swapped["plan_menu"]["plans"].reverse()
                swapped["plans"].reverse()
                _bind_menu(swapped)
                variants.append(("swapped", swapped))
            opaque, opaque_elements, opaque_plans = _opaque_request(request, case["case_id"])
            variants.append(("opaque_ids", opaque))
            for condition, variant in variants:
                images = [variant["source_render"]] + [
                    row["render"] for row in variant.get("plans", [])
                ]
                for image in images:
                    if Path(image["path"]).is_absolute():
                        image["path"] = Path(image["path"]).relative_to(out).as_posix()
                probe_id = digest([case["case_id"], mode, condition])[:24]
                relative = f"requests/{probe_id}.json"
                _write_json(out / relative, variant)
                elements = variant["source_summary"]["elements"]
                plans = variant.get("plan_menu", {}).get("plans", [])
                probes.append(
                    {
                        "probe_id": probe_id,
                        "case_id": case["case_id"],
                        "family_id": case["family_id"],
                        "split": case["split"],
                        "condition": condition,
                        "mode": mode,
                        "request_path": relative,
                        "request_digest": request_digest(variant),
                        "source_sha256": variant["source_render"]["sha256"],
                        "original_source_sha256": original_ref["sha256"],
                        "menu_digest": variant.get("plan_menu", {}).get("menu_digest"),
                        "known_element_ids": [row["id"] for row in elements],
                        "plan_map": opaque_plans
                        if condition == "opaque_ids"
                        else {row["plan_id"]: row["plan_id"] for row in plans},
                        "element_map": opaque_elements
                        if condition == "opaque_ids"
                        else {row["id"]: row["id"] for row in elements},
                    }
                )
    if not probes:
        raise ValueError("No evaluable probes were generated")
    if collection_path.read_bytes() != collection_bytes or any(
        sha256_file(ROOT / relative) != expected for relative, expected in originals.items()
    ):
        raise ValueError("Input/code changed during freezing; use a fresh output directory")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    bundle = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "purpose": collection["purpose"],
        "collection_sha256": hashlib.sha256(collection_bytes).hexdigest(),
        "snapshot": {"source_revision": revision, "code_hashes": code_hashes},
        "cases": cases,
        "probes": probes,
        "files": {
            path.relative_to(out).as_posix(): sha256_file(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
        "limitations": limitations
        + [
            "No labels were opened or copied. Test inputs are opt-in; test labels remain separate.",
            "Synthetic collections establish infrastructure behavior, not aesthetic quality.",
            "Image-control changes are sensitivity diagnostics, not proof of grounded reasoning.",
            "A published synthetic test label is not a pristine holdout.",
        ],
    }
    bundle["bundle_sha256"] = digest(bundle)
    _write_json(out / "bundle.json", bundle)
    load_bundle(out / "bundle.json")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--modes", nargs="+", default=["critique"])
    parser.add_argument("--max-cases", type=int, default=4)
    args = parser.parse_args(argv)
    bundle = prepare_collection(
        args.collection, args.out, splits=args.splits, max_cases=args.max_cases, modes=args.modes
    )
    print(json.dumps({"bundle_sha256": bundle["bundle_sha256"], "probes": len(bundle["probes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
