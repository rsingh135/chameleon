"""CPU-only JSONL prepare/freeze/complete transport for adapter diagnostics.

No model is imported or launched. The operator runs the separately supervised
prepared worker between freeze and complete, including on the cloud GPU host.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from .adapter_analysis import (
    CLASSES,
    _identity,
    _purpose_views,
    build_adapter_request,
    canonical,
    complete_adapter_analysis,
    request_bytes,
)
from .adapter_context import (
    build_source_contexts,
    build_source_views,
    sha_file,
    verify_source_views,
)

MAX_PACKET_BYTES = 16 * 1024 * 1024


def _parse(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate adapter transport JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("Adapter transport JSON must contain finite values")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _read(path, maximum=MAX_PACKET_BYTES):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Adapter transport artifact is missing or exceeds its byte budget")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Adapter transport artifact grew beyond its byte budget")
    return _parse(data)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")


def _job(root, job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise ValueError("Adapter job IDs must be bounded opaque local names")
    root = Path(root).resolve()
    target = (root / job_id).resolve()
    if not target.is_relative_to(root) or target.is_symlink():
        raise ValueError("Adapter job escapes its operator-owned artifact root")
    return target


def _prepared_identity(directory, bundle):
    recipe_path = directory / "recipe.json"
    recipe_hash = sha_file(recipe_path, 4 * 1024 * 1024)
    recipe = _read(recipe_path)
    views = _purpose_views(bundle["request"].get("purpose_views", 2))
    version = "two-view" if views == 2 else "three-view"
    if type(recipe.get("purpose_views", 2)) is not int or recipe.get("purpose_views", 2) != views:
        raise ValueError(f"This bridge requires the exact trained {version} purpose contract")
    if (
        recipe.get("external_evaluation") is not True
        or recipe.get("inference_only") is not True
        or type(recipe.get("steps")) is not int
        or recipe["steps"] != 0
    ):
        raise ValueError("Bridge freeze accepts label-free adapter-only inference recipes")
    dataset_path = Path(recipe["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = directory / dataset_path
    if sha_file(dataset_path) != recipe["dataset_sha256"]:
        raise ValueError("Prepared inference dataset changed")
    dataset = _read(dataset_path)
    if type(dataset.get("purpose_views", 2)) is not int or dataset.get("purpose_views", 2) != views:
        raise ValueError(f"This bridge requires the exact {version} purpose dataset")
    if (
        dataset.get("task") != "purpose"
        or set(dataset.get("classes", [])) != set(CLASSES)
        or len(dataset["classes"]) != len(CLASSES)
        or dataset.get("provenance", {}).get("request_sha256") != bundle["request_sha256"]
    ):
        raise ValueError("Prepared inference dataset differs from the grounded ten-word request")
    expected = {row["record_id"]: row for row in bundle["request"]["records"]}
    rows = dataset.get("records", [])
    if len(rows) != len(expected) or len({row.get("record_id") for row in rows}) != len(rows):
        raise ValueError("Prepared inference records do not cover the exact request")
    for row in rows:
        if set(row) - {"record_id", "family_id", "split", "images", "prompt", "control"}:
            raise ValueError("Prepared inference record contains labels or unsupported metadata")
        original = expected.get(row["record_id"])
        if (
            not original
            or row["prompt"] != original["prompt"]
            or [ref["sha256"] for ref in row["images"]]
            != [ref["sha256"] for ref in original["images"]]
        ):
            raise ValueError("Prepared inference prompt/images differ from native source grounding")
    manifest_path = Path(recipe["base_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = directory / manifest_path
    if sha_file(manifest_path) != recipe["base_manifest_sha256"]:
        raise ValueError("Prepared base manifest changed")
    manifest = _read(manifest_path)
    package = directory / "code/experiments/visual_finetune"
    for name, checksum in recipe["code_sha256"].items():
        if (
            not isinstance(name, str)
            or "/" in name
            or "\\" in name
            or sha_file(package / name, 1024 * 1024) != checksum
        ):
            raise ValueError("Prepared adapter worker code changed")
    identity = _identity(
        {
            "base": {
                "repo": manifest["repo_id"],
                "revision": manifest["revision"],
                "model_sha256": manifest["files"]["model.safetensors"]["sha256"],
            },
            "adapter_sha256": recipe["adapter_sha256"],
            "adapter_config_sha256": recipe["adapter_config_sha256"],
            "recipe_sha256": recipe_hash,
            "dataset_sha256": recipe["dataset_sha256"],
            "code_sha256": recipe["code_sha256"],
        }
    )
    if sha_file(recipe_path, 4 * 1024 * 1024) != recipe_hash:
        raise ValueError("Prepared recipe changed while freezing its identity")
    return identity


def handle(packet, *, artifact_root, diagnostic=False, purpose_views=2, view_builder=None):
    """Process one packet; operator flags alone permit diagnostic observations."""
    if not isinstance(packet, dict):
        raise ValueError("Adapter JSONL input must contain an operation object")
    views = _purpose_views(purpose_views)
    if views == 2 and view_builder is not None:
        raise ValueError("Two-view transport does not accept an isolated view builder")
    operation = packet.get("operation")
    job = _job(artifact_root, packet.get("job_id"))
    if operation == "prepare":
        if set(packet) != {"operation", "job_id", "source", "source_render"}:
            raise ValueError("Adapter prepare accepts original source and native render only")
        if views == 3 and not callable(view_builder):
            raise ValueError("Three-view prepare requires a trusted operator native view builder")
        job.mkdir(parents=True, exist_ok=False)
        if views == 2:
            crop = build_source_contexts(
                packet["source"], packet["source_render"], job / "context-images"
            )
            bundle = build_adapter_request(
                packet["source"], packet["source_render"], crop["context_images"]
            )
        else:
            prepared = view_builder(
                deepcopy(packet["source"]), deepcopy(packet["source_render"]), job / "source-views"
            )
            bundle = build_adapter_request(
                packet["source"],
                packet["source_render"],
                prepared["context_images"],
                purpose_views=3,
                isolated_images=prepared["isolated_images"],
                isolated_receipts=prepared["isolated_receipts"],
            )
            if bundle["records"]:
                verify_source_views(packet["source"], packet["source_render"], bundle)
        _write(job / "source.json", packet["source"])
        _write(job / "source-render.json", packet["source_render"])
        _write(job / "binding.json", bundle)
        with (job / "adapter-request.json").open("xb") as stream:
            stream.write(request_bytes(bundle))
        return {
            "ok": True,
            "operation": operation,
            "job_id": packet["job_id"],
            "request_path": str(job / "adapter-request.json"),
            "binding_path": str(job / "binding.json"),
            "source_digest": bundle["source_digest"],
            "source_render_sha256": bundle["source_render_sha256"],
            "record_count": len(bundle["records"]),
            "purpose_views": views,
            "initial_abstentions": bundle["initial_abstentions"],
            "worker_request_ready": bool(bundle["records"]),
            "model_executed": False,
            "deployment_enabled": False,
        }
    if operation == "freeze":
        if set(packet) != {"operation", "job_id", "evaluation_dir"}:
            raise ValueError("Adapter freeze accepts an already prepared evaluation directory only")
        directory = Path(packet["evaluation_dir"]).resolve()
        if not directory.is_relative_to(job):
            raise ValueError("Prepared evaluation must stay within this adapter job directory")
        run_markers = (
            "phases.jsonl",
            "stdout.log",
            "stderr.log",
            "supervision.json",
            "result.json",
            "inference_receipt.json",
            "external_adapted.json",
        )
        if any((directory / marker).exists() for marker in run_markers):
            raise ValueError("Expected identity must be frozen before model execution starts")
        bundle = _read(job / "binding.json")
        if _purpose_views(bundle["request"].get("purpose_views", 2)) != views:
            raise ValueError("Frozen binding differs from the operator's explicit view profile")
        identity = _prepared_identity(directory, bundle)
        if any((directory / marker).exists() for marker in run_markers):
            raise ValueError("Model execution started during expected-identity freezing")
        _write(
            job / "frozen-evaluation.json",
            {
                "evaluation_dir": str(directory),
                "prepared_identity": identity,
                "binding_digest": bundle["binding_digest"],
                "model_binary_bytes_verified_here": False,
            },
        )
        return {
            "ok": True,
            "operation": operation,
            "job_id": packet["job_id"],
            "worker_recipe": str(directory / "recipe.json"),
            "prepared_identity": identity,
            "model_executed": False,
            "deployment_enabled": False,
        }
    if operation == "complete":
        if set(packet) != {"operation", "job_id"}:
            raise ValueError("Adapter completion uses only the previously frozen job")
        frozen = _read(job / "frozen-evaluation.json")
        bundle = _read(job / "binding.json")
        if _purpose_views(bundle["request"].get("purpose_views", 2)) != views:
            raise ValueError("Completion binding differs from the operator's explicit view profile")
        if bundle["binding_digest"] != frozen["binding_digest"]:
            raise ValueError("Adapter binding changed after pre-inference freezing")
        directory = Path(frozen["evaluation_dir"])
        if _prepared_identity(directory, bundle) != frozen["prepared_identity"]:
            raise ValueError("Prepared inference identity changed after model execution")
        unique_refs = {
            ref["path"]: ref["sha256"]
            for row in bundle["request"]["records"]
            for ref in row["images"]
        }
        for path, checksum in unique_refs.items():
            if sha_file(path) != checksum:
                raise ValueError("Source/context PNG changed before adapter completion")
        if views == 3:
            verify_source_views(
                _read(job / "source.json"), _read(job / "source-render.json"), bundle
            )
        prediction_path = directory / "external_adapted.json"
        if not prediction_path.is_file() or prediction_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Adapter prediction file is missing or oversized")
        with prediction_path.open("rb") as stream:
            predictions = stream.read(8 * 1024 * 1024 + 1)
        result = complete_adapter_analysis(
            _read(job / "source.json"),
            _read(job / "source-render.json"),
            bundle,
            predictions,
            _read(directory / "inference_receipt.json"),
            expected_identity=frozen["prepared_identity"],
            mode="diagnostic" if diagnostic else "off",
        )
        return {
            "ok": result["valid"],
            "operation": operation,
            "job_id": packet["job_id"],
            "result": result,
        }
    raise ValueError("Unknown adapter transport operation")


def main(argv=None, *, stdin=None, stdout=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("output/adapter-analyzer"))
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Explicit unqualified diagnostic observations only",
    )
    parser.add_argument("--purpose-views", type=int, choices=(2, 3), default=2)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--fonts-dir", type=Path)
    parser.add_argument("--resource-digest")
    args = parser.parse_args(argv)
    if args.purpose_views == 2 and any(
        value is not None for value in (args.assets_dir, args.fonts_dir, args.resource_digest)
    ):
        parser.error("Native isolation resource flags require explicit --purpose-views 3")
    if args.purpose_views == 3 and (args.assets_dir is None or args.fonts_dir is None):
        parser.error("Three-view preparation requires trusted --assets-dir and --fonts-dir")

    def views_builder(source, render, destination):
        return build_source_views(
            source,
            render,
            destination,
            purpose_views=3,
            assets_dir=args.assets_dir,
            fonts_dir=args.fonts_dir,
            expected_resource_digest=args.resource_digest,
        )

    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    while True:
        line = stdin.readline(MAX_PACKET_BYTES + 1)
        if not line:
            break
        try:
            if len(line.encode("utf-8")) > MAX_PACKET_BYTES or not line.endswith("\n"):
                raise ValueError("Adapter JSONL packet exceeds its line budget or lacks a newline")
            output = handle(
                _parse(line),
                artifact_root=args.artifact_root,
                diagnostic=args.diagnostic,
                purpose_views=args.purpose_views,
                view_builder=views_builder if args.purpose_views == 3 else None,
            )
        except (ValueError, OSError, KeyError, TypeError, UnicodeError, RecursionError) as exc:
            output = {
                "ok": False,
                "error": str(exc)[:600],
                "model_executed": False,
                "deployment_enabled": False,
            }
        stdout.write(canonical(output) + "\n")
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
