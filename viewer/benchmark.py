"""Bounded, source-bound evaluation of one frozen method across native scenes.

This runner uses the current deterministic engine. It does not invoke a model,
train anything, approve visual quality, or turn flattened images into elements.
The browser can submit one job at a time while retaining the expected method
digest; a changed implementation is rejected before rendering the next job.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from copy import deepcopy
from pathlib import Path

from engine.resources import ResourceContext
from tools.scene_harness import resource_hashes
from viewer.document_bridge import MAX_DOCUMENT_REQUEST_BYTES, prepare_document

ROOT = Path(__file__).resolve().parents[1]
METHOD_ID = "deterministic-current"
MAX_SCENES = 8
MAX_TARGETS = 4
MAX_JOBS = 16


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def method_receipt() -> dict:
    """Fingerprint the local implementation and relevant installed runtime.

    This describes checkout files, not a hermetic environment or a claim that
    an already-running Python process hot-reloads edits. Restart after edits.
    """
    paths = [
        *(ROOT / "engine").rglob("*.py"),
        *(ROOT / "renderer").rglob("*.py"),
        ROOT / "tools" / "scene_harness.py",
        ROOT / "viewer" / "server.py",
        ROOT / "viewer" / "document_bridge.py",
        Path(__file__),
    ]
    files = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }
    packages = {}
    for name in ("Pillow", "numpy", "skia-python", "cr-renderer"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed-as-distribution"
    receipt = {
        "id": METHOD_ID,
        "protocol": "scene-benchmark-v1",
        "implementation_sha256": _digest(files),
        "runtime": {"python": platform.python_version(), "packages": packages},
        "analyzer": {"enabled": False},
        "critic": {"enabled": False},
        "options": {"layout": "current-engine-defaults", "native_renderer": True},
        "fingerprint_scope": "engine+renderer+viewer resize implementation and runtime versions",
        "restart_after_code_changes": True,
    }
    return {**receipt, "digest": _digest(receipt)}


def _text(value: object, name: str, maximum: int = 200) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be nonempty text of at most {maximum} characters.")
    return value


def validate_benchmark(payload: object) -> dict:
    """Normalize plain sources/tasks and validate every job before rendering."""
    from viewer.server import validate_request

    if not isinstance(payload, dict):
        raise ValueError("Benchmark request must be an object.")
    if payload.get("method", METHOD_ID) != METHOD_ID:
        raise ValueError("Only deterministic-current is available; no vision model is enabled.")
    if len(_canonical(payload)) > MAX_DOCUMENT_REQUEST_BYTES:
        raise ValueError("Benchmark request exceeds 24 MiB.")
    scenes, targets = payload.get("scenes"), payload.get("targets")
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= MAX_SCENES:
        raise ValueError(f"Provide between 1 and {MAX_SCENES} scenes.")
    if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_TARGETS:
        raise ValueError(f"Provide between 1 and {MAX_TARGETS} target canvases.")
    if len(scenes) * len(targets) > MAX_JOBS:
        raise ValueError(f"A benchmark can contain at most {MAX_JOBS} jobs.")
    expected = payload.get("expected_method_digest")
    if expected is not None and (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise ValueError("expected_method_digest must be a lowercase SHA-256 digest.")
    normalized = []
    seen = set()
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError("Each scene must be an object.")
        sid = _text(scene.get("id"), f"scenes[{index}].id", 120)
        if sid in seen:
            raise ValueError("Scene IDs must be unique within a benchmark.")
        seen.add(sid)
        task = scene.get("payload")
        if isinstance(task, dict) and "canvas" in task and "elements" in task:
            task = {"source": task}
        if not isinstance(task, dict) or not isinstance(task.get("source"), dict):
            raise ValueError(f"Scene {sid} must contain native JSON as payload.")
        task = deepcopy(task)
        # Each job uses the shared targets, never a source's preselected target.
        task.pop("target_canvas", None)
        for target in targets:
            validate_request({**task, "target_canvas": target})
        input_kind = scene.get("input_kind", "editable-design")
        if not isinstance(input_kind, str) or input_kind not in {"editable-design", "flat-image"}:
            raise ValueError("input_kind must be editable-design or flat-image.")
        normalized.append(
            {
                "id": sid,
                "label": _text(scene.get("label", sid), f"scenes[{index}].label"),
                "family": _text(scene.get("family", "unassigned"), "family", 120),
                "input_kind": input_kind,
                "payload": task,
            }
        )
    return {
        "method": METHOD_ID,
        "expected_method_digest": expected,
        "scenes": normalized,
        "targets": deepcopy(targets),
    }


def _counts(rows: list[dict]) -> dict:
    return {
        "jobs": len(rows),
        "succeeded": sum(row["status"] == "ok" for row in rows),
        "failed": sum(row["status"] == "error" for row in rows),
        "contract_passed": sum(row["contract_passed"] is True for row in rows),
        "visual_review": "not_reviewed",
    }


def run_benchmark(payload: object) -> dict:
    """Render validated jobs sequentially and bind every result to its evidence."""
    from viewer.server import perform_resize

    plan = validate_benchmark(payload)
    method = method_receipt()
    expected = plan["expected_method_digest"]
    if expected is not None and expected != method["digest"]:
        raise ValueError("The method changed. Restart the server and start a new benchmark.")
    results = []
    for scene in plan["scenes"]:
        task = scene["payload"]
        source_digest = _digest(task["source"])
        resources = ResourceContext()
        preparation_error = None
        document = None
        hashes = None
        try:
            if any(key in task for key in ("document", "assets", "fonts")):
                document = prepare_document(task)
                resources = document.resources
            hashes = resource_hashes(task["source"], resources.assets_dir, resources.fonts_dir)
        except (ValueError, OSError, RuntimeError) as exc:
            preparation_error = str(exc)
        for target in plan["targets"]:
            started = time.perf_counter()
            row = {
                "scene_id": scene["id"],
                "label": scene["label"],
                "family": scene["family"],
                "input_kind": scene["input_kind"],
                "target_canvas": deepcopy(target),
                "source_sha256": source_digest,
                "resources_sha256": _digest(hashes) if hashes is not None else None,
                "resource_hashes": hashes,
                "status": "error",
                "contract_passed": None,
                "result": None,
            }
            try:
                if preparation_error is not None:
                    raise ValueError(preparation_error)
                # Never let a failing processor mutate the frozen plan or next job.
                request = deepcopy({**task, "target_canvas": target})
                result = perform_resize(request, resources=resources)
                after = resource_hashes(task["source"], resources.assets_dir, resources.fonts_dir)
                if _digest(task["source"]) != source_digest or after != hashes:
                    raise RuntimeError("Source or resource bytes changed during the benchmark.")
                if document is not None:
                    result["document"] = document.receipt
                row.update(
                    status="ok",
                    contract_passed=not bool(result["report"]["violations"]),
                    result=result,
                )
            except Exception as exc:
                # One bad resource/renderer must not conceal the other scene outcomes.
                row["error"] = str(exc)[:1000]
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            results.append(row)
    comparable = method_receipt()["digest"] == method["digest"]
    families = sorted({row["family"] for row in results})
    summary = {
        **_counts(results),
        "by_family": {
            family: _counts([row for row in results if row["family"] == family])
            for family in families
        },
    }
    return {
        "schema_version": "scene-benchmark-v1",
        "method": method,
        "comparable": comparable,
        "comparison_warning": (
            None if comparable else "Implementation files changed during this run; rerun all jobs."
        ),
        "results": results,
        "summary": summary,
    }
