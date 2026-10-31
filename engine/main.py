"""
Headless orchestrator. It:
  1. Reads every *.json task file from /app/input (source + target_canvas)
  2. Calls the shared configured resize pipeline, with strict stage-one bounds
  3. Checks the output contract (local checks — NOT the official scoring;
     see RULES.md for what the judge actually checks)
  4. Writes <task>_output.json and <task>_output.png to /app/output

Mirrors the same input/ -> output/ convention as the earlier India-program
challenges: drop task files in, get one JSON + one PNG out per task, no
manual steps in between.
"""

import hashlib
import json
import sys
import time
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

from engine.eval import check_contract
from engine.stage_one import strictly_in_bounds
from engine.visual_pipeline import resize_configured

# Relative to the current working directory, which is /app both inside the
# Docker image (WORKDIR /app) and when run locally from this repo's root -
# so this works identically in both "python -m engine.main" (no Docker) and
# the containerized entrypoint.
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
ASSETS_DIR = Path("brand_kit/assets")
_TASK_PROVENANCE = ContextVar("chameleon_batch_provenance", default=None)

REQUIRED_COMMON = ["id", "type", "x", "y", "width", "height"]
REQUIRED_BY_TYPE = {
    "text": ["content", "font_family", "font_size", "color"],
    "image": ["asset_id"],
    "shape": ["shape_type", "fill_color"],
}


def self_check(source: dict, output: dict) -> list[str]:
    """Quick, non-authoritative sanity check so obvious mistakes are caught
    locally before you submit. This is NOT the official scorer — see
    RULES.md for what actually gates and grades your submission."""
    problems = []

    if "canvas" not in output or "elements" not in output:
        return ["output is missing top-level 'canvas' or 'elements'"]

    src_ids = {el["id"] for el in source["elements"]}
    out_ids = {el.get("id") for el in output["elements"]}
    if src_ids != out_ids:
        problems.append(
            f"element id mismatch — missing {sorted(src_ids - out_ids)}, "
            f"unexpected {sorted(out_ids - src_ids)}"
        )

    cw, ch = output["canvas"].get("width"), output["canvas"].get("height")
    for el in output["elements"]:
        for field in REQUIRED_COMMON:
            if field not in el:
                problems.append(f"element '{el.get('id')}': missing '{field}'")
                continue
        for field in REQUIRED_BY_TYPE.get(el.get("type"), []):
            if field not in el:
                problems.append(
                    f"element '{el.get('id')}' (type={el.get('type')}): missing '{field}'"
                )
        if cw and ch and all(k in el for k in ("x", "y", "width", "height")):
            x1, y1 = el["x"] + el["width"], el["y"] + el["height"]
            if el["x"] < -0.5 or el["y"] < -0.5 or x1 > cw + 0.5 or y1 > ch + 0.5:
                problems.append(f"element '{el.get('id')}' is off-canvas ({cw}x{ch})")

    return problems


def render_to_png(design: dict, assets_dir: Path, png_path: Path) -> None:
    """Export the same native pixels used by visual planning, at full size."""
    try:
        # Keep the renderer dependency check explicit even if eval.render was
        # imported previously; a missing backend must never produce a preview.
        from engine.eval.render import render_image
        from engine.resources import get_resources

        __import__("renderer.converter.convert")
    except ImportError as exc:
        raise RuntimeError(
            "Real renderer is unavailable. Install requirements.txt before running the engine."
        ) from exc
    current = get_resources()
    # Honor an active document context. Standalone three-argument callers keep
    # the original convention of a fonts directory next to their assets folder.
    fonts_dir = (
        current.fonts_dir
        if current.assets_dir == Path(assets_dir).resolve()
        else Path(assets_dir).parent / "fonts"
    )
    image = render_image(
        design,
        assets_dir=assets_dir,
        fonts_dir=fonts_dir,
        max_size=max(design["canvas"].values()),
    )
    image.save(png_path)


def resize(source: dict, target: dict) -> dict:
    """Preserve the batch's two-argument design API over configured dispatch."""
    plan = resize_configured(source, target, assets_dir=ASSETS_DIR, strict_stage_one=True)
    provenance = _TASK_PROVENANCE.get()
    if provenance is not None:
        provenance.update(deepcopy(plan.get("state", {})))
    return plan["design"]


def _resize_task(source: dict, target: dict) -> tuple[dict, dict]:
    # Scope provenance to this task, including callers that replace the public
    # resize facade. No receipt can leak from a previous task or another thread.
    provenance = {}
    token = _TASK_PROVENANCE.set(provenance)
    try:
        return resize(source, target), provenance
    finally:
        _TASK_PROVENANCE.reset(token)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_files = sorted(INPUT_DIR.glob("*.json"))
    if not task_files:
        print(f"No task files found in {INPUT_DIR}", file=sys.stderr)
        return 1

    exit_code = 0
    for task_path in task_files:
        stem = task_path.stem
        json_path = OUTPUT_DIR / f"{stem}_output.json"
        png_path = OUTPUT_DIR / f"{stem}_output.png"
        pipeline_path = OUTPUT_DIR / f"{stem}_pipeline.json"
        try:
            # A failed rerun must not leave an old preview masquerading as its result.
            json_path.unlink(missing_ok=True)
            png_path.unlink(missing_ok=True)
            pipeline_path.unlink(missing_ok=True)
            task = json.loads(task_path.read_text(encoding="utf-8"))
            source, target_canvas = task["source"], task["target_canvas"]

            t0 = time.perf_counter()
            output, provenance = _resize_task(source, target_canvas)
            elapsed = time.perf_counter() - t0
            violations = check_contract(source, output, target_canvas)
            if violations:
                raise ValueError("Output contract failed: " + "; ".join(map(str, violations)))
            if not strictly_in_bounds(output):
                raise ValueError("Output failed strict stage-one bounds")

            json_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
            render_to_png(output, ASSETS_DIR, png_path)
            if "visual_model" in provenance:
                pipeline_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "task_file": task_path.name,
                            "output_json_sha256": hashlib.sha256(
                                json_path.read_bytes()
                            ).hexdigest(),
                            "output_png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
                            "visual_model": provenance["visual_model"],
                        },
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
        except Exception as exc:
            exit_code = 1
            # A renderer can fail after writing part of a PNG.
            png_path.unlink(missing_ok=True)
            pipeline_path.unlink(missing_ok=True)
            print(f"[{task_path.name}] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        print(f"[{task_path.name}] OK in {elapsed:.2f}s -> {stem}_output.json, {stem}_output.png")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
