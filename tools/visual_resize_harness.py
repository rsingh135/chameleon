"""Audit the common configured visual resize path using native evidence.

The production phase implementation lives in engine.visual_pipeline; this tool
adds source loading, personal-test reserve exclusions and explicit review files.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from engine import visual_pipeline as pipeline
from tools import scene_harness as scene

ROOT = Path(__file__).resolve().parents[1]
MAX_PROVIDER_BYTES = pipeline.MAX_PROVIDER_BYTES
_engine_api = pipeline._engine_api
run_provider = pipeline.run_provider
make_request = pipeline.make_request
_candidate_id = pipeline._candidate_id
_winner = pipeline._winner
_validated_result = pipeline._validated_result


def _implementation_hashes():
    files = scene._implementation_hashes()
    files[Path(__file__).relative_to(ROOT).as_posix()] = scene._sha(__file__)
    for path in sorted((ROOT / "experiments/visual_planner").glob("*.py")):
        files[path.relative_to(ROOT).as_posix()] = scene._sha(path)
    return files


def _reserve_identities():
    """Consult exclusion identities only; never render reserve source pixels."""
    catalogue = ROOT / "fixtures/open_posters/catalogue.json"
    ids, hashes = set(), set()
    for row in scene._read(catalogue)["examples"]:
        if row.get("split") != "reserve":
            continue
        ids.add(row["id"])
        source = catalogue.parent / "prepared" / row["id"] / "source.json"
        if source.is_file():
            hashes.add(scene._digest(scene._read(source)))
    return ids, hashes


def reject_reserved_source(source, source_id=None):
    ids, hashes = _reserve_identities()
    if source_id in ids or scene._digest(source) in hashes:
        raise ValueError("Personal-test reserve sources are excluded from visual-model experiments")


def source_image_ablation(request, replacement_render):
    """Build a marked diagnostic; it never overwrites the real source asset."""
    altered = deepcopy(request)
    altered.pop("request_digest", None)
    altered["source_render"] = deepcopy(replacement_render)
    altered["diagnostic"] = {
        "kind": "source_image_ablation",
        "original_source_render_sha256": request["source_render"]["sha256"],
        "not_production_source_evidence": True,
    }
    altered["request_digest"] = scene._digest(altered)
    return altered


def candidate_order_control(request):
    """Reverse critic letters while retaining each candidate's identity/image."""
    if request.get("mode") != "critique":
        raise ValueError("Candidate order control requires a critique request")
    altered = deepcopy(request)
    altered.pop("request_digest", None)
    altered["plan_menu"]["plans"].reverse()
    altered["plan_menu"].pop("menu_digest", None)
    altered["plan_menu"]["menu_digest"] = scene._digest(altered["plan_menu"])
    altered["plans"].reverse()
    altered["diagnostic"] = {"kind": "candidate_order", "not_production_selection": True}
    altered["request_digest"] = scene._digest(altered)
    return altered


def target_control(source, source_render, swapped_menu):
    request = make_request(source, source_render, swapped_menu, mode="plan")
    request.pop("request_digest", None)
    request["diagnostic"] = {"kind": "target_swap", "not_production_selection": True}
    request["request_digest"] = scene._digest(request)
    return request


def _attach_review(review, record):
    if review is None:
        return
    if (
        not isinstance(review, dict)
        or review.get("origin") not in {"agent_visual_review", "human_review"}
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"]
        or review.get("baseline_sha256") != record["baseline_render"]["sha256"]
        or review.get("final_sha256") != record["final_render"]["sha256"]
        or review.get("source_digest") != record["source_digest"]
        or review.get("target_canvas") != record["target_canvas"]
        or review.get("baseline_design_digest") != record["baseline_design_digest"]
        or review.get("final_design_digest") != record["final_design_digest"]
        or review.get("verdict") not in {"improved", "same", "worse", "uncertain"}
    ):
        raise ValueError("Visual review must bind both exact PNGs and its reviewer/origin")
    if review["verdict"] == "improved" and record["same_png"]:
        raise ValueError("Identical rendered pixels cannot establish a visual improvement")
    record["quality_review"] = deepcopy(review)
    record["quality_improved"] = (
        None if review["verdict"] == "uncertain" else review["verdict"] == "improved"
    )


def review_existing(report_path, reviews, destination=None):
    """Attach an explicit review without rerunning a model or rewriting evidence."""
    report_path = Path(report_path).resolve()
    report = scene._read(report_path)
    if not isinstance(reviews, dict) or not report.get("implementation_unchanged"):
        raise ValueError("Review requires a fixed-implementation report and review mapping")
    if not report.get("source_immutable") or not report.get("resources_immutable"):
        raise ValueError("Review cannot approve a mutable-source experiment")
    root = report_path.parent
    if scene._sha(root / "source.png") != report["source_render"]["sha256"]:
        raise ValueError("Original source render changed before review")
    if scene._digest(scene._read(root / "source.json")) != report["source_json_digest"]:
        raise ValueError("Original source JSON changed before review")
    for row in report["targets"]:
        dimensions = row["target_canvas"]
        if any(
            type(dimensions.get(k)) is not int or dimensions[k] <= 0 for k in ("width", "height")
        ):
            raise ValueError("Review target dimensions are invalid")
        name = f"{dimensions['width']}x{dimensions['height']}"
        directory = root / name
        baseline = _winner(scene._read(directory / "baseline-snapshot.json"))["design"]
        final = scene._read(directory / "final-plan.json")["design"]
        if (
            scene._sha(directory / "baseline.png") != row["baseline_render"]["sha256"]
            or scene._sha(directory / "final.png") != row["final_render"]["sha256"]
            or scene._digest(baseline) != row["baseline_design_digest"]
            or scene._digest(final) != row["final_design_digest"]
        ):
            raise ValueError("A reviewed native image or design changed")
        _attach_review(reviews.get(name), row)
    report["reviewed_from_report_sha256"] = scene._sha(report_path)
    destination = Path(destination or root / "review-report.json").resolve()
    if destination == report_path:
        raise ValueError("Review must not overwrite original experiment evidence")
    scene._write(destination, report)
    return report


def inspect_visual_resize(
    source,
    assets_dir,
    fonts_dir,
    out,
    *,
    targets,
    command,
    timeout=180,
    source_id=None,
    reviews=None,
    strict_stage_one=True,
):
    reject_reserved_source(source, source_id)
    report = pipeline.evaluate_visual_resize(
        source,
        assets_dir,
        fonts_dir,
        out,
        targets=targets,
        command=command,
        timeout=timeout,
        source_id=source_id,
        strict_stage_one=strict_stage_one,
        api=_engine_api(),
        provider=run_provider,
        implementation_hashes=_implementation_hashes,
    )
    for row in report["targets"]:
        target = row["target_canvas"]
        name = f"{target['width']}x{target['height']}"
        _attach_review((reviews or {}).get(name), row)
        scene._write(Path(out) / name / "result.json", row)
    scene._write(Path(out) / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prepared", type=Path)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--bundle", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--fonts-dir", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--target", type=scene.target_dimensions, action="append", required=True)
    parser.add_argument("--model-command", required=True, help="Explicit local JSON argv; no shell")
    parser.add_argument("--model-timeout", type=float, default=180)
    parser.add_argument("--reviews", type=Path, help="Optional PNG-bound visual review mapping")
    parser.add_argument("--out", type=Path, default=ROOT / "output/visual-resize")
    args = parser.parse_args(argv)
    try:
        original_input = (
            args.prepared / "source.json" if args.prepared else args.source or args.bundle
        )
        original_payload = scene._read(original_input)
        original_source = (
            original_payload.get("source", original_payload)
            if isinstance(original_payload, dict)
            else original_payload
        )
        reject_reserved_source(
            original_source, args.source_id or (args.prepared.name if args.prepared else None)
        )
        source, assets, fonts, receipt = scene.load_input(
            prepared=args.prepared,
            source_path=args.source,
            bundle=args.bundle,
            assets_dir=args.assets_dir,
            fonts_dir=args.fonts_dir,
            out=args.out,
        )
        report = inspect_visual_resize(
            source,
            assets,
            fonts,
            args.out,
            targets=args.target,
            command=args.model_command,
            timeout=args.model_timeout,
            source_id=args.source_id or (args.prepared.name if args.prepared else None),
            reviews=scene._read(args.reviews) if args.reviews else None,
        )
        if receipt is not None:
            scene._write(args.out / "bundle-receipt.json", receipt)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Visual resize failed: {exc}\n")
    print(
        scene._canonical(
            {
                "report": str(args.out / "report.json"),
                "targets": [
                    {
                        key: row[key]
                        for key in (
                            "target_canvas",
                            "model_executed",
                            "new_candidates_generated",
                            "model_candidate_selected",
                            "layout_changed",
                            "quality_improved",
                            "fallback_reason",
                        )
                    }
                    for row in report["targets"]
                ],
            }
        )
    )
    return int(
        not report["implementation_unchanged"]
        or any(r["status"] != "contract_valid" for r in report["targets"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
