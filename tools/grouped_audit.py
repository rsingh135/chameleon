"""Render and compare source-aware grouped resizing against saved plans.

python -m tools.grouped_audit --baseline-dir output/grouped-baseline \
    --out output/grouped-audit

The bounded matrix is five supplied/authored sources at four canvas sizes, plus
one unequal-products wide case. Without --baseline-dir, only current outputs are
evaluated. Prior-policy replays must be labeled; they are not historical runs.
Repeat --case TASK_ID or --target WIDTHxHEIGHT to filter the default matrix.
Filtered reports contain only selected cells; use a separate --out to retain a
previous full-matrix report.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops

from engine.eval import build_context, check_contract, score
from engine.eval.proportions import analyze_proportions
from engine.layout.appearance import unit_visible_box
from engine.layout.hierarchy import infer_hierarchy
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from tools.composition_cases import _render, authored_diagnostics, generate_cases
from tools.composition_demo import ASSETS, FONTS, ROOT, _contact_sheet, _digest, _write_json

TARGETS = (
    {"width": 1600, "height": 400},
    {"width": 913, "height": 517},
    {"width": 1080, "height": 1920},
    {"width": 728, "height": 90},
)
UNEQUAL_ID = "authored_pair_unequal_products"


def unequal_products_case() -> dict:
    """Shrink the second authored product about its center, without engine hints."""
    task = deepcopy(generate_cases()[0])
    task["task_id"] = UNEQUAL_ID
    product_ids = set(task["expected_campaigns"][1]["product_ids"])
    for element in task["source"]["elements"]:
        if element["id"] not in product_ids:
            continue
        width, height = element["width"] * 0.65, element["height"] * 0.65
        element["x"] += (element["width"] - width) / 2
        element["y"] += (element["height"] - height) / 2
        element["width"], element["height"] = width, height
    task["audit_note"] = (
        "Second product authored at 65% of the first product's frame dimensions. "
        "No historical run is claimed. An optional prior-policy comparison must "
        "be explicitly labeled as a replay on this authored source."
    )
    return task


def audit_cases() -> list[dict]:
    supplied = [
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in sorted((ROOT / "input").glob("chameleon_sample_*.json"))
    ]
    return [*supplied, *generate_cases(), unequal_products_case()]


def _font_metrics(source: dict, design: dict, graph) -> dict:
    originals = {element["id"]: element for element in source["elements"]}
    outputs = {element["id"]: element for element in design["elements"]}
    roles, pairs = {}, []
    for unit in graph.units:
        for eid in unit.element_ids:
            original, output = originals[eid], outputs[eid]
            if original["type"] != "text":
                continue
            roles.setdefault(unit.role, []).append(
                {
                    "element_id": eid,
                    "source_font_px": original["font_size"],
                    "output_font_px": output["font_size"],
                    "scale": output["font_size"] / original["font_size"],
                }
            )
    for cluster in graph.clusters:
        if cluster.kind != "message":
            continue
        headings, bodies = [], []
        for index in cluster.unit_indices:
            unit = graph.units[index]
            if unit.role not in {"headline", "body"}:
                continue
            ids = [eid for eid in unit.element_ids if originals[eid]["type"] == "text"]
            (headings if unit.role == "headline" else bodies).extend(ids)
        for heading in headings:
            for body in bodies:
                source_ratio = originals[heading]["font_size"] / originals[body]["font_size"]
                output_ratio = outputs[heading]["font_size"] / outputs[body]["font_size"]
                pairs.append(
                    {
                        "message": cluster.id,
                        "heading_id": heading,
                        "body_id": body,
                        "source_ratio": source_ratio,
                        "output_ratio": output_ratio,
                        "ratio_preservation": output_ratio / source_ratio,
                        "absolute_log_drift": abs(math.log(output_ratio / source_ratio)),
                    }
                )
    texts = [element["font_size"] for element in design["elements"] if element["type"] == "text"]
    return {
        "minimum_text_px": min(texts, default=None),
        "below_10px": sum(size < 10 for size in texts),
        "source_role_fonts": roles,
        "heading_body_ratios": pairs,
    }


def _authored_product_metrics(task: dict, design: dict) -> dict | None:
    """Measure authored product ratios independently of inferred campaign membership."""
    campaigns = task.get("expected_campaigns")
    if not campaigns:
        return None
    original = {element["id"]: element for element in task["source"]["elements"]}
    outputs = {element["id"]: element for element in design["elements"]}
    rows = []
    for campaign in campaigns:
        ids = campaign["product_ids"]
        before = unit_visible_box([original[eid] for eid in ids], ASSETS)
        after = unit_visible_box([outputs[eid] for eid in ids], ASSETS)
        text_ids = [
            eid for eid in campaign.get("element_ids", []) if original[eid]["type"] == "text"
        ]
        rows.append(
            {
                "campaign": campaign["id"],
                "product_ids": ids,
                "source_visible_height": before.height,
                "output_visible_height": after.height,
                "visible_height_scale": after.height / before.height,
                "median_text_scale": median(
                    outputs[eid]["font_size"] / original[eid]["font_size"] for eid in text_ids
                )
                if text_ids
                else None,
            }
        )
    sibling_pairs = []
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            before = second["source_visible_height"] / first["source_visible_height"]
            after = second["output_visible_height"] / first["output_visible_height"]
            sibling_pairs.append(
                {
                    "first": first["campaign"],
                    "second": second["campaign"],
                    "source_visible_height_ratio": before,
                    "output_visible_height_ratio": after,
                    "ratio_preservation": after / before,
                }
            )
    return {"campaigns": rows, "sibling_product_ratios": sibling_pairs}


def _measure(task, plan, target, context, graph) -> dict:
    source, design = task["source"], plan["design"]
    if design["canvas"] != target:
        raise ValueError("Selected plan canvas does not match its requested target")
    state = plan["state"]
    if state.get("source_digest") != _digest(source):
        raise ValueError("Selected plan is not tied to the original source digest")
    contract = check_contract(source, design, target, context)
    evaluation = score(source, design, context, target_canvas=target)
    result = {
        "state": state,
        "contract_violations": [{"code": item.code, "message": str(item)} for item in contract],
        "evaluation": {
            "total": evaluation["total"],
            "metrics": evaluation["metrics"],
            "violations": [
                {"code": item.code, "message": str(item)} for item in evaluation["violations"]
            ],
        },
        "proportions": analyze_proportions(source, design, graph),
        "relationships": analyze_relationships(graph, design, state.get("cluster_axes", {})),
        "typography": _font_metrics(source, design, graph),
    }
    if task.get("expected_campaigns"):
        result["authored"] = authored_diagnostics(task, design)
        result["authored_products"] = _authored_product_metrics(task, design)
    return result


def _baseline(task, target, directory, context, graph, case_out):
    if directory is None:
        return None
    folder = directory / task["task_id"]
    stem = f"{target['width']}x{target['height']}"
    plan_path = folder / f"{stem}.json"
    provenance_path = folder / "provenance.json"
    unequal = task["task_id"] == UNEQUAL_ID
    if unequal and not all(
        path.is_file() for path in (folder / "task.json", plan_path, provenance_path)
    ):
        return None
    saved_task = json.loads((folder / "task.json").read_text(encoding="utf-8-sig"))
    if _digest(saved_task["source"]) != _digest(task["source"]):
        raise ValueError("Saved baseline source differs from this audit source")
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        if provenance_path.is_file()
        else {"kind": "captured_pre_change", "label": "Captured pre-change selected plan"}
    )
    if not isinstance(provenance, dict):
        raise ValueError("Baseline provenance must be a JSON object")
    if unequal and (
        provenance.get("kind") != "replayed_prior_policy"
        or not isinstance(provenance.get("policy_revision"), str)
        or not provenance["policy_revision"]
        or provenance.get("source_digest") != _digest(task["source"])
    ):
        raise ValueError("Unequal-products baseline needs source-matching prior-policy provenance")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if not isinstance(plan.get("design"), dict) or not isinstance(plan.get("state"), dict):
        raise ValueError("Baseline must contain the captured design and selected state")
    measured = _measure(task, plan, target, context, graph)
    measured["provenance"] = provenance
    picture, measured["render_warnings"] = _render(plan["design"])
    picture.save(case_out / f"{stem}.baseline.png")
    saved_picture = folder / f"{stem}.png"
    if saved_picture.exists():
        with Image.open(saved_picture) as saved:
            if saved.size != picture.size:
                raise ValueError("Saved baseline PNG has the wrong canvas dimensions")
            difference = ImageChops.difference(saved.convert("RGBA"), picture.convert("RGBA"))
            measured["pixels_match_saved_baseline"] = all(
                low == high == 0 for low, high in difference.getextrema()
            )
    measured["plan_file"] = str(plan_path.resolve())
    return measured, picture


def _target_filter(value: str) -> str:
    sizes = {f"{target['width']}x{target['height']}" for target in TARGETS}
    if value not in sizes:
        raise argparse.ArgumentTypeError("Choose a matrix target: " + ", ".join(sorted(sizes)))
    return value


def selected_cases(tasks: list[dict], case_ids=None, target_ids=None) -> list[tuple[dict, tuple]]:
    """Filter the bounded matrix without adding targets or inventing missing cells."""
    case_ids, target_ids = set(case_ids or ()), set(target_ids or ())
    known = {task["task_id"] for task in tasks}
    if case_ids - known:
        raise ValueError("Unknown task ID: " + ", ".join(sorted(case_ids - known)))
    result = []
    for task in tasks:
        if case_ids and task["task_id"] not in case_ids:
            continue
        targets = TARGETS[:1] if task["task_id"] == UNEQUAL_ID else TARGETS
        targets = tuple(
            target
            for target in targets
            if not target_ids or f"{target['width']}x{target['height']}" in target_ids
        )
        if targets:
            result.append((task, targets))
    if not result:
        raise ValueError("The case/target filters select no cells in the default matrix")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/grouped-audit"))
    parser.add_argument("--baseline-dir", type=Path, help="Read-only captured pre-change plans")
    parser.add_argument("--case", dest="case_ids", action="append", metavar="TASK_ID")
    parser.add_argument(
        "--target", dest="target_ids", action="append", type=_target_filter, metavar="WIDTHxHEIGHT"
    )
    args = parser.parse_args(argv)
    try:
        selected = selected_cases(audit_cases(), args.case_ids, args.target_ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.baseline_dir:
        out, baseline = args.out.resolve(), args.baseline_dir.resolve()
        if out == baseline or out in baseline.parents or baseline in out.parents:
            parser.error("--out and --baseline-dir must be separate, non-nested directories")
    # Runtime import avoids making fixture/helper consumers load the resize engine.
    from engine.resize import resize_with_state

    started = time.perf_counter()
    report = {
        "cases": [],
        "errors": [],
        "scope": "Full-resolution rendered audit; metrics are not semantic/aesthetic proof.",
        "baseline_policy": (
            "Source-matching captured plans or explicitly tagged prior-policy replays. "
            "Absent unequal-products replay evidence means current-only."
        ),
        "baseline_directory": str(args.baseline_dir.resolve()) if args.baseline_dir else None,
        "case_filter": sorted(set(args.case_ids)) if args.case_ids else None,
        "target_filter": sorted(set(args.target_ids)) if args.target_ids else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for task, targets in selected:
        source, case_id = task["source"], task["task_id"]
        before = deepcopy(source)
        source_digest = _digest(source)
        row = {"task": case_id, "source_digest": source_digest, "targets": []}
        report["cases"].append(row)
        case_out = args.out / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        cells = []
        try:
            graph = build_relationship_graph(source, ASSETS)
            graph_digest = _digest(asdict(graph))
            hierarchy = infer_hierarchy(source, graph, ASSETS)
            hierarchy_digest = _digest(asdict(hierarchy))
            row.update(graph_digest=graph_digest, hierarchy_digest=hierarchy_digest)
            _write_json(case_out / "task.json", task)
            _write_json(case_out / "source.json", source)
            _write_json(case_out / "graph.json", asdict(graph))
            _write_json(case_out / "hierarchy.json", asdict(hierarchy))
            context = build_context(source, ASSETS, FONTS)
            source_image, row["source_render_warnings"] = _render(source)
            source_image.save(case_out / "source.png")
            cells.append((case_id, "ORIGINAL SOURCE", source_image))
            for target in targets:
                stem = f"{target['width']}x{target['height']}"
                target_row = {"target": dict(target)}
                row["targets"].append(target_row)
                for suffix in (".json", ".state.json", ".plan.json", ".png", ".baseline.png"):
                    (case_out / f"{stem}{suffix}").unlink(missing_ok=True)
                try:
                    baseline = _baseline(task, target, args.baseline_dir, context, graph, case_out)
                    if baseline is not None:
                        target_row["baseline"], picture = baseline
                        provenance = target_row["baseline"]["provenance"]
                        replay = provenance.get("kind") == "replayed_prior_policy"
                        cells.append(
                            (
                                f"PRIOR POLICY {stem}" if replay else f"BEFORE {stem}",
                                provenance.get("label", "Saved source-matching plan"),
                                picture,
                            )
                        )
                    elif case_id == UNEQUAL_ID:
                        target_row["baseline_omitted"] = (
                            "No source-matching saved prior-policy replay for this authored source"
                        )
                    plan = resize_with_state(source, target)
                    target_row["current"] = _measure(task, plan, target, context, graph)
                    picture, target_row["current"]["render_warnings"] = _render(plan["design"])
                    _write_json(case_out / f"{stem}.json", plan["design"])
                    _write_json(case_out / f"{stem}.state.json", plan["state"])
                    _write_json(case_out / f"{stem}.plan.json", plan)
                    picture.save(case_out / f"{stem}.png")
                    cells.append((f"CURRENT {stem}", plan["state"]["name"], picture))
                    if target_row["current"]["contract_violations"]:
                        report["errors"].append(
                            {"task": case_id, "target": target, "phase": "contract"}
                        )
                    current = target_row["current"]
                    campaign_drift = current["proportions"]["metrics"].get("campaign_scale_rms_log")
                    print(
                        f"{case_id} {stem}: {plan['state']['name']} "
                        f"min_font={current['typography']['minimum_text_px']} "
                        f"campaign_drift={campaign_drift} "
                        f"collisions={current['evaluation']['metrics'].get('raw_collisions')}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - retain each explicit audit failure
                    target_row["error"] = f"{type(exc).__name__}: {exc}"
                    report["errors"].append(
                        {"task": case_id, "target": target, "error": target_row["error"]}
                    )
                    cells.append((f"FAILED {stem}", "See report.json", None))
            row["source_immutable"] = source == before and _digest(source) == source_digest
            row["graph_immutable"] = _digest(asdict(graph)) == graph_digest
            row["hierarchy_immutable"] = _digest(asdict(hierarchy)) == hierarchy_digest
            if not all(
                row[key] for key in ("source_immutable", "graph_immutable", "hierarchy_immutable")
            ):
                raise ValueError("Source, graph or hierarchy was mutated during audit")
        except Exception as exc:  # noqa: BLE001 - retain source failures with partial evidence
            row["error"] = f"{type(exc).__name__}: {exc}"
            report["errors"].append({"task": case_id, "error": row["error"]})
        if cells:
            _contact_sheet(cells).save(case_out / "contact_sheet.png")
        _write_json(case_out / "report.json", row)
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    report["output_count"] = sum(len(row["targets"]) for row in report["cases"])
    report["success"] = not report["errors"]
    _write_json(args.out / "report.json", report)
    print(
        f"{report['output_count']} targets; {len(report['errors'])} failures. "
        f"{args.out.resolve() / 'report.json'}"
    )
    return int(not report["success"])


if __name__ == "__main__":
    raise SystemExit(main())
