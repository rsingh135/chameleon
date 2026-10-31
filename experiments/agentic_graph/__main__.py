"""Run a bounded three-model graph or its explicit deterministic control."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.resources import DEFAULT_ASSETS_DIR, DEFAULT_FONTS_DIR
from engine.visual_pipeline import _io
from experiments.agentic_graph.fundamentals import fundamental_tasks
from experiments.agentic_graph.graph import GraphLimits, TaskCriteria, run_graph, write_json
from experiments.agentic_graph.providers import LocalModelRoles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", type=Path)
    source.add_argument("--fundamentals", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="JSON with analyzer/organizer/verifier commands")
    mode.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="A new evidence directory")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--fonts", type=Path, default=DEFAULT_FONTS_DIR)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--minimum-font", type=float, default=12)
    args = parser.parse_args()
    roles = LocalModelRoles(_io._parse(args.config.read_text("utf-8-sig"))) if args.config else None
    tasks = (
        fundamental_tasks()
        if args.fundamentals
        else [{**_io._parse(args.task.read_text("utf-8-sig")), "id": args.task.stem}]
    )
    criteria = TaskCriteria(minimum_font_px=args.minimum_font)
    limits = GraphLimits(max_rounds=args.max_rounds, max_seconds=args.max_seconds)
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for task in tasks:
        result = run_graph(
            task["source"],
            task["target_canvas"],
            args.out / task["id"],
            source_id=task["id"],
            assets_dir=args.assets,
            fonts_dir=args.fonts,
            roles=roles,
            criteria=criteria,
            limits=limits,
        )
        results.append(
            {
                "task_id": task["id"],
                "status": result["status"],
                "task_passed": result["task_passed"],
                "all_three_roles_executed": result["all_three_roles_executed"],
                "stop_reason": result["stop_reason"],
            }
        )
        print(f"{task['id']}: {result['status']} ({result['stop_reason']})", flush=True)
    report = {
        "schema_version": 1,
        "mode": "baseline_only" if roles is None else "three_roles",
        "all_tasks_passed": all(row["task_passed"] for row in results),
        "tasks": results,
        "qualification": "Fundamental checks are not proof of trained visual generalization.",
    }
    write_json(args.out / "suite-report.json", report)
    return 0 if report["all_tasks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
