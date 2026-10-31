"""Render and measure focal-axis proposals or the selected source adaptation."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from engine.eval import build_context, score
from engine.eval.proportions import analyze_proportions
from engine.eval.render import render_image
from engine.layout.focal import analyze_focal, infer_focal_profile
from engine.layout.focal_states import generate_focal_states
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.resize import resize_with_state

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"core": "level2_core", "advanced": "level3_advanced"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*SOURCES, "all"), default="all")
    parser.add_argument("--target", default="1600x400")
    parser.add_argument("--out", type=Path, default=Path("output/focal-audit"))
    parser.add_argument("--proposals", action="store_true")
    args = parser.parse_args()
    try:
        width, height = map(int, args.target.lower().split("x"))
        if min(width, height) < 1 or max(width, height) > 8192 or width * height > 20_000_000:
            raise ValueError
    except ValueError:
        parser.error("--target must be WIDTHxHEIGHT inside the local canvas limits")
    target = {"width": width, "height": height}
    cases = SOURCES if args.case == "all" else {args.case: SOURCES[args.case]}
    report = {}
    for name, suffix in cases.items():
        source = json.loads((ROOT / f"input/chameleon_sample_{suffix}.json").read_text())["source"]
        before = deepcopy(source)
        graph = build_relationship_graph(source)
        graph_before = asdict(graph)
        profile = infer_focal_profile(source, graph)
        context = build_context(source)
        out = args.out / name / args.target
        out.mkdir(parents=True, exist_ok=True)
        render_image(source, max_size=max(source["canvas"].values())).save(out / "source.png")
        if args.proposals:
            plans = [
                {
                    "design": s.design,
                    "state": {
                        "name": s.name,
                        "cluster_axes": dict(s.cluster_axes),
                        "sizing_exceptions": list(s.exceptions),
                    },
                }
                for s in generate_focal_states(source, target, graph, profile)
            ]
        else:
            plans = [resize_with_state(source, target)]
        rows = []
        for plan in plans:
            design, state = plan["design"], plan["state"]
            filename = state["name"]
            render_image(design, max_size=max(target.values())).save(out / f"{filename}.png")
            row = {
                "state": state,
                "evaluation": score(source, design, context, target_canvas=target),
                "focal": analyze_focal(source, design, graph, profile),
                "relationships": analyze_relationships(graph, design, state["cluster_axes"]),
                "proportions": analyze_proportions(source, design, graph),
            }
            (out / f"{filename}.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            rows.append(row)
            print(
                name,
                filename,
                json.dumps(row["focal"]["metrics"]),
                "collisions",
                row["evaluation"]["metrics"].get("raw_collisions"),
                "relationships",
                [w["code"] for w in row["relationships"]["warnings"]],
                flush=True,
            )
        assert source == before and asdict(graph) == graph_before
        report[name] = {
            "source_immutable": True,
            "profile": asdict(profile) if profile else None,
            "target": target,
            "results": rows,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
