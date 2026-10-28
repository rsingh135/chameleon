"""Render source relationships and bounded composition states for visual review.

python -m tools.composition_demo --task input/chameleon_sample_level2_core.json \
    --baseline output/research/baseline/core_01-1600x400.json --out output/composition

Without --baseline, the comparison uses the CURRENT default engine, which may
already include these states. This is an experiment report, not a quality proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from engine.eval import build_context, score
from engine.eval.render import render_image
from engine.eval.style import analyze_style
from engine.layout.composition import generate_composition_states
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from engine.resize import resize
from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello

ROOT = Path(__file__).resolve().parents[1]
ASSETS, FONTS = ROOT / "brand_kit/assets", ROOT / "brand_kit/fonts"


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _canvas(value: dict) -> None:
    dimensions = [value.get(key) for key in ("width", "height")]
    if any(type(size) is not int or not 1 <= size <= 8192 for size in dimensions):
        raise ValueError("Canvas dimensions must be integers from 1 to 8192 px")
    if dimensions[0] * dimensions[1] > 20_000_000:
        raise ValueError("Canvas must not exceed 20 million pixels")


def _load_design(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("canvas"), dict):
        raise ValueError(f"{path} must contain a design with canvas and elements")
    if not isinstance(value.get("elements"), list):
        raise ValueError(f"{path} must contain an elements list")
    _canvas(value["canvas"])
    return value


def _contact_sheet(cells: list[tuple[str, str, Image.Image | None]]) -> Image.Image:
    columns, cell_width, cell_height = min(3, len(cells)), 520, 470
    sheet = Image.new(
        "RGB",
        (columns * cell_width, ((len(cells) + columns - 1) // columns) * cell_height),
        "#efeff0",
    )
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=14)
    for index, (label, detail, picture) in enumerate(cells):
        x, y = index % columns * cell_width, index // columns * cell_height
        draw.text((x + 16, y + 12), label, fill="#222222", font=font)
        draw.text((x + 16, y + 35), detail, fill="#555555", font=font)
        if picture is None:
            draw.text(
                (x + 16, y + 200), "RENDER FAILED - see states.json", fill="#a12b26", font=font
            )
            continue
        thumb = picture.convert("RGBA").copy()
        thumb.thumbnail((cell_width - 32, cell_height - 88))
        backdrop = Image.new("RGBA", thumb.size, "white")
        thumb = Image.alpha_composite(backdrop, thumb).convert("RGB")
        sheet.paste(
            thumb,
            (x + (cell_width - thumb.width) // 2, y + 70 + (cell_height - 88 - thumb.height) // 2),
        )
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task", type=Path, default=ROOT / "input/chameleon_sample_level2_core.json"
    )
    parser.add_argument("--out", type=Path, default=Path("output/composition"))
    parser.add_argument(
        "--baseline", type=Path, help="Previously saved design JSON, not a task JSON"
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args(argv)
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")

    started = time.perf_counter()
    errors, cells = [], []
    report = {
        "task": str(args.task.resolve()),
        "success": False,
        "errors": errors,
        "states": [],
        "limitations": [
            "Relationship inference is a source hypothesis, not universal semantic recognition.",
            "Proxy metrics and zero warnings do not establish aesthetic or endpoint quality.",
            "Only the listed states belong to this run; previous unlisted files are not results.",
        ],
    }
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "contact_sheet.png").unlink(missing_ok=True)
        task = json.loads(args.task.read_text(encoding="utf-8-sig"))
        source = task["source"]
        before = deepcopy(source)
        target = (
            {"width": args.width, "height": args.height}
            if args.width is not None
            else deepcopy(task["target_canvas"])
        )
        _canvas(source["canvas"])
        _canvas(target)
        source_digest = _digest(source)
        graph = build_relationship_graph(source, ASSETS)
        graph_data = asdict(graph)
        graph_digest = _digest(graph_data)
        _write_json(
            args.out / "graph.json",
            {"source_digest": source_digest, "graph_digest": graph_digest, "graph": graph_data},
        )
        context = build_context(source, ASSETS, FONTS)
        baseline = _load_design(args.baseline) if args.baseline else resize(source, target)
        if baseline["canvas"] != target:
            raise ValueError("Saved baseline canvas must match the requested target")
        baseline_label = "SAVED BASELINE" if args.baseline else "CURRENT DEFAULT (includes states)"
        report.update(
            target_canvas=target,
            source_digest=source_digest,
            graph_digest=graph_digest,
            graph_file="graph.json",
            baseline_kind="saved_json" if args.baseline else "current_default",
            baseline_path=str(args.baseline.resolve()) if args.baseline else None,
            baseline_digest=_digest(baseline),
        )

        def save(design, stem, label, axes=None):
            summary = {
                "name": label,
                "design_file": f"{stem}.json",
                "png_file": f"{stem}.png",
                "source_digest": source_digest,
                "graph_digest": graph_digest,
            }
            png_path = args.out / f"{stem}.png"
            png_path.unlink(missing_ok=True)
            _write_json(args.out / f"{stem}.json", design)
            try:
                expected_canvas = source["canvas"] if stem == "source" else target
                evaluation = score(source, design, context, target_canvas=expected_canvas)
                violations = evaluation["violations"]
                summary["contract"] = {
                    "valid": not violations,
                    "violations": [
                        {"code": item.code, "message": str(item)} for item in violations
                    ],
                }
                summary["quality_proxy"] = evaluation["total"]
                summary["metrics"] = evaluation["metrics"]
                summary["style"] = analyze_style(source, design, ASSETS)
                summary["relationships"] = analyze_relationships(graph, design, axes)
                metrics = evaluation["metrics"]
                minimum = metrics.get("raw_min_font_px")
                summary["readability"] = {
                    "min_font_px": minimum,
                    "comfort_score": metrics.get("text_size"),
                    "below_10px": minimum is not None and minimum < 10,
                }
                summary["collisions"] = metrics.get("raw_collisions")
                summary["diagnostic_codes"] = (
                    [item.code for item in violations]
                    + [f"style:{item['code']}" for item in summary["style"]["warnings"]]
                    + [
                        f"relationship:{item['code']}"
                        for item in summary["relationships"]["warnings"]
                    ]
                    + (["readability:below_10px"] if summary["readability"]["below_10px"] else [])
                    + (["collision:overlap"] if summary["collisions"] else [])
                )
                if violations:
                    errors.append(
                        {
                            "state": stem,
                            "phase": "contract",
                            "violations": summary["contract"]["violations"],
                        }
                    )
                _, warnings = convert_chameleon_to_crello(
                    design, _get_renderer(str(FONTS)).font_manager, str(ASSETS)
                )
                summary["render_warnings"] = warnings
                picture = render_image(
                    design,
                    ASSETS,
                    FONTS,
                    max_size=max(design["canvas"]["width"], design["canvas"]["height"]),
                )
                expected = (design["canvas"]["width"], design["canvas"]["height"])
                if picture.size != expected:
                    raise ValueError(
                        f"Renderer returned {picture.size}, expected full-size {expected}"
                    )
                picture.save(png_path)
                summary["rendered_dimensions"] = list(picture.size)
            except Exception as exc:  # noqa: BLE001 - write a truthful failed experiment report
                picture = None
                png_path.unlink(missing_ok=True)
                summary["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(
                    {"state": stem, "phase": "diagnostic_or_render", "error": summary["error"]}
                )
            detail = (
                f"{design['canvas']['width']}x{design['canvas']['height']}  "
                f"min text {summary.get('readability', {}).get('min_font_px')} px"
            )
            cells.append((label, detail, picture))
            return summary

        report["source"] = save(source, "source", "ORIGINAL SOURCE")
        report["baseline"] = save(baseline, "baseline", baseline_label)
        states = generate_composition_states(source, target, graph)
        report["proposal_status"] = "generated" if states else "unsupported_or_no_feasible_state"
        for index, state in enumerate(states):
            summary = save(state.design, f"state_{index:03d}", state.name, dict(state.cluster_axes))
            summary.update(
                topology=state.topology,
                width=state.width,
                height=state.height,
                cluster_axes=dict(state.cluster_axes),
                state_source_digest=state.source_digest,
                source_matches_graph=state.source_digest == source_digest,
            )
            if state.source_digest != source_digest:
                errors.append(
                    {
                        "state": state.name,
                        "phase": "source_digest",
                        "error": "State refers to a different source",
                    }
                )
            if (state.width, state.height) != (target["width"], target["height"]):
                errors.append(
                    {
                        "state": state.name,
                        "phase": "canvas",
                        "error": "State metadata target differs",
                    }
                )
            report["states"].append(summary)
        report["source_immutable"] = source == before and _digest(source) == source_digest
        report["graph_immutable"] = _digest(asdict(graph)) == graph_digest
        if not report["source_immutable"] or not report["graph_immutable"]:
            errors.append(
                {"phase": "immutability", "error": "Source or original relationship graph changed"}
            )
        report["contact_sheet"] = "contact_sheet.png"
        _contact_sheet(cells).save(args.out / "contact_sheet.png")
        report["success"] = not errors
    except Exception as exc:  # noqa: BLE001 - the CLI must return nonzero on a partial run
        errors.append({"phase": "setup_or_output", "error": f"{type(exc).__name__}: {exc}"})
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    try:
        _write_json(args.out / "states.json", report)
    except OSError as exc:
        print(f"Cannot write states.json: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(json.dumps(error), file=sys.stderr)
    for summary in report["states"]:
        print(
            f"{summary['name']}: codes={','.join(summary.get('diagnostic_codes', [])) or 'none'}; "
            f"min_font={summary.get('readability', {}).get('min_font_px')}; "
            f"collisions={summary.get('collisions')}"
        )
    print(
        f"{len(report['states'])} proposals; {len(errors)} failures. "
        f"Report: {args.out.resolve() / 'states.json'}"
    )
    print(
        "Inspect the source/baseline/proposal renders; "
        "these diagnostics do not prove visual quality."
    )
    return int(not report["success"])


if __name__ == "__main__":
    raise SystemExit(main())
