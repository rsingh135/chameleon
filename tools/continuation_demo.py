"""Render the isolated orientation-continuation prototype and its default baseline.

python -m tools.continuation_demo --task input/chameleon_sample_level2_core.json \
    --out output/continuation --steps 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from engine.eval import build_context, score
from engine.eval.render import render_plate
from engine.layout.continuation import ROOT, continue_layout
from renderer.converter.convert import render_design_to_png


def _sheet(cells: list[tuple[str, str, Image.Image | None]], columns: int) -> Image.Image:
    width, height = 480, 450
    columns = min(columns, len(cells))
    sheet = Image.new(
        "RGB", (columns * width, ((len(cells) + columns - 1) // columns) * height), "#eeeae3"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, (title, detail, image) in enumerate(cells):
        x, y = index % columns * width, index // columns * height
        draw.text((x + 14, y + 12), title, fill="#17282b", font=font)
        draw.text((x + 14, y + 34), detail, fill="#596367", font=font)
        if image is None:
            draw.text(
                (x + 14, y + 180), "RENDER FAILED - see report.json", fill="#a12b26", font=font
            )
            continue
        thumb = image.convert("RGB").copy()
        thumb.thumbnail((width - 28, height - 82))
        sheet.paste(
            thumb, (x + (width - thumb.width) // 2, y + 66 + (height - 82 - thumb.height) // 2)
        )
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/continuation"))
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--beam-width", type=int, default=4)
    args = parser.parse_args(argv)
    task = json.loads(args.task.read_text(encoding="utf-8"))
    started = time.perf_counter()
    report = continue_layout(
        task["source"],
        task["target_canvas"],
        steps=args.steps,
        beam_width=args.beam_width,
    )
    report["task"] = str(args.task.resolve())
    report["search_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    assets, fonts = ROOT / "brand_kit/assets", ROOT / "brand_kit/fonts"
    context = build_context(task["source"], assets, fonts)
    render_errors = []

    def save(design: dict, stem: str, summary: dict) -> Image.Image | None:
        (args.out / f"{stem}.json").write_text(
            json.dumps(design, indent=2, allow_nan=False), encoding="utf-8"
        )
        canvas = design["canvas"]
        try:
            warnings = render_design_to_png(
                design,
                assets,
                args.out / f"{stem}.png",
                fonts_dir=fonts,
                max_size=max(canvas["width"], canvas["height"]),
            )
            summary["render_warnings"] = warnings
            plate = render_plate(design, assets, fonts, max_size=1024)
            rendered = score(task["source"], design, context, plate, canvas)
            summary["rendered_quality"] = rendered["total"]
            summary["rendered_metrics"] = rendered["metrics"]
            summary["rendered_violations"] = [
                str(violation) for violation in rendered["violations"]
            ]
            with Image.open(args.out / f"{stem}.png") as image:
                return image.copy()
        except Exception as exc:
            summary["render_error"] = str(exc)
            render_errors.append({"frame": stem, "error": str(exc)})
            print(f"{stem}: render failed: {exc}", file=sys.stderr)
            return None

    source_summary = {}
    source_image = save(task["source"], "source", source_summary)
    baseline_image = save(report["baseline_frames"][-1], "baseline", report["baseline_anchors"][-1])
    path_images = [
        save(design, f"anchor_{index:02d}", summary)
        for index, (design, summary) in enumerate(
            zip(report["frames"], report["anchors"], strict=True)
        )
    ]
    prototype_image = save(report["frames"][-1], "prototype", report["anchors"][-1])
    report["source_render"] = source_summary
    report["render_errors"] = render_errors
    comparison = report["comparison"]
    comparison["prototype_rendered_endpoint_quality"] = report["anchors"][-1].get(
        "rendered_quality"
    )
    comparison["default_rendered_endpoint_quality"] = report["baseline_anchors"][-1].get(
        "rendered_quality"
    )
    comparison["rendered_quality_guard_enforced"] = comparison["endpoint_design_matches_baseline"]
    report["limitations"].append(
        "The final design exactly matches the default endpoint; only the transition path is varied."
    )
    report["files"] = {
        "comparison": "comparison.png",
        "path": "path.png",
        "source": "source.png",
        "baseline": "baseline.png",
        "prototype": "prototype.png",
    }
    baseline_quality = comparison["default_endpoint_quality"]
    prototype_quality = comparison["prototype_endpoint_quality"]
    _sheet(
        [
            ("ORIGINAL SOURCE", "The supplied design, unchanged", source_image),
            ("CURRENT DEFAULT ENGINE", f"JSON proxy {baseline_quality:.3f}", baseline_image),
            (
                "CONTINUATION PROTOTYPE",
                f"JSON proxy {prototype_quality:.3f}; discrete anchors only",
                prototype_image,
            ),
        ],
        3,
    ).save(args.out / "comparison.png")
    _sheet(
        [
            (
                f"{anchor['index']:02d}  {anchor['canvas']['width']}x{anchor['canvas']['height']}  "
                f"{anchor['origin']}",
                f"quality {anchor['quality']:.3f}  min font {anchor['min_font_px']}  "
                f"jump {anchor['movement']['max_center_jump']:.3f}",
                image,
            )
            for anchor, image in zip(report["anchors"], path_images, strict=True)
        ],
        4,
    ).save(args.out / "path.png")
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Prototype: {args.out.resolve() / 'comparison.png'}")
    print(f"Discrete path: {args.out.resolve() / 'path.png'}")
    print(
        f"JSON endpoint quality: default {baseline_quality:.4f}; prototype {prototype_quality:.4f}"
    )
    print(
        "Max normalized center jump: "
        f"default anchors {comparison['default_anchor_max_center_jump']:.4f}; "
        f"prototype {comparison['prototype_max_center_jump']:.4f}"
    )
    print("Experimental discrete search; intermediate motion is not validated. See report.json.")
    return int(bool(render_errors or comparison["selected_constraint_violations"]))


if __name__ == "__main__":
    raise SystemExit(main())
