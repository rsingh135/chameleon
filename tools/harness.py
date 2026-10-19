"""Regression harness: run resize() over many tasks and targets, score every
result, render it, and lay everything out on a contact sheet.

    python -m tools.harness                       # samples x their own targets
    python -m tools.harness --targets all         # samples x whole catalogue
    python -m tools.harness --synth 40 --seed 1   # plus synthetic tasks
    python -m tools.harness --no-render           # JSON metrics only, fast

Outputs in --out (default eval_out/):
    report.json      every row: task, target, violations, metrics, timing
    report.md        summary tables (by family, by target, worst cases)
    sheet_*.png      contact sheets, one per task, source first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from engine.eval import build_context, score
from engine.eval.catalogue import CATALOGUE
from engine.eval.render import render_image, render_plate
from engine.resize import resize

ASSETS_DIR = Path("brand_kit/assets")
THUMB = (320, 320)
CELL_W, CELL_H, LABEL_H = 340, 400, 60


def load_tasks(input_dir: Path) -> list[dict]:
    tasks = []
    for path in sorted(input_dir.glob("*.json")):
        task = json.loads(path.read_text())
        task.setdefault("task_id", path.stem)
        task.setdefault("family", task.get("difficulty", "sample"))
        task.setdefault("holdout", False)
        tasks.append(task)
    return tasks


def targets_for(task: dict, spec: str) -> dict[str, dict]:
    if spec == "own":
        return {task.get("target_name", "own"): task["target_canvas"]}
    if spec == "all":
        return dict(CATALOGUE)
    out = {}
    for name in spec.split(","):
        name = name.strip()
        if name not in CATALOGUE:
            sys.exit(f"unknown target '{name}'; known: {', '.join(CATALOGUE)}")
        out[name] = CATALOGUE[name]
    return out


def run_one(
    task: dict, target_name: str, target: dict, render: bool
) -> tuple[dict, Image.Image | None]:
    source = task["source"]
    ctx = build_context(source, ASSETS_DIR)
    row = {
        "task_id": task["task_id"],
        "family": task["family"],
        "holdout": task["holdout"],
        "target": target_name,
        "target_canvas": target,
        "source_canvas": source["canvas"],
        "n_elements": len(source["elements"]),
    }
    t0 = time.perf_counter()
    try:
        candidate = resize(source, target)
    except Exception:  # noqa: BLE001 - a crash is a reliability failure we must record
        row.update(
            elapsed=time.perf_counter() - t0,
            crashed=True,
            error=traceback.format_exc(limit=3),
            violations=["crash"],
            total=0.0,
            metrics={},
        )
        return row, None
    row["elapsed"] = time.perf_counter() - t0
    row["crashed"] = False

    plate = image = None
    if render:
        try:
            plate = render_plate(candidate, ASSETS_DIR, max_size=1024)
            image = render_image(candidate, ASSETS_DIR, max_size=1024)
        except Exception:  # noqa: BLE001
            row["render_error"] = traceback.format_exc(limit=3)
    result = score(source, candidate, ctx, plate, target)
    row["violations"] = [str(v) for v in result["violations"]]
    row["violation_codes"] = sorted({v.code for v in result["violations"]})
    row["total"] = result["total"]
    row["metrics"] = {k: round(v, 4) for k, v in result["metrics"].items()}
    row["candidate"] = candidate
    return row, image


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _cell(image: Image.Image | None, title: str, lines: list[str], good: bool) -> Image.Image:
    cell = Image.new("RGB", (CELL_W, CELL_H), (244, 240, 234))
    draw = ImageDraw.Draw(cell)
    if image is not None:
        thumb = image.convert("RGB").copy()
        thumb.thumbnail(THUMB)
        x = (CELL_W - thumb.width) // 2
        y = LABEL_H + (CELL_H - LABEL_H - thumb.height) // 2
        cell.paste(thumb, (x, y))
        draw.rectangle([x - 1, y - 1, x + thumb.width, y + thumb.height], outline=(120, 120, 120))
    else:
        draw.text((20, CELL_H // 2), "render failed / crashed", fill=(160, 30, 30), font=_font(14))
    draw.rectangle([0, 0, CELL_W, LABEL_H], fill=(60, 180, 90) if good else (200, 60, 60))
    draw.text((8, 4), title, fill="white", font=_font(14))
    for i, line in enumerate(lines[:2]):
        draw.text((8, 22 + i * 17), line[:48], fill="white", font=_font(12))
    return cell


def contact_sheet(
    task: dict, rows: list[dict], images: list[Image.Image | None], render: bool
) -> Image.Image:
    cells = []
    src = task["source"]
    src_img = render_image(src, ASSETS_DIR, max_size=1024) if render else None
    cells.append(
        _cell(
            src_img,
            f"SOURCE {src['canvas']['width']}x{src['canvas']['height']}",
            [task["task_id"], task["family"]],
            True,
        )
    )
    for row, img in zip(rows, images, strict=True):
        tc = row["target_canvas"]
        ok = not row["violations"]
        detail = ", ".join(row.get("violation_codes") or []) or f"total {row['total']:.2f}"
        m = row["metrics"]
        second = (
            f"col {m.get('collision_free', 0):.2f} txt {m.get('text_size', 0):.2f} "
            f"con {m.get('text_contrast', 0):.2f} grp {m.get('group_integrity', 0):.2f}"
            if m
            else (row.get("error", "").strip().splitlines() or [""])[-1]
        )
        cells.append(
            _cell(img, f"{row['target']} {tc['width']}x{tc['height']}", [detail, second], ok)
        )
    cols = min(5, len(cells))
    rows_n = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL_W, rows_n * CELL_H), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * CELL_W, (i // cols) * CELL_H))
    return sheet


def summarize(rows: list[dict]) -> str:
    def block(title: str, key) -> str:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(str(key(r)), []).append(r)
        out = [
            f"\n### {title}\n",
            "| group | n | valid | mean total (valid) | crashes |",
            "|---|---|---|---|---|",
        ]
        for name, g in sorted(groups.items()):
            valid = [r for r in g if not r["violations"]]
            mean = sum(r["total"] for r in valid) / len(valid) if valid else 0.0
            crashes = sum(r["crashed"] for r in g)
            out.append(f"| {name} | {len(g)} | {len(valid)} | {mean:.3f} | {crashes} |")
        return "\n".join(out)

    valid = [r for r in rows if not r["violations"]]
    codes = Counter(c for r in rows for c in r.get("violation_codes") or ["crash"] * r["crashed"])
    mean_total = sum(r["total"] for r in valid) / len(valid) if valid else 0.0
    mean_elapsed = sum(r["elapsed"] for r in rows) / len(rows)
    max_elapsed = max(r["elapsed"] for r in rows)
    lines = [
        "# Harness report\n",
        f"- rows: {len(rows)}  valid: {len(valid)}  crashes: {sum(r['crashed'] for r in rows)}",
        f"- mean total over valid rows: {mean_total:.3f}",
        f"- resize() time: mean {mean_elapsed * 1000:.1f}ms  max {max_elapsed * 1000:.1f}ms",
        "\n### Violations by code\n",
        "| code | rows |",
        "|---|---|",
        *[f"| {c} | {n} |" for c, n in codes.most_common()],
        block("By family", lambda r: r["family"]),
        block("Seen vs holdout", lambda r: "holdout" if r["holdout"] else "seen"),
        block("By target", lambda r: r["target"]),
        "\n### Worst valid rows\n",
        "| task | target | total | weakest metrics |",
        "|---|---|---|---|",
    ]
    for r in sorted(valid, key=lambda r: r["total"])[:15]:
        weakest = sorted(((v, k) for k, v in r["metrics"].items() if not k.startswith("raw_")))[:3]
        lines.append(
            f"| {r['task_id']} | {r['target']} | {r['total']:.3f} | "
            + ", ".join(f"{k} {v:.2f}" for v, k in weakest)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=Path("input"))
    ap.add_argument(
        "--targets", default="own", help="'own', 'all', or comma-separated catalogue names"
    )
    ap.add_argument("--synth", type=int, default=0, help="add N synthetic tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval_out"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N tasks")
    args = ap.parse_args()

    tasks = load_tasks(args.input)
    if args.synth:
        from tools.synth import generate

        tasks += generate(args.synth, args.seed)
    if args.limit:
        tasks = tasks[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    render = not args.no_render

    all_rows: list[dict] = []
    for task in tasks:
        rows, images = [], []
        for name, target in targets_for(task, args.targets).items():
            row, img = run_one(task, name, target, render)
            rows.append(row)
            images.append(img)
            flag = "OK " if not row["violations"] else "BAD"
            codes = ",".join(row.get("violation_codes") or [])
            print(f"{flag} {task['task_id']:<28} {name:<20} total={row['total']:.3f} {codes}")
        if render and not args.no_sheets:
            contact_sheet(task, rows, images, render).save(
                args.out / f"sheet_{task['task_id']}.png"
            )
        all_rows += rows

    (args.out / "report.json").write_text(json.dumps(all_rows, indent=1, default=str))
    (args.out / "report.md").write_text(summarize(all_rows))
    print(f"\n{len(all_rows)} rows -> {args.out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
