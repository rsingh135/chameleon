"""Audit prepared external posters and publish an offline evidence gallery.

python -m tools.poster_audit --manifest fixtures/open_posters/catalogue.json \
    --out output/poster-audit

Reserve examples are import-only: this module never resizes them. A technically
clean render still requires human comparison with the native source; proxy
metrics never assign a visual approval.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import textwrap
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

TARGETS = {
    "landscape": {"width": 1600, "height": 900},
    "square": {"width": 1080, "height": 1080},
    "story": {"width": 1080, "height": 1920},
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}\Z")
_OWNERSHIP = {
    "reading_order",
    "explicit_group",
    "peer_order",
    "campaign_ownership",
    "row_attachment",
}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _inside(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Catalogue paths must be relative to their corpus root")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Catalogue path escapes its corpus root")
    return path


def _prepared_digest(prepared):
    files = {}
    for path in sorted(prepared.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(prepared):
                raise ValueError("Prepared resource escapes its artifact directory")
            files[path.relative_to(prepared).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return _digest(files)


def _resize(source, target):
    from engine.resize import resize_with_state

    return resize_with_state(source, target)


def _render(design):
    from engine.resources import get_renderer, get_resources, render_input
    from renderer.converter.convert import convert_chameleon_to_crello

    resources = get_resources()
    renderer = get_renderer()
    example, warnings = convert_chameleon_to_crello(
        render_input(design), renderer.font_manager, str(resources.assets_dir)
    )
    from io import BytesIO

    image = renderer.render(example, max_size=max(design["canvas"].values()), format="png")
    return Image.open(BytesIO(image)).convert("RGBA"), warnings


def _resource_check(source, resources):
    from engine.resources import load_font_manager

    if not (resources.fonts_dir / "font-manifest.json").is_file():
        raise ValueError("Imported resources need an explicit font-manifest.json")
    manager = load_font_manager(resources.fonts_dir)
    fonts = {}
    for element in source["elements"]:
        if element["type"] == "image":
            path = (resources.assets_dir / element["asset_id"]).resolve()
            if not path.is_relative_to(resources.assets_dir) or not path.is_file():
                raise ValueError(f"Missing imported image bytes: {element['asset_id']}")
        if element["type"] == "text":
            weight = element.get("font_weight", "regular")
            italic = element.get("font_style") == "italic"
            style = (
                "bolditalic"
                if weight == "bold" and italic
                else ("bold" if weight == "bold" else "italic" if italic else "regular")
            )
            family = element["font_family"]
            data = manager.lookup(family, weight, style)
            fonts[f"{family}|{weight}|{style}"] = hashlib.sha256(data).hexdigest()
    return {"passed": True, "font_faces": fonts}


def _pixel_comparison(native, converted):
    if native.size != converted.size:
        return {
            "same_canvas": False,
            "native_size": list(native.size),
            "converted_size": list(converted.size),
            "rgb_mae": None,
        }

    def paper(image):
        image = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")

    difference = ImageChops.difference(paper(native), paper(converted))
    return {
        "same_canvas": True,
        "rgb_mae": sum(ImageStat.Stat(difference).mean) / (3 * 255),
        "interpretation": "Pixel diagnostic only; inspect native versus converted source.",
    }


def _visible(design, ids):
    from engine.layout.appearance import unit_visible_box

    members = {element["id"]: element for element in design["elements"]}
    if not ids or any(eid not in members for eid in ids):
        raise ValueError("Invariant element_ids must name existing source/output elements")
    return unit_visible_box([members[eid] for eid in ids])


def evaluate_invariants(source, design, definitions, target_name, source_pixels=None):
    """Evaluate only source-authored rules; never derive expectations from outputs.

    Rules: center_x(element_ids,max_drift), relative_order(first,second,axis),
    scale_ratio(first,second,min_ratio,max_ratio), minimum_font(element_ids,min_px),
    group_similarity(element_ids,max_log_error), and source_pixel_mae(maximum).
    Optional targets limits a rule to named formats (or source).
    """
    results = []
    originals = {e["id"]: e for e in source["elements"]}
    outputs = {e["id"]: e for e in design["elements"]}
    for index, rule in enumerate(definitions):
        row = {"name": rule.get("name", f"rule-{index + 1}"), "kind": rule.get("kind")}
        if target_name not in rule.get("targets", [*TARGETS]):
            continue
        try:
            kind = rule["kind"]
            ids = rule.get("element_ids", [])
            if kind == "center_x":
                before, after = _visible(source, ids), _visible(design, ids)
                expected = (before.x + before.width / 2) / source["canvas"]["width"]
                actual = (after.x + after.width / 2) / design["canvas"]["width"]
                value = abs(actual - expected)
                passed = value <= rule.get("max_drift", 0.04)
            elif kind == "relative_order":
                first, second = _visible(design, rule["first"]), _visible(design, rule["second"])
                # Requiring source corroboration catches a mis-authored rule.
                a, b = _visible(source, rule["first"]), _visible(source, rule["second"])
                axis = rule.get("axis", "y")
                if axis not in {"x", "y"}:
                    raise ValueError("relative_order axis must be x or y")
                length = "width" if axis == "x" else "height"

                def center(box, axis=axis, length=length):
                    return getattr(box, axis) + getattr(box, length) / 2

                if center(a) > center(b):
                    raise ValueError("Declared order contradicts the original source")
                value = (center(second) - center(first)) / design["canvas"][length]
                passed = value >= -rule.get("tolerance", 0.0)
            elif kind == "scale_ratio":
                a, b = _visible(source, rule["first"]), _visible(source, rule["second"])
                first, second = _visible(design, rule["first"]), _visible(design, rule["second"])
                value = (first.height / a.height) / (second.height / b.height)
                passed = rule.get("min_ratio", 0.85) <= value <= rule.get("max_ratio", 1.15)
            elif kind == "minimum_font":
                selected = ids or [eid for eid, e in originals.items() if e["type"] == "text"]
                if not selected or any(originals[eid]["type"] != "text" for eid in selected):
                    raise ValueError("minimum_font must identify original text elements")
                value = min(outputs[eid]["font_size"] for eid in selected)
                passed = value >= rule.get("min_px", 10.0)
            elif kind == "group_similarity":
                before, after = _visible(source, ids), _visible(design, ids)
                scales = [
                    outputs[eid][field] / originals[eid][field]
                    for eid in ids
                    for field in ("width", "height", "font_size")
                    if originals[eid].get(field, 0) > 0
                ]
                scale = math.exp(sum(math.log(value) for value in scales) / len(scales))
                errors = [abs(math.log(value / scale)) for value in scales]
                for eid in ids:
                    for axis, start in (("x", "x"), ("y", "y")):
                        delta = (outputs[eid][axis] - getattr(after, start)) / scale
                        expected = originals[eid][axis] - getattr(before, start)
                        errors.append(abs(delta - expected) / max(before.width, before.height, 1))
                value = max(errors, default=0.0)
                passed = value <= rule.get("max_log_error", 0.05)
            elif kind == "source_pixel_mae":
                if source_pixels is None or source_pixels.get("rgb_mae") is None:
                    raise ValueError("Native/converted pixel comparison is unavailable")
                value = source_pixels["rgb_mae"]
                passed = value <= rule["maximum"]
            else:
                raise ValueError(f"Unknown invariant kind: {kind!r}")
            row.update(passed=bool(passed), value=value)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        results.append(row)
    return {
        "status": "configured" if results else "unconfigured",
        "rules": results,
        "passed": all(row["passed"] for row in results) if results else None,
    }


def _measure(source, design, target, state, context, graph, definitions, target_name):
    from engine.eval import check_contract, score
    from engine.eval.proportions import analyze_proportions
    from engine.eval.style import analyze_style
    from engine.layout.relationships import analyze_relationships

    violations = check_contract(source, design, target, context)
    result = score(source, design, context, target_canvas=target)
    source_ids = [e["id"] for e in source["elements"]]
    output_ids = [e["id"] for e in design["elements"]]
    originals = {e["id"]: e for e in source["elements"]}
    outputs = {e["id"]: e for e in design["elements"]}
    membership = Counter(source_ids) == Counter(output_ids) and len(set(output_ids)) == len(
        output_ids
    )
    copied = membership and all(
        original.get(field) == outputs[eid].get(field)
        for eid, original in originals.items()
        for field in (
            "content",
            "font_family",
            "font_weight",
            "font_style",
            "style_runs",
            "asset_id",
        )
    )
    fonts = [e["font_size"] for e in design["elements"] if e["type"] == "text"]
    threshold = max(10.0, 0.012 * min(target.values()))
    relationships = (
        analyze_relationships(graph, design, state.get("cluster_axes", {})) if membership else None
    )
    ownership = (
        [w for w in relationships["warnings"] if w["code"] in _OWNERSHIP] if relationships else []
    )
    invariants = evaluate_invariants(source, design, definitions, target_name)
    checks = {
        "contract": {"passed": not violations, "violations": [str(v) for v in violations]},
        "identity": {
            "passed": copied,
            "exact_membership": membership,
            "copy_fonts_assets_unchanged": copied,
        },
        "readability": {
            "passed": all(size >= threshold for size in fonts),
            "minimum_px": min(fonts, default=None),
            "threshold_px": threshold,
            "below_threshold": sum(size < threshold for size in fonts),
        },
        "collisions": {
            "passed": result["metrics"].get("raw_collisions", math.inf) == 0,
            "raw": result["metrics"].get("raw_collisions"),
        },
        "ownership": {"passed": membership and not ownership, "warnings": ownership},
    }
    return {
        "checks": checks,
        "invariants": invariants,
        "technical_pass": all(row["passed"] for row in checks.values())
        and invariants["passed"] is not False,
        "evaluation": {"total": result["total"], "metrics": result["metrics"]},
        "relationships": relationships,
        "proportions": analyze_proportions(source, design, graph) if membership else None,
        "style": analyze_style(source, design, relationship_graph=graph) if membership else None,
    }


def _failure(row, phase, exc):
    row.setdefault("failures", []).append({"phase": phase, "error": f"{type(exc).__name__}: {exc}"})


def _review(entry, key, evidence_digest):
    review = entry.get("visual_reviews", {}).get(key)
    if not isinstance(review, dict) or review.get("evidence_digest") != evidence_digest:
        return {
            "status": "pending",
            "notes": "Human comparison with the native source is required.",
        }
    if review.get("status") not in {"approved", "rejected", "needs_changes"} or not review.get(
        "reviewer"
    ):
        return {"status": "pending", "notes": "Review needs a reviewer and explicit decision."}
    return deepcopy(review)


def _copy_evidence(entry, root, prepared, folder, row):
    original = _inside(root, entry["original_path"])
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError("Original poster bytes do not match the curated SHA-256")
    if (prepared / "original.svg").read_bytes() != original.read_bytes():
        raise ValueError("Prepared original.svg differs from the curated original")
    shutil.copyfile(original, folder / "original.svg")
    row["original_file"] = "original.svg"
    license_path = _inside(root, entry["license_path"])
    shutil.copyfile(license_path, folder / "license.txt")
    row["license_file"] = "license.txt"
    native_path = prepared / "native.png"
    (folder / "native.png").unlink(missing_ok=True)
    try:
        with Image.open(native_path) as image:
            image.load()
            row["native_size"] = list(image.size)
        shutil.copyfile(native_path, folder / "native.png")
        row["native_file"] = "native.png"
    except (OSError, ValueError, SyntaxError) as exc:
        _failure(row, "native_render", exc)


def _case(entry, root, out, gallery_only):
    from engine.resources import using_resources

    folder = out / entry["id"]
    folder.mkdir(parents=True, exist_ok=True)
    saved = (
        _read(folder / "report.json")
        if gallery_only and (folder / "report.json").is_file()
        else None
    )
    row = {
        "id": entry["id"],
        "title": entry.get("title", entry["id"]),
        "split": entry["split"],
        "attribution": {
            key: entry.get(key)
            for key in ("author", "source_url", "license_id", "license_url", "sha256")
        },
        "targets": [],
        "failures": [],
        "visual_review": {"status": "pending"},
    }
    try:
        prepared = _inside(root, entry["prepared_path"])
        imported = _read(prepared / "import-report.json")
        row["import"] = imported
        _copy_evidence(entry, root, prepared, folder, row)
        if imported.get("status") != "ready":
            raise ValueError(
                f"Import {imported.get('status', 'unknown')}: {imported.get('failures', [])}"
            )
        source = _read(prepared / "source.json")
        before = deepcopy(source)
        source_digest = _digest(source)
        row["source_digest"] = source_digest
        row["prepared_digest"] = _prepared_digest(prepared)
        _json(folder / "source.json", source)
        row["source_file"] = "source.json"
        if entry["split"] == "reserve":
            row["status"] = "reserve_untested"
            row["visual_review"] = {
                "status": "untested",
                "notes": "Prepared import only; never sent through the resize engine.",
            }
            return row
        if gallery_only:
            if (
                saved
                and saved.get("source_digest") == source_digest
                and saved.get("prepared_digest") == row["prepared_digest"]
            ):
                for key in (
                    "targets",
                    "converted_file",
                    "source_pixels",
                    "source_invariants",
                    "source_resources",
                    "source_immutable",
                ):
                    if key in saved:
                        row[key] = saved[key]
                row["failures"].extend(saved.get("failures", []))
                for target in row["targets"]:
                    filename = folder / target.get("png_file", "missing-render.png")
                    if not filename.is_file() or hashlib.sha256(
                        filename.read_bytes()
                    ).hexdigest() != target.get("png_digest"):
                        target.pop("png_file", None)
                        _failure(
                            target, "saved_evidence", ValueError("Saved PNG is missing or changed")
                        )
                row["status"] = "saved_evidence_pending_visual_review"
            else:
                row["status"] = "untested"
                row["failures"].append(
                    {
                        "phase": "saved_evidence",
                        "error": "No source-matching prior audit artifacts; no engine was run.",
                    }
                )
            return row
        resources_dir = prepared / "resources"
        with using_resources(resources_dir / "assets", resources_dir / "fonts") as resources:
            try:
                row["source_resources"] = _resource_check(source, resources)
            except Exception as exc:
                _failure(row, "resources", exc)
                return row
            from engine.eval import build_context
            from engine.layout.relationships import build_relationship_graph

            try:
                context = build_context(source)
                graph = build_relationship_graph(source)
            except Exception as exc:
                _failure(row, "source_analysis", exc)
                return row
            (folder / "converted-source.png").unlink(missing_ok=True)
            try:
                picture, warnings = _render(source)
                picture.save(folder / "converted-source.png")
                row["converted_file"] = "converted-source.png"
                row["source_render_warnings"] = warnings
                if warnings:
                    _failure(row, "source_render", ValueError("; ".join(warnings)))
                if row.get("native_file"):
                    with Image.open(folder / "native.png") as native:
                        row["source_pixels"] = _pixel_comparison(native, picture)
                else:
                    row["source_pixels"] = {"same_canvas": False, "rgb_mae": None}
                row["source_invariants"] = evaluate_invariants(
                    source, source, entry.get("invariants", []), "source", row["source_pixels"]
                )
            except Exception as exc:
                _failure(row, "source_render", exc)
            for name, target in TARGETS.items():
                cell = {"name": name, "canvas": dict(target), "failures": []}
                row["targets"].append(cell)
                for suffix in (".png", ".plan.json"):
                    (folder / f"{name}{suffix}").unlink(missing_ok=True)
                try:
                    working = deepcopy(source)
                    plan = _resize(working, dict(target))
                    if working != before:
                        raise ValueError("The engine mutated the immutable original source")
                    if plan["state"].get("source_digest") != source_digest:
                        raise ValueError("Selected plan is not tied to the original source digest")
                    _json(folder / f"{name}.plan.json", plan)
                    cell.update(
                        plan_file=f"{name}.plan.json",
                        state=plan["state"],
                        evidence_digest=_digest(plan["design"]),
                    )
                except Exception as exc:
                    _failure(cell, "engine", exc)
                    continue
                try:
                    cell.update(
                        _measure(
                            source,
                            plan["design"],
                            target,
                            plan["state"],
                            context,
                            graph,
                            entry.get("invariants", []),
                            name,
                        )
                    )
                except Exception as exc:
                    _failure(cell, "checks", exc)
                try:
                    picture, warnings = _render(plan["design"])
                    if picture.size != (target["width"], target["height"]):
                        raise ValueError(
                            "Rendered PNG dimensions do not match the requested canvas"
                        )
                    picture.save(folder / f"{name}.png")
                    cell.update(png_file=f"{name}.png", render_warnings=warnings)
                    if warnings:
                        _failure(cell, "render", ValueError("; ".join(warnings)))
                    cell["png_digest"] = hashlib.sha256(
                        (folder / cell["png_file"]).read_bytes()
                    ).hexdigest()
                    cell["evidence_digest"] = _digest(
                        {
                            "design": plan["design"],
                            "png": cell["png_digest"],
                            "prepared": row["prepared_digest"],
                        }
                    )
                except Exception as exc:
                    _failure(cell, "render", exc)
                cell["visual_review"] = _review(entry, name, cell["evidence_digest"])
            row["source_immutable"] = source == before and _digest(source) == source_digest
            if not row["source_immutable"]:
                _failure(row, "source_integrity", ValueError("Source changed during audit"))
        row["status"] = "rendered_pending_visual_review"
    except Exception as exc:
        _failure(row, "import", exc)
        row["status"] = "import_failed"
    return row


def _cells(row, out):
    folder = out / row["id"]
    cells = []
    for key, label in (
        ("native_file", "Native original SVG raster"),
        ("converted_file", "Imported source renderer"),
    ):
        if row.get(key):
            cells.append((f"{row['id']} · {label}", row["split"], folder / row[key]))
    for target in row["targets"]:
        status = (
            "CHECKS FAILED"
            if target.get("technical_pass") is False or target["failures"]
            else "visual review pending"
        )
        cells.append(
            (
                f"{row['id']} · {target['name']}",
                status,
                folder / target["png_file"] if target.get("png_file") else None,
            )
        )
    return cells or [(row["id"], row.get("status", "failed"), None)]


def _contact_sheet(cells, path):
    columns, width, height = min(3, len(cells)), 440, 400
    sheet = Image.new("RGB", (columns * width, math.ceil(len(cells) / columns) * height), "#f0f0ee")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=13)
    for index, (label, detail, filename) in enumerate(cells):
        x, y = index % columns * width, index // columns * height
        draw.text((x + 14, y + 12), textwrap.shorten(label, 58), fill="#222222", font=font)
        draw.text((x + 14, y + 33), textwrap.shorten(detail, 58), fill="#666666", font=font)
        if filename and filename.is_file():
            with Image.open(filename) as image:
                image = image.convert("RGBA")
                image.thumbnail((width - 28, height - 80))
                image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
                sheet.paste(
                    image.convert("RGB"),
                    (x + (width - image.width) // 2, y + 64 + (height - 80 - image.height) // 2),
                )
        else:
            draw.text((x + 14, y + 180), "NO RENDER · see report.json", fill="#9d302b", font=font)
    sheet.save(path)


def _link(value, label):
    if isinstance(value, str) and urlsplit(value).scheme in {"http", "https"}:
        return f'<a href="{html.escape(value, quote=True)}">{html.escape(label)}</a>'
    return html.escape(label)


def _gallery(report, out):
    sections, all_cells = [], []
    for row in report["examples"]:
        cells = _cells(row, out)
        all_cells.extend(cells)
        _contact_sheet(cells, out / row["id"] / "contact_sheet.png")
        pictures = []
        for label, detail, filename in cells:
            relative = (
                html.escape(filename.relative_to(out).as_posix(), quote=True) if filename else ""
            )
            alternative = html.escape(label, quote=True)
            content = (
                f'<img loading="lazy" src="{relative}" alt="{alternative}">'
                if filename
                else '<div class="missing">No render available</div>'
            )
            pictures.append(
                f"<figure>{content}<figcaption>{html.escape(label)}<br><small>{html.escape(detail)}</small></figcaption></figure>"
            )
        attribution = row["attribution"]
        source = _link(attribution.get("source_url"), "Original source")
        license_link = _link(
            attribution.get("license_url"), str(attribution.get("license_id", "License"))
        )
        failures = row["failures"] + [
            failure for target in row["targets"] for failure in target["failures"]
        ]
        errors = "".join(
            f"<li>{html.escape(failure['phase'] + ': ' + failure['error'])}</li>"
            for failure in failures
        )
        sections.append(
            f'<section id="{row["id"]}"><h2>{html.escape(row["title"])}</h2>'
            f'<p class="status">{html.escape(row.get("status", "failed"))} · '
            f"{html.escape(row['split'])} set · human review pending</p>"
            f"<p>{html.escape(str(attribution.get('author', '')))} · {source} · "
            f'{license_link} · <a href="{row["id"]}/report.json">Full checks</a></p>'
            f'<ul class="errors">{errors}</ul><div class="grid">'
            f"{''.join(pictures)}</div></section>"
        )
    _contact_sheet(all_cells, out / "contact_sheet.png")
    summary = html.escape(json.dumps(report["summary"], indent=2))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>External poster audit</title><style>
body{{margin:0;background:#f5f5f2;color:#222;font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1720px;margin:auto;padding:40px 24px}}
h1{{font-size:32px;letter-spacing:-.7px;margin:0 0 12px}}h2{{font-size:22px;margin-bottom:5px}}
a{{color:#36539a}}section{{margin:40px 0;border-top:1px solid #d5d5d1;padding-top:18px}}
.status{{font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}
figure{{margin:0}}figure img,.missing{{width:100%;height:400px;object-fit:contain;
background:white;border:1px solid #ddd;box-sizing:border-box}}
figcaption{{padding:8px 0}}small{{color:#666}}.errors{{color:#9b342c}}
.missing{{display:grid;place-content:center;color:#9b342c}}
pre{{white-space:pre-wrap;background:#eaeae6;padding:16px;font-size:12px}}
.notice{{max-width:900px}}
</style><main><h1>External poster audit</h1><p class="notice">
Native original, imported source, and selected adaptations. Structural checks and metrics are
diagnostics; human visual review remains pending unless a matching review record explicitly
approves the rendered evidence. Reserve posters are prepared imports only and have never been
resized by this harness.</p><p><a href="report.json">Summary JSON</a> ·
<a href="contact_sheet.png">Contact sheet</a></p><pre>{summary}</pre>
{"".join(sections)}</main></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def run_audit(manifest_path, out, case_ids=None, gallery_only=False):
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    root, catalogue = manifest_path.parent, _read(manifest_path)
    if catalogue.get("schema_version") != 1 or not isinstance(catalogue.get("examples"), list):
        raise ValueError("Expected catalogue schema_version 1 and examples array")
    entries = catalogue["examples"]
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Catalogue examples must be objects")
    ids = [entry.get("id") for entry in entries]
    if any(not isinstance(eid, str) or not _SAFE_ID.fullmatch(eid) for eid in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("Poster IDs must be unique portable identifiers")
    if any(entry.get("split") not in {"reviewed", "reserve"} for entry in entries):
        raise ValueError("Poster split must be reviewed or reserve")
    selected = set(case_ids or ids)
    if selected - set(ids) or not selected:
        raise ValueError("The case filter names unknown or no posters")
    if out == root or out in root.parents or root in out.parents:
        raise ValueError("Audit output must not overwrite the corpus root")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "gallery_only": gallery_only,
        "scope": "External poster evidence; metrics never imply visual approval.",
        "examples": [],
    }
    started = time.perf_counter()
    for entry in entries:
        if entry["id"] not in selected:
            continue
        row = _case(entry, root, out, gallery_only)
        _json(out / entry["id"] / "report.json", row)
        report["examples"].append(row)
        print(
            f"{row['id']}: {row.get('status', 'failed')} ({len(row['targets'])} target results)",
            flush=True,
        )
    rows = report["examples"]
    targets = [target for row in rows for target in row["targets"]]
    report["summary"] = {
        "reviewed_posters": sum(row["split"] == "reviewed" for row in rows),
        "reserve_posters_untested": sum(row["split"] == "reserve" for row in rows),
        "import_failures": sum(row.get("status") == "import_failed" for row in rows),
        "phase_failures": sum(len(row["failures"]) for row in rows)
        + sum(len(t["failures"]) for t in targets),
        "target_results": len(targets),
        "technical_check_failures": sum(
            target.get("technical_pass") is False for target in targets
        ),
        "source_check_failures": sum(
            row.get("source_invariants", {}).get("passed") is False for row in rows
        ),
        "visual_reviews_pending": sum(
            t.get("visual_review", {}).get("status", "pending") == "pending" for t in targets
        ),
        "visual_reviews_rejected": sum(
            t.get("visual_review", {}).get("status") in {"rejected", "needs_changes"}
            for t in targets
        ),
        "unconfigured_target_invariants": sum(
            t.get("invariants", {}).get("status") == "unconfigured" for t in targets
        ),
    }
    report["technical_pass"] = not any(
        report["summary"][key]
        for key in ("phase_failures", "technical_check_failures", "source_check_failures")
    )
    report["visual_approval"] = False
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    _gallery(report, out)
    _json(out / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("fixtures/open_posters/catalogue.json")
    )
    parser.add_argument("--out", type=Path, default=Path("output/poster-audit"))
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--gallery-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.manifest, args.out, args.case_ids, args.gallery_only)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return int(not report["technical_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
