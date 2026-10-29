"""Audit prepared external posters and publish an offline evidence gallery.

python -m tools.poster_audit --manifest fixtures/open_posters/catalogue.json \
    --out output/poster-audit

Reserve examples are import-only: this module never resizes them. A technically
clean render still requires visual comparison with the native source; proxy
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


def _implementation_digest():
    root = Path(__file__).resolve().parents[1]
    files = [
        *sorted((root / "engine").rglob("*.py")),
        *sorted((root / "renderer/converter").rglob("*.py")),
    ]
    return _digest(
        {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
    )


def _resize(source, target):
    from engine.resize import resize_with_state

    return resize_with_state(source, target)


def _render(design):
    from engine.resources import get_renderer, get_resources, native_render_example, render_input
    from renderer.converter.convert import convert_chameleon_to_crello

    resources = get_resources()
    renderer = get_renderer()
    example, warnings = convert_chameleon_to_crello(
        render_input(design), renderer.font_manager, str(resources.assets_dir)
    )
    example = native_render_example(example, design)
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
    same_canvas = native.size == converted.size
    native_size, converted_size = list(native.size), list(converted.size)
    aspect_error = abs(native.width / native.height - converted.width / converted.height)
    if not same_canvas and aspect_error > 0.005:
        return {
            "same_canvas": False,
            "native_size": native_size,
            "converted_size": converted_size,
            "rgb_mae": None,
            "interpretation": "Reference and imported source have different aspect ratios.",
        }
    if not same_canvas:
        # Author thumbnails remain the independent reference. Compare at their
        # resolution; never describe an upsampled thumbnail as a full-size proof.
        converted = converted.resize(native.size, Image.Resampling.LANCZOS)

    def paper(image):
        image = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")

    difference = ImageChops.difference(paper(native), paper(converted))
    return {
        "same_canvas": same_canvas,
        "native_size": native_size,
        "converted_size": converted_size,
        "comparison_size": list(native.size),
        "rgb_mae": sum(ImageStat.Stat(difference).mean) / (3 * 255),
        "interpretation": "Pixel diagnostic at native reference resolution; inspect both sources.",
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
    from engine.layout.rules import analyze_rules

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
            "native_text_effects",
            "source_art",
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
    rules = analyze_rules(source, design)
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
        "dividers": {"passed": not rules["distorted_rules"], **rules},
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
            "notes": "Visual comparison with the native source is required.",
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
    original_name = "original" + original.suffix.lower()
    if (prepared / original_name).read_bytes() != original.read_bytes():
        raise ValueError("Prepared original differs from the curated original")
    shutil.copyfile(original, folder / original_name)
    row["original_file"] = original_name
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
        "source_note": entry.get("source_note"),
        "native_label": "Author's original app thumbnail"
        if entry.get("native_reference")
        else "Native SVG raster",
        "attribution": {
            key: entry.get(key)
            for key in (
                "author",
                "attribution_text",
                "source_url",
                "license_id",
                "license_url",
                "sha256",
            )
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
        row["implementation_digest"] = _implementation_digest()
        row["invariant_digest"] = _digest(entry.get("invariants", []))
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
                and saved.get("implementation_digest") == row["implementation_digest"]
                and saved.get("invariant_digest") == row["invariant_digest"]
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
                    target["visual_review"] = _review(
                        entry, target["name"], target["evidence_digest"]
                    )
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
                        "error": "No matching source, implementation and invariant evidence; "
                        "no engine was run.",
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
        ("native_file", row.get("native_label", "Native reference")),
        ("converted_file", "Imported source renderer"),
    ):
        if row.get(key):
            cells.append((f"{row['id']} · {label}", row["split"], folder / row[key]))
    for target in row["targets"]:
        status = (
            "CHECKS FAILED"
            if target.get("technical_pass") is False or target["failures"]
            else "checks passed"
        )
        status += " · visual " + target.get("visual_review", {}).get("status", "pending")
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
    for row in sorted(report["examples"], key=lambda row: row["split"] == "reserve"):
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
                f'<a href="{relative}"><img loading="lazy" src="{relative}" '
                f'alt="{alternative}"></a>'
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
        credit = html.escape(
            str(attribution.get("attribution_text") or attribution.get("author", ""))
        )
        source_note = f"<p>{html.escape(row['source_note'])}</p>" if row.get("source_note") else ""
        failures = row["failures"] + [
            failure for target in row["targets"] for failure in target["failures"]
        ]
        errors = "".join(
            f"<li>{html.escape(failure['phase'] + ': ' + failure['error'])}</li>"
            for failure in failures
        )
        label = "Personal test · import only" if row["split"] == "reserve" else "Render review"
        if row.get("status") == "import_failed":
            label = "Import needs attention"
        sections.append(
            f'<section id="{row["id"]}"><h2>{html.escape(row["title"])}</h2>'
            f'<p class="status">{label}</p>'
            f"<p>{credit} · {source} · "
            f'{license_link} · <a href="{row["id"]}/report.json">Full checks</a></p>'
            f"{source_note}"
            f'<p><a href="/?poster={row["id"]}">Open in studio</a> · '
            f'<a href="/api/posters/{row["id"]}/original">Editable original</a> · '
            f'<a href="/api/posters/{row["id"]}/bundle">Portable JSON bundle</a></p>'
            f'<ul class="errors">{errors}</ul><div class="grid">'
            f"{''.join(pictures)}</div></section>"
        )
    _contact_sheet(all_cells, out / "contact_sheet.png")
    summary = html.escape(json.dumps(report["summary"], indent=2))
    reviewed = sum(row["split"] == "reviewed" for row in report["examples"])
    reserves = sum(row["split"] == "reserve" for row in report["examples"])
    rendered = sum(
        bool(target.get("png_file")) for row in report["examples"] for target in row["targets"]
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chameleon · Poster tests</title><style>
body{{margin:0;background:#f6f6f6;color:#242424;font:14px/1.5 system-ui,sans-serif}}
header{{display:flex;align-items:center;gap:20px;padding:18px 28px;background:white;
border-bottom:1px solid #dedede;flex-wrap:wrap}}
header .brand{{margin-right:auto;color:#242424;font-size:18px;font-weight:650;text-decoration:none}}
.mark{{display:inline-grid;place-items:center;background:#eb1000;color:white;width:30px;
height:32px;border-radius:7px;margin-right:10px;font-size:15px}}
.button{{padding:9px 15px;background:#242424;color:white;border-radius:7px;text-decoration:none}}
main{{max-width:1720px;margin:auto;padding:32px 28px}}
h1{{font-size:30px;letter-spacing:-.7px;margin:0 0 10px}}h2{{font-size:21px;margin-bottom:4px}}
a{{color:#3d5e94;text-underline-offset:3px}}
a:focus-visible{{outline:3px solid #1473e6;outline-offset:4px}}
section{{margin:36px 0;border-top:1px solid #d8d8d8;padding-top:18px;scroll-margin-top:20px}}
.status{{font-weight:600;color:#686868;margin:4px 0 12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}
figure{{margin:0}}figure img,.missing{{width:100%;height:400px;object-fit:contain;
background:white;border:1px solid #ddd;border-radius:5px;box-sizing:border-box}}
figcaption{{padding:8px 0}}small{{color:#666}}.errors{{color:#9b342c}}
.missing{{display:grid;place-content:center;color:#9b342c}}
pre{{white-space:pre-wrap;background:#ededed;padding:16px;font-size:12px}}
.notice{{max-width:850px;color:#606060;margin:0 0 20px}}
.counts{{font-size:15px;font-weight:600}}details{{margin:12px 0;color:#606060}}
summary{{cursor:pointer}}
@media(max-width:600px){{header,main{{padding:20px}}figure img{{height:340px}}}}
</style><header><a class="brand" href="/"><span class="mark">Ch</span>Chameleon</a>
<a href="/">Open studio</a><a class="button" href="reserve-10.zip">Personal-test pack ↓</a>
</header><main><h1>Poster comparisons</h1><p class="notice">
Original artwork, editable imports, and selected adaptations. Open any image at full size.
Render checks and visual reviews are reported separately. Personal tests are import-only;
this audit has never resized them.</p>
<p class="counts">{reviewed} reviewed designs &nbsp; · &nbsp; {rendered} adaptations
&nbsp; · &nbsp; {reserves} personal-test inputs</p>
<details><summary>Verification details and downloads</summary><p>
<a href="report.json">Full checks</a> · <a href="contact_sheet.png">Contact sheet</a></p>
<pre>{summary}</pre></details>
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
