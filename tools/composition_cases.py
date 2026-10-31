"""Deterministic real-asset fixtures and rendered audits for compound campaigns.

The authored expectations are test metadata, never a production inference hint.
Every image uses the supplied brand kit; no supplied source or asset is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from engine.eval import build_context, check_contract, score
from engine.eval.render import render_image
from engine.eval.style import analyze_style
from engine.layout.appearance import unit_visible_box
from engine.layout.relationships import analyze_relationships, build_relationship_graph
from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello
from tools.composition_demo import ASSETS, FONTS, ROOT, _contact_sheet, _digest, _write_json

TARGETS = (
    {"width": 728, "height": 90},
    {"width": 1600, "height": 400},
    {"width": 913, "height": 517},
    {"width": 1080, "height": 1920},
)


def _make_case(layout: str) -> dict:
    """Author one independent source without using production layout functions."""
    elements, expected = [], []
    counter = 0

    def add(kind, x, y, width, height, **fields):
        nonlocal counter
        counter += 1
        element = {
            "id": f"n{counter * 7919 % 104729:06d}",
            "type": kind,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            **fields,
        }
        elements.append(element)
        return element["id"]

    def text(content, x, y, width, height, size, **fields):
        return add(
            "text",
            x,
            y,
            width,
            height,
            content=content,
            font_family="Solstice Sans",
            font_size=size,
            color="rgba(255,255,255,1)",
            text_align="center",
            z_index=5,
            **fields,
        )

    def control(content, x, y, width, height, size, color):
        group = f"q{(counter + 1) * 12347 % 99991:05d}"
        plate = add(
            "shape",
            x,
            y,
            width,
            height,
            z_index=3,
            group_id=group,
            shape_type="rectangle",
            fill_color=color,
            corner_radius=height / 2,
        )
        label_height = size * 1.2
        label = text(
            content,
            x,
            y + (height - label_height) / 2,
            width,
            label_height,
            size,
            font_weight="bold",
            group_id=group,
        )
        return [plate, label]

    canvas = (
        {"width": 1600, "height": 1100} if layout == "columns" else {"width": 1000, "height": 2200}
    )
    add(
        "image",
        0,
        0,
        canvas["width"],
        canvas["height"],
        z_index=0,
        asset_id="solstice_bg_gradient.png",
        fit="cover",
    )
    for index in range(2):
        center = 400 + index * 800 if layout == "columns" else 500
        top = 0 if layout == "columns" else index * 1100
        logo = add(
            "image",
            center - 125,
            top + 40,
            250,
            82,
            z_index=4,
            asset_id="solstice_logo_lockup.png",
            fit="contain",
        )
        product = add(
            "image",
            center - 180,
            top + 140,
            360,
            360,
            z_index=3,
            asset_id="solstice_bottle.png",
            fit="contain",
        )
        headline = text(
            ("A bright new ritual", "Make every gathering a little brighter")[index],
            center - 330,
            top + 540,
            660,
            110,
            40,
            font_weight="bold",
        )
        # Use the supplied serif face for both authored headlines.
        elements[-1]["font_family"] = "Solstice Serif"
        body = text(
            (
                "Bright botanicals for a quiet moment.",
                "A slower infusion of citrus peel and garden herbs, crafted for long "
                "evenings and generous tables. Open a bottle, pour it slowly, and share "
                "the last light with friends.",
            )[index],
            center - 320,
            top + 665,
            640,
            135,
            22,
        )
        badges = [
            control(
                ("BOTANICAL BLEND", "SLOW-CRAFTED")[index],
                center - 249,
                top + 830,
                230,
                46,
                14,
                "rgba(232,106,58,1)",
            ),
            control(
                ("SMALL BATCH", "FOR SHARING")[index],
                center - 1,
                top + 830,
                250,
                46,
                14,
                "rgba(43,27,46,1)",
            ),
        ]
        action = control(
            ("Discover the blend", "Bring everyone together")[index],
            center - 180,
            top + 925,
            360,
            72,
            20,
            "rgba(232,106,58,1)",
        )
        expected.append(
            {
                "id": f"offer_{index}",
                "logo_ids": [logo],
                "product_ids": [product],
                "headline_ids": [headline],
                "body_ids": [body],
                "action_ids": action,
                "badge_groups": badges,
                "element_ids": [logo, product, headline, body, *sum(badges, []), *action],
            }
        )
    return {
        "task_id": f"authored_pair_{layout}",
        "source_layout": layout,
        "source": {"canvas": canvas, "elements": elements},
        "target_canvas": {"width": 1600, "height": 400},
        "targets": deepcopy(list(TARGETS)),
        "expected_campaigns": expected,
    }


def generate_cases() -> list[dict]:
    """Return independent real-asset column and vertical two-campaign tasks.

    ``expected_campaigns`` is authored test metadata, never an inference hint.
    Only atomic controls have group_id. Repeated bottle instances test spatial
    ownership, not recognition of different product classes.
    """
    return [_make_case(layout) for layout in ("columns", "stacked")]


def advanced_case() -> dict:
    """Attach authored audit expectations without modifying the supplied task."""
    task = json.loads((ROOT / "input/chameleon_sample_level3_advanced.json").read_text())
    task["targets"] = deepcopy(list(TARGETS))
    task["expected_campaigns"] = [
        {
            "id": "advanced_offer",
            "logo_ids": ["logo"],
            "product_ids": ["product"],
            "headline_ids": ["headline"],
            "body_ids": ["subhead"],
            "action_ids": ["cta_bg", "cta_label"],
            "badge_groups": [[f"badge{i}_bg", f"badge{i}_label"] for i in (1, 2, 3)],
        }
    ]
    return task


def authored_diagnostics(task: dict, design: dict) -> dict:
    """Report authored pair/order evidence independently of inferred graph scores.

    This is not universal semantic recognition or an aesthetic score. Reading
    order and nearest-neighbor ties are recorded separately; ties need review.
    Frame positions and local centers remain inspectable in the report.
    """
    by_id = {element["id"]: element for element in design["elements"]}

    def box(ids):
        return unit_visible_box([by_id[eid] for eid in ids], ASSETS)

    def center(value):
        return value.x + value.width / 2, value.y + value.height / 2

    def gap(a, b):
        return math.hypot(
            max(0, a.x - b.x - b.width, b.x - a.x - a.width),
            max(0, a.y - b.y - b.height, b.y - a.y - a.height),
        )

    expected = task["expected_campaigns"]
    headlines = [box(campaign["headline_ids"]) for campaign in expected]
    products = [box(campaign["product_ids"]) for campaign in expected]
    rows = []
    for index, campaign in enumerate(expected):
        logo, product = box(campaign["logo_ids"]), products[index]
        headline, body = headlines[index], box(campaign["body_ids"])
        action = box(campaign["action_ids"])
        badges = [box(group) for group in campaign["badge_groups"]]
        peer_row = box([eid for group in campaign["badge_groups"] for eid in group])
        ordered = [headline, body, peer_row, action]
        inversions = [
            label
            for label, first, second in zip(
                ("headline_before_body", "body_before_badges", "badges_before_action"),
                ordered,
                ordered[1:],
                strict=False,
            )
            if second.y < first.y - 0.5
        ]

        def nearest(first, candidates):
            distances = [gap(first, item) for item in candidates]
            minimum = min(distances)
            return [i for i, distance in enumerate(distances) if distance <= minimum + 0.5]

        nearest_headlines = nearest(product, headlines)
        nearest_products = nearest(headline, products)
        nearest_action_headlines = nearest(action, headlines)
        centers = [center(item)[0] for item in ordered]
        rows.append(
            {
                "campaign": campaign["id"],
                "reading_order_failures": inversions,
                "message_axis_spread_px": round(max(centers) - min(centers), 4),
                "message_axis_spread_fraction": round(
                    (max(centers) - min(centers)) / design["canvas"]["width"], 6
                ),
                "brand_product_axis_spread_px": round(abs(center(logo)[0] - center(product)[0]), 4),
                "product_nearest_headlines": nearest_headlines,
                "headline_nearest_products": nearest_products,
                "action_nearest_headlines": nearest_action_headlines,
                "own_product_message_pair_present": index in nearest_headlines
                and index in nearest_products,
                "own_action_message_pair_present": index in nearest_action_headlines,
                "ambiguous_nearest_pair": any(
                    len(values) > 1
                    for values in (nearest_headlines, nearest_products, nearest_action_headlines)
                ),
                "product_headline_gap_px": round(gap(product, headline), 4),
                "message_boxes": [asdict(item) for item in ordered],
                "badge_control_boxes": [asdict(item) for item in badges],
            }
        )
    return {
        "campaigns": rows,
        "reading_order_failures": sum(len(row["reading_order_failures"]) for row in rows),
        "detached_pairs": sum(
            not row["own_product_message_pair_present"]
            or not row["own_action_message_pair_present"]
            for row in rows
        ),
        "ambiguous_pairs": sum(row["ambiguous_nearest_pair"] for row in rows),
        "limitations": (
            "Authored fixture expectations and visible geometry; inspect actual renders."
        ),
    }


def validate_source(task: dict) -> None:
    """Validate authored sources using the supplied schema and hard contract."""
    from jsonschema import Draft202012Validator

    schema = json.loads((ROOT / "chameleon_design_schema_v0.9.json").read_text())
    source = task["source"]
    Draft202012Validator(schema).validate(source)
    context = build_context(source, ASSETS, FONTS)
    violations = check_contract(source, source, source["canvas"], context)
    if violations:
        raise ValueError("Invalid authored source: " + "; ".join(str(item) for item in violations))


def _render(design: dict):
    """Keep converter warnings alongside the actual full-resolution pixels."""
    _, warnings = convert_chameleon_to_crello(
        design, _get_renderer(str(FONTS)).font_manager, str(ASSETS)
    )
    picture = render_image(design, ASSETS, FONTS, max_size=max(design["canvas"].values()))
    if picture.size != (design["canvas"]["width"], design["canvas"]["height"]):
        raise ValueError("Output render is not at full target resolution")
    return picture, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/advanced-audit"))
    parser.add_argument(
        "--baseline-dir", type=Path, help="Read-only pre-change case/size artifacts"
    )
    parser.add_argument("--cases", choices=("all", "advanced", "authored"), default="all")
    args = parser.parse_args(argv)
    if args.baseline_dir and args.baseline_dir.resolve() == args.out.resolve():
        parser.error("--baseline-dir must differ from --out")
    # Import at execution time: fixture consumers do not require a selected-state API.
    from engine.resize import resize_with_state

    tasks = ([advanced_case()] if args.cases != "authored" else []) + (
        generate_cases() if args.cases != "advanced" else []
    )
    started = time.perf_counter()
    report = {"cases": [], "errors": [], "scope": "rendered audit, not semantic/aesthetic proof"}
    args.out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        case_id, source = task["task_id"], task["source"]
        before = deepcopy(source)
        case_out = args.out / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        cells = []
        source_digest = _digest(source)
        row = {"task": case_id, "source_digest": source_digest, "targets": []}
        report["cases"].append(row)
        try:
            validate_source(task)
            _write_json(case_out / "task.json", task)
            _write_json(case_out / "source.json", source)
            graph = build_relationship_graph(source, ASSETS)
            graph_digest = _digest(asdict(graph))
            _write_json(
                case_out / "graph.json", {"source_digest": source_digest, "graph": asdict(graph)}
            )
            context = build_context(source, ASSETS, FONTS)
            source_image, row["source_render_warnings"] = _render(source)
            source_image.save(case_out / "source.png")
            cells.append((case_id, "AUTHORED SOURCE / SUPPLIED ADVANCED", source_image))
            row["source_authored"] = authored_diagnostics(task, source)
            for target in task["targets"]:
                stem = f"{target['width']}x{target['height']}"
                target_row = {"target": target}
                row["targets"].append(target_row)
                for suffix in (".json", ".png"):
                    (case_out / f"{stem}{suffix}").unlink(missing_ok=True)
                try:
                    if args.baseline_dir:
                        baseline_case = args.baseline_dir / case_id
                        baseline_task = json.loads((baseline_case / "task.json").read_text())
                        if _digest(baseline_task["source"]) != source_digest:
                            raise ValueError(
                                "Baseline source differs from the current authored source"
                            )
                        baseline = json.loads((baseline_case / f"{stem}.json").read_text())
                        if baseline["canvas"] != target:
                            raise ValueError("Baseline target does not match")
                        baseline_image, target_row["baseline_render_warnings"] = _render(baseline)
                        cells.append(
                            (f"BEFORE {stem}", "Frozen pre-change selected output", baseline_image)
                        )
                        target_row["baseline_authored"] = authored_diagnostics(task, baseline)
                        target_row["baseline_file"] = str(
                            (baseline_case / f"{stem}.json").resolve()
                        )
                    selected = resize_with_state(source, target)
                    design, state = selected["design"], selected["state"]
                    result = score(source, design, context, target_canvas=target)
                    target_row.update(
                        state=state,
                        violations=[
                            {"code": item.code, "message": str(item)}
                            for item in result["violations"]
                        ],
                        metrics=result["metrics"],
                        style=analyze_style(source, design, ASSETS, relationship_graph=graph),
                        relationships=analyze_relationships(graph, design, state["cluster_axes"]),
                        authored=authored_diagnostics(task, design),
                    )
                    _write_json(case_out / f"{stem}.json", design)
                    picture, target_row["render_warnings"] = _render(design)
                    picture.save(case_out / f"{stem}.png")
                    cells.append((f"CURRENT {stem}", state["name"], picture))
                    if result["violations"]:
                        report["errors"].append(
                            {"task": case_id, "target": target, "phase": "contract"}
                        )
                    print(
                        f"{case_id} {stem}: min_font={result['metrics'].get('raw_min_font_px')} "
                        f"order_failures={target_row['authored']['reading_order_failures']} "
                        f"detached_pairs={target_row['authored']['detached_pairs']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - preserve explicit per-case failures
                    (case_out / f"{stem}.png").unlink(missing_ok=True)
                    target_row["error"] = f"{type(exc).__name__}: {exc}"
                    report["errors"].append(
                        {"task": case_id, "target": target, "error": target_row["error"]}
                    )
                    cells.append((f"FAILED {stem}", "See report.json", None))
            row["source_immutable"] = source == before and _digest(source) == source_digest
            row["graph_immutable"] = _digest(asdict(graph)) == graph_digest
            if not row["source_immutable"] or not row["graph_immutable"]:
                raise ValueError("Source or graph was mutated during audit")
        except Exception as exc:  # noqa: BLE001 - do not lose the report after a source failure
            row["error"] = f"{type(exc).__name__}: {exc}"
            report["errors"].append({"task": case_id, "error": row["error"]})
        (case_out / "contact_sheet.png").unlink(missing_ok=True)
        if cells:
            _contact_sheet(cells).save(case_out / "contact_sheet.png")
        _write_json(case_out / "report.json", row)
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    report["success"] = not report["errors"]
    _write_json(args.out / "report.json", report)
    print(
        f"{len(report['cases'])} sources; {len(report['errors'])} failures. "
        f"{args.out.resolve() / 'report.json'}"
    )
    return int(not report["success"])


if __name__ == "__main__":
    raise SystemExit(main())
