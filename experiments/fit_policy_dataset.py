"""Build development-only synthetic geometry supervision; never read reserve pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = {
    "core_01",
    "doory_grand-opening-flyer",
    "doory_rooftop-event-story",
    "doory_night-bloom-concert-poster",
    "opensuse_booth_poster",
    "opensuse_trifold_front",
    "synthetic_materials_brief",
    "synthetic_guild_bulletin",
    "synthetic_garden_plan",
}
TARGETS = ({"width": 800, "height": 500}, {"width": 900, "height": 1600})
COLUMN_TARGETS = ({"width": 1080, "height": 1080}, {"width": 1600, "height": 900})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def development_sources():
    """The catalogue is an allowlist; reserve source JSON/assets are never opened."""
    catalogue = read(ROOT / "fixtures/open_posters/catalogue.json")
    entries = catalogue["examples"]
    reserves = {entry["id"] for entry in entries if entry["split"] == "reserve"}
    if len(reserves) != 10 or "doory_modern-resume" not in reserves:
        raise ValueError("Expected ten excluded personal reserves, including Modern Resume")
    sources = [
        {
            "source_id": "core_01",
            "source_path": ROOT / "input/chameleon_sample_level2_core.json",
            "assets_dir": ROOT / "brand_kit/assets",
            "fonts_dir": ROOT / "brand_kit/fonts",
        }
    ]
    for entry in entries:
        if entry["split"] != "reviewed":
            continue
        folder = ROOT / "fixtures/open_posters" / entry["prepared_path"]
        sources.append(
            {
                "source_id": entry["id"],
                "source_path": folder / "source.json",
                "assets_dir": folder / "resources/assets",
                "fonts_dir": folder / "resources/fonts",
            }
        )
    if len(sources) != 9 or any(s["source_id"] in reserves for s in sources):
        raise ValueError("Fit training requires Core and exactly eight reviewed works")
    return sources, sorted(reserves)


def synthetic_column_sources(out):
    """Original fictional documents, independent of all catalogue/resume content."""
    definitions = (
        (
            "materials_brief",
            "Materials Field Report",
            794,
            1123,
            0.46,
            13,
            (
                "Purpose",
                "Methods",
                "Findings",
                "Next study",
                "Samples",
                "Observations",
                "Questions",
                "Review",
            ),
            "The workshop compares small material trials and records "
            "the conditions behind each observation.",
        ),
        (
            "guild_bulletin",
            "Harbor Guild Bulletin",
            816,
            1056,
            0.54,
            14,
            (
                "This month",
                "Workshops",
                "Community",
                "Calendar",
                "Members",
                "Projects",
                "Resources",
                "Contact desk",
            ),
            "Local makers share practical notes, arrange open sessions, "
            "and keep a clear record of upcoming activities.",
        ),
        (
            "garden_plan",
            "Civic Garden Plan",
            900,
            1400,
            0.48,
            16,
            (
                "Purpose",
                "Phases",
                "Care plan",
                "Budget",
                "People",
                "Work days",
                "Supplies",
                "Tracking",
            ),
            "A small neighborhood team coordinates planting, shared tools, "
            "and regular care for an experimental garden.",
        ),
        (
            "operations_digest",
            "Orchard Operations Digest",
            760,
            1180,
            0.44,
            12,
            (
                "Harvest",
                "Storage",
                "Packing",
                "Delivery",
                "Resources",
                "Training",
                "Checks",
                "Review",
            ),
            "The orchard team summarizes seasonal work so that each handoff "
            "remains clear, traceable, and easy to review.",
        ),
    )
    directory = Path(out) / "synthetic-sources"
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for family, title, width, height, division, size, headings, introduction in definitions:
        margin, top = width * 0.06, height * 0.235
        cut, bottom = width * division, height - width * 0.06
        step = (bottom - top) / 4

        def text(identity, content, x, y, frame_width, frame_height, font_size):
            return {
                "id": identity,
                "type": "text",
                "content": content,
                "x": x,
                "y": y,
                "width": frame_width,
                "height": frame_height,
                "font_family": "Solstice Sans",
                "font_size": font_size,
                "color": "#203947",
                "text_align": "left",
                "z_index": 2,
            }

        elements = [
            {
                "id": "paper",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 0,
                "y": 0,
                "width": width,
                "height": height,
                "fill_color": "#fafaf8",
                "z_index": 0,
            },
            text(
                "document-title",
                title,
                margin,
                height * 0.045,
                width - 2 * margin,
                size * 4,
                size * 2.5,
            ),
            text(
                "document-summary",
                introduction,
                margin,
                height * 0.125,
                width - 2 * margin,
                size * 4,
                size,
            ),
            {
                "id": "heading-rule",
                "type": "shape",
                "shape_type": "rectangle",
                "x": margin,
                "y": height * 0.19,
                "width": width - 2 * margin,
                "height": 2,
                "fill_color": "#889aa3",
                "z_index": 1,
            },
            {
                "id": "column-rule",
                "type": "shape",
                "shape_type": "rectangle",
                "x": cut,
                "y": top,
                "width": width * 0.002,
                "height": bottom - top,
                "fill_color": "#889aa3",
                "z_index": 1,
            },
        ]
        for index, heading in enumerate(headings):
            side, row = divmod(index, 4)
            x = margin if side == 0 else cut + margin * 0.55
            available = cut - margin * 0.55 - x if side == 0 else width - margin - x
            y = top + row * step
            body = (
                f"{heading} records one practical part of the work. "
                "The team notes the current condition, assigns a clear next action, "
                "and checks progress at the following review."
            )
            elements.extend(
                [
                    text(
                        f"section-{index}-label", heading, x, y, available, size * 2.6, size * 1.4
                    ),
                    text(
                        f"section-{index}-body",
                        body,
                        x,
                        y + size * 2.5,
                        available,
                        step - size * 3.0,
                        size,
                    ),
                ]
            )
        source = {"canvas": {"width": width, "height": height}, "elements": elements}
        path = directory / (family + ".json")
        path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        entries.append(
            {
                "source_id": "synthetic_" + family,
                "source_kind": "synthetic_column",
                "source_path": path,
                "assets_dir": ROOT / "brand_kit/assets",
                "fonts_dir": ROOT / "brand_kit/fonts",
            }
        )
    return entries


def synthetic_teacher(features):
    """A disclosed fit proxy, not human preference or a visual/aesthetic judgment."""
    occupancy = 0.7 * math.sqrt(min(1, max(0, features["foreground_bbox_area"])))
    occupancy += 0.3 * min(1, max(0, features["foreground_area"]))
    readability = 0.5 * min(1, max(0, features["minimum_font"]) / 1.8)
    readability += 0.5 * min(1, max(0, features["median_font"]) / 2.8)
    group_style = (
        sum(
            features[key]
            for key in (
                "font_scale_consistency",
                "hierarchy_retention",
                "unit_scale_consistency",
                "unit_attachment",
            )
        )
        / 4
    )
    terms = {
        "occupied_area": occupancy,
        "readability": readability,
        "group_style": group_style,
        "reading_order": features["reading_order_retention"],
        "image_proportions": features["image_proportion_retention"],
        "text_crowding": max(0, features["text_frame_fill"] - 1.3),
        "overflow": features["overflow_fraction"],
    }
    score = (
        3.0 * terms["occupied_area"]
        + 1.8 * terms["readability"]
        + terms["group_style"]
        + 0.8 * terms["reading_order"]
        + 0.7 * terms["image_proportions"]
        - 0.7 * terms["text_crowding"]
        - 1.2 * terms["overflow"]
    )
    if not math.isfinite(score):
        raise ValueError("Synthetic teacher score must be finite")
    return score, terms


def generate_dataset(out, *, targets=TARGETS, max_candidates=10, base_dataset=None):
    from engine.eval import check_contract
    from engine.fit_policy import (
        FEATURE_VERSION,
        LABEL_KIND,
        digest,
        extract_features,
        source_digest,
    )
    from engine.layout.candidates import generate_candidates
    from engine.layout.fitted_columns import generate_fitted_column_states
    from engine.layout.grouped import GroupedState, generate_grouped_states
    from engine.layout.relationships import build_relationship_graph
    from engine.resources import using_resources
    from engine.stage_one import contain_resolution_decorations, strictly_in_bounds

    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / "output"):
        raise ValueError("Generated training evidence belongs under project output/")
    out.mkdir(parents=True, exist_ok=True)
    sources, reserves = development_sources()
    authored = {entry["source_id"]: entry for entry in sources}
    synthetic = synthetic_column_sources(out)
    records, groups, skipped = [], [], []
    if base_dataset is not None:
        base = read(base_dataset)
        if (
            base.get("schema_version") != 1
            or base.get("label_kind") != LABEL_KIND
            or base.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError("Reused authored data must have the same teacher and feature schema")
        source_cache, grouped = {}, Counter()
        for original_row in base["records"]:
            row = dict(original_row)
            entry = authored.get(row.get("source_id"))
            if entry is None or row.get("reserve") or row.get("hard_valid") is not True:
                raise ValueError(
                    "Only previously hard-valid allowlisted authored records may be reused"
                )
            identity = row["source_id"]
            if identity not in source_cache:
                payload = read(entry["source_path"])
                source_cache[identity] = payload.get("source", payload)
            source = source_cache[identity]
            design = read(row["design_path"])
            if (
                source_digest(source) != row["source_digest"]
                or digest(design) != row["design_digest"]
                or row["split"] != ("train" if identity in TRAIN else "validation")
                or extract_features(source, row["target_canvas"], design, units=row["units"])
                != row["features"]
            ):
                raise ValueError("Reused authored geometry or features changed")
            score, terms = synthetic_teacher(row["features"])
            if score != row["teacher_score"] or terms != row["teacher_terms"]:
                raise ValueError("Reused authored teacher definition changed")
            row["source_kind"] = "authored_development"
            records.append(row)
            target = row["target_canvas"]
            grouped[f"{identity}-{target['width']}x{target['height']}"] += 1
        groups.extend(
            {"group": name, "candidates": count, "reused_verified_authored_data": True}
            for name, count in grouped.items()
        )
        sources = synthetic
    else:
        sources.extend(synthetic)
    code_paths = [
        Path(__file__),
        ROOT / "engine/fit_policy.py",
        ROOT / "engine/layout/candidates.py",
        ROOT / "engine/layout/grouped.py",
        ROOT / "engine/layout/fitted_columns.py",
        ROOT / "engine/layout/relationships.py",
        ROOT / "engine/stage_one.py",
    ]
    before = {str(path.relative_to(ROOT)): file_sha(path) for path in code_paths}
    for entry in sources:
        identity = entry["source_id"]
        payload = read(entry["source_path"])
        source = payload.get("source", payload)
        original_digest = source_digest(source)
        sw, sh = (source["canvas"][key] for key in ("width", "height"))
        with using_resources(entry["assets_dir"], entry["fonts_dir"]):
            graph = build_relationship_graph(source)
            units = [
                {
                    "unit_id": f"u{unit.index}",
                    "element_ids": list(unit.element_ids),
                    "role": unit.role,
                    "box": [
                        unit.box.x / sw,
                        unit.box.y / sh,
                        unit.box.width / sw,
                        unit.box.height / sh,
                    ],
                }
                for unit in graph.units
            ]
            for target in (
                COLUMN_TARGETS if entry.get("source_kind") == "synthetic_column" else targets
            ):
                label = f"{identity}-{target['width']}x{target['height']}"
                if math.isclose(sw / sh, target["width"] / target["height"], rel_tol=1e-9):
                    skipped.append(
                        {"group": label, "reason": "strict_resolution_not_policy_training"}
                    )
                    continue
                if (
                    entry.get("source_kind") == "synthetic_column"
                    and (target["width"] / target["height"]) / (sw / sh) < 1.35
                ):
                    skipped.append({"group": label, "reason": "outside_fitted_widening_contract"})
                    continue
                fitted = generate_fitted_column_states(source, target, graph)
                if entry.get("source_kind") == "synthetic_column" and not fitted:
                    raise RuntimeError(f"Synthetic source did not exercise fitted columns: {label}")
                offered = [(state.name, state.design) for state in fitted]
                offered.extend(
                    (f"deterministic/{i}", design)
                    for i, design in enumerate(
                        generate_candidates(source, target)[
                            : 2 if entry.get("source_kind") == "synthetic_column" else None
                        ]
                    )
                )
                if entry.get("source_kind") != "synthetic_column":
                    offered.extend(
                        (state.name, state.design)
                        for state in generate_grouped_states(source, target, graph)[:4]
                    )
                candidates, rejected = {}, Counter()
                for name, design in offered:
                    state = GroupedState(name, "dataset-real-generator", (), json.dumps(design))
                    state = contain_resolution_decorations(source, target, state)
                    if state is None:
                        rejected["strict_frame"] += 1
                        continue
                    design = state.design
                    if check_contract(source, design, target) or not strictly_in_bounds(design):
                        rejected["hard_contract"] += 1
                        continue
                    design_hash = digest(design)
                    if design_hash in candidates:
                        continue
                    features = extract_features(source, target, design, units=units)
                    score, terms = synthetic_teacher(features)
                    candidates[design_hash] = {
                        "design": design,
                        "generator_state": name,
                        "plan_id": "candidate-" + design_hash,
                        "features": features,
                        "teacher_score": score,
                        "teacher_terms": terms,
                    }
                values = sorted(candidates.items(), key=lambda pair: pair[1]["teacher_score"])
                if (
                    len(values) < 2
                    or values[-1][1]["teacher_score"] - values[0][1]["teacher_score"] < 0.01
                ):
                    skipped.append(
                        {
                            "group": label,
                            "reason": "insufficient_non_tied_hard_valid_states",
                            "valid": len(values),
                        }
                    )
                    continue
                # Preserve both undersized and fitted actual proposals, not only winners.
                if len(values) > max_candidates:
                    values = (
                        values[: max_candidates // 2]
                        + values[-(max_candidates - max_candidates // 2) :]
                    )
                menu_digest = digest(
                    {
                        "source_digest": original_digest,
                        "target": target,
                        "units": units,
                        "plans": [v[1]["plan_id"] for v in values],
                    }
                )
                directory = out / "candidates" / label
                directory.mkdir(parents=True, exist_ok=True)
                for design_hash, candidate in values:
                    path = directory / (design_hash[:20] + ".json")
                    path.write_text(
                        json.dumps(candidate.pop("design"), indent=2) + "\n", encoding="utf-8"
                    )
                    records.append(
                        {
                            "source_id": identity,
                            "source_kind": entry.get("source_kind", "authored_development"),
                            "family": identity,
                            "split": "train" if identity in TRAIN else "validation",
                            "source_digest": original_digest,
                            "source_path": str(entry["source_path"]),
                            "assets_dir": str(entry["assets_dir"]),
                            "fonts_dir": str(entry["fonts_dir"]),
                            "target_canvas": target,
                            "menu_digest": menu_digest,
                            "units": units,
                            "design_digest": design_hash,
                            "design_path": str(path),
                            "hard_valid": True,
                            "reserve": False,
                            **candidate,
                        }
                    )
                groups.append(
                    {"group": label, "candidates": len(values), "rejected": dict(rejected)}
                )
                print(json.dumps(groups[-1]), flush=True)
        if source_digest(source) != original_digest:
            raise RuntimeError("A generator mutated the original source")
    after = {str(path.relative_to(ROOT)): file_sha(path) for path in code_paths}
    dataset = {
        "schema_version": 1,
        "label_kind": LABEL_KIND,
        "feature_version": FEATURE_VERSION,
        "records": records,
    }
    dataset_path = out / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    summary = {
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha(dataset_path),
        "label_kind": LABEL_KIND,
        "verified_base_dataset": str(base_dataset) if base_dataset else None,
        "verified_base_sha256": file_sha(base_dataset) if base_dataset else None,
        "records": len(records),
        "groups": groups,
        "skipped": skipped,
        "families": {
            split: sorted({r["family"] for r in records if r["split"] == split})
            for split in ("train", "validation")
        },
        "excluded_reserve_ids": reserves,
        "reserve_pixels_read": False,
        "implementation_before": before,
        "implementation_after": after,
        "implementation_unchanged": before == after,
        "human_preference_labels": False,
        "visual_quality_validated": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/fit-policy-training")
    parser.add_argument("--base-dataset", type=Path)
    args = parser.parse_args()
    summary = generate_dataset(args.out, base_dataset=args.base_dataset)
    print(
        json.dumps(
            {k: summary[k] for k in ("dataset", "records", "families", "implementation_unchanged")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
