"""Original synthetic intent counterfactuals for local infrastructure exercises.

All sources are constructed here from shapes and new short text. Generated weak
preferences test declared rules, not human taste or real-design generalization.
No external posters, reserve catalogue, models, cloud clients or engine changes.
Every labeled candidate passes the actual hard contract and is shadow-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from copy import deepcopy
from pathlib import Path

from .contracts import (
    canonical_json_sha256,
    load_collection,
    make_pair_id,
    sha256_file,
    validate_label,
)

CURRICULUM_VERSION = "synthetic-intent-v1"
FAMILIES = ("message-left", "message-right", "centered-stack")
AXES = ("headline_hierarchy", "reading_order", "action_attachment", "hero_prominence")
ISSUES = {
    "headline_hierarchy": ("weak_hierarchy", ["headline"]),
    "reading_order": ("broken_reading_order", ["headline", "description"]),
    "action_attachment": ("detached_action", ["action-label", "action-plate"]),
    "hero_prominence": ("weak_product_prominence", ["hero"]),
}
TARGETS = (
    {"width": 960, "height": 540},
    {"width": 540, "height": 960},
    {"width": 720, "height": 720},
)


def _shape(identity, x, y, width, height, color, group=None):
    result = {
        "id": identity,
        "type": "shape",
        "shape_type": "rectangle",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "fill_color": color,
    }
    if group:
        result["group_id"] = group
    return result


def _text(identity, content, x, y, width, height, size, group=None):
    result = {
        "id": identity,
        "type": "text",
        "content": content,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "font_family": "Solstice Sans",
        "font_size": size,
        "color": "#20372f",
    }
    if group:
        result["group_id"] = group
    return result


def fixture(family: str) -> tuple[dict, dict]:
    """Return a new source and its explicitly authored, non-gaze intent profile."""
    if family not in FAMILIES:
        raise ValueError("Unknown synthetic fixture family")
    index = FAMILIES.index(family)
    text_x = (40, 50, 56)[index]
    hero_x, hero_y = ((40, 300), (300, 306), (160, 294))[index]
    hero_width = (320, 300, 320)[index]
    support_x, support_y = ((420, 328), (40, 338), (70, 354))[index]
    support_width = (160, 180, 100)[index]
    action_x = (40, 360, 210)[index]
    design = {
        "canvas": {"width": 640, "height": 720},
        "elements": [
            _shape(
                "paper",
                0,
                0,
                640,
                720,
                ("rgba(242,239,222,1)", "rgba(234,241,246,1)", "rgba(248,233,223,1)")[index],
            ),
            _text("eyebrow", "ORIGINAL SYNTHETIC STUDY", text_x, 30, 550, 30, 15),
            _text(
                "headline",
                ("Shape and space", "A clear message", "A quiet rhythm")[index],
                text_x,
                78,
                550,
                84,
                42,
            ),
            _text("description", "A source-aware layout exercise.", text_x, 188, 550, 72, 23),
            _shape("support", support_x, support_y, support_width, 110, "rgba(202,173,111,1)"),
            _shape("hero", hero_x, hero_y, hero_width, 170, "rgba(103,141,115,1)"),
            _shape("action-plate", action_x, 550, 220, 64, "rgba(201,218,195,1)", "action"),
            _text("action-label", "Compare studies", action_x + 14, 568, 192, 32, 20, "action"),
            _text("footer", "Synthetic infrastructure only", 40, 666, 560, 28, 14),
        ],
    }
    profile = {
        "schema_version": 1,
        "family_id": family,
        "evidence": "authored synthetic intent",
        "attention_tiers": [
            ["headline", "hero"],
            ["description", "action-label"],
            ["eyebrow", "footer"],
        ],
        "reading_order_edges": [
            ["headline", "description"],
            ["description", "action-label"],
            ["action-label", "footer"],
        ],
        "groups": {"action": ["action-plate", "action-label"]},
        "required_element_ids": [element["id"] for element in design["elements"]],
        "omission_policy": "none",
        "addition_policy": "none",
        "framing_policy": "all source boxes in frame",
        "limits": (
            "Hero is an abstract rectangle; no claim of product semantics or measured human gaze."
        ),
    }
    return design, profile


def uniform_reference(source: dict, target: dict) -> dict:
    """Uniform foreground, canvas-filling paper; not the engine's selected winner."""
    result = deepcopy(source)
    width, height = source["canvas"]["width"], source["canvas"]["height"]
    scale = min(target["width"] / width, target["height"] / height)
    offset_x = (target["width"] - width * scale) / 2
    offset_y = (target["height"] - height * scale) / 2
    result["canvas"] = dict(target)
    for element in result["elements"]:
        element["x"] = element["x"] * scale + offset_x
        element["y"] = element["y"] * scale + offset_y
        for field in ("width", "height", "font_size", "corner_radius", "stroke_width"):
            if field in element:
                element[field] *= scale
        if element["id"] == "paper":
            # This existing authored background remains a single shape with its
            # original color, avoiding transparent letterbox bars in model input.
            element.update(x=0, y=0, width=target["width"], height=target["height"])
    return result


def counterfactual(reference: dict, axis: str) -> dict:
    if axis not in AXES:
        raise ValueError("Unknown curriculum axis")
    result = deepcopy(reference)
    elements = {element["id"]: element for element in result["elements"]}
    if axis == "headline_hierarchy":
        elements["headline"]["font_size"] = elements["description"]["font_size"] * 0.75
    elif axis == "reading_order":
        elements["headline"]["y"], elements["description"]["y"] = (
            elements["description"]["y"],
            elements["headline"]["y"],
        )
    elif axis == "action_attachment":
        label = elements["action-label"]
        center = label["x"] + label["width"] / 2
        # Move the label to the opposite side while retaining its y position.
        label["x"] = (
            0
            if center > result["canvas"]["width"] / 2
            else result["canvas"]["width"] - label["width"]
        )
    else:
        hero = elements["hero"]
        for position, size in (("x", "width"), ("y", "height")):
            hero[position] += hero[size] * 0.275
            hero[size] *= 0.45
    return result


def hard_probes(reference: dict) -> dict[str, tuple[dict, str]]:
    """Invalid policy tests stay outside every labeled comparison pool."""
    off_canvas, added, omitted = (deepcopy(reference) for _ in range(3))
    hero = next(element for element in off_canvas["elements"] if element["id"] == "hero")
    hero["x"] = off_canvas["canvas"]["width"] - hero["width"] / 2
    added["elements"].append({**deepcopy(added["elements"][-1]), "id": "unexpected-addition"})
    omitted["elements"] = [element for element in omitted["elements"] if element["id"] != "footer"]
    return {
        "off_canvas": (off_canvas, "off_canvas"),
        "addition": (added, "id_unexpected"),
        "omission": (omitted, "id_missing"),
    }


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save(renderer, design: dict, path: Path) -> None:
    fingerprint = canonical_json_sha256(design)
    supplied = deepcopy(design)
    image = renderer(supplied)
    if canonical_json_sha256(supplied) != fingerprint:
        raise ValueError("Curriculum renderer mutated its input")
    if image.size != (design["canvas"]["width"], design["canvas"]["height"]):
        raise ValueError("Curriculum render dimensions differ from design")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def create_curriculum(destination: str | Path, *, renderer=None) -> dict:
    """Publish twelve counterbalanced weak pairs and nine separate policy probes."""
    from engine.eval import build_context, check_contract
    from engine.resources import using_resources

    from .collect import code_provenance, validate_source_resources

    if renderer is None:
        from engine.eval.render import render_image

        renderer = render_image
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Curriculum destination must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (".curriculum-partial-" + uuid.uuid4().hex)
    stage.mkdir()
    root = stage / "collection"
    provenance = code_provenance()
    provenance["code_files"]["experiments/vision_critic/curriculum.py"] = sha256_file(__file__)
    provenance["working_tree_digest"] = canonical_json_sha256(provenance["code_files"])
    collection = {
        "schema_version": 1,
        "collection_id": "",
        "purpose": "synthetic",
        "provenance": {
            **provenance,
            "policy_version": CURRICULUM_VERSION,
            "pool": "synthetic-counterfactual-shadow",
            "baseline_kind": "synthetic_source_uniform_fit",
        },
        "cases": [],
    }
    tasks, labels, probes = [], [], []
    with using_resources() as resources:
        for family_index, (family, split) in enumerate(
            zip(FAMILIES, ("train", "validation", "test"), strict=True)
        ):
            source, profile = fixture(family)
            validate_source_resources(source, resources)
            context = build_context(source)
            if check_contract(source, source, source["canvas"], context):
                raise ValueError("Authored synthetic source is not contract-valid")
            source_prefix = f"sources/{family}"
            _write(root / source_prefix / "source.json", source)
            _write(root / source_prefix / "intent.json", profile)
            _save(renderer, source, root / source_prefix / "source.png")
            for axis_index, axis in enumerate(AXES):
                target = dict(TARGETS[axis_index % len(TARGETS)])
                reference = uniform_reference(source, target)
                changed = counterfactual(reference, axis)
                for design in (reference, changed):
                    if check_contract(source, design, target, context):
                        raise ValueError(
                            f"Counterfactual {family}/{axis} violates the hard contract"
                        )
                case_id = canonical_json_sha256(
                    {
                        "family": family,
                        "axis": axis,
                        "target": target,
                        "source": canonical_json_sha256(source),
                    }
                )
                prefix = f"cases/{case_id}"
                candidates = []
                for name, design in (("reference", reference), ("counterfactual", changed)):
                    _write(root / prefix / (name + ".json"), design)
                    _save(renderer, design, root / prefix / (name + ".png"))
                    candidates.append(
                        {
                            "candidate_id": canonical_json_sha256(design),
                            "design": f"{prefix}/{name}.json",
                            "design_sha256": canonical_json_sha256(design),
                            "render": f"{prefix}/{name}.png",
                            "render_sha256": sha256_file(root / prefix / (name + ".png")),
                            "admitted": False,
                            "diagnostics": {"contract_valid": True, "contract_violations": []},
                            "provenance": {
                                "element_ids": [element["id"] for element in design["elements"]],
                                "curriculum_axis": axis,
                                "generator_kind": name,
                                "synthetic_rule_only": True,
                            },
                        }
                    )
                collection["cases"].append(
                    {
                        "case_id": case_id,
                        "source_id": f"synthetic-intent-{family}",
                        "family_id": family,
                        "split": split,
                        "source_parity_verified": False,
                        "training_rights_reviewed": True,
                        "rights_reference": (
                            "Original synthetic shapes/text in curriculum.py; bundled font OFL "
                            "provenance in brand_kit/README.md; infrastructure only."
                        ),
                        "target_canvas": target,
                        "source": {
                            "design": f"{source_prefix}/source.json",
                            "json_sha256": canonical_json_sha256(source),
                            "render": f"{source_prefix}/source.png",
                            "render_sha256": sha256_file(root / source_prefix / "source.png"),
                        },
                        "baseline_candidate_id": candidates[0]["candidate_id"],
                        "admitted_candidate_ids": [],
                        "candidates": candidates,
                    }
                )
                # Counterbalance A/B position; the preferred teaching reference is not always A.
                a, b = (
                    candidates
                    if (family_index + axis_index) % 2 == 0
                    else list(reversed(candidates))
                )
                choice = "A" if a is candidates[0] else "B"
                bad_side = "B" if choice == "A" else "A"
                issue, element_ids = ISSUES[axis]
                labels.append(
                    {
                        "schema_version": 1,
                        "pair_id": "",
                        "collection_id": "",
                        "case_id": case_id,
                        "candidate_a_id": a["candidate_id"],
                        "candidate_b_id": b["candidate_id"],
                        "decision": choice,
                        "issues": [
                            {"candidate": bad_side, "code": issue, "element_ids": element_ids}
                        ],
                        "annotator": "synthetic-rule-v1",
                        "created_at": "2026-10-30T00:00:00Z",
                        "source_render_sha256": collection["cases"][-1]["source"]["render_sha256"],
                        "a_render_sha256": a["render_sha256"],
                        "b_render_sha256": b["render_sha256"],
                    }
                )
                tasks.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "family_id": family,
                        "split": split,
                        "intent_profile": f"collection/{source_prefix}/intent.json",
                        "weak_expectation": choice,
                        "supervision": "programmatic counterfactual rule; not human preference",
                        "assumption": (
                            "Uniform-fit reference retains the declared relationship better on "
                            "this one manipulated axis; not a global quality optimum."
                        ),
                    }
                )
            for name, (probe, expected) in hard_probes(
                uniform_reference(source, TARGETS[2])
            ).items():
                violations = check_contract(source, probe, TARGETS[2], context)
                codes = sorted({violation.code for violation in violations})
                if expected not in codes:
                    raise ValueError(f"Hard probe did not trigger expected {expected}")
                path = f"hard-probes/{family}/{name}.json"
                _write(stage / path, probe)
                probes.append(
                    {
                        "design": path,
                        "source": f"collection/{source_prefix}/source.json",
                        "expected_code": expected,
                        "actual_codes": codes,
                        "eligible_for_preferences": False,
                    }
                )
    collection["collection_id"] = canonical_json_sha256(
        {key: value for key, value in collection.items() if key != "collection_id"}
    )
    _write(root / "collection.json", collection)
    load_collection(root)
    for label in labels:
        label["collection_id"] = collection["collection_id"]
        label["pair_id"] = make_pair_id(
            label["collection_id"],
            label["case_id"],
            label["candidate_a_id"],
            label["candidate_b_id"],
        )
        validate_label(label, collection, allow_synthetic=True)
    with (stage / "weak-labels.jsonl").open("x", encoding="utf-8") as stream:
        for label in labels:
            stream.write(json.dumps(label, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema_version": 1,
        "curriculum_version": CURRICULUM_VERSION,
        "purpose": "synthetic_infrastructure_only",
        "human_reviewed_preferences": False,
        "real_quality_validated": False,
        "collection_id": collection["collection_id"],
        "tasks": tasks,
        "hard_probes": probes,
        "candidate_count": 24,
        "pair_count": len(labels),
        "baseline_kind": "synthetic_source_uniform_fit; not the current engine winner",
        "selection_policy": "all candidates shadow-only; production admission unchanged",
        "labels_sha256": sha256_file(stage / "weak-labels.jsonl"),
    }
    _write(stage / "curriculum.json", manifest)
    if destination.exists():
        raise ValueError("Curriculum destination appeared during generation")
    stage.rename(destination)
    return {
        "output": str(destination),
        "collection": str(destination / "collection"),
        "labels": str(destination / "weak-labels.jsonl"),
        "pair_count": len(labels),
        "purpose": "synthetic_infrastructure_only",
        "real_quality_validated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = create_curriculum(args.out)
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
