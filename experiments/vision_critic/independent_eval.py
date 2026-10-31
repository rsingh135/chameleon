"""Score frozen visual probes and supplied transcripts without running a model.

Output validity, execution evidence, control sensitivity and labeled agreement
are separate diagnostics. A fallback never earns a model recommendation match.
Test labels are opened only after an explicit bundle-bound recipe freeze.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import (
    canonical_json_sha256,
    parse_json,
    safe_artifact_path,
    sha256_file,
)
from .train_sft import parse_model_verdict

PROTOCOL = "chameleon-independent-eval-v1"
CONDITIONS = {"original", "swapped", "blank_source", "replaced_source", "opaque_ids"}
AGREEMENT_CONDITIONS = {"original", "swapped"}
AXES = {"hierarchy", "attachment", "reading_order", "prominence"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_TRANSCRIPTS = 16384


def _json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Evaluation JSON exceeds its byte budget")
    return parse_json(path.read_text(encoding="utf-8-sig"))


def _hash(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4096


def _mean(values):
    return sum(values) / len(values) if values else None


def _image_ref(value, root, files):
    if not isinstance(value, dict) or not _hash(value.get("sha256")):
        raise ValueError("Probe images require bound SHA-256 identities")
    path = Path(value.get("path", ""))
    path = path if path.is_absolute() else root / path
    try:
        relative = path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as exc:
        raise ValueError("Probe image leaves the frozen bundle") from exc
    safe_artifact_path(root, relative)
    if files.get(relative) != value["sha256"]:
        raise ValueError("Probe image is absent or differs from the frozen file inventory")
    return value["sha256"]


def _request(probe, root, files):
    from experiments.visual_planner.provider import digest, summary_element_ids

    if probe.get("request_path") not in files:
        raise ValueError("Probe request is absent from the frozen file inventory")
    request = _json(safe_artifact_path(root, probe["request_path"]))
    if not isinstance(request, dict) or digest(request) != probe.get("request_digest"):
        raise ValueError("Probe request digest differs from its frozen identity")
    mode = request.get("mode", "plan")
    if mode != probe.get("mode") or mode not in {"plan", "critique", "analyze"}:
        raise ValueError("Probe mode differs from its request")
    known = probe.get("known_element_ids")
    if (
        not isinstance(known, list)
        or not known
        or len(known) > 256
        or any(not _text(identity) for identity in known)
        or len(set(known)) != len(known)
        or set(summary_element_ids(request.get("source_summary"))) != set(known)
    ):
        raise ValueError("Probe source metadata differs from its known element IDs")
    mapping = probe.get("element_map")
    if (
        not isinstance(mapping, dict)
        or set(mapping) != set(known)
        or any(not _text(identity) for identity in mapping.values())
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError("Probe element map must cover known IDs bijectively")
    source_sha = _image_ref(request.get("source_render"), root, files)
    if source_sha != probe.get("source_sha256") or not _hash(probe.get("original_source_sha256")):
        raise ValueError("Probe source image differs from its identity")
    menu, plans, candidate_images = request.get("plan_menu"), [], {}
    image_shas = [source_sha]
    if mode == "analyze":
        if any(key in request for key in ("plan_menu", "target_canvas", "source_analysis")):
            raise ValueError("Analyze probes must be source-only")
        if probe.get("menu_digest") is not None or probe.get("plan_map") != {}:
            raise ValueError("Analyze probes cannot select plans")
        source_digest = request.get("source_digest")
    else:
        if not isinstance(menu, dict) or menu.get("schema_version") != 1:
            raise ValueError("Choice probes require a versioned plan menu")
        menu_digest = digest({key: value for key, value in menu.items() if key != "menu_digest"})
        if menu.get("menu_digest") != menu_digest or probe.get("menu_digest") != menu_digest:
            raise ValueError("Probe menu differs from its bound digest")
        plans = menu.get("plans")
        plan_ids = [plan.get("plan_id") for plan in plans or [] if isinstance(plan, dict)]
        if (
            not isinstance(plans, list)
            or not 1 <= len(plans) <= 7
            or len(plan_ids) != len(plans)
            or any(not _text(identity) for identity in plan_ids)
            or len(set(plan_ids)) != len(plan_ids)
            or not isinstance(probe.get("plan_map"), dict)
            or set(probe["plan_map"]) != set(plan_ids)
            or any(not _text(identity) for identity in probe["plan_map"].values())
        ):
            raise ValueError("Probe plan map differs from the known menu")
        supplied = request.get("plans", plans)
        if not isinstance(supplied, list) or len(supplied) != len(plans):
            raise ValueError("Rendered choices differ from the known menu")
        rendered = {plan.get("plan_id"): plan for plan in supplied if isinstance(plan, dict)}
        if len(rendered) != len(supplied) or set(rendered) != set(plan_ids):
            raise ValueError("Rendered choices contain unknown or duplicate plan IDs")
        for plan in plans:
            item = rendered[plan["plan_id"]]
            if {key: value for key, value in item.items() if key != "render"} != {
                key: value for key, value in plan.items() if key != "render"
            }:
                raise ValueError("Rendered choice changed clean menu metadata")
            if item.get("render") is not None:
                candidate_images[plan["plan_id"]] = _image_ref(item["render"], root, files)
                image_shas.append(candidate_images[plan["plan_id"]])
        if mode == "critique" and (len(plans) != 2 or len(image_shas) != 3):
            raise ValueError("Critique probes require source and two rendered choices")
        source_digest = menu.get("source_digest")
        if request.get("source_digest", source_digest) != source_digest:
            raise ValueError("Request source and menu identities differ")
        if request.get("target_canvas", menu.get("target_canvas")) != menu.get("target_canvas"):
            raise ValueError("Request target and menu identities differ")
    if not _hash(source_digest):
        raise ValueError("Probe requires a bound source design digest")
    analysis_digest = None
    if "source_analysis" in request or "source_analysis_digest" in request:
        from engine.visual_analysis import validate_analysis

        analysis = validate_analysis(
            request.get("source_analysis"),
            source_digest=source_digest,
            source_render_sha256=source_sha,
            known_element_ids=known,
        )
        analysis_digest = digest(analysis)
        if request.get("source_analysis_digest") != analysis_digest:
            raise ValueError("Source analysis differs from its bound handoff digest")
    return {
        "request": request,
        "source_digest": source_digest,
        "image_shas": image_shas,
        "plans": plans,
        "candidate_images": candidate_images,
        "analysis_digest": analysis_digest,
    }


def load_bundle(path):
    """Verify the immutable artifact, including copied implementation files."""
    path = Path(path)
    bundle, root = _json(path), path.parent
    if (
        not isinstance(bundle, dict)
        or type(bundle.get("schema_version")) is not int
        or bundle["schema_version"] != 1
        or bundle.get("protocol") != PROTOCOL
        or bundle.get("purpose") not in {"synthetic", "research"}
        or bundle.get("bundle_sha256")
        != canonical_json_sha256({k: v for k, v in bundle.items() if k != "bundle_sha256"})
    ):
        raise ValueError("Unsupported or changed evaluation bundle")
    files = bundle.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise ValueError("A frozen file inventory is required")
    for name, digest in files.items():
        artifact = safe_artifact_path(root, name)
        if (
            not _hash(digest)
            or not artifact.is_file()
            or artifact.stat().st_size > 32_000_000
            or sha256_file(artifact) != digest
        ):
            raise ValueError("Evaluation input file changed: " + str(name))
    code = bundle.get("snapshot", {}).get("code_hashes")
    if not isinstance(code, dict) or not code or any(files.get(k) != v for k, v in code.items()):
        raise ValueError("Frozen code copies must be included in the file inventory")
    cases, probes = bundle.get("cases"), bundle.get("probes")
    if not isinstance(cases, list) or not isinstance(probes, list) or not 1 <= len(probes) <= 4096:
        raise ValueError("Evaluation cases and bounded probes are required")
    by_case, by_probe, conditions = {}, {}, set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not _text(case.get("case_id"))
            or not _text(case.get("family_id"))
            or case.get("split") not in {"train", "validation", "test"}
            or case["case_id"] in by_case
        ):
            raise ValueError("Cases require unique IDs, family IDs and known splits")
        by_case[case["case_id"]] = case
        if "source_json_path" in case:
            if case["source_json_path"] not in files:
                raise ValueError("Original source JSON is absent from the frozen inventory")
            source = _json(safe_artifact_path(root, case["source_json_path"]))
            if canonical_json_sha256(source) != case.get("source_json_sha256"):
                raise ValueError("Original source JSON differs from its canonical identity")
            if "source_visual_digest" in case:
                from engine.visual_plan import source_digest

                if source_digest(source) != case["source_visual_digest"]:
                    raise ValueError("Original source differs from its visual-plan digest protocol")
            ids = [element.get("id") for element in source.get("elements", [])]
            if (
                not ids
                or any(not _text(identity) for identity in ids)
                or len(set(ids)) != len(ids)
                or case.get("source_element_ids") != ids
            ):
                raise ValueError("Original source element IDs differ from the frozen lineage")
    for probe in probes:
        if not isinstance(probe, dict) or not _text(probe.get("probe_id")):
            raise ValueError("Probe requires a nonempty ID")
        case = by_case.get(probe.get("case_id"))
        key = probe.get("case_id"), probe.get("mode"), probe.get("condition")
        if (
            case is None
            or probe["probe_id"] in by_probe
            or probe.get("condition") not in CONDITIONS
            or probe.get("family_id") != case["family_id"]
            or probe.get("split") != case["split"]
            or key in conditions
        ):
            raise ValueError("Probe identity, condition, family or split is inconsistent")
        conditions.add(key)
        by_probe[probe["probe_id"]] = probe
    return bundle, by_case, by_probe


def _receipt(receipt, probe, prepared, code):
    if not isinstance(receipt, dict):
        return False, True
    bindings = {
        "request_digest": probe["request_digest"],
        "source_render_sha256": probe["source_sha256"],
        "source_digest": prepared["source_digest"],
        "menu_digest": probe["menu_digest"],
    }
    if prepared["analysis_digest"] is not None or "source_analysis_digest" in receipt:
        bindings["source_analysis_digest"] = prepared["analysis_digest"]
    intact = all(key not in receipt or receipt[key] == value for key, value in bindings.items())
    images = receipt.get("images")
    if images is not None:
        intact = (
            intact
            and isinstance(images, list)
            and [item.get("sha256") if isinstance(item, dict) else None for item in images]
            == prepared["image_shas"]
        )
    weights = receipt.get("weights")
    model_bound = _hash(receipt.get("weight_sha256")) or (
        isinstance(weights, dict) and bool(weights) and all(_hash(v) for v in weights.values())
    )
    implementation = receipt.get("implementation_hashes")
    frozen_names = {Path(name).name: value for name, value in code.items()}
    code_bound = (
        receipt.get("implementation_unchanged") is True
        and isinstance(implementation, dict)
        and bool(implementation)
        and all(frozen_names.get(name) == value for name, value in implementation.items())
    )
    complete = (
        intact
        and receipt.get("executed") is True
        and all(key in receipt for key in bindings)
        and images is not None
        and receipt.get("input_image_count") == len(prepared["image_shas"])
        and model_bound
        and bool(re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("model_revision", ""))))
        and _hash(receipt.get("prompt_sha256"))
        and code_bound
    )
    return bool(complete), bool(intact)


def _prediction(row, probe, prepared, code):
    result = {
        "probe_id": probe["probe_id"],
        "transcript_present": True,
        "case_id": probe["case_id"],
        "family_id": probe["family_id"],
        "split": probe["split"],
        "condition": probe["condition"],
        "mode": probe["mode"],
        "protocol": row["protocol"],
        "executed": row["executed"],
        "fallback": row["fallback"],
        "valid": False,
        "known_ids_valid": False,
        "input_integrity": True,
        "receipt_verified": False,
        "qualified_model_success": False,
        "choice": None,
        "canonical_outcome": None,
        "abstain": False,
        "issues": [],
        "observation_fingerprint": None,
        "raw_fingerprint": None,
        "error": row["error"],
    }
    response = row["response"]
    receipt = response.get("receipt") if isinstance(response, dict) else None
    verified, intact = _receipt(receipt, probe, prepared, code)
    result.update(receipt_verified=verified, input_integrity=intact)
    raw = row["raw_response"]
    generated = receipt.get("raw_output") if isinstance(receipt, dict) else None
    if raw is None:
        raw = generated
    if isinstance(raw, str):
        result["raw_fingerprint"] = canonical_json_sha256(raw)
    try:
        if row["error"] is not None:
            raise ValueError("Transcript records an execution error")
        if row["protocol"] == "critic":
            if probe["mode"] == "analyze" or len(prepared["plans"]) != 2:
                raise ValueError("Critic protocol requires two rendered candidates")
            parsed = parse_model_verdict(raw, {side: probe["known_element_ids"] for side in "AB"})
            if not parsed["valid"]:
                raise ValueError("Invalid critic JSON or unknown grounded element IDs")
            choice = parsed["choice"]
            result["issues"] = [
                {
                    "candidate_id": probe["plan_map"][
                        prepared["plans"]["AB".index(i["candidate"])]["plan_id"]
                    ],
                    "code": i["code"],
                    "element_ids": [probe["element_map"][eid] for eid in i["element_ids"]],
                }
                for i in parsed["issues"]
            ]
            result["abstain"] = choice in {"tie", "neither"}
            result["choice"] = choice
            result["canonical_outcome"] = (
                "tie"
                if choice == "tie"
                else "abstain"
                if choice == "neither"
                else probe["plan_map"][prepared["plans"]["AB".index(choice)]["plan_id"]]
            )
        else:
            if not isinstance(response, dict):
                raise ValueError("Provider protocol requires a response object")
            if probe["mode"] == "analyze":
                from engine.visual_analysis import validate_analysis

                analysis = validate_analysis(
                    response.get("analysis"),
                    source_digest=prepared["source_digest"],
                    source_render_sha256=probe["source_sha256"],
                    known_element_ids=probe["known_element_ids"],
                )
                observations = [
                    {
                        **item,
                        "element_ids": sorted(
                            probe["element_map"][eid] for eid in item["element_ids"]
                        ),
                    }
                    for item in analysis["observations"]
                ]
                result["observation_fingerprint"] = canonical_json_sha256(
                    sorted(observations, key=canonical_json_sha256)
                )
                result["abstain"] = analysis["abstain"]
                if isinstance(generated, str) and generated:
                    from engine.visual_analysis import bind_analysis

                    bound = bind_analysis(
                        parse_json(generated),
                        source_digest=prepared["source_digest"],
                        source_render_sha256=probe["source_sha256"],
                        known_element_ids=probe["known_element_ids"],
                    )
                    if bound != analysis:
                        raise ValueError("Generated analysis differs from its bound sidecar")
            else:
                decision = response.get("decision")
                if (
                    not isinstance(decision, dict)
                    or decision.get("schema_version") != 1
                    or decision.get("menu_digest") != probe["menu_digest"]
                    or decision.get("plan_id") not in probe["plan_map"]
                ):
                    raise ValueError("Provider decision is not bound to a known menu plan")
                plan_id = decision["plan_id"]
                index = [plan["plan_id"] for plan in prepared["plans"]].index(plan_id)
                result["choice"] = "ABCDEFG"[index]
                result["canonical_outcome"] = probe["plan_map"][plan_id]
                if isinstance(receipt, dict) and receipt.get("choice_letter") not in {
                    None,
                    result["choice"],
                }:
                    raise ValueError("Receipt choice differs from the bound decision")
                # The collector's raw stdout can be the provider envelope.
                # Actual model JSON, when recorded, lives in receipt.raw_output.
                choice_json = generated
                if not isinstance(choice_json, str) and isinstance(raw, str):
                    try:
                        stdout = parse_json(raw)
                    except ValueError:
                        stdout = None
                    if isinstance(stdout, dict) and "choice" in stdout:
                        choice_json = raw
                    elif (
                        isinstance(stdout, dict)
                        and "decision" in stdout
                        and (stdout["decision"] != decision)
                    ):
                        raise ValueError("Provider stdout envelope differs from its decision")
                if isinstance(choice_json, str) and choice_json:
                    from experiments.visual_planner.provider import parse_generated

                    parsed = parse_generated(
                        choice_json,
                        {
                            "mode": probe["mode"],
                            "menu": {"menu_digest": probe["menu_digest"]},
                            "plans": prepared["plans"],
                        },
                    )
                    if parsed is None or parsed["decision"]["plan_id"] != plan_id:
                        raise ValueError("Provider raw response differs from its decision")
        result["known_ids_valid"] = True
        if not intact:
            raise ValueError("Receipt input identities differ from the frozen request")
        result["valid"] = True
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        result["error"] = str(exc)
    result["recommendation_eligible"] = (
        result["valid"] and result["executed"] and not result["fallback"]
    )
    result["qualified_model_success"] = (
        result["recommendation_eligible"] and verified and not result["abstain"]
    )
    return result


def _labels(path, bundle, cases, probes, *, test):
    labels = _json(path)
    if (
        not isinstance(labels, dict)
        or labels.get("schema_version") != 1
        or labels.get("bundle_sha256") != bundle["bundle_sha256"]
        or labels.get("label_kind") not in {"human", "synthetic"}
        or not isinstance(labels.get("decisions"), list)
    ):
        raise ValueError("Labels differ from the frozen bundle or label schema")
    known, decisions = defaultdict(set), {}
    for probe in probes.values():
        known[probe["case_id"]].update(probe["plan_map"].values())
    for row in labels["decisions"]:
        case = cases.get(row.get("case_id")) if isinstance(row, dict) else None
        if case is None or row["case_id"] in decisions or (case["split"] == "test") != test:
            raise ValueError("Labels contain duplicate, unknown or incorrectly sealed cases")
        outcomes = known[row["case_id"]] | {"tie", "abstain"}
        axes = row.get("axes")
        if (
            row.get("preferred_candidate_id") not in outcomes
            or not isinstance(axes, dict)
            or axes.keys() - AXES
            or any(value not in outcomes for value in axes.values())
        ):
            raise ValueError("Labels reference unknown outcomes or axes")
        decisions[row["case_id"]] = {**row, "label_kind": labels["label_kind"]}
    return decisions


def _agreement(rows, labels):
    relevant = [
        row
        for row in rows
        if row["condition"] in AGREEMENT_CONDITIONS
        and row["mode"] != "analyze"
        and row["case_id"] in labels
    ]

    def metrics(items, axis=None):
        labeled_count = len(items)

        def expected(row):
            label = labels[row["case_id"]]
            return label["axes"][axis] if axis else label["preferred_candidate_id"]

        abstain_count = sum(expected(row) == "abstain" for row in items)
        items = [row for row in items if expected(row) != "abstain"]

        def matches(row):
            return row["recommendation_eligible"] and row["canonical_outcome"] == expected(row)

        eligible = [row for row in items if row["recommendation_eligible"]]
        qualified = [row for row in items if row["qualified_model_success"]]
        families = defaultdict(list)
        for row in items:
            families[row["family_id"]].append(float(matches(row)))
        return {
            "labeled_probe_count": labeled_count,
            "scored_probe_count": len(items),
            "abstain_label_count": abstain_count,
            "eligible_recommendations": len(eligible),
            "matches": sum(matches(row) for row in items),
            "strict_agreement": _mean([float(matches(row)) for row in items]),
            "eligible_only_agreement": _mean([float(matches(row)) for row in eligible]),
            "qualified_recommendations": len(qualified),
            "qualified_only_agreement": _mean([float(matches(row)) for row in qualified]),
            "family_macro_agreement": _mean([_mean(values) for values in families.values()]),
        }

    originals = [
        row
        for row in relevant
        if row["condition"] == "original"
        and labels[row["case_id"]]["preferred_candidate_id"] != "abstain"
    ]
    orders = defaultdict(dict)
    for row in relevant:
        if row["condition"] in {"original", "swapped"} and (
            labels[row["case_id"]]["preferred_candidate_id"] != "abstain"
        ):
            orders[row["case_id"], row["mode"]][row["condition"]] = row
    both_correct = [
        all(
            row["recommendation_eligible"]
            and row["canonical_outcome"] == labels[row["case_id"]]["preferred_candidate_id"]
            for row in pair.values()
        )
        for pair in orders.values()
        if {"original", "swapped"} <= pair.keys()
    ]
    baseline = {}
    for choice in ("A", "B"):
        binary = [row for row in originals if row.get("letter_outcomes", {}).get(choice)]
        baseline["always_" + choice] = _mean(
            [
                float(
                    row["letter_outcomes"][choice]
                    == labels[row["case_id"]]["preferred_candidate_id"]
                )
                for row in binary
            ]
        )
    baseline["case_baseline"] = _mean(
        [
            float(
                row.get("baseline_candidate_id") == labels[row["case_id"]]["preferred_candidate_id"]
            )
            for row in originals
            if row.get("baseline_candidate_id") is not None
        ]
    )
    return {
        "scope": "Supplied labels only; synthetic labels measure weak rule agreement",
        "label_kinds": sorted({labels[row["case_id"]]["label_kind"] for row in relevant}),
        "fallbacks_excluded_from_matches": True,
        "conditions": sorted(AGREEMENT_CONDITIONS),
        **metrics(relevant),
        "both_orders_pair_count": len(both_correct),
        "both_orders_correct_count": sum(both_correct),
        "both_orders_correct_rate": _mean([float(value) for value in both_correct]),
        "per_axis": {
            axis: metrics([row for row in relevant if axis in labels[row["case_id"]]["axes"]], axis)
            for axis in sorted(AXES)
        },
        "constant_baselines_on_originals": baseline,
    }


def _controls(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["case_id"], row["mode"]][row["condition"]] = row
    results = {}
    for condition in sorted(CONDITIONS - {"original"}):
        values, qualified = [], []
        paired = 0
        for group in grouped.values():
            original, changed = group.get("original"), group.get(condition)
            if original is None or changed is None:
                continue
            paired += 1
            if not all(row["recommendation_eligible"] for row in (original, changed)):
                continue
            field = (
                "observation_fingerprint" if original["mode"] == "analyze" else "canonical_outcome"
            )
            value = original[field] == changed[field]
            values.append(float(value))
            if original["qualified_model_success"] and changed["qualified_model_success"]:
                qualified.append(float(value))
        results[condition] = {
            "paired_count": paired,
            "comparable_count": len(values),
            "invariance_rate": _mean(values),
            "change_rate": _mean([1 - value for value in values]),
            "qualified_comparable_count": len(qualified),
            "qualified_invariance_rate": _mean(qualified),
            "interpretation": "Sensitivity diagnostic; no automatic correct-answer claim"
            if condition in {"blank_source", "replaced_source"}
            else "Canonical identity consistency across equivalent presentations",
        }
    return results


def evaluate(
    bundle_path,
    transcripts_paths,
    labels_path=None,
    test_labels_path=None,
    recipe_freeze_path=None,
    include_test=False,
):
    """Validate and score supplied artifacts; this function never runs inference."""
    bundle, cases, probes = load_bundle(bundle_path)
    if include_test or test_labels_path is not None:
        if not include_test or recipe_freeze_path is None:
            raise ValueError("Test labels require include_test and a frozen recipe before opening")
        freeze = _json(recipe_freeze_path)
        if (
            not isinstance(freeze, dict)
            or freeze.get("bundle_sha256") != bundle["bundle_sha256"]
            or not _hash(freeze.get("recipe_sha256"))
        ):
            raise ValueError("Test labels require a matching bundle-bound recipe freeze")
    active = {key: p for key, p in probes.items() if include_test or p["split"] != "test"}
    root = Path(bundle_path).parent
    prepared = {key: _request(probe, root, bundle["files"]) for key, probe in active.items()}
    originals = {}
    for probe in active.values():
        if probe["condition"] == "original":
            originals[probe["case_id"], probe["mode"]] = probe
    for probe in active.values():
        original = originals.get((probe["case_id"], probe["mode"]))
        if original is None or probe["original_source_sha256"] != original["source_sha256"]:
            raise ValueError("Control probes require their bound original source")
        if set(probe["element_map"].values()) != set(original["element_map"].values()):
            raise ValueError("Control element identity differs from the original")
        if set(probe["plan_map"].values()) != set(original["plan_map"].values()):
            raise ValueError("Control candidate identity differs from the original")
        case = cases[probe["case_id"]]
        inputs, original_inputs = prepared[probe["probe_id"]], prepared[original["probe_id"]]
        if case.get("source_json_sha256") is not None and (
            inputs["source_digest"] != case.get("source_visual_digest", case["source_json_sha256"])
            or not set(probe["element_map"].values()) <= set(case["source_element_ids"])
            or probe["original_source_sha256"] != case.get("source_render_sha256")
        ):
            raise ValueError("Probe source or known IDs differ from the original JSON lineage")
        if "candidate_ids" in case and not set(probe["plan_map"].values()) <= set(
            case["candidate_ids"]
        ):
            raise ValueError("Probe maps to a candidate outside the original case")
        canonical_images = {
            probe["plan_map"][plan_id]: sha for plan_id, sha in inputs["candidate_images"].items()
        }
        original_images = {
            original["plan_map"][plan_id]: sha
            for plan_id, sha in original_inputs["candidate_images"].items()
        }
        if canonical_images != original_images:
            raise ValueError("Control changed physical candidate image identities")
        summary = {
            **inputs["request"]["source_summary"],
            "elements": [
                {**row, "id": probe["element_map"][row["id"]]}
                for row in inputs["request"]["source_summary"]["elements"]
            ],
        }
        original_summary = {
            **original_inputs["request"]["source_summary"],
            "elements": [
                {**row, "id": original["element_map"][row["id"]]}
                for row in original_inputs["request"]["source_summary"]["elements"]
            ],
        }
        if summary != original_summary:
            raise ValueError("Control changed source metadata beyond opaque identifiers")
        if probe["condition"] in {"original", "swapped", "opaque_ids"} and (
            probe["source_sha256"] != original["source_sha256"]
        ):
            raise ValueError("Equivalent controls changed the original source pixels")
    labels = _labels(labels_path, bundle, cases, probes, test=False) if labels_path else {}
    if test_labels_path:
        labels.update(_labels(test_labels_path, bundle, cases, probes, test=True))
    methods, identities, seen, receipt_models = defaultdict(list), {}, set(), {}
    row_count = 0
    for path in transcripts_paths:
        path = Path(path)
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("Transcript file exceeds its byte budget")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row_count += 1
            if row_count > MAX_TRANSCRIPTS:
                raise ValueError("Transcript count exceeds the bounded evaluator budget")
            row = parse_json(line)
            fields = {
                "probe_id",
                "bundle_sha256",
                "request_digest",
                "method_name",
                "method_identity",
                "method_identity_sha256",
                "protocol",
                "executed",
                "fallback",
                "response",
                "raw_response",
                "error",
            }
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError("Transcript must match the versioned row schema")
            probe = probes.get(row["probe_id"])
            if (
                probe is None
                or row["bundle_sha256"] != bundle["bundle_sha256"]
                or (row["request_digest"] != probe["request_digest"])
            ):
                raise ValueError("Transcript names an unknown probe or stale input digest")
            name, identity = row["method_name"], row["method_identity"]
            if (
                not _text(name)
                or not isinstance(identity, dict)
                or not identity
                or row["method_identity_sha256"] != canonical_json_sha256(identity)
                or row["protocol"] not in {"provider", "critic"}
                or type(row["executed"]) is not bool
                or type(row["fallback"]) is not bool
                or row["response"] is not None
                and not isinstance(row["response"], dict)
                or row["raw_response"] is not None
                and not isinstance(row["raw_response"], str)
                or row["error"] is not None
                and not isinstance(row["error"], str)
            ):
                raise ValueError("Transcript method identity or output types are invalid")
            key = name, row["probe_id"]
            if key in seen:
                raise ValueError("Duplicate probe/method transcript")
            seen.add(key)
            if name in identities and identities[name] != row["method_identity_sha256"]:
                raise ValueError("One method name cannot mix immutable method identities")
            identities[name] = row["method_identity_sha256"]
            if row["probe_id"] not in active:
                continue
            receipt = (row["response"] or {}).get("receipt")
            if isinstance(receipt, dict) and receipt.get("executed") is True:
                model_identity = canonical_json_sha256(
                    {
                        k: receipt.get(k)
                        for k in (
                            "model_id",
                            "model_revision",
                            "weights",
                            "weight_sha256",
                            "implementation_hashes",
                            "runtime_archives",
                            "max_new_tokens",
                        )
                    }
                )
                if name in receipt_models and receipt_models[name] != model_identity:
                    raise ValueError("One method name cannot mix receipt model/code identities")
                receipt_models[name] = model_identity
            prediction = _prediction(
                row, probe, prepared[row["probe_id"]], bundle["snapshot"]["code_hashes"]
            )
            prediction["letter_outcomes"] = {
                "ABCDEFG"[index]: probe["plan_map"][plan["plan_id"]]
                for index, plan in enumerate(prepared[row["probe_id"]]["plans"])
            }
            prediction["baseline_candidate_id"] = cases[probe["case_id"]].get(
                "baseline_candidate_id"
            )
            methods[name].append(prediction)
    reports = {}
    for name in sorted(identities):
        rows = methods[name]
        present = len(rows)
        present_ids = {row["probe_id"] for row in rows}
        for probe_id in active.keys() - present_ids:
            probe = active[probe_id]
            row = _prediction(
                {
                    "protocol": "provider",
                    "executed": False,
                    "fallback": False,
                    "response": None,
                    "raw_response": None,
                    "error": "Missing transcript",
                },
                probe,
                prepared[probe_id],
                bundle["snapshot"]["code_hashes"],
            )
            row["transcript_present"] = False
            row["baseline_candidate_id"] = cases[probe["case_id"]].get("baseline_candidate_id")
            row["letter_outcomes"] = {
                "ABCDEFG"[index]: probe["plan_map"][plan["plan_id"]]
                for index, plan in enumerate(prepared[probe_id]["plans"])
            }
            rows.append(row)
        rows.sort(key=lambda row: row["probe_id"])
        for row in rows:
            case = cases[row["case_id"]]
            if row["mode"] == "analyze":
                row["admitted_recommendation"] = False
                row["selection"] = {"candidate_id": None, "fallback": False, "applicable": False}
                continue
            admitted = set(case.get("admitted_candidate_ids", []))
            requested = row["canonical_outcome"] if row["recommendation_eligible"] else None
            chosen = requested if requested in admitted else None
            if chosen is None and case.get("baseline_candidate_id") in admitted:
                chosen = case["baseline_candidate_id"]
            row["admitted_recommendation"] = requested in admitted
            row["selection"] = {
                "candidate_id": chosen,
                "fallback": requested not in admitted,
                "applicable": True,
            }
            if row["selection"]["fallback"]:
                row["qualified_model_success"] = False
        letters = Counter(
            row["choice"]
            for row in rows
            if row["recommendation_eligible"] and row["choice"] is not None
        )
        raw = [row["raw_fingerprint"] for row in rows if row["raw_fingerprint"] is not None]
        reports[name] = {
            "method_identity_sha256": identities[name],
            "counts": {
                "expected_probes": len(active),
                "transcripts": present,
                "missing_transcripts": len(active) - present,
                **{
                    key: sum(bool(row[key]) for row in rows)
                    for key in (
                        "executed",
                        "fallback",
                        "valid",
                        "known_ids_valid",
                        "receipt_verified",
                        "recommendation_eligible",
                        "qualified_model_success",
                        "abstain",
                    )
                },
                "invalid_outputs": sum(not row["valid"] for row in rows),
                "input_integrity_failures": sum(not row["input_integrity"] for row in rows),
                "unverified_transcripts": sum(not row["receipt_verified"] for row in rows),
            },
            "letter_distribution": dict(letters),
            "dominant_letter_fraction": max(letters.values()) / sum(letters.values())
            if letters
            else None,
            "unique_raw_responses": len(set(raw)),
            "constant_raw_response": len(raw) >= 2 and len(set(raw)) == 1,
            "admissibility": {
                "admitted_recommendations": sum(row["admitted_recommendation"] for row in rows),
                "baseline_fallbacks": sum(row["selection"]["fallback"] for row in rows),
                "no_admitted_selection": sum(
                    row["selection"]["candidate_id"] is None for row in rows
                ),
                "scope": "Supplied admission list; baseline fallbacks earn no model match",
            },
            "controls": _controls(rows),
            "agreement": _agreement(rows, labels),
            "predictions": rows,
        }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "bundle_sha256": bundle["bundle_sha256"],
        "purpose": bundle["purpose"],
        "test_split_unblinded": include_test,
        "input_files_verified": True,
        "snapshot_code_verified": True,
        "inference_executed_by_evaluator": False,
        "quality_claim": "Protocol, control and supplied-label diagnostics; no aesthetic approval",
        "receipt_scope": "Supplied receipt identities; execution is not independently rerun",
        "methods": reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--recipe-freeze", type=Path)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(
        args.bundle,
        args.transcripts,
        args.labels,
        args.test_labels,
        args.recipe_freeze,
        args.include_test,
    )
    from tools.vision_lora_smoke import write_json

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
