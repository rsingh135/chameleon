"""Pure JSON split freezes, baselines and predeclared model-evaluation gates.

No files, gold labels, model packages, networks or inference are opened here.
Trusted registries come from the caller's independently verified inventory.
References establish declared provenance, not license conclusions or quality.

Manifest schema::

    {"schema_version": 1, "families": [
      {"family_id": "campaign", "split": "prospective_test",
       "asset_family_ids": ["primary-artwork-lineage"], "common_asset_ids": [],
       "sources": [{"source_id": "work", "original_sha256": "<64 hex>",
         "json_sha256": "<64 hex>", "render_sha256": "<64 hex>",
         "primary_asset_sha256s": [], "request_path": "requests/work.json",
         "license_id": "CC0-1.0", "license_reference": "source/license",
         "provenance": {"origin": "licensed_source", "reference": "source/url"}}]}],
     "common_asset_ids": {"brand-logo": {"kind": "logo", "sha256": "<64 hex>"}},
     "local_gold_refs": [{"path": "sealed/purpose.json", "sha256": "<64 hex>"}]}

Splits: train, tuning_validation, prospective_test, legacy_development. At least
three independent families are required in each new evaluation split. Reserve
mapping: trusted source ID -> original_sha256/json_sha256. Gradient/legacy
registries: source_ids, family_ids, original_sha256s, primary_asset_sha256s lists.
Denylist and all registries are explicit mandatory arguments, not inferred.

Gate report schema::

    {"task": "purpose", "protocol_sha256": "...", "manifest_sha256": "...",
     "gold_sha256": "...", "method_identity_sha256": "...", "identity_bound": true,
     "request_set_sha256": "...", "record_ids": ["unit-1", "unit-2"],
     "family_ids": ["family-1", "family-2", "family-3"],
     "coverage": {"expected": 100, "completed": 100, "invalid": 0,
                  "missing": 0, "fallback": 0},
     "raw": {"accuracy": .8, "macro_f1": .8, "balanced_accuracy": .8,
             "family_macro_accuracy": .8,
             "support": {"gold_count": 6, "family_count": 2, "recall": .8, "f1": .8}},
     "controls": {"blank": {"gold_count": 5, "accuracy": .8},
                  "replaced": {"gold_count": 5, "accuracy": .8}}}

Purpose baselines: pretrained, always_body, train_majority, metadata. Accuracy
exists for all; pretrained/metadata additionally supply macro_f1,
balanced_accuracy and family_macro_accuracy; pretrained supplies support recall/f1.
Every baseline also supplies manifest_sha256, gold_sha256, request_set_sha256,
record_ids, family_ids, identity_bound, method_identity_sha256 and coverage,
matching the report's evaluation identities and complete coverage (its method
identity identifies the baseline separately). Unit/family lists are sorted and
unique. The request-set hash binds exact request identities and bytes.
Critic raw adds swap_consistency, both_orders_correct, non_n_pair_count and
class_support {A/B/N: {gold_count, family_count}}. Its baselines are pretrained,
always_A, always_B, majority, order_baseline (accuracy and balanced_accuracy for
all, macro_f1 for pretrained). Critic controls: neither and source_ablated,
each {gold_count, expected_n_accuracy}. Report mix: original_count/control_count.

Purpose raw metrics cover originals only; critic raw metrics cover matched
originals and their physical swaps. Artificial controls are always separate.
Only raw metrics gate acceptance. An explicit exact_tie_to_N policy can also
report policy_metrics; it never replaces or retroactively alters raw results.
All thresholds and identities must be frozen before gold is joined. Passing
these gates is not statistical significance or end-to-end design approval.
Prospective feedback is disclosed only after candidate, code and tie policy are
frozen. A failed prospective split is retired to legacy_development before its
feedback is used, and fresh final families are required. This pure module freezes
that policy; it cannot independently prove disclosure timing or data provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import PurePosixPath

SHA = re.compile(r"[0-9a-f]{64}\Z")
MANUAL_IDS = {"doory_modern-resume", "opensuse_banner_we_believe"}
SPLITS = {"train", "tuning_validation", "prospective_test", "legacy_development"}
DEFAULTS = {
    "purpose": {
        "accuracy_margin": 0.03,
        "macro_f1_margin": 0.03,
        "balanced_accuracy_margin": 0.03,
        "support_recall_min": 0.5,
        "support_f1_min": 0.5,
        "support_gold_min": 6,
        "support_families_min": 2,
        "control_accuracy_min": 0.8,
    },
    "critic": {
        "accuracy_margin": 0.05,
        "balanced_accuracy_margin": 0.05,
        "macro_f1_margin": 0.05,
        "swap_consistency_min": 0.9,
        "both_orders_correct_min": 0.6,
        "control_accuracy_min": 0.8,
        "class_gold_min": 4,
        "class_families_min": 3,
        "control_fraction_max": 0.25,
    },
}
PROSPECTIVE_POLICY = {
    "candidate_code_and_tie_policy_frozen_before_feedback": True,
    "failed_families_retired_to_legacy_before_feedback_use": True,
    "new_prospective_families_required_after_failure": True,
}


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("References require exact lowercase 64-character SHA256")
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Identity and provenance fields require bounded nonempty text")
    return value


def _strings(value):
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("Identity lists must be unique JSON string arrays")
    return {_text(item) for item in value}


def _path(value):
    _text(value)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("Gold and request references must be normalized local relative paths")
    if path.as_posix() != value or value == ".":
        raise ValueError("Reference path is not normalized")
    reserved = {"con", "prn", "aux", "nul"} | {
        name + str(number) for name in ("com", "lpt") for number in range(1, 10)
    }
    if any(
        part.endswith((".", " "))
        or part.split(".")[0].casefold() in reserved
        or any(ord(char) < 32 or char in '<>"|?*' for char in part)
        for part in path.parts
    ):
        raise ValueError("Reference paths must avoid portable filesystem aliases")
    return value


def _registry(value):
    keys = {"source_ids", "family_ids", "original_sha256s", "primary_asset_sha256s"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Trusted gradient/legacy registries require four explicit identity lists")
    result = {key: _strings(value[key]) for key in keys}
    for key in ("original_sha256s", "primary_asset_sha256s"):
        for item in result[key]:
            _sha(item)
    return result


def _freeze(key, value):
    text = canonical_json(value)
    return {
        key: deepcopy(value),
        "canonical_json": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def freeze_manifest(
    manifest,
    *,
    denylist,
    catalogue_reserves,
    previous_gradient_registry,
    legacy_development_registry,
):
    """Validate source/campaign separation and freeze references without opening them."""
    denied = _strings(denylist)
    if not isinstance(catalogue_reserves, dict) or not catalogue_reserves.keys() >= MANUAL_IDS:
        raise ValueError("Trusted reserve hash mapping must include Resume and WeBelieve")
    if not MANUAL_IDS | set(catalogue_reserves) <= denied:
        raise ValueError("Explicit denylist must cover manual and current catalogue reserves")
    reserved_hashes = set()
    for source_id, refs in catalogue_reserves.items():
        _text(source_id)
        if not isinstance(refs, dict) or set(refs) != {"original_sha256", "json_sha256"}:
            raise ValueError("Trusted reserve references require original and JSON hashes")
        reserved_hashes.update(_sha(value) for value in refs.values())
    trained, legacy = _registry(previous_gradient_registry), _registry(legacy_development_registry)
    manifest_fields = {"schema_version", "families", "common_asset_ids", "local_gold_refs"}
    if (
        not isinstance(manifest, dict)
        or set(manifest) != manifest_fields
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Manifest must use schema_version 1")
    common = manifest.get("common_asset_ids")
    if not isinstance(common, dict):
        raise ValueError("Common assets must be explicitly declared with hashes and kinds")
    common_hashes = set()
    for asset_id, value in common.items():
        _text(asset_id)
        if not isinstance(value, dict) or set(value) != {"sha256", "kind"}:
            raise ValueError("Common assets require kind and SHA256")
        if value["kind"] not in {"font", "logo", "icon"}:
            raise ValueError("Primary artwork cannot be declared a common font/logo/icon")
        common_hashes.add(_sha(value["sha256"]))
    if common_hashes & (trained["primary_asset_sha256s"] | legacy["primary_asset_sha256s"]):
        raise ValueError("Common assets cannot override trusted primary-artwork lineage")
    families = manifest.get("families")
    gold = manifest.get("local_gold_refs")
    if not isinstance(families, list) or not families or not isinstance(gold, list):
        raise ValueError("Manifest requires families and separate local_gold_refs lists")
    if any(not isinstance(row, dict) or set(row) != {"path", "sha256"} for row in gold):
        raise ValueError("Gold references may contain only local path and SHA256")
    gold_paths = {_path(row["path"]).casefold() for row in gold}
    if len(gold_paths) != len(gold):
        raise ValueError("Gold references must be unique separate local paths")
    for row in gold:
        if set(row) != {"path", "sha256"}:
            raise ValueError("Gold references may contain only local path and SHA256")
        _sha(row["sha256"])
    family_ids, source_ids, request_paths = set(), set(), set()
    lineage_owner, counts = {}, Counter()
    source_fields = {
        "source_id",
        "original_sha256",
        "json_sha256",
        "render_sha256",
        "primary_asset_sha256s",
        "request_path",
        "license_id",
        "license_reference",
        "provenance",
    }
    for family in families:
        if not isinstance(family, dict) or set(family) != {
            "family_id",
            "split",
            "asset_family_ids",
            "common_asset_ids",
            "sources",
        }:
            raise ValueError("Each family must contain only the declared metadata fields")
        fid, split = _text(family.get("family_id")), family.get("split")
        if fid in family_ids or split not in SPLITS:
            raise ValueError("Duplicate family identity or unknown split")
        family_ids.add(fid)
        counts[split] += 1
        lineages = _strings(family.get("asset_family_ids"))
        if not lineages:
            raise ValueError("Every family needs its authored campaign/artwork lineage")
        shared = _strings(family.get("common_asset_ids"))
        if not shared <= common.keys():
            raise ValueError("Shared common assets lack an explicit hash declaration")
        sources = family.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("Every family needs original source references")
        for source in sources:
            if not isinstance(source, dict) or set(source) != source_fields:
                raise ValueError("Source references have missing or unknown fields")
            sid = _text(source["source_id"])
            hashes = {
                _sha(source[key]) for key in ("original_sha256", "json_sha256", "render_sha256")
            }
            if sid in source_ids or sid in denied or hashes & reserved_hashes:
                raise ValueError("Duplicate, denied or reserved-alias source")
            source_ids.add(sid)
            primary = _strings(source["primary_asset_sha256s"])
            for item in primary:
                _sha(item)
            if primary & common_hashes:
                raise ValueError("A primary asset cannot also use the common-asset exception")
            is_trained = (
                sid in trained["source_ids"]
                or fid in trained["family_ids"]
                or bool(lineages & trained["family_ids"])
                or bool(hashes & trained["original_sha256s"])
                or bool(primary & trained["primary_asset_sha256s"])
            )
            is_legacy = (
                sid in legacy["source_ids"]
                or fid in legacy["family_ids"]
                or bool(lineages & legacy["family_ids"])
                or bool(hashes & legacy["original_sha256s"])
                or bool(primary & legacy["primary_asset_sha256s"])
            )
            if is_trained and split != "train":
                raise ValueError("Previously gradient-exposed source/artwork must remain TRAIN")
            if is_legacy and split != "legacy_development":
                raise ValueError("Old development sources must remain legacy_development")
            request = _path(source["request_path"]).casefold()
            if request in request_paths or any(
                request == path or request.startswith(path + "/") or path.startswith(request + "/")
                for path in gold_paths
            ):
                raise ValueError("Gold references must stay separate from unique model requests")
            request_paths.add(request)
            for key in ("license_id", "license_reference"):
                _text(source[key])
            provenance = source["provenance"]
            if not isinstance(provenance, dict) or set(provenance) != {"origin", "reference"}:
                raise ValueError("Source provenance requires origin and reference")
            for value in provenance.values():
                _text(value)
            for identity in [
                *("campaign:" + value for value in lineages),
                *("original:" + value for value in hashes),
                *("primary:" + value for value in primary),
            ]:
                if identity in lineage_owner and lineage_owner[identity] != fid:
                    raise ValueError("Source or primary-artwork alias crosses independent families")
                lineage_owner[identity] = fid
    if counts["prospective_test"] < 3 or counts["tuning_validation"] < 3:
        raise ValueError("Prospective test and tuning validation each need at least three families")
    body = deepcopy(manifest)
    body["trusted_split_bindings"] = {
        "denylist": sorted(denied),
        "catalogue_reserves": deepcopy(catalogue_reserves),
        "previous_gradient_registry": deepcopy(previous_gradient_registry),
        "legacy_development_registry": deepcopy(legacy_development_registry),
    }
    return _freeze("manifest", body)


def freeze_protocol(task, manifest_sha256, *, tie_policy="raw_argmax", thresholds=None):
    """Freeze thresholds before gold is joined; no threshold can be silently changed."""
    if task not in DEFAULTS or tie_policy not in {"raw_argmax", "exact_tie_to_N"}:
        raise ValueError("Unknown evaluation task or predeclared exact-tie policy")
    values = deepcopy(DEFAULTS[task] if thresholds is None else thresholds)
    if set(values) != set(DEFAULTS[task]):
        raise ValueError("Threshold fields must match the versioned protocol")
    for key, value in values.items():
        if (
            type(value) not in {int, float}
            or value < 0
            or (type(value) is float and not math.isfinite(value))
        ):
            raise ValueError("Thresholds must be finite nonnegative numbers")
        if key.endswith(("_gold_min", "_families_min")):
            if type(value) is not int or value < 1:
                raise ValueError("Support thresholds must be positive integers")
        elif value > 1:
            raise ValueError("Rate and margin thresholds must be in [0,1]")
    return _freeze(
        "protocol",
        {
            "schema_version": 1,
            "task": task,
            "manifest_sha256": _sha(manifest_sha256),
            "thresholds": values,
            "tie_policy": tie_policy,
            "primary_metrics": "raw",
            "prospective_test_policy": deepcopy(PROSPECTIVE_POLICY),
            "quality_approved": False,
        },
    )


def constant_predictions(record_ids, label):
    """Deterministic baseline outputs; this does not inspect gold labels."""
    ids = _strings(record_ids)
    _text(label)
    return {identity: label for identity in sorted(ids)}


def train_majority(rows):
    """Use only unique TRAIN units; ties resolve lexicographically, never via gold."""
    counts, units = Counter(), set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Train-majority baseline cannot consume non-TRAIN labels")
        unit, label = _text(row.get("unit_id")), _text(row.get("label"))
        if unit in units:
            raise ValueError("Repeated training units cannot inflate the majority prior")
        units.add(unit)
        counts[label] += 1
    if not counts:
        raise ValueError("Train-majority baseline requires training units")
    label = min(counts, key=lambda name: (-counts[name], name))
    return {"label": label, "counts": dict(sorted(counts.items())), "unit_count": len(units)}


def gate(report, baselines, protocol):
    """Return pass/insufficient/fail for supplied evidence; never approve design quality."""
    failures, insufficient = [], []
    try:
        body = protocol["protocol"]
        text = canonical_json(body)
        if (
            text != protocol["canonical_json"]
            or hashlib.sha256(text.encode()).hexdigest() != protocol["sha256"]
        ):
            raise ValueError("Protocol changed after freeze")
        task, thresholds = body["task"], body["thresholds"]
        if protocol != freeze_protocol(
            task,
            body["manifest_sha256"],
            tie_policy=body["tie_policy"],
            thresholds=thresholds,
        ):
            raise ValueError("Invalid protocol schema or metric policy")
        if report.get("task") != task or report.get("protocol_sha256") != protocol["sha256"]:
            raise ValueError("Report is not bound to the frozen protocol")
        if (
            report.get("manifest_sha256") != body["manifest_sha256"]
            or report.get("identity_bound") is not True
        ):
            raise ValueError("Report lacks complete source/model identity binding")
        _sha(report.get("gold_sha256"))
        _sha(report.get("method_identity_sha256"))
        _sha(report.get("request_set_sha256"))
        records, families = _strings(report.get("record_ids")), _strings(report.get("family_ids"))
        if report["record_ids"] != sorted(records) or report["family_ids"] != sorted(families):
            raise ValueError("Report unit/family identities must be canonical sorted unique lists")
        if len(families) < 3:
            raise ValueError("Evaluation coverage requires at least three independent families")
        coverage = report["coverage"]
        for key in ("expected", "completed", "invalid", "missing", "fallback"):
            if type(coverage.get(key)) is not int or coverage[key] < 0:
                raise ValueError("Coverage counts must be explicit nonnegative integers")
        if (
            not coverage["expected"]
            or coverage["completed"] != coverage["expected"]
            or any(coverage[key] for key in ("invalid", "missing", "fallback"))
        ):
            failures.append("Incomplete identity-bound coverage or invalid/missing/fallback output")
        if len(records) != coverage["expected"]:
            raise ValueError("Report unit identities differ from expected coverage")
        names = (
            ("pretrained", "always_body", "train_majority", "metadata")
            if task == "purpose"
            else ("pretrained", "always_A", "always_B", "majority", "order_baseline")
        )
        for name in names:
            baseline = baselines[name]
            if (
                baseline.get("identity_bound") is not True
                or baseline.get("manifest_sha256") != report["manifest_sha256"]
                or baseline.get("gold_sha256") != report["gold_sha256"]
                or baseline.get("request_set_sha256") != report["request_set_sha256"]
                or baseline.get("record_ids") != report["record_ids"]
                or baseline.get("family_ids") != report["family_ids"]
                or baseline.get("coverage") != coverage
            ):
                raise ValueError(name + " baseline has unmatched evaluation identity or coverage")
            _sha(baseline.get("method_identity_sha256"))
        raw = report["raw"]

        def rate(container, key):
            value = container[key]
            if type(value) not in {int, float} or not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"Metric {key} must be finite and in [0,1]")
            return value

        def at_least(value, target, label):
            if target > 1:
                failures.append(f"{label}: required threshold {target:.6f} exceeds the 1.0 ceiling")
            elif value + 1e-12 < target:
                failures.append(f"{label}: {value:.6f} below required {target:.6f}")

        def enough(container, minimum, families, label):
            n, f = container.get("gold_count"), container.get("family_count")
            if (
                type(n) is not int
                or type(f) is not int
                or n < 0
                or f < 0
                or f > n
                or f > len(report["family_ids"])
                or n > coverage["expected"]
            ):
                raise ValueError(f"{label} needs explicit gold/family support counts")
            if n < minimum or f < families:
                insufficient.append(
                    f"{label}: need at least {minimum} gold across {families} families"
                )

        def control_gold_count(control, condition):
            count = control.get("gold_count")
            if type(count) is not int or count < 0:
                raise ValueError(condition + " control gold count must be a nonnegative integer")
            if not count:
                insufficient.append(condition + " control has no gold")
            return count

        if task == "purpose":
            at_least(
                rate(raw, "accuracy"),
                max(rate(baselines[n], "accuracy") for n in names) + thresholds["accuracy_margin"],
                "purpose accuracy",
            )
            for key in ("macro_f1", "balanced_accuracy"):
                at_least(
                    rate(raw, key),
                    max(rate(baselines[n], key) for n in ("pretrained", "metadata"))
                    + thresholds[key + "_margin"],
                    key,
                )
            at_least(
                rate(raw, "family_macro_accuracy"),
                max(
                    rate(baselines[n], "family_macro_accuracy") for n in ("pretrained", "metadata")
                ),
                "original family macro accuracy",
            )
            support = raw["support"]
            enough(
                support,
                thresholds["support_gold_min"],
                thresholds["support_families_min"],
                "support class",
            )
            for key in ("recall", "f1"):
                at_least(
                    rate(support, key),
                    max(
                        thresholds["support_" + key + "_min"],
                        rate(baselines["pretrained"]["support"], key),
                    ),
                    "support " + key,
                )
            total_controls = 0
            for condition in ("blank", "replaced"):
                control = report["controls"][condition]
                total_controls += control_gold_count(control, condition)
                at_least(
                    rate(control, "accuracy"),
                    thresholds["control_accuracy_min"],
                    condition + " accuracy",
                )
            originals = coverage["expected"] - total_controls
            if originals <= 0 or support["gold_count"] > originals:
                raise ValueError(
                    "Purpose support/control counts exceed original evaluation coverage"
                )
        else:
            at_least(
                rate(raw, "balanced_accuracy"),
                max(rate(baselines[n], "balanced_accuracy") for n in names)
                + thresholds["balanced_accuracy_margin"],
                "critic balanced accuracy",
            )
            at_least(
                rate(raw, "accuracy"),
                max(rate(baselines[n], "accuracy") for n in names) + thresholds["accuracy_margin"],
                "critic raw accuracy",
            )
            at_least(
                rate(raw, "macro_f1"),
                rate(baselines["pretrained"], "macro_f1") + thresholds["macro_f1_margin"],
                "critic macro F1",
            )
            at_least(
                rate(raw, "swap_consistency"),
                thresholds["swap_consistency_min"],
                "physical preference swap consistency",
            )
            if type(raw.get("non_n_pair_count")) is not int or raw["non_n_pair_count"] <= 0:
                insufficient.append("No non-N physical pairs for both-orders-correct evaluation")
            at_least(
                rate(raw, "both_orders_correct"),
                thresholds["both_orders_correct_min"],
                "both orders correct",
            )
            for label in ("A", "B", "N"):
                enough(
                    raw["class_support"][label],
                    thresholds["class_gold_min"],
                    thresholds["class_families_min"],
                    "critic class " + label,
                )
            total_controls = 0
            for condition in ("neither", "source_ablated"):
                control = report["controls"][condition]
                total_controls += control_gold_count(control, condition)
                at_least(
                    rate(control, "expected_n_accuracy"),
                    thresholds["control_accuracy_min"],
                    condition + " expected N accuracy",
                )
            original, controls = report["mix"]["original_count"], report["mix"]["control_count"]
            if (
                type(original) is not int
                or type(controls) is not int
                or original <= 0
                or controls < 0
            ):
                raise ValueError("Critic mix needs positive originals and nonnegative controls")
            if original + controls != coverage["expected"]:
                raise ValueError("Critic original/control mix differs from expected coverage")
            class_counts = {label: raw["class_support"][label]["gold_count"] for label in "ABN"}
            if sum(class_counts.values()) != original or total_controls != controls:
                raise ValueError(
                    "Critic class/control gold counts differ from declared evaluation mix"
                )
            if (
                class_counts["A"] != class_counts["B"]
                or class_counts["N"] % 2
                or raw["non_n_pair_count"] != class_counts["A"]
            ):
                raise ValueError(
                    "Critic gold support is inconsistent with matched physical swap pairs"
                )
            if controls / (original + controls) > thresholds["control_fraction_max"]:
                failures.append("Artificial controls exceed the predeclared 25% evaluation mix cap")
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        failures.append(str(error))
    status = "fail" if failures else "insufficient" if insufficient else "pass"
    return {"status": status, "reasons": failures + insufficient, "quality_approved": False}
