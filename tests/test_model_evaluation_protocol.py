"""Protocol gates use synthetic metrics only; no gold, models or source files."""

import builtins
import hashlib
from copy import deepcopy

import pytest

from tools import model_evaluation_protocol as module


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def registry():
    return {
        key: [] for key in ("source_ids", "family_ids", "original_sha256s", "primary_asset_sha256s")
    }


def family(identity, split):
    return {
        "family_id": identity,
        "split": split,
        "asset_family_ids": ["campaign-" + identity],
        "common_asset_ids": ["shared-logo"],
        "sources": [
            {
                "source_id": "source-" + identity,
                "original_sha256": sha("original-" + identity),
                "json_sha256": sha("json-" + identity),
                "render_sha256": sha("render-" + identity),
                "primary_asset_sha256s": [sha("primary-" + identity)],
                "request_path": "requests/" + identity + ".json",
                "license_id": "CC0-1.0",
                "license_reference": "source/license-" + identity,
                "provenance": {"origin": "licensed_source", "reference": "source/" + identity},
            }
        ],
    }


@pytest.fixture
def inventory():
    families = [
        family(f"new-{split}-{number}", split)
        for split in ("tuning_validation", "prospective_test")
        for number in range(3)
    ]
    families += [family("rare-" + str(n), "train") for n in range(3)]
    families += [family("old-dev-" + str(n), "legacy_development") for n in range(3)]
    trained, legacy = registry(), registry()
    for entry in families[6:9]:
        trained["family_ids"].append(entry["family_id"])
        trained["source_ids"].append(entry["sources"][0]["source_id"])
        trained["original_sha256s"].append(entry["sources"][0]["original_sha256"])
        trained["primary_asset_sha256s"].extend(entry["sources"][0]["primary_asset_sha256s"])
    for entry in families[9:]:
        legacy["family_ids"].append(entry["family_id"])
        legacy["source_ids"].append(entry["sources"][0]["source_id"])
        legacy["original_sha256s"].append(entry["sources"][0]["original_sha256"])
        legacy["primary_asset_sha256s"].extend(entry["sources"][0]["primary_asset_sha256s"])
    reserves = {
        identity: {
            "original_sha256": sha(identity + "original"),
            "json_sha256": sha(identity + "json"),
        }
        for identity in (*sorted(module.MANUAL_IDS), "current-catalogue-reserve")
    }
    return {
        "manifest": {
            "schema_version": 1,
            "families": families,
            "common_asset_ids": {"shared-logo": {"kind": "logo", "sha256": sha("logo")}},
            "local_gold_refs": [{"path": "sealed/gold.json", "sha256": sha("sealed-gold")}],
        },
        "arguments": {
            "denylist": sorted(reserves),
            "catalogue_reserves": reserves,
            "previous_gradient_registry": trained,
            "legacy_development_registry": legacy,
        },
    }


def freeze(inventory):
    return module.freeze_manifest(inventory["manifest"], **inventory["arguments"])


def bind_baselines(report, baselines):
    report.update(
        request_set_sha256=sha("exact-request-identity-set"),
        record_ids=sorted("unit-" + str(n) for n in range(report["coverage"]["expected"])),
        family_ids=["family-1", "family-2", "family-3"],
    )
    for name, baseline in baselines.items():
        baseline.update(
            manifest_sha256=report["manifest_sha256"],
            gold_sha256=report["gold_sha256"],
            request_set_sha256=report["request_set_sha256"],
            record_ids=deepcopy(report["record_ids"]),
            family_ids=deepcopy(report["family_ids"]),
            identity_bound=True,
            method_identity_sha256=sha(name),
            coverage=deepcopy(report["coverage"]),
        )


def test_manifest_freezes_exact_canonical_bytes_and_explicit_common_logo(inventory, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No source files or gold may be opened")

    monkeypatch.setattr(builtins, "open", forbidden)
    frozen = freeze(inventory)
    assert frozen["sha256"] == sha(frozen["canonical_json"])
    assert frozen["canonical_json"] == module.canonical_json(frozen["manifest"])
    assert frozen["manifest"]["common_asset_ids"] == inventory["manifest"]["common_asset_ids"]
    assert len({row["family_id"] for row in frozen["manifest"]["families"]}) == 12
    inventory["manifest"]["families"][0]["family_id"] = "later change"
    assert "later change" not in frozen["canonical_json"]


def test_reserve_and_manual_denylist_must_be_explicit_and_complete(inventory):
    with pytest.raises(TypeError):
        module.freeze_manifest(inventory["manifest"])
    inventory["arguments"]["denylist"].remove("current-catalogue-reserve")
    with pytest.raises(ValueError, match="denylist"):
        freeze(inventory)


@pytest.mark.parametrize("field", ["source_id", "original_sha256", "json_sha256"])
def test_renamed_reserved_sources_cannot_bypass_hash_or_id_denial(inventory, field):
    source = inventory["manifest"]["families"][0]["sources"][0]
    reserved = "current-catalogue-reserve"
    source[field] = (
        reserved
        if field == "source_id"
        else inventory["arguments"]["catalogue_reserves"][reserved][field]
    )
    with pytest.raises(ValueError, match="reserved-alias"):
        freeze(inventory)


@pytest.mark.parametrize("alias", ["original_sha256", "json_sha256", "primary_asset", "campaign"])
def test_asset_or_source_aliases_cannot_inflate_distinct_family_counts(inventory, alias):
    first, second = inventory["manifest"]["families"][:2]
    if alias == "campaign":
        second["asset_family_ids"] = first["asset_family_ids"][:]
    elif alias == "primary_asset":
        second["sources"][0]["primary_asset_sha256s"] = first["sources"][0][
            "primary_asset_sha256s"
        ][:]
    else:
        second["sources"][0][alias] = first["sources"][0][alias]
    with pytest.raises(ValueError, match="alias"):
        freeze(inventory)


@pytest.mark.parametrize(
    "registry_kind", ["previous_gradient_registry", "legacy_development_registry"]
)
@pytest.mark.parametrize("alias", ["source_id", "family_id", "original", "primary_asset"])
def test_trained_and_legacy_sources_cannot_be_renamed_into_prospective_test(
    inventory, registry_kind, alias
):
    entry = inventory["manifest"]["families"][3]
    trusted = inventory["arguments"][registry_kind]
    if alias == "source_id":
        entry["sources"][0]["source_id"] = trusted["source_ids"][0]
    elif alias == "family_id":
        entry["asset_family_ids"] = [trusted["family_ids"][0]]
    elif alias == "original":
        entry["sources"][0]["json_sha256"] = trusted["original_sha256s"][0]
    else:
        entry["sources"][0]["primary_asset_sha256s"] = trusted["primary_asset_sha256s"][:1]
    with pytest.raises(ValueError, match="TRAIN|legacy_development"):
        freeze(inventory)


def test_common_asset_exception_cannot_hide_known_primary_training_artwork(inventory):
    inventory["manifest"]["common_asset_ids"]["shared-logo"]["sha256"] = inventory["arguments"][
        "previous_gradient_registry"
    ]["primary_asset_sha256s"][0]
    with pytest.raises(ValueError, match="override"):
        freeze(inventory)


@pytest.mark.parametrize("split", ["tuning_validation", "prospective_test"])
def test_evaluation_splits_need_three_independent_families(inventory, split):
    families = inventory["manifest"]["families"]
    identity = next(row for row in families if row["split"] == split)
    families.remove(identity)
    with pytest.raises(ValueError, match="three families"):
        freeze(inventory)


@pytest.mark.parametrize(
    "bad", ["https://example.com/gold", "../gold.json", "requests/new-tuning_validation-0.json"]
)
def test_gold_is_a_separate_local_reference_never_a_model_request(inventory, bad):
    inventory["manifest"]["local_gold_refs"][0]["path"] = bad
    with pytest.raises(ValueError, match="local|separate"):
        freeze(inventory)


@pytest.fixture
def purpose():
    protocol = module.freeze_protocol("purpose", sha("manifest"))
    report = {
        "task": "purpose",
        "protocol_sha256": protocol["sha256"],
        "manifest_sha256": sha("manifest"),
        "gold_sha256": sha("gold"),
        "method_identity_sha256": sha("model"),
        "identity_bound": True,
        "coverage": {"expected": 120, "completed": 120, "invalid": 0, "missing": 0, "fallback": 0},
        "raw": {
            "accuracy": 0.8,
            "macro_f1": 0.8,
            "balanced_accuracy": 0.8,
            "family_macro_accuracy": 0.8,
            "support": {"gold_count": 6, "family_count": 2, "recall": 0.75, "f1": 0.75},
        },
        "controls": {
            "blank": {"gold_count": 10, "accuracy": 0.8},
            "replaced": {"gold_count": 10, "accuracy": 0.9},
        },
    }
    baselines = {
        "pretrained": {
            "accuracy": 0.6,
            "macro_f1": 0.6,
            "balanced_accuracy": 0.6,
            "family_macro_accuracy": 0.6,
            "support": {"recall": 0.6, "f1": 0.6},
        },
        "always_body": {"accuracy": 0.55},
        "train_majority": {"accuracy": 0.6},
        "metadata": {
            "accuracy": 0.65,
            "macro_f1": 0.65,
            "balanced_accuracy": 0.65,
            "family_macro_accuracy": 0.62,
        },
    }
    bind_baselines(report, baselines)
    return report, baselines, protocol


@pytest.fixture
def critic():
    protocol = module.freeze_protocol("critic", sha("manifest"))
    report = {
        "task": "critic",
        "protocol_sha256": protocol["sha256"],
        "manifest_sha256": sha("manifest"),
        "gold_sha256": sha("gold"),
        "method_identity_sha256": sha("model"),
        "identity_bound": True,
        "coverage": {"expected": 120, "completed": 120, "invalid": 0, "missing": 0, "fallback": 0},
        "raw": {
            "accuracy": 0.8,
            "balanced_accuracy": 0.8,
            "macro_f1": 0.8,
            "swap_consistency": 0.95,
            "both_orders_correct": 0.8,
            "non_n_pair_count": 30,
            "class_support": {label: {"gold_count": 30, "family_count": 3} for label in "ABN"},
        },
        "controls": {
            "neither": {"gold_count": 15, "expected_n_accuracy": 0.8},
            "source_ablated": {"gold_count": 15, "expected_n_accuracy": 0.9},
        },
        "mix": {"original_count": 90, "control_count": 30},
    }
    baselines = {
        "pretrained": {"accuracy": 0.6, "balanced_accuracy": 0.6, "macro_f1": 0.6},
        "always_A": {"accuracy": 0.34, "balanced_accuracy": 1 / 3},
        "always_B": {"accuracy": 0.33, "balanced_accuracy": 1 / 3},
        "majority": {"accuracy": 0.4, "balanced_accuracy": 0.4},
        "order_baseline": {"accuracy": 0.45, "balanced_accuracy": 0.45},
    }
    bind_baselines(report, baselines)
    return report, baselines, protocol


@pytest.mark.parametrize("fixture", ["purpose", "critic"])
def test_genuine_metric_gate_pass_is_not_quality_approval(fixture, request):
    report, baselines, protocol = request.getfixturevalue(fixture)
    result = module.gate(report, baselines, protocol)
    assert result == {"status": "pass", "reasons": [], "quality_approved": False}


def test_100_percent_valid_output_cannot_hide_purpose_regression(purpose):
    report, baselines, protocol = purpose
    report["raw"].update(accuracy=0.59, macro_f1=0.58)
    result = module.gate(report, baselines, protocol)
    assert result["status"] == "fail"
    assert any("purpose accuracy" in reason for reason in result["reasons"])


def test_purpose_must_beat_strongest_metadata_and_majority_baselines(purpose):
    report, baselines, protocol = purpose
    baselines["metadata"]["accuracy"] = 0.79
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize("field,value", [("gold_count", 5), ("family_count", 1)])
def test_purpose_support_class_with_too_few_independent_labels_is_insufficient(
    purpose, field, value
):
    report, baselines, protocol = purpose
    report["raw"]["support"][field] = value
    assert module.gate(report, baselines, protocol)["status"] == "insufficient"


@pytest.mark.parametrize("field,value", [("recall", 0.49), ("f1", 0.49)])
def test_support_class_floor_and_pretrained_nonregression_are_required(purpose, field, value):
    report, baselines, protocol = purpose
    report["raw"]["support"][field] = value
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize("fixture", ["purpose", "critic"])
@pytest.mark.parametrize("failure", ["invalid", "missing", "fallback", "identity"])
def test_incomplete_or_unbound_coverage_never_passes(fixture, failure, request):
    report, baselines, protocol = request.getfixturevalue(fixture)
    if failure == "identity":
        report["identity_bound"] = False
    else:
        report["coverage"][failure] = 1
    assert module.gate(report, baselines, protocol)["status"] == "fail"


def test_always_n_cannot_pass_just_because_it_is_swap_consistent(critic):
    report, baselines, protocol = critic
    report["raw"].update(
        swap_consistency=1,
        both_orders_correct=0,
        accuracy=1 / 3,
        balanced_accuracy=1 / 3,
        macro_f1=1 / 6,
    )
    result = module.gate(report, baselines, protocol)
    assert result["status"] == "fail"
    assert any("both orders" in reason for reason in result["reasons"])


def test_always_b_constant_and_order_bias_fail_the_critic_gate(critic):
    report, baselines, protocol = critic
    report["raw"].update(
        accuracy=0.5, balanced_accuracy=0.5, macro_f1=0.4, swap_consistency=0, both_orders_correct=0
    )
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize("label", list("ABN"))
def test_critic_class_support_is_insufficient_below_three_families(critic, label):
    report, baselines, protocol = critic
    report["raw"]["class_support"][label]["family_count"] = 2
    assert module.gate(report, baselines, protocol)["status"] == "insufficient"


def test_artificial_control_success_cannot_swamp_critic_originals(critic):
    report, baselines, protocol = critic
    report["mix"] = {"original_count": 89, "control_count": 31}
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize("condition", ["neither", "source_ablated"])
def test_critic_expected_n_controls_have_separate_accuracy_floors(critic, condition):
    report, baselines, protocol = critic
    report["controls"][condition]["expected_n_accuracy"] = 0.79
    assert module.gate(report, baselines, protocol)["status"] == "fail"


def test_predeclared_margin_above_perfect_score_is_an_explicit_failure(purpose):
    report, baselines, protocol = purpose
    report["raw"]["accuracy"] = 1
    baselines["metadata"]["accuracy"] = 0.99
    result = module.gate(report, baselines, protocol)
    assert result["status"] == "fail"
    assert any("ceiling" in reason for reason in result["reasons"])


def test_changed_protocol_and_retroactive_tie_metrics_cannot_replace_raw_results(critic):
    report, baselines, protocol = critic
    report["raw"]["both_orders_correct"] = 0
    report["policy_metrics"] = {"accuracy": 1, "balanced_accuracy": 1, "both_orders_correct": 1}
    assert module.gate(report, baselines, protocol)["status"] == "fail"
    frozen = module.freeze_protocol("critic", sha("manifest"), tie_policy="exact_tie_to_N")
    assert frozen["protocol"]["tie_policy"] == "exact_tie_to_N"
    protocol["protocol"]["thresholds"]["both_orders_correct_min"] = 0
    assert any(
        "changed after freeze" in reason
        for reason in module.gate(report, baselines, protocol)["reasons"]
    )


def test_train_majority_and_constant_predictions_never_use_nontrain_gold():
    rows = [
        {"unit_id": str(n), "split": "train", "label": label}
        for n, label in enumerate(("body", "body", "support"))
    ]
    assert module.train_majority(rows)["label"] == "body"
    assert module.constant_predictions(["b", "a"], "body") == {"a": "body", "b": "body"}
    with pytest.raises(ValueError, match="non-TRAIN"):
        module.train_majority(
            rows + [{"unit_id": "test", "split": "prospective_test", "label": "A"}]
        )
    with pytest.raises(ValueError, match="Repeated"):
        module.train_majority(rows + [deepcopy(rows[0])])


@pytest.mark.parametrize(
    "field",
    ["manifest_sha256", "gold_sha256", "method_identity_sha256", "identity_bound", "coverage"],
)
def test_unbound_or_incomplete_baseline_cannot_make_candidate_pass(purpose, field):
    report, baselines, protocol = purpose
    baseline = baselines["metadata"]
    if field == "coverage":
        baseline[field]["completed"] -= 1
    elif field == "identity_bound":
        baseline[field] = False
    elif field == "method_identity_sha256":
        baseline[field] = "not-a-hash"
    else:
        baseline[field] = sha("different evaluation")
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize("location", ["manifest", "family", "source"])
def test_labels_cannot_be_embedded_in_metadata_schema(inventory, location):
    target = inventory["manifest"]
    if location in {"family", "source"}:
        target = target["families"][0]
    if location == "source":
        target = target["sources"][0]
    target["gold_labels"] = {"headline": "support"}
    with pytest.raises(ValueError):
        freeze(inventory)


def test_self_hashed_invalid_protocol_is_still_rejected(purpose):
    report, baselines, protocol = purpose
    protocol["protocol"]["thresholds"]["accuracy_margin"] = -1
    protocol = module._freeze("protocol", protocol["protocol"])
    report["protocol_sha256"] = protocol["sha256"]
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize("field", ["record_ids", "family_ids", "request_set_sha256"])
def test_baseline_evaluation_units_and_request_bytes_must_match(purpose, field):
    report, baselines, protocol = purpose
    if field.endswith("ids"):
        baselines["pretrained"][field][0] = "different-identity"
    else:
        baselines["pretrained"][field] = sha("different-request-bytes")
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize(
    "path", ["sealed/GOLD.json", "sealed/gold.json.", "sealed/gold.json ", "sealed/CON.json"]
)
def test_portable_filesystem_aliases_cannot_expose_gold_as_request(inventory, path):
    inventory["manifest"]["families"][0]["sources"][0]["request_path"] = path
    with pytest.raises(ValueError, match="separate|aliases"):
        freeze(inventory)


def test_prospective_one_shot_feedback_rule_is_frozen(purpose):
    _, _, protocol = purpose
    assert protocol["protocol"]["prospective_test_policy"] == {
        "candidate_code_and_tie_policy_frozen_before_feedback": True,
        "failed_families_retired_to_legacy_before_feedback_use": True,
        "new_prospective_families_required_after_failure": True,
    }


def test_critic_raw_accuracy_must_also_beat_pretrained_and_order_baseline(critic):
    report, baselines, protocol = critic
    baselines["pretrained"]["accuracy"] = 0.79
    assert module.gate(report, baselines, protocol)["status"] == "fail"
    baselines["pretrained"]["accuracy"] = 0.6
    baselines["order_baseline"]["accuracy"] = 0.79
    assert module.gate(report, baselines, protocol)["status"] == "fail"


@pytest.mark.parametrize(
    "change", ["class_total", "control_total", "pair_count", "unpaired_labels"]
)
def test_critic_count_conservation_prevents_invented_gold_support(critic, change):
    report, baselines, protocol = critic
    if change == "class_total":
        report["raw"]["class_support"]["N"]["gold_count"] = 28
    elif change == "control_total":
        report["controls"]["neither"]["gold_count"] = 16
    elif change == "pair_count":
        report["raw"]["non_n_pair_count"] = 31
    else:
        report["raw"]["class_support"]["A"]["gold_count"] = 31
        report["raw"]["class_support"]["B"]["gold_count"] = 29
    assert module.gate(report, baselines, protocol)["status"] == "fail"


def test_purpose_gold_support_cannot_exceed_covered_originals(purpose):
    report, baselines, protocol = purpose
    report["raw"]["support"]["gold_count"] = 101
    assert module.gate(report, baselines, protocol)["status"] == "fail"


def test_huge_integer_metric_fails_without_numeric_overflow(purpose):
    report, baselines, protocol = purpose
    report["raw"]["accuracy"] = 10**400
    assert module.gate(report, baselines, protocol)["status"] == "fail"
    thresholds = deepcopy(module.DEFAULTS["purpose"])
    thresholds["accuracy_margin"] = 10**400
    with pytest.raises(ValueError, match="in \\[0,1\\]"):
        module.freeze_protocol("purpose", sha("manifest"), thresholds=thresholds)
