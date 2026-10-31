"""Offline purpose scoring uses synthetic local gold and never a model worker."""

import builtins
import hashlib
import json
import math
import struct
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools import score_visual_purpose as scorer


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction(request, label):
    scores = {name: 2.0 if name == label else 0.0 for name in scorer.PURPOSE_CLASSES}
    total = sum(math.exp(value) for value in scores.values())
    return {
        "record_id": request["record_id"],
        "family_id": request["family_id"],
        "control": request["control"],
        "label": None,
        "prediction": label,
        "closed_scores": scores,
        "closed_probabilities": {key: math.exp(value) / total for key, value in scores.items()},
        "raw_next_token_id": 100,
        "raw_next_token": label,
        "raw_next_token_is_allowed": True,
        "input_token_count": 40,
        "image_grid_thw": [[1, 2, 2], [1, 2, 2]],
        "input_image_sha256": [image["sha256"] for image in request["images"]],
    }


@pytest.fixture
def frozen(tmp_path):
    paths = {
        name: tmp_path / (name + ".json")
        for name in (
            "request",
            "predictions",
            "receipt",
            "gold",
            "freeze",
        )
    }
    labels = ("body", "headline", "background", "unknown", "unknown", "body")
    controls = ("original", "original", "original", "both_blank", "source_replaced", "swapped")
    requests, gold = [], []
    for index, (label, control) in enumerate(zip(labels, controls, strict=True)):
        row = {
            "record_id": f"record-{index}",
            "family_id": f"family-{index % 2}",
            "control": control,
            "prompt": "Classify the target from source and context.",
            "images": [
                {
                    "path": f"image-{index}-{role}.png",
                    "sha256": hashlib.sha256(f"synthetic-{index}-{role}".encode()).hexdigest(),
                }
                for role in ("source", "context")
            ],
        }
        requests.append(row)
        annotation = {"record_id": row["record_id"], "label": label, "label_origin": "synthetic"}
        if index >= 3:
            annotation["parent_record_id"] = "record-1" if index == 4 else "record-0"
        gold.append(annotation)
    request = {"protocol": scorer.REQUEST_PROTOCOL, "task": "purpose", "records": requests}
    write(paths["request"], request)
    request_sha = sha(paths["request"])
    write(
        paths["gold"],
        {
            "protocol": scorer.GOLD_PROTOCOL,
            "request_sha256": request_sha,
            "label_scope": "synthetic_local_test_only",
            "records": gold,
        },
    )
    freeze = {
        "protocol": scorer.FREEZE_PROTOCOL,
        "request_sha256": request_sha,
        "adapter_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
        "gold_sha256": sha(paths["gold"]),
    }
    write(paths["freeze"], freeze)
    rows = [prediction(row, label) for row, label in zip(requests, labels, strict=True)]
    receipt = {
        "execution_success": True,
        "task": "purpose",
        "request_sha256": request_sha,
        "adapter_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
    }
    fixture = SimpleNamespace(paths=paths, requests=requests, rows=rows, receipt=receipt, gold=gold)
    publish(fixture)
    return fixture


def publish(frozen):
    write(
        frozen.paths["predictions"],
        {"scoring": scorer.SCORING, "metrics": None, "predictions": frozen.rows},
    )
    frozen.receipt["prediction_file_sha256"] = sha(frozen.paths["predictions"])
    write(frozen.paths["receipt"], frozen.receipt)


def score(frozen):
    return scorer.score(
        *(
            frozen.paths[name]
            for name in (
                "request",
                "predictions",
                "receipt",
                "gold",
                "freeze",
            )
        )
    )


def test_perfect_predictions_report_support_confusion_controls_and_no_quality_approval(frozen):
    report = score(frozen)
    metrics = report["metrics"]
    assert metrics["gold_count"] == metrics["valid_count"] == 6
    assert metrics["accuracy_all_gold"] == metrics["macro_f1_supported"] == 1
    assert metrics["macro_recall_supported"] == 1
    assert metrics["supported_class_count"] == 4
    assert metrics["per_class"]["accent"] == {
        "support": 0,
        "predicted": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    assert metrics["confusion"]["body"]["body"] == 2
    assert metrics["abstain_count"] == 2 and metrics["qualified_prediction_count"] == 4
    assert report["per_control"]["both_blank"]["accuracy_all_gold"] == 1
    assert report["per_control"]["both_blank"]["qualified_prediction_count"] == 0
    assert report["paired_controls"]["swapped"]["prediction_consistency"] == 1
    assert report["paired_controls"]["both_blank"]["prediction_consistency"] == 0
    assert report["paired_controls"]["source_replaced"]["mean_probability_l1"] > 0
    assert not report["inference_executed"] and not report["quality_approved"]
    assert not report["promotion_eligible"] and not report["cold_load_independently_verified"]


def test_constant_role_prediction_is_exposed_against_majority_and_macro_metrics(frozen):
    frozen.rows = [prediction(row, "body") for row in frozen.requests]
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["prediction_histogram"] == {"body": 6}
    assert metrics["dominant_prediction_fraction"] == 1
    assert (
        metrics["accuracy_all_gold"] == metrics["majority_baseline"]["accuracy_all_gold"] == 1 / 3
    )
    assert metrics["macro_f1_supported"] == 0.125
    assert metrics["macro_recall_supported"] == 0.25


def test_missing_and_nonfinite_rows_remain_failures_in_all_gold_denominators(frozen):
    frozen.rows.pop(1)
    frozen.rows[1]["closed_scores"]["body"] = float("nan")
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["gold_count"] == 6
    assert metrics["missing_count"] == metrics["invalid_count"] == 1
    assert metrics["accuracy_all_gold"] == 4 / 6
    assert metrics["confusion"]["headline"][scorer.MISSING] == 1
    assert metrics["confusion"]["background"][scorer.INVALID] == 1
    assert metrics["per_class"]["headline"]["false_negative"] == 1
    assert metrics["per_class"]["background"]["recall"] == 0


def test_raw_disallowed_token_does_not_turn_closed_class_scores_into_generated_correctness(frozen):
    frozen.rows[0].update(raw_next_token="An explanation", raw_next_token_is_allowed=False)
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["valid_count"] == 6 and metrics["accuracy_all_gold"] == 1
    assert (
        metrics["raw_allowed_rate_valid"] == metrics["strict_raw_token_accuracy_all_gold"] == 5 / 6
    )


@pytest.mark.parametrize(
    "field",
    [
        "request_sha256",
        "prediction_file_sha256",
        "adapter_sha256",
        "adapter_config_sha256",
        "execution_success",
    ],
)
def test_stale_or_unsuccessful_receipts_are_rejected(frozen, field):
    frozen.receipt[field] = False if field == "execution_success" else "0" * 64
    write(frozen.paths["receipt"], frozen.receipt)
    with pytest.raises(ValueError, match="receipt|freeze"):
        score(frozen)


@pytest.mark.parametrize(
    "mutation", ["family", "control", "images", "classes", "argmax", "probabilities"]
)
def test_malformed_or_rebound_prediction_rows_are_invalid_not_dropped(frozen, mutation):
    row = frozen.rows[0]
    if mutation == "family":
        row["family_id"] = "other-family"
    elif mutation == "control":
        row["control"] = "source_replaced"
    elif mutation == "images":
        row["input_image_sha256"].reverse()
    elif mutation == "classes":
        row["closed_scores"].pop("body")
    elif mutation == "argmax":
        row["prediction"] = "headline"
    else:
        row["closed_probabilities"]["body"] = 0
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["invalid_count"] == 1 and metrics["gold_count"] == 6
    assert metrics["accuracy_all_gold"] == 5 / 6


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_prediction",
        "unknown_prediction",
        "duplicate_gold",
        "unknown_gold",
        "unknown_label",
        "stale_gold",
        "request_gold",
    ],
)
def test_unknown_duplicate_or_leaked_answer_identities_are_rejected(frozen, mutation):
    gold = json.loads(frozen.paths["gold"].read_text())
    if mutation == "duplicate_prediction":
        frozen.rows.append(deepcopy(frozen.rows[0]))
        publish(frozen)
    elif mutation == "unknown_prediction":
        frozen.rows[0]["record_id"] = "unrequested"
        publish(frozen)
    elif mutation == "duplicate_gold":
        gold["records"].append(deepcopy(gold["records"][0]))
    elif mutation == "unknown_gold":
        gold["records"][0]["record_id"] = "unrequested"
    elif mutation == "unknown_label":
        gold["records"][0]["label"] = "invented-role"
    elif mutation == "stale_gold":
        gold["request_sha256"] = "0" * 64
    else:
        request = json.loads(frozen.paths["request"].read_text())
        request["records"][0]["label"] = "body"
        write(frozen.paths["request"], request)
    write(frozen.paths["gold"], gold)
    freeze = json.loads(frozen.paths["freeze"].read_text())
    freeze["gold_sha256"] = sha(frozen.paths["gold"])
    write(frozen.paths["freeze"], freeze)
    with pytest.raises(ValueError):
        score(frozen)


def test_scoring_and_cli_import_no_gpu_or_worker_modules(frozen, monkeypatch, capsys):
    import_module = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "accelerate"}
        assert not name.endswith(("worker", "train_sft"))
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    out = frozen.paths["request"].parent / "report.json"
    args = [part for name, path in frozen.paths.items() for part in ("--" + name, str(path))]
    assert scorer.main([*args, "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["bindings"]["prediction_file_sha256"] == sha(frozen.paths["predictions"])
    assert report["metrics"]["accuracy_all_gold"] == 1


@pytest.mark.parametrize("raw", ["An explanation", "headline"])
def test_allowed_raw_token_must_be_an_allowed_maximal_class(frozen, raw):
    frozen.rows[0]["raw_next_token"] = raw
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["invalid_count"] == 1 and metrics["gold_count"] == 6
    assert metrics["raw_allowed_rate_valid"] == 1
    assert metrics["strict_raw_token_accuracy_all_gold"] == 5 / 6


def test_true_raw_argmax_tie_uses_its_decoded_class_for_strict_accuracy(frozen):
    row = frozen.rows[0]
    row["closed_scores"]["background"] = row["closed_scores"]["body"]
    total = sum(math.exp(value) for value in row["closed_scores"].values())
    row["closed_probabilities"] = {
        key: math.exp(value) / total for key, value in row["closed_scores"].items()
    }
    row["prediction"] = "background"  # Fixed class ordering wins this closed-class tie.
    row["raw_next_token"] = " body "  # Global token ordering may instead select body.
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["valid_count"] == 6
    assert metrics["accuracy_all_gold"] == 5 / 6
    assert metrics["strict_raw_token_accuracy_all_gold"] == 1


def test_huge_json_integer_is_an_invalid_row_instead_of_an_overflow_crash(frozen):
    frozen.rows[0]["closed_scores"]["body"] = 10**400
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["invalid_count"] == 1 and metrics["gold_count"] == 6
    assert metrics["accuracy_all_gold"] == 5 / 6


def test_finite_extreme_integer_logits_do_not_overflow_during_softmax_validation(frozen):
    row = frozen.rows[0]
    row["closed_scores"] = {
        key: 10**308 if key == "body" else -(10**308) for key in scorer.PURPOSE_CLASSES
    }
    row["closed_probabilities"] = {key: float(key == "body") for key in scorer.PURPOSE_CLASSES}
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["valid_count"] == 6 and metrics["accuracy_all_gold"] == 1


def test_float32_probability_rounding_ties_preserve_the_unique_logit_winner(frozen):
    row = frozen.rows[0]
    row["closed_scores"] = {key: 1e-8 if key == "body" else 0 for key in scorer.PURPOSE_CLASSES}
    rounded = struct.unpack("f", struct.pack("f", 0.1))[0]
    row["closed_probabilities"] = dict.fromkeys(scorer.PURPOSE_CLASSES, rounded)
    publish(frozen)
    metrics = score(frozen)["metrics"]
    assert metrics["valid_count"] == 6 and metrics["accuracy_all_gold"] == 1


def test_changed_gold_is_hash_rejected_before_parsing_its_answers(frozen, monkeypatch):
    annotation = json.loads(frozen.paths["gold"].read_text())
    annotation["records"][0]["label"] = "visual"
    write(frozen.paths["gold"], annotation)
    parse = scorer.json.loads

    def guarded(text, *args, **kwargs):
        assert scorer.GOLD_PROTOCOL not in text, "Changed sealed gold was parsed"
        return parse(text, *args, **kwargs)

    monkeypatch.setattr(scorer.json, "loads", guarded)
    with pytest.raises(ValueError, match="SHA-256"):
        score(frozen)


def bind_changed_request(frozen, request):
    write(frozen.paths["request"], request)
    request_sha = sha(frozen.paths["request"])
    gold = json.loads(frozen.paths["gold"].read_text())
    gold["request_sha256"] = request_sha
    write(frozen.paths["gold"], gold)
    freeze = json.loads(frozen.paths["freeze"].read_text())
    freeze.update(request_sha256=request_sha, gold_sha256=sha(frozen.paths["gold"]))
    write(frozen.paths["freeze"], freeze)
    frozen.receipt["request_sha256"] = request_sha
    publish(frozen)


@pytest.mark.parametrize("declared", [None, 2, 3])
def test_exact_declared_view_count_and_order_are_bound(frozen, declared):
    request = json.loads(frozen.paths["request"].read_text())
    if declared is not None:
        request["purpose_views"] = declared
    if declared == 3:
        for row, prediction in zip(request["records"], frozen.rows, strict=True):
            image = {"path": "isolated.png", "sha256": "c" * 64}
            row["images"].append(image)
            prediction["input_image_sha256"].append(image["sha256"])
    bind_changed_request(frozen, request)
    report = score(frozen)
    assert report["purpose_views"] == (declared or 2)
    assert report["metrics"]["accuracy_all_gold"] == 1
    if declared == 3:
        frozen.rows[0]["input_image_sha256"][-2:] = reversed(
            frozen.rows[0]["input_image_sha256"][-2:]
        )
        publish(frozen)
        assert score(frozen)["metrics"]["invalid_count"] == 1


@pytest.mark.parametrize("declared,actual", [(None, 3), (2, 3), (3, 2), (True, 2), (4, 2)])
def test_undeclared_or_mismatched_three_view_requests_fail(frozen, declared, actual):
    request = json.loads(frozen.paths["request"].read_text())
    if declared is not None:
        request["purpose_views"] = declared
    if actual == 3:
        request["records"][0]["images"].append({"path": "isolated.png", "sha256": "c" * 64})
    bind_changed_request(frozen, request)
    with pytest.raises(ValueError, match="views|ordered images"):
        score(frozen)


def test_invalid_receipt_does_not_open_local_gold(frozen, monkeypatch):
    frozen.receipt["execution_success"] = False
    write(frozen.paths["receipt"], frozen.receipt)
    read = scorer._json

    def guarded(path, **kwargs):
        assert path != frozen.paths["gold"], "Gold was opened before receipt validation"
        return read(path, **kwargs)

    monkeypatch.setattr(scorer, "_json", guarded)
    with pytest.raises(ValueError, match="receipt"):
        score(frozen)


@pytest.mark.parametrize("content", ['{"task":"purpose","task":"purpose"}', '{"task":NaN}'])
def test_duplicate_or_nonfinite_identity_metadata_is_rejected(frozen, content):
    frozen.paths["receipt"].write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate|finite"):
        score(frozen)
