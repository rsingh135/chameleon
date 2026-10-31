"""A TRAIN-only, metadata-only control cannot learn image interventions."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.visual_finetune.data import build_purpose_prompt
from tools import purpose_metadata_baseline as baseline


def row(identity, label="body", *, kind="text", box=None, text="Details", **extra):
    context = {
        "bbox_normalized": box or [0.1, 0.2, 0.5, 0.1],
        "type": kind,
        "text_data": text,
        "paint_order_index": 1,
        "element_count": 3,
    }
    return {
        "record_id": identity,
        "family_id": "training-family",
        "split": "train",
        "control": "original",
        "label": label,
        "prompt": build_purpose_prompt(context),
        "images": [
            {"path": "missing-source.png", "sha256": "1" * 64},
            {"path": "missing-context.png", "sha256": "2" * 64},
        ],
        **extra,
    }


def request(*rows, views=2):
    stripped = [{k: v for k, v in value.items() if k not in {"label", "split"}} for value in rows]
    return {
        "protocol": baseline.REQUEST_PROTOCOL,
        "task": "purpose",
        "purpose_views": views,
        "records": stripped,
    }


def predict(fitted, value):
    return baseline.predict(fitted, value, request_sha256=baseline.digest(value))


def training():
    return [
        row("a", "background", kind="shape", box=[0, 0, 1, 1], text=""),
        row("b", "body", text="A paragraph of details"),
    ]


def test_separable_classes_train_only_scaler_centroids_and_bindings():
    rows = training()
    fitted = baseline.fit(rows)
    before = deepcopy(fitted)
    result = predict(fitted, request(*rows))
    assert [r["prediction"] for r in result["predictions"]] == ["background", "body"]
    assert fitted == before
    assert fitted["class_counts"]["unknown"] == 0 and "unknown" not in fitted["centroids"]
    assert fitted["train_family_ids"] == ["training-family"]
    assert fitted["train_records_sha256"] == baseline.digest(rows)
    assert fitted["feature_version"] == baseline.FEATURE_VERSION
    index = baseline.FEATURES.index("area")
    assert fitted["means"][index] == pytest.approx((1 + 0.05) / 2)
    assert fitted["scales"][index] == pytest.approx(0.475)
    assert result["method"] == "train_metadata_nearest_centroid_v1"
    assert result["model_executed"] is result["image_pixels_used"] is False
    assert result["quality_approved"] is result["promotion_eligible"] is False
    assert all(r["request_sha256"] == result["request_sha256"] for r in result["predictions"])
    assert "closed_scores" not in result and "inference_receipt" not in result


@pytest.mark.parametrize(
    "change",
    [
        {"split": "validation"},
        {"split": "test"},
        {"control": "both_blank"},
        {"control": "source_replaced"},
        {"label": "not-a-class"},
    ],
)
def test_fit_refuses_leaked_splits_and_synthetic_controls_instead_of_filtering(change):
    with pytest.raises(ValueError, match="TRAIN-only"):
        baseline.fit([*training(), row("bad", **change)])


def test_all_aliases_allowed_zero_variance_safe_and_ties_deterministic():
    rows = [row(f"r{i}", label) for i, label in enumerate(baseline.PURPOSE_CLASSES)]
    fitted = baseline.fit(rows)
    assert set(fitted["centroids"]) == set(baseline.PURPOSE_CLASSES)
    assert all(value == 1 for value in fitted["scales"])
    assert predict(fitted, request(row("target")))["predictions"][0]["prediction"] == "accent"


def test_features_use_first_static_marker_and_fixed_text_flags():
    text = (
        "Visit https://example.org or a@example.org, call +1 212 555 1234. "
        + baseline.MARKER
        + "{}"
    )
    value = baseline.features(row("text", text=text)["prompt"])
    assert len(value) == len(baseline.FEATURES)
    assert [value[baseline.FEATURES.index(k)] for k in ("url", "email", "phone", "cta")] == [1] * 4
    assert value[baseline.FEATURES.index("paint_order_fraction")] == 0.5
    assert value[baseline.FEATURES.index("type_text")] == 1


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"text","type":"image"}',
        '{"bbox_normalized":[0,0,1,1],"type":"text","text_data":"x","paint_order_index":0,"element_count":1,"label":"body"}',
        '{"bbox_normalized":[0,0,1e999,1],"type":"text","text_data":"x","paint_order_index":0,"element_count":1}',
    ],
)
def test_context_rejects_duplicate_unknown_and_nonfinite_metadata(payload):
    with pytest.raises(ValueError):
        baseline.features("Static prompt. " + baseline.MARKER + payload)


@pytest.mark.parametrize("where", ["root", "row", "nested"])
def test_prediction_rejects_answers_anywhere_in_request(where):
    value = request(row("predict"))
    if where == "root":
        value["labels"] = {"predict": "body"}
    elif where == "row":
        value["records"][0]["label"] = None
    else:
        value["records"][0]["images"][0]["extra"] = {"answer": "body"}
    with pytest.raises(ValueError, match="label-free"):
        predict(baseline.fit(training()), value)


def test_image_paths_never_opened_and_controls_do_not_change_metadata_decision(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Metadata baseline opened a file")

    monkeypatch.setattr(Path, "open", fail)
    fitted = baseline.fit(training())
    original = row("original")
    blank = row("blank", control="both_blank")
    replaced = row("replaced", control="source_replaced")
    for value in (blank, replaced):
        value["images"] = [{"path": "different-blank.png", "sha256": "f" * 64}] * 2
    result = predict(fitted, request(original, blank, replaced))
    assert {r["prediction"] for r in result["predictions"]} == {"body"}


def test_three_views_preserved_without_pixels_and_fit_serialization_is_deterministic():
    first, second = baseline.fit(training()), baseline.fit(list(reversed(training())))
    assert baseline.canonical(first) == baseline.canonical(second)
    value = request(row("predict"), views=3)
    value["records"][0]["images"].append({"path": "missing-isolated.png", "sha256": "3" * 64})
    assert baseline.canonical(predict(first, value)) == baseline.canonical(predict(second, value))
    first["means"][0] += 1
    with pytest.raises(ValueError, match="fit identity"):
        predict(first, value)


def test_duplicates_empty_inputs_and_wrong_view_count_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        baseline.fit([row("same"), row("same")])
    with pytest.raises(ValueError, match="nonempty"):
        baseline.fit([])
    value = request(row("one"), views=3)
    with pytest.raises(ValueError, match="ordered image"):
        predict(baseline.fit(training()), value)
    value = request(row("one"))
    del value["records"][0]["control"]
    with pytest.raises(ValueError, match="control metadata"):
        predict(baseline.fit(training()), value)


def test_cli_binds_raw_request_bytes_and_never_overwrites(tmp_path):
    train, query, out = tmp_path / "train.jsonl", tmp_path / "request.json", tmp_path / "new"
    train.write_text("\n".join(json.dumps(r) for r in training()), encoding="utf-8")
    query.write_text(json.dumps(request(row("predict")), indent=2), encoding="utf-8")
    args = ["--train", str(train), "--request", str(query), "--out", str(out)]
    baseline.main(args)
    result = json.loads((out / "predictions.json").read_text())
    assert result["request_sha256"] == hashlib.sha256(query.read_bytes()).hexdigest()
    assert set(p.name for p in out.iterdir()) == {"fit.json", "predictions.json"}
    before = (out / "predictions.json").read_bytes()
    with pytest.raises(FileExistsError):
        baseline.main(args)
    assert (out / "predictions.json").read_bytes() == before


def test_cli_rejects_bad_train_before_creating_output(tmp_path):
    train, query, out = tmp_path / "train.jsonl", tmp_path / "request.json", tmp_path / "out"
    train.write_text(json.dumps(row("leak", split="validation")), encoding="utf-8")
    query.write_text(json.dumps(request(row("predict"))), encoding="utf-8")
    with pytest.raises(ValueError, match="TRAIN-only"):
        baseline.main(["--train", str(train), "--request", str(query), "--out", str(out)])
    assert not out.exists()


def test_fresh_import_and_execution_need_no_image_or_gpu_packages():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','peft','transformers','PIL','skia'}:
        raise AssertionError('Forbidden image/model import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools import purpose_metadata_baseline
import json
context = {'bbox_normalized':[0,0,1,1], 'type':'text', 'text_data':'Details',
           'paint_order_index':0, 'element_count':1}
row = {'record_id':'one', 'family_id':'train', 'split':'train', 'control':'original',
       'label':'body', 'prompt':purpose_metadata_baseline.MARKER + json.dumps(context)}
fitted = purpose_metadata_baseline.fit([row])
query = {key:value for key,value in row.items() if key not in {'split','label'}}
query['images'] = [{'path':'not-opened.png','sha256':'0'*64}]*2
request = {'protocol':purpose_metadata_baseline.REQUEST_PROTOCOL, 'task':'purpose',
           'records':[query]}
result = purpose_metadata_baseline.predict(fitted,request,request_sha256='1'*64)
assert result['predictions'][0]['prediction'] == 'body'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
