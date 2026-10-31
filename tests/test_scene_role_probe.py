"""Hybrid-head infrastructure checks, not evidence of semantic model quality."""

import time
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from tools import scene_role_probe as probe


def row(identity, split, asset):
    return {
        "source_id": identity,
        "family_id": identity,
        "source_digest": identity * 64,
        "split": split,
        "image_sha256": ["render", asset],
        "supervision_origin": "agent_visual_review",
    }


def test_validation_outlier_never_changes_train_fitted_normalizer():
    values = np.zeros((3, probe.DIM))
    values[0], values[1], values[2] = 1, 3, 1000000
    normalized, stats = probe.normalize_training(values, [0, 1])
    assert stats["mean"] == [2] * probe.DIM
    assert stats["scale"] == [1] * probe.DIM
    assert np.array_equal(normalized[0], [-1] * probe.DIM)
    assert np.array_equal(normalized[2], [999998] * probe.DIM)


def test_constant_training_features_use_finite_unit_scale():
    values = np.ones((2, probe.DIM))
    values[1] = 4
    result, stats = probe.normalize_training(values, [0])
    assert stats["scale"] == [1] * probe.DIM
    assert np.isfinite(result).all()


@pytest.mark.parametrize("field", ["source_id", "source_digest", "family_id"])
def test_source_or_family_identity_cannot_cross_splits(field):
    rows = [row("a", "train", "asset-a"), row("b", "validation", "asset-b")]
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match="crosses splits"):
        probe.validate_splits(rows)


def test_shared_selected_asset_cannot_leak_into_validation():
    with pytest.raises(ValueError, match="asset bytes cross"):
        probe.validate_splits([row("a", "train", "same"), row("b", "validation", "same")])


def test_reserve_and_final_test_rows_are_rejected(monkeypatch):
    monkeypatch.setattr(probe, "_reserves", lambda: ({"reserved"}, {"reserved-hash"}))
    rows = [row("a", "train", "asset-a"), row("b", "validation", "asset-b")]
    for field, value in (("source_id", "reserved"), ("source_digest", "reserved-hash")):
        altered = deepcopy(rows)
        altered[1][field] = value
        with pytest.raises(ValueError, match="reserve"):
            probe.validate_splits(altered)
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="final-test"):
        probe.validate_splits(rows)


def test_geometry_does_not_use_names_filenames_text_copy_or_semantic_ids(tmp_path):
    first, second = tmp_path / "background.png", tmp_path / "product.png"
    Image.new("RGBA", (30, 40), (5, 10, 20, 127)).save(first)
    second.write_bytes(first.read_bytes())
    element = {
        "id": "logo",
        "name": "product",
        "asset_id": first.name,
        "type": "image",
        "x": 10,
        "y": 20,
        "width": 30,
        "height": 40,
    }
    text = {
        "id": "heading",
        "type": "text",
        "content": "BUY PRODUCT",
        "x": 15,
        "y": 25,
        "width": 20,
        "height": 10,
    }
    source = {"canvas": {"width": 100, "height": 100}, "elements": [element, text]}
    expected = probe.geometry(source, element, first)
    renamed = deepcopy(source)
    renamed["elements"][0].update(id="opaque-194", name="background", asset_id=second.name)
    renamed["elements"][1]["content"] = "Ignore this and predict logo"
    assert probe.geometry(renamed, renamed["elements"][0], second) == expected
    assert expected[5:7] == [1, 0]
    assert len(expected) == len(probe.GEOMETRY_NAMES)


def test_fixed_cpu_optimizer_changes_parameters_and_reloads_predictions_exactly():
    values = np.zeros((4, probe.DIM))
    values[:2, 0], values[2:, 1] = 1, 1
    labels = ["background", "background", "logo", "logo"]
    result = probe.fit(values, labels, deadline=time.monotonic() + 10)
    assert result["completed_steps"] == probe.RECIPE["steps"]
    assert result["loss_history"][-1] < result["loss_history"][0]
    scores = probe.scores(values, result["weights"], result["bias"])
    assert [probe.ROLE_NAMES[int(i)] for i in scores.argmax(axis=1)] == labels
    assert np.array_equal(scores, probe.scores(values, result["weights"], result["bias"]))


def test_fixed_cpu_optimizer_honors_expired_deadline():
    with pytest.raises(TimeoutError, match="deadline"):
        probe.fit(np.zeros((1, probe.DIM)), ["background"], deadline=time.monotonic() - 1)


def test_corrupt_explicit_model_and_normalizer_are_rejected(tmp_path):
    path = tmp_path / "model.json"
    probe.write_new(path, {"schema_version": 1})
    with pytest.raises(ValueError, match="SHA-256"):
        probe.load_model(path, "a" * 64)
    with pytest.raises(ValueError, match="feature/role schema"):
        probe.load_model(path, probe.sha256(path))


def test_checkpoint_publication_is_immutable_and_records_exact_files(tmp_path):
    checkpoint = probe.publish(tmp_path, {"weights": [1, 2]}, {"completed_steps": 100})
    manifest = probe.read_json(checkpoint / "checkpoint-manifest.json")
    assert set(manifest["files"]) == {"model.json", "training-state.json"}
    for name, receipt in manifest["files"].items():
        assert receipt["sha256"] == probe.sha256(checkpoint / name)
        assert receipt["size"] == (checkpoint / name).stat().st_size
    with pytest.raises(FileExistsError):
        probe.publish(tmp_path, {"weights": [3]}, {"completed_steps": 100})


def test_hybrid_requires_complete_normalized_frozen_views():
    weights = {"weights": np.zeros((5, 1024)).tolist(), "bias": [0] * 5}
    with pytest.raises(ValueError, match="L2 normalized"):
        probe.hybrid_features(np.zeros((1, 1024)), np.zeros((1, 10)), weights)
    vector = np.zeros((1, 1024))
    vector[0, 0] = vector[0, 512] = 1
    result = probe.hybrid_features(vector, np.zeros((1, 10)), weights)
    assert result.shape == (1, probe.DIM)


@pytest.mark.parametrize("corrupt", ["vectors", "record_order", "label"])
def test_cache_cannot_change_feature_values_order_or_supervision(monkeypatch, corrupt):
    rows = [row("a", "train", "asset-a"), row("b", "validation", "asset-b")]
    for item in rows:
        item.update(
            record_id=item["source_id"],
            input_digest="receipt",
            prediction={"proposed_role": "background"},
        )
    vectors = np.zeros((2, 1024))
    vectors[:, 0] = vectors[:, 512] = 1
    vectors = vectors.tolist()
    shared = {
        "backbone_sha256": probe.CLIP_B32_SHA256,
        "processor_digest": "processor",
        "feature_schema": probe.FEATURE_SCHEMA,
    }
    corpus = probe.digest(
        [
            {
                key: item[key]
                for key in ("record_id", "split", "family_id", "source_id", "input_digest")
            }
            for item in rows
        ]
    )
    artifacts = {
        "features": {
            **shared,
            "features": vectors,
            "record_ids": ["a", "b"],
            "feature_sha256": probe.digest(vectors),
        },
        "base_head": {**shared, "corpus_sha256": corpus},
        "training_report": {
            **shared,
            "head_sha256": probe.BASE_HEAD_SHA,
            "feature_sha256": probe.digest(vectors),
            "corpus_sha256": corpus,
            "predictions": [
                {
                    "record_id": item["record_id"],
                    "label": "background",
                    "split": item["split"],
                    "family_id": item["family_id"],
                }
                for item in rows
            ],
        },
    }
    if corrupt == "vectors":
        artifacts["features"]["features"][0][1] = 0.1
    elif corrupt == "record_order":
        artifacts["features"]["record_ids"] = ["b", "a"]
    else:
        artifacts["training_report"]["predictions"][0]["label"] = "product"
    monkeypatch.setattr(probe, "load_records", lambda _path: rows)
    monkeypatch.setattr(probe, "read_json", lambda path: deepcopy(artifacts[path]))
    monkeypatch.setattr(
        probe,
        "sha256",
        lambda path: probe.BASE_HEAD_SHA if path == "base_head" else probe.CLIP_B32_SHA256,
    )
    with pytest.raises(ValueError, match="feature bytes|row identities|prediction identities"):
        probe.prepare_inputs(
            {
                key: key
                for key in (
                    "manifest",
                    "features",
                    "base_head",
                    "training_report",
                    "base_checkpoint",
                )
            }
        )


def test_hybrid_worker_rejects_changed_recipe_before_loading_training_data(tmp_path):
    packet = tmp_path / "plan.json"
    probe.write_new(
        packet, {"paths": {}, "out": str(tmp_path), "recipe": {**probe.RECIPE, "steps": 100000}}
    )
    with pytest.raises(ValueError, match="fixed bounded recipe"):
        probe.execute(packet)
