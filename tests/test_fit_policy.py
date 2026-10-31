"""Actual tiny-policy training and causal geometry selection; no VLM claim."""

import hashlib
import json
from copy import deepcopy

import pytest

from engine import fit_policy as policy


def example(family):
    source = {
        "canvas": {"width": 100, "height": 200},
        "elements": [
            {
                "id": "heading",
                "type": "text",
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 25,
                "font_size": 16,
                "content": "Heading " + family,
            },
            {
                "id": "body",
                "type": "text",
                "x": 10,
                "y": 50,
                "width": 80,
                "height": 120,
                "font_size": 10,
                "content": "Supporting design copy " + family,
            },
        ],
    }
    target = {"width": 200, "height": 100}
    small, fitted = deepcopy(source), deepcopy(source)
    small["canvas"], fitted["canvas"] = dict(target), dict(target)
    small["elements"][0].update(x=85, y=20, width=24, height=8, font_size=4.8)
    small["elements"][1].update(x=85, y=35, width=24, height=36, font_size=3)
    fitted["elements"][0].update(x=10, y=8, width=180, height=25)
    fitted["elements"][1].update(x=10, y=40, width=180, height=50)
    candidates = [
        {"plan_id": "z-underfilled", "design": small, "hard_valid": True},
        {"plan_id": "a-fitted", "design": fitted, "hard_valid": True},
    ]
    return source, target, candidates


def dataset():
    rows = []
    for family, split in (
        ("event", "train"),
        ("product", "train"),
        ("editorial", "validation"),
        ("brochure", "validation"),
    ):
        source, target, candidates = example(family)
        menu = policy.build_fit_menu(source, target, candidates)
        for candidate in candidates:
            rows.append(
                {
                    "source_id": family,
                    "family": family,
                    "split": split,
                    "source_digest": policy.source_digest(source),
                    "target_canvas": target,
                    "menu_digest": menu["menu_digest"],
                    "plan_id": candidate["plan_id"],
                    "features": policy.extract_features(source, target, candidate["design"]),
                    "teacher_score": 0.9 if candidate["plan_id"] == "a-fitted" else 0.3,
                    "hard_valid": True,
                    "design_digest": policy.digest(candidate["design"]),
                }
            )
    return {
        "schema_version": 1,
        "feature_version": policy.FEATURE_VERSION,
        "label_kind": policy.LABEL_KIND,
        "records": rows,
    }


def test_actual_training_generalizes_and_causally_changes_executed_geometry(
    tmp_path,
):
    model_path = tmp_path / "policy.json"
    artifact = policy.train_model(dataset(), model_path, epochs=80)
    assert artifact["training"]["pairs"] > 0
    assert any(abs(weight) > 0.01 for weight in artifact["weights"])
    losses = artifact["training"]["losses"]
    assert losses[-1]["mean_pairwise_log_loss"] < losses[0]["mean_pairwise_log_loss"]
    assert artifact["metrics"]["validation"]["pairwise_accuracy"] == 1
    assert artifact["metrics"]["untrained_validation"]["pairwise_accuracy"] == 0.5
    assert not set(artifact["families"]["train"]) & set(artifact["families"]["validation"])
    assert artifact["vlm_fine_tuning"] is False

    source, target, candidates = example("unseen-acceptance")
    menu = policy.build_fit_menu(source, target, candidates)
    prediction = policy.choose_fit_plan(source, target, menu, candidates, model_path)
    assert prediction["receipt"]["executed"] is True
    assert prediction["decision"]["plan_id"] == "a-fitted"
    executed = next(
        row["design"] for row in candidates if row["plan_id"] == prediction["decision"]["plan_id"]
    )
    assert executed["elements"][0]["width"] == 180
    assert executed != candidates[0]["design"]

    zero = deepcopy(artifact)
    zero["weights"] = [0.0] * len(policy.FEATURE_NAMES)
    zero["test_ablation"] = "Zero learned weights; not a promoted model"
    zero.pop("model_digest")
    zero["model_digest"] = policy.digest(zero)
    zero_path = tmp_path / "zero-ablation.json"
    zero_path.write_text(json.dumps(zero))
    ablated = policy.choose_fit_plan(source, target, menu, candidates, zero_path)
    assert ablated["decision"]["plan_id"] == "z-underfilled"
    assert ablated["receipt"]["model_digest"] != prediction["receipt"]["model_digest"]
    assert prediction["receipt"]["quality_improved"] is None


@pytest.mark.parametrize("leak", ["family", "source", "reserve"])
def test_training_rejects_family_leakage_and_reserve_rows(tmp_path, leak):
    data = dataset()
    if leak == "family":
        data["records"][-1]["family"] = data["records"][0]["family"]
    elif leak == "source":
        data["records"][-1]["source_digest"] = data["records"][0]["source_digest"]
    else:
        data["records"][0]["reserve"] = True
    with pytest.raises(ValueError):
        policy.train_model(data, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_prediction_rejects_stale_target_unknown_and_changed_geometry(tmp_path):
    source, target, candidates = example("acceptance")
    menu = policy.build_fit_menu(source, target, candidates)
    wrong = dict(target, width=300)
    with pytest.raises(ValueError, match="source/target"):
        policy.choose_fit_plan(source, wrong, menu, candidates, tmp_path / "missing.json")
    changed = deepcopy(candidates)
    changed[0]["plan_id"] = "unknown"
    with pytest.raises(ValueError, match="exactly match"):
        policy.choose_fit_plan(source, target, menu, changed, tmp_path / "missing.json")
    path = tmp_path / "trained.json"
    policy.train_model(dataset(), path, epochs=10)
    changed = deepcopy(candidates)
    changed[0]["design"]["elements"][0]["x"] += 1
    with pytest.raises(ValueError, match="geometry differs"):
        policy.choose_fit_plan(source, target, menu, changed, path)


def test_missing_model_reports_no_execution_and_invalid_model_is_visible(tmp_path):
    source, target, candidates = example("acceptance")
    menu = policy.build_fit_menu(source, target, candidates)
    missing = tmp_path / "missing.json"
    result = policy.choose_fit_plan(source, target, menu, candidates, missing)
    assert result["decision"] is None
    assert result["receipt"]["executed"] is False
    assert result["receipt"]["status"] == "no_model"
    path = tmp_path / "broken.json"
    path.write_text('{"kind":"fake"}')
    identity = policy.fit_policy_identity(path)
    assert not identity["enabled"] and identity["status"] == "invalid_model"


def test_features_preserve_source_and_capture_semantic_group_and_reading_failures():
    source, target, candidates = example("groups")
    for row in source["elements"]:
        row["group_id"] = "message"
    original = deepcopy(source)
    good = policy.extract_features(source, target, candidates[1]["design"])
    bad = deepcopy(candidates[1]["design"])
    bad["elements"][0]["y"], bad["elements"][1]["y"] = 50, 8
    bad["elements"][0]["font_size"] = 5
    bad["elements"][1]["width"] = 30
    poor = policy.extract_features(source, target, bad)
    assert good["reading_order_retention"] > poor["reading_order_retention"]
    assert good["hierarchy_retention"] > poor["hierarchy_retention"]
    assert good["unit_scale_consistency"] > poor["unit_scale_consistency"]
    assert source == original


def test_metrics_keep_synthetic_and_authored_validation_domains_separate(tmp_path):
    data = dataset()
    for row in data["records"]:
        row["source_kind"] = (
            "synthetic_column" if row["family"] == "brochure" else "authored_development"
        )
    artifact = policy.train_model(data, tmp_path / "domains.json", epochs=10)
    domains = artifact["metrics"]["validation_by_source_kind"]
    assert set(domains) == {"synthetic_column", "authored_development"}
    assert domains["synthetic_column"]["groups"] == 1
    assert domains["authored_development"]["groups"] == 1
    assert sum(row["records"] for row in domains.values()) == 4


def test_invalid_identity_hashes_small_bytes_and_bounds_oversized_reads(tmp_path, monkeypatch):
    small = tmp_path / "broken.json"
    small.write_bytes(b"{broken model")
    identity = policy.fit_policy_identity(small)
    assert identity["status"] == "invalid_model"
    assert identity["model_sha256"] == hashlib.sha256(small.read_bytes()).hexdigest()
    assert identity["model_bytes_verified"] is True
    large = tmp_path / "large.json"
    large.write_bytes(b"x" * 1_000_001)
    original = policy.Path.open

    def guarded(path, *args, **kwargs):
        if path == large:
            pytest.fail("Oversized model bytes must not be read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(policy.Path, "open", guarded)
    oversized = policy.fit_policy_identity(large)
    assert oversized["status"] == "invalid_model"
    assert oversized["model_file_size"] == 1_000_001
    assert oversized["model_mtime_ns"] > 0
    assert oversized["model_sha256"] is None
    assert oversized["model_bytes_verified"] is False


def test_training_replaces_only_a_complete_checkpoint_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "active.json"
    old = b'{"previous":"artifact"}'
    destination.write_bytes(old)
    replace = policy.os.replace
    observed = []

    def checked(temporary, target):
        assert destination.read_bytes() == old
        replacement = json.loads(policy.Path(temporary).read_text())
        assert replacement["kind"] == "trained_pairwise_fit_policy"
        observed.append(replacement["model_digest"])
        replace(temporary, target)

    monkeypatch.setattr(policy.os, "replace", checked)
    result = policy.train_model(dataset(), destination, epochs=10)
    assert observed == [result["model_digest"]]
    assert policy.load_model(destination)["model_digest"] == result["model_digest"]
    assert not list(tmp_path.glob("*.tmp"))
