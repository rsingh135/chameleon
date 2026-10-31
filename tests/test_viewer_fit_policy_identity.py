"""Method provenance distinguishes the geometry fit planner from optional vision."""

from copy import deepcopy

import pytest

from viewer import benchmark


@pytest.fixture
def fit_identity(monkeypatch):
    identity = {
        "enabled": True,
        "status": "ready",
        "path": "/models/fit-policy.json",
        "implementation_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "model_digest": "c" * 64,
        "feature_version": "fit-features-v1",
        "label_kind": "synthetic_geometry_teacher_v1",
        "dataset_digest": "d" * 64,
        "vlm_fine_tuning": False,
    }
    monkeypatch.setattr(benchmark, "fit_policy_identity", lambda: deepcopy(identity))
    monkeypatch.setattr(
        benchmark,
        "_visual_identity",
        lambda: (
            {"enabled": False},
            False,
            {
                "studio_timeout_ms": 60_000,
                "benchmark_timeout_ms": 120_000,
                "model_configured": False,
            },
        ),
    )
    return identity


def test_default_trained_fit_planner_is_reported_separately_from_vision(fit_identity):
    receipt = benchmark.method_receipt()
    assert receipt["fit_planner"] == {
        **fit_identity,
        "kind": "trained_pairwise_fit_policy",
        "uses_vision": False,
    }
    assert receipt["analyzer"] == receipt["critic"] == {"enabled": False}
    assert receipt["options"]["layout"] == "trained-fit-policy"
    # Geometry ranking does not opt into the long-running visual provider.
    assert receipt["options"]["request_policy"]["studio_timeout_ms"] == 60_000
    assert receipt["options"]["request_policy"]["model_configured"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_sha256", "e" * 64),
        ("model_digest", "f" * 64),
        ("path", "/models/another-fit-policy.json"),
        ("feature_version", "fit-features-v2"),
    ],
)
def test_fit_artifact_and_configuration_change_method_without_code_edits(
    fit_identity, field, value
):
    before = benchmark.method_receipt()
    fit_identity[field] = value
    after = benchmark.method_receipt()
    assert before["implementation_sha256"] == after["implementation_sha256"]
    assert before["digest"] != after["digest"]
    assert after["fit_planner"][field] == value


@pytest.mark.parametrize("status", ["no_model", "invalid_model"])
def test_unavailable_fit_artifact_does_not_claim_a_trained_planner(fit_identity, status):
    fit_identity.update(enabled=False, status=status, model_sha256=None)
    receipt = benchmark.method_receipt()
    assert receipt["fit_planner"]["enabled"] is False
    assert receipt["fit_planner"]["status"] == status
    assert receipt["options"]["layout"] == "current-engine-defaults"
    assert receipt["analyzer"]["enabled"] is False


def test_configured_vision_identity_remains_independent_of_fit_planner(fit_identity, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_visual_identity",
        lambda: ({"enabled": True, "model_sha256": "v" * 64}, True, {}),
    )
    receipt = benchmark.method_receipt()
    assert receipt["fit_planner"]["model_sha256"] == fit_identity["model_sha256"]
    assert receipt["fit_planner"]["uses_vision"] is False
    assert receipt["analyzer"]["model_sha256"] == "v" * 64
    assert receipt["critic"]["enabled"] is True
    assert receipt["options"]["layout"] == "configured-visual-pipeline"
