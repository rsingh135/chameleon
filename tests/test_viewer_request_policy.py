"""Request deadlines follow declared local configuration without running models."""

import http.client
import json
import threading
from collections import Counter
from copy import deepcopy

import pytest

from engine import visual_config, visual_pipeline
from viewer import benchmark, server

DEFAULT_POLICY = {
    "studio_timeout_ms": 60000,
    "benchmark_timeout_ms": 120000,
    "model_configured": False,
}


@pytest.fixture(autouse=True)
def no_ambient_model_configuration(monkeypatch):
    monkeypatch.delenv("CHAMELEON_VISUAL_PLANNER_CONFIG", raising=False)

    def unexpected_provider(*_args, **_kwargs):
        pytest.fail("Reading request policy must not run a visual provider")

    monkeypatch.setattr(visual_pipeline, "resize_configured", unexpected_provider)


def _mock_configuration(monkeypatch, *, enabled=True, critic=True, timeout=240, error=None):
    monkeypatch.setenv("CHAMELEON_VISUAL_PLANNER_CONFIG", "mocked-operator-config")
    calls = Counter()
    identity = {
        "enabled": enabled,
        "valid": error is None,
        "identity": "mocked-static-identity" if error is None else None,
    }

    def resolve():
        calls["resolve"] += 1
        if error is not None:
            raise error
        return {"enabled": enabled, "critic": critic, "timeout_seconds": timeout}, {}

    def execution_identity():
        calls["identity"] += 1
        return deepcopy(identity)

    monkeypatch.setattr(visual_config, "resolve_config", resolve)
    monkeypatch.setattr(visual_pipeline, "visual_execution_identity", execution_identity)
    return identity, calls


def test_default_policy_does_not_even_resolve_or_identify_a_model(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("No configured model should need configuration or runtime inspection")

    monkeypatch.setattr(visual_config, "resolve_config", unexpected)
    monkeypatch.setattr(visual_pipeline, "visual_execution_identity", unexpected)
    identity, critic, policy = benchmark._visual_identity()
    assert identity == {"enabled": False}
    assert critic is False and policy == DEFAULT_POLICY


@pytest.mark.parametrize("timeout,critic", [(1, False), (240, False), (240, True), (600, True)])
def test_enabled_operator_policy_covers_planner_critic_and_native_overhead(
    monkeypatch, timeout, critic
):
    expected_identity, calls = _mock_configuration(monkeypatch, critic=critic, timeout=timeout)
    identity, actual_critic, policy = benchmark._visual_identity()
    budget = (timeout * (1 + bool(critic)) + 150) * 1000
    assert identity == expected_identity and actual_critic is critic
    assert policy == {
        "studio_timeout_ms": max(60000, budget),
        "benchmark_timeout_ms": max(120000, budget),
        "model_configured": True,
    }
    assert calls["resolve"] == calls["identity"] == 1
    assert set(policy) == set(DEFAULT_POLICY)  # No provider commands, paths or run metadata.


def test_disabled_operator_uses_default_deadlines_and_is_not_model_configured(monkeypatch):
    expected_identity, _ = _mock_configuration(monkeypatch, enabled=False, critic=True, timeout=600)
    identity, critic, policy = benchmark._visual_identity()
    assert identity == expected_identity
    assert critic is False and policy == DEFAULT_POLICY


@pytest.mark.parametrize("error", [ValueError("invalid"), OSError("missing"), TypeError("bad")])
def test_invalid_explicit_configuration_keeps_default_deadlines_but_remains_configured(
    monkeypatch, error
):
    expected_identity, _ = _mock_configuration(monkeypatch, error=error)
    identity, critic, policy = benchmark._visual_identity()
    assert identity == expected_identity and critic is False
    assert policy == {**DEFAULT_POLICY, "model_configured": True}


def test_method_receipt_exposes_the_same_computed_policy(monkeypatch):
    policy = {
        "studio_timeout_ms": 630000,
        "benchmark_timeout_ms": 630000,
        "model_configured": True,
    }
    monkeypatch.setattr(
        benchmark,
        "_visual_identity",
        lambda: ({"enabled": True, "valid": True, "identity": "mocked"}, True, policy),
    )
    receipt = benchmark.method_receipt()
    assert receipt["options"]["request_policy"] is policy
    assert receipt["critic"] == {"enabled": True}


@pytest.fixture
def policy_server(monkeypatch):
    receipt = {
        "id": "deterministic-current",
        "digest": "a" * 64,
        "options": {"request_policy": deepcopy(DEFAULT_POLICY)},
    }
    monkeypatch.setattr(benchmark, "method_receipt", lambda: deepcopy(receipt))
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance, receipt
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=3)


def _http(instance, method, path):
    connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=10)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_policy_endpoint_returns_only_default_deadlines_and_supports_head(policy_server):
    instance, _ = policy_server
    status, headers, body = _http(instance, "GET", "/api/resize/policy")
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert json.loads(body) == DEFAULT_POLICY
    status, headers, body = _http(instance, "HEAD", "/api/resize/policy")
    assert status == 200 and body == b"" and int(headers["Content-Length"]) > 0
    status, _, body = _http(instance, "GET", "/api/health")
    assert status == 200 and json.loads(body) == {"status": "ok"}


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_policy_endpoint_rejects_code_identity_drift_after_server_start(policy_server, method):
    instance, receipt = policy_server
    receipt["digest"] = "b" * 64
    status, _, body = _http(instance, method, "/api/resize/policy")
    assert status == 409
    if method == "HEAD":
        assert body == b""
    else:
        assert "Restart" in json.loads(body)["error"]
    # Health remains the small availability response, independent of policy.
    status, _, body = _http(instance, "GET", "/api/health")
    assert status == 200 and json.loads(body) == {"status": "ok"}
