"""Optional add-on drafts cannot become mandatory release fingerprint inputs."""

from pathlib import Path

import pytest

from viewer import benchmark


@pytest.fixture
def receipt_checkout(tmp_path, monkeypatch):
    for relative in (
        "tools/scene_harness.py",
        "viewer/server.py",
        "viewer/document_bridge.py",
        "viewer/resize_cache.py",
        "viewer/benchmark.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixed mandatory implementation\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "__file__", str(tmp_path / "viewer" / "benchmark.py"))
    monkeypatch.setattr(benchmark, "fit_policy_identity", lambda: {"enabled": False})
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
    return tmp_path


def test_receipt_works_without_an_optional_unreleased_addon_bridge(receipt_checkout):
    assert not (receipt_checkout / "viewer" / "addon_bridge.py").exists()
    receipt = benchmark.method_receipt()
    assert len(receipt["digest"]) == len(receipt["implementation_sha256"]) == 64
    assert receipt["options"]["layout"] == "current-engine-defaults"


def test_present_optional_addon_bytes_are_bound_to_the_method_fingerprint(receipt_checkout):
    absent = benchmark.method_receipt()
    optional = receipt_checkout / "viewer" / "addon_bridge.py"
    optional.write_text("# optional add-on implementation A\n", encoding="utf-8")
    present = benchmark.method_receipt()
    optional.write_text("# optional add-on implementation B\n", encoding="utf-8")
    changed = benchmark.method_receipt()
    assert len({r["implementation_sha256"] for r in (absent, present, changed)}) == 3
    assert len({r["digest"] for r in (absent, present, changed)}) == 3
    assert absent["options"] == present["options"] == changed["options"]


def test_mandatory_fingerprint_inputs_still_cannot_be_missing(receipt_checkout):
    mandatory = receipt_checkout / "viewer" / "document_bridge.py"
    assert mandatory.is_relative_to(Path(receipt_checkout).resolve())
    mandatory.unlink()
    with pytest.raises(FileNotFoundError):
        benchmark.method_receipt()
