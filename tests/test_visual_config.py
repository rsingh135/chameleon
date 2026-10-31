"""Operator-only visual configuration is strict, local and honest about pins."""

import json
import sys
from pathlib import Path

import pytest

from engine import visual_config as module


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    package = tmp_path / "experiments/visual_planner"
    package.mkdir(parents=True)
    (package / "provider.py").write_text('PROMPT = "inspect original pixels"\n')
    (package / "worker.py").write_text("# verifies actual model files before load\n")
    model = tmp_path / "models/pilot"
    model.mkdir(parents=True)
    (model / "verified-model.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": "public/model",
                "revision": "a" * 40,
                "weight_sha256": "b" * 64,
                "files": {"model.safetensors": {"sha256": "b" * 64, "bytes": 100}},
            }
        )
    )
    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps({"model_dir": str(model), "python_executable": sys.executable}))
    config = {
        "schema_version": 1,
        "enabled": True,
        "provider_command": [
            sys.executable,
            "-m",
            "experiments.visual_planner.provider",
            "--config",
            str(provider),
        ],
        "timeout_seconds": 240,
        "artifact_root": str(tmp_path / "output/live"),
        "critic": True,
    }
    return config, provider, model


def test_unconfigured_and_minimal_disabled_never_load_provider(monkeypatch):
    monkeypatch.delenv("CHAMELEON_VISUAL_PLANNER_CONFIG", raising=False)
    config, identity = module.resolve_config()
    assert config is None and not identity["enabled"]
    assert module.visual_model_enabled() is False
    disabled, identity = module.resolve_config({"schema_version": 1, "enabled": False})
    assert disabled["enabled"] is False and not identity["enabled"]


def test_valid_config_records_manifest_pins_without_claiming_fresh_model_hash(configured):
    config, _, _ = configured
    actual, identity = module.resolve_config(config)
    assert actual["provider_command"][0] == str(Path(config["provider_command"][0]).resolve())
    assert actual["provider_command"][1:] == config["provider_command"][1:]
    assert actual["timeout_seconds"] == 240
    assert identity["enabled"] and identity["model"]["binary_bytes_verified"] is False
    assert identity["model"]["manifest_sha256"]
    assert identity["provider_configuration_sha256"]
    assert identity["configuration_sha256"] != identity["provider_configuration_sha256"]
    assert module.visual_model_enabled(config) is True
    assert module.visual_execution_identity(config)["identity"] == identity["digest"]
    assert "torch" not in module.__dict__


@pytest.mark.parametrize(
    "key,value",
    [
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", 601),
        ("timeout_seconds", True),
        ("critic", "yes"),
        ("enabled", "yes"),
        ("schema_version", True),
        ("mode", "shell"),
        ("provider_command", ["cmd.exe", "/c", "echo secret"]),
        ("provider_command", [sys.executable, "-c", "print('untrusted')"]),
        ("artifact_root", "../outside"),
    ],
)
def test_invalid_operator_fields_reject_and_still_bypass_model_response_cache(
    configured, key, value
):
    config, _, _ = configured
    config[key] = value
    with pytest.raises(ValueError):
        module.resolve_config(config)
    assert module.visual_model_enabled(config) is True
    execution = module.visual_execution_identity(config)
    assert execution["enabled"] and not execution["valid"]


def test_missing_env_config_is_enabled_invalid_not_silently_model_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAMELEON_VISUAL_PLANNER_CONFIG", str(tmp_path / "missing.json"))
    assert module.visual_model_enabled() is True
    assert module.visual_execution_identity()["valid"] is False


def test_identity_changes_with_provider_config_code_or_manifest(configured):
    config, provider, model = configured
    _, initial = module.resolve_config(config)
    value = json.loads(provider.read_text())
    value["max_new_tokens"] = 17
    provider.write_text(json.dumps(value))
    _, changed = module.resolve_config(config)
    assert changed["digest"] != initial["digest"]
    worker = module.ROOT / "experiments/visual_planner/worker.py"
    worker.write_text(worker.read_text() + "# changed prompt/code\n")
    _, code = module.resolve_config(config)
    assert code["digest"] != changed["digest"]
    manifest = model / "verified-model.json"
    data = json.loads(manifest.read_text())
    data["weight_sha256"] = "c" * 64
    manifest.write_text(json.dumps(data))
    _, pins = module.resolve_config(config)
    assert pins["digest"] != code["digest"]


def test_config_file_duplicate_keys_and_nonfinite_json_reject(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"schema_version":1,"enabled":false,"enabled":true}')
    with pytest.raises(ValueError):
        module.resolve_config(path)


@pytest.mark.parametrize(
    "change", [{"backend": "shell"}, {"strategy": "eval"}, {"model_dir": "relative-model"}]
)
def test_provider_modes_and_ambiguous_relative_paths_reject(configured, change):
    config, provider, _ = configured
    data = json.loads(provider.read_text())
    data.update(change)
    provider.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.resolve_config(config)


@pytest.mark.parametrize("change", [{"backend": []}, {"strategy": {}}])
def test_nested_provider_mode_types_raise_validation_error(configured, change):
    config, provider, _ = configured
    data = json.loads(provider.read_text())
    data.update(change)
    provider.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.resolve_config(config)


def test_analyzer_has_independent_configuration_and_execution_identity(configured):
    config, provider, _ = configured
    analyzer_path = provider.with_name("analyzer.json")
    data = json.loads(provider.read_text())
    data["max_new_tokens"] = 96
    analyzer_path.write_text(json.dumps(data))
    config["analyzer_command"] = [*config["provider_command"][:-1], str(analyzer_path)]
    resolved, initial = module.resolve_config(config)
    assert resolved["analyzer_command"][-1] == str(analyzer_path)
    assert initial["analyzer"]["provider_configuration"]["max_new_tokens"] == 96
    data["max_new_tokens"] = 128
    analyzer_path.write_text(json.dumps(data))
    _, changed = module.resolve_config(config)
    assert initial["digest"] != changed["digest"]
    assert initial["provider_configuration_sha256"] == changed["provider_configuration_sha256"]


def test_analyzer_command_retains_same_local_execution_boundary(configured):
    config, _, _ = configured
    config["analyzer_command"] = [sys.executable, "-c", "print('untrusted')"]
    with pytest.raises(ValueError, match="fixed local"):
        module.resolve_config(config)
