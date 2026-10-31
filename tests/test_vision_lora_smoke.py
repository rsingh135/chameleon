"""Offline infrastructure checks; these do not certify model/CUDA compatibility."""

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import vision_lora_smoke as smoke


@pytest.fixture
def checkpoint_stage(tmp_path):
    stage = tmp_path / ".checkpoint.partial"
    stage.mkdir()
    (stage / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (stage / "adapter_model.safetensors").write_bytes(b"stand-in transport bytes\x00\xff")
    (stage / "training-state.pt").write_bytes(b"stand-in optimizer/RNG transport bytes")
    (stage / "progress.json").write_text('{"completed_steps":2}', encoding="utf-8")
    return stage


def test_complete_checkpoint_is_published_and_independently_verifiable(checkpoint_stage, tmp_path):
    expected = {path.name: path.read_bytes() for path in checkpoint_stage.iterdir()}
    published = tmp_path / "checkpoint-000002"
    smoke.publish_checkpoint(checkpoint_stage, published)
    assert not checkpoint_stage.exists()
    assert smoke.verify_checkpoint(published)["format_version"] == 1
    assert {
        path.name: path.read_bytes()
        for path in published.iterdir()
        if path.name != smoke.CHECKPOINT_MANIFEST
    } == expected
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("missing", sorted(smoke.REQUIRED_CHECKPOINT_FILES))
def test_incomplete_save_never_becomes_a_published_checkpoint(checkpoint_stage, tmp_path, missing):
    (checkpoint_stage / missing).unlink()
    published = tmp_path / "checkpoint-000002"
    with pytest.raises(RuntimeError, match="Incomplete"):
        smoke.publish_checkpoint(checkpoint_stage, published)
    assert not published.exists()
    assert checkpoint_stage.is_dir()


@pytest.mark.parametrize("change", ["corrupt", "remove", "add"])
def test_saved_checkpoint_detects_damage_before_reload(checkpoint_stage, tmp_path, change):
    published = tmp_path / "checkpoint-000002"
    smoke.publish_checkpoint(checkpoint_stage, published)
    if change == "corrupt":
        weights = published / "adapter_model.safetensors"
        data = weights.read_bytes()
        weights.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif change == "remove":
        (published / "training-state.pt").unlink()
    else:
        (published / "unexpected.safetensors").write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="Incomplete|SHA256"):
        smoke.verify_checkpoint(published)


def test_generation_cannot_overwrite_a_previous_checkpoint(checkpoint_stage, tmp_path):
    published = tmp_path / "checkpoint-000002"
    smoke.publish_checkpoint(checkpoint_stage, published)
    manifest = (published / smoke.CHECKPOINT_MANIFEST).read_bytes()
    another = tmp_path / ".another.partial"
    another.mkdir()
    with pytest.raises(FileExistsError):
        smoke.publish_checkpoint(another, published)
    assert (published / smoke.CHECKPOINT_MANIFEST).read_bytes() == manifest
    smoke.verify_checkpoint(published)


def test_interrupted_publication_preserves_previous_generation(
    checkpoint_stage, tmp_path, monkeypatch
):
    first = tmp_path / "checkpoint-000001"
    smoke.publish_checkpoint(checkpoint_stage, first)
    last_good = smoke.verify_checkpoint(first)
    partial = tmp_path / ".second.partial"
    partial.mkdir()
    for name in smoke.REQUIRED_CHECKPOINT_FILES:
        (partial / name).write_bytes((first / name).read_bytes())
    second = tmp_path / "checkpoint-000002"
    rename = Path.rename

    def fail_final_rename(source, destination):
        if source == partial:
            raise OSError("simulated interrupted disk operation")
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_final_rename)
    with pytest.raises(OSError, match="interrupted"):
        smoke.publish_checkpoint(partial, second)
    assert not second.exists()
    assert partial.is_dir()
    assert smoke.verify_checkpoint(first) == last_good


def test_symlink_is_not_imported_into_a_checkpoint(checkpoint_stage, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_bytes(b"not part of a checkpoint")
    try:
        (checkpoint_stage / "linked.txt").symlink_to(secret)
    except OSError:
        pytest.skip("This Windows account cannot create test symlinks")
    with pytest.raises(RuntimeError, match="symlink"):
        smoke.publish_checkpoint(checkpoint_stage, tmp_path / "checkpoint-000002")
    assert secret.read_bytes() == b"not part of a checkpoint"


def test_manifest_never_selects_arbitrary_external_files(checkpoint_stage, tmp_path):
    published = tmp_path / "checkpoint-000002"
    smoke.publish_checkpoint(checkpoint_stage, published)
    manifest_path = published / smoke.CHECKPOINT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../outside.txt"] = {"size": 4, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="SHA256"):
        smoke.verify_checkpoint(published)


@pytest.mark.parametrize(
    "expected,actual",
    [
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (float("inf"), 1.0),
        (1.0, float("inf")),
        (1.0, 1.2),
    ],
)
def test_reload_with_nonfinite_or_changed_loss_cannot_pass(expected, actual):
    with pytest.raises(RuntimeError, match="loss"):
        smoke.validate_reload("same-adapter", "same-adapter", expected, actual)


def test_wrong_adapter_is_rejected_even_with_a_matching_loss():
    with pytest.raises(RuntimeError, match="digest"):
        smoke.validate_reload("saved-adapter", "wrong-adapter", 1.0, 1.0)


def test_bf16_reload_tolerance_accepts_a_small_finite_difference():
    smoke.validate_reload("same-adapter", "same-adapter", 1.0, 1.01)


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--local-files-only", "--revision", "main"], "40-hex"),
        (["--local-files-only", "--revision", "v1"], "40-hex"),
        (["--local-files-only", "--revision", "1234"], "40-hex"),
        (["--steps", "3"], "invalid choice"),
        (["--timeout-minutes", "31"], "between 1 and 30"),
    ],
)
def test_invalid_recipe_is_rejected_without_importing_or_installing_torch(arguments, message):
    script = Path(smoke.__file__)
    # -S disables all site packages: validation must work with only the standard library.
    result = subprocess.run(
        [sys.executable, "-S", str(script), *arguments],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "torch" not in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Bootstrap targets the Linux GPU VM")
def test_bootstrap_repairs_an_interrupted_torch_install_and_reuses_a_completed_stack(tmp_path):
    """Stub executables prove the install retry path without downloading packages."""
    project = tmp_path / "fake persistent workspace"
    (project / "tools").mkdir(parents=True)
    for relative in (
        "requirements-vision.txt",
        "tools/vision_gpu_probe.py",
        "tools/vision_lora_smoke.py",
    ):
        (project / relative).touch()
    executable = project / ".venv-vision" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ $# -eq 1 && "$1" == "-" ]]; then
  cat > /dev/null
  test -f "$CHAMELEON_ROOT/cuda-ready"
  exit $?
fi
printf '%s\\n' "$*" >> "$CHAMELEON_ROOT/calls.log"
if [[ "$*" == "-m pip install torch=="* ]]; then
  touch "$CHAMELEON_ROOT/cuda-ready"
fi
if [[ "$*" == "-m pip freeze --all" ]]; then
  printf '%s\\n' 'stub-only==1'
fi
if [[ "$1" == "tools/vision_gpu_probe.py" ]]; then
  test -f "$CHAMELEON_ROOT/cuda-ready"
fi
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 'stub GPU; no real hardware used'\n", encoding="utf-8")
    nvidia_smi.chmod(0o700)
    environment = {
        **os.environ,
        "CHAMELEON_ROOT": str(project),
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
    }
    bootstrap = Path(smoke.__file__).with_name("brev_bootstrap.sh")
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(bootstrap)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    calls = (project / "calls.log").read_text().splitlines()
    assert sum(line.startswith("-m pip install torch==") for line in calls) == 1
    assert calls.count("-m pip install -r requirements-vision.txt") == 2
    assert sum(line.startswith("tools/vision_gpu_probe.py") for line in calls) == 2
    assert (project / "vision-env.sh").is_file()


def test_three_image_fixture_uses_source_a_b_order_without_importing_model_libraries(
    tmp_path, monkeypatch
):
    from PIL import Image

    original = builtins.__import__

    def no_model_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "huggingface_hub"}:
            raise AssertionError("Fixture preparation imported a model library")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_import)
    example = smoke.synthetic_example(tmp_path, "source-ab")
    assert example["image_count"] == 3
    assert example["task_model_trained"] is False
    assert [Path(path).name for path in example["images"]] == [
        "synthetic-source.png",
        "synthetic-A.png",
        "synthetic-B.png",
    ]
    with (
        Image.open(example["images"][0]) as source,
        Image.open(example["images"][1]) as a,
        Image.open(example["images"][2]) as b,
    ):
        assert source.resize(a.size).tobytes() == a.tobytes()
        assert a.tobytes() != b.tobytes()
    assert json.loads(example["response"]) == {"choice": "A", "issues": []}


def test_task_shaped_fixture_cli_makes_no_gpu_compatibility_claim(tmp_path):
    output = tmp_path / "three-image-fixture"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.vision_lora_smoke",
            "--scenario",
            "source-ab",
            "--prepare-only",
            "--out-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "prepared-report.json").read_text())
    assert report["fixture_prepared"] is True
    assert report["gpu_compatibility_validated"] is False
    assert report["task_model_trained"] is False
    assert not (output / "published").exists()


def test_synthetic_generation_schema_and_target_match_are_reported_separately():
    correct = smoke.strict_synthetic_verdict('{"choice":"A","issues":[]}')
    wrong = smoke.strict_synthetic_verdict('{"choice":"B","issues":[]}')
    invalid = smoke.strict_synthetic_verdict('{"choice":["A"],"issues":[]}')
    assert correct["valid"] and correct["matches_synthetic_target"]
    assert wrong["valid"] and not wrong["matches_synthetic_target"]
    assert not invalid["valid"] and invalid["fallback_required"]
    assert all(result["malformed_output_rejected"] for result in (correct, wrong, invalid))


@pytest.mark.skipif(os.name != "posix", reason="Bootstrap repairs target the Linux GPU VM")
def test_partial_venv_missing_pip_fails_actionably_then_repairs_after_prerequisite(tmp_path):
    project = tmp_path / "partial environment workspace"
    (project / "tools").mkdir(parents=True)
    for relative in (
        "requirements-vision.txt",
        "tools/vision_gpu_probe.py",
        "tools/vision_lora_smoke.py",
    ):
        (project / relative).touch()
    executable = project / ".venv-vision" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$CHAMELEON_ROOT/calls.log"
if [[ "$*" == "-m pip --version" ]]; then
  test -f "$CHAMELEON_ROOT/pip-ready"
  exit $?
fi
if [[ "$*" == "-m ensurepip --upgrade" ]]; then
  touch "$CHAMELEON_ROOT/pip-ready"
fi
if [[ $# -eq 1 && "$1" == "-" ]]; then
  cat > /dev/null
fi
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    original = executable.read_bytes()
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    (fake_modules / "ensurepip.py").write_text(
        "import os\nif os.environ['PREREQ_SUPPORTED'] == '0':\n"
        "    raise ImportError('simulated missing ensurepip')\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 'stub GPU; no real hardware used'\n", encoding="utf-8")
    nvidia_smi.chmod(0o700)
    environment = {
        **os.environ,
        "CHAMELEON_ROOT": str(project),
        "PREREQ_SUPPORTED": "0",
        "PYTHONPATH": str(fake_modules),
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
    }
    bootstrap = Path(smoke.__file__).with_name("brev_bootstrap.sh")
    result = subprocess.run(
        ["bash", str(bootstrap)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert "-venv" in result.stderr and "did not install system packages" in result.stderr
    assert executable.read_bytes() == original
    assert "-m pip install" not in (project / "calls.log").read_text()
    environment["PREREQ_SUPPORTED"] = "1"
    result = subprocess.run(
        ["bash", str(bootstrap)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "pip-ready").exists()
    assert "-m ensurepip --upgrade" in (project / "calls.log").read_text()
    assert executable.read_bytes() == original
