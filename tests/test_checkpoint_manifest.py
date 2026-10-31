"""Checkpoint transport works without optional training modules or model libraries."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import checkpoint_manifest as checkpoint


@pytest.fixture
def stage(tmp_path):
    directory = tmp_path / ".checkpoint.partial"
    directory.mkdir()
    for name in checkpoint.REQUIRED_CHECKPOINT_FILES:
        (directory / name).write_bytes(("opaque transport: " + name).encode() + b"\x00\xff")
    return directory


def test_published_checkpoint_retains_existing_manifest_format_and_exact_bytes(stage, tmp_path):
    expected = {path.name: path.read_bytes() for path in stage.iterdir()}
    target = tmp_path / "checkpoint-2"
    checkpoint.publish_checkpoint(stage, target)
    manifest = checkpoint.verify_checkpoint(target)
    assert manifest == {
        "format_version": 1,
        "files": {
            name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in expected.items()
        },
    }
    assert json.loads((target / checkpoint.CHECKPOINT_MANIFEST).read_text()) == manifest
    assert all((target / name).read_bytes() == data for name, data in expected.items())


@pytest.mark.parametrize("missing", sorted(checkpoint.REQUIRED_CHECKPOINT_FILES))
def test_incomplete_checkpoint_is_never_published(stage, tmp_path, missing):
    (stage / missing).unlink()
    target = tmp_path / "checkpoint-2"
    with pytest.raises(RuntimeError, match="Incomplete"):
        checkpoint.publish_checkpoint(stage, target)
    assert not target.exists() and stage.is_dir()


def test_corruption_is_detected_and_previous_generation_is_never_overwritten(stage, tmp_path):
    target = tmp_path / "checkpoint-2"
    checkpoint.publish_checkpoint(stage, target)
    previous = (target / checkpoint.CHECKPOINT_MANIFEST).read_bytes()
    replacement = tmp_path / ".replacement.partial"
    replacement.mkdir()
    with pytest.raises(FileExistsError):
        checkpoint.publish_checkpoint(replacement, target)
    assert (target / checkpoint.CHECKPOINT_MANIFEST).read_bytes() == previous
    weight = target / "adapter_model.safetensors"
    weight.write_bytes(b"x" * weight.stat().st_size)
    with pytest.raises(RuntimeError, match="SHA256"):
        checkpoint.verify_checkpoint(target)


def test_backup_imports_in_isolation_without_training_drafts_or_model_libraries(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    original = Path(__file__).resolve().parents[1] / "tools"
    for name in ("checkpoint_backup.py", "checkpoint_manifest.py", "brev_backup.py"):
        shutil.copyfile(original / name, tools / name)
    script = """
import builtins, sys
original_import = builtins.__import__
forbidden = {'torch', 'transformers', 'peft', 'experiments'}
def safe_import(name, *args, **kwargs):
    if name.split('.')[0] in forbidden or name == 'tools.vision_lora_smoke':
        raise AssertionError('optional training dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = safe_import
import tools.checkpoint_backup
import tools.checkpoint_manifest
assert 'tools.vision_lora_smoke' not in sys.modules
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
print('standalone stdlib backup import passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert "standalone stdlib backup import passed" in result.stdout
