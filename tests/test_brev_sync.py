"""Off-VM transfer contracts with a fake Brev CLI and real local hash verification."""

import ast
import json
import os
import shlex
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tools import brev_backup, brev_guard, brev_sync, vision_lora_smoke

NOW = datetime(2026, 10, 29, 16, tzinfo=brev_guard.UTC)
ROOT = "/home/ubuntu/workspace/chameleon"
SOURCE = ROOT + "/training_runs/smoke/published/checkpoint-000002"


@pytest.fixture
def config():
    return brev_guard.Config(
        "instance-123", "chameleon", "org-123", "our-org", NOW, NOW + timedelta(hours=2)
    )


@pytest.fixture
def remote_generation(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in vision_lora_smoke.REQUIRED_CHECKPOINT_FILES:
        (stage / name).write_bytes((name + " test bytes").encode())
    leaf = tmp_path / "published" / "checkpoint-000002"
    leaf.parent.mkdir()
    vision_lora_smoke.publish_checkpoint(stage, leaf)
    receipt = brev_backup.snapshot(leaf, tmp_path / "remote-exports")
    local_generation = Path(receipt["generation_dir"])
    stored = json.loads((local_generation / "receipt.json").read_text())
    stored["source"] = SOURCE
    (local_generation / "receipt.json").write_text(json.dumps(stored))
    receipt = {
        **stored,
        "generation_dir": ROOT + "/.export-snapshots/" + stored["generation"],
    }
    return local_generation, receipt


class FakeCLI:
    def __init__(self, generation, receipt, *, placement="nested", status="RUNNING"):
        self.generation = generation
        self.receipt = dict(receipt)
        self.placement = placement
        self.status = status
        self.instance_id = "instance-123"
        self.calls = []
        self.copy_error = False
        self.corrupt_copy = False

    def __call__(self, command, timeout):
        self.calls.append((command, timeout))
        if "orgs" in command:
            return json.dumps([{"id": "org-123", "name": "our-org", "is_active": True}])
        if "ls" in command:
            return json.dumps(
                {
                    "workspaces": [
                        {"id": self.instance_id, "name": "chameleon", "status": self.status}
                    ]
                }
            )
        if "exec" in command:
            assert command[:3] == ["brev", "--org", "our-org"]
            words = shlex.split(command[-1])
            assert words[:2] == ["python3", "-c"]
            ast.parse(words[2])
            assert "verify_checkpoint(source)" in words[2]
            return "Brev informational output\n" + brev_sync.MARKER + json.dumps(self.receipt)
        if "copy" in command:
            assert command[:3] == ["brev", "--org", "our-org"]
            # Live Brev copy has only --host, accepts exactly two paths, and
            # handles directory recursion internally. Public -r examples are stale.
            assert command[3] == "copy"
            assert len(command) == 6
            assert "-r" not in command
            if self.copy_error:
                raise brev_guard.GuardError("transfer failed")
            incoming = Path(command[-1])
            target = (
                incoming / self.receipt["generation"] if self.placement == "nested" else incoming
            )
            if self.placement == "incomplete":
                (incoming / "random.partial").write_text("partial data")
            else:
                shutil.copytree(self.generation, target, dirs_exist_ok=True)
                if self.corrupt_copy:
                    (target / "data/adapter_model.safetensors").write_bytes(b"damaged")
            return ""
        raise AssertionError(f"Unexpected command: {command}")


def run_sync(config, fake, tmp_path, **kwargs):
    return brev_sync.sync_checkpoint(
        config,
        remote_root=ROOT,
        remote_source=SOURCE,
        backup_root=tmp_path / "backups",
        runner=fake,
        now=lambda: NOW,
        **kwargs,
    )


@pytest.mark.parametrize("placement", ["nested", "root"])
@pytest.mark.parametrize("status", ["RUNNING", "READY"])
def test_pull_verifies_and_publishes_one_checkpoint(
    config, remote_generation, tmp_path, placement, status
):
    fake = FakeCLI(*remote_generation, placement=placement, status=status)
    result = run_sync(config, fake, tmp_path)
    assert result["verified"]
    assert result["generation"] == remote_generation[1]["generation"]
    final = Path(result["generation_dir"])
    brev_backup.verify(final)
    vision_lora_smoke.verify_checkpoint(final / "data")
    assert all(timeout <= 120 for _, timeout in fake.calls)
    assert not any(word in {"start", "stop", "delete"} for call, _ in fake.calls for word in call)


def test_pulled_checkpoint_restores_without_remote_copy(config, remote_generation, tmp_path):
    """Recovery depends on the verified local generation, not VM disk survival."""
    remote, _ = remote_generation
    result = run_sync(config, FakeCLI(*remote_generation), tmp_path)
    shutil.rmtree(remote)
    destination = tmp_path / "restored-checkpoint"
    receipt = brev_backup.restore(Path(result["generation_dir"]), destination)
    assert receipt["restored_to"] == str(destination)
    assert receipt["generation"] == result["generation"]
    vision_lora_smoke.verify_checkpoint(destination)
    original = Path(result["generation_dir"]) / "data"
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == {
        path.name: path.read_bytes() for path in original.iterdir()
    }


@pytest.mark.parametrize("status", ["STOPPED", "DEPLOYING", "STOPPING", "UNKNOWN"])
def test_never_starts_or_uses_a_nonrunning_vm(config, remote_generation, tmp_path, status):
    fake = FakeCLI(*remote_generation, status=status)
    with pytest.raises(brev_sync.SyncError, match="already RUNNING/READY"):
        run_sync(config, fake, tmp_path)
    assert not any("exec" in call or "copy" in call for call, _ in fake.calls)


def test_wrong_id_refuses_transfer(config, remote_generation, tmp_path):
    fake = FakeCLI(*remote_generation)
    fake.instance_id = "wrong-id"
    with pytest.raises(brev_guard.IdentityError):
        run_sync(config, fake, tmp_path)
    assert not any("exec" in call or "copy" in call for call, _ in fake.calls)


@pytest.mark.parametrize(
    "source",
    [
        "/outside/published/checkpoint-000002",
        ROOT + "/training_runs/smoke/.checkpoint-000002.partial",
        ROOT + "/training_runs/smoke/published",
        ROOT + "/../escape/published/checkpoint-000002",
        ROOT + "/training_runs/smoke/published/checkpoint-000002\n;evil",
    ],
)
def test_unsafe_or_incomplete_source_path_does_not_contact_vm(
    config, remote_generation, tmp_path, source
):
    fake = FakeCLI(*remote_generation)
    with pytest.raises(brev_sync.SyncError):
        brev_sync.sync_checkpoint(
            config,
            remote_root=ROOT,
            remote_source=source,
            backup_root=tmp_path / "backups",
            runner=fake,
            now=lambda: NOW,
        )
    assert not fake.calls


def test_wrong_remote_generation_path_is_rejected(config, remote_generation, tmp_path):
    fake = FakeCLI(*remote_generation)
    fake.receipt["generation_dir"] = "/home/ubuntu/unrelated"
    with pytest.raises(brev_sync.SyncError, match="path/source"):
        run_sync(config, fake, tmp_path)
    assert not any("copy" in call for call, _ in fake.calls)


@pytest.mark.parametrize("mode", ["copy_error", "incomplete", "corrupt", "wrong_hash"])
def test_failed_transfer_never_publishes_or_changes_existing_backups(
    config, remote_generation, tmp_path, mode
):
    backup_root = tmp_path / "backups"
    previous = backup_root / "previous-generation"
    previous.mkdir(parents=True)
    (previous / "keep.bin").write_bytes(b"existing verified data")
    fake = FakeCLI(*remote_generation, placement="incomplete" if mode == "incomplete" else "nested")
    fake.copy_error = mode == "copy_error"
    fake.corrupt_copy = mode == "corrupt"
    if mode == "wrong_hash":
        fake.receipt["manifest_sha256"] = "0" * 64
    with pytest.raises((brev_sync.SyncError, brev_guard.GuardError, brev_backup.BackupError)):
        run_sync(config, fake, tmp_path)
    assert (previous / "keep.bin").read_bytes() == b"existing verified data"
    assert not (backup_root / remote_generation[1]["generation"]).exists()


def test_expired_deadline_prevents_all_remote_commands(config, remote_generation, tmp_path):
    fake = FakeCLI(*remote_generation)
    with pytest.raises(brev_sync.SyncError, match="deadline"):
        brev_sync.sync_checkpoint(
            config,
            remote_root=ROOT,
            remote_source=SOURCE,
            backup_root=tmp_path / "backups",
            runner=fake,
            now=lambda: config.deadline,
        )
    assert not fake.calls


def test_remaining_allowance_bounds_each_subprocess(config, remote_generation, tmp_path):
    fake = FakeCLI(*remote_generation)
    current = config.deadline - timedelta(seconds=7)
    brev_sync.sync_checkpoint(
        config,
        remote_root=ROOT,
        remote_source=SOURCE,
        backup_root=tmp_path / "backups",
        runner=fake,
        now=lambda: current,
    )
    assert all(timeout == 7 for _, timeout in fake.calls)


def test_remote_command_quotes_a_project_path_containing_apostrophe():
    root = brev_sync.remote_path("/home/ubuntu/workspace/owner's project")
    source = root / "training_runs/smoke/published/checkpoint-000002"
    words = shlex.split(brev_sync.remote_command(root, source))
    assert words[:2] == ["python3", "-c"]
    ast.parse(words[2])
    assert repr(str(root)) in words[2]


def test_duplicate_or_bad_marker_is_not_a_verified_receipt(remote_generation):
    marker = brev_sync.MARKER + json.dumps(remote_generation[1])
    for output in (marker + "\n" + marker, brev_sync.MARKER + "not-json", "no receipt"):
        with pytest.raises(brev_sync.SyncError):
            brev_sync.parse_receipt(
                output, brev_sync.remote_path(ROOT), brev_sync.remote_path(SOURCE)
            )


def test_same_generation_is_never_overwritten(config, remote_generation, tmp_path):
    final = tmp_path / "backups" / remote_generation[1]["generation"]
    final.mkdir(parents=True)
    (final / "keep.bin").write_bytes(b"previous generation")
    fake = FakeCLI(*remote_generation)
    with pytest.raises(brev_sync.SyncError, match="already exists"):
        run_sync(config, fake, tmp_path)
    assert (final / "keep.bin").read_bytes() == b"previous generation"
    assert not any("copy" in call for call, _ in fake.calls)


def test_deadline_expiring_after_export_prevents_copy(config, remote_generation, tmp_path):
    fake = FakeCLI(*remote_generation)
    current = NOW

    def advancing_runner(command, timeout):
        nonlocal current
        result = fake(command, timeout)
        if "exec" in command:
            current = config.deadline
        return result

    with pytest.raises(brev_sync.SyncError, match="deadline"):
        brev_sync.sync_checkpoint(
            config,
            remote_root=ROOT,
            remote_source=SOURCE,
            backup_root=tmp_path / "backups",
            runner=advancing_runner,
            now=lambda: current,
        )
    assert not any("copy" in call for call, _ in fake.calls)
    assert not (tmp_path / "backups").exists()


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup is the WSL/Linux contract")
def test_posix_timeout_terminates_descendant_before_delayed_write(tmp_path):
    started = tmp_path / "child-started"
    marker = tmp_path / "orphan-marker"
    child = (
        "import time; from pathlib import Path; "
        f"Path({str(started)!r}).write_text('started'); "
        f"time.sleep(2); Path({str(marker)!r}).write_text('orphaned child')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(20)"
    )
    before = time.monotonic()
    try:
        brev_sync.run_transfer_command([sys.executable, "-c", parent], 0.8)
    except brev_guard.GuardError as error:
        assert "process group timed out" in str(error)
    else:
        raise AssertionError("The process exceeded its timeout without rejection")
    assert time.monotonic() - before < 5
    assert started.exists(), "The test must actually spawn the child before timing out"
    time.sleep(2.2)
    assert not marker.exists(), "A child process survived the parent timeout"
