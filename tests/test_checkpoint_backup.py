"""Published checkpoint copies must retain every resume byte without loading it."""

import hashlib
import json
from pathlib import Path

import pytest

from tools import brev_backup as storage
from tools import checkpoint_backup as backup
from tools.checkpoint_manifest import publish_checkpoint


@pytest.fixture
def source(tmp_path):
    stage = tmp_path / "checkpoint.partial"
    stage.mkdir()
    (stage / "adapter_model.safetensors").write_bytes(b"opaque adapter weights\x00\xff")
    (stage / "training-state.pt").write_bytes(b"opaque optimizer and RNG, not a pickle")
    (stage / "optimizer.pt").write_bytes(b"extra optimizer state")
    (stage / "scheduler.pt").write_bytes(b"extra scheduler state")
    (stage / "rng_state.pth").write_bytes(b"extra RNG state")
    (stage / "verified-model.json").write_text('{"revision":"' + "a" * 40 + '"}')
    (stage / "progress.json").write_text(
        json.dumps({"completed_steps": 24, "base_model": "known/base", "base_revision": "a" * 40})
    )
    (stage / "adapter_config.json").write_text('{"base_model_name_or_path":"known/base"}')
    destination = tmp_path / "published" / "checkpoint-000024"
    destination.parent.mkdir()
    publish_checkpoint(stage, destination)
    return destination


def saved(receipt):
    return Path(receipt["generation_dir"])


def test_round_trip_preserves_all_resume_files_and_base_references(source, tmp_path):
    receipt = backup.backup_checkpoint(source, tmp_path / "backups")
    assert backup.verify_checkpoint_backup(saved(receipt)) == receipt
    assert receipt["checkpoint"]["completed_steps"] == 24
    assert receipt["checkpoint"]["base_revision"] == "a" * 40
    assert {"optimizer.pt", "scheduler.pt", "rng_state.pth", "verified-model.json"} <= set(
        receipt["checkpoint"]["files"]
    )
    target = tmp_path / "restored checkpoint"
    restored = backup.restore_checkpoint(saved(receipt), target)
    assert restored["checkpoint"] == receipt["checkpoint"]
    for path in source.iterdir():
        assert (target / path.name).read_bytes() == path.read_bytes()


def test_repeated_saves_preserve_prior_generations_and_source(source, tmp_path):
    original = storage._inventory(source)
    root = tmp_path / "backups"
    first = backup.backup_checkpoint(source, root)
    second = backup.backup_checkpoint(source, root)
    assert first["generation"] != second["generation"]
    assert backup.verify_checkpoint_backup(saved(first)) == first
    assert backup.verify_checkpoint_backup(saved(second)) == second
    assert storage._inventory(source) == original


def test_incomplete_or_unpublished_checkpoint_is_rejected_before_publication(source, tmp_path):
    (source / "training-state.pt").unlink()
    with pytest.raises(backup.BackupError, match="incomplete or corrupt"):
        backup.backup_checkpoint(source, tmp_path / "backups")
    assert not (tmp_path / "backups").exists()


@pytest.mark.parametrize("change", ["mutate", "add", "remove"])
def test_source_mutation_keeps_last_good_and_retains_failed_staging(
    source, tmp_path, monkeypatch, change
):
    root = tmp_path / "backups"
    first = backup.backup_checkpoint(source, root)
    copy = storage._copy

    def mutate(src, destination, manifest):
        copy(src, destination, manifest)
        if change == "mutate":
            (src / "training-state.pt").write_bytes(b"state changed")
        elif change == "add":
            (src / "new-state").write_bytes(b"state added")
        else:
            (src / "optimizer.pt").unlink()

    monkeypatch.setattr(storage, "_copy", mutate)
    with pytest.raises((backup.BackupError, OSError)):
        backup.backup_checkpoint(source, root)
    assert backup.verify_checkpoint_backup(saved(first)) == first
    assert len(list(root.glob(".checkpoint-staging-*"))) == 1
    assert len([p for p in root.iterdir() if not p.name.startswith(".")]) == 1


def test_failed_publication_does_not_delete_staging_or_prior_backup(source, tmp_path, monkeypatch):
    root = tmp_path / "backups"
    first = backup.backup_checkpoint(source, root)

    def fail(*_args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(storage, "_publish", fail)
    with pytest.raises(OSError, match="injected"):
        backup.backup_checkpoint(source, root)
    assert backup.verify_checkpoint_backup(saved(first)) == first
    assert len(list(root.glob(".checkpoint-staging-*"))) == 1


def test_corruption_is_rejected_before_restore(source, tmp_path):
    receipt = backup.backup_checkpoint(source, tmp_path / "backups")
    (saved(receipt) / "data" / "training-state.pt").write_bytes(b"corrupt")
    destination = tmp_path / "restored"
    with pytest.raises(backup.BackupError):
        backup.restore_checkpoint(saved(receipt), destination)
    assert not destination.exists()


def test_restore_never_overwrites_existing_data(source, tmp_path):
    receipt = backup.backup_checkpoint(source, tmp_path / "backups")
    destination = tmp_path / "restore"
    destination.mkdir()
    (destination / "keep").write_bytes(b"important")
    with pytest.raises(backup.BackupError, match="already exist"):
        backup.restore_checkpoint(saved(receipt), destination)
    assert (destination / "keep").read_bytes() == b"important"


@pytest.mark.parametrize("placement", ["source", "file", "backup-root"])
def test_symlinks_and_escapes_are_refused(source, tmp_path, placement):
    link = tmp_path / "link"
    target = source if placement != "file" else source / "optimizer.pt"
    try:
        link.symlink_to(target, target_is_directory=placement != "file")
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer privileges")
    if placement == "file":
        (source / "escape").symlink_to(target)
        candidate, root = source, tmp_path / "backups"
    elif placement == "backup-root":
        candidate, root = source, link
    else:
        candidate, root = link, tmp_path / "backups"
    with pytest.raises(backup.BackupError, match="Links|symlinks|overlap"):
        backup.backup_checkpoint(candidate, root)


def test_overlapping_paths_are_refused(source):
    with pytest.raises(backup.BackupError, match="overlap"):
        backup.backup_checkpoint(source, source / "backups")


def test_hash_bound_training_receipt_weights_are_preserved_without_exact_resume_claim(tmp_path):
    source = tmp_path / "checkpoint-8"
    source.mkdir()
    weights = b"new adapter weights, no optimizer/RNG saved"
    (source / "adapter_model.safetensors").write_bytes(weights)
    (source / "adapter_config.json").write_text('{"base_model_name_or_path":"known/base"}')
    receipt = {
        "actual_training": True,
        "step": 8,
        "losses": [1.0] * 8,
        "base_revision": "c" * 40,
        "dataset_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
        "adapter_sha256": hashlib.sha256(weights).hexdigest(),
    }
    (source / "training_receipt.json").write_text(json.dumps(receipt))
    result = backup.backup_checkpoint(source, tmp_path / "backups")
    assert result["checkpoint"]["source_format"] == "adapter_training_receipt"
    assert result["checkpoint"]["completed_steps"] == 8
    assert result["checkpoint"]["exact_resume_available"] is False
    assert result["checkpoint"]["resume_state_files"] == []
    destination = tmp_path / "restored"
    restored = backup.restore_checkpoint(result["generation_dir"], destination)
    assert restored["checkpoint"] == result["checkpoint"]
    assert (destination / "adapter_model.safetensors").read_bytes() == weights
    (source / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(backup.BackupError, match="does not bind"):
        backup.backup_checkpoint(source, tmp_path / "backups")


def test_cli_backup_verify_restore_without_model_import(source, tmp_path, capsys):
    assert (
        backup.main(["backup", "--source", str(source), "--backup-root", str(tmp_path / "backups")])
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert backup.main(["verify", "--generation", receipt["generation_dir"]]) == 0
    capsys.readouterr()
    target = tmp_path / "restored"
    assert (
        backup.main(
            ["restore", "--generation", receipt["generation_dir"], "--destination", str(target)]
        )
        == 0
    )


@pytest.fixture
def guardian(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    checkpoint = root / "training-run" / "checkpoint-8"
    checkpoint.mkdir(parents=True)
    weights = b"guardian fixture adapter weights"
    (checkpoint / "adapter_model.safetensors").write_bytes(weights)
    (checkpoint / "adapter_config.json").write_text('{"base_model_name_or_path":"known/base"}')
    (checkpoint / "training_receipt.json").write_text(
        json.dumps(
            {
                "actual_training": True,
                "step": 8,
                "losses": [1.0] * 8,
                "base_revision": "c" * 40,
                "dataset_sha256": "a" * 64,
                "recipe_sha256": "b" * 64,
                "adapter_sha256": hashlib.sha256(weights).hexdigest(),
            }
        )
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(backup, "DEFAULT_RUNS_ROOT", root)
    monkeypatch.setattr(backup, "monotonic", lambda: clock["now"])

    def advance(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(backup, "sleep", advance)
    return {
        "root": root,
        "checkpoint": checkpoint,
        "weights": weights,
        "backups": tmp_path / "backups",
        "state": tmp_path / "guardian" / "state.json",
        "clock": clock,
        "advance": advance,
    }


def run_guardian(fixture, **kwargs):
    return backup.watch_checkpoints(
        fixture["root"],
        state_path=fixture["state"],
        backup_root=fixture["backups"],
        max_seconds=kwargs.pop("max_seconds", 10),
        poll_seconds=5,
        **kwargs,
    )


def test_guardian_saves_new_checkpoint_once_and_stops_quietly(guardian, capsys):
    result = run_guardian(guardian)
    assert result["status"] == "stopped" and result["stop_reason"] == "max_seconds"
    assert len(result["snapshots"]) == 1
    assert not result["errors"]
    persisted = json.loads(guardian["state"].read_text())
    assert persisted == result
    assert len(list(guardian["backups"].iterdir())) == 1
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events == ["guardian_started", "checkpoint_backed_up", "guardian_stopped"]


def test_guardian_recognizes_existing_verified_backup_and_restarts_without_duplicates(guardian):
    receipt = backup.backup_checkpoint(guardian["checkpoint"], guardian["backups"])
    result = run_guardian(guardian)
    assert result["snapshots"][0]["generation_dir"] == receipt["generation_dir"]
    again = run_guardian(guardian, max_seconds=1)
    assert len(again["snapshots"]) == 1
    assert len(list(guardian["backups"].iterdir())) == 1


def test_guardian_retries_incomplete_mutating_save_without_losing_progress(
    guardian, monkeypatch, capsys
):
    weight = guardian["checkpoint"] / "adapter_model.safetensors"
    weight.write_bytes(b"incomplete save")

    def finish_save(seconds):
        weight.write_bytes(guardian["weights"])
        guardian["advance"](seconds)

    monkeypatch.setattr(backup, "sleep", finish_save)
    result = run_guardian(guardian, max_seconds=15)
    assert len(result["snapshots"]) == 1 and not result["errors"]
    assert backup.verify_checkpoint_backup(result["snapshots"][0]["generation_dir"])
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events.count("checkpoint_waiting") == 1
    assert events.count("checkpoint_backed_up") == 1


def test_guardian_expired_deadline_prevents_new_backups(guardian):
    result = run_guardian(guardian, until_utc="2000-01-01T00:00:00Z")
    assert result["stop_reason"] == "until_utc"
    assert not result["snapshots"]
    assert not guardian["backups"].exists()
