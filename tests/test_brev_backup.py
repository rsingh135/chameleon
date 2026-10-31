"""Off-VM snapshots must remain recoverable after failed saves and transfers."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import brev_backup as backup


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "pulled checkpoint"
    (directory / "nested").mkdir(parents=True)
    (directory / "empty directory").mkdir()
    (directory / "adapter.safetensors").write_bytes(b"adapter-weights\x00\xff")
    (directory / "nested" / "trainer.json").write_text('{"step": 25}', encoding="utf-8")
    (directory / "zero").write_bytes(b"")
    return directory


def generation(receipt):
    return Path(receipt["generation_dir"])


def rewrite_manifest(path, change):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    change(manifest)
    data = json.dumps(manifest).encode()
    manifest_path.write_bytes(data)
    receipt_path = path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_round_trip_is_independent_of_the_lost_source(source, tmp_path):
    receipt = backup.snapshot(source, tmp_path / "backups")
    saved = generation(receipt)
    assert backup.verify(saved) == receipt
    (source / "adapter.safetensors").write_bytes(b"changed after the backup")
    (source / "nested" / "trainer.json").unlink()
    restored = tmp_path / "fresh restore"
    result = backup.restore(saved, restored)
    assert result["generation"] == receipt["generation"]
    assert (restored / "adapter.safetensors").read_bytes() == b"adapter-weights\x00\xff"
    assert json.loads((restored / "nested" / "trainer.json").read_text())["step"] == 25
    assert (restored / "empty directory").is_dir()
    assert (restored / "zero").read_bytes() == b""
    assert receipt["file_count"] == 3


def test_repeated_snapshots_never_overwrite_prior_generations(source, tmp_path):
    root = tmp_path / "backups"
    first, second = backup.snapshot(source, root), backup.snapshot(source, root)
    assert first["generation"] != second["generation"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert backup.verify(generation(first)) == first
    assert backup.verify(generation(second)) == second
    assert set(root.iterdir()) == {generation(first), generation(second)}


@pytest.mark.parametrize("change", ["mutate", "remove", "add"])
def test_source_changes_during_copy_do_not_publish_or_destroy_last_good(
    source, tmp_path, monkeypatch, change
):
    root = tmp_path / "backups"
    good = backup.snapshot(source, root)
    copy = backup._copy

    def changing_copy(src, destination, manifest):
        copy(src, destination, manifest)
        if change == "mutate":
            (src / "adapter.safetensors").write_bytes(b"new content")
        elif change == "remove":
            (src / "zero").unlink()
        else:
            (src / "new file").write_bytes(b"new content")

    monkeypatch.setattr(backup, "_copy", changing_copy)
    with pytest.raises((backup.BackupError, OSError)):
        backup.snapshot(source, root)
    assert list(root.iterdir()) == [generation(good)]
    assert backup.verify(generation(good)) == good


def test_corrupt_staging_copy_is_rejected(source, tmp_path, monkeypatch):
    root = tmp_path / "backups"
    copy = backup._copy

    def corrupt_copy(src, destination, manifest):
        copy(src, destination, manifest)
        (destination / "adapter.safetensors").write_bytes(b"truncated")

    monkeypatch.setattr(backup, "_copy", corrupt_copy)
    with pytest.raises(backup.BackupError, match="manifest"):
        backup.snapshot(source, root)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("failure", ["rename", "receipt"])
def test_atomic_publication_failure_preserves_last_good(source, tmp_path, monkeypatch, failure):
    root = tmp_path / "backups"
    good = backup.snapshot(source, root)
    if failure == "rename":

        def reject_publish(*_args):
            raise OSError("injected rename failure")

        monkeypatch.setattr(backup, "_publish", reject_publish)
    else:
        write = backup._write

        def fail_receipt(path, data):
            if path.name == "receipt.json":
                raise OSError("injected disk-full receipt write")
            write(path, data)

        monkeypatch.setattr(backup, "_write", fail_receipt)
    with pytest.raises(OSError, match="injected"):
        backup.snapshot(source, root)
    assert list(root.iterdir()) == [generation(good)]
    assert backup.verify(generation(good)) == good


@pytest.mark.parametrize("change", ["contents", "missing", "extra", "empty_dir", "metadata"])
def test_verify_rejects_corruption_and_exact_tree_changes(source, tmp_path, change):
    saved = generation(backup.snapshot(source, tmp_path / "backups"))
    if change == "contents":
        data = saved / "data" / "adapter.safetensors"
        data.write_bytes(b"X" * data.stat().st_size)
    elif change == "missing":
        (saved / "data" / "zero").unlink()
    elif change == "extra":
        (saved / "data" / "extra").write_bytes(b"unexpected")
    elif change == "empty_dir":
        (saved / "data" / "unexpected").mkdir()
    else:
        (saved / "extra-metadata").write_bytes(b"unexpected")
    with pytest.raises(backup.BackupError):
        backup.verify(saved)


@pytest.mark.parametrize(
    "unsafe",
    ["../outside", "/absolute", "C:/Windows/file", "a\\b", "a/../b", "CON", "x:stream", "a//b"],
)
def test_manifest_paths_are_validated_even_if_receipt_hash_is_updated(source, tmp_path, unsafe):
    saved = generation(backup.snapshot(source, tmp_path / "backups"))
    rewrite_manifest(saved, lambda value: value["files"][0].update(path=unsafe))
    with pytest.raises(backup.BackupError, match="path"):
        backup.verify(saved)


def test_manifest_requires_explicit_parents_and_no_case_collisions(source, tmp_path):
    first = generation(backup.snapshot(source, tmp_path / "backups"))
    rewrite_manifest(first, lambda value: value.update(directories=[]))
    with pytest.raises(backup.BackupError, match="parent"):
        backup.verify(first)
    second = generation(backup.snapshot(source, tmp_path / "backups"))
    rewrite_manifest(second, lambda value: value["files"].append({**value["files"][0]}))
    with pytest.raises(backup.BackupError, match="colliding"):
        backup.verify(second)


def test_manifest_checksum_and_duplicate_json_keys_are_rejected(source, tmp_path):
    saved = generation(backup.snapshot(source, tmp_path / "backups"))
    (saved / "manifest.json").write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(backup.BackupError, match="Duplicate"):
        backup.verify(saved)
    saved = generation(backup.snapshot(source, tmp_path / "backups"))
    (saved / "manifest.json").write_text('{"schema_version":1,"directories":[],"files":[]}')
    with pytest.raises(backup.BackupError, match="checksum"):
        backup.verify(saved)


@pytest.mark.parametrize("kind", ["file", "directory", "source_root", "metadata"])
def test_observed_symlinks_are_rejected(source, tmp_path, kind):
    root = tmp_path / "backups"
    if kind == "metadata":
        saved = generation(backup.snapshot(source, root))
        target = saved / "receipt.json"
        external = tmp_path / "external-receipt"
        target.rename(external)
    elif kind == "source_root":
        target, external = tmp_path / "linked source", source
    else:
        target = source / "link"
        external = source / ("nested" if kind == "directory" else "zero")
    try:
        target.symlink_to(external, target_is_directory=external.is_dir())
    except OSError as error:
        pytest.skip(f"Creating symlinks is unavailable: {error}")
    with pytest.raises(backup.BackupError, match="Links"):
        if kind == "metadata":
            backup.verify(saved)
        else:
            backup.snapshot(target if kind == "source_root" else source, root)


def test_hardlinks_are_rejected(source, tmp_path):
    try:
        os.link(source / "zero", tmp_path / "external-hardlink")
    except OSError as error:
        pytest.skip(f"Creating hardlinks is unavailable: {error}")
    with pytest.raises(backup.BackupError, match="hardlinked"):
        backup.snapshot(source, tmp_path / "backups")


@pytest.mark.parametrize("overlap", ["same", "inside", "outside"])
def test_overlapping_snapshot_paths_are_rejected(source, overlap):
    destination = {"same": source, "inside": source / "backups", "outside": source.parent}[overlap]
    with pytest.raises(backup.BackupError, match="overlap"):
        backup.snapshot(source, destination)
    assert not (source / "backups").exists()


def test_restore_never_overwrites_an_existing_directory(source, tmp_path):
    saved = generation(backup.snapshot(source, tmp_path / "backups"))
    destination = tmp_path / "restore"
    destination.mkdir()
    (destination / "keep").write_text("important")
    with pytest.raises(backup.BackupError, match="already exist"):
        backup.restore(saved, destination)
    assert (destination / "keep").read_text() == "important"
    with pytest.raises(backup.BackupError, match="overlap"):
        backup.restore(saved, saved / "data" / "restore")


def test_failed_restore_leaves_destination_absent_and_backup_intact(source, tmp_path, monkeypatch):
    receipt = backup.snapshot(source, tmp_path / "backups")
    saved, destination = generation(receipt), tmp_path / "restore"

    def partial_copy(_source, target, _manifest):
        target.mkdir()
        (target / "partial").write_bytes(b"partial")
        raise OSError("injected interrupted copy")

    monkeypatch.setattr(backup, "_copy", partial_copy)
    with pytest.raises(OSError, match="injected"):
        backup.restore(saved, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".brev-staging-*"))
    assert backup.verify(saved) == receipt


def test_cli_emits_receipts_and_reports_failed_verification(source, tmp_path, capsys):
    assert (
        backup.main(["snapshot", "--source", str(source), "--backup-root", str(tmp_path / "b")])
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    saved = receipt["generation_dir"]
    assert backup.main(["verify", "--generation", saved]) == 0
    assert json.loads(capsys.readouterr().out)["generation"] == receipt["generation"]
    destination = tmp_path / "restore"
    assert backup.main(["restore", "--generation", saved, "--destination", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["restored_to"] == str(destination)
    (Path(saved) / "data" / "zero").write_bytes(b"corrupt")
    assert backup.main(["verify", "--generation", saved]) == 1
    assert "manifest" in json.loads(capsys.readouterr().err)["error"]
