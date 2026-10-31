"""Preserve published training checkpoints as verified, append-only local snapshots.

All checkpoint files are copied, including opaque optimizer/scheduler/RNG state.
Verification reads bytes and JSON only; it never imports Torch or unpickles state.
Base-model references in adapter_config.json, progress.json or training_receipt.json
are retained; large base caches are not downloaded or followed implicitly. Back up
a base cache once with tools.brev_backup rather than copying it for every adapter.

Failed operations retain their uniquely named partial staging directories. They
never remove, prune, or overwrite source checkpoints or previous snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

from tools import brev_backup as storage
from tools.checkpoint_manifest import verify_checkpoint

UTC = timezone.utc  # noqa: UP017 -- backup/guardian also supports Python 3.10.
DEFAULT_BACKUP_ROOT = Path(__file__).resolve().parents[1] / "local_backups/checkpoints"
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "output/visual-finetune/runs"
BackupError = storage.BackupError


def _checkpoint(directory: Path) -> dict:
    """Check a publication manifest or hash-bound receipt and summarize saved metadata."""
    if directory.name.endswith(".partial"):
        raise BackupError("Back up a published checkpoint, not an active partial save")
    if not (directory / "checkpoint-manifest.json").exists():
        return _receipt_checkpoint(directory)
    try:
        manifest = verify_checkpoint(directory)
        progress, _ = storage._metadata(directory / "progress.json")
        adapter, _ = storage._metadata(directory / "adapter_config.json")
    except (OSError, ValueError, RuntimeError) as exc:
        raise BackupError(f"Checkpoint is incomplete or corrupt: {exc}") from exc
    return {
        "source_format": "published_training_checkpoint",
        "completed_steps": progress.get("completed_steps"),
        "base_model": progress.get("base_model", adapter.get("base_model_name_or_path")),
        "base_revision": progress.get("base_revision", adapter.get("revision")),
        "purpose": progress.get("purpose"),
        "task_model_trained": progress.get("task_model_trained"),
        "checkpoint_manifest_sha256": storage._stream(directory / "checkpoint-manifest.json")[
            "sha256"
        ],
        "files": sorted(manifest["files"]),
    }


def _receipt_checkpoint(directory: Path) -> dict:
    """Preserve hash-bound adapter-only saves without claiming optimizer resumption."""
    try:
        receipt, _ = storage._metadata(directory / "training_receipt.json")
        adapter, _ = storage._metadata(directory / "adapter_config.json")
        weight = storage._stream(directory / "adapter_model.safetensors")
        step = receipt.get("step")
        losses = receipt.get("losses")
        if (
            receipt.get("actual_training") is not True
            or type(step) is not int
            or step < 1
            or not isinstance(losses, list)
            or len(losses) != step
            or any(type(value) not in {int, float} or not math.isfinite(value) for value in losses)
            or not isinstance(adapter.get("base_model_name_or_path"), str)
            or not adapter["base_model_name_or_path"]
            or not weight["size"]
            or receipt.get("adapter_sha256") != weight["sha256"]
            or any(
                not isinstance(receipt.get(key), str)
                or not re.fullmatch(r"[0-9a-f]{64}", receipt[key])
                for key in ("dataset_sha256", "recipe_sha256")
            )
            or not isinstance(receipt.get("base_revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", receipt["base_revision"])
        ):
            raise BackupError("Training receipt does not bind valid adapter progress and bytes")
        inventory = storage._inventory(directory)
    except (OSError, ValueError) as exc:
        raise BackupError(f"Checkpoint is incomplete or corrupt: {exc}") from exc
    names = sorted(row["path"] for row in inventory["files"])
    resume_files = [
        name
        for name in names
        if any(
            token in name.lower() for token in ("training-state", "optimizer", "scheduler", "rng")
        )
    ]
    return {
        "source_format": "adapter_training_receipt",
        "completed_steps": step,
        "base_model": adapter["base_model_name_or_path"],
        "base_revision": receipt["base_revision"],
        "dataset_sha256": receipt["dataset_sha256"],
        "recipe_sha256": receipt["recipe_sha256"],
        "adapter_sha256": weight["sha256"],
        "training_receipt_sha256": storage._stream(directory / "training_receipt.json")["sha256"],
        "resume_state_files": resume_files,
        "exact_resume_available": False,
        "resume_note": (
            "Saved state bytes were not loaded or resume-tested"
            if resume_files
            else "Optimizer/scheduler/RNG states were not saved; weights-only recovery"
        ),
        "files": names,
    }


def verify_checkpoint_backup(generation: str | Path) -> dict:
    """Verify the exact saved tree and its original checkpoint publication manifest."""
    generation = storage._path(generation)
    receipt = storage.verify(generation)
    checkpoint = _checkpoint(generation / "data")
    # Metadata and checkpoint validation must not observe different payloads.
    storage.verify(generation)
    return {**receipt, "checkpoint": checkpoint}


def backup_checkpoint(source: str | Path, backup_root: str | Path = DEFAULT_BACKUP_ROOT) -> dict:
    """Copy one quiescent published checkpoint into a new atomic generation.

    Call after publication or a complete hash-bound training_receipt save. A
    receipt-only adapter is preserved as weights-only recovery, never presented
    as an exact optimizer/RNG resume checkpoint.
    """
    source, backup_root = storage._path(source), storage._path(backup_root)
    storage._separate(source, backup_root)
    _checkpoint(source)
    manifest = storage._inventory(source)
    storage._validate_manifest(manifest)
    encoded = storage._encoded(manifest)
    if len(encoded) > storage.MAX_METADATA_BYTES:
        raise BackupError("Checkpoint manifest exceeds the metadata byte budget")
    backup_root.mkdir(parents=True, exist_ok=True)
    storage._no_links(backup_root)
    identity = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex
    staging = backup_root / (".checkpoint-staging-" + uuid4().hex)
    destination = backup_root / identity
    staging.mkdir()
    storage._copy(source, staging / "data", manifest)
    storage._matches(staging / "data", manifest)
    storage._matches(source, manifest)
    _checkpoint(staging / "data")
    receipt = {
        "schema_version": storage.VERSION,
        "generation": identity,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(manifest["files"]),
        "byte_count": sum(entry["size"] for entry in manifest["files"]),
    }
    storage._write(staging / "manifest.json", encoded)
    storage._write(staging / "receipt.json", storage._encoded(receipt))
    verify_checkpoint_backup(staging)
    storage._matches(source, manifest)
    storage._sync_payload(staging / "data", manifest)
    storage._sync_directory(staging)
    storage._publish(staging, destination)
    storage._sync_directory(backup_root)
    return verify_checkpoint_backup(destination)


def restore_checkpoint(generation: str | Path, destination: str | Path) -> dict:
    """Restore a verified checkpoint into a new absent directory, atomically."""
    generation, destination = storage._path(generation), storage._path(destination)
    storage._separate(generation, destination)
    if destination.exists():
        raise BackupError("Restore destination must not already exist")
    receipt = verify_checkpoint_backup(generation)
    manifest, _ = storage._metadata(generation / "manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage._no_links(destination.parent)
    staging = destination.parent / (".checkpoint-restore-" + uuid4().hex)
    storage._copy(generation / "data", staging, manifest)
    storage._matches(staging, manifest)
    checkpoint = _checkpoint(staging)
    verify_checkpoint_backup(generation)
    storage._sync_payload(staging, manifest)
    storage._publish(staging, destination)
    storage._sync_directory(destination.parent)
    return {
        "generation": receipt["generation"],
        "manifest_sha256": receipt["manifest_sha256"],
        "restored_to": str(destination),
        "checkpoint": checkpoint,
    }


def _watch_candidates(root: Path) -> list[Path]:
    """Inspect checkpoint folders only; never traverse training data or base weights."""
    candidates = []
    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        storage._no_links(run)
        for parent in (run, run / "published"):
            if not parent.is_dir():
                continue
            storage._no_links(parent)
            for path in sorted(parent.glob("checkpoint-*")):
                if not path.is_dir() or path.name.endswith(".partial"):
                    continue
                storage._no_links(path)
                if (path / "checkpoint-manifest.json").is_file() or (
                    (path / "training_receipt.json").is_file()
                    and (path / "adapter_model.safetensors").is_file()
                ):
                    candidates.append(path)
    return candidates


def _save_watch_state(path: Path, state: dict) -> None:
    """Only guardian control metadata is replaced; model snapshots remain append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    storage._no_links(path)
    encoded = storage._encoded(state)
    if len(encoded) > storage.MAX_METADATA_BYTES:
        raise BackupError("Guardian state exceeds the metadata byte budget")
    temporary = path.parent / (".checkpoint-guardian-" + uuid4().hex + ".partial")
    storage._write(temporary, encoded)
    storage._no_links(path)
    temporary.replace(path)
    storage._sync_directory(path.parent)


def _watch_stamp(root: Path) -> tuple:
    """Cheap change detection after validation; keep repeated scans off weight bytes."""
    rows = []

    def visit(path: Path) -> None:
        storage._no_links(path)
        if path.is_dir():
            rows.append((path.relative_to(root).as_posix(), *storage._stamp(path.lstat())))
            for child in sorted(path.iterdir()):
                visit(child)
        else:
            rows.append(
                (path.relative_to(root).as_posix(), *storage._stamp(storage._file_info(path)))
            )

    visit(root)
    return tuple(rows)


def _watch_event(state: dict, path: Path, kind: str, **details) -> None:
    event = {"event": kind, "at": datetime.now(UTC).isoformat(), **details}
    state["updated_at"] = event["at"]
    state["last_event"] = event
    _save_watch_state(path, state)
    print(json.dumps(event, sort_keys=True), flush=True)


def watch_checkpoints(
    runs_root: str | Path | None = None,
    *,
    state_path: str | Path,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
    max_seconds: int,
    until_utc: str | None = None,
    poll_seconds: int = 5,
) -> dict:
    """Guard the explicit local training root until a cooperative bounded deadline.

    One guardian owns its state path. State must be outside training and backup
    trees. Deadlines are checked between checkpoint copies, never by interrupting
    a copy halfway. Incomplete or changed saves are retried on the next scan.
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= 86400:
        raise BackupError("max_seconds must be an integer from 1 to 86400")
    if type(poll_seconds) is not int or not 5 <= poll_seconds <= 10:
        raise BackupError("poll_seconds must be an integer from 5 to 10")
    root = storage._path(runs_root if runs_root is not None else DEFAULT_RUNS_ROOT)
    if root != storage._path(DEFAULT_RUNS_ROOT) or not root.is_dir():
        raise BackupError("Guardian may watch only the existing output/visual-finetune/runs root")
    backup_root, state_path = storage._path(backup_root), storage._path(state_path)
    storage._separate(root, backup_root)
    storage._separate(root, state_path)
    storage._separate(backup_root, state_path)
    budget, stop_reason = float(max_seconds), "max_seconds"
    if until_utc is not None:
        deadline = datetime.fromisoformat(until_utc.replace("Z", "+00:00"))
        if deadline.utcoffset() is None:
            raise BackupError("until_utc must include an explicit UTC offset")
        remaining = max(0.0, (deadline.astimezone(UTC) - datetime.now(UTC)).total_seconds())
        if remaining <= budget:
            budget, stop_reason = remaining, "until_utc"
    previous = {}
    if state_path.exists():
        previous, _ = storage._metadata(state_path)
        if (
            previous.get("kind") != "checkpoint_guardian_state"
            or previous.get("schema_version") != 1
            or previous.get("runs_root") != str(root)
            or previous.get("backup_root") != str(backup_root)
        ):
            raise BackupError("Existing state is not this guardian's control metadata")
    state = {
        "schema_version": 1,
        "kind": "checkpoint_guardian_state",
        "pid": os.getpid(),
        "runs_root": str(root),
        "backup_root": str(backup_root),
        "started_at": datetime.now(UTC).isoformat(),
        "max_seconds": max_seconds,
        "until_utc": until_utc,
        "poll_seconds": poll_seconds,
        "status": "running",
        "snapshots": [],
        "errors": {},
    }
    started = monotonic()
    unchanged = {}
    known = set()
    entries = list(previous.get("snapshots", []))
    if backup_root.is_dir():
        for folder in sorted(backup_root.iterdir()):
            if folder.is_dir() and storage.GENERATION_PATTERN.fullmatch(folder.name):
                entries.append({"generation_dir": str(folder)})
    # Discover already protected sources, including emergency generic snapshots.
    for entry in entries:
        if monotonic() - started >= budget:
            break
        try:
            folder = storage._path(entry["generation_dir"])
            if folder.parent != backup_root:
                continue
            receipt, _ = storage._metadata(folder / "receipt.json")
            if not storage._path(receipt["source"]).is_relative_to(root):
                continue
            verified = verify_checkpoint_backup(folder)
            identity = (verified["source"], verified["manifest_sha256"])
            if identity not in known:
                state["snapshots"].append(verified)
                known.add(identity)
        except (OSError, ValueError, KeyError, TypeError):
            # A stale/corrupt older backup cannot suppress a fresh copy.
            continue
    _watch_event(state, state_path, "guardian_started", protected=len(known))
    try:
        while monotonic() - started < budget:
            try:
                candidates = _watch_candidates(root)
            except (OSError, ValueError) as exc:
                candidates = []
                error = str(exc)[:600]
                if state["errors"].get("scan") != error:
                    state["errors"]["scan"] = error
                    _watch_event(state, state_path, "scan_error", error=error)
            else:
                if state["errors"].pop("scan", None) is not None:
                    _watch_event(state, state_path, "scan_recovered")
            for candidate in candidates:
                if monotonic() - started >= budget:
                    break
                name = str(candidate)
                try:
                    stamp = _watch_stamp(candidate)
                    if unchanged.get(name) == stamp:
                        continue
                    _checkpoint(candidate)
                    identity = hashlib.sha256(
                        storage._encoded(storage._inventory(candidate))
                    ).hexdigest()
                    if (name, identity) in known:
                        if _watch_stamp(candidate) == stamp:
                            unchanged[name] = stamp
                        if state["errors"].pop(name, None) is not None:
                            _watch_event(state, state_path, "checkpoint_recovered", source=name)
                        continue
                    receipt = backup_checkpoint(candidate, backup_root)
                    if _watch_stamp(candidate) == stamp:
                        unchanged[name] = stamp
                    known.add((name, receipt["manifest_sha256"]))
                    state["snapshots"].append(receipt)
                    state["errors"].pop(name, None)
                    _watch_event(
                        state,
                        state_path,
                        "checkpoint_backed_up",
                        source=name,
                        generation_dir=receipt["generation_dir"],
                        completed_steps=receipt["checkpoint"]["completed_steps"],
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    error = str(exc)[:600]
                    if state["errors"].get(name) != error:
                        state["errors"][name] = error
                        _watch_event(
                            state, state_path, "checkpoint_waiting", source=name, error=error
                        )
            remaining = budget - (monotonic() - started)
            if remaining > 0:
                sleep(min(float(poll_seconds), remaining))
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    state.update(
        status="stopped", finished_at=datetime.now(UTC).isoformat(), stop_reason=stop_reason
    )
    _watch_event(state, state_path, "guardian_stopped", reason=stop_reason, protected=len(known))
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--source", type=Path, required=True)
    backup.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    verify = commands.add_parser("verify")
    verify.add_argument("--generation", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--generation", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    watch = commands.add_parser("watch")
    watch.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    watch.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    watch.add_argument("--state", type=Path, required=True)
    watch.add_argument("--max-seconds", type=int, required=True)
    watch.add_argument("--until-utc")
    watch.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == "backup":
            result = backup_checkpoint(args.source, args.backup_root)
        elif args.command == "verify":
            result = verify_checkpoint_backup(args.generation)
        elif args.command == "restore":
            result = restore_checkpoint(args.generation, args.destination)
        else:
            watch_checkpoints(
                args.runs_root,
                state_path=args.state,
                backup_root=args.backup_root,
                max_seconds=args.max_seconds,
                until_utc=args.until_utc,
                poll_seconds=args.poll_seconds,
            )
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
