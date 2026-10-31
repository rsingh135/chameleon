"""Publish and restore verified local snapshots after an off-VM checkpoint pull.

Use quiescent, trusted directories. Observed links/reparse points and ordinary
concurrent mutations are rejected; portable pathname checks are not a sandbox
against an adversary swapping filesystem entries. Generations are append-only:
each receipt and payload become visible together through one directory rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

VERSION = 1
UTC = timezone.utc  # noqa: UP017 - the local WSL CLI also runs on Python 3.10
CHUNK_SIZE = 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
GENERATION_PATTERN = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}\Z")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class BackupError(ValueError):
    """A snapshot could not be safely verified or published."""


def _path(value: str | Path) -> Path:
    result = Path(os.path.abspath(value))
    _no_links(result)
    return result


def _no_links(path: Path) -> None:
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            raise BackupError(f"Links and reparse points are not allowed: {part}")


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise BackupError(f"Unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise BackupError(f"Unsafe relative path: {value!r}")
    for part in value.split("/"):
        if (
            part in {"", ".", ".."}
            or part.endswith((".", " "))
            or any(ord(character) < 32 for character in part)
            or any(character in '<>"|?*' for character in part)
            or PureWindowsPath(part).is_reserved()
        ):
            raise BackupError(f"Unsafe or nonportable relative path: {value!r}")
    return value


def _separate(first: Path, second: Path) -> None:
    if first.is_relative_to(second) or second.is_relative_to(first):
        raise BackupError("Source, backup and restore paths must not overlap")


def _file_info(path: Path):
    _no_links(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise BackupError(f"Only regular, non-hardlinked files are allowed: {path}")
    return info


def _stamp(info) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _stream(path: Path, destination: Path | None = None) -> dict:
    before = _file_info(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(os.open(path, flags), "rb") as source:
        opened = os.fstat(source.fileno())
        # Windows path-stat and handle-stat can differ in timestamp precision.
        # Compare identity across APIs, and mutation stamps within each API.
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise BackupError(f"File changed while opening: {path}")
        output = destination.open("xb") if destination is not None else None
        try:
            while block := source.read(CHUNK_SIZE):
                digest.update(block)
                size += len(block)
                if output is not None:
                    output.write(block)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
            if _stamp(os.fstat(source.fileno())) != _stamp(opened):
                raise BackupError(f"File changed while reading: {path}")
        finally:
            if output is not None:
                output.close()
    if _stamp(_file_info(path)) != _stamp(before):
        raise BackupError(f"File changed while reading: {path}")
    return {"size": size, "sha256": digest.hexdigest()}


def _inventory(root: Path) -> dict:
    _no_links(root)
    if not root.is_dir():
        raise BackupError(f"A directory is required: {root}")
    files, directories, names = [], [], set()

    def visit(folder: Path) -> None:
        _no_links(folder)
        with os.scandir(folder) as entries:
            children = sorted(entries, key=lambda item: item.name)
        for entry in children:
            path = Path(entry.path)
            relative = _relative(path.relative_to(root).as_posix())
            if relative.casefold() in names:
                raise BackupError(f"Case-colliding paths are not portable: {relative}")
            names.add(relative.casefold())
            _no_links(path)
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
                visit(path)
            else:
                files.append({"path": relative, **_stream(path)})

    visit(root)
    return {"schema_version": VERSION, "directories": sorted(directories), "files": files}


def _encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _metadata(path: Path) -> tuple[dict, bytes]:
    if _file_info(path).st_size > MAX_METADATA_BYTES:
        raise BackupError("Metadata exceeds the 8 MiB limit")
    data = path.read_bytes()
    if len(data) > MAX_METADATA_BYTES:
        raise BackupError("Metadata exceeds the 8 MiB limit")
    value = json.loads(data, object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise BackupError("Metadata must be a JSON object")
    return value, data


def _validate_manifest(manifest: dict) -> None:
    if (
        set(manifest) != {"schema_version", "directories", "files"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != VERSION
    ):
        raise BackupError("Unsupported manifest schema")
    if not isinstance(manifest["directories"], list) or not isinstance(manifest["files"], list):
        raise BackupError("Manifest file and directory lists are required")
    names = {}
    for value in manifest["directories"]:
        relative = _relative(value)
        if relative.casefold() in names:
            raise BackupError("Duplicate/case-colliding manifest paths")
        names[relative.casefold()] = "directory"
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise BackupError("Invalid manifest file entry")
        relative = _relative(entry["path"])
        if relative.casefold() in names:
            raise BackupError("Duplicate/case-colliding manifest paths")
        names[relative.casefold()] = "file"
        if type(entry["size"]) is not int or entry["size"] < 0:
            raise BackupError("File sizes must be nonnegative integers")
        if not isinstance(entry["sha256"], str) or not HASH_PATTERN.fullmatch(entry["sha256"]):
            raise BackupError("Invalid SHA-256 value")
    for relative in names:
        for parent in PurePosixPath(relative).parents:
            if str(parent) != "." and names.get(str(parent)) != "directory":
                raise BackupError("Manifest must explicitly include every parent directory")


def _matches(root: Path, expected: dict) -> None:
    if _inventory(root) != expected:
        raise BackupError("File set, sizes or SHA-256 values do not match the manifest")


def _copy(source: Path, destination: Path, manifest: dict) -> None:
    destination.mkdir()
    for relative in sorted(manifest["directories"], key=lambda item: (item.count("/"), item)):
        (destination / relative).mkdir()
    for entry in manifest["files"]:
        relative = entry["path"]
        observed = _stream(source / relative, destination / relative)
        if observed != {"size": entry["size"], "sha256": entry["sha256"]}:
            raise BackupError(f"Source changed during copying: {relative}")


def _publish(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise BackupError("Publication destination already exists; nothing was overwritten")
    staging.rename(destination)


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _sync_payload(root: Path, manifest: dict) -> None:
    for relative in reversed(manifest["directories"]):
        _sync_directory(root / relative)
    _sync_directory(root)


def _cleanup(staging: Path, parent: Path) -> None:
    # Only remove the unique staging directory created by this invocation.
    if staging.parent != parent or not staging.name.startswith(".brev-staging-"):
        raise BackupError("Refusing to remove an unexpected staging path")
    _no_links(parent)
    if staging.exists():
        _no_links(staging)
        shutil.rmtree(staging)


def snapshot(source: str | Path, backup_root: str | Path) -> dict:
    """Byte-copy a quiescent source and publish an independently verified generation."""
    source, backup_root = _path(source), _path(backup_root)
    _separate(source, backup_root)
    manifest = _inventory(source)
    _validate_manifest(manifest)
    manifest_bytes = _encoded(manifest)
    if len(manifest_bytes) > MAX_METADATA_BYTES:
        raise BackupError("Manifest exceeds the 8 MiB limit")
    backup_root.mkdir(parents=True, exist_ok=True)
    _no_links(backup_root)
    identity = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    staging = backup_root / (".brev-staging-" + uuid.uuid4().hex)
    final = backup_root / identity
    staging.mkdir()
    try:
        _copy(source, staging / "data", manifest)
        _matches(staging / "data", manifest)
        _matches(source, manifest)
        receipt = {
            "schema_version": VERSION,
            "generation": identity,
            "created_at": datetime.now(UTC).isoformat(),
            "source": str(source),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "file_count": len(manifest["files"]),
            "byte_count": sum(entry["size"] for entry in manifest["files"]),
        }
        _write(staging / "manifest.json", manifest_bytes)
        _write(staging / "receipt.json", _encoded(receipt))
        verify(staging)
        _sync_payload(staging / "data", manifest)
        _sync_directory(staging)
        _publish(staging, final)
        _sync_directory(backup_root)
        return {**receipt, "generation_dir": str(final)}
    finally:
        _cleanup(staging, backup_root)


def verify(generation: str | Path) -> dict:
    """Check exact contents and hashes; this is integrity checking, not authentication."""
    generation = _path(generation)
    if not generation.is_dir() or set(item.name for item in generation.iterdir()) != {
        "data",
        "manifest.json",
        "receipt.json",
    }:
        raise BackupError("A generation requires only data/, manifest.json and receipt.json")
    manifest, manifest_bytes = _metadata(generation / "manifest.json")
    receipt, _ = _metadata(generation / "receipt.json")
    _validate_manifest(manifest)
    expected_keys = {
        "schema_version",
        "generation",
        "created_at",
        "source",
        "manifest_sha256",
        "file_count",
        "byte_count",
    }
    if (
        set(receipt) != expected_keys
        or type(receipt["schema_version"]) is not int
        or (receipt["schema_version"] != VERSION)
    ):
        raise BackupError("Unsupported receipt schema")
    if not isinstance(receipt["generation"], str) or not GENERATION_PATTERN.fullmatch(
        receipt["generation"]
    ):
        raise BackupError("Invalid generation identity")
    if not isinstance(receipt["created_at"], str) or not isinstance(receipt["source"], str):
        raise BackupError("Receipt provenance is required")
    if receipt["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise BackupError("Manifest checksum does not match the receipt")
    if (
        type(receipt["file_count"]) is not int
        or receipt["file_count"] != len(manifest["files"])
        or type(receipt["byte_count"]) is not int
        or receipt["byte_count"] != sum(entry["size"] for entry in manifest["files"])
    ):
        raise BackupError("Receipt totals do not match the manifest")
    _matches(generation / "data", manifest)
    return {**receipt, "generation_dir": str(generation)}


def restore(generation: str | Path, destination: str | Path) -> dict:
    """Restore through sibling staging into an absent destination; never overwrite."""
    generation, destination = _path(generation), _path(destination)
    _separate(generation, destination)
    if destination.exists():
        raise BackupError("Restore destination must not already exist")
    receipt = verify(generation)
    manifest, _ = _metadata(generation / "manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _no_links(destination.parent)
    staging = destination.parent / (".brev-staging-" + uuid.uuid4().hex)
    try:
        _copy(generation / "data", staging, manifest)
        _matches(staging, manifest)
        verify(generation)
        _sync_payload(staging, manifest)
        _publish(staging, destination)
        _sync_directory(destination.parent)
    finally:
        _cleanup(staging, destination.parent)
    return {"generation": receipt["generation"], "restored_to": str(destination)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("snapshot", help="Publish a new immutable backup generation")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--backup-root", type=Path, required=True)
    check = commands.add_parser("verify", help="Verify one published generation")
    check.add_argument("--generation", type=Path, required=True)
    recover = commands.add_parser("restore", help="Restore into a new directory")
    recover.add_argument("--generation", type=Path, required=True)
    recover.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            result = snapshot(args.source, args.backup_root)
        elif args.command == "verify":
            result = verify(args.generation)
        else:
            result = restore(args.generation, args.destination)
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
