"""Stdlib-only checkpoint byte integrity and atomic publication helpers.

The manifest describes opaque files; no model or optimizer state is deserialized.
These functions retain the checkpoint format used by the training smoke tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

CHECKPOINT_MANIFEST = "checkpoint-manifest.json"
REQUIRED_CHECKPOINT_FILES = frozenset(
    {"adapter_config.json", "adapter_model.safetensors", "training-state.pt", "progress.json"}
)


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint_files(directory: Path) -> dict[str, dict]:
    """Hash a quiescent checkpoint tree; do not follow links or special files."""
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or getattr(directory.lstat(), "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise RuntimeError("Checkpoint must be a real directory")
    files = {}
    for path in sorted(directory.rglob("*")):
        info = path.lstat()
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        ):
            raise RuntimeError("Checkpoint symlinks and reparse points are not permitted")
        if path.is_dir():
            continue
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("Checkpoint special files are not permitted")
        relative = path.relative_to(directory).as_posix()
        if relative == CHECKPOINT_MANIFEST:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[relative] = {"size": info.st_size, "sha256": digest.hexdigest()}
    if not REQUIRED_CHECKPOINT_FILES.issubset(files) or any(
        files[name]["size"] == 0 for name in REQUIRED_CHECKPOINT_FILES if name in files
    ):
        raise RuntimeError(
            "Incomplete checkpoint: adapter, optimizer state, and progress are required"
        )
    return files


def verify_checkpoint(directory: Path) -> dict:
    """Detect missing, extra, or damaged files against the published manifest.

    This detects accidental corruption. An attacker able to replace both files
    and manifest is outside this check's scope; off-VM receipts provide a
    separate trusted copy. It does not deserialize or validate tensor contents.
    """
    actual_files = checkpoint_files(directory)
    manifest_path = directory / CHECKPOINT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != 1
        or not isinstance(manifest.get("files"), dict)
    ):
        raise RuntimeError("Unsupported checkpoint manifest")
    if actual_files != manifest["files"]:
        raise RuntimeError("Checkpoint file set or SHA256 differs from its manifest")
    return manifest


def publish_checkpoint(stage: Path, destination: Path) -> None:
    """Verify, flush, and atomically publish a complete immutable checkpoint.

    Stage and destination must be on the same filesystem. Failed publication
    preserves every previous generation; the partial stage is never a success.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Checkpoint generation already exists: {destination}")
    files = checkpoint_files(stage)
    write_json(stage / CHECKPOINT_MANIFEST, {"format_version": 1, "files": files})
    verify_checkpoint(stage)
    for path in stage.rglob("*"):
        if path.is_file():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
    if os.name == "posix":
        directory = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    stage.rename(destination)
    if os.name == "posix":
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
