"""Publish complete model/state generations atomically, preserving earlier saves."""

import os
from pathlib import Path
from uuid import uuid4

from .data import read_json, sha, write_json

REQUIRED = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "training-state.pt",
    "progress.json",
}


def publish(stage, destination):
    stage, destination = Path(stage), Path(destination)
    if destination.exists() or stage.parent != destination.parent:
        raise ValueError("Checkpoint publication requires a new sibling destination")
    files = {
        path.relative_to(stage).as_posix(): {"size": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    if not REQUIRED.issubset(files) or any(files[key]["size"] == 0 for key in REQUIRED):
        raise ValueError("Atomic checkpoint is incomplete")
    write_json(stage / "checkpoint-manifest.json", {"format_version": 1, "files": files})
    for relative, expected in files.items():
        if sha(stage / relative) != expected["sha256"]:
            raise ValueError("Checkpoint changed during publication")
    for path in stage.rglob("*"):
        if path.is_file():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
    stage.rename(destination)
    return read_json(destination / "checkpoint-manifest.json")


def staging_directory(destination):
    destination = Path(destination)
    stage = destination.parent / ("." + destination.name + "-" + uuid4().hex + ".partial")
    stage.mkdir()
    return stage
