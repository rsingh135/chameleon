"""Explicitly fetch a pinned public local runtime and VLM; inference never fetches."""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path

from .native import (
    CUDA_ARCHIVES,
    MODEL,
    PINNED,
    PROFILE_IDS,
    PROJECTOR,
    QWEN_MODEL,
    QWEN_PINNED,
    QWEN_PROJECTOR,
    QWEN_REVISION,
    REVISION,
    RUNTIME_SHA,
    RUNTIME_TAG,
    profile,
    runtime_profile,
)
from .provider import ROOT, sha256

ARCHIVE = f"llama-{RUNTIME_TAG}-bin-win-cpu-x64.zip"
SOURCES = {
    ARCHIVE: {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_TAG}/{ARCHIVE}",
        "sha256": RUNTIME_SHA,
        "bytes": 24_574_610,
        "license": "MIT",
    },
    MODEL: {
        "url": f"https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/{REVISION}/{MODEL}",
        "sha256": PINNED[MODEL],
        "bytes": 436_806_912,
        "license": "Apache-2.0",
    },
    PROJECTOR: {
        "url": f"https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/{REVISION}/{PROJECTOR}",
        "sha256": PINNED[PROJECTOR],
        "bytes": 199_468_800,
        "license": "Apache-2.0",
    },
}
QWEN_SOURCES = {
    ARCHIVE: SOURCES[ARCHIVE],
    QWEN_MODEL: {
        "url": (
            "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/"
            f"{QWEN_REVISION}/{QWEN_MODEL}"
        ),
        "sha256": QWEN_PINNED[QWEN_MODEL],
        "bytes": 579_615_840,
        "license": "Apache-2.0",
    },
    QWEN_PROJECTOR: {
        "url": (
            "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/"
            f"{QWEN_REVISION}/{QWEN_PROJECTOR}"
        ),
        "sha256": QWEN_PINNED[QWEN_PROJECTOR],
        "bytes": 204_987_104,
        "license": "Apache-2.0",
    },
}


def fetch(path, specification):
    """No HF client, tokens, authenticated headers, package install, or auto code."""
    if (
        path.is_file()
        and path.stat().st_size == specification["bytes"]
        and sha256(path) == specification["sha256"]
    ):
        return
    partial = path.with_name(path.name + ".part")
    try:
        request = urllib.request.Request(
            specification["url"], headers={"User-Agent": "Chameleon/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > specification["bytes"]:
                    raise ValueError("Public download exceeds its pinned byte size")
                out.write(chunk)
        if partial.stat().st_size != specification["bytes"]:
            raise ValueError("Public download does not match its pinned byte size")
        if sha256(partial) != specification["sha256"]:
            raise ValueError("Public download does not match its pinned SHA-256")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def extract_runtime(archive, destination, expected_sha=None, maximum_bytes=256 * 1024 * 1024):
    if sha256(archive) != (RUNTIME_SHA if expected_sha is None else expected_sha):
        raise ValueError("Runtime ZIP differs from its pinned release")
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        if len(entries) > 1024 or sum(p.file_size for p in entries) > maximum_bytes:
            raise ValueError("Pinned runtime exceeds its extraction budget")
        # Validate the complete archive before writing any member.
        for member in entries:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Runtime ZIP contains an escaping path or symbolic link")
        destination.mkdir(parents=True, exist_ok=True)
        for member in entries:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)


def download(out, profile_id="smolvlm_500m", device="cpu"):
    specification = profile({"native_profile": profile_id})
    selected_runtime = runtime_profile({"native_device": device})
    sources = dict(SOURCES if profile_id == "smolvlm_500m" else QWEN_SOURCES)
    if device == "cuda":
        sources.pop(ARCHIVE)
        sources.update(
            {
                name: {
                    **value,
                    "url": (
                        f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_TAG}/{name}"
                    ),
                }
                for name, value in CUDA_ARCHIVES.items()
            }
        )
    out = Path(out).resolve()
    if not out.is_relative_to((ROOT / "output/visual-planner").resolve()):
        raise ValueError("Native setup belongs under this project's output/visual-planner")
    out.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        fetch(out / name, source)
    runtime = out / selected_runtime["directory"]
    for archive, expected in selected_runtime["archives"].items():
        extract_runtime(
            out / archive,
            runtime,
            expected_sha=expected,
            maximum_bytes=(2048 if device == "cuda" else 256) * 1024 * 1024,
        )
    manifest = {
        "schema_version": 1,
        "profile": profile_id,
        "model_revision": specification["revision"],
        "base_model_id": specification["base_model_id"],
        "quantizer": specification["quantizer"],
        "runtime_tag": RUNTIME_TAG,
        "native_device": device,
        "sources": sources,
        "files": {
            str(p.relative_to(runtime)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(runtime.rglob("*"))
            if p.is_file()
        },
    }
    manifest_name = (
        "verified-native.json" if profile_id == "smolvlm_500m" else "verified-native-qwen.json"
    )
    if device == "cuda":
        manifest_name = manifest_name.replace(".json", "-cuda.json")
    (out / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "model_dir": str(out),
        "native_profile": profile_id,
        "native_device": device,
        "runtime_tag": RUNTIME_TAG,
        "model_revision": specification["revision"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/visual-planner/native")
    parser.add_argument("--profile", choices=PROFILE_IDS, default="smolvlm_500m")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(download(args.out, args.profile, args.device)), flush=True)


if __name__ == "__main__":
    main()
