"""Fetch an explicitly pinned public model once; inference remains offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MODELS = {
    "500m": {
        "repo_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "revision": "a7da5b986cb59b408707209984f360a5f4ad7e47",
        "weight_sha256": "d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2",
        "weight_bytes": 1015025832,
    },
    "256m": {
        "repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "revision": "7e3e67edbbed1bf9888184d9df282b700a323964",
        "weight_sha256": "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
        "weight_bytes": 513028808,
    },
}
FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(size, out):
    spec = MODELS[size]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    from huggingface_hub import hf_hub_download

    for filename in FILES:
        hf_hub_download(
            spec["repo_id"],
            filename,
            revision=spec["revision"],
            local_dir=out,
            token=False,
            local_files_only=False,
        )
    weight = out / "model.safetensors"
    if weight.stat().st_size != spec["weight_bytes"] or sha256(weight) != spec["weight_sha256"]:
        raise ValueError("Downloaded model differs from its pinned public weight identity")
    manifest = {
        "schema_version": 1,
        **spec,
        "license": "Apache-2.0",
        "source": f"https://huggingface.co/{spec['repo_id']}/tree/{spec['revision']}",
        "files": {
            name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
            for name in FILES
        },
    }
    (out / "verified-model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "model_dir": str(out),
        "revision": spec["revision"],
        "weight_sha256": spec["weight_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=MODELS, default="500m")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download(args.size, args.out)), flush=True)


if __name__ == "__main__":
    main()
