"""Build a project-local verified base overlay from cached weights; no network access."""

import argparse
import os
import shutil
from pathlib import Path

from experiments.visual_finetune.data import read_json, sha, write_json


def prepare(weights, metadata, manifest_path, out):
    weights, metadata, out = Path(weights).resolve(), Path(metadata), Path(out)
    manifest = read_json(manifest_path)
    if out.exists():
        raise ValueError("Verified base overlay must use a new directory")
    if sha(weights) != manifest["files"]["model.safetensors"]["sha256"]:
        raise ValueError("Cached model weights differ from the pinned public base")
    out.mkdir(parents=True)
    for name, expected in manifest["files"].items():
        if Path(name).name != name or name.endswith(".py"):
            raise ValueError("Base metadata whitelist cannot contain paths or remote code")
        target = out / name
        if name == "model.safetensors":
            if os.name == "posix":
                target.symlink_to(weights)
            else:
                shutil.copyfile(weights, target)
        else:
            source = metadata / name
            if sha(source) != expected["sha256"]:
                raise ValueError("Packaged base metadata differs from its pinned identity")
            shutil.copyfile(source, target)
    shutil.copyfile(manifest_path, out / "verified-base.json")
    receipt = {
        "base": str(out.resolve()),
        "weights_source": str(weights),
        "model_sha256": sha(weights),
        "network_used": False,
        "cache_modified": False,
    }
    write_json(out / "overlay-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(prepare(args.weights, args.metadata, args.manifest, args.out), indent=2))


if __name__ == "__main__":
    main()
