"""Supervise a single read-only cloud adapter reload diagnostic."""

import argparse
import os
import shutil
import sys
from pathlib import Path

from experiments.visual_finetune.data import read_json, sha, write_json
from experiments.visual_finetune.supervise import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="checkpoint-32")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    old, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.exists():
        raise ValueError("Diagnostic output must be a new directory")
    out.mkdir(parents=True)
    package = out / "code/experiments/visual_finetune"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_bytes(b"")
    hashes = {}
    for name in ("__init__.py", "data.py", "diagnose_reload.py"):
        source = Path(__file__).resolve().parents[1] / "experiments/visual_finetune" / name
        shutil.copyfile(source, package / name)
        hashes[name] = sha(source)
    source_files = [
        "recipe.json",
        "adapted_validation.json",
        "reload_probe.json",
        args.checkpoint + "/adapter_config.json",
        args.checkpoint + "/adapter_model.safetensors",
    ]
    recipe = read_json(old / "recipe.json")
    source_files.append(recipe["dataset"])
    source_files.append(recipe["base_manifest"])
    config = {
        "output": str(out),
        "source_run": str(old),
        "checkpoint": str(old / args.checkpoint),
        "source_sha256": {name: sha(old / name) for name in source_files},
        "code_sha256": hashes,
    }
    write_json(out / "diagnostic-config.json", config)
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(out / "code"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    receipt = run(
        [
            args.python,
            "-m",
            "experiments.visual_finetune.diagnose_reload",
            str(out / "diagnostic-config.json"),
        ],
        out,
        cwd=str(out / "code"),
        environment=environment,
        max_seconds=180,
        max_rss_mb=16384,
        minimum_available_mb=256,
        deadline_utc="2026-10-31T05:01:00+00:00",
    )
    print(receipt)
    raise SystemExit(0 if receipt["exit_code"] == 0 else 1)


if __name__ == "__main__":
    main()
