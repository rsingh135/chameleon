"""Execute one already frozen inference/training recipe without preparing it again."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from experiments.visual_finetune.data import read_json, sha
from experiments.visual_finetune.supervise import run


def execute(recipe_path, *, python=None):
    recipe_path = Path(recipe_path).resolve()
    recipe = read_json(recipe_path)
    out = Path(recipe["output"])
    if (out / "supervision.json").exists() or (out / "result.json").exists():
        raise ValueError("A completed/failed frozen recipe must not be executed again in place")
    with (out / "execution-started.sha256").open("x", encoding="utf-8") as marker:
        marker.write(sha(recipe_path) + "\n")
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
    guard = recipe["supervision"]
    measured = "C:/Users/edmun/anaconda3/envs/acestep/python.exe"
    executable = python or (
        measured if os.name == "nt" and Path(measured).is_file() else sys.executable
    )
    return run(
        [executable, "-m", "experiments.visual_finetune.worker", str(recipe_path)],
        out,
        cwd=str(out / "code"),
        environment=environment,
        max_seconds=guard["max_seconds"],
        max_rss_mb=guard.get("max_rss_mb", 4096),
        minimum_available_mb=guard["minimum_available_mb"],
        deadline_utc=guard["absolute_deadline_utc"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--python")
    args = parser.parse_args()
    import json

    receipt = execute(args.recipe, python=args.python)
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if receipt["exit_code"] == 0 else 1)


if __name__ == "__main__":
    main()
