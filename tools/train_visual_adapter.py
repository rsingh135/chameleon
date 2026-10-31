"""Freeze and supervise a local short-answer visual LoRA run; never enable it by default."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from experiments.visual_finetune.data import freeze_dataset, sha, write_json
from experiments.visual_finetune.supervise import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", choices=("purpose", "critic"), required=True)
    parser.add_argument("--out", required=True)
    measured_python = "C:/Users/edmun/anaconda3/envs/acestep/python.exe"
    parser.add_argument(
        "--python",
        default=(
            measured_python
            if os.name == "nt" and Path(measured_python).is_file()
            else sys.executable
        ),
    )
    parser.add_argument(
        "--base", default="output/visual-finetune/base/89644892e4d85e24eaac8bacfd4f463576704203"
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--base-manifest")
    parser.add_argument("--allow-train-only", action="store_true")
    parser.add_argument("--validation-request")
    parser.add_argument("--evaluation-limit", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument(
        "--sampling",
        choices=("balanced_class_cycle", "sqrt_class_frequency"),
        default="balanced_class_cycle",
    )
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--max-rss-mb", type=int, default=4096)
    parser.add_argument("--deadline-utc", default="2026-10-31T05:01:00+00:00")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.max_rss_mb <= 0:
        raise ValueError("The owned-worker RSS cap must be positive")
    root, out = Path(__file__).resolve().parents[1], Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("A run output must be empty; previous failures are retained")
    out.mkdir(parents=True, exist_ok=True)
    base_manifest = Path(args.base_manifest or Path(args.base) / "verified-base.json").resolve()
    shutil.copyfile(base_manifest, out / "base_identity.json")
    dataset = freeze_dataset(
        args.manifest,
        out / "data",
        task=args.task,
        allow_train_only=args.allow_train_only,
        evaluation_request=args.validation_request,
    )
    validation = dataset["evaluation_records"] or [
        row for row in dataset["records"] if row["split"] == "validation"
    ]
    # Deterministic round-robin class/control coverage, independent of model scores.
    groups = {}
    for row in sorted(validation, key=lambda r: r["record_id"]):
        groups.setdefault((row["control"], row.get("label", "unlabeled")), []).append(row)
    chosen = []
    while len(chosen) < min(args.evaluation_limit, len(validation)):
        for key in sorted(groups):
            if groups[key] and len(chosen) < args.evaluation_limit:
                chosen.append(groups[key].pop(0)["record_id"])
    code = out / "code"
    package = code / "experiments" / "visual_finetune"
    package.mkdir(parents=True)
    (code / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    code_hashes = {}
    for source in (root / "experiments" / "visual_finetune").glob("*.py"):
        shutil.copyfile(source, package / source.name)
        code_hashes[source.name] = sha(source)
    recipe = {
        "schema_version": 1,
        "output": str(out),
        "base": str(Path(args.base).resolve()),
        "dataset": "data/dataset.json",
        "base_manifest": "base_identity.json",
        "base_manifest_sha256": sha(out / "base_identity.json"),
        "dataset_sha256": sha(out / "data" / "dataset.json"),
        "code_sha256": code_hashes,
        "seed": 1729,
        "steps": args.steps,
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "sampling": "balanced_A_B_N_cycle" if args.task == "critic" else args.sampling,
        "learning_rate": args.learning_rate,
        "accumulation": 6 if args.task == "critic" else 4,
        "checkpoint_every": 4,
        "min_pixels": 128 * 128,
        "max_pixels": args.max_pixels or (512 * 512 if args.task == "critic" else 256 * 256),
        "max_input_tokens": 2048,
        "evaluation_record_ids": chosen,
        "label_free_evaluation": bool(dataset["evaluation_records"]),
        "quality_approved": False,
        "supervision": {
            "minimum_available_mb": 256,
            "owned_working_set_trim": True,
            "max_seconds": args.max_seconds,
            "max_rss_mb": args.max_rss_mb,
            "absolute_deadline_utc": args.deadline_utc,
        },
    }
    write_json(out / "recipe.json", recipe)
    if args.prepare_only:
        print(str(out / "recipe.json"))
        return
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
            "PYTHONPATH": str(code),
            "PYTHONUNBUFFERED": "1",
        }
    )
    receipt = run(
        [args.python, "-m", "experiments.visual_finetune.worker", str(out / "recipe.json")],
        out,
        cwd=str(code),
        environment=environment,
        max_seconds=args.max_seconds,
        max_rss_mb=args.max_rss_mb,
        deadline_utc=args.deadline_utc,
    )
    print(receipt)
    if receipt["exit_code"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
