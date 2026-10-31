"""Score label-free independent visual probes without reading held-out answers."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from experiments.visual_finetune.data import _copy_image, read_json, sha, write_json
from experiments.visual_finetune.snapshot import FROZEN_TOOL_KEYS, freeze_coverage_helper
from experiments.visual_finetune.supervise import run


def prepare(
    request_path,
    training_run,
    adapter_path,
    out,
    *,
    base=None,
    base_manifest=None,
    inference_only=False,
    max_seconds=900,
    max_rss_mb=4096,
    deadline_utc="2026-10-31T05:01:00+00:00",
    image_root=None,
):
    request_path, training_run, out = map(Path, (request_path, training_run, out))
    request = read_json(request_path)
    if max_seconds <= 0 or max_rss_mb <= 0:
        raise ValueError("Owned-worker resource limits must remain positive")
    if request.get("protocol") != "visual-adapter-label-free-evaluation-v1":
        raise ValueError("Unexpected independent evaluation protocol")
    training_recipe = read_json(training_run / "recipe.json")
    training_dataset = Path(training_recipe["dataset"])
    if not training_dataset.is_absolute():
        training_dataset = training_run / training_dataset
    if sha(training_dataset) != training_recipe["dataset_sha256"]:
        raise ValueError("Original training dataset differs from its frozen receipt")
    training_data = read_json(training_dataset)
    if request["task"] != training_data["task"]:
        raise ValueError("Probe task and adapter task differ")
    definition_sha = training_data.get("provenance", {}).get("task_definition_sha256")
    if request.get("task_definition_sha256") != definition_sha:
        raise ValueError("Purpose request must pin the exact trained task definition")
    if request.get("context_version", "v2") != training_recipe.get("context_version", "v2"):
        raise ValueError("Request context version differs from the trained observation contract")
    purpose_views = training_data.get("purpose_views", 2)
    if request["task"] == "purpose" and request.get("purpose_views", 2) != purpose_views:
        raise ValueError("Purpose request view count differs from the trained input contract")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be empty")
    image_dir = out / "data" / "images"
    image_dir.mkdir(parents=True)
    records = []
    seen = set()
    image_index = {}
    if image_root is not None:
        for path in Path(image_root).iterdir():
            if path.is_file():
                image_index.setdefault(path.stem, []).append(path)
    for original in request["records"]:
        if any(key in original for key in ("label", "target", "expected", "answer")):
            raise ValueError("Independent scoring must not receive sealed answers")
        if original["record_id"] in seen:
            raise ValueError("Duplicate independent probe ID")
        seen.add(original["record_id"])
        if len(original["images"]) != (purpose_views if request["task"] == "purpose" else 3):
            raise ValueError("Probe image order/count differs from the trained task")
        refs = []
        for ref in original["images"]:
            if image_root is not None:
                matches = image_index.get(ref["sha256"], [])
                if len(matches) != 1:
                    raise ValueError("Cross-host image root needs one exact SHA-named input")
                path = matches[0]
            else:
                path = Path(ref["path"])
                if not path.is_absolute():
                    path = request_path.parent / path
            refs.append(_copy_image(path, ref["sha256"], image_dir))
        records.append(
            {
                "record_id": original["record_id"],
                "family_id": original.get("family_id", "independent"),
                "split": "external_evaluation",
                "images": refs,
                "prompt": original["prompt"],
                "control": original.get("control", "independent"),
            }
        )
    if not records:
        raise ValueError("Independent probes must be nonempty")
    dataset = {
        "schema_version": 1,
        "task": request["task"],
        "classes": training_data["classes"],
        "records": records,
        "provenance": {"request_sha256": sha(request_path), "label_free": True},
        "purpose_views": purpose_views if request["task"] == "purpose" else None,
    }
    write_json(out / "data" / "dataset.json", dataset)
    package = out / "code" / "experiments" / "visual_finetune"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    hashes = {}
    for source in (Path(__file__).resolve().parents[1] / "experiments" / "visual_finetune").glob(
        "*.py"
    ):
        shutil.copyfile(source, package / source.name)
        hashes[source.name] = sha(source)
    if FROZEN_TOOL_KEYS & training_recipe["code_sha256"].keys():
        hashes.update(
            freeze_coverage_helper(
                training_run / "code", out / "code", expected=training_recipe["code_sha256"]
            )
        )
    adapter = Path(adapter_path).resolve()
    receipt = read_json(adapter / "training_receipt.json")
    if receipt["recipe_sha256"] != sha(training_run / "recipe.json"):
        raise ValueError("Original training recipe differs from its checkpoint receipt")
    adapter_sha = sha(adapter / "adapter_model.safetensors")
    config_sha = sha(adapter / "adapter_config.json")
    if (
        receipt["dataset_sha256"] != training_recipe["dataset_sha256"]
        or receipt["adapter_sha256"] != adapter_sha
    ):
        raise ValueError("Adapter receipt differs from the declared training run")
    if receipt.get("adapter_config_sha256", config_sha) != config_sha:
        raise ValueError("Saved adapter configuration differs from its training receipt")
    manifest_source = Path(
        base_manifest
        or training_recipe.get(
            "base_manifest", str(Path(training_recipe["base"]) / "verified-base.json")
        )
    )
    if not manifest_source.is_absolute():
        manifest_source = training_run / manifest_source
    if (
        training_recipe.get("base_manifest_sha256")
        and sha(manifest_source) != training_recipe["base_manifest_sha256"]
    ):
        raise ValueError("Evaluation base identity differs from the training recipe")
    shutil.copyfile(manifest_source, out / "base_identity.json")
    recipe = {
        **training_recipe,
        "output": str(out.resolve()),
        "steps": 0,
        "external_evaluation": True,
        "dataset": "data/dataset.json",
        "dataset_sha256": sha(out / "data" / "dataset.json"),
        "code_sha256": hashes,
        "evaluation_record_ids": [row["record_id"] for row in records],
        "adapter": str(adapter),
        "adapter_sha256": adapter_sha,
        "adapter_config_sha256": config_sha,
        "adapter_config_identity_scope": (
            "training_checkpoint"
            if "adapter_config_sha256" in receipt
            else "post_training_frozen_before_evaluation"
        ),
        "training_recipe_sha256": sha(training_run / "recipe.json"),
        "base": str(Path(base or training_recipe["base"]).resolve()),
        "base_manifest": "base_identity.json",
        "base_manifest_sha256": sha(out / "base_identity.json"),
        "inference_only": bool(inference_only),
        "quality_approved": False,
        "supervision": {
            "minimum_available_mb": 256,
            "owned_working_set_trim": True,
            "max_seconds": max_seconds,
            "max_rss_mb": max_rss_mb,
            "absolute_deadline_utc": deadline_utc,
        },
    }
    write_json(out / "recipe.json", recipe)
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request")
    parser.add_argument("--training-run")
    parser.add_argument("--adapter")
    parser.add_argument("--out")
    parser.add_argument(
        "--run-prepared", help="Execute this immutable recipe using its frozen guards"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--base-manifest")
    parser.add_argument("--image-root", help="Resolve cross-host image refs by exact SHA filename")
    measured_python = "C:/Users/edmun/anaconda3/envs/acestep/python.exe"
    parser.add_argument(
        "--python",
        default=(
            measured_python
            if os.name == "nt" and Path(measured_python).is_file()
            else sys.executable
        ),
    )
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-rss-mb", type=int, default=4096)
    parser.add_argument("--deadline-utc", default="2026-10-31T05:01:00+00:00")
    args = parser.parse_args()
    if args.run_prepared:
        from tools.run_prepared_visual_adapter import execute

        receipt = execute(args.run_prepared, python=args.python)
        print(receipt)
        raise SystemExit(0 if receipt["exit_code"] == 0 else 1)
    if not all((args.request, args.training_run, args.adapter, args.out)):
        parser.error("--request, --training-run, --adapter and --out are required for preparation")
    if args.max_rss_mb <= 0 or args.max_seconds <= 0:
        parser.error("Owned-worker resource limits must remain positive")
    out = Path(args.out).resolve()
    prepare(
        args.request,
        args.training_run,
        args.adapter,
        out,
        base=args.base,
        base_manifest=args.base_manifest,
        inference_only=args.inference_only,
        max_seconds=args.max_seconds,
        max_rss_mb=args.max_rss_mb,
        deadline_utc=args.deadline_utc,
        image_root=args.image_root,
    )
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
            "PYTHONPATH": str(out / "code"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    receipt = run(
        [
            args.python,
            "-m",
            "experiments.visual_finetune.worker",
            str(out / "recipe.json"),
        ],
        out,
        cwd=str(out / "code"),
        environment=environment,
        max_seconds=args.max_seconds,
        max_rss_mb=args.max_rss_mb,
        deadline_utc=args.deadline_utc,
    )
    print(receipt)
    raise SystemExit(0 if receipt["exit_code"] == 0 else 1)


if __name__ == "__main__":
    main()
