"""Create an offline-ready cloud bundle without starting or contacting a cloud VM."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from experiments.visual_finetune.data import (
    ALIASES,
    BASE_MODEL_SHA256,
    BASE_REPO,
    BASE_REVISION,
    PURPOSE_CLASSES,
    _copy_image,
    _local,
    critic_rows,
    label_free_rows,
    read_json,
    sha,
    write_json,
)

PINS = {
    "python": "3.11.14",
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "transformers": "4.57.6",
    "peft": "0.18.1",
    "accelerate": "1.13.0",
    "safetensors": "0.7.0",
    "Pillow": "12.1.1",
}


def normalize_shell_scripts(directory):
    for script in Path(directory).glob("*.sh"):
        script.write_bytes(script.read_bytes().replace(b"\r", b""))


def portable_export(manifest_path, destination):
    manifest_path, destination = Path(manifest_path).resolve(), Path(destination)
    original = read_json(manifest_path)
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in manifest["files_sha256"].items():
        source = _local(root, relative)
        if sha(source) != expected:
            raise ValueError("Reviewed export changed before packaging: " + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for entry in manifest["sources"]:
        receipt = read_json(_local(root, entry["receipt"]))
        source = (
            _local(root, entry["source_json"])
            if entry.get("source_json")
            else Path(receipt["source_path"])
        )
        expected = receipt["source_json_sha256"]
        if sha(source) != expected:
            raise ValueError("Native source differs from its reviewed receipt")
        relative = "native-sources/" + expected + ".json"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entry["source_json"] = relative
        manifest["files_sha256"][relative] = expected
    manifest["portable_export"] = {
        "original_manifest_sha256": sha(manifest_path),
        "source_json_copies_sha_verified": True,
        "absolute_receipt_paths": "historical_provenance_only_not_consumed_by_training_loader",
        "original_human_reviewed": original["human_reviewed"],
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def portable_critic_export(manifest_path, evaluation_request, destination):
    destination = Path(destination)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    training, provenance = critic_rows(manifest_path)
    evaluation = label_free_rows(evaluation_request, task="critic")
    if any(row["split"] != "train" for row in training):
        raise ValueError("This critic bundle is explicitly train-only")
    if {row["family_id"] for row in training} & {row["family_id"] for row in evaluation}:
        raise ValueError("Critic training and independent evaluation families overlap")
    for row in training + evaluation:
        row["images"] = [
            {"path": "images/" + Path(ref["path"]).name, "sha256": ref["sha256"]}
            for ref in [_copy_image(ref["path"], ref["sha256"], image_dir) for ref in row["images"]]
        ]
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "label_scope": "algorithmic_weak_not_human_or_aesthetic",
            "provenance": provenance,
            "records": training,
        },
    )
    write_json(
        destination / "label-free-evaluation.json",
        {
            "protocol": "visual-adapter-label-free-evaluation-v1",
            "task": "critic",
            "original_request_sha256": sha(evaluation_request),
            "records": evaluation,
        },
    )
    return {
        "portable_export": {"original_manifest_sha256": sha(manifest_path)},
        "evaluation_request_sha256": sha(evaluation_request),
    }


def package(manifest_path, out, base_manifest, *, task="purpose", evaluation_request=None):
    root, out = Path(__file__).resolve().parents[1], Path(out).resolve()
    archive = out.with_suffix(".tar.gz")
    if out.exists() or archive.exists():
        raise ValueError("A cloud bundle must use a new immutable output directory")
    out.mkdir(parents=True)
    if task == "critic":
        if not evaluation_request:
            raise ValueError("Critic bundle requires separate immutable label-free inputs")
        export = portable_critic_export(manifest_path, evaluation_request, out / "export")
    else:
        export = portable_export(manifest_path, out / "export")
    for name in ("experiments", "tools"):
        (out / name).mkdir(exist_ok=True)
        (out / name / "__init__.py").write_text("", encoding="utf-8")
    package_dir = out / "experiments" / "visual_finetune"
    package_dir.mkdir()
    for source in (root / "experiments" / "visual_finetune").glob("*.py"):
        shutil.copyfile(source, package_dir / source.name)
    shutil.copyfile(root / "experiments/visual_finetune/README.md", package_dir / "README.md")
    for name in (
        "train_visual_adapter.py",
        "evaluate_visual_adapter.py",
        "package_visual_training.py",
        "prepare_visual_base.py",
    ):
        shutil.copyfile(root / "tools" / name, out / "tools" / name)
    references = out / "references"
    references.mkdir()
    identity = read_json(base_manifest)
    if (
        identity["repo_id"],
        identity["revision"],
        identity["files"]["model.safetensors"]["sha256"],
    ) != (BASE_REPO, BASE_REVISION, BASE_MODEL_SHA256):
        raise ValueError("Bundle base differs from the verified closed model profile")
    shutil.copyfile(base_manifest, references / "base-identity.json")
    metadata = references / "base-metadata"
    metadata.mkdir()
    for name, expected in identity["files"].items():
        if name == "model.safetensors":
            continue
        source = Path(base_manifest).parent / name
        if sha(source) != expected["sha256"]:
            raise ValueError("Local base metadata changed before packaging")
        shutil.copyfile(source, metadata / name)
    write_json(
        references / "runtime-pins.json",
        {
            "measured_local": PINS,
            "cloud_operator_reported": {
                "python": "3.10",
                "torch": "2.14.1+cu126",
                "transformers": "4.57.6",
                "peft": "0.18.1",
                "accelerate": "1.12",
                "Pillow": "12",
            },
            "cloud_runtime_verified": False,
            "install_requested": False,
            "global_install_permitted": False,
        },
    )
    write_json(
        out / "purpose-contract.json",
        {
            "schema_version": 1,
            "task": "purpose",
            "model_answers": list(PURPOSE_CLASSES),
            "canonical_role_to_answer": ALIASES,
            "answer_token_count": 1,
            "GUI_12_letter_taxonomy": "not_accepted_or_implicitly_mapped",
            "critic_A_B_N": "separate_task_and_adapter_not_a_purpose_mapping",
            "label_scope": "agent_visual_review_and_programmatic_controls_not_human",
        },
    )
    if task == "critic":
        write_json(
            out / "critic-contract.json",
            {
                "task": "critic",
                "model_answers": ["A", "B", "N"],
                "canonical_choice": {"A": "candidate_A", "B": "candidate_B", "N": "NEITHER"},
                "purpose_words": "separate_task_and_adapter_not_used",
                "label_scope": "algorithmic_weak_not_human_or_aesthetic",
                "evaluation_labels_available_to_trainer": False,
            },
        )
    (out / "run-purpose.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        ': "${MODEL_BASE:?Set MODEL_BASE to the verified pinned local base snapshot}"\n'
        ': "${RUN_OUT:?Set RUN_OUT to a new project-local output directory}"\n'
        'python -m tools.train_visual_adapter --python "$(command -v python)" '
        '--manifest export/manifest.json --task purpose --base "$MODEL_BASE" '
        '--base-manifest references/base-identity.json --out "$RUN_OUT" '
        "--steps 32 --evaluation-limit 48 --sampling balanced_class_cycle "
        "--learning-rate 0.00002 --max-pixels 65536 --max-seconds 900 --max-rss-mb 16384 "
        "--deadline-utc 2026-10-31T05:01:00+00:00\n",
        encoding="utf-8",
    )
    if task == "critic":
        (out / "run-critic.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            ': "${MODEL_BASE:?Set MODEL_BASE to the verified pinned local base snapshot}"\n'
            ': "${RUN_OUT:?Set RUN_OUT to a new project-local output directory}"\n'
            'python -m tools.train_visual_adapter --python "$(command -v python)" '
            '--manifest export/manifest.json --task critic --base "$MODEL_BASE" '
            '--base-manifest references/base-identity.json --out "$RUN_OUT" '
            "--allow-train-only --validation-request export/label-free-evaluation.json "
            "--steps 32 --evaluation-limit 48 --learning-rate 0.00002 "
            "--max-pixels 262144 --max-seconds 900 --max-rss-mb 16384 "
            "--deadline-utc 2026-10-31T05:01:00+00:00\n",
            encoding="utf-8",
        )
    (out / "RUNBOOK.md").write_text(
        "# Purpose context-v2 cloud experiment\n\n"
        "Preparation only: this bundle does not start a VM, install packages, "
        "or authorize spending.\n\n"
        "Use the existing approved Python3.10 cloud environment; do not install or "
        "alter packages merely to match the measured local runtime. Actual package "
        "versions are emitted by the worker. See references/runtime-pins.json for "
        "measured local versus operator-reported cloud facts. MODEL_BASE points to a local "
        "Qwen/Qwen3-VL-2B-Instruct snapshot at revision " + BASE_REVISION + ". "
        "Base weights are deliberately not included; every model/config/tokenizer byte "
        "is checked against references/base-identity.json before inference. "
        "The worker is offline and never fetches weights implicitly.\n\n"
        "If the cached snapshot lacks optional metadata files, create a project-local "
        "overlay without modifying the old cache: `python -m tools.prepare_visual_base "
        "--weights /actual/cached/model.safetensors --metadata references/base-metadata "
        "--manifest references/base-identity.json --out models/qwen3-vl-2b`. "
        "Set MODEL_BASE to that new overlay; Linux uses a SHA-verified weight symlink.\n\n"
        "From this extracted directory, set MODEL_BASE and a new RUN_OUT, then run "
        "`bash run-purpose.sh`. This uses fresh pretrained weights, not an exact resume "
        "of the historical step-8 adapter. The fixed run uses 32 steps, r4/alpha8, "
        "language q/v only, BF16 SDPA, LR2e-5, accumulation4, class-balanced sampling, "
        "on-demand two-image preprocessing and atomic full-state saves every4steps. "
        "Vision/base weights are frozen. Validation is scored before training and after.\n\n"
        "The owned Linux process group is time/RAM bounded (900s, sustained host "
        "availability256MiB, cloud hard stop2026-10-31T05:01:00Z). A process timeout "
        "does not stop cloud billing: the operator must separately arrange approved "
        "instance shutdown and the external spending deadline before starting.\n\n"
        "Inputs use SHA-verified relative native-source copies. Historical absolute "
        "paths inside receipts are provenance only. No reserve/manual cases or sealed "
        "test labels are included. The purpose answer contract is exactly ten words; "
        "GUI twelve-letter labels and the separate critic A/B/N task are not mapped. "
        "Semantic purpose does not authorize cover/contain, grouping, or geometry.\n",
        encoding="utf-8",
    )
    if task == "critic":
        (out / "RUNBOOK.md").write_text(
            "# Separate source-conditioned A/B/N critic adapter\n\n"
            "Use the approved existing Python3.10 environment and verified cached base; "
            "this bundle starts no VM and installs nothing. MODEL_BASE and RUN_OUT must "
            "be set to project-local paths, then run `bash run-critic.sh`. The base overlay "
            "helper and references/base-identity.json work as in the purpose pilot.\n\n"
            "Thirty train-only rows are algorithmic weak labels over five source families; "
            "A/B/N each have ten examples. They demonstrate fitting corruption examples, "
            "not human aesthetic preference. Twenty separate label-free independent inputs "
            "are scored before/after training and never supply gradients or gold labels. "
            "Only the independent evaluator holds gold. "
            "No accuracy is computed by this trainer.\n\n"
            "Images are source,A,B. Fixed answer words are A,B,N where N=NEITHER; this is "
            "not the ten-word purpose task. Training is fresh-base r4/alpha8, LR2e-5, "
            "balanced six-example accumulation (two of each label),32steps,512px image "
            "budget, full-state atomic checkpoints4steps. Owned-group supervision caps "
            "the run at900s, guards256MiB available RAM, and stops by05:01UTC. Cloud "
            "billing shutdown is a separate operator responsibility. Deployment remains off.\n",
            encoding="utf-8",
        )
    # Windows text mode translates LF to CRLF. Portable shell payloads must be
    # written as bytes, then hashed only after their exact LF bytes are fixed.
    if task == "critic":
        (out / "run-purpose.sh").unlink()
        (out / "purpose-contract.json").unlink()
    normalize_shell_scripts(out)
    files = {
        path.relative_to(out).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    write_json(
        out / "bundle-manifest.json",
        {
            "schema_version": 1,
            "protocol": "visual-" + task + "-cloud-bundle-v1",
            "files": files,
            "original_export_sha256": export["portable_export"]["original_manifest_sha256"],
            "portable_export_sha256": sha(out / "export" / "manifest.json"),
            "base": {
                "repo": BASE_REPO,
                "revision": BASE_REVISION,
                "model_sha256": BASE_MODEL_SHA256,
                "weights_included": False,
            },
            "task": task,
            "classes": list(PURPOSE_CLASSES) if task == "purpose" else ["A", "B", "N"],
            "paid_launch_authorized": False,
            "quality_approved": False,
        },
    )
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                stream.add(path, arcname=path.relative_to(out).as_posix(), recursive=False)
    return {
        "bundle": str(out),
        "manifest_sha256": sha(out / "bundle-manifest.json"),
        "archive": str(archive),
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "files": len(files),
        "paid_launch_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--task", choices=("purpose", "critic"), default="purpose")
    parser.add_argument("--validation-request")
    args = parser.parse_args()
    import json

    print(
        json.dumps(
            package(
                args.manifest,
                args.out,
                args.base_manifest,
                task=args.task,
                evaluation_request=args.validation_request,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
