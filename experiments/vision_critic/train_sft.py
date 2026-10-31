"""Reviewed three-image Qwen3-VL LoRA SFT, offline and dry-run by default.

Dry runs validate the corpus and recipe without importing torch, loading a
tokenizer, downloading weights, or claiming model compatibility. Actual GPU
execution requires --execute and a local exact base-model commit. The separate
GPU smoke must first establish package/model compatibility. External stop and
backup controls remain necessary; this process cannot stop Brev billing.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def package_evidence() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def parse_model_verdict(text: str, candidate_element_ids: dict | None = None) -> dict:
    """Untrusted generated JSON must match the small choice/issues contract."""
    from .contracts import parse_json, validate_grounded_issues

    invalid = {"choice": None, "issues": [], "valid": False}
    try:
        value = parse_json(text)
    except ValueError:
        return invalid
    if not isinstance(value, dict) or set(value) != {"choice", "issues"}:
        return invalid
    choice, issues = value["choice"], value["issues"]
    if not isinstance(choice, str) or choice not in {"A", "B", "tie", "neither"}:
        return invalid
    try:
        issues = validate_grounded_issues(issues, candidate_element_ids or {"A": [], "B": []})
    except ValueError:
        return invalid
    return {"choice": choice, "issues": issues, "valid": True}


def make_verdict_record(row: dict, text: str) -> dict:
    from .features import pair_input_identity

    return {
        "schema_version": 1,
        "pair_id": row["pair_id"],
        "corpus_sha256": row["corpus_sha256"],
        "input_sha256": pair_input_identity(row),
        **parse_model_verdict(text, row.get("candidate_element_ids")),
        "raw_response": text,
    }


def _terminate_worker(worker: subprocess.Popen) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(worker.pid, signal.SIGTERM)
    elif worker.poll() is None:
        worker.terminate()
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "posix":
            worker.kill()
    finally:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGKILL)
        worker.wait(timeout=5)


def supervise_command(command: list[str], timeout_seconds: float) -> None:
    """Bound the GPU worker, including model loading and native calls.

    Linux GPU hosts stop the worker's process group. This stops the workload,
    not Brev billing. Existing published checkpoints remain usable.
    """
    if timeout_seconds <= 0:
        raise ValueError("Worker timeout must be positive")
    worker = subprocess.Popen(command, start_new_session=os.name == "posix")
    try:
        result = worker.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_worker(worker)
        raise TimeoutError(
            "SFT worker wall-time limit reached; preserve published checkpoints"
        ) from error
    except BaseException:
        _terminate_worker(worker)
        raise
    if result:
        raise subprocess.CalledProcessError(result, command)


def prepare_plan(
    preferences: list[dict], rows: list[dict], recipe: dict, *, allow_synthetic: bool = False
) -> dict:
    from PIL import Image

    from .features import full_corpus_provenance, pair_input_identity

    provenance = full_corpus_provenance(preferences)
    preferences = [record for record in preferences if record["split"] != "test"]
    rows = [row for row in rows if row["split"] != "test"]

    if not re.fullmatch(r"[0-9a-fA-F]{40}", recipe["base_revision"]):
        raise ValueError("An exact 40-hex base-model commit is required")
    if (
        not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}", recipe["model"]
        )
        or ".." in recipe["model"]
        or Path(recipe["model"]).exists()
    ):
        raise ValueError(
            "A Hub repository ID is required; local model paths need a separate file manifest"
        )
    if not 1 <= recipe["max_steps"] <= 2000 or not 1 <= recipe["max_minutes"] <= 240:
        raise ValueError("Training requires bounded positive steps and wall time")
    if not 128 <= recipe["image_side"] <= 1024 or not 512 <= recipe["max_tokens"] <= 8192:
        raise ValueError("Image/token limits are outside the supported bounded recipe")
    if not 1 <= recipe["save_every"] <= 100 or recipe["learning_rate"] <= 0:
        raise ValueError("Invalid save interval or learning rate")
    if not allow_synthetic and any(
        "synthetic" in str(r.get("purpose", "")).lower() for r in preferences
    ):
        raise ValueError("Reviewed task training cannot use synthetic infrastructure records")
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("SFT requires non-abstain reviewed train and validation records")
    for field in ("family_id", "source_id"):
        if {row[field] for row in train} & {row[field] for row in validation}:
            raise ValueError(f"SFT {field} overlaps between train and validation")
    inspected_images = []
    for row in rows:
        if len(row["images"]) != 3 or not row["prompt"].strip():
            raise ValueError("Every SFT example needs source+A+B images and a nonempty prompt")
        response = json.loads(row["response"])
        if response.get("choice") not in {"A", "B", "tie", "neither"} or not isinstance(
            response.get("issues"), list
        ):
            raise ValueError("SFT response must contain a canonical choice and issue list")
        for path in row["images"]:
            with Image.open(path) as image:
                if image.width * image.height > 20_000_000:
                    raise ValueError("Image exceeds the framework's 20 megapixel decode limit")
                image.verify()
                inspected_images.append(
                    {"path": str(path), "width": image.width, "height": image.height}
                )
    identity = canonical_hash(
        [
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "input_sha256": pair_input_identity(row),
                "prompt": row["prompt"],
                "response": row["response"],
            }
            for row in rows
        ]
    )
    trainer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    recipe = {
        **recipe,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "lora_targets": "language_model q_proj/v_proj only",
        "mask_version": 1,
    }
    stable_recipe = {
        key: value
        for key, value in recipe.items()
        if key not in {"max_steps", "max_minutes", "save_every"}
    }
    return {
        "schema_version": 1,
        "dry_run_passed": True,
        "gpu_compatibility_validated": False,
        "task_model_trained": False,
        "model": recipe["model"],
        "base_revision": recipe["base_revision"],
        "recipe": recipe,
        "recipe_sha256": canonical_hash({**stable_recipe, "trainer_sha256": trainer_sha256}),
        "trainer_sha256": trainer_sha256,
        "corpus_sha256": identity,
        "identity_scope": "non-test input bytes and literal prompts/responses",
        "full_corpus_provenance": provenance,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_families": sorted({r["family_id"] for r in train}),
        "validation_families": sorted({r["family_id"] for r in validation}),
        "images_inspected": inspected_images,
        "packages": package_evidence(),
        "purpose": "synthetic_infrastructure_only"
        if allow_synthetic
        else "reviewed_preference_training",
        "mask_format_plan": {
            "input": "user: source+A+B images and prompt; assistant: JSON",
            "labels": "mask the exact multimodal prefix with -100; train assistant JSON only",
            "truncation": "never truncate; reject runtime token sequences above max_tokens",
            "tokenizer_validation": "pending actual local processor/tokenizer execution",
        },
    }


def validate_resume(progress: dict, plan: dict) -> None:
    for key in (
        "base_model",
        "base_revision",
        "corpus_sha256",
        "recipe_sha256",
        "packages",
        "trainer_sha256",
    ):
        expected = plan["model"] if key == "base_model" else plan[key]
        if progress.get(key) != expected:
            raise ValueError(f"Resume checkpoint identity differs: {key}")
    if not isinstance(progress.get("completed_steps"), int) or progress["completed_steps"] < 1:
        raise ValueError("Resume checkpoint has no completed optimizer step")
    if progress["completed_steps"] >= plan["recipe"]["max_steps"]:
        raise ValueError("Resume target steps must exceed the saved completed steps")


def execute(
    plan: dict, rows: list[dict], out: Path, resume: Path | None, allow_download: bool
) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    from tools.vision_lora_smoke import (
        adapter_digest,
        publish_checkpoint,
        verify_checkpoint,
        write_json,
    )

    if plan["purpose"] != "reviewed_preference_training":
        raise ValueError(
            "Task SFT requires reviewed real preferences; use the separate synthetic smoke"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT requires CUDA with BF16 support")
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError("SFT requires at least 16 GiB free VRAM; actual usage must be measured")
    recipe = plan["recipe"]
    random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    deadline = started + recipe["max_minutes"] * 60
    loading = {
        "revision": plan["base_revision"],
        "token": False,
        "trust_remote_code": False,
        "local_files_only": not allow_download,
    }
    configuration = AutoConfig.from_pretrained(recipe["model"], **loading)
    if configuration.model_type != "qwen3_vl":
        raise ValueError("This trainer supports dense Qwen3-VL checkpoints only")
    if configuration._commit_hash != plan["base_revision"]:
        raise ValueError("Resolved base-model configuration differs from the pinned commit")
    processor = AutoProcessor.from_pretrained(
        recipe["model"], **loading, min_pixels=128 * 128, max_pixels=recipe["image_side"] ** 2
    )

    def base():
        return Qwen3VLForConditionalGeneration.from_pretrained(
            recipe["model"],
            **loading,
            config=configuration,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

    def inputs(row: dict, *, generation: bool = False):
        with_images = [{"type": "image"} for _ in range(3)]
        user = [
            {"role": "user", "content": [*with_images, {"type": "text", "text": row["prompt"]}]}
        ]
        prefix_text = processor.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True
        )
        images = []
        for path in row["images"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        prefix = processor(text=[prefix_text], images=images, return_tensors="pt")
        prefix.pop("token_type_ids", None)
        if generation:
            if prefix["input_ids"].shape[1] > recipe["max_tokens"]:
                raise ValueError("Multimodal prompt exceeds max_tokens; truncation is forbidden")
            return {name: value.to("cuda") for name, value in prefix.items()}
        full_text = processor.apply_chat_template(
            [*user, {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = processor(text=[full_text], images=images, return_tensors="pt")
        batch.pop("token_type_ids", None)
        prefix_length = prefix["input_ids"].shape[1]
        if not torch.equal(batch["input_ids"][:, :prefix_length], prefix["input_ids"]):
            raise ValueError("Processor prefix differs; refusing an incorrect assistant-only mask")
        if batch["input_ids"].shape[1] > recipe["max_tokens"]:
            raise ValueError(
                "Multimodal training sequence exceeds max_tokens; truncation is forbidden"
            )
        labels = batch["input_ids"].clone()
        labels[:, :prefix_length] = -100
        if not bool((labels != -100).any()):
            raise ValueError("No assistant tokens remain for loss")
        batch["labels"] = labels
        return {name: value.to("cuda") for name, value in batch.items()}

    model = base()
    completed, history = 0, []
    if resume:
        verify_checkpoint(resume)
        progress = json.loads((resume / "progress.json").read_text())
        validate_resume(progress, plan)
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
    else:
        targets = [
            name
            for name, layer in model.named_modules()
            if ".language_model." in name
            and isinstance(layer, torch.nn.Linear)
            and name.endswith((".q_proj", ".v_proj"))
        ]
        if not targets:
            raise RuntimeError("No language-only q/v projection targets were found")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=targets,
                r=8,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
            ),
        )
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not named or any(
        "lora_" not in name or ".visual." in name or "merger" in name for name, _ in named
    ):
        raise RuntimeError("Vision/projector/base weights must remain frozen")
    parameters = [parameter for _, parameter in named]
    optimizer = torch.optim.AdamW(parameters, lr=recipe["learning_rate"])
    if resume:
        state = torch.load(resume / "training-state.pt", map_location="cpu", weights_only=True)
        validate_resume(state, plan)
        optimizer.load_state_dict(state["optimizer"])
        completed, history = state["completed_steps"], state["loss_history"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
    model.config.use_cache = False
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    published = out / "published"
    published.mkdir()
    last_checkpoint = None

    def save():
        metadata = {
            "base_model": recipe["model"],
            "base_revision": plan["base_revision"],
            "corpus_sha256": plan["corpus_sha256"],
            "recipe_sha256": plan["recipe_sha256"],
            "completed_steps": completed,
            "loss_history": history,
            "task_model_trained": True,
            "purpose": plan["purpose"],
            "packages": plan["packages"],
            "trainer_sha256": plan["trainer_sha256"],
            "recipe": recipe,
        }
        stage = out / f".checkpoint-{completed:06d}.partial"
        stage.mkdir()
        model.save_pretrained(stage, safe_serialization=True)
        torch.save(
            {
                **metadata,
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
            },
            stage / "training-state.pt",
        )
        write_json(stage / "progress.json", metadata)
        checkpoint = published / f"checkpoint-{completed:06d}"
        publish_checkpoint(stage, checkpoint)
        return checkpoint

    model.train()
    for index in range(completed, recipe["max_steps"]):
        if time.monotonic() >= deadline:
            break
        optimizer.zero_grad(set_to_none=True)
        batch = inputs(train[index % len(train)])
        loss = model(**batch, use_cache=False).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Training produced a non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise RuntimeError("Trainable adapter gradients are invalid")
        optimizer.step()
        completed = index + 1
        history.append(float(loss.detach()))
        if completed % recipe["save_every"] == 0 or completed == recipe["max_steps"]:
            last_checkpoint = save()
        del batch, loss
    if completed == 0 or (resume and completed == progress["completed_steps"]):
        raise TimeoutError("Deadline reached before a new training step completed")
    if last_checkpoint is None or int(last_checkpoint.name.rsplit("-", 1)[1]) != completed:
        last_checkpoint = save()
    expected_digest = adapter_digest(model)
    verify_checkpoint(last_checkpoint)
    optimizer = None
    del parameters, named, model, norm
    gc.collect()
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(base(), last_checkpoint, is_trainable=False)
    if adapter_digest(model) != expected_digest:
        raise RuntimeError("Reloaded trained adapter differs from the saved adapter")
    model.eval()
    validation_losses, verdicts = [], []
    for row in validation:
        if time.monotonic() >= deadline:
            break
        with torch.inference_mode():
            loss = float(model(**inputs(row), use_cache=False).loss)
            if not math.isfinite(loss):
                raise RuntimeError("Reloaded model produced a non-finite held-out loss")
            validation_losses.append(loss)
            batch = inputs(row, generation=True)
            generated = model.generate(**batch, max_new_tokens=96, do_sample=False, use_cache=True)
            text = processor.batch_decode(
                generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
        verdicts.append(make_verdict_record(row, text))
    with (out / "verdicts.jsonl").open("w", encoding="utf-8") as stream:
        for verdict in verdicts:
            stream.write(json.dumps(verdict) + "\n")
    report = {
        "schema_version": 1,
        "task_model_trained": True,
        "gpu_training_completed": True,
        "adapter_reload_verified": True,
        "base_revision": plan["base_revision"],
        "corpus_sha256": plan["corpus_sha256"],
        "recipe_sha256": plan["recipe_sha256"],
        "completed_steps": completed,
        "validation_examples_evaluated": len(validation_losses),
        "validation_examples_total": len(validation),
        "validation_complete": len(validation_losses) == len(validation),
        "mean_validation_loss": sum(validation_losses) / len(validation_losses)
        if validation_losses
        else None,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(last_checkpoint),
        "packages": package_evidence(),
    }
    write_json(out / "training-report.json", report)
    return report


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker-plan":
        payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        execute(
            payload["plan"],
            payload["rows"],
            Path(payload["out"]),
            Path(payload["resume"]) if payload["resume"] else None,
            payload["allow_download"],
        )
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--base-revision", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-minutes", type=int, default=15)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--image-side", type=int, default=448)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.execute and args.allow_synthetic:
        parser.error("Synthetic corpus is permitted for dry-run infrastructure checks only")
    if args.allow_download and not args.execute:
        parser.error("Dry-run never downloads weights; remove --allow-download")
    from .dataset import load_preferences, sft_records
    from .train_ranker import atomic_json

    preferences = load_preferences(
        args.collection, args.labels, allow_synthetic=args.allow_synthetic
    )
    training_preferences = [record for record in preferences if record["split"] != "test"]
    rows = list(sft_records(training_preferences))
    recipe = {
        "model": args.model,
        "base_revision": args.base_revision,
        "max_steps": args.max_steps,
        "max_minutes": args.max_minutes,
        "save_every": args.save_every,
        "image_side": args.image_side,
        "max_tokens": args.max_tokens,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    plan = prepare_plan(training_preferences, rows, recipe, allow_synthetic=args.allow_synthetic)
    if args.resume:
        from tools.vision_lora_smoke import verify_checkpoint

        verify_checkpoint(args.resume)
        validate_resume(json.loads((args.resume / "progress.json").read_text()), plan)
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "dry-run-report.json", plan)
    if args.execute:
        payload_path = args.out / "worker-plan.json"
        atomic_json(
            payload_path,
            {
                "plan": plan,
                "rows": rows,
                "out": str(args.out),
                "resume": str(args.resume) if args.resume else None,
                "allow_download": args.allow_download,
            },
        )
        supervise_command(
            [
                sys.executable,
                "-m",
                "experiments.vision_critic.train_sft",
                "--_worker-plan",
                str(payload_path),
            ],
            args.max_minutes * 60,
        )
        report = json.loads((args.out / "training-report.json").read_text(encoding="utf-8"))
    else:
        report = plan
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
