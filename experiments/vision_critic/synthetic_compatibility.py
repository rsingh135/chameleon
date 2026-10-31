"""Bounded weak-label curriculum experiment, never a task-quality claim.

Requires explicit synthetic opt-in. Dry-run is the default. Execution uses only
an already cached exact public base commit and compares synthetic validation
agreement before/after a tiny LoRA run. Test judgments remain concealed. This
does not weaken the reviewed-data gate in train_sft.py or stop Brev billing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path


def prepare(
    collection: Path, labels: Path, revision: str, steps: int, max_seconds: int
) -> tuple[dict, list]:
    from .dataset import load_preferences, sft_records
    from .features import corpus_identity, full_corpus_provenance

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An exact lowercase 40-hex cached base commit is required")
    if steps not in {12, 24} or not 30 <= max_seconds <= 300:
        raise ValueError("Synthetic run must use 12/24 steps and 30..300 seconds")
    preferences = load_preferences(collection, labels, allow_synthetic=True)
    if not preferences or any(row["purpose"] != "synthetic" for row in preferences):
        raise ValueError("This compatibility harness accepts explicitly synthetic collections only")
    preferences = [row for row in preferences if row["split"] != "test"]
    rows = sft_records(preferences)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation or any(len(row["images"]) != 3 for row in rows):
        raise ValueError("Source+A+B train and validation records are required")
    for key in ("family_id", "source_id"):
        if {row[key] for row in train} & {row[key] for row in validation}:
            raise ValueError("Synthetic training and validation sources/families overlap")
    plan = {
        "schema_version": 1,
        "purpose": "synthetic_curriculum_compatibility_only",
        "task_model_trained": False,
        "gpu_compatibility_validated": False,
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_revision": revision,
        "requested_steps": steps,
        "max_seconds": max_seconds,
        "training_identity": corpus_identity(preferences),
        "full_corpus_provenance": full_corpus_provenance(preferences),
        "train_count": len(train),
        "validation_count": len(validation),
        "test_split_unblinded": False,
        "image_count_per_example": 3,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "label_scope": "Programmatic weak expectations, not human preference ground truth",
    }
    return plan, rows


def agreement(verdicts: list[dict]) -> dict:
    return {
        "count": len(verdicts),
        "schema_valid_count": sum(row["valid"] for row in verdicts),
        "weak_label_choice_agreement": sum(row["matches_weak_target"] for row in verdicts)
        / len(verdicts)
        if verdicts
        else None,
        "scope": "Synthetic weak-label agreement only; no real design-quality measurement",
    }


def execute(plan: dict, rows: list[dict], out: Path) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    from tools.vision_lora_smoke import (
        adapter_digest,
        packages,
        publish_checkpoint,
        verify_checkpoint,
        write_json,
    )

    from .train_sft import parse_model_verdict

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Synthetic compatibility requires CUDA and BF16")
    started = time.monotonic()
    deadline = started + plan["max_seconds"]
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.cuda.reset_peak_memory_stats()
    loading = {
        "revision": plan["base_revision"],
        "local_files_only": True,
        "trust_remote_code": False,
        "token": False,
    }
    config = AutoConfig.from_pretrained(plan["base_model"], **loading)
    if config.model_type != "qwen3_vl" or config._commit_hash != plan["base_revision"]:
        raise ValueError("Cached base configuration differs from the pinned dense Qwen3-VL model")
    processor = AutoProcessor.from_pretrained(
        plan["base_model"], **loading, min_pixels=128**2, max_pixels=192**2
    )

    def base():
        return Qwen3VLForConditionalGeneration.from_pretrained(
            plan["base_model"],
            **loading,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

    def batch(row, training):
        user = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(3)],
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        prompt = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        images = []
        for path in row["images"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        prefix = processor(text=[prompt], images=images, return_tensors="pt")
        prefix.pop("token_type_ids", None)
        if prefix["image_grid_thw"].shape[0] != 3:
            raise ValueError("Expected three source/A/B images")
        if training:
            text = processor.apply_chat_template(
                [
                    *user,
                    {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            values = processor(text=[text], images=images, return_tensors="pt")
            values.pop("token_type_ids", None)
            length = prefix["input_ids"].shape[1]
            if not torch.equal(values["input_ids"][:, :length], prefix["input_ids"]):
                raise ValueError("Assistant-only prefix mask differs from the processor template")
            labels = values["input_ids"].clone()
            labels[:, :length] = -100
            if not bool((labels != -100).any()):
                raise ValueError("Assistant loss has no unmasked tokens")
            values["labels"] = labels
        else:
            values = prefix
        if values["input_ids"].shape[1] > 1536:
            raise ValueError("Synthetic sequence exceeds 1536 tokens; truncation is forbidden")
        return {name: value.to("cuda") for name, value in values.items()}

    validation = [row for row in rows if row["split"] == "validation"]
    train = [row for row in rows if row["split"] == "train"]

    def evaluate(model):
        verdicts = []
        model.eval()
        for row in validation:
            if time.monotonic() >= deadline - 5:
                break
            inputs = batch(row, False)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=96, do_sample=False, use_cache=True
                )
            text = processor.batch_decode(
                output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            parsed = parse_model_verdict(text, row["candidate_element_ids"])
            verdicts.append(
                {
                    "pair_id": row["pair_id"],
                    **parsed,
                    "raw_response": text,
                    "matches_weak_target": parsed["valid"]
                    and parsed["choice"] == json.loads(row["response"])["choice"],
                }
            )
        return verdicts

    model = base()
    before = evaluate(model)
    write_json(out / "before-validation.json", {"metrics": agreement(before), "verdicts": before})
    targets = [
        name
        for name, layer in model.named_modules()
        if ".language_model." in name
        and isinstance(layer, torch.nn.Linear)
        and name.endswith((".q_proj", ".v_proj"))
    ]
    if not targets:
        raise RuntimeError("Language-only LoRA projection targets were not found")
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
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    initial_digest = adapter_digest(model)
    losses, checkpoint, completed = [], None, 0
    (out / "published").mkdir()
    model.train()
    model.config.use_cache = False
    for index in range(plan["requested_steps"]):
        if time.monotonic() > started + plan["max_seconds"] * 0.65:
            break
        optimizer.zero_grad(set_to_none=True)
        values = batch(train[index % len(train)], True)
        loss = model(**values, use_cache=False).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Synthetic training loss is non-finite")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise RuntimeError("Synthetic adapter gradients are invalid")
        optimizer.step()
        losses.append(float(loss.detach()))
        completed = index + 1
        # Publish every completed step: a deadline interruption retains the latest complete work.
        stage = out / f".checkpoint-{completed:06d}.partial"
        stage.mkdir()
        model.save_pretrained(stage, safe_serialization=True)
        progress = {
            "base_model": plan["base_model"],
            "base_revision": plan["base_revision"],
            "completed_steps": completed,
            "corpus_sha256": plan["training_identity"],
            "purpose": plan["purpose"],
            "task_model_trained": False,
            "loss": losses[-1],
            "packages": packages(),
            "script_sha256": plan["script_sha256"],
        }
        torch.save(
            {
                **progress,
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            stage / "training-state.pt",
        )
        write_json(stage / "progress.json", progress)
        checkpoint = out / "published" / f"checkpoint-{completed:06d}"
        publish_checkpoint(stage, checkpoint)
        print(json.dumps({"synthetic_step": completed, "loss": losses[-1]}), flush=True)
        del values, loss, norm
    if not completed:
        raise TimeoutError(
            "No optimization step completed within the reserved synthetic training budget"
        )
    digest = adapter_digest(model)
    if digest == initial_digest:
        raise RuntimeError("Synthetic LoRA adapter did not update")
    optimizer = None
    del model, parameters, named
    gc.collect()
    torch.cuda.empty_cache()
    verify_checkpoint(checkpoint)
    model = PeftModel.from_pretrained(base(), checkpoint, is_trainable=False)
    if adapter_digest(model) != digest:
        raise RuntimeError("Synthetic adapter reload digest differs")
    after = evaluate(model)
    write_json(out / "after-validation.json", {"metrics": agreement(after), "verdicts": after})
    before_metrics, after_metrics = agreement(before), agreement(after)
    complete = len(before) == len(after) == len(validation)
    delta = (
        after_metrics["weak_label_choice_agreement"] - before_metrics["weak_label_choice_agreement"]
        if complete
        else None
    )
    report = {
        **plan,
        "gpu_compatibility_validated": True,
        "completed_steps": completed,
        "task_model_trained": False,
        "base_validation": before_metrics,
        "trained_validation": after_metrics,
        "validation_complete": complete,
        "synthetic_agreement_delta": delta,
        "losses": losses,
        "checkpoint": str(checkpoint),
        "adapter_reload_verified": True,
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
        "packages": packages(),
        "quality_claim": "No real quality claim; weak-label synthetic experiment only",
    }
    write_json(out / "synthetic-report.json", report)
    return report


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker-plan":
        payload = json.loads(Path(sys.argv[2]).read_text())
        execute(payload["plan"], payload["rows"], Path(payload["out"]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--steps", type=int, choices=(12, 24), default=12)
    parser.add_argument("--max-seconds", type=int, default=240)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.allow_synthetic:
        parser.error("Explicit --allow-synthetic is required; this is not reviewed task training")
    plan, rows = prepare(
        args.collection, args.labels, args.base_revision, args.steps, args.max_seconds
    )
    args.out.mkdir(parents=True, exist_ok=False)
    from tools.vision_lora_smoke import write_json

    from .train_sft import supervise_command

    write_json(args.out / "synthetic-plan.json", plan)
    if args.execute:
        payload = args.out / "worker-plan.json"
        write_json(payload, {"plan": plan, "rows": rows, "out": str(args.out)})
        supervise_command(
            [
                sys.executable,
                "-m",
                "experiments.vision_critic.synthetic_compatibility",
                "--_worker-plan",
                str(payload),
            ],
            args.max_seconds,
        )
        report = json.loads((args.out / "synthetic-report.json").read_text())
    else:
        report = plan
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
