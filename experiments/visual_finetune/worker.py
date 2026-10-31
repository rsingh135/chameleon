"""A bounded, offline, answer-only Qwen3-VL LoRA experiment on one CUDA GPU."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from .checkpoint import publish, staging_directory
from .data import (
    BASE_MODEL_SHA256,
    BASE_REPO,
    BASE_REVISION,
    PURPOSE_CLASSES,
    digest,
    read_json,
    sha,
    write_json,
)


def main(recipe_path):
    recipe_path = Path(recipe_path).resolve()
    recipe = read_json(recipe_path)
    for name, expected in recipe["code_sha256"].items():
        if Path(name).name != name or sha(Path(__file__).parent / name) != expected:
            raise ValueError("Isolated experiment code differs from its frozen recipe")
    out = Path(recipe["output"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def phase(name, **details):
        event = {"phase": name, "seconds": time.monotonic() - started, **details}
        with (out / "phases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event), flush=True)

    phase("imports_begin")
    import torch

    phase("torch_imported")
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    phase("transformers_imported")

    phase(
        "imports_complete",
        python=sys.version,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "Pillow", "safetensors")
        },
    )
    torch.set_num_threads(2)
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires the measured BF16 CUDA device")
    dataset_path = Path(recipe["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = recipe_path.parent / dataset_path
    if sha(dataset_path) != recipe["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    dataset = read_json(dataset_path)
    base = Path(recipe["base"])
    manifest_path = Path(recipe.get("base_manifest", str(base / "verified-base.json")))
    if not manifest_path.is_absolute():
        manifest_path = recipe_path.parent / manifest_path
    if recipe.get("base_manifest_sha256") and sha(manifest_path) != recipe["base_manifest_sha256"]:
        raise ValueError("Frozen base identity manifest changed")
    manifest = read_json(manifest_path)
    if (
        manifest["repo_id"],
        manifest["revision"],
        manifest["files"]["model.safetensors"]["sha256"],
    ) != (BASE_REPO, BASE_REVISION, BASE_MODEL_SHA256):
        raise ValueError("Base differs from this closed experiment's pinned Qwen3-VL profile")
    for name, expected in manifest["files"].items():
        if sha(base / name) != expected["sha256"]:
            raise ValueError("Pinned base file changed: " + name)
    phase("base_verified", revision=manifest["revision"], device=torch.cuda.get_device_name(0))
    processor = AutoProcessor.from_pretrained(
        str(base),
        local_files_only=True,
        trust_remote_code=False,
        min_pixels=recipe["min_pixels"],
        max_pixels=recipe["max_pixels"],
    )
    classes = dataset["classes"]
    expected_classes = list(PURPOSE_CLASSES) if dataset["task"] == "purpose" else ["A", "B", "N"]
    if classes != expected_classes:
        raise ValueError("Dataset classes differ from the exact task-specific word contract")
    token_ids = []
    for label in classes:
        encoded = processor.tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError("Short-answer experiment requires single-token class aliases")
        token_ids.append(encoded[0])
    phase(
        "load_begin",
        host_copy="direct_cuda_low_cpu_mem_usage",
        classes=classes,
        class_token_ids=token_ids,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("Silent CPU parameter offload is outside this run recipe")
    phase(
        "load_complete",
        allocated_mb=torch.cuda.memory_allocated() / 2**20,
        reserved_mb=torch.cuda.memory_reserved() / 2**20,
    )
    tokens = torch.tensor(token_ids, device="cuda:0")
    label_index = {name: index for index, name in enumerate(classes)}

    def prepare(row):
        pictures = []
        for ref in row["images"]:
            path = Path(ref["path"])
            if not path.is_absolute():
                path = dataset_path.parent / path
            if sha(path) != ref["sha256"]:
                raise ValueError("Frozen input image changed")
            with Image.open(path) as picture:
                pictures.append(picture.convert("RGB"))
        message = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in pictures]
                + [{"type": "text", "text": row["prompt"]}],
            }
        ]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[prompt], images=pictures, padding=False, return_tensors="pt")
        for picture in pictures:
            picture.close()
        if batch["input_ids"].shape[1] > recipe["max_input_tokens"]:
            raise ValueError("Input exceeds the declared token bound; no silent truncation")
        if len(batch["image_grid_thw"]) != len(row["images"]):
            raise ValueError("The processor did not embed every declared image")
        return batch.to(device="cuda:0", dtype=torch.bfloat16)

    def evaluate(tag, rows):
        phase(tag + "_begin", records=len(rows))
        model.eval()
        predictions = []
        with torch.inference_mode():
            for i, row in enumerate(rows):
                inputs = prepare(row)
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
                selected = logits.index_select(0, tokens)
                probabilities = selected.softmax(-1)
                index = int(selected.argmax().item())
                raw_id = int(logits.argmax().item())
                predictions.append(
                    {
                        "record_id": row["record_id"],
                        "family_id": row["family_id"],
                        "control": row["control"],
                        "label": row.get("label"),
                        "prediction": classes[index],
                        "closed_scores": {
                            name: float(value)
                            for name, value in zip(classes, selected.tolist(), strict=True)
                        },
                        "closed_probabilities": {
                            name: float(value)
                            for name, value in zip(classes, probabilities.tolist(), strict=True)
                        },
                        "raw_next_token_id": raw_id,
                        "raw_next_token": processor.tokenizer.decode([raw_id]),
                        "raw_next_token_is_allowed": raw_id in token_ids,
                        "input_token_count": int(inputs["input_ids"].shape[1]),
                        "image_grid_thw": inputs["image_grid_thw"].tolist(),
                        "input_image_sha256": [ref["sha256"] for ref in row["images"]],
                    }
                )
                del inputs, logits, selected, probabilities
                if (i + 1) % 8 == 0:
                    phase(tag + "_progress", complete=i + 1)
        metrics = (
            classification_metrics(predictions, classes)
            if all(row.get("label") is not None for row in rows)
            else None
        )
        write_json(
            out / (tag + ".json"),
            {
                "scoring": "visual_prefix_single_token_closed_class_logits_not_generated_prose",
                "metrics": metrics,
                "predictions": predictions,
            },
        )
        phase(tag + "_complete", metrics=metrics)
        return predictions

    validation = [
        row
        for row in (dataset.get("evaluation_records") or dataset["records"])
        if row["record_id"] in recipe["evaluation_record_ids"]
    ]
    if not validation or len(validation) != len(recipe["evaluation_record_ids"]):
        raise ValueError("Evaluation IDs must bind exact nonempty frozen records")
    if recipe.get("external_evaluation"):
        if recipe["steps"] != 0 or any("label" in row for row in validation):
            raise ValueError("External evaluation accepts no training steps or sealed labels")
        if not recipe.get("inference_only"):
            evaluate("external_base", validation)
        from peft import PeftModel

        adapter = Path(recipe["adapter"])
        if sha(adapter / "adapter_model.safetensors") != recipe["adapter_sha256"]:
            raise ValueError("Evaluation adapter changed")
        if sha(adapter / "adapter_config.json") != recipe["adapter_config_sha256"]:
            raise ValueError("Evaluation adapter configuration changed")
        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=False, local_files_only=True
        )
        evaluate("external_adapted", validation)
        write_json(
            out / "inference_receipt.json",
            {
                "execution_success": True,
                "task": dataset["task"],
                "request_sha256": dataset["provenance"]["request_sha256"],
                "base": {
                    "repo": manifest["repo_id"],
                    "revision": manifest["revision"],
                    "model_sha256": manifest["files"]["model.safetensors"]["sha256"],
                },
                "adapter_sha256": recipe["adapter_sha256"],
                "adapter_config_sha256": recipe["adapter_config_sha256"],
                "recipe_sha256": sha(recipe_path),
                "dataset_sha256": recipe["dataset_sha256"],
                "code_sha256": recipe["code_sha256"],
                "prediction_file_sha256": sha(out / "external_adapted.json"),
                "quality_approved": False,
                "experimental_only": True,
            },
        )
        write_json(
            out / "result.json",
            {
                "status": "external_label_free_scoring_complete",
                "actual_training": False,
                "adapter_sha256": recipe["adapter_sha256"],
                "adapter_config_sha256": recipe["adapter_config_sha256"],
                "dataset_sha256": recipe["dataset_sha256"],
                "recipe_sha256": sha(recipe_path),
                "quality_approved": False,
            },
        )
        return
    expected_split = "external_evaluation" if recipe.get("label_free_evaluation") else "validation"
    if any(row["split"] != expected_split for row in validation) or (
        recipe.get("label_free_evaluation") and any("label" in row for row in validation)
    ):
        raise ValueError("Evaluation IDs must bind only frozen validation records")
    base_predictions = evaluate("base_validation", validation)
    if recipe["steps"] == 0:
        write_json(out / "result.json", {"status": "base_evaluated", "actual_training": False})
        return
    from peft import LoraConfig, get_peft_model

    phase("peft_imported")
    module_names = [
        name
        for name, _ in model.named_modules()
        if "language_model" in name and name.endswith(("q_proj", "v_proj"))
    ]
    if not module_names:
        raise ValueError("No language-only LoRA target modules found")
    config = LoraConfig(
        r=recipe["rank"],
        lora_alpha=recipe["alpha"],
        lora_dropout=recipe["dropout"],
        bias="none",
        target_modules=module_names,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name or "visual" in name for name, _ in trainable):
        raise RuntimeError("Only language LoRA tensors may be trained")
    initial = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=recipe["learning_rate"], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    training = [row for row in dataset["records"] if row["split"] == "train"]
    pools = {label: [row for row in training if row["label"] == label] for label in classes}
    pools = {label: rows for label, rows in pools.items() if rows}
    # Temper the 82:1 class skew without repeating the sole visual every ten microsteps.
    sampling_labels = sorted(pools)
    weights = [math.sqrt(len(pools[label])) for label in sampling_labels]
    rng = random.Random(recipe["seed"])
    sampling = recipe.get("sampling", "sqrt_class_frequency")
    purpose_cycle = list(sampling_labels)
    phase(
        "training_begin",
        trainable_parameters=sum(p.numel() for _, p in trainable),
        target_module_count=len(module_names),
        rank=recipe["rank"],
        learning_rate=recipe["learning_rate"],
        sampling="balanced_A_B_N_cycle" if dataset["task"] == "critic" else sampling,
        vision_frozen=True,
    )
    losses = []
    sampled = Counter()
    for step in range(recipe["steps"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for microstep in range(recipe["accumulation"]):
            if dataset["task"] == "critic":
                if set(pools) != {"A", "B", "N"}:
                    raise ValueError("A critic run requires all three train labels")
                label = sampling_labels[(step * recipe["accumulation"] + microstep) % 3]
            elif sampling == "balanced_class_cycle":
                position = (step * recipe["accumulation"] + microstep) % len(purpose_cycle)
                if position == 0:
                    rng.shuffle(purpose_cycle)
                label = purpose_cycle[position]
            else:
                label = rng.choices(sampling_labels, weights=weights, k=1)[0]
            row = rng.choice(pools[label])
            sampled[label] += 1
            inputs = prepare(row)
            logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            target = torch.tensor([token_ids[label_index[label]]], device="cuda:0")
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite answer-only loss")
            (loss / recipe["accumulation"]).backward()
            step_loss += float(loss.detach()) / recipe["accumulation"]
            del inputs, logits, target, loss
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite LoRA gradient")
        optimizer.step()
        scheduler.step()
        losses.append(step_loss)
        phase(
            "training_step",
            step=step + 1,
            loss=step_loss,
            gradient_norm=float(norm),
            peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20,
        )
        if (step + 1) % recipe["checkpoint_every"] == 0 or step + 1 == recipe["steps"]:
            checkpoint = out / ("checkpoint-" + str(step + 1))
            stage = staging_directory(checkpoint)
            model.save_pretrained(stage, safe_serialization=True)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "sampler": {
                        "rng_state": rng.getstate(),
                        "sampled_classes": dict(sampled),
                        "labels": sampling_labels,
                        "weights": weights,
                        "sampling": sampling,
                        "purpose_cycle": purpose_cycle,
                        "completed_microsteps": (step + 1) * recipe["accumulation"],
                    },
                    "completed_steps": step + 1,
                    "losses": losses,
                },
                stage / "training-state.pt",
            )
            write_json(
                stage / "progress.json",
                {
                    "completed_steps": step + 1,
                    "base_model": str(base),
                    "base_revision": manifest["revision"],
                    "purpose": dataset["task"],
                    "task_model_trained": True,
                    "resume_state_saved": True,
                    "resume_replay_validated": False,
                    "dataset_sha256": recipe["dataset_sha256"],
                    "recipe_sha256": sha(recipe_path),
                },
            )
            write_json(
                stage / "training_receipt.json",
                {
                    "actual_training": True,
                    "step": step + 1,
                    "losses": losses,
                    "dataset_sha256": recipe["dataset_sha256"],
                    "recipe_sha256": sha(recipe_path),
                    "base_revision": manifest["revision"],
                    "trainable_module_names": [name for name, _ in trainable],
                    "sampled_classes": dict(sampled),
                    "optimizer": "AdamW",
                    "answer_only_loss": True,
                    "vision_frozen": True,
                    "adapter_sha256": sha(stage / "adapter_model.safetensors"),
                    "adapter_config_sha256": sha(stage / "adapter_config.json"),
                    "optimizer_scheduler_rng_sampler_saved": True,
                },
            )
            publish(stage, checkpoint)
            phase("checkpoint_saved", step=step + 1, path=str(checkpoint))
    delta = (
        sum(
            float((p.detach().float().cpu() - initial[name]).square().sum())
            for name, p in trainable
        )
        ** 0.5
    )
    if not delta > 0 or not math.isfinite(delta):
        raise RuntimeError("The adapter did not learn a finite nonzero parameter delta")
    del initial, optimizer, scheduler
    gc.collect()
    adapted = evaluate("adapted_validation", validation)
    checkpoint = out / ("checkpoint-" + str(recipe["steps"]))
    # Reload the saved adapter into the resident base; avoid a second 2B model allocation.
    model.load_adapter(str(checkpoint), adapter_name="saved_reload", is_trainable=False)
    model.set_adapter("saved_reload")
    reloaded = evaluate("reload_probe", validation[:1])
    original_probe = adapted[0]
    reload_error = max(
        abs(reloaded[0]["closed_scores"][label] - original_probe["closed_scores"][label])
        for label in classes
    )
    if reload_error > 0.02:
        raise RuntimeError("Saved adapter reload changed the probe scores")
    result = {
        "status": "trained_and_reloaded",
        "actual_training": True,
        "adapter": str(checkpoint),
        "adapter_sha256": sha(checkpoint / "adapter_model.safetensors"),
        "adapter_config_sha256": sha(checkpoint / "adapter_config.json"),
        "parameter_delta_l2": delta,
        "base_metrics": classification_metrics(base_predictions, classes),
        "adapted_metrics": classification_metrics(adapted, classes),
        "reload_max_logit_error": reload_error,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "dataset_sha256": recipe["dataset_sha256"],
        "recipe_sha256": sha(recipe_path),
        "label_scope": dataset["provenance"],
        "task": dataset["task"],
        "quality_approved": False,
        "run_identity": digest(recipe),
    }
    write_json(out / "result.json", result)
    phase("complete", **result)


def classification_metrics(rows, classes):
    if any(row.get("label") is None for row in rows):
        return None
    correct = sum(row["label"] == row["prediction"] for row in rows)
    by_class = {}
    for label in classes:
        tp = sum(row["label"] == label and row["prediction"] == label for row in rows)
        actual = sum(row["label"] == label for row in rows)
        predicted = sum(row["prediction"] == label for row in rows)
        by_class[label] = {
            "n": actual,
            "predicted": predicted,
            "recall": tp / actual if actual else None,
            "f1": 2 * tp / (actual + predicted) if actual + predicted else None,
        }
    available = [item["f1"] for item in by_class.values() if item["n"]]
    controls = {
        control: {
            "n": len(subset),
            "accuracy": sum(r["label"] == r["prediction"] for r in subset) / len(subset),
        }
        for control in sorted({r["control"] for r in rows})
        if (subset := [r for r in rows if r["control"] == control])
    }
    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": sum(available) / len(available) if available else None,
        "per_class": by_class,
        "prediction_histogram": dict(Counter(r["prediction"] for r in rows)),
        "controls": controls,
        "raw_allowed_rate": sum(r["raw_next_token_is_allowed"] for r in rows) / len(rows)
        if rows
        else None,
    }


if __name__ == "__main__":
    if (
        os.environ.get("VISUAL_ADAPTER_REQUIRE_OWNED_JOB") == "1"
        and sys.stdin.readline().strip() != "owned-job-assigned"
    ):
        raise RuntimeError("Model execution requires its assigned supervisor job")
    main(sys.argv[1])
