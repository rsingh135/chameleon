"""Validate image-conditioned LoRA forward/backward/save/reload on a CUDA GPU.

Tiny synthetic image examples exercise infrastructure only. They are not
Chameleon preference labels, a quality benchmark, or a task-trained critic.
Use an external billing deadline and a shell timeout in addition to this script.
Public base weights are downloaded to HF_HOME; credentials are not required.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc  # noqa: UP017 -- remote VM support includes Python 3.10.
_checkpoint_io = importlib.import_module(
    "tools.checkpoint_manifest" if __package__ else "checkpoint_manifest"
)
CHECKPOINT_MANIFEST = _checkpoint_io.CHECKPOINT_MANIFEST
REQUIRED_CHECKPOINT_FILES = _checkpoint_io.REQUIRED_CHECKPOINT_FILES
write_json = _checkpoint_io.write_json
checkpoint_files = _checkpoint_io.checkpoint_files
verify_checkpoint = _checkpoint_io.verify_checkpoint
publish_checkpoint = _checkpoint_io.publish_checkpoint


def packages() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "Pillow")
    }


def validate_reload(
    expected_digest: str, actual_digest: str, expected_loss: float, actual_loss: float
) -> None:
    if expected_digest != actual_digest:
        raise RuntimeError("Reloaded adapter digest differs from trained adapter")
    if not all(math.isfinite(value) for value in (expected_loss, actual_loss)):
        raise RuntimeError("Reloaded image-conditioned loss must be finite")
    if abs(expected_loss - actual_loss) > 0.05:
        raise RuntimeError("Reloaded image-conditioned loss differs beyond BF16 tolerance")


def adapter_digest(model) -> str:
    import torch
    from peft import get_peft_model_state_dict

    digest = hashlib.sha256()
    for name, tensor in sorted(get_peft_model_state_dict(model).items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(
            tensor.detach().cpu().contiguous().view(-1).view(dtype=torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def synthetic_example(run: Path, scenario: str) -> dict:
    """Build inspectable local fixtures without any model or torch import."""
    from PIL import Image, ImageDraw

    if scenario == "single-image":
        image = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(image).rectangle((64, 64, 192, 192), fill="red")
        images = [image]
        names = ["synthetic-input.png"]
        prompt = "What color is the square? Answer with one word."
        response = "red"
    elif scenario == "source-ab":
        source = Image.new("RGB", (256, 320), (245, 241, 230))
        drawing = ImageDraw.Draw(source)
        drawing.rectangle((24, 24, 232, 56), fill=(24, 42, 68))
        drawing.rectangle((56, 88, 200, 240), fill=(194, 58, 48))
        drawing.rectangle((24, 272, 232, 292), fill=(24, 42, 68))
        candidate_a = source.resize((192, 240))
        candidate_b = Image.new("RGB", (192, 240), (245, 241, 230))
        drawing_b = ImageDraw.Draw(candidate_b)
        drawing_b.rectangle((4, 180, 100, 236), fill=(56, 100, 190))
        drawing_b.rectangle((100, 16, 190, 48), fill=(24, 42, 68))
        images = [source, candidate_a, candidate_b]
        names = ["synthetic-source.png", "synthetic-A.png", "synthetic-B.png"]
        prompt = (
            "Images are ordered source, candidate A, candidate B. Which resized candidate "
            "better preserves the source colors, visual hierarchy, and layout? Return only "
            "JSON with choice (A, B, tie, or neither) and issues: []. "
            "This is a synthetic infrastructure example."
        )
        response = '{"choice":"A","issues":[]}'
    else:
        raise ValueError("Unknown synthetic compatibility scenario")
    paths = []
    for image, name in zip(images, names, strict=True):
        path = run / name
        image.save(path)
        paths.append(str(path))
    example = {
        "scenario": scenario,
        "images": paths,
        "prompt": prompt,
        "response": response,
        "image_count": len(images),
        "task_model_trained": False,
        "purpose": "synthetic_compatibility_only",
    }
    write_json(run / "synthetic-example.json", example)
    return example


def strict_synthetic_verdict(raw_response: str) -> dict:
    from experiments.vision_critic.train_sft import parse_model_verdict

    verdict = parse_model_verdict(raw_response, {"A": [], "B": []})
    malformed = parse_model_verdict('{"choice":["A"],"issues":[]}', {"A": [], "B": []})
    return {
        **verdict,
        "raw_response": raw_response,
        "matches_synthetic_target": verdict["valid"] and verdict["choice"] == "A",
        "fallback_required": not verdict["valid"] or verdict["choice"] in {"tie", "neither"},
        "malformed_output_rejected": not malformed["valid"],
        "quality_claim": "One synthetic example; this is not design-quality evaluation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--revision", default="main", help="Resolved and recorded as an exact commit"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout-minutes", type=int, default=20)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--scenario", choices=("single-image", "source-ab"), default="single-image")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write synthetic fixtures only; no model imports/download/GPU",
    )
    args = parser.parse_args()
    if not 1 <= args.timeout_minutes <= 30:
        parser.error("--timeout-minutes must be between 1 and 30")
    if args.local_files_only and not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("Offline runs require an exact 40-hex base-model commit in --revision")
    if args.prepare_only:
        run = args.out_dir or Path("training_runs") / (
            "prepared-compatibility-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        run.mkdir(parents=True, exist_ok=False)
        example = synthetic_example(run, args.scenario)
        report = {
            **example,
            "fixture_prepared": True,
            "gpu_compatibility_validated": False,
            "scope": "Fixture preparation only; multimodal tokenizer/mask still unverified",
        }
        write_json(run / "prepared-report.json", report)
        print(json.dumps(report, indent=2))
        return 0
    if hasattr(signal, "SIGALRM"):

        def expired(_signal, _frame):
            raise TimeoutError(
                "Smoke wall-time limit reached; external billing guard is still required"
            )

        signal.signal(signal.SIGALRM, expired)
        signal.alarm(args.timeout_minutes * 60)

    import torch
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This smoke requires CUDA and BF16 support")
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError("Need at least 16 GiB free VRAM for this bounded smoke")
    random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    revision = (
        args.revision
        if args.local_files_only
        else HfApi(token=False).model_info(args.model, revision=args.revision, token=False).sha
    )
    run = args.out_dir or Path("training_runs") / (
        "vision-smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    run.mkdir(parents=True, exist_ok=False)
    published = run / "published"
    published.mkdir()
    example = synthetic_example(run, args.scenario)
    report = {
        "vision_lora_smoke_passed": False,
        "task_model_trained": False,
        "scope": "synthetic infrastructure check only; no quality or task-learning claim",
        "base_model": args.model,
        "base_revision": revision,
        "packages": packages(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "steps": args.steps,
        "scenario": args.scenario,
        "image_count": example["image_count"],
        "losses": [],
    }
    write_json(run / "smoke-report.json", report)

    def load_base():
        return Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            revision=revision,
            token=False,
            trust_remote_code=False,
            local_files_only=args.local_files_only,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=revision,
        token=False,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    images = []
    for path in example["images"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in images],
                {"type": "text", "text": example["prompt"]},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        [
            *messages,
            {"role": "assistant", "content": [{"type": "text", "text": example["response"]}]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(text=[full], images=images, return_tensors="pt")
    prefix = processor(text=[prompt], images=images, return_tensors="pt")
    if inputs["image_grid_thw"].shape[0] != len(images):
        raise RuntimeError("Processor did not preserve the expected number of images")
    prefix_length = prefix["input_ids"].shape[1]
    if not torch.equal(inputs["input_ids"][:, :prefix_length], prefix["input_ids"]):
        raise RuntimeError(
            "Chat-template prefix differs; refusing an incorrect assistant label mask"
        )
    max_tokens = 2048 if args.scenario == "source-ab" else 512
    if inputs["input_ids"].shape[1] > max_tokens:
        raise RuntimeError("Smoke sequence exceeds the token budget; no truncation is permitted")
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_length] = -100
    if not bool((labels != -100).any()):
        raise RuntimeError("No assistant tokens available for loss")
    report.update(
        {
            "input_tokens": inputs["input_ids"].shape[1],
            "assistant_loss_tokens": int((labels != -100).sum()),
            "prefix_tokens": prefix_length,
            "assistant_prefix_mask_verified": True,
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
        }
    )
    inputs.pop("token_type_ids", None)
    inputs = {name: value.to("cuda") for name, value in inputs.items()}
    inputs["labels"] = labels.to("cuda")
    del prefix

    model = load_base()
    targets = [
        name
        for name, layer in model.named_modules()
        if ".language_model." in name
        and isinstance(layer, torch.nn.Linear)
        and name.endswith((".q_proj", ".v_proj"))
    ]
    if not targets:
        raise RuntimeError("No language-only q/v attention projections found")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=targets,
            bias="none",
        ),
    )
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        ".visual." in name or "merger" in name or "lora_" not in name for name, _ in trainable
    ):
        raise RuntimeError("Unexpected trainable parameters: vision and projector must stay frozen")
    report["trainable_parameters"] = sum(parameter.numel() for _, parameter in trainable)
    report["target_modules"] = targets
    before = adapter_digest(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-4)
    model.config.use_cache = False
    model.train()
    final_checkpoint = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, use_cache=False).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite image-conditioned loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0:
            raise RuntimeError("Adapter gradients are non-finite or zero")
        optimizer.step()
        report["losses"].append(float(loss.detach()))
        report["last_gradient_norm"] = float(grad_norm)
        stage = run / f".checkpoint-{step:06d}.partial"
        stage.mkdir()
        model.save_pretrained(stage, safe_serialization=True)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "completed_steps": step,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "base_revision": revision,
                "packages": report["packages"],
                "script_sha256": report["script_sha256"],
            },
            stage / "training-state.pt",
        )
        write_json(
            stage / "progress.json",
            {
                "completed_steps": step,
                "base_model": args.model,
                "base_revision": revision,
                "task_model_trained": False,
                "loss": report["losses"][-1],
                "resume_scope": (
                    "adapter + AdamW + RNG; requires exact base revision and environment"
                ),
            },
        )
        final_checkpoint = published / f"checkpoint-{step:06d}"
        publish_checkpoint(stage, final_checkpoint)
        write_json(run / "smoke-report.json", report)
        print(
            json.dumps(
                {
                    "step": step,
                    "loss": report["losses"][-1],
                    "published_checkpoint": str(final_checkpoint),
                }
            ),
            flush=True,
        )
    after = adapter_digest(model)
    if before == after:
        raise RuntimeError("LoRA weights did not change after optimizer steps")
    model.eval()
    with torch.inference_mode():
        expected_loss = float(model(**inputs, use_cache=False).loss)
    del optimizer, trainable, model, loss, grad_norm
    gc.collect()
    torch.cuda.empty_cache()
    verify_checkpoint(final_checkpoint)
    restored = PeftModel.from_pretrained(load_base(), final_checkpoint, is_trainable=True)
    reloaded_digest = adapter_digest(restored)
    restored.eval()
    with torch.inference_mode():
        actual_loss = float(restored(**inputs, use_cache=False).loss)
    validate_reload(after, reloaded_digest, expected_loss, actual_loss)
    state = torch.load(
        final_checkpoint / "training-state.pt", map_location="cpu", weights_only=True
    )
    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in restored.parameters() if parameter.requires_grad], lr=1e-4
    )
    resumed_optimizer.load_state_dict(state["optimizer"])
    if state["completed_steps"] != args.steps or not resumed_optimizer.state:
        raise RuntimeError("Optimizer checkpoint could not be restored")
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    restored.train()
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss = restored(**inputs, use_cache=False).loss
    if not bool(torch.isfinite(resumed_loss)):
        raise RuntimeError("Resumed optimization produced a non-finite loss")
    resumed_loss.backward()
    resumed_parameters = [
        parameter for parameter in restored.parameters() if parameter.requires_grad
    ]
    resumed_grad_norm = torch.nn.utils.clip_grad_norm_(resumed_parameters, 1.0)
    if not bool(torch.isfinite(resumed_grad_norm)) or float(resumed_grad_norm) <= 0:
        raise RuntimeError("Resumed adapter gradients are non-finite or zero")
    resumed_optimizer.step()
    resumed_digest = adapter_digest(restored)
    if resumed_digest == reloaded_digest:
        raise RuntimeError("Resumed optimizer step failed to update adapter weights")
    if args.scenario == "source-ab":
        resumed_step = args.steps + 1
        stage = run / f".checkpoint-{resumed_step:06d}.partial"
        stage.mkdir()
        restored.save_pretrained(stage, safe_serialization=True)
        torch.save(
            {
                "optimizer": resumed_optimizer.state_dict(),
                "completed_steps": resumed_step,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "base_revision": revision,
                "packages": report["packages"],
                "script_sha256": report["script_sha256"],
            },
            stage / "training-state.pt",
        )
        write_json(
            stage / "progress.json",
            {
                "completed_steps": resumed_step,
                "base_model": args.model,
                "base_revision": revision,
                "task_model_trained": False,
                "resume_validation_checkpoint": True,
                "loss": float(resumed_loss.detach()),
                "purpose": "synthetic_compatibility_only",
            },
        )
        final_checkpoint = published / f"checkpoint-{resumed_step:06d}"
        publish_checkpoint(stage, final_checkpoint)
        verify_checkpoint(final_checkpoint)
        restored.eval()
        generation = processor(text=[prompt], images=images, return_tensors="pt")
        generation.pop("token_type_ids", None)
        generation = {name: value.to("cuda") for name, value in generation.items()}
        with torch.inference_mode():
            generated = restored.generate(
                **generation, max_new_tokens=96, do_sample=False, use_cache=True
            )
        text = processor.batch_decode(
            generated[:, generation["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        report["strict_output_evaluation"] = strict_synthetic_verdict(text)
    report.update(
        {
            "vision_lora_smoke_passed": True,
            "adapter_sha256": resumed_digest if args.scenario == "source-ab" else after,
            "pre_resume_adapter_sha256": after,
            "reloaded_loss": actual_loss,
            "pre_reload_loss": expected_loss,
            "optimizer_restore_passed": True,
            "resumed_optimizer_step_passed": True,
            "resume_validation_steps": 1,
            "total_completed_optimizer_steps": args.steps + 1,
            "resumed_loss": float(resumed_loss.detach()),
            "resumed_adapter_sha256": resumed_digest,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "final_checkpoint": str(final_checkpoint),
        }
    )
    write_json(run / "smoke-report.json", report)
    write_json(published / "smoke-report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
