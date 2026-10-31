"""Compare saved FP32 LoRA tensors with cold, second-adapter and pre-cast loads."""

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from .data import read_json, sha, write_json


def main(config_path):
    config = read_json(config_path)
    out, old = Path(config["output"]), Path(config["source_run"])
    started = time.monotonic()

    def phase(name, **values):
        row = {"phase": name, "seconds": time.monotonic() - started, **values}
        with (out / "phases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    for name, expected in config["code_sha256"].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError("Diagnostic code changed")
    checkpoint = Path(config["checkpoint"])
    for relative, expected in config["source_sha256"].items():
        if sha(old / relative) != expected:
            raise ValueError("Frozen source run changed: " + relative)
    phase("imports_begin")
    import torch
    from peft import PeftConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Diagnostic requires the approved BF16 CUDA device")
    phase(
        "imports_complete",
        python=sys.version,
        versions={p: importlib.metadata.version(p) for p in ("torch", "peft", "transformers")},
    )
    recipe = read_json(old / "recipe.json")
    base = Path(recipe["base"])
    identity_path = Path(recipe["base_manifest"])
    if not identity_path.is_absolute():
        identity_path = old / identity_path
    identity = read_json(identity_path)
    if sha(identity_path) != recipe["base_manifest_sha256"]:
        raise ValueError("Frozen base identity changed")
    for name, expected in identity["files"].items():
        if sha(base / name) != expected["sha256"]:
            raise ValueError("Pinned base changed")
    data_path = Path(recipe["dataset"])
    if not data_path.is_absolute():
        data_path = old / data_path
    if sha(data_path) != recipe["dataset_sha256"]:
        raise ValueError("Frozen training input changed")
    dataset = read_json(data_path)
    old_adapted = read_json(old / "adapted_validation.json")["predictions"][0]
    old_reload = read_json(old / "reload_probe.json")["predictions"][0]
    row = next(r for r in dataset["records"] if r["record_id"] == old_adapted["record_id"])
    processor = AutoProcessor.from_pretrained(
        str(base),
        local_files_only=True,
        trust_remote_code=False,
        min_pixels=recipe["min_pixels"],
        max_pixels=recipe["max_pixels"],
    )
    pictures = []
    for ref in row["images"]:
        path = Path(ref["path"])
        if not path.is_absolute():
            path = data_path.parent / path
        if sha(path) != ref["sha256"]:
            raise ValueError("Probe image changed")
        with Image.open(path) as picture:
            pictures.append(picture.convert("RGB"))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in pictures]
            + [{"type": "text", "text": row["prompt"]}],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=pictures, return_tensors="pt", padding=False)
    for picture in pictures:
        picture.close()
    classes = dataset["classes"]
    token_ids = [
        processor.tokenizer.encode(label, add_special_tokens=False)[0] for label in classes
    ]
    inputs = inputs.to(device="cuda:0", dtype=torch.bfloat16)
    phase(
        "load_begin",
        input_token_count=int(inputs["input_ids"].shape[1]),
        image_grid_thw=inputs["image_grid_thw"].tolist(),
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint),
        adapter_name="default",
        is_trainable=False,
        autocast_adapter_dtype=True,
        local_files_only=True,
    )
    expected = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    result = {
        "config_sha256": sha(config_path),
        "source_sha256": config["source_sha256"],
        "record_id": row["record_id"],
        "input_image_sha256": [ref["sha256"] for ref in row["images"]],
        "old_adapted": old_adapted["closed_scores"],
        "old_reload": old_reload["closed_scores"],
        "stages": {},
        "quality_approved": False,
    }

    def audit_and_score(name):
        model.set_adapter(name)
        model.eval()
        current = get_peft_model_state_dict(model, adapter_name=name, save_embedding_layers=False)
        if set(current) != set(expected):
            raise ValueError("Loaded adapter keys differ from the saved file")
        tensors = {}
        for key, value in current.items():
            actual = value.detach().float().cpu()
            saved = expected[key].float()
            rounded = saved.to(torch.bfloat16).float()
            tensors[key] = {
                "live_dtype": str(value.dtype),
                "saved_dtype": str(expected[key].dtype),
                "exact_saved": torch.equal(actual, saved),
                "exact_bf16_roundtrip": torch.equal(actual, rounded),
                "max_abs_saved": float((actual - saved).abs().max()),
                "max_abs_bf16_roundtrip": float((actual - rounded).abs().max()),
            }
        repeated = []
        with torch.inference_mode():
            for _ in range(2):
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
                repeated.append(
                    {
                        label: float(logits[token])
                        for label, token in zip(classes, token_ids, strict=True)
                    }
                )
        scales = {
            key: {
                "scale": float(module.scaling[name]),
                "A_dtype": str(module.lora_A[name].weight.dtype),
                "B_dtype": str(module.lora_B[name].weight.dtype),
                "active": list(module.active_adapters),
            }
            for key, module in model.named_modules()
            if hasattr(module, "lora_A") and name in module.lora_A
        }
        stage = {
            "tensors": tensors,
            "layer_status": scales,
            "repeated_scores": repeated,
            "all_parameters_exact_saved": all(v["exact_saved"] for v in tensors.values()),
            "all_parameters_exact_bf16_roundtrip": all(
                v["exact_bf16_roundtrip"] for v in tensors.values()
            ),
            "max_abs_parameter_error": max(v["max_abs_saved"] for v in tensors.values()),
            "max_abs_repeat_score_error": max(
                abs(repeated[0][k] - repeated[1][k]) for k in classes
            ),
            "difference_from_original_adapted": {
                k: repeated[0][k] - result["old_adapted"][k] for k in classes
            },
            "difference_from_original_reload": {
                k: repeated[0][k] - result["old_reload"][k] for k in classes
            },
        }
        result["stages"][name] = stage
        write_json(out / "diagnostic.json", result)
        phase(
            "stage_complete",
            adapter=name,
            exact_saved=stage["all_parameters_exact_saved"],
            exact_bf16=stage["all_parameters_exact_bf16_roundtrip"],
            max_parameter_error=stage["max_abs_parameter_error"],
            repeat_error=stage["max_abs_repeat_score_error"],
            original_score_error=max(
                abs(v) for v in stage["difference_from_original_adapted"].values()
            ),
        )

    audit_and_score("default")
    model.load_adapter(
        str(checkpoint),
        adapter_name="second",
        is_trainable=False,
        autocast_adapter_dtype=True,
        local_files_only=True,
    )
    audit_and_score("second")
    precast_config = PeftConfig.from_pretrained(str(checkpoint), local_files_only=True)
    model.add_adapter("precast", precast_config)
    # Explicitly upcast BEFORE FP32 file values are copied, rather than after a BF16 copy.
    model.base_model._cast_adapter_dtype(adapter_name="precast", autocast_adapter_dtype=True)
    model.load_adapter(
        str(checkpoint),
        adapter_name="precast",
        is_trainable=False,
        autocast_adapter_dtype=True,
        local_files_only=True,
    )
    audit_and_score("precast")
    result["status"] = "diagnostic_complete"
    write_json(out / "diagnostic.json", result)
    phase("complete")


if __name__ == "__main__":
    if (
        os.environ.get("VISUAL_ADAPTER_REQUIRE_OWNED_JOB") == "1"
        and sys.stdin.readline().strip() != "owned-job-assigned"
    ):
        raise RuntimeError("Diagnostic requires its owned process lifecycle")
    main(sys.argv[1])
