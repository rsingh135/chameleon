"""Dedicated CPU VLM process. Invoked only by the bounded provider supervisor."""

from __future__ import annotations

import argparse
import importlib.metadata
import time
from pathlib import Path

from .download_model import FILES, MODELS
from .provider import (
    LETTERS,
    bind_decision,
    canonical,
    digest,
    parse_generated,
    prepare_request,
    prompt,
    read_json,
    sha256,
)


def whole_view(path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    fitted = ImageOps.contain(rgb, (512, 512), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 512), "white")
    canvas.paste(fitted, ((512 - fitted.width) // 2, (512 - fitted.height) // 2))
    return canvas


def interpret_generation(raw, prepared):
    """Invalid descriptions fail instead of entering the letter-scoring path."""
    interpreted = parse_generated(raw, prepared)
    if interpreted is None and prepared.get("mode") == "analyze":
        raise ValueError("VLM source analysis was invalid; no choice fallback is allowed")
    return interpreted


def run(directory):
    started = time.monotonic()
    directory = Path(directory)

    def event(name):
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                canonical(
                    {
                        "event": name,
                        "seconds": time.monotonic() - started,
                        "cpu_seconds": time.process_time(),
                    }
                )
                + "\n"
            )

    event("worker_started")
    request, config = read_json(directory / "request.json"), read_json(directory / "config.json")
    prepared = prepare_request(request)
    analyze = prepared.get("mode") == "analyze"
    if analyze and config["strategy"] == "choice_logits":
        raise ValueError("Source analysis requires generation, never menu-choice scoring")
    model_dir = Path(config["model_dir"])
    manifest = read_json(model_dir / "verified-model.json")
    spec = next((s for s in MODELS.values() if s["revision"] == manifest.get("revision")), None)
    if spec is None or manifest.get("weight_sha256") != spec["weight_sha256"]:
        raise ValueError("Unrecognized pinned VLM identity")
    for filename in FILES:
        if sha256(model_dir / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError("VLM model or processor file bytes changed")
    event("model_files_verified")
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    event("runtime_imported")
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    model = (
        AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        .to("cpu")
        .eval()
    )
    event("model_loaded")
    load_seconds = time.monotonic() - started
    images = [whole_view(i["path"]) for i in prepared["images"]]
    question = prompt(prepared)
    scoring_question = (
        question.split("\nReply with JSON only:", 1)[0]
        + "\nReply using ONLY the single letter of the best choice."
    )
    if config["strategy"] == "choice_logits":
        question = scoring_question
    messages = [
        {
            "role": "user",
            "content": [*({"type": "image"} for _ in images), {"type": "text", "text": question}],
        }
    ]
    rendered_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=rendered_prompt, images=images, return_tensors="pt")
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            inputs[key] = value.to(dtype=dtype)
    input_tokens = int(inputs["input_ids"].shape[-1])
    if input_tokens > 4096:
        raise ValueError("Visual input exceeded its context budget")
    event("inputs_ready")
    raw, interpreted, scores = "", None, {}
    with torch.inference_mode():
        if config["strategy"] == "generate_then_score":
            event("generation_started")
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=config["max_new_tokens"]
            )
            raw = processor.batch_decode(output[:, input_tokens:], skip_special_tokens=True)[0]
            event("generation_finished")
            interpreted = interpret_generation(raw, prepared)
        if interpreted is None:
            if analyze:
                raise ValueError("VLM source analysis was invalid; no choice fallback is allowed")
            # This is real model conditional scoring, not an algorithmic choice.
            # One finite letter token is evaluated for each already-known plan.
            if config["strategy"] != "choice_logits":
                messages[0]["content"][-1]["text"] = scoring_question
                rendered_prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = processor(text=rendered_prompt, images=images, return_tensors="pt")
                for key, value in inputs.items():
                    if torch.is_floating_point(value):
                        inputs[key] = value.to(dtype=dtype)
            event("choice_scoring_started")
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            event("choice_scoring_finished")
            values = []
            for letter in LETTERS[: len(prepared["plans"])]:
                tokens = processor.tokenizer.encode(letter, add_special_tokens=False)
                if len(tokens) != 1:
                    raise ValueError("Conditional letter scoring requires single-token choices")
                score = float(torch.log_softmax(logits, dim=-1)[tokens[0]])
                scores[letter] = score
                values.append(score)
            best = max(range(len(values)), key=values.__getitem__)
            letter = LETTERS[best]
            interpreted = {
                "decision": bind_decision(
                    prepared,
                    letter,
                    "Selected by constrained VLM next-token scores; no reliable prose was parsed.",
                ),
                "observations": "",
                "choice": letter,
            }
            method = "conditional_letter_log_likelihood"
        else:
            method = "generated_json"
    prepare_request(request)
    response = {
        "analysis" if analyze else "decision": interpreted["analysis" if analyze else "decision"],
        "receipt": {
            "schema_version": 1,
            "executed": True,
            "mode": prepared.get("mode", "plan"),
            "backend": "local-generative-vlm",
            "model_id": spec["repo_id"],
            "model_revision": spec["revision"],
            "weight_sha256": spec["weight_sha256"],
            "model_manifest_digest": digest(manifest),
            "provider_sha256": sha256(Path(__file__).with_name("provider.py")),
            "worker_sha256": sha256(__file__),
            "request_digest": prepared["request_digest"],
            "source_analysis_digest": digest(prepared["source_analysis"])
            if prepared.get("source_analysis") is not None
            else None,
            "source_digest": prepared["menu"]["source_digest"],
            "source_render_sha256": prepared["images"][0]["sha256"],
            "menu_digest": prepared["menu"]["menu_digest"],
            "target_canvas": prepared["target"],
            "images": prepared["images"],
            "input_image_count": len(prepared["images"]),
            "input_tokens": input_tokens,
            "raw_output": raw,
            "visual_observations": interpreted["observations"],
            "choice_letter": interpreted.get("choice"),
            "decision_method": method,
            "choice_log_likelihoods": scores,
            "prompt_sha256": digest(rendered_prompt),
            "model_load_seconds": load_seconds,
            "worker_seconds": time.monotonic() - started,
            "dtype": config["dtype"],
            "device": "cpu",
            "threads": config["threads"],
            "view_policy": "full-image proportional white letterbox 512; no crop or splitting",
            "packages": {
                k: importlib.metadata.version(k) for k in ("torch", "transformers", "Pillow")
            },
            "quality_validated": False,
            "regions_predicted": False,
            "crop_authority": False,
        },
    }
    (directory / "response.json").write_text(canonical(response) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.directory)


if __name__ == "__main__":
    main()
