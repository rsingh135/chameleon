"""Pinned project-local llama.cpp CPU/CUDA backend; no runtime fetching."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from .provider import LETTERS, canonical, digest, parse_generated, prompt, sha256
from .worker import whole_view

REVISION = "72e986006ef53e37cdd3f6d4241c90b0f01df376"
RUNTIME_TAG = "b12017"
RUNTIME_SHA = "e89d351a603e91a9e3e944c4c4c4ccf5b4af6acbd44c5434b61bacfa3999b1f6"
CUDA_ARCHIVES = {
    "llama-b12017-bin-win-cuda-12.4-x64.zip": {
        "sha256": "294eb8f846bdcd21a7b59b8483fac56fc44f14f7d7580107022a911ee76fab48",
        "bytes": 279_943_357,
        "license": "MIT and bundled component notices",
    },
    "cudart-llama-bin-win-cuda-12.4-x64.zip": {
        "sha256": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        "bytes": 391_443_627,
        "license": "NVIDIA CUDA Toolkit redistribution terms",
    },
}
MODEL = "SmolVLM-500M-Instruct-Q8_0.gguf"
PROJECTOR = "mmproj-SmolVLM-500M-Instruct-f16.gguf"
PINNED = {
    MODEL: "9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4",
    PROJECTOR: "4b2fca922e549e9b9608f4ba56f93565dd5220e935ba5570a2a864a0fcc988a6",
}
PROFILE_IDS = ("smolvlm_500m", "qwen35_08b")
QWEN_REVISION = "f36b1ea49a332ede8fe5f389bbf5b3575ef71f48"
QWEN_MODEL = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
QWEN_PROJECTOR = "mmproj-Qwen_Qwen3.5-0.8B-f16.gguf"
QWEN_PINNED = {
    QWEN_MODEL: "fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526",
    QWEN_PROJECTOR: "1dc1351c82e41b48edb55fd6ddfa7ca60fb5a16b3d5abf3ce7054880dd022847",
}
# The original Qwen template's single-turn, no-tools, enable_thinking=False
# branch. Media expansion remains the pinned mtmd implementation's job.
QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
)


def runtime_profile(config):
    device = config.get("native_device", "cpu")
    if device == "cpu":
        return {
            "device": device,
            "directory": f"runtime-{RUNTIME_TAG}",
            "archives": {f"llama-{RUNTIME_TAG}-bin-win-cpu-x64.zip": RUNTIME_SHA},
        }
    if device == "cuda":
        return {
            "device": device,
            "directory": f"runtime-{RUNTIME_TAG}-cuda-12.4",
            "archives": {name: value["sha256"] for name, value in CUDA_ARCHIVES.items()},
        }
    raise ValueError("Native device must explicitly be cpu or cuda")


def archive_member_sha256(zipped, member):
    value = hashlib.sha256()
    with zipped.open(member) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cuda_proof(stderr, config):
    """CUDA selection alone is insufficient: require actual language and vision offload."""
    devices = re.findall(r"Device\s+(\d+):\s*([^,\r\n]+),\s*compute capability", stderr)
    selected = next((name.strip() for index, name in devices if index == "0"), None)
    if selected is None:
        actual = re.search(r"using device CUDA0 \(([^)\r\n]+)\)", stderr)
        selected = actual.group(1).strip() if actual else None
    layers = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", stderr)
    vision = [
        line
        for line in stderr.splitlines()
        if "CUDA0" in line
        and re.search(r"clip|vision|mtmd", line, re.IGNORECASE)
        and re.search(r"using|backend|buffer", line, re.IGNORECASE)
    ]
    if not selected or not layers or not vision:
        raise ValueError("Requested CUDA execution lacks device, layer, or vision-backend proof")
    offloaded, total = map(int, layers[-1])
    if not total or offloaded != total:
        raise ValueError("Requested CUDA all-layer offload fell back partially to CPU")
    expected = config.get("native_expected_cuda_device")
    if expected is not None and selected != expected:
        raise ValueError("Native CUDA selected a different device from the operator config")
    return {
        "cuda_execution_verified": True,
        "cuda_device": {"index": 0, "backend": "CUDA0", "name": selected},
        "gpu_layers": offloaded,
        "gpu_layers_total": total,
        "image_encoder_backend": "CUDA0",
        "projector_offloaded": True,
        "cuda_log_excerpt": [
            line
            for line in stderr.splitlines()
            if line in vision or "layers to GPU" in line or "Device 0:" in line
        ],
    }


def profile(config):
    selected = config.get("native_profile", "smolvlm_500m")
    if selected == "smolvlm_500m":
        return {
            "name": selected,
            "repo_id": "ggml-org/SmolVLM-500M-Instruct-GGUF",
            "base_model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
            "revision": REVISION,
            "model": MODEL,
            "projector": PROJECTOR,
            "weights": PINNED,
            "quantization": {"language": "Q8_0", "vision_projector": "F16"},
            "quantizer": "ggml-org",
        }
    if selected == "qwen35_08b":
        return {
            "name": selected,
            "repo_id": "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
            "base_model_id": "Qwen/Qwen3.5-0.8B",
            "revision": QWEN_REVISION,
            "model": QWEN_MODEL,
            "projector": QWEN_PROJECTOR,
            "weights": QWEN_PINNED,
            "quantization": {"language": "Q4_K_M", "vision_projector": "F16"},
            "quantizer": "bartowski",
        }
    raise ValueError("Unknown pinned native VLM profile")


def prepare(prepared, config, directory):
    specification = profile(config)
    selected_runtime = runtime_profile(config)
    root = Path(config["model_dir"]).resolve()
    runtime = root / selected_runtime["directory"]
    for name, expected in specification["weights"].items():
        if sha256(root / name) != expected:
            raise ValueError("Native VLM/projector differs from the official pinned weights")
    # Verify extracted executable/DLLs against the authenticated archive, not a
    # self-reported mutable sidecar. No archive content is executed at this step.
    binary_hashes = {}
    for name, expected_zip_sha in selected_runtime["archives"].items():
        archive = root / name
        if sha256(archive) != expected_zip_sha:
            raise ValueError("Native runtime ZIP differs from the official pinned release")
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                if item.is_dir():
                    continue
                path = (runtime / item.filename).resolve()
                if not path.is_relative_to(runtime.resolve()) or not path.is_file():
                    raise ValueError("Native runtime file leaves its pinned directory")
                expected = archive_member_sha256(zipped, item)
                if sha256(path) != expected:
                    raise ValueError("Native executable or DLL differs from the pinned archive")
                binary_hashes[item.filename] = expected
    views = []
    for i, image in enumerate(prepared["images"]):
        path = directory / f"view-{i}.png"
        whole_view(image["path"]).save(path)
        views.append({"path": str(path), "sha256": sha256(path), "purpose": image["purpose"]})
    # b12017 interleaves each marker with the corresponding comma-separated
    # image. Explicit labels beside the markers keep the three-view order clear.
    media = "\n".join(f"{view['purpose']}: <__media__>" for view in views)
    question = media + "\n" + prompt(prepared).split("\nReply with JSON only:", 1)[0]
    analyze = prepared.get("mode") == "analyze"
    if not analyze:
        question += (
            "\nReply with a JSON object containing choice, observations and reason. "
            "Use the best option letter. Each prose field must use at most eight words."
        )
    evidence_first = not analyze and config.get("native_evidence_first", False)
    if evidence_first:
        question += (
            " First describe visible subjects and image coverage in observations, "
            "then give a short design reason, then choose. "
            "Use direct noun phrases, at most five words per prose field."
        )
    schema = {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": list(LETTERS[: len(prepared["plans"])])},
            "observations": {"type": "string", "maxLength": 64},
            "reason": {"type": "string", "maxLength": 64},
        },
        "required": ["choice", "observations", "reason"],
        "additionalProperties": False,
    }
    if analyze:
        from engine.visual_analysis import analysis_json_schema

        schema = analysis_json_schema(prepared["known_element_ids"])
    if evidence_first:
        order = ("observations", "reason", "choice")
        schema["properties"] = {key: schema["properties"][key] for key in order}
        schema["required"] = list(order)
        for key in ("observations", "reason"):
            schema["properties"][key]["maxLength"] = 32
    schema_file = directory / "response-schema.json"
    # JSON-schema-to-grammar conversion preserves property order. Sorting keys
    # here would silently move choice before the visible-evidence fields.
    schema_file.write_text(
        json.dumps(schema, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    command = [
        str(runtime / "llama-mtmd-cli.exe"),
        "-m",
        str(root / specification["model"]),
        "--mmproj",
        str(root / specification["projector"]),
        "-ngl",
        "all" if selected_runtime["device"] == "cuda" else "0",
        "--no-warmup",
        "-t",
        str(config["threads"]),
        "-c",
        "4096",
        "-n",
        str(config["max_new_tokens"]),
        "--temp",
        "0",
        "--seed",
        "2026",
        "--json-schema-file",
        str(schema_file),
        "-p",
        question,
    ]
    if selected_runtime["device"] == "cuda":
        command.extend(
            ["--device", "CUDA0", "--mmproj-device", "CUDA0", "--split-mode", "none", "-fit", "off"]
        )
    else:
        command.append("--no-mmproj-offload")
    if specification["name"] == "qwen35_08b":
        # The default 512-token physical batch allocates a large CPU scratch
        # buffer. Bound it so the engine and this owned worker can coexist.
        command.extend(["-ub", "64", "--jinja", "--chat-template", QWEN_TEMPLATE])
    # Repeating --image is deprecated and only the last argument is guaranteed
    # to survive parsing. One comma-separated argument is the pinned CLI API.
    command.extend(["--image", ",".join(view["path"] for view in views)])
    (directory / "prompt.txt").write_text(question, encoding="utf-8")
    (directory / "command.json").write_text(canonical(command), encoding="utf-8")
    return (
        command,
        runtime,
        {
            "views": views,
            "prompt_sha256": digest(question),
            "prompt_file_sha256": sha256(directory / "prompt.txt"),
            "command_sha256": digest(command),
            "native_image_argument_count": len(views),
            "native_media_marker_count": question.count("<__media__>"),
            "response_schema_sha256": sha256(schema_file),
            "response_field_order": list(schema["properties"]),
            "evidence_first": evidence_first,
            "prose_character_limit": 160 if analyze else (32 if evidence_first else 64),
            "physical_ubatch_size": 64 if specification["name"] == "qwen35_08b" else None,
            "native_profile": specification["name"],
            "chat_template_policy": (
                "qwen35-single-turn-no-tools-thinking-disabled"
                if specification["name"] == "qwen35_08b"
                else "pinned-model-default"
            ),
            "chat_template_sha256": (
                digest(QWEN_TEMPLATE) if specification["name"] == "qwen35_08b" else None
            ),
            "native_device_requested": selected_runtime["device"],
            "gpu_layers_requested": "all" if selected_runtime["device"] == "cuda" else 0,
            "runtime_sha256": next(iter(selected_runtime["archives"].values())),
            "runtime_archives": selected_runtime["archives"],
            "runtime_binary_sha256": binary_hashes,
            "runtime_executable_sha256": sha256(runtime / "llama-mtmd-cli.exe"),
        },
    )


def result(prepared, config, directory, identity):
    specification = profile(config)
    if sha256(directory / "response-schema.json") != identity["response_schema_sha256"]:
        raise ValueError("Native response schema changed during inference")
    if sha256(directory / "prompt.txt") != identity["prompt_file_sha256"]:
        raise ValueError("Native prompt changed during inference")
    if sha256(directory / "command.json") != identity["command_sha256"]:
        raise ValueError("Native command changed during inference")
    path = directory / "stdout.log"
    if path.stat().st_size > 256 * 1024:
        raise ValueError("Native VLM output exceeded its byte budget")
    raw = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    stderr = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
    cuda = config.get("native_device", "cpu") == "cuda"
    accelerator = (
        cuda_proof(stderr, config)
        if cuda
        else {"gpu_layers": 0, "projector_offloaded": False, "image_encoder_backend": "CPU"}
    )
    if "only last value will be used" in stderr or "Number of media markers" in stderr:
        raise ValueError("Native CLI did not bind the requested distinct image sequence")
    encoded = re.findall(r"encoding mtmd batch, n_chunks = (\d+)", stderr)
    completed = re.findall(r"mtmd batch encoding done in (\d+) ms", stderr)
    if len(completed) != len(encoded) or sum(map(int, encoded)) < len(prepared["images"]):
        raise ValueError("Native logs do not confirm encoding every submitted image")
    for view in identity["views"]:
        if sha256(view["path"]) != view["sha256"]:
            raise ValueError("Model input view changed during native inference")
    parsed = parse_generated(raw, prepared)
    if parsed is None:
        raise ValueError(f"Native VLM did not produce valid structured output; inspect {path}")
    analyze = prepared.get("mode") == "analyze"
    return {
        "analysis" if analyze else "decision": parsed["analysis" if analyze else "decision"],
        "receipt": {
            "schema_version": 1,
            "executed": True,
            "mode": prepared.get("mode", "plan"),
            "backend": "llama.cpp-cuda" if cuda else "llama.cpp-cpu",
            "model_id": specification["repo_id"],
            "model_revision": specification["revision"],
            "base_model_id": specification["base_model_id"],
            "quantizer": specification["quantizer"],
            "model_license": "Apache-2.0",
            "runtime_license": "MIT",
            "runtime_dependency_license": "NVIDIA CUDA Toolkit redistribution terms"
            if cuda
            else None,
            "weights": specification["weights"],
            "runtime_tag": RUNTIME_TAG,
            **identity,
            "source_digest": prepared["menu"]["source_digest"],
            "source_render_sha256": prepared["images"][0]["sha256"],
            "menu_digest": prepared["menu"]["menu_digest"],
            "request_digest": prepared["request_digest"],
            "source_analysis_digest": digest(prepared["source_analysis"])
            if prepared.get("source_analysis") is not None
            else None,
            "target_canvas": prepared["target"],
            "images": prepared["images"],
            "input_image_count": len(prepared["images"]),
            "native_media_chunks_encoded": sum(map(int, encoded)),
            "native_encoding_batches": len(completed),
            "native_image_encoding_verified": True,
            "native_encoding_milliseconds": [int(v) for v in completed],
            "native_stderr_sha256": sha256(directory / "stderr.log"),
            "native_log_excerpt": [line for line in stderr.splitlines() if "mtmd batch" in line],
            "quantization": specification["quantization"],
            "max_new_tokens": config["max_new_tokens"],
            "raw_output": raw,
            "visual_observations": parsed["observations"],
            "choice_letter": parsed.get("choice"),
            "decision_method": "schema_constrained_vlm_generation",
            "device": "cuda:0" if cuda else "cpu",
            "threads": config["threads"],
            **accelerator,
            "quality_validated": False,
            "regions_predicted": False,
            "crop_authority": False,
        },
    }
