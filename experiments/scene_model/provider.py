"""Pinned, offline CPU scene-role inference for the source-only local harness.

No model import, download, training or process starts at module import time.
The provider returns proposals only; identity and permissions remain external.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

from .schema import ROLE_NAMES, validate_prediction

CLIP_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
FEATURE_SCHEMA = "clip-two-view-letterbox-l2-source-asset-v2"
PROVIDER_VERSION = "offline-clip-scene-v1"
MAX_REQUEST_BYTES = 512 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ROLE_PROMPTS = {
    "background": (
        "a photograph of people at an event used as the background of a poster",
        "a photograph of a place or environment behind graphic design text",
        "a full scene photograph supporting a poster composition",
    ),
    "product": (
        "a standalone product package promoted in an advertisement",
        "an isolated consumer product object on a plain background",
        "a product bottle or box shown as the main advertised object",
    ),
    "logo": (
        "a brand logo or wordmark used as graphic identity",
        "a simple company symbol or emblem on a plain background",
        "a small graphic brand mark",
    ),
    "foreground": (
        "a decorative illustration used as a foreground design element",
        "a diagram icon or piece of vector artwork",
        "a framed supporting illustration in a poster",
    ),
    "unknown": (
        "a smooth abstract gradient or translucent fade layer",
        "a plain monochrome rectangle or unrecognizable texture",
        "an abstract image with no identifiable subject",
    ),
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("JSON numbers must be finite")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Input must be bounded, finite JSON") from exc


def _json_file(path, limit=MAX_REQUEST_BYTES):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("Local JSON file exceeds its size limit")
    return _parse(path.read_text(encoding="utf-8-sig"))


def validate_config(value, *, base_dir=None):
    required = {"schema_version", "backend", "checkpoint_path", "checkpoint_sha256"}
    optional = {"device", "threads", "max_images", "head_path", "head_sha256", "processor_digest"}
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        raise ValueError("Provider configuration has missing or unknown fields")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["backend"] != "openai_clip"
    ):
        raise ValueError("Only offline OpenAI CLIP schema1 is supported")
    if value["checkpoint_sha256"] != CLIP_B32_SHA256:
        raise ValueError(
            "This provider requires the pinned original OpenAI CLIP ViT-B/32 checkpoint"
        )
    if value.get("device", "cpu") != "cpu":
        raise ValueError("This local provider is CPU-only; it never starts GPU or cloud compute")
    result = deepcopy(value)
    for name, default, minimum, maximum in (("threads", 4, 1, 8), ("max_images", 16, 1, 16)):
        number = value.get(name, default)
        if type(number) is not int or not minimum <= number <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        result[name] = number
    if ("head_path" in value) != ("head_sha256" in value):
        raise ValueError("An optional trained head requires an explicit path and SHA-256")
    for name in ("checkpoint_path", "head_path"):
        if name not in result:
            continue
        raw = result[name]
        if not isinstance(raw, str) or not raw or "://" in raw:
            raise ValueError(
                "Model files must be explicit local paths, never repository IDs or URLs"
            )
        path = Path(raw).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        result[name] = str(path.resolve())
    result["device"] = "cpu"
    return result


def _rgb(path):
    from PIL import Image

    path = Path(path)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Source views must be existing local images of at most32MiB")
    with Image.open(path) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Source views must be still images of at most20MP")
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def whole_view(path, side=224):
    """Keep every source/asset edge before CLIP's otherwise square center crop."""
    from PIL import Image

    image = _rgb(path)
    scale = side / max(image.size)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(resized, ((side - width) // 2, (side - height) // 2))
    return canvas


def processor_recipe(preprocess):
    """Serialize typed transform parameters, never process-specific repr values."""
    recipe = []
    for transform in preprocess.transforms:
        if inspect.isfunction(transform):
            recipe.append(
                {"type": "function", "module": transform.__module__, "name": transform.__qualname__}
            )
            continue
        kind = type(transform).__name__
        row = {"type": kind, "module": type(transform).__module__}
        if kind == "Resize":
            row.update(
                size=transform.size,
                interpolation=transform.interpolation.value,
                max_size=transform.max_size,
                antialias=transform.antialias,
            )
        elif kind == "CenterCrop":
            row["size"] = list(transform.size)
        elif kind == "Normalize":
            row.update(
                mean=list(transform.mean), std=list(transform.std), inplace=transform.inplace
            )
        elif kind != "ToTensor":
            raise ValueError("Unrecognized transform in the pinned CLIP processor")
        recipe.append(row)
    _canonical(recipe)
    return recipe


def _function_hash(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


class ClipEncoder:
    """One immutable local backbone plus its reproducible two-view processor."""

    def __init__(self, model, preprocess, clip, torch, checkpoint_sha256):
        self.model, self.preprocess, self.clip, self.torch = model, preprocess, clip, torch
        self.checkpoint_sha256 = checkpoint_sha256
        directory = Path(clip.__file__).resolve().parent
        versions = {}
        for name in ("torch", "torchvision", "clip", "Pillow"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.processor_identity = {
            "feature_schema": FEATURE_SCHEMA,
            "full_view_policy": (
                "aspect-preserving bicubic resize into224×224 RGB white letterbox "
                "before CLIP preprocess; no view crop"
            ),
            "preprocess": processor_recipe(preprocess),
            "view_code_sha256": {
                function.__name__: _function_hash(function)
                for function in (_rgb, whole_view, encode_clip_views)
            },
            "alpha_policy": "composite exact source RGBA on white before CLIP preprocessing",
            "clip_files": {
                name: _file_hash(directory / name)
                for name in (
                    "clip.py",
                    "model.py",
                    "simple_tokenizer.py",
                    "bpe_simple_vocab_16e6.txt.gz",
                )
            },
            "packages": versions,
        }
        self.processor_digest = _digest(self.processor_identity)
        self._text_features = None


def load_clip_encoder(checkpoint_path, *, expected_sha256, device="cpu", threads=4):
    if device != "cpu" or type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Only a bounded local CPU runtime is enabled")
    path = Path(checkpoint_path).expanduser().resolve()
    if (
        expected_sha256 != CLIP_B32_SHA256
        or not path.is_file()
        or _file_hash(path) != expected_sha256
    ):
        raise ValueError("Pinned local CLIP checkpoint is missing or its SHA-256 differs")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import clip
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Use an existing local Python with OpenAI clip and torch installed; "
            "no dependency is downloaded"
        ) from exc
    torch.set_num_threads(threads)
    # Passing an existing absolute filename takes CLIP's local-file branch;
    # never pass a model nickname or URL that could trigger its downloader.
    model, preprocess = clip.load(str(path), device="cpu", jit=False)
    model.eval()
    return ClipEncoder(model, preprocess, clip, torch, expected_sha256)


def encode_clip_views(encoder, source_path, asset_path):
    torch = encoder.torch
    images = torch.stack(
        [encoder.preprocess(whole_view(path)) for path in (source_path, asset_path)]
    )
    with torch.inference_mode():
        values = encoder.model.encode_image(images.to("cpu")).float()
        values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if tuple(values.shape) != (2, 512) or not torch.isfinite(values).all():
        raise ValueError("Pinned CLIP backbone must emit two finite512-dimensional image features")
    return values.reshape(1, 1024).cpu()


def clip_role_probabilities(encoder, features):
    torch = encoder.torch
    if encoder._text_features is None:
        texts = []
        with torch.inference_mode():
            for role in ROLE_NAMES:
                tokens = encoder.clip.tokenize(list(ROLE_PROMPTS[role]))
                values = encoder.model.encode_text(tokens).float()
                values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                pooled = values.mean(dim=0)
                texts.append(pooled / pooled.norm().clamp_min(1e-12))
            encoder._text_features = torch.stack(texts)
    with torch.inference_mode():
        # Asset appearance is primary; source context helps distinguish an
        # isolated promoted object from photographic scene support.
        values = features.reshape(2, 512)
        combined = 0.25 * values[0] + 0.75 * values[1]
        combined = combined / combined.norm().clamp_min(1e-12)
        logits = encoder.model.logit_scale.exp() * combined @ encoder._text_features.T
        probabilities = logits.softmax(dim=-1)
    return [float(v) for v in probabilities.tolist()]


def clip_zero_shot(encoder, features):
    probabilities = clip_role_probabilities(encoder, features)
    best = max(range(len(ROLE_NAMES)), key=probabilities.__getitem__)
    return validate_prediction(
        {
            "proposed_role": ROLE_NAMES[best],
            "confidence": probabilities[best],
            "protected_regions": [],
            "rationale": (
                "Offline CLIP two-view zero-shot role similarity; confidence is uncalibrated. "
                "No protected-region detector was run."
            ),
        }
    )


def load_head(path, expected_sha256, encoder):
    if _file_hash(path) != expected_sha256:
        raise ValueError("Explicit trained role-head SHA-256 differs")
    head = _json_file(path, 2 * 1024 * 1024)
    if (
        head.get("feature_schema") != FEATURE_SCHEMA
        or head.get("backbone_sha256") != encoder.checkpoint_sha256
        or head.get("processor_digest") != encoder.processor_digest
        or head.get("roles") != list(ROLE_NAMES)
    ):
        raise ValueError("Role head uses a different feature, backbone, processor or role order")
    weights, bias = head.get("weights"), head.get("bias")
    if (
        not isinstance(weights, list)
        or len(weights) != 5
        or any(not isinstance(row, list) or len(row) != 1024 for row in weights)
        or not isinstance(bias, list)
        or len(bias) != 5
    ):
        raise ValueError("Role head must contain5×1024 weights and5 biases")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for row in [*weights, bias]
        for v in row
    ):
        raise ValueError("Role-head parameters must be finite real numbers")
    return head


def _head_prediction(encoder, features, head):
    torch = encoder.torch
    with torch.inference_mode():
        weights, bias = torch.tensor(head["weights"]), torch.tensor(head["bias"])
        probabilities = (features @ weights.T + bias).softmax(dim=-1)[0].tolist()
    best = max(range(5), key=probabilities.__getitem__)
    return validate_prediction(
        {
            "proposed_role": ROLE_NAMES[best],
            "confidence": float(probabilities[best]),
            "protected_regions": [],
            "rationale": (
                "Explicit pinned CPU role head on frozen two-view CLIP features; "
                "confidence is uncalibrated. No protected-region detector was run."
            ),
        }
    )


def prepare_request(request, max_images=16):
    if not isinstance(request, dict) or len(_canonical(request).encode()) > MAX_REQUEST_BYTES:
        raise ValueError("Scene request must be a bounded finite object")
    source, profile = request.get("source"), request.get("deterministic_scene")
    if not isinstance(source, dict) or not isinstance(profile, dict):
        raise ValueError("Source design and deterministic scene are required")
    if (
        request.get("source_digest") != _digest(source)
        or profile.get("source_digest") != request["source_digest"]
        or request.get("scene_digest") != profile.get("scene_digest")
    ):
        raise ValueError("Scene request source/profile digest binding differs")
    if _digest(
        {key: value for key, value in profile.items() if key != "scene_digest"}
    ) != profile.get("scene_digest"):
        raise ValueError("Deterministic scene profile content differs from its digest")
    elements = source.get("elements")
    if (
        not isinstance(elements, list)
        or len(elements) > 200
        or not all(isinstance(e, dict) and isinstance(e.get("id"), str) for e in elements)
    ):
        raise ValueError("Scene source needs at most200 known elements")
    by_id = {e["id"]: e for e in elements}
    if len(by_id) != len(elements):
        raise ValueError("Duplicate source element IDs")
    source_images = {e["id"] for e in elements if e.get("type") == "image"}
    images = profile.get("images")
    files = request.get("asset_files")
    if not isinstance(images, list) or len(images) > max_images or not isinstance(files, dict):
        raise ValueError("Scene image count exceeds the configured local limit")
    ids = [row.get("element_id") for row in images if isinstance(row, dict)]
    if (
        len(ids) != len(images)
        or not all(isinstance(identity, str) for identity in ids)
        or len(set(ids)) != len(ids)
        or set(ids) != source_images
        or set(files) != source_images
    ):
        raise ValueError("Scene images and asset paths must match every known source image exactly")
    rendered = request.get("source_render")
    if (
        not isinstance(rendered, dict)
        or not isinstance(rendered.get("path"), str)
        or _file_hash(rendered["path"]) != rendered.get("sha256")
    ):
        raise ValueError("Source render bytes differ from the bound input")
    _rgb(rendered["path"])
    selected = []
    for row in images:
        identity = row["element_id"]
        path = files[identity]
        if (
            not isinstance(path, str)
            or "://" in path
            or _file_hash(path) != row.get("asset_sha256")
        ):
            raise ValueError("Source asset bytes differ from their known image binding")
        _rgb(path)
        selected.append((by_id[identity], row, path))
    return rendered["path"], selected


class SceneModelProvider:
    def __init__(self, config):
        self.config = validate_config(config)
        self.encoder = None
        self.head = None
        self.receipt = None

    def predict(self, request):
        source_path, selected = prepare_request(request, self.config["max_images"])
        before = _digest(request)
        if self.encoder is None and selected:
            self.encoder = load_clip_encoder(
                self.config["checkpoint_path"],
                expected_sha256=self.config["checkpoint_sha256"],
                threads=self.config["threads"],
            )
            if (
                self.config.get("processor_digest", self.encoder.processor_digest)
                != self.encoder.processor_digest
            ):
                raise ValueError("Installed CLIP processor differs from the explicit pin")
            if "head_path" in self.config:
                self.head = load_head(
                    self.config["head_path"], self.config["head_sha256"], self.encoder
                )
        predictions = []
        for element, evidence, path in selected:
            features = encode_clip_views(self.encoder, source_path, path)
            prediction = (
                _head_prediction(self.encoder, features, self.head)
                if self.head
                else clip_zero_shot(self.encoder, features)
            )
            prediction = validate_prediction(prediction)
            predictions.append(
                {
                    "element_id": element["id"],
                    "asset_sha256": evidence["asset_sha256"],
                    **prediction,
                }
            )
        # Recheck actual input bytes, not only dictionaries, after inference.
        prepare_request(request, self.config["max_images"])
        if before != _digest(request):
            raise RuntimeError("Inference mutated its source request")
        response = {
            "schema_version": 1,
            "source_digest": request["source_digest"],
            "scene_digest": request["scene_digest"],
            "images": predictions,
        }
        self.receipt = {
            "provider_version": PROVIDER_VERSION,
            "provider_sha256": _file_hash(__file__),
            "prompt_recipe_sha256": _digest(ROLE_PROMPTS),
            "backend": "frozen_clip_role_head" if self.head else "clip_zero_shot",
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "processor_digest": self.encoder.processor_digest if self.encoder else None,
            "processor_identity": self.encoder.processor_identity if self.encoder else None,
            "head_sha256": self.config.get("head_sha256"),
            "feature_schema": FEATURE_SCHEMA,
            "source_digest": request["source_digest"],
            "scene_digest": request["scene_digest"],
            "source_immutable": True,
            "input_bytes_unchanged": True,
            "device": "cpu",
            "image_count": len(predictions),
            "protected_region_detector": False,
            "confidence_calibrated": False,
            "crop_authority": False,
            "model_weights_updated": False,
        }
        return response


def predict_scene(request, config):
    """Callable counterpart of the stdin/stdout CLI; returns harness-compatible JSON."""
    return SceneModelProvider(config).predict(request)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--request", type=Path, help="Otherwise read the harness JSON request on stdin"
    )
    parser.add_argument("--receipt", type=Path, help="Optional provider/runtime provenance sidecar")
    args = parser.parse_args(argv)
    try:
        config = validate_config(_json_file(args.config), base_dir=args.config.resolve().parent)
        if args.request:
            request = _json_file(args.request)
        else:
            text = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1).decode("utf-8")
            if len(text.encode()) > MAX_REQUEST_BYTES:
                raise ValueError("Scene request exceeds its byte limit")
            request = _parse(text)
        provider = SceneModelProvider(config)
        response = provider.predict(request)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(
                json.dumps(provider.receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        print(json.dumps(response, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Offline scene provider unavailable: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
