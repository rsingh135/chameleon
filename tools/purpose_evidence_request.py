"""Prepare explicit three-view purpose requests without running a model.

Existing source/context PNG bytes are copied unchanged. Context receipts must
bind the source JSON/PNG, original element index/frame and exact context PNG.
Native painting runs serially on the calling thread; cohosted callers must also
own their application's renderer lock. This tool grants no layout authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from copy import deepcopy
from io import BytesIO
from pathlib import Path

from experiments.visual_finetune.data import build_purpose_prompt, canonical

PROTOCOL = "native-purpose-three-view-request-evidence-v1"
WORKER_PROTOCOL = "visual-adapter-label-free-evaluation-v1"
ORDERED_VIEWS = ("SOURCE", "CONTEXT", "ISOLATED_TARGET_ON_FIXED_CHECKER")
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 32_000_000
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_ELEMENTS = 256
SHA = re.compile(r"[0-9a-f]{64}\Z")
ANSWER_KEYS = frozenset({"label", "labels", "gold", "target", "expected", "answer"})


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _compact(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _no_answers(value):
    if isinstance(value, dict):
        if ANSWER_KEYS & value.keys():
            raise ValueError("Purpose evidence accepts raw source inputs, never gold answers")
        for item in value.values():
            _no_answers(item)
    elif isinstance(value, list):
        for item in value:
            _no_answers(item)


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate purpose evidence JSON key")
            result[key] = value
        return result

    result = json.loads(
        data.decode("utf-8-sig"),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite evidence JSON")),
    )
    if not isinstance(result, dict) or len(_compact(result)) > MAX_JSON_BYTES:
        raise ValueError("Purpose evidence metadata must be a bounded finite object")
    _no_answers(result)
    return result


def _snapshot(path, maximum, snapshots, budget):
    path = Path(path).resolve()
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Purpose evidence inputs must be regular files")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    budget[0] += len(data)
    if not data or len(data) > maximum or budget[0] > MAX_TOTAL_BYTES:
        raise ValueError("Purpose evidence input exceeds its byte budget")
    snapshots.append((path, _sha(data), maximum))
    return data


def _png(data, *, maximum_axis=8192, expected_size=None, expected_mode=None):
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        if (
            image.format != "PNG"
            or image.width * image.height > MAX_PIXELS
            or max(image.size) > maximum_axis
            or min(image.size) < 1
            or expected_size is not None
            and image.size != tuple(expected_size)
            or expected_mode is not None
            and image.mode != expected_mode
        ):
            raise ValueError("Purpose PNG dimensions, format or mode differ from its contract")
        image.load()
        return {
            "sha256": _sha(data),
            "size": list(image.size),
            "mode": image.mode,
            "pixel_sha256": _sha(image.tobytes()),
            "worker_rgb_pixel_sha256": _sha(image.convert("RGB").tobytes()),
        }


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _source(source):
    elements, canvas = source.get("elements"), source.get("canvas", {})
    if (
        not isinstance(elements, list)
        or not 1 <= len(elements) <= MAX_ELEMENTS
        or any(
            type(canvas.get(key)) is not int or not 1 <= canvas[key] <= 8192
            for key in ("width", "height")
        )
        or canvas["width"] * canvas["height"] > MAX_PIXELS
    ):
        raise ValueError("Purpose source requires bounded native canvas and original elements")
    ids = []
    for element in elements:
        identity = element.get("id") if isinstance(element, dict) else None
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or len(identity) > 200
            or any(ord(char) < 32 for char in identity)
            or element.get("type") not in {"text", "image", "shape"}
            or any(not _finite(element.get(key)) for key in ("x", "y", "width", "height"))
            or min(element["width"], element["height"]) <= 0
            or not _finite(element.get("z_index", 0))
        ):
            raise ValueError("Purpose source requires valid unique original IDs and geometry")
        ids.append(identity)
    if len(ids) != len(set(ids)):
        raise ValueError("Purpose source contains duplicate original IDs")
    return ids


def _contexts(receipt, source, source_sha, source_png_sha):
    allowed = {
        "schema_version",
        "protocol",
        "source_json_sha256",
        "source_render_sha256",
        "records",
        "crop_recipe",
        "implementation_sha256",
        "source_digest",
        "source_digest_protocol",
    }
    row_keys = {
        "element_id",
        "element_index",
        "frame_xywh",
        "context_sha256",
        "context_size",
        "context_box_pixels",
        "source_frame_visible",
        "crop_kind",
        "labels_or_id_overlays_added",
    }
    if (
        set(receipt) - allowed
        or receipt.get("schema_version") != 1
        or not isinstance(receipt.get("protocol"), str)
        or not 1 <= len(receipt["protocol"]) <= 160
        or receipt.get("source_json_sha256") != source_sha
        or receipt.get("source_render_sha256") != source_png_sha
        or not isinstance(receipt.get("records"), list)
        or len(receipt["records"]) > MAX_ELEMENTS
    ):
        raise ValueError("Context receipt must bind the exact source JSON and PNG")
    indexed = {}
    for row in receipt["records"]:
        index = row.get("element_index") if isinstance(row, dict) else None
        if type(index) is not int or not 0 <= index < len(source["elements"]):
            raise ValueError("Context receipt requires an original element index")
        element = source["elements"][index]
        key = element["id"]
        if (
            set(row) - row_keys
            or row.get("labels_or_id_overlays_added", False) is not False
            or key in indexed
            or row.get("element_id") != key
            or row.get("frame_xywh") != [element[k] for k in ("x", "y", "width", "height")]
            or not isinstance(row.get("context_sha256"), str)
            or not SHA.fullmatch(row["context_sha256"])
        ):
            raise ValueError("Context receipt differs from its original element frame or identity")
        indexed[key] = row
    return indexed


def _resource_budget(source, assets, fonts):
    from viewer.resize_cache import _image_paths

    paths = set()
    for element in source["elements"]:
        if element["type"] == "image":
            _, path = _image_paths(assets, element.get("asset_id"))
            if path.exists():
                paths.add(path)
    paths.update(
        path for path in fonts.rglob("*") if path.suffix.lower() in {".ttf", ".otf", ".json"}
    )
    total = 0
    for path in paths:
        info = path.stat()
        if (
            not path.resolve().is_relative_to(assets)
            and not path.resolve().is_relative_to(fonts)
            or not stat.S_ISREG(info.st_mode)
            or info.st_size > 64 * 1024 * 1024
        ):
            raise ValueError("Purpose resources must be bounded local regular files")
        total += info.st_size
    if len(paths) > 400 or total > MAX_TOTAL_BYTES:
        raise ValueError("Purpose resources exceed their inspection budget")


def _render(element, assets, fonts):
    from tools.native_element_evidence import render_isolated_element

    return render_isolated_element(element, assets_dir=assets, fonts_dir=fonts)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def prepare_purpose_evidence(
    source_path,
    source_render,
    context_images,
    *,
    context_receipt,
    assets_dir,
    fonts_dir,
    family_id,
    out,
    unavailable_policy="fail",
    excluded_element_ids=(),
    expected_source_sha256=None,
    expected_resource_digest=None,
    inference_identity=None,
):
    """Write a fresh, label-free three-view request and model-independent receipts.

    ``context_receipt`` is a JSON path with source_json_sha256,
    source_render_sha256 and records containing element_id, element_index,
    frame_xywh and context_sha256. Optional exclusions never invoke native paint.
    expected_resource_digest identifies canonical resource hashes, excluding paths.
    Failure leaves no READY.json; callers must use a fresh directory for a retry.
    """
    from viewer.resize_cache import ResourceBindings

    implementation_sha = _sha(Path(__file__).read_bytes())
    if (
        not isinstance(family_id, str)
        or not family_id.strip()
        or len(family_id) > 200
        or any(ord(char) < 32 for char in family_id)
        or unavailable_policy not in {"fail", "exclude"}
    ):
        raise ValueError("Purpose evidence needs a bounded family and explicit unavailable policy")
    requested = Path(out)
    if requested.is_symlink() or requested.exists():
        raise ValueError("Purpose evidence requires a fresh output directory")
    destination, assets, fonts = (
        requested.resolve(),
        Path(assets_dir).resolve(),
        Path(fonts_dir).resolve(),
    )
    if (
        not assets.is_dir()
        or not fonts.is_dir()
        or any(
            destination.is_relative_to(root) or root.is_relative_to(destination)
            for root in (assets, fonts)
        )
    ):
        raise ValueError("Purpose output must be independent of existing resource directories")
    snapshots, budget = [], [0]
    source_bytes = _snapshot(source_path, MAX_JSON_BYTES, snapshots, budget)
    source_sha, source = _sha(source_bytes), _json(source_bytes)
    if expected_source_sha256 is not None and expected_source_sha256 != source_sha:
        raise ValueError("Purpose source JSON differs from its expected bytes")
    ids = _source(source)
    frozen = deepcopy(source)
    if (
        not isinstance(context_images, dict)
        or set(context_images) - set(ids)
        or isinstance(excluded_element_ids, (str, bytes))
    ):
        raise ValueError("Purpose contexts and exclusions must identify original elements")
    excluded = set(excluded_element_ids)
    if excluded - set(ids):
        raise ValueError("Purpose exclusions contain an unknown original element")
    if inference_identity is not None:
        inference_identity = _json(_compact(inference_identity))
        if inference_identity.get("purpose_views") != 3:
            raise ValueError("Optional inference identity must explicitly bind purpose_views=3")

    def image_snapshot(ref, **kwargs):
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            raise ValueError("Purpose evidence requires explicit local PNG references")
        data = _snapshot(ref["path"], MAX_IMAGE_BYTES, snapshots, budget)
        info = _png(data, **kwargs)
        if ref.get("sha256") != info["sha256"]:
            raise ValueError("Purpose PNG bytes differ from their declared SHA-256")
        return data, info

    source_png, source_info = image_snapshot(
        source_render, expected_size=[source["canvas"][k] for k in ("width", "height")]
    )
    receipt_bytes = _snapshot(context_receipt, MAX_JSON_BYTES, snapshots, budget)
    context_rows = _contexts(_json(receipt_bytes), source, source_sha, source_info["sha256"])
    contexts = {}
    for key, ref in context_images.items():
        if key not in context_rows:
            raise ValueError("Context PNG has no source/frame-bound receipt")
        data, info = image_snapshot(ref, maximum_axis=256)
        if context_rows[key]["context_sha256"] != info["sha256"]:
            raise ValueError("Context PNG differs from its source/frame-bound receipt")
        if context_rows[key].get("context_size", info["size"]) != info["size"]:
            raise ValueError("Context PNG dimensions differ from its source/frame-bound receipt")
        contexts[key] = data, info
    if any(path.is_relative_to(destination) for path, _, _ in snapshots):
        raise ValueError("Purpose output cannot contain original evidence inputs")
    _resource_budget(source, assets, fonts)
    resource_bindings = ResourceBindings()
    resources = resource_bindings.bind(source, assets, fonts, allow_missing=True)
    resource_digest = _sha(_compact(resources.hashes()))
    if expected_resource_digest is not None and expected_resource_digest != resource_digest:
        raise ValueError("Purpose resources differ from their expected digest")
    destination.mkdir(parents=True, exist_ok=False)
    written = set()

    def save_image(data, info):
        relative = "images/" + info["sha256"] + ".png"
        if relative not in written:
            _write(destination / relative, data)
            written.add(relative)
        return {"path": relative, "sha256": info["sha256"]}

    render_ref = save_image(source_png, source_info)
    order = sorted(range(len(ids)), key=lambda i: source["elements"][i].get("z_index", 0))
    ranks = {index: rank for rank, index in enumerate(order)}
    request = {"protocol": WORKER_PROTOCOL, "task": "purpose", "purpose_views": 3, "records": []}
    bindings, omitted = [], []
    for index, element in enumerate(source["elements"]):
        identity = element["id"]
        reason = "explicit_admission_exclusion" if identity in excluded else None
        if reason is None and identity not in contexts:
            reason = "missing_source_bound_context"
        if reason is None:
            native = _render(element, assets, fonts)
            if native.get("status") != "available":
                reason = (
                    "native_isolation_unavailable: " + str(native.get("reason", "unknown"))[:500]
                )
        if reason is not None:
            if identity not in excluded and unavailable_policy == "fail":
                raise ValueError(reason)
            omitted.append({"element_id": identity, "element_index": index, "reason": reason})
            continue
        receipt = native["receipt"]
        if (
            not isinstance(receipt, dict)
            or receipt.get("resources_unchanged") is not True
            or receipt.get("status") != "available"
            or receipt.get("clipped") is not False
            or receipt.get("native_single_original_layer") is not True
            or receipt.get("frame_or_asset_fallback_used") is not False
            or receipt.get("geometry_authority") is not False
            or receipt.get("model_executed") is not False
            or not isinstance(receipt.get("policy"), dict)
            or receipt["policy"].get("version") != "native-element-isolation-v1"
            or not isinstance(receipt.get("implementation_sha256"), str)
            or not SHA.fullmatch(receipt["implementation_sha256"])
            or receipt.get("source_element_sha256") != _sha(_compact(element))
            or receipt.get("receipt_digest")
            != _sha(_compact({k: v for k, v in receipt.items() if k != "receipt_digest"}))
        ):
            raise ValueError(
                "Native isolated receipt differs from its original element or metadata"
            )
        alpha, checker = native["alpha_png"], native["checker_png"]
        if not isinstance(alpha, bytes) or not isinstance(checker, bytes):
            raise ValueError("Available native evidence requires both encoded PNGs")
        budget[0] += len(alpha) + len(checker)
        if (
            len(alpha) > MAX_IMAGE_BYTES
            or len(checker) > MAX_IMAGE_BYTES
            or budget[0] > MAX_TOTAL_BYTES
        ):
            raise ValueError("Native purpose artifacts exceed their total byte budget")
        alpha_info = _png(alpha, maximum_axis=512, expected_mode="RGBA")
        checker_info = _png(
            checker, maximum_axis=512, expected_mode="RGB", expected_size=alpha_info["size"]
        )
        if (
            receipt.get("alpha_png_sha256") != alpha_info["sha256"]
            or receipt.get("checker_png_sha256") != checker_info["sha256"]
        ):
            raise ValueError("Native isolated PNG bytes differ from their receipt")
        crop, crop_info = contexts[identity]
        crop_ref, checker_ref = save_image(crop, crop_info), save_image(checker, checker_info)
        alpha_ref = save_image(alpha, alpha_info)
        relative = f"native-receipts/element-{index:04d}.json"
        encoded_receipt = _compact(receipt) + b"\n"
        _write(destination / relative, encoded_receipt)
        context = {
            "type": element["type"],
            "text_data": str(element.get("content", ""))[:240],
            "bbox_normalized": [
                element[k] / source["canvas"][axis]
                for k, axis in (
                    ("x", "width"),
                    ("y", "height"),
                    ("width", "width"),
                    ("height", "height"),
                )
            ],
            "paint_order_index": ranks[index],
            "element_count": len(ids),
        }
        prompt = build_purpose_prompt(context, purpose_views=3)
        record_id = f"element-{index:04d}"
        request["records"].append(
            {
                "record_id": record_id,
                "family_id": family_id,
                "control": "original",
                "prompt": prompt,
                "images": [render_ref, crop_ref, checker_ref],
            }
        )
        bindings.append(
            {
                "record_id": record_id,
                "element_id": identity,
                "element_index": index,
                "source_element_sha256": _sha(_compact(element)),
                "paint_order_index": ranks[index],
                "prompt_sha256": _sha(prompt.encode()),
                "ordered_image_sha256": [
                    ref["sha256"] for ref in (render_ref, crop_ref, checker_ref)
                ],
                "views": [source_info, crop_info, checker_info],
                "alpha_image": {**alpha_ref, **alpha_info},
                "native_receipt": {"path": relative, "sha256": _sha(encoded_receipt)},
                "native_policy": receipt.get("policy"),
                "native_implementation_sha256": receipt.get("implementation_sha256"),
            }
        )
    if not request["records"]:
        raise ValueError("Purpose request has no admitted three-view records")
    if (
        source != frozen
        or resource_bindings.bind(source, assets, fonts, allow_missing=True) != resources
    ):
        raise ValueError("Original source or resources changed during purpose evidence preparation")
    for path, expected, maximum in snapshots:
        if _sha(_snapshot(path, maximum, [], [0])) != expected:
            raise ValueError("Original source/context input changed before request publication")
    if _sha(Path(__file__).read_bytes()) != implementation_sha:
        raise ValueError("Purpose evidence implementation changed before request publication")
    _write(destination / "source.json", source_bytes)
    _write(destination / "context-receipt.json", receipt_bytes)
    request_bytes = (canonical(request) + "\n").encode("utf-8")
    _write(destination / "request.json", request_bytes)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete_model_independent_evidence",
        "source_json_sha256": source_sha,
        "source_canonical_sha256": _sha(_compact(source)),
        "source_canonical_digest_protocol": "sorted-compact-utf8-no-ascii-escape-v1",
        "source_render": {**render_ref, **source_info},
        "context_receipt": {"path": "context-receipt.json", "sha256": _sha(receipt_bytes)},
        "resource_identity": {
            "assets_dir": str(assets),
            "fonts_dir": str(fonts),
            "resource_digest": resource_digest,
            "hashes": resources.hashes(),
        },
        "purpose_views": 3,
        "ordered_views": list(ORDERED_VIEWS),
        "preprocessing": {
            "source_and_context": "exact_supplied_png_bytes_no_recrop_or_resave",
            "model_image_conversion": "PIL.Image.convert('RGB')",
            "isolated_alpha_is_sidecar_not_fourth_model_image": True,
        },
        "records": bindings,
        "coverage": {
            "source_count": len(ids),
            "admitted_count": len(bindings),
            "excluded_count": len(omitted),
            "excluded": omitted,
        },
        "unavailable_policy": unavailable_policy,
        "request": {"path": "request.json", "sha256": _sha(request_bytes)},
        "inference_identity": None,
        "implementation_sha256": implementation_sha,
        "source_context_resources_unchanged": True,
        "inference_executed": False,
        "quality_approved": False,
        "promotion_eligible": False,
    }
    if inference_identity is not None:
        encoded = _compact(inference_identity) + b"\n"
        _write(destination / "expected-inference-identity.json", encoded)
        manifest["inference_identity"] = {
            "path": "expected-inference-identity.json",
            "sha256": _sha(encoded),
            "scope": "expected_only_no_execution",
        }
    manifest_bytes = _compact(manifest) + b"\n"
    _write(destination / "manifest.json", manifest_bytes)
    _write(
        destination / "READY.json",
        _compact(
            {
                "manifest_sha256": _sha(manifest_bytes),
                "request_sha256": _sha(request_bytes),
                "inference_executed": False,
            }
        )
        + b"\n",
    )
    return manifest
