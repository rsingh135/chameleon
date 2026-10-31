"""Ground actual adapter purpose scores without granting layout authority.

This module loads no model and launches no process. An operator-owned runner
prepares and scores label-free records with the saved adapter. Only its pinned,
source/image/request-bound results can become advisory subject observations.
Deployment is disabled; explicit diagnostic use is separate from qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from copy import deepcopy

from engine.visual_analysis import bind_analysis
from experiments.visual_finetune.data import ALIASES, build_purpose_prompt

PROTOCOL = "purpose-adapter-analyzer-binding-v1"
WORKER_PROTOCOL = "visual-adapter-label-free-evaluation-v1"
SCORING = "visual_prefix_single_token_closed_class_logits_not_generated_prose"
MAX_ELEMENTS = 256
MAX_PREDICTION_BYTES = 8 * 1024 * 1024
MIN_PROBABILITY = 0.85
MIN_MARGIN = 0.30
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CANONICAL = {
    "action": "action",
    "background": "background",
    "body": "body",
    "brand": "brand_mark",
    "accent": "decoration",
    "footer": "footer_destination",
    "headline": "headline",
    "visual": "primary_visual",
    "support": "supporting_copy",
    "unknown": "unknown",
}
CLASSES = tuple(sorted(_CANONICAL))
_IDENTITY_KEYS = {
    "base",
    "adapter_sha256",
    "adapter_config_sha256",
    "recipe_sha256",
    "dataset_sha256",
    "code_sha256",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_digest(source):
    return hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("Adapter analyzer identities must be exact SHA-256 values")
    return value


def _ref(value):
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError("Purpose inference requires an explicit source/context image reference")
    path = value["path"]
    if not path or "://" in path or any(ord(character) < 32 for character in path):
        raise ValueError("Purpose inference uses local image paths only")
    return {"path": path, "sha256": _sha(value.get("sha256"))}


def _source(source):
    canonical(source)
    if not isinstance(source, dict) or not isinstance(source.get("elements"), list):
        raise ValueError("Purpose inference requires original editable source elements")
    if not 1 <= len(source["elements"]) <= MAX_ELEMENTS:
        raise ValueError("Purpose inference source element budget exceeded")
    canvas = source.get("canvas", {})
    if any(
        type(canvas.get(k)) not in {int, float} or not math.isfinite(canvas[k]) or canvas[k] <= 0
        for k in ("width", "height")
    ):
        raise ValueError("Purpose inference requires finite source dimensions")
    ids = []
    for element in source["elements"]:
        key = element.get("id")
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 200
            or not key.strip()
            or any(ord(c) < 32 for c in key)
            or element.get("type") not in {"text", "image", "shape"}
        ):
            raise ValueError("Purpose inference needs valid original IDs/types")
        if any(
            type(element.get(k)) not in {int, float} or not math.isfinite(element[k])
            for k in ("x", "y", "width", "height")
        ):
            raise ValueError("Purpose inference source geometry must be finite")
        if min(element["width"], element["height"]) <= 0:
            raise ValueError("Purpose inference source frames must have positive area")
        if type(element.get("z_index", 0)) not in {int, float} or not math.isfinite(
            element.get("z_index", 0)
        ):
            raise ValueError("Purpose inference paint order must be finite")
        ids.append(key)
    if len(set(ids)) != len(ids):
        raise ValueError("Purpose inference IDs must be unique")
    return ids


def _purpose_views(value):
    if type(value) is not int or value not in {2, 3}:
        raise ValueError("Purpose views must explicitly be integer two or three")
    return value


def _native_digest(value):
    """Match the source owner's UTF-8 native receipt contract exactly."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _isolated_ref(element, reference, receipt):
    """Accept only the actual native isolation policy; unsupported paint abstains."""
    if isinstance(receipt, dict) and receipt.get("status") == "unavailable":
        return None, "native_isolation_unavailable"
    if isinstance(receipt, dict) and receipt.get("clipped") is True:
        return None, "native_isolation_clipped_or_unverified"
    if reference is None:
        return None, "missing_isolated_image"
    if not isinstance(receipt, dict):
        return None, "missing_isolated_receipt"
    if receipt.get("status") != "available":
        return None, "native_isolation_unavailable"
    if receipt.get("clipped") is not False:
        return None, "native_isolation_clipped_or_unverified"
    ref = _ref(reference)
    if (
        len(canonical(receipt).encode()) > 512 * 1024
        or receipt.get("protocol") != "native-element-isolation-v1"
        or not isinstance(receipt.get("policy"), dict)
        or receipt["policy"].get("version") != "native-element-isolation-v1"
        or receipt.get("resources_unchanged") is not True
        or receipt.get("native_single_original_layer") is not True
        or receipt.get("frame_or_asset_fallback_used") is not False
        or receipt.get("geometry_authority") is not False
        or receipt.get("model_executed") is not False
        or receipt.get("source_element_sha256") != _native_digest(element)
        or receipt.get("checker_png_sha256") != ref["sha256"]
        or receipt.get("receipt_digest")
        != _native_digest({k: v for k, v in receipt.items() if k != "receipt_digest"})
    ):
        raise ValueError(
            "Isolated view receipt does not bind exact native original-element evidence"
        )
    _sha(receipt.get("implementation_sha256"))
    _sha(receipt.get("alpha_png_sha256"))
    resources = receipt.get("resource_identity")
    if not isinstance(resources, dict) or not isinstance(resources.get("hashes"), dict):
        raise ValueError("Native isolated view must bind trusted operator resources")
    _sha(resources.get("resource_digest"))
    if receipt["implementation_sha256"] != _native_digest(
        {"files": receipt.get("implementation_hashes"), "runtime": receipt.get("runtime")}
    ):
        raise ValueError("Native isolation implementation identity is inconsistent")
    return ref, None


def build_adapter_request(
    source,
    source_render,
    context_images,
    *,
    purpose_views=2,
    isolated_images=None,
    isolated_receipts=None,
):
    """Build the explicitly versioned prompt; IDs/paths never enter features.

    Default two-view bytes are unchanged. Three views require actual SOURCE,
    CONTEXT and native ISOLATED/checker evidence; unavailable elements abstain.
    Image byte verification and actual embedding remain the worker's job.
    """
    if dict((alias, purpose) for purpose, alias in ALIASES.items()) != _CANONICAL:
        raise ValueError(
            "The trained purpose taxonomy changed; an explicit bridge update is required"
        )
    ids = _source(source)
    views = _purpose_views(purpose_views)
    render = _ref(source_render)
    if not isinstance(context_images, dict) or set(context_images) - set(ids):
        raise ValueError("Context images may reference original source IDs only")
    if views == 2 and (isolated_images is not None or isolated_receipts is not None):
        raise ValueError("Two-view purpose input cannot silently consume isolated evidence")
    if views == 3 and (
        not isinstance(isolated_images, dict)
        or not isinstance(isolated_receipts, dict)
        or set(isolated_images) - set(ids)
        or set(isolated_receipts) - set(ids)
    ):
        raise ValueError("Three-view purpose input requires known-ID native isolated evidence maps")
    order = sorted(range(len(ids)), key=lambda i: source["elements"][i].get("z_index", 0))
    rank = {index: position for position, index in enumerate(order)}
    request = {
        "protocol": WORKER_PROTOCOL,
        "task": "purpose",
        "purpose_views": views,
        "records": [],
    }
    bindings, absent = [], []
    for index, element in enumerate(source["elements"]):
        key = element["id"]
        if key not in context_images:
            absent.append({"element_id": key, "reason": "missing_context_image"})
            continue
        crop = _ref(context_images[key])
        isolated = None
        if views == 3:
            isolated, reason = _isolated_ref(
                element, isolated_images.get(key), isolated_receipts.get(key)
            )
            if reason:
                absent.append({"element_id": key, "reason": reason})
                continue
        context = {
            "type": element["type"],
            "bbox_normalized": [
                element[k] / source["canvas"][axis]
                for k, axis in (
                    ("x", "width"),
                    ("y", "height"),
                    ("width", "width"),
                    ("height", "height"),
                )
            ],
            "text_data": str(element.get("content", ""))[:240],
            "paint_order_index": rank[index],
            "element_count": len(ids),
        }
        prompt = build_purpose_prompt(context, purpose_views=views)
        record_id = f"element-{index:04d}"
        request["records"].append(
            {
                "record_id": record_id,
                "prompt": prompt,
                "images": [render, crop] + ([isolated] if isolated is not None else []),
            }
        )
        bindings.append(
            {
                "record_id": record_id,
                "element_id": key,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_image_sha256": [render["sha256"], crop["sha256"]]
                + ([isolated["sha256"]] if isolated is not None else []),
            }
        )
    bundle = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_digest": source_digest(source),
        "source_render_sha256": render["sha256"],
        "known_element_ids": ids,
        "request": request,
        "records": bindings,
        "initial_abstentions": absent,
        "classes": list(CLASSES),
        "request_sha256": hashlib.sha256((canonical(request) + "\n").encode()).hexdigest(),
    }
    if views == 3:
        bundle.update(
            purpose_views=3,
            context_images={key: _ref(value) for key, value in context_images.items()},
            isolated_images={key: _ref(value) for key, value in isolated_images.items()},
            isolated_receipts=deepcopy(isolated_receipts),
        )
    bundle["binding_digest"] = digest(bundle)
    return bundle


def request_bytes(bundle):
    """Write these exact bytes to the worker request file before preparing it."""
    encoded = (canonical(bundle["request"]) + "\n").encode()
    if hashlib.sha256(encoded).hexdigest() != bundle["request_sha256"]:
        raise ValueError("Adapter request changed after binding")
    return encoded


def _binding(source, source_render, bundle):
    if (
        not isinstance(bundle, dict)
        or bundle.get("protocol") != PROTOCOL
        or bundle.get("binding_digest")
        != digest({k: v for k, v in bundle.items() if k != "binding_digest"})
        or bundle.get("source_digest") != source_digest(source)
        or bundle.get("source_render_sha256") != _ref(source_render)["sha256"]
        or bundle.get("known_element_ids") != _source(source)
    ):
        raise ValueError("Adapter analysis binding is stale or belongs to different source pixels")
    request = bundle.get("request")
    if not isinstance(request, dict):
        raise ValueError("Adapter binding has no versioned worker request")
    views = _purpose_views(request.get("purpose_views", 2))
    bindings, records = bundle.get("records"), request.get("records")
    if (
        not isinstance(bindings, list)
        or not isinstance(records, list)
        or len(bindings) != len(records)
        or len(records) > MAX_ELEMENTS
        or any(not isinstance(row, dict) for row in bindings)
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("images"), list)
            or len(row["images"]) != views
            for row in records
        )
    ):
        raise ValueError("Adapter binding requires exact record and versioned image counts")
    refs = {
        row["element_id"]: bundle["request"]["records"][index]["images"][1]
        for index, row in enumerate(bundle["records"])
    }
    kwargs = {}
    if views == 3:
        refs = bundle["context_images"]
        kwargs = {
            "purpose_views": 3,
            "isolated_images": bundle["isolated_images"],
            "isolated_receipts": bundle["isolated_receipts"],
        }
    if build_adapter_request(source, source_render, refs, **kwargs) != bundle:
        raise ValueError("Adapter prompt/grounding inputs differ from the original source")
    request_bytes(bundle)
    if not bundle["records"]:
        raise ValueError("No grounded image inputs can attest classifier execution")


def _identity(value):
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        raise ValueError(
            "Prepared adapter identity must pin base, adapter, recipe, dataset and worker code"
        )
    base = value["base"]
    if not isinstance(base, dict) or set(base) != {"repo", "revision", "model_sha256"}:
        raise ValueError("Adapter base identity requires repo, pinned revision and weight SHA")
    if (
        not isinstance(base["repo"], str)
        or not base["repo"]
        or not isinstance(base["revision"], str)
    ):
        raise ValueError("Adapter base repository/revision are missing")
    if not re.fullmatch(r"[0-9a-f]{40}", base["revision"]):
        raise ValueError("Adapter base revision must be an immutable commit")
    _sha(base["model_sha256"])
    for key in _IDENTITY_KEYS - {"base", "code_sha256"}:
        _sha(value[key])
    code = value["code_sha256"]
    if not isinstance(code, dict) or not 1 <= len(code) <= 32 or "worker.py" not in code:
        raise ValueError("Adapter execution must pin the actual worker code")
    for name, checksum in code.items():
        if not isinstance(name, str) or "/" in name or "\\" in name or len(name) > 200:
            raise ValueError("Adapter code identity must use bounded file names")
        _sha(checksum)
    return deepcopy(value)


def _parse_predictions(encoded):
    if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_PREDICTION_BYTES:
        raise ValueError("Adapter predictions must be bounded exact file bytes")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate adapter prediction JSON key")
            result[key] = value
        return result

    value = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda _v: (_ for _ in ()).throw(ValueError("Nonfinite prediction")),
    )
    if (
        not isinstance(value, dict)
        or set(value) != {"scoring", "metrics", "predictions"}
        or value["scoring"] != SCORING
        or value["metrics"] is not None
        or not isinstance(value["predictions"], list)
    ):
        raise ValueError("Only actual label-free closed-logit adapter outputs are accepted")
    return value


def _base_prediction_binding(receipt, encoded):
    """Verify an optional actual base score sidecar, without using its scores."""
    expected = receipt.get("base_prediction_file_sha256")
    if expected is None:
        if encoded is not None:
            raise ValueError("Base prediction bytes were supplied without a receipt hash")
        return False
    _sha(expected)
    if (
        not isinstance(encoded, bytes)
        or not 0 < len(encoded) <= MAX_PREDICTION_BYTES
        or hashlib.sha256(encoded).hexdigest() != expected
    ):
        raise ValueError("Actual base prediction file bytes differ from their receipt hash")
    return True


def _prediction(row, binding, *, purpose_views=2):
    required = {
        "record_id",
        "prediction",
        "closed_scores",
        "closed_probabilities",
        "raw_next_token_id",
        "raw_next_token",
        "raw_next_token_is_allowed",
        "input_token_count",
        "image_grid_thw",
        "input_image_sha256",
    }
    if (
        not isinstance(row, dict)
        or not required <= row.keys()
        or row.keys() - required - {"label", "family_id", "control"}
        or row.get("label") is not None
    ):
        raise ValueError("Purpose predictions contain missing fields or sealed answers")
    if (
        row["record_id"] != binding["record_id"]
        or row["input_image_sha256"] != binding["input_image_sha256"]
    ):
        raise ValueError("Purpose prediction does not bind its ordered source/context images")
    if type(row["input_token_count"]) is not int or not 0 < row["input_token_count"] <= 100_000:
        raise ValueError("Purpose prediction has no valid embedded token count")
    grid = row["image_grid_thw"]
    if (
        not isinstance(grid, list)
        or len(grid) != _purpose_views(purpose_views)
        or any(
            not isinstance(item, list)
            or len(item) != 3
            or any(type(v) is not int or v <= 0 for v in item)
            for item in grid
        )
    ):
        raise ValueError("Purpose prediction must attest the exact versioned image embeddings")
    scores, probabilities = row["closed_scores"], row["closed_probabilities"]
    for values in (scores, probabilities):
        if (
            not isinstance(values, dict)
            or set(values) != set(CLASSES)
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in values.values())
        ):
            raise ValueError(
                "Purpose score taxonomy must exactly match the trained ten-word aliases"
            )
    if (
        any(not 0 <= v <= 1 for v in probabilities.values())
        or abs(sum(probabilities.values()) - 1) > 1e-4
    ):
        raise ValueError("Closed purpose probabilities must form a finite distribution")
    highest = max(scores.values())
    denominator = sum(math.exp(value - highest) for value in scores.values())
    if any(
        abs(probabilities[key] - math.exp(scores[key] - highest) / denominator) > 5e-4
        for key in CLASSES
    ):
        raise ValueError("Purpose probabilities disagree with the actual closed logits")
    prediction = row["prediction"]
    if (
        not isinstance(prediction, str)
        or prediction not in CLASSES
        or scores[prediction] < highest - 1e-5
    ):
        raise ValueError("Purpose prediction is not its declared closed-logit maximum")
    if (
        type(row["raw_next_token_id"]) is not int
        or row["raw_next_token_id"] < 0
        or not isinstance(row["raw_next_token"], str)
        or len(row["raw_next_token"]) > 200
        or type(row["raw_next_token_is_allowed"]) is not bool
    ):
        raise ValueError("Full-vocabulary next-token evidence is missing")
    p = probabilities[prediction]
    margin = p - max(v for key, v in probabilities.items() if key != prediction)
    reason = (
        "raw_token_outside_class_set"
        if not row["raw_next_token_is_allowed"]
        else "raw_token_disagrees_with_closed_choice"
        if row["raw_next_token"].strip() != prediction
        else "classifier_unknown"
        if prediction == "unknown"
        else "low_closed_class_probability"
        if p < MIN_PROBABILITY
        else "low_closed_class_margin"
        if margin < MIN_MARGIN
        else None
    )
    return {
        "element_id": binding["element_id"],
        "record_id": binding["record_id"],
        "alias": prediction,
        "proposed_purpose": None if reason else _CANONICAL[prediction],
        "normalized_candidate_score": p,
        "candidate_score_margin": margin,
        "abstention_reason": reason,
        "score_kind": "normalized_closed_class_softmax_not_calibrated",
        "prompt_sha256": binding["prompt_sha256"],
        "input_image_sha256": deepcopy(binding["input_image_sha256"]),
    }


def complete_adapter_analysis(
    source,
    source_render,
    bundle,
    prediction_bytes,
    inference_receipt,
    *,
    expected_identity,
    mode="off",
    base_prediction_bytes=None,
):
    """Return the graph analyzer result; bad/cold scores abstain, never organize.

    ``expected_identity`` comes from the operator's prepared recipe BEFORE the
    worker runs, not from model output. Only diagnostic opt-in can emit proposals
    today. No deployment qualification path is implemented or implied.
    """
    empty = bind_analysis(
        {"observations": [], "abstain": True},
        source_digest=source_digest(source),
        source_render_sha256=_ref(source_render)["sha256"],
        known_element_ids=_source(source),
    )
    result = {
        "role": "analyzer",
        "executed": False,
        "valid": False,
        "abstained": True,
        "analysis": empty,
        "trace": {
            "invoked": True,
            "receipt_valid": False,
            "deployment_enabled": False,
            "qualified": False,
            "experimental_only": True,
            "organizer": False,
            "layout_authority": False,
            "crop_authority": False,
        },
    }
    try:
        result["trace"]["execution_attested"] = (
            isinstance(inference_receipt, dict)
            and inference_receipt.get("execution_success") is True
        )
        if mode not in {"off", "diagnostic"}:
            raise ValueError("Adapter analyzer supports off or explicit diagnostic mode only")
        _binding(source, source_render, bundle)
        expected = _identity(expected_identity)
        receipt_keys = _IDENTITY_KEYS | {
            "execution_success",
            "task",
            "request_sha256",
            "prediction_file_sha256",
            "quality_approved",
            "experimental_only",
        }
        if (
            not isinstance(inference_receipt, dict)
            or not receipt_keys <= inference_receipt.keys()
            or inference_receipt.keys() - receipt_keys - {"base_prediction_file_sha256"}
            or inference_receipt.get("execution_success") is not True
            or inference_receipt.get("task") != "purpose"
            or inference_receipt.get("quality_approved") is not False
            or inference_receipt.get("experimental_only") is not True
        ):
            raise ValueError(
                "Adapter execution did not attest the experimental label-free purpose task"
            )
        if any(inference_receipt[key] != value for key, value in expected.items()):
            raise ValueError(
                "Adapter execution differs from its pinned base/adapter/prepared recipe"
            )
        if (
            inference_receipt["request_sha256"] != bundle["request_sha256"]
            or inference_receipt["prediction_file_sha256"]
            != hashlib.sha256(prediction_bytes).hexdigest()
        ):
            raise ValueError("Adapter execution request/prediction file bytes changed")
        base_verified = _base_prediction_binding(inference_receipt, base_prediction_bytes)
        result["executed"] = True
        result["trace"]["receipt_valid"] = True
        payload = _parse_predictions(prediction_bytes)
        bindings = {row["record_id"]: row for row in bundle["records"]}
        rows = payload["predictions"]
        if (
            any(not isinstance(row, dict) for row in rows)
            or len(rows) != len(bindings)
            or len({r.get("record_id") for r in rows}) != len(rows)
        ):
            raise ValueError("Adapter predictions must cover each grounded request exactly once")
        if {row.get("record_id") for row in rows} != set(bindings):
            raise ValueError("Adapter returned an unknown or missing record identity")
        views = _purpose_views(bundle["request"].get("purpose_views", 2))
        predictions = [
            _prediction(row, bindings[row["record_id"]], purpose_views=views) for row in rows
        ]
        result.update(executed=True, valid=True)
        result["trace"].update(
            receipt_valid=True,
            receipt=deepcopy(inference_receipt),
            binding_digest=bundle["binding_digest"],
            taxonomy="trained_ten_word_purpose_aliases_v1",
            adapter_is_organizer=False,
            confidence_calibrated=False,
            diagnostic_candidate_score_screen=MIN_PROBABILITY,
            diagnostic_margin_screen=MIN_MARGIN,
            source_element_ids_are_features=False,
            element_predictions=predictions,
            initial_abstentions=deepcopy(bundle["initial_abstentions"]),
            model_visual_input_count=len(rows) * views,
            purpose_views=views,
            base_prediction_file_verified=base_verified,
        )
        if views == 3:
            result["trace"]["native_isolation"] = {
                row["element_id"]: {
                    key: deepcopy(bundle["isolated_receipts"][row["element_id"]][key])
                    for key in (
                        "receipt_digest",
                        "implementation_sha256",
                        "source_element_sha256",
                        "checker_png_sha256",
                        "resource_identity",
                    )
                }
                for row in bundle["records"]
            }
        observations = []
        accepted = [row for row in predictions if row["abstention_reason"] is None]
        if mode == "diagnostic":
            grouped = defaultdict(list)
            for row in sorted(accepted, key=lambda row: -row["normalized_candidate_score"]):
                grouped[row["proposed_purpose"]].append(row["element_id"])
            for purpose, ids in grouped.items():
                for index in range(0, len(ids), 16):
                    if len(observations) == 8:
                        break
                    observations.append(
                        {
                            "kind": "subject",
                            "element_ids": ids[index : index + 16],
                            "evidence": (
                                f"Independent adapter role proposal: {purpose}; "
                                "uncalibrated closed-class scores, "
                                "not a relationship or layout instruction."
                            ),
                        }
                    )
        result["analysis"] = bind_analysis(
            {"observations": observations, "abstain": not observations},
            source_digest=bundle["source_digest"],
            source_render_sha256=bundle["source_render_sha256"],
            known_element_ids=bundle["known_element_ids"],
        )
        result["abstained"] = not observations
        published_ids = {key for row in observations for key in row["element_ids"]}
        result["trace"].update(
            mode=mode,
            accepted_role_predictions=len(accepted),
            published_observations=len(observations),
            source_analysis_digest=digest(result["analysis"]),
            source_analysis_digest_protocol="canonical-sorted-compact-ascii-v1",
            published_element_ids=[
                key for key in bundle["known_element_ids"] if key in published_ids
            ],
            observation_budget_omissions=[
                row["element_id"]
                for row in accepted
                if mode == "diagnostic" and row["element_id"] not in published_ids
            ],
            fallback_reason="bridge_disabled"
            if mode == "off"
            else "all_predictions_abstained"
            if not observations
            else None,
        )
    except (ValueError, OSError, KeyError, TypeError, UnicodeError, RecursionError) as exc:
        result["trace"]["fallback_reason"] = str(exc)[:600]
    return result
