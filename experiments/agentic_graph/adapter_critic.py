"""Bind saved-adapter A/B/N scores to an admitted native comparison.

This is a classifier transport, not generated visual reasoning. It imports no
model runtime, grants no geometry authority, and is disabled unless diagnostic
use is explicit. The native executor remains responsible for candidate gates.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path

from experiments.visual_finetune.data import CRITIC_PROMPT

from .adapter_analysis import (
    _IDENTITY_KEYS,
    MAX_PREDICTION_BYTES,
    WORKER_PROTOCOL,
    _identity,
    _parse_predictions,
    _ref,
    _sha,
    _source,
    canonical,
    digest,
    source_digest,
)
from .adapter_analysis import SCORING as SCORING

PROTOCOL = "critic-adapter-verifier-binding-v1"
CLASSES = ("A", "B", "N")
MIN_PROBABILITY = 0.85
MIN_MARGIN = 0.30
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def _menu(source, menu):
    required = {
        "schema_version",
        "task",
        "source_digest",
        "target_canvas",
        "choice_available",
        "plans",
        "menu_digest",
    }
    if (
        not isinstance(menu, dict)
        or set(menu) != required
        or type(menu["schema_version"]) is not int
        or menu["schema_version"] != 1
        or menu["task"] != "critic"
        or menu["source_digest"] != source_digest(source)
        or menu["choice_available"] is not True
        or not isinstance(menu["plans"], list)
        or len(menu["plans"]) != 2
    ):
        raise ValueError("Critic bridge requires two source-bound admitted menu choices")
    target = menu["target_canvas"]
    if (
        not isinstance(target, dict)
        or set(target) != {"width", "height"}
        or any(
            type(target.get(k)) not in {int, float}
            or not math.isfinite(target[k])
            or target[k] <= 0
            for k in ("width", "height")
        )
    ):
        raise ValueError("Critic comparison target must have finite positive dimensions")
    ids = []
    for index, plan in enumerate(menu["plans"]):
        if (
            not isinstance(plan, dict)
            or set(plan) != {"plan_id", "label", "description"}
            or plan["label"] != f"Alternative {index + 1}"
            or plan["description"] != f"Rendered alternative {index + 1}"
            or not isinstance(plan["plan_id"], str)
            or not plan["plan_id"].startswith("candidate-")
        ):
            raise ValueError("Critic menu must contain native neutral candidate identities only")
        _sha(plan["plan_id"][10:])
        ids.append(plan["plan_id"])
    if len(set(ids)) != 2 or menu["menu_digest"] != digest(
        {k: v for k, v in menu.items() if k != "menu_digest"}
    ):
        raise ValueError("Critic menu is stale or contains repeated candidate identities")
    return ids


def build_adapter_critic_request(source, source_render, critic_menu, renders):
    """Freeze the trained SOURCE/A/B prompt; IDs and paths are grounding only.

    ``critic_menu`` and ``renders`` must come from the native admitted snapshot.
    Application revalidates that origin against the actual snapshot; a caller's
    self-authored menu cannot admit new geometry through this bridge.
    """
    _source(source)
    ids = _menu(source, critic_menu)
    if not isinstance(renders, dict) or set(renders) != set(ids):
        raise ValueError("Critic renders must cover exactly the two known candidate IDs")
    images = [_ref(source_render), *[_ref(renders[key]) for key in ids]]
    if images[1]["sha256"] == images[2]["sha256"]:
        raise ValueError("Critic comparison requires distinct rendered choices")
    request = {
        "protocol": WORKER_PROTOCOL,
        "task": "critic",
        "records": [{"record_id": "comparison-0000", "prompt": CRITIC_PROMPT, "images": images}],
    }
    bundle = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_digest": source_digest(source),
        "source_render_sha256": images[0]["sha256"],
        "source_canvas": deepcopy(source["canvas"]),
        "target_canvas": deepcopy(critic_menu["target_canvas"]),
        "critic_menu": deepcopy(critic_menu),
        "choice_ids": {"A": ids[0], "B": ids[1]},
        "classes": list(CLASSES),
        "request": request,
        "record": {
            "record_id": "comparison-0000",
            "prompt_sha256": hashlib.sha256(CRITIC_PROMPT.encode()).hexdigest(),
            "input_image_sha256": [image["sha256"] for image in images],
        },
        "request_sha256": hashlib.sha256((canonical(request) + "\n").encode()).hexdigest(),
    }
    bundle["context_digest"] = digest(bundle)
    return bundle


def request_bytes(bundle):
    """Return the exact label-free JSON bytes to freeze before inference."""
    encoded = (canonical(bundle["request"]) + "\n").encode()
    if hashlib.sha256(encoded).hexdigest() != bundle["request_sha256"]:
        raise ValueError("Critic worker request changed after binding")
    return encoded


def _binding(source, source_render, critic_menu, renders, bundle):
    if build_adapter_critic_request(source, source_render, critic_menu, renders) != bundle:
        raise ValueError("Critic context differs from its frozen source/menu/render binding")
    request_bytes(bundle)


def verify_render_files(bundle):
    """Check actual local bytes and native dimensions before/after scoring.

    No image decoding or file access happens merely by importing this module.
    The worker independently binds the same ordered three image hashes.
    """
    from PIL import Image

    images = bundle["request"]["records"][0]["images"]
    for index, image in enumerate(images):
        path = Path(image["path"])
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
            raise ValueError("Critic native image is missing or exceeds its byte budget")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != image["sha256"]:
            raise ValueError("Critic native source/candidate image bytes changed")
        expected = bundle["source_canvas"] if index == 0 else bundle["target_canvas"]
        try:
            with Image.open(path) as picture:
                if (
                    picture.format != "PNG"
                    or picture.width * picture.height > MAX_IMAGE_PIXELS
                    or picture.size != (expected["width"], expected["height"])
                ):
                    raise ValueError("Critic images must be full native PNG source/target renders")
        except Image.DecompressionBombError as exc:
            raise ValueError("Critic native image exceeds the decode pixel budget") from exc


def _prediction(row, binding):
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
        raise ValueError("Critic scores contain missing fields or sealed answers")
    if (
        row["record_id"] != binding["record_id"]
        or row["input_image_sha256"] != binding["input_image_sha256"]
    ):
        raise ValueError("Critic scores do not bind ordered ORIGINAL/A/B image hashes")
    grid = row["image_grid_thw"]
    if (
        type(row["input_token_count"]) is not int
        or not 0 < row["input_token_count"] <= 100_000
        or not isinstance(grid, list)
        or len(grid) != 3
        or any(
            not isinstance(item, list)
            or len(item) != 3
            or any(type(v) is not int or v <= 0 for v in item)
            for item in grid
        )
    ):
        raise ValueError("Critic score must attest all three actual image embeddings")
    scores, probabilities = row["closed_scores"], row["closed_probabilities"]
    for values in (scores, probabilities):
        if (
            not isinstance(values, dict)
            or set(values) != set(CLASSES)
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in values.values())
        ):
            raise ValueError("Critic score taxonomy must be exactly A/B/N")
    if (
        any(not 0 <= p <= 1 for p in probabilities.values())
        or abs(sum(probabilities.values()) - 1) > 1e-4
    ):
        raise ValueError("Critic normalized scores must form a finite distribution")
    highest = max(scores.values())
    denominator = sum(math.exp(v - highest) for v in scores.values())
    if any(
        abs(probabilities[key] - math.exp(scores[key] - highest) / denominator) > 5e-4
        for key in CLASSES
    ):
        raise ValueError("Critic normalized scores disagree with the actual class logits")
    prediction = row["prediction"]
    if (
        not isinstance(prediction, str)
        or prediction not in CLASSES
        or scores[prediction] < highest - 1e-5
    ):
        raise ValueError("Critic choice is not the declared closed-logit maximum")
    if (
        type(row["raw_next_token_id"]) is not int
        or row["raw_next_token_id"] < 0
        or not isinstance(row["raw_next_token"], str)
        or len(row["raw_next_token"]) > 200
        or type(row["raw_next_token_is_allowed"]) is not bool
    ):
        raise ValueError("Critic full-vocabulary next-token evidence is missing")
    p = probabilities[prediction]
    margin = p - max(value for key, value in probabilities.items() if key != prediction)
    reason = (
        "raw_token_outside_class_set"
        if not row["raw_next_token_is_allowed"]
        else "raw_token_disagrees_with_closed_choice"
        if row["raw_next_token"].strip() != prediction
        else "classifier_neither"
        if prediction == "N"
        else "low_closed_class_probability"
        if p < MIN_PROBABILITY
        else "low_closed_class_margin"
        if margin < MIN_MARGIN
        else None
    )
    return {
        "class": prediction,
        "normalized_candidate_score": p,
        "candidate_score_margin": margin,
        "score_kind": "normalized_closed_class_softmax_not_calibrated",
        "abstention_reason": reason,
        "input_image_sha256": deepcopy(binding["input_image_sha256"]),
        "prompt_sha256": binding["prompt_sha256"],
    }


def complete_adapter_critic(
    source,
    source_render,
    critic_menu,
    renders,
    bundle,
    prediction_bytes,
    inference_receipt,
    *,
    expected_identity,
    mode="off",
):
    """Map a verified A/B score to a known ID; N/weak/invalid emits no decision.

    ``expected_identity`` is frozen by trusted preparation before inference.
    ``prediction_valid`` is distinct from graph ``valid``: the current graph
    requires a concrete decision for a valid verifier and cannot execute N.
    """
    result = {
        "role": "verifier",
        "executed": False,
        "valid": False,
        "prediction_valid": False,
        "abstained": True,
        "decision": None,
        "trace": {
            "invoked": True,
            "receipt_valid": False,
            "deployment_enabled": False,
            "qualified": False,
            "experimental_only": True,
            "mode": mode,
            "generated_reasoning": False,
            "confidence_calibrated": False,
            "geometry_authority": False,
            "admission_override": False,
        },
    }
    try:
        result["trace"]["execution_attested"] = (
            isinstance(inference_receipt, dict)
            and inference_receipt.get("execution_success") is True
        )
        if mode not in {"off", "diagnostic"}:
            raise ValueError("Unqualified critic supports off or explicit diagnostic mode only")
        _binding(source, source_render, critic_menu, renders, bundle)
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
            or set(inference_receipt) != receipt_keys
            or inference_receipt.get("execution_success") is not True
            or inference_receipt.get("task") != "critic"
            or inference_receipt.get("quality_approved") is not False
            or inference_receipt.get("experimental_only") is not True
            or any(inference_receipt[key] != value for key, value in expected.items())
        ):
            raise ValueError("Critic execution differs from the pinned experimental adapter task")
        if (
            not isinstance(prediction_bytes, bytes)
            or not 0 < len(prediction_bytes) <= MAX_PREDICTION_BYTES
            or inference_receipt["request_sha256"] != bundle["request_sha256"]
            or inference_receipt["prediction_file_sha256"]
            != hashlib.sha256(prediction_bytes).hexdigest()
        ):
            raise ValueError("Critic inference request/prediction bytes changed")
        result["executed"] = True
        result["trace"].update(receipt_valid=True, receipt=deepcopy(inference_receipt))
        payload = _parse_predictions(prediction_bytes)
        if len(payload["predictions"]) != 1:
            raise ValueError("Critic must return exactly its one bound A/B comparison")
        prediction = _prediction(payload["predictions"][0], bundle["record"])
        result["prediction_valid"] = True
        result["trace"].update(
            context_digest=bundle["context_digest"],
            request_sha256=bundle["request_sha256"],
            critic_menu_digest=critic_menu["menu_digest"],
            source_digest=bundle["source_digest"],
            target_canvas=deepcopy(bundle["target_canvas"]),
            choice_ids=deepcopy(bundle["choice_ids"]),
            prediction=prediction,
            model_visual_input_count=3,
            diagnostic_candidate_score_screen=MIN_PROBABILITY,
            diagnostic_margin_screen=MIN_MARGIN,
        )
        reason = prediction["abstention_reason"] or (
            "adapter_critic_disabled" if mode == "off" else None
        )
        if reason:
            result["trace"]["fallback_reason"] = reason
        else:
            decision = {
                "schema_version": 1,
                "menu_digest": critic_menu["menu_digest"],
                "plan_id": bundle["choice_ids"][prediction["class"]],
            }
            choice = {
                "context_digest": bundle["context_digest"],
                "prediction_file_sha256": inference_receipt["prediction_file_sha256"],
                "adapter_sha256": expected["adapter_sha256"],
                "decision": decision,
            }
            result.update(valid=True, abstained=False, decision=decision)
            result["trace"].update(choice_digest=digest(choice), choice_binding=choice)
    except (ValueError, OSError, KeyError, TypeError, UnicodeError, RecursionError) as exc:
        result["trace"]["fallback_reason"] = str(exc)[:600]
    return result


def complete_and_apply_adapter_critic(
    source,
    target,
    snapshot,
    source_render,
    critic_menu,
    renders,
    bundle,
    prediction_bytes,
    inference_receipt,
    *,
    expected_identity,
    mode="off",
    assets_dir=None,
    fonts_dir=None,
):
    """Revalidate actual score evidence and apply through the native executor.

    No fallback design is invented. Callers retain their existing deterministic
    winner when ``applied`` is None. A choice digest is joined to the resulting
    full-design digest only after current native admission/contract checks pass.
    """
    from engine.visual_selection import apply_visual_selection

    result = complete_adapter_critic(
        source,
        source_render,
        critic_menu,
        renders,
        bundle,
        prediction_bytes,
        inference_receipt,
        expected_identity=expected_identity,
        mode=mode,
    )
    output = {"verifier": result, "applied": None}
    if not result["valid"]:
        return output
    try:
        verify_render_files(bundle)
        applied = apply_visual_selection(
            source,
            target,
            snapshot,
            result["decision"],
            assets_dir=assets_dir,
            fonts_dir=fonts_dir,
            critic_menu=critic_menu,
        )
        actual_digest = digest(applied["design"])
        if result["decision"]["plan_id"] != "candidate-" + actual_digest:
            raise ValueError("Native executor returned geometry different from the scored choice")
        processing = {
            "context_digest": bundle["context_digest"],
            "choice_digest": result["trace"]["choice_digest"],
            "selected_candidate_id": result["decision"]["plan_id"],
            "applied_design_sha256": actual_digest,
            "native_trace": deepcopy(applied["trace"]),
        }
        processing["processing_digest"] = digest(processing)
        applied["trace"]["trained_critic"] = processing
        output["applied"] = applied
        result["trace"]["native_execution"] = processing
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result.update(valid=False, abstained=True, decision=None)
        result["trace"]["fallback_reason"] = str(exc)[:600]
    return output
