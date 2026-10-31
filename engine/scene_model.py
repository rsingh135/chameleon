"""Source-bound semantic suggestions can restrict crops, never authorize new ones.

No model library is imported by the engine. Optional local providers return the
same strict JSON protocol as the scene harness; their outputs are untrusted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy

MAX_MODEL_BYTES = 256 * 1024
MAX_MODEL_REGIONS = 256
_SHA = re.compile(r"[0-9a-f]{64}\Z")
MODEL_ROLES = {"background", "product", "logo", "foreground", "unknown"}
REGION_KINDS = {"face", "person", "product", "logo", "other"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _closed(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise ValueError("Model response has missing or unknown fields")


def _confidence(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Confidence must be a finite number from 0 to 1")


def validate_model_response(response, profile):
    """Validate suggestions only; this function cannot grant cover permission."""
    if len(_canonical(response).encode()) > MAX_MODEL_BYTES:
        raise ValueError("Model response exceeds its byte limit")
    _closed(response, {"schema_version", "source_digest", "scene_digest", "images"})
    if type(response["schema_version"]) is not int or response["schema_version"] != 1:
        raise ValueError("Unsupported model response schema")
    if any(response[key] != profile[key] for key in ("source_digest", "scene_digest")):
        raise ValueError("Model response belongs to another source or scene")
    known = {image["element_id"]: image for image in profile["images"]}
    images = response["images"]
    if not isinstance(images, list) or len(images) > min(200, len(known)):
        raise ValueError("Model image suggestions exceed the known image set")
    seen, count = set(), 0
    for image in images:
        _closed(
            image,
            {"element_id", "asset_sha256", "proposed_role", "confidence"},
            {"protected_regions", "rationale"},
        )
        identity = image["element_id"]
        if not isinstance(identity, str) or identity not in known or identity in seen:
            raise ValueError("Unknown or duplicate model element ID")
        seen.add(identity)
        asset = image["asset_sha256"]
        if (
            not isinstance(asset, str)
            or not _SHA.fullmatch(asset)
            or asset != known[identity]["asset_sha256"]
        ):
            raise ValueError("Model image bytes do not match the bound source asset")
        if not isinstance(image["proposed_role"], str) or image["proposed_role"] not in MODEL_ROLES:
            raise ValueError("Unknown proposed image role")
        _confidence(image["confidence"])
        if "rationale" in image and (
            not isinstance(image["rationale"], str) or len(image["rationale"]) > 1000
        ):
            raise ValueError("Model rationale must be text of at most 1000 characters")
        regions = image.get("protected_regions", [])
        if not isinstance(regions, list) or len(regions) > 32:
            raise ValueError("An image may have at most 32 proposed protected regions")
        count += len(regions)
        if count > MAX_MODEL_REGIONS:
            raise ValueError("Model protected regions exceed their total limit")
        for region in regions:
            _closed(region, {"kind", "box", "confidence"})
            if not isinstance(region["kind"], str) or region["kind"] not in REGION_KINDS:
                raise ValueError("Unknown protected region kind")
            _confidence(region["confidence"])
            box = region["box"]
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in box
                )
            ):
                raise ValueError("Protected region must contain four finite normalized coordinates")
            x, y, width, height = box
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
                raise ValueError("Protected region must stay inside the image's normalized bounds")
    return deepcopy(response)


def constrain_candidates(source, candidates, response, profile):
    """Veto introduced cropping of confidently classified products and logos.

    Background predictions only corroborate the physical scene policy. Unknown
    and low-confidence roles abstain. CLIP does not localize objects, so proposed
    regions remain review evidence and never claim measured crop protection.
    Authored source crops stay intact; product placement remains the existing
    source-derived focal/group policy's responsibility.
    """
    proposal = validate_model_response(response, profile)
    threshold = 0.8
    protected = {
        row["element_id"]
        for row in proposal["images"]
        if row["proposed_role"] in {"product", "logo"} and row["confidence"] >= threshold
    }
    originals = {e["id"]: e for e in source["elements"]}

    def visible_interval(element, canvas, position, extent):
        return (
            max(0.0, min(1.0, -element[position] / element[extent])),
            max(0.0, min(1.0, (canvas[extent] - element[position]) / element[extent])),
        )

    def retained(candidate):
        images = {e["id"]: e for e in candidate["elements"]}
        for identity in protected:
            before, after = originals[identity], images.get(identity)
            if after is None:
                return False
            if after.get("fit") != before.get("fit"):
                return False
            if after.get("image_position") != before.get("image_position"):
                return False
            # A uniform image-frame scale retains the source crop; translation
            # is allowed so source-derived clusters can move with the product.
            if not math.isclose(
                before["width"] * after["height"],
                after["width"] * before["height"],
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                return False
            for position, extent in (("x", "width"), ("y", "height")):
                old_start, old_end = visible_interval(before, source["canvas"], position, extent)
                new_start, new_end = visible_interval(after, candidate["canvas"], position, extent)
                if new_start > old_start + 1e-6 or new_end < old_end - 1e-6:
                    return False
        return True

    remaining = [candidate for candidate in candidates if retained(candidate)]
    return remaining, {
        "policy": "protect",
        "proposal_digest": hashlib.sha256(_canonical(proposal).encode()).hexdigest(),
        "confidence_threshold": threshold,
        "images_reviewed": len(proposal["images"]),
        "protected_element_ids": sorted(protected),
        "candidates_vetoed": len(candidates) - len(remaining),
        "background_permissions_granted": 0,
        "protected_regions_applied": 0,
        "quality_validated": False,
    }
