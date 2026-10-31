"""Label-free native source evidence with an explicit isolated third view.

The default context path reads original PNG pixels and editable source geometry.
Only explicit three-view preparation opens operator-bound original resources.
No purpose export, annotations, semantic rules or target layout is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from .adapter_analysis import (
    _binding,
    _isolated_ref,
    _purpose_views,
    _ref,
    _source,
    digest,
    source_digest,
)

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_CONTEXT_SIDE = 256


def sha_file(path, maximum=MAX_FILE_BYTES):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Source/context file is missing or exceeds its byte budget")
    result, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            count += len(chunk)
            if count > maximum:
                raise ValueError("Source/context file grew beyond its byte budget")
            result.update(chunk)
    return result.hexdigest()


def context_box(element, size):
    """Exact padding/clamping recipe from tools.semantic_training_data._context_box."""
    width, height = size
    x, y, w, h = (element[key] for key in ("x", "y", "width", "height"))
    padding = max(8, min(width, height) * 0.025)
    left, top = max(0, math.floor(x - padding)), max(0, math.floor(y - padding))
    right, bottom = min(width, math.ceil(x + w + padding)), min(height, math.ceil(y + h + padding))
    if right <= left or bottom <= top:
        center_x, center_y = min(width - 1, max(0, x + w / 2)), min(height - 1, max(0, y + h / 2))
        left, top = max(0, math.floor(center_x - padding)), max(0, math.floor(center_y - padding))
        right, bottom = (
            min(width, math.ceil(center_x + padding)),
            min(height, math.ceil(center_y + padding)),
        )
    return left, top, right, bottom


def build_source_contexts(source, source_render, out, *, expected_source_digest=None):
    """Write unannotated original-pixel contexts into a fresh output directory."""
    from PIL import Image

    _source(source)
    source_pin = source_digest(source)
    if expected_source_digest is not None and source_pin != expected_source_digest:
        raise ValueError("Editable source differs from its expected source identity")
    reference = _ref(source_render)
    source_path = Path(reference["path"]).resolve()
    if sha_file(source_path) != reference["sha256"]:
        raise ValueError("Original source PNG differs from its declared bytes")
    destination = Path(out).resolve()
    if source_path.is_relative_to(destination) or destination.is_symlink():
        raise ValueError("Context output cannot contain or overwrite the original source PNG")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Native context output must be a fresh empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    refs, records = {}, []
    with Image.open(source_path) as image:
        if image.format != "PNG" or image.width * image.height > MAX_PIXELS:
            raise ValueError("Context inference requires a bounded native original PNG")
        if image.size != (source["canvas"]["width"], source["canvas"]["height"]):
            raise ValueError("Source PNG dimensions differ from the editable native canvas")
        image.load()
        for index, element in enumerate(source["elements"]):
            box = context_box(element, image.size)
            visible = (
                element["x"] < image.width
                and element["y"] < image.height
                and element["x"] + element["width"] > 0
                and element["y"] + element["height"] > 0
            )
            record = {
                "element_id": element["id"],
                "context_box_pixels": list(box),
                "source_frame_visible": visible,
                "crop_kind": "native_render_context_not_isolated_asset",
                "labels_or_id_overlays_added": False,
            }
            if not visible:
                record["abstention_reason"] = "element_outside_source_canvas"
                records.append(record)
                continue
            crop = image.crop(box)
            try:
                crop.thumbnail((MAX_CONTEXT_SIDE, MAX_CONTEXT_SIDE), Image.Resampling.LANCZOS)
                path = destination / f"element-{index:04d}.png"
                with path.open("xb") as stream:
                    crop.save(stream, format="PNG")
                refs[element["id"]] = {"path": str(path), "sha256": sha_file(path)}
                record.update(
                    context_size=list(crop.size), context_sha256=refs[element["id"]]["sha256"]
                )
            finally:
                crop.close()
            records.append(record)
    if source_digest(source) != source_pin or sha_file(source_path) != reference["sha256"]:
        raise ValueError("Original source changed while preparing native contexts")
    receipt = {
        "schema_version": 1,
        "source_digest": source_pin,
        "source_render_sha256": reference["sha256"],
        "crop_recipe": "purpose-context-v2;padding=max(8,short*0.025);thumbnail=256-LANCZOS",
        "source_ids_are_grounding_only": True,
        "model_executed": False,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "supervision_read": False,
        "records": records,
    }
    (destination / "context-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return {"context_images": refs, "receipt": receipt}


def build_source_views(
    source,
    source_render,
    out,
    *,
    purpose_views=2,
    assets_dir=None,
    fonts_dir=None,
    expected_source_digest=None,
    expected_resource_digest=None,
):
    """Explicitly build native three-view evidence using the source-owner helper.

    Resource paths come only from a trusted operator fixture binding. Call on
    one native worker/thread while owning any surrounding renderer lock. Native
    caches require immutable paths; this function never clears shared caches.
    The default path remains the existing two-view context producer.
    """
    views = _purpose_views(purpose_views)
    if views == 2:
        if assets_dir is not None or fonts_dir is not None or expected_resource_digest is not None:
            raise ValueError(
                "Two-view context preparation does not accept native isolation resources"
            )
        return build_source_contexts(
            source, source_render, out, expected_source_digest=expected_source_digest
        )
    if assets_dir is None or fonts_dir is None:
        raise ValueError("Three-view native evidence requires explicit trusted assets and fonts")
    _source(source)
    if expected_source_digest is not None and expected_source_digest != source_digest(source):
        raise ValueError("Editable source differs from its expected source identity")
    assets, fonts = Path(assets_dir).resolve(), Path(fonts_dir).resolve()
    destination = Path(out).resolve()
    source_path = Path(_ref(source_render)["path"]).resolve()
    if (
        not assets.is_dir()
        or not fonts.is_dir()
        or destination.exists()
        or Path(out).is_symlink()
        or source_path.is_relative_to(destination)
        or any(
            destination.is_relative_to(root) or root.is_relative_to(destination)
            for root in (assets, fonts)
        )
    ):
        raise ValueError(
            "Native source evidence requires fresh output independent of original resources"
        )

    # Lazy imports preserve the existing no-renderer-import two-view path.
    from tools.native_element_evidence import ISOLATION_V1, render_isolated_element
    from viewer.resize_cache import ResourceBindings

    frozen = deepcopy(source)
    bindings = ResourceBindings()
    resources = bindings.bind(source, assets, fonts, allow_missing=True)
    if expected_resource_digest is not None and expected_resource_digest != resources.digest:
        raise ValueError(
            "Native source evidence resources differ from the operator's expected digest"
        )
    destination.mkdir(parents=True, exist_ok=False)
    contexts = build_source_contexts(
        source,
        source_render,
        destination / "contexts",
        expected_source_digest=expected_source_digest,
    )
    isolated, receipts, rows = {}, {}, []
    total = 0
    for index, element in enumerate(source["elements"]):
        key = element["id"]
        if key not in contexts["context_images"]:
            rows.append({"element_id": key, "reason": "missing_source_bound_context"})
            continue
        native = render_isolated_element(
            deepcopy(element), assets_dir=assets, fonts_dir=fonts, policy=ISOLATION_V1
        )
        receipt = native["receipt"]
        receipts[key] = deepcopy(receipt)
        relative = f"element-{index:04d}"
        receipt_path = destination / f"{relative}-native.json"
        with receipt_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
        checker = native.get("checker_png")
        ref = (
            {
                "path": str(destination / f"{relative}-isolated.png"),
                "sha256": hashlib.sha256(checker).hexdigest(),
            }
            if isinstance(checker, bytes)
            else None
        )
        accepted, reason = _isolated_ref(element, ref, receipt)
        row = {"element_id": key, "receipt_digest": receipt["receipt_digest"]}
        if reason:
            rows.append({**row, "reason": reason})
            continue
        alpha = native.get("alpha_png")
        if (
            not isinstance(alpha, bytes)
            or hashlib.sha256(alpha).hexdigest() != receipt["alpha_png_sha256"]
        ):
            raise ValueError("Native isolated alpha sidecar differs from its receipt")
        total += len(checker) + len(alpha)
        if max(len(checker), len(alpha)) > 32 * 1024 * 1024 or total > 256 * 1024 * 1024:
            raise ValueError("Native isolated evidence exceeds its encoded byte budget")
        with Path(accepted["path"]).open("xb") as stream:
            stream.write(checker)
        with (destination / f"{relative}-alpha.png").open("xb") as stream:
            stream.write(alpha)
        isolated[key] = accepted
        rows.append(
            {
                **row,
                "isolated_sha256": accepted["sha256"],
                "alpha_sidecar_sha256": receipt["alpha_png_sha256"],
            }
        )
    if (
        source != frozen
        or sha_file(source_path) != _ref(source_render)["sha256"]
        or bindings.bind(source, assets, fonts, allow_missing=True) != resources
    ):
        raise ValueError("Original source or trusted native resources changed during preparation")
    receipt = {
        "schema_version": 1,
        "purpose_views": 3,
        "ordered_views": ["SOURCE", "CONTEXT", "ISOLATED_TARGET_ON_FIXED_CHECKER"],
        "source_digest": source_digest(source),
        "source_render_sha256": source_render["sha256"],
        "context_receipt": contexts["receipt"],
        "native_policy": ISOLATION_V1,
        "resource_identity": {
            "assets_dir": str(assets),
            "fonts_dir": str(fonts),
            "resource_digest": resources.digest,
            "hashes": resources.hashes(),
        },
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": rows,
        "model_executed": False,
        "supervision_read": False,
        "geometry_authority": False,
    }
    receipt["receipt_digest"] = digest(receipt)
    with (destination / "views-receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    return {
        "context_images": contexts["context_images"],
        "isolated_images": isolated,
        "isolated_receipts": receipts,
        "receipt": receipt,
    }


def verify_source_views(source, source_render, bundle):
    """Recheck actual SOURCE/CONTEXT/ISOLATED bytes around scoring, no repaint."""
    from PIL import Image

    _binding(source, source_render, bundle)
    views = _purpose_views(bundle["request"].get("purpose_views", 2))
    checked = set()
    for row in bundle["request"]["records"]:
        if len(row["images"]) != views:
            raise ValueError("Purpose input image count changed from its explicit view contract")
        for index, ref in enumerate(row["images"]):
            identity = (ref["path"], ref["sha256"], index)
            if identity in checked:
                continue
            if sha_file(ref["path"]) != ref["sha256"]:
                raise ValueError("Source/context/isolated PNG changed around adapter scoring")
            try:
                with Image.open(ref["path"]) as image:
                    if (
                        image.format != "PNG"
                        or image.width * image.height > MAX_PIXELS
                        or index == 0
                        and image.size != tuple(source["canvas"][k] for k in ("width", "height"))
                        or index == 1
                        and max(image.size) > MAX_CONTEXT_SIDE
                        or index == 2
                        and (max(image.size) > 512 or image.mode != "RGB")
                    ):
                        raise ValueError(
                            "Purpose evidence PNG differs from its native view contract"
                        )
            except Image.DecompressionBombError as exc:
                raise ValueError("Purpose evidence exceeds its decode pixel budget") from exc
            checked.add(identity)
