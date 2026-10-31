"""Label-free native source context crops matching purpose-context-v2 export.

Only the supplied original PNG and editable source geometry are read. No purpose
export, annotations, semantic rules, target layout or source asset path is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .adapter_analysis import _ref, _source, source_digest

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
