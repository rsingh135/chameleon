"""Compose a diagnostic contrast matte from retained native RGBA bytes only.

This opt-in preprocessing policy does not render an element, read source assets,
run a model, or change the existing checker policies. Original RGB and alpha are
preserved during compositing; the resulting opaque PNG is a diagnostic view.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import zlib
from io import BytesIO
from pathlib import Path

from PIL import Image, PngImagePlugin
from PIL import __version__ as PILLOW_VERSION

CONTRAST_MATTE_V1 = "isolated-alpha-contrast-matte-v1"
MAX_AXIS = 512
MAX_PNG_BYTES = 4 * 1024 * 1024
MAX_CORE_SAMPLES = 4096
MATTE_PREFERENCE = (117, 255, 0)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LINEAR = tuple(
    value / 255 / 12.92 if value / 255 <= 0.04045 else ((value / 255 + 0.055) / 1.055) ** 2.4
    for value in range(256)
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity():
    return {
        "implementation_sha256": _file_sha(__file__),
        "pillow": {
            "version": PILLOW_VERSION,
            "image_module_sha256": _file_sha(Image.__file__),
            "png_module_sha256": _file_sha(PngImagePlugin.__file__),
            "core_binary_sha256": _file_sha(Image.core.__file__),
        },
        "python": sys.version,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
    }


def _quantile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _luminance(red, green, blue):
    return 0.2126 * _LINEAR[red] + 0.7152 * _LINEAR[green] + 0.0722 * _LINEAR[blue]


def _statistics(image, positions, matte):
    if not positions:
        return {
            key: None for key in ("p10_contrast", "median_contrast", "min_contrast", "max_contrast")
        }
    pixels = image.tobytes()
    background = _luminance(matte, matte, matte)
    contrasts = []
    for position in positions:
        offset = position * 3
        foreground = _luminance(*pixels[offset : offset + 3])
        contrasts.append(
            (max(foreground, background) + 0.05) / (min(foreground, background) + 0.05)
        )
    contrasts.sort()
    return {
        "p10_contrast": _quantile(contrasts, 0.1),
        "median_contrast": _quantile(contrasts, 0.5),
        "min_contrast": contrasts[0],
        "max_contrast": contrasts[-1],
    }


def compose_contrast_view(
    alpha_png: bytes, *, expected_alpha_sha256=None, policy=CONTRAST_MATTE_V1
):
    """Return ``{png: bytes, receipt: dict}`` without changing retained alpha bytes.

    Core pixels have positive alpha and alpha >= 75% of the image's maximum.
    At most 4096 positions are sampled uniformly by row-major core rank, including
    both endpoints. Percentiles use linear interpolation at (n - 1) * fraction.
    Exact score ties prefer neutral 117, then white 255, then black 0.
    """
    if policy != CONTRAST_MATTE_V1:
        raise ValueError("Unsupported isolated contrast matte policy")
    if not isinstance(alpha_png, bytes) or not 0 < len(alpha_png) <= MAX_PNG_BYTES:
        raise ValueError("Native alpha PNG must be bounded nonempty immutable bytes")
    input_sha = _sha(alpha_png)
    if expected_alpha_sha256 is not None and (
        not isinstance(expected_alpha_sha256, str)
        or not _SHA.fullmatch(expected_alpha_sha256)
        or expected_alpha_sha256 != input_sha
    ):
        raise ValueError("Native alpha PNG differs from its expected SHA-256")
    identity = _identity()
    try:
        with Image.open(BytesIO(alpha_png)) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGBA"
                or min(image.size) < 1
                or max(image.size) > MAX_AXIS
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError(
                    "Native alpha requires one RGBA PNG raster no larger than 512 per axis"
                )
            image.verify()
        with Image.open(BytesIO(alpha_png)) as image:
            rgba = image.copy()
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("Native alpha PNG could not be decoded exactly") from error
    alpha = rgba.getchannel("A").tobytes()
    maximum_alpha = max(alpha)
    core = [
        index for index, value in enumerate(alpha) if value > 0 and 4 * value >= 3 * maximum_alpha
    ]
    if len(core) > MAX_CORE_SAMPLES:
        samples = [
            core[index * (len(core) - 1) // (MAX_CORE_SAMPLES - 1)]
            for index in range(MAX_CORE_SAMPLES)
        ]
    else:
        samples = core
    candidates, images = [], {}
    for matte in MATTE_PREFERENCE:
        background = Image.new("RGBA", rgba.size, (matte, matte, matte, 255))
        composed = Image.alpha_composite(background, rgba).convert("RGB")
        images[matte] = composed
        candidates.append({"matte_rgb": [matte] * 3, **_statistics(composed, samples, matte)})
    selected = max(candidates, key=lambda row: row["p10_contrast"]) if samples else candidates[0]
    selected_matte = selected["matte_rgb"][0]
    output = BytesIO()
    images[selected_matte].save(output, format="PNG")
    encoded = output.getvalue()
    if _identity() != identity:
        raise ValueError(
            "Contrast composer implementation or Pillow identity changed during composition"
        )
    receipt = {
        "schema_version": 1,
        "protocol": CONTRAST_MATTE_V1,
        "policy": {
            "version": CONTRAST_MATTE_V1,
            "max_axis": MAX_AXIS,
            "max_pixels": MAX_AXIS * MAX_AXIS,
            "max_input_bytes": MAX_PNG_BYTES,
            "candidate_mattes": [0, 117, 255],
            "core_alpha_fraction": 0.75,
            "core_requires_positive_alpha": True,
            "max_core_samples": MAX_CORE_SAMPLES,
            "sampling": "uniform_row_major_core_rank_including_endpoints_floor",
            "score": "p10_actual_composited_rgb_relative_luminance_contrast",
            "percentile": "linear_interpolation_at_(n-1)*fraction",
            "luminance": "sRGB_linearization_0.04045_12.92_2.4_weights_0.2126_0.7152_0.0722",
            "contrast_offset": 0.05,
            "exact_tie_preference": list(MATTE_PREFERENCE),
            "compositing": "PIL.Image.alpha_composite_RGBA_then_convert_RGB",
            "selection_uses_labels_or_context": False,
        },
        "input_alpha_png_sha256": input_sha,
        "input_rgba_pixel_sha256": _sha(rgba.tobytes()),
        "output_png_sha256": _sha(encoded),
        "output_rgb_pixel_sha256": _sha(images[selected_matte].tobytes()),
        "size": list(rgba.size),
        "input_mode": "RGBA",
        "output_mode": "RGB",
        "maximum_alpha": maximum_alpha,
        "paint_empty": not core,
        "core_pixel_count": len(core),
        "core_sample_count": len(samples),
        "sample_positions_sha256": _sha(json.dumps(samples, separators=(",", ":")).encode()),
        "candidates": candidates,
        "selected_matte_rgb": [selected_matte] * 3,
        "selected_p10_contrast": selected["p10_contrast"],
        **identity,
        "input_rgba_unchanged": True,
        "native_redraw": False,
        "model_executed": False,
        "quality_approved": False,
        "geometry_authority": False,
        "promotion_eligible": False,
    }
    receipt["receipt_digest"] = _sha(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    )
    return {"png": encoded, "receipt": receipt}
