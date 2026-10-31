"""Native-document crop geometry, isolated from the original vendor renderer."""

from __future__ import annotations

import math
from pathlib import Path


def image_position(value=None):
    """Normalized viewport alignment; absent values retain the original center."""
    if value is None:
        return 0.5, 0.5
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in value
        )
    ):
        raise ValueError("image_position must contain two finite fractions between 0 and 1.")
    return tuple(value)


def positioned_image(element, assets_dir):
    """Read original bytes and proportionally crop a native image's viewport."""
    from PIL import Image

    if element.get("fit") != "cover":
        raise ValueError("Positioned native image viewports require proportional cover.")
    root = Path(assets_dir).resolve()
    path = (root / element["asset_id"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Positioned image must use an original local document asset.")
    px, py = image_position(element.get("image_position"))
    width, height = max(1, round(element["width"])), max(1, round(element["height"]))
    if width * height > 32_000_000:
        raise ValueError("Positioned image viewport exceeds the bounded pixel budget.")
    with Image.open(path) as original:
        if original.width * original.height > 32_000_000:
            raise ValueError("Positioned source image exceeds the bounded pixel budget.")
        image = original.convert("RGBA")
    scale = max(width / image.width, height / image.height)
    fitted = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if fitted[0] * fitted[1] > 32_000_000:
        raise ValueError("Positioned image scale exceeds the bounded pixel budget.")
    resized = image.resize(fitted, Image.Resampling.LANCZOS)
    x, y = int((fitted[0] - width) * px), int((fitted[1] - height) * py)
    return resized.crop((x, y, x + width, y + height))
