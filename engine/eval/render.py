"""In-memory rendering through the official renderer, plus pixel statistics.

`render_image` is the same pipeline as `renderer.converter.convert.
render_design_to_png`, returning a PIL image instead of writing a file.
`render_plate` renders the design with all text removed: what sits *under*
the text, which is what readability depends on.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from engine.eval.geometry import Box
from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello


def render_image(
    design: dict,
    assets_dir: str | Path,
    fonts_dir: str | Path | None = None,
    max_size: int | None = None,
) -> Image.Image:
    assets_dir = str(assets_dir)
    fonts = str(fonts_dir) if fonts_dir is not None else str(Path(assets_dir).parent / "fonts")
    renderer = _get_renderer(fonts)
    example, _warnings = convert_chameleon_to_crello(design, renderer.font_manager, assets_dir)
    png = renderer.render(example, max_size=max_size, format="png")
    return Image.open(io.BytesIO(png)).convert("RGBA")


def render_plate(
    design: dict,
    assets_dir: str | Path,
    fonts_dir: str | Path | None = None,
    max_size: int | None = None,
) -> Image.Image:
    """Render with every text element removed."""
    plate = {
        "canvas": design["canvas"],
        "elements": [el for el in design["elements"] if el["type"] != "text"],
    }
    if not plate["elements"]:
        w, h = design["canvas"]["width"], design["canvas"]["height"]
        return Image.new("RGBA", (w, h), (255, 255, 255, 255))
    return render_image(plate, assets_dir, fonts_dir, max_size)


def region_stats(image: Image.Image, box: Box, design_canvas: dict) -> tuple[float, float]:
    """(mean luminance 0..1, busyness 0..1) of the image under `box`.
    Handles the renderer having scaled the canvas down."""
    sx = image.width / design_canvas["width"]
    sy = image.height / design_canvas["height"]
    x0 = max(0, int(box.x0 * sx))
    y0 = max(0, int(box.y0 * sy))
    x1 = min(image.width, int(round(box.x1 * sx)))
    y1 = min(image.height, int(round(box.y1 * sy)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 1.0, 0.0
    # Composite onto white first: transparent regions read as white paper.
    crop = image.crop((x0, y0, x1, y1))
    backdrop = Image.new("RGBA", crop.size, (255, 255, 255, 255))
    lum = Image.alpha_composite(backdrop, crop).convert("L")
    mean = ImageStat.Stat(lum).mean[0] / 255.0
    lum.thumbnail((128, 128))
    edges = lum.filter(ImageFilter.FIND_EDGES)
    busyness = ImageStat.Stat(edges).mean[0] / 255.0
    return mean, busyness
