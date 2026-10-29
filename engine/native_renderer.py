"""Document-only renderer compatibility for genuine fractional text sizes.

The installed cr-renderer truncates fontSize to int when creating text blobs,
although its converter measures using floats. This adapter keeps the vendor's
span, paint, image, coordinate and encoding helpers, changing only blob creation
for manifest-backed documents. It never replaces vendor functions or globals.
"""

from __future__ import annotations

import skia
from cr_renderer import CrelloV5Renderer, image_utils, text_utils
from cr_renderer.color_utils import parse_color
from cr_renderer.renderer import _decode_class_label, _get_scale_size
from cr_renderer.schema import OutlineTextEffect, TextElement


def make_native_text_blob(element, span, font_manager):
    """Build a vendor-compatible blob using the original fractional font size."""
    content = (span.text.upper() if element.capitalize else span.text).strip("\n")
    weight, style = "regular", "regular"
    if span.style.bold:
        weight = "bold"
        style = "bolditalic" if span.style.italic else "bold"
    elif span.style.italic:
        style = "italic"
    data = font_manager.lookup(element.font, weight, style)
    font = skia.Font(skia.Typeface.MakeFromData(data), float(element.fontSize))
    glyphs = font.textToGlyphs(content)
    positions = [x + element.letterSpacing * index for index, x in enumerate(font.getXPos(glyphs))]
    builder = skia.TextBlobBuilder()
    builder.allocRunPosH(font, glyphs, positions, 0)
    advance = sum(font.getWidths(glyphs)) + element.letterSpacing * max(0, len(glyphs) - 1)
    return text_utils.BlobItem(
        builder.make(),
        advance,
        parse_color(span.style.color, span.style.opacity),
        font.getMetrics(),
    )


def render_native_text(canvas, font_manager, element, descriptor=None):
    """Use vendor text paint/effects with document-local fractional blobs."""
    if descriptor is not None:
        from engine.native_text_effects import draw_native_text

        draw_native_text(canvas, font_manager, element, descriptor)
        return
    text_utils.check_supported_effects(element)
    for line_index, line in enumerate(text_utils.make_text_spans(element)):
        y = line_index * element.fontSize * (element.lineHeight or 1)
        blobs = [make_native_text_blob(element, span, font_manager) for span in line]
        x = text_utils.get_textalign_offset(element.textAlign, element.width, blobs)
        text_utils.render_text_box_effect(canvas, element, x, y, blobs)
        outline = text_utils.find_text_effect(element, OutlineTextEffect)
        if outline is None or outline.hasInnerText:
            text_utils.render_text_fill(canvas, element, x, y, blobs)
        if outline is not None:
            text_utils.render_text_outline(canvas, element, outline, x, y, blobs)


class NativeDocumentRenderer(CrelloV5Renderer):
    """Exact manifest face rendering isolated from the bundled brand renderer."""

    @classmethod
    def from_template(cls, template, font_manager):
        renderer = cls.__new__(cls)
        renderer.features = template.features
        renderer.font_manager = font_manager
        return renderer

    def render(
        self, example, max_size=360, render_text=True, format="jpeg", canvas_color=skia.ColorWHITE
    ):
        # The vendor has no injectable text-painter seam. Keep its small surface
        # traversal here, calling its unchanged helpers for all non-text work.
        descriptors = example.get("native_text_effects", [])
        example = _decode_class_label(self.features, example)
        scale, size = _get_scale_size(example["canvas_width"], example["canvas_height"], max_size)
        surface = skia.Surface(*size)
        raster_images = []
        with surface as canvas:
            canvas.scale(*scale)
            if canvas_color is not None:
                canvas.clear(canvas_color)
            for index in range(example["length"]):
                with skia.AutoCanvasRestore(canvas):
                    canvas.translate(example["left"][index], example["top"][index])
                    if example["angle"][index]:
                        canvas.rotate(
                            example["angle"][index],
                            example["width"][index] / 2,
                            example["height"][index] / 2,
                        )
                    if (
                        example["type"][index] == "TextElement"
                        and self.font_manager
                        and render_text
                    ):
                        element = TextElement(
                            uuid="",
                            type="textElement",
                            width=float(example["width"][index]),
                            height=float(example["height"][index]),
                            text=str(example["text"][index]),
                            fontSize=float(example["font_size"][index]),
                            font=str(example["font"][index]),
                            lineHeight=float(example["line_height"][index]),
                            textAlign=str(example["text_align"][index]),
                            capitalize=bool(example["capitalize"][index]),
                            letterSpacing=float(example["letter_spacing"][index]),
                            opacity=float(example["opacity"][index]),
                            boldMap=text_utils.generate_map(example["font_bold"][index]),
                            italicMap=text_utils.generate_map(example["font_italic"][index]),
                            colorMap=text_utils.generate_map(example["text_color"][index]),
                            lineMap=text_utils.generate_map(example["text_line"][index]),
                        )
                        render_native_text(
                            canvas,
                            self.font_manager,
                            element,
                            descriptors[index] if index < len(descriptors) else None,
                        )
                    else:
                        # Keep copied encoded bytes and decoded pixels owned for
                        # this render. Native imports create many short-lived
                        # prototype images before their final art composition.
                        encoded = image_utils.encode_pil_image(example["image"][index], "PNG")
                        data = skia.Data.MakeWithCopy(encoded)
                        image = skia.Image.MakeFromEncoded(data)
                        if image is None or (image := image.makeRasterImage()) is None:
                            raise ValueError("Native artwork could not be decoded exactly.")
                        raster_images.append((data, image))
                        canvas.drawImageRect(
                            image,
                            skia.Rect(image.width(), image.height()),
                            skia.Rect(example["width"][index], example["height"][index]),
                            paint=skia.Paint(Alphaf=example["opacity"][index], AntiAlias=True),
                        )
        return image_utils.encode_surface(surface, format)
