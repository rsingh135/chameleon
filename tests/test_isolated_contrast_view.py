"""Contrast composition uses synthetic RGBA pixels and no model or source assets."""

import builtins
import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from tools import isolated_contrast_view as composer


def png(color=(255, 255, 255, 255), size=(24, 20), mode="RGBA", format="PNG"):
    output = BytesIO()
    Image.new(mode, size, color).save(output, format=format)
    return output.getvalue()


def decoded(data):
    with Image.open(BytesIO(data)) as image:
        return image.copy()


@pytest.mark.parametrize(
    "color,matte", [((255, 255, 255, 255), 0), ((207, 199, 174, 255), 0), ((0, 0, 0, 255), 255)]
)
def test_opaque_white_cream_and_black_choose_opposite_matte(color, matte):
    result = composer.compose_contrast_view(png(color))
    receipt = result["receipt"]
    assert receipt["selected_matte_rgb"] == [matte] * 3
    assert receipt["selected_p10_contrast"] > 12
    assert decoded(result["png"]).getpixel((0, 0)) == color[:3]


def test_mixed_black_white_ink_chooses_neutral_using_lower_tail():
    image = Image.new("RGBA", (20, 10), (255, 255, 255, 255))
    image.paste((0, 0, 0, 255), (0, 0, 10, 10))
    source = BytesIO()
    image.save(source, format="PNG")
    receipt = composer.compose_contrast_view(source.getvalue())["receipt"]
    assert receipt["selected_matte_rgb"] == [117] * 3
    assert receipt["selected_p10_contrast"] > 4.5
    assert {row["matte_rgb"][0]: row["p10_contrast"] for row in receipt["candidates"]}[0] == 1


def test_authored_opacity_is_composited_once_and_original_bytes_unchanged():
    source = png((255, 255, 255, 128))
    original = source[:]
    result = composer.compose_contrast_view(source)
    assert source == original and decoded(source).getpixel((0, 0)) == (255, 255, 255, 128)
    assert result["receipt"]["selected_matte_rgb"] == [0] * 3
    assert decoded(result["png"]).getpixel((0, 0)) == (128, 128, 128)
    assert result["receipt"]["maximum_alpha"] == 128
    assert 5 < result["receipt"]["selected_p10_contrast"] < 5.4


def test_empty_alpha_uses_neutral_with_explicit_empty_paint_and_no_fake_scores():
    result = composer.compose_contrast_view(png((250, 10, 50, 0)))
    receipt = result["receipt"]
    assert (
        receipt["paint_empty"] and receipt["core_pixel_count"] == receipt["core_sample_count"] == 0
    )
    assert receipt["selected_matte_rgb"] == [117] * 3
    assert decoded(result["png"]).getpixel((0, 0)) == (117, 117, 117)
    assert receipt["selected_p10_contrast"] is None
    assert all(row["p10_contrast"] is None for row in receipt["candidates"])
    assert receipt["policy"]["exact_tie_preference"] == [117, 255, 0]


def test_sample_cap_is_uniform_deterministic_and_binds_exact_input_output_pixels():
    source = png(size=(512, 512))
    source_sha = hashlib.sha256(source).hexdigest()
    first = composer.compose_contrast_view(source, expected_alpha_sha256=source_sha)
    assert first == composer.compose_contrast_view(source)
    receipt = first["receipt"]
    assert receipt["core_pixel_count"] == 512 * 512
    assert receipt["core_sample_count"] == 4096
    positions = [index * (512 * 512 - 1) // 4095 for index in range(4096)]
    assert (
        receipt["sample_positions_sha256"]
        == hashlib.sha256(json.dumps(positions, separators=(",", ":")).encode()).hexdigest()
    )
    assert receipt["input_alpha_png_sha256"] == source_sha
    assert receipt["output_png_sha256"] == hashlib.sha256(first["png"]).hexdigest()
    assert (
        receipt["output_rgb_pixel_sha256"]
        == hashlib.sha256(decoded(first["png"]).tobytes()).hexdigest()
    )
    assert receipt["pillow"]["version"] and len(receipt["implementation_sha256"]) == 64
    assert not receipt["native_redraw"] and not receipt["model_executed"]
    assert not receipt["quality_approved"] and not receipt["promotion_eligible"]


@pytest.mark.parametrize(
    "source",
    [
        b"",
        b"not PNG",
        png(size=(513, 1)),
        png((255, 255, 255), mode="RGB"),
        png((255, 255, 255), mode="RGB", format="JPEG"),
        b"0" * (composer.MAX_PNG_BYTES + 1),
    ],
    ids=["empty", "invalid", "oversized-axis", "rgb", "jpeg", "oversized-bytes"],
)
def test_invalid_oversized_or_non_rgba_inputs_are_rejected(source):
    with pytest.raises(ValueError):
        composer.compose_contrast_view(source)


@pytest.mark.parametrize("expected", ["a" * 64, "invalid", True])
def test_expected_input_hash_must_be_valid_and_match(expected):
    with pytest.raises(ValueError, match="SHA-256"):
        composer.compose_contrast_view(png(), expected_alpha_sha256=expected)


def test_unknown_policy_and_truncated_png_are_rejected():
    with pytest.raises(ValueError, match="policy"):
        composer.compose_contrast_view(png(), policy="other")
    with pytest.raises(ValueError):
        composer.compose_contrast_view(png()[:55])


def test_core_threshold_excludes_faint_antialias_fringe():
    image = Image.new("RGBA", (10, 1))
    image.putdata(
        [(255, 255, 255, alpha) for alpha in (0, 10, 50, 100, 190, 191, 192, 200, 254, 255)]
    )
    data = BytesIO()
    image.save(data, format="PNG")
    assert composer.compose_contrast_view(data.getvalue())["receipt"]["core_pixel_count"] == 4


def test_no_models_renderer_labels_or_context_inputs_are_imported(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "engine", "renderer"}
        assert not name.endswith(("worker", "train_sft"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    receipt = composer.compose_contrast_view(png())["receipt"]
    assert receipt["policy"]["selection_uses_labels_or_context"] is False
