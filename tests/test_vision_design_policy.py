"""Offline strict framing cannot relax backend identity or admission policy."""

from copy import deepcopy

import pytest

from engine.eval import build_context, check_contract
from experiments.vision_critic.design_policy import strict_keep_all_in_frame


@pytest.fixture
def source():
    return {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "paper",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 100,
                "fill_color": "rgba(255,255,255,1)",
            },
            {
                "id": "hero",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 20,
                "y": 40,
                "width": 30,
                "height": 20,
                "fill_color": "rgba(50,100,150,1)",
                "group_id": "main",
            },
            {
                "id": "title",
                "type": "text",
                "content": "Study",
                "x": 20,
                "y": 10,
                "width": 70,
                "height": 24,
                "font_family": "Solstice Sans",
                "font_size": 12,
                "color": "#20372f",
            },
        ],
    }


def codes(violations):
    return {violation.code for violation in violations}


def test_valid_geometry_change_is_read_only_and_remains_backend_valid(source):
    candidate = deepcopy(source)
    candidate["elements"][1]["x"] = 30
    snapshot = deepcopy((source, candidate))
    assert strict_keep_all_in_frame(source, candidate, source["canvas"]) == []
    assert (source, candidate) == snapshot
    assert check_contract(source, candidate, source["canvas"]) == []


@pytest.mark.parametrize(
    "operation,expected",
    [("add", "id_unexpected"), ("omit", "id_missing"), ("duplicate", "id_duplicate")],
)
def test_backend_rejections_are_preserved(source, operation, expected):
    candidate = deepcopy(source)
    if operation == "add":
        candidate["elements"].append({**candidate["elements"][1], "id": "new-asset"})
    elif operation == "omit":
        candidate["elements"].pop()
    else:
        candidate["elements"].append(deepcopy(candidate["elements"][1]))
    assert expected in codes(strict_keep_all_in_frame(source, candidate, source["canvas"]))


@pytest.mark.parametrize(
    "field,value", [("content", "Replacement"), ("font_family", "Other Font"), ("color", "#ff0000")]
)
def test_source_copy_fonts_and_colors_are_frozen(source, field, value):
    candidate = deepcopy(source)
    candidate["elements"][2][field] = value
    assert "frozen_field_changed" in codes(
        strict_keep_all_in_frame(source, candidate, source["canvas"])
    )


def test_added_semantic_fields_and_layer_reorder_cannot_evade_backend_checks(source):
    candidate = deepcopy(source)
    candidate["elements"][1]["new_brand_asset"] = "unapproved.png"
    assert check_contract(source, candidate, source["canvas"]) == []
    assert "strict_identity_changed" in codes(
        strict_keep_all_in_frame(source, candidate, source["canvas"])
    )
    candidate = deepcopy(source)
    candidate["elements"][1:] = reversed(candidate["elements"][1:])
    assert check_contract(source, candidate, source["canvas"]) == []
    assert "strict_layer_order" in codes(
        strict_keep_all_in_frame(source, candidate, source["canvas"])
    )


def test_source_authored_crop_is_not_a_strict_foreground_exemption(source):
    source["elements"][1]["x"] = -10
    candidate = deepcopy(source)
    assert check_contract(source, candidate, source["canvas"]) == []
    assert "strict_out_of_frame" in codes(
        strict_keep_all_in_frame(source, candidate, source["canvas"])
    )
    # This policy does not alter the backend's existing authored-crop authority.
    assert check_contract(source, candidate, source["canvas"]) == []
    with pytest.raises(ValueError, match="full-frame"):
        strict_keep_all_in_frame(
            source, candidate, source["canvas"], background_bleed_ids=("hero",)
        )


def test_rotation_and_subpixel_frame_escape_are_checked(source):
    source["elements"][1].update(x=0, rotation=45)
    candidate = deepcopy(source)
    assert check_contract(source, candidate, source["canvas"]) == []
    assert "strict_out_of_frame" in codes(
        strict_keep_all_in_frame(source, candidate, source["canvas"])
    )
    source["elements"][1].update(x=20, rotation=0)
    candidate = deepcopy(source)
    candidate["elements"][1]["x"] = -0.25
    assert check_contract(source, candidate, source["canvas"]) == []
    assert "strict_out_of_frame" in codes(
        strict_keep_all_in_frame(source, candidate, source["canvas"])
    )


def test_only_explicit_background_bleed_can_be_exempt_and_never_relaxes_backend(source):
    source["elements"][0].update(x=-5, width=110)
    candidate = deepcopy(source)
    assert "strict_out_of_frame" in codes(
        strict_keep_all_in_frame(source, candidate, source["canvas"])
    )
    assert (
        strict_keep_all_in_frame(
            source, candidate, source["canvas"], background_bleed_ids=("paper",)
        )
        == []
    )
    source["elements"][0].update(x=0, width=100)
    candidate = deepcopy(source)
    candidate["elements"][0].update(x=-5, width=110)
    assert "off_canvas" in codes(
        strict_keep_all_in_frame(
            source, candidate, source["canvas"], background_bleed_ids=("paper",)
        )
    )


@pytest.mark.parametrize("allowlist", ["paper", ("unknown",), ("title",), ("paper", "paper")])
def test_invalid_bleed_configuration_is_rejected(source, allowlist):
    with pytest.raises(ValueError):
        strict_keep_all_in_frame(source, source, source["canvas"], background_bleed_ids=allowlist)


@pytest.mark.parametrize(
    "target",
    [
        {"width": 100, "height": 100, "ignore_crop": True},
        {"width": True, "height": 100},
        {"width": float("nan"), "height": 100},
    ],
)
def test_target_configuration_is_closed_and_finite(source, target):
    assert "strict_target_invalid" in codes(strict_keep_all_in_frame(source, source, target))


def test_foreign_context_and_unknown_policy_keywords_are_rejected(source):
    other = deepcopy(source)
    other["elements"][1]["x"] = 25
    with pytest.raises(ValueError, match="supplied source"):
        strict_keep_all_in_frame(source, source, source["canvas"], context=build_context(other))
    with pytest.raises(TypeError, match="unexpected keyword"):
        strict_keep_all_in_frame(source, source, source["canvas"], allow_omissions=True)


def test_malformed_source_and_overflowing_numbers_fail_closed(source):
    malformed = deepcopy(source)
    malformed["elements"][0]["type"] = ["shape"]
    assert "strict_source_invalid" in codes(
        strict_keep_all_in_frame(malformed, source, source["canvas"])
    )
    huge = {"width": 10**400, "height": 100}
    assert "strict_target_invalid" in codes(strict_keep_all_in_frame(source, source, huge))
    malformed = deepcopy(source)
    malformed["elements"][0]["x"] = 10**400
    assert "strict_source_invalid" in codes(
        strict_keep_all_in_frame(malformed, source, source["canvas"])
    )
