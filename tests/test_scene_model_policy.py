"""Untrusted local semantic predictions may only narrow crop permissions."""

from copy import deepcopy

import pytest

from engine.scene_model import constrain_candidates


def case(role="product", confidence=0.95):
    source = {
        "elements": [
            {
                "id": "photo",
                "type": "image",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 200,
                "fit": "fill",
            }
        ]
    }
    profile = {
        "source_digest": "a" * 64,
        "scene_digest": "b" * 64,
        "images": [{"element_id": "photo", "asset_sha256": "c" * 64}],
    }
    response = {
        "schema_version": 1,
        "source_digest": profile["source_digest"],
        "scene_digest": profile["scene_digest"],
        "images": [
            {
                "element_id": "photo",
                "asset_sha256": "c" * 64,
                "proposed_role": role,
                "confidence": confidence,
            }
        ],
    }
    translated = deepcopy(source)
    translated["elements"][0].update(x=15, y=20, width=50, height=100)
    cropped = deepcopy(source)
    cropped["elements"][0].update(width=200, height=100, fit="cover")
    return source, profile, response, translated, cropped


@pytest.mark.parametrize("role", ["product", "logo"])
def test_model_crop_veto_retains_uniform_source_crop_and_translation(role):
    source, profile, response, translated, cropped = case(role)
    original = deepcopy((source, profile, response))
    remaining, trace = constrain_candidates(source, [translated, cropped], response, profile)
    assert remaining == [translated]
    assert trace["protected_element_ids"] == ["photo"]
    assert trace["candidates_vetoed"] == 1
    assert trace["background_permissions_granted"] == 0
    assert (source, profile, response) == original


@pytest.mark.parametrize(
    "role,confidence", [("product", 0.79), ("unknown", 1), ("background", 1), ("foreground", 1)]
)
def test_uncertain_or_nonprotective_predictions_do_not_change_candidate_pool(role, confidence):
    source, profile, response, translated, cropped = case(role, confidence)
    remaining, trace = constrain_candidates(source, [translated, cropped], response, profile)
    assert remaining == [translated, cropped]
    assert trace["protected_element_ids"] == []
    assert trace["background_permissions_granted"] == 0


@pytest.mark.parametrize("field", ["source_digest", "scene_digest"])
def test_stale_model_prediction_is_rejected_before_selection(field):
    source, profile, response, translated, _ = case()
    response[field] = "d" * 64
    with pytest.raises(ValueError, match="another source or scene"):
        constrain_candidates(source, [translated], response, profile)


def test_position_change_does_not_hide_a_new_product_crop():
    source, profile, response, translated, _ = case()
    translated["elements"][0]["image_position"] = [0.5, 0.1]
    remaining, _ = constrain_candidates(source, [translated], response, profile)
    assert remaining == []
