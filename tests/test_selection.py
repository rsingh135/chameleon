"""Selection must enforce the contract before considering visual quality."""

from copy import deepcopy

import pytest

import engine.resize as module


@pytest.fixture(autouse=True)
def clear_resize_cache():
    module._resize_cached.cache_clear()
    yield
    module._resize_cached.cache_clear()


@pytest.fixture
def design():
    return {
        "canvas": {"width": 200, "height": 100},
        "elements": [
            {
                "id": "panel",
                "type": "shape",
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 80,
                "shape_type": "rectangle",
                "fill_color": "rgba(255, 0, 0, 1)",
            }
        ],
    }


def test_invalid_candidate_cannot_win_a_quality_tie(monkeypatch, design):
    target = {"width": 300, "height": 100}
    valid = {**deepcopy(design), "canvas": target}
    invalid = deepcopy(valid)
    invalid["elements"][0]["fill_color"] = "rgba(0, 0, 0, 1)"
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [invalid, valid])
    assert module.resize(design, target) == valid


def test_all_invalid_candidates_report_the_contract_failure(monkeypatch, design):
    invalid = {**deepcopy(design), "canvas": {"width": 300, "height": 100}}
    invalid["elements"] = []
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [invalid])
    with pytest.raises(ValueError, match="id_missing"):
        module.resize(design, invalid["canvas"])


def test_source_aware_ranking_keeps_the_hero_prominent(monkeypatch):
    source = {
        "canvas": {"width": 200, "height": 100},
        "elements": [
            {
                "id": "product",
                "type": "image",
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 80,
                "asset_id": "missing-product.png",
                "fit": "contain",
            }
        ],
    }
    target = {"width": 300, "height": 100}
    prominent = {**deepcopy(source), "canvas": target}
    tiny = deepcopy(prominent)
    tiny["elements"][0].update(width=10, height=10)
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [tiny, prominent])
    assert module.resize(source, target) == prominent


def test_valid_identity_resize_preserves_composition_and_returns_a_copy(design):
    first = module.resize(design, design["canvas"])
    assert first == design
    first["elements"][0]["x"] = 999
    assert module.resize(design, design["canvas"]) == design


def test_identity_resize_does_not_preserve_source_text_overflow():
    source = {
        "canvas": {"width": 200, "height": 100},
        "elements": [
            {
                "id": "copy",
                "type": "text",
                "x": 10,
                "y": 10,
                "width": 150,
                "height": 50,
                "content": "Overflowingword" * 3,
                "font_family": "Solstice Sans",
                "font_size": 48,
                "color": "rgba(0, 0, 0, 1)",
            }
        ],
    }
    result = module.resize(source, source["canvas"])
    text = result["elements"][0]
    measured = module.get_measurer().measure(text, text["font_size"], text["width"])
    assert result != source
    assert measured.width <= text["width"] + 0.05
    assert measured.height <= text["height"] + 0.05
