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


def test_source_style_beats_a_higher_proxy_for_equally_readable_layouts(monkeypatch):
    import engine.eval as evaluation

    source = {
        "canvas": {"width": 400, "height": 300},
        "elements": [
            {
                "id": "corner-plate",
                "type": "shape",
                "x": 10,
                "y": 10,
                "width": 100,
                "height": 40,
                "shape_type": "rectangle",
                "fill_color": "rgba(230,100,40,1)",
                "group_id": "corner",
                "z_index": 1,
            },
            {
                "id": "corner-label",
                "type": "text",
                "x": 16,
                "y": 20,
                "width": 88,
                "height": 20,
                "content": "New",
                "font_family": "Solstice Sans",
                "font_size": 14,
                "color": "rgba(255,255,255,1)",
                "group_id": "corner",
                "z_index": 2,
            },
        ],
    }
    target = {"width": 600, "height": 300}
    faithful = {**deepcopy(source), "canvas": target}
    moved = deepcopy(faithful)
    for element in moved["elements"]:
        element["x"] += 480

    def proxy(_source, candidate, *_args, **_kwargs):
        return {
            "violations": [],
            "metrics": {"raw_collisions": 0, "text_size": 1},
            "total": 1.0 if candidate == moved else 0.8,
        }

    monkeypatch.setattr(evaluation, "score", proxy)
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [moved, faithful])
    assert module.resize(source, target) == faithful


@pytest.mark.parametrize(
    "unsafe_metrics",
    [
        {"raw_collisions": 0.1, "text_size": 1},
        {"raw_collisions": 0, "text_size": 0.4},
    ],
)
def test_style_cannot_outweigh_collisions_or_unreadable_text(monkeypatch, design, unsafe_metrics):
    import engine.eval as evaluation
    import engine.eval.style as style_analysis

    target = {"width": 300, "height": 100}
    safe = {**deepcopy(design), "canvas": target}
    unsafe = deepcopy(safe)
    unsafe["elements"][0]["x"] += 20

    def proxy(_source, candidate, *_args, **_kwargs):
        return {
            "violations": [],
            "metrics": unsafe_metrics
            if candidate == unsafe
            else {"raw_collisions": 0, "text_size": 1},
            "total": 1,
        }

    monkeypatch.setattr(evaluation, "score", proxy)
    monkeypatch.setattr(
        style_analysis,
        "analyze_style",
        lambda _source, candidate, *_args: {"score": 1 if candidate == unsafe else 0.5},
    )
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [unsafe, safe])
    assert module.resize(design, target) == safe
