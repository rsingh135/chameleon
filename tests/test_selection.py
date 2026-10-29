"""Selection must enforce the contract before considering visual quality."""

from copy import deepcopy

import pytest

import engine.resize as module


@pytest.fixture(autouse=True)
def clear_resize_cache(monkeypatch):
    # Selection fixtures supply their candidate sets explicitly. New production
    # state families must not quietly add unrelated proposals to those sets.
    for factory in (
        "generate_composition_states",
        "generate_grouped_states",
        "generate_focal_states",
        "generate_source_guides",
    ):
        monkeypatch.setattr(module, factory, lambda *_args: ())
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
        lambda _source, candidate, *_args, **_kwargs: {"score": 1 if candidate == unsafe else 0.5},
    )
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [unsafe, safe])
    assert module.resize(design, target) == safe


@pytest.mark.parametrize("failure", ["reversed_copy", "detached_action"])
def test_relationship_loss_cannot_win_with_a_higher_aggregate_proxy(monkeypatch, failure):
    import json
    from pathlib import Path

    import engine.eval as evaluation
    from engine.layout.composition import generate_composition_states

    task = json.loads(
        (
            Path(__file__).resolve().parents[1] / "input/chameleon_sample_level2_core.json"
        ).read_text()
    )
    source, target = task["source"], task["target_canvas"]
    connected = generate_composition_states(source, target)[-1].design
    broken = deepcopy(connected)
    if failure == "reversed_copy":
        head = next(e for e in broken["elements"] if e["id"] == "headline")
        next(e for e in broken["elements"] if e["id"] == "subhead")["y"] = head["y"] - 40
    else:
        for element in broken["elements"]:
            if element.get("group_id") == "cta":
                element["x"] += 450

    def proxy(_source, candidate, *_args, **_kwargs):
        return {
            "violations": [],
            "metrics": {"raw_collisions": 0, "text_size": 1},
            "total": 1 if candidate == broken else 0.5,
        }

    monkeypatch.setattr(evaluation, "score", proxy)
    assert module._select_candidate(source, target, [broken, connected]) == connected


def test_exact_gap_fidelity_cannot_hide_large_typography_and_style_loss():
    import json
    from pathlib import Path

    from engine.eval.style import analyze_style
    from engine.layout.composition import generate_composition_states
    from engine.layout.relationships import analyze_relationships, build_relationship_graph

    source = json.loads(
        (
            Path(__file__).resolve().parents[1] / "input/chameleon_sample_level2_core.json"
        ).read_text()
    )["source"]
    target = {"width": 1080, "height": 1920}
    exact_gaps = module.generate_candidates(source, target)[0]
    coherent = generate_composition_states(source, target)[-1].design
    graph = build_relationship_graph(source)
    assert (
        analyze_relationships(graph, exact_gaps)["score"]
        > analyze_relationships(graph, coherent)["score"]
    )
    assert (
        analyze_style(source, coherent)["score"] > analyze_style(source, exact_gaps)["score"] + 0.1
    )
    assert module._select_candidate(source, target, [exact_gaps, coherent]) == coherent


def test_resize_plan_returns_independent_design_and_state_copies(design):
    import hashlib
    import json

    target = {"width": 300, "height": 100}
    first = module.resize_with_state(design, target)
    assert first["design"] == module.resize(design, target)
    assert (
        first["state"]["source_digest"]
        == hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    )
    first["state"]["cluster_axes"]["invented"] = "horizontal"
    first["state"]["sizing_exceptions"].append("invented")
    first["design"]["elements"][0]["x"] = 9000
    second = module.resize_with_state(design, target)
    assert "invented" not in second["state"]["cluster_axes"]
    assert "invented" not in second["state"]["sizing_exceptions"]
    assert second["design"]["elements"][0]["x"] != 9000


def test_compact_state_json_retains_selected_name_and_axes(monkeypatch, design):
    import json

    from engine.layout.composition import CompositionState

    target = {"width": 300, "height": 100}
    candidate = {**deepcopy(design), "canvas": target}
    compact = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    assert ": " not in compact
    state = CompositionState(
        "compact-state",
        "source-guides/compact",
        "fixture",
        300,
        100,
        (("opaque-cluster", "horizontal"),),
        compact,
    )
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [])
    monkeypatch.setattr(module, "generate_source_guides", lambda *_: (state,))

    def select_supplied_state(_source, _target, candidates, axes):
        # Isolate serialization lookup; relationship-axis semantics are covered
        # by the source-evidenced declared-flow test below.
        assert axes[json.dumps(candidate, sort_keys=True)] == {"opaque-cluster": "horizontal"}
        return candidates[0]

    monkeypatch.setattr(module, "_select_candidate", select_supplied_state)
    plan = module.resize_with_state(design, target)
    assert plan["design"] == candidate
    assert plan["state"]["name"] == "compact-state"
    assert plan["state"]["topology"] == "source-guides/compact"
    assert plan["state"]["cluster_axes"] == {"opaque-cluster": "horizontal"}


def test_selected_declared_axes_survive_the_public_resize_plan(monkeypatch):
    import json
    from dataclasses import asdict

    from engine.layout.composition import CompositionState
    from engine.layout.relationships import analyze_relationships, build_relationship_graph

    def text(eid, y, size):
        return {
            "id": eid,
            "type": "text",
            "x": 20,
            "y": y,
            "width": 300,
            "height": 60,
            "font_size": size,
            "font_family": "Solstice Sans",
            "content": "Title" if size == 40 else "Body",
            "color": "#000000",
        }

    source = {
        "canvas": {"width": 700, "height": 500},
        "elements": [text("a", 20, 40), text("b", 100, 20)],
    }
    graph = build_relationship_graph(source)
    cluster = graph.clusters[0]
    first, second = (graph.units[i].box for i in cluster.unit_indices)
    gap = next(e.source_gap for e in graph.edges if e.code == "message_attachment")
    target = {"width": 800, "height": 500}
    candidate = deepcopy(source)
    candidate["canvas"] = target
    candidate["elements"][1].update(
        x=first.x + first.width + gap * max(first.width, second.width), y=first.y
    )
    state = CompositionState(
        "declared-flow",
        "test-horizontal",
        "fixture",
        800,
        500,
        ((cluster.id, "horizontal"),),
        json.dumps(candidate, sort_keys=True),
    )
    before = asdict(graph)
    monkeypatch.setattr(module, "generate_candidates", lambda *_: [])
    monkeypatch.setattr(module, "generate_composition_states", lambda *_: (state,))
    plan = module.resize_with_state(source, target)
    assert plan["state"]["name"] == "declared-flow"
    assert plan["state"]["cluster_axes"] == {cluster.id: "horizontal"}
    assert (
        analyze_relationships(graph, plan["design"], plan["state"]["cluster_axes"])["warnings"]
        == []
    )
    assert analyze_relationships(graph, plan["design"])["warnings"]
    assert asdict(graph) == before


def test_larger_label_cannot_win_by_distorting_its_rigid_control(monkeypatch):
    import engine.eval as evaluation
    import engine.eval.style as style_analysis

    source = {
        "canvas": {"width": 400, "height": 200},
        "elements": [
            {
                "id": "plate",
                "type": "shape",
                "x": 20,
                "y": 20,
                "width": 200,
                "height": 60,
                "shape_type": "rectangle",
                "fill_color": "#cc3300",
                "corner_radius": 30,
                "group_id": "control",
            },
            {
                "id": "label",
                "type": "text",
                "x": 30,
                "y": 32,
                "width": 180,
                "height": 36,
                "font_size": 14,
                "font_family": "Solstice Sans",
                "content": "New",
                "color": "#ffffff",
                "group_id": "control",
            },
        ],
    }
    faithful = deepcopy(source)
    target = {"width": 600, "height": 200}
    faithful["canvas"] = target
    inflated = deepcopy(faithful)
    inflated["elements"][1]["font_size"] = 21
    monkeypatch.setattr(
        evaluation,
        "score",
        lambda _source, candidate, *_args, **_kwargs: {
            "violations": [],
            "metrics": {"raw_collisions": 0, "text_size": 1},
            "total": 1.0 if candidate == inflated else 0.5,
        },
    )
    monkeypatch.setattr(style_analysis, "analyze_style", lambda *_args, **_kwargs: {"score": 1})
    assert module._select_candidate(source, target, [inflated, faithful]) == faithful


@pytest.mark.parametrize("footer", [True, False])
def test_fine_print_floor_does_not_apply_to_small_body_copy(monkeypatch, footer):
    import engine.eval as evaluation
    import engine.eval.style as style_analysis

    source = {
        "canvas": {"width": 500, "height": 400},
        "elements": [
            {
                "id": "title",
                "type": "text",
                "x": 40,
                "y": 40,
                "width": 400,
                "height": 70,
                "font_size": 40,
                "font_family": "Solstice Sans",
                "content": "A title",
                "color": "#000000",
            },
            {
                "id": "detail",
                "type": "text",
                "x": 40,
                "y": 365 if footer else 200,
                "width": 400,
                "height": 30,
                "font_size": 10,
                "font_family": "Solstice Sans",
                "content": "Detailed conditions. " * 4,
                "color": "#000000",
            },
        ],
    }
    target = {"width": 800, "height": 400}
    faithful = deepcopy(source)
    faithful["canvas"] = target
    for element in faithful["elements"]:
        element["font_size"] *= 0.96
    distorted = deepcopy(faithful)
    distorted["elements"][0]["font_size"] = 28
    distorted["elements"][1]["font_size"] = 12
    monkeypatch.setattr(
        evaluation,
        "score",
        lambda _source, candidate, *_args, **_kwargs: {
            "violations": [],
            "metrics": {
                "raw_collisions": 0,
                "text_size": 0.96 if candidate == faithful else 1,
            },
            "total": 1,
        },
    )
    monkeypatch.setattr(
        style_analysis,
        "analyze_style",
        lambda _source, candidate, *_args, **_kwargs: {
            "score": 1 if candidate == faithful else 0.8,
        },
    )
    expected = faithful if footer else distorted
    assert module._select_candidate(source, target, [distorted, faithful]) == expected
