"""Measured document-column fitting preserves source identity and hierarchy."""

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import check_contract
from engine.layout.fitted_columns import generate_fitted_column_states
from engine.resources import using_resources
from engine.stage_one import strictly_in_bounds

ROOT = Path(__file__).resolve().parents[1]


def _source():
    def text(eid, content, x, y, width, size):
        return {
            "id": eid,
            "type": "text",
            "content": content,
            "x": x,
            "y": y,
            "width": width,
            "height": size * 3,
            "font_family": "Inter-Regular",
            "font_size": size,
            "color": "#222222",
            "text_align": "left",
        }

    elements = [
        {
            "id": "paper",
            "type": "shape",
            "shape_type": "rectangle",
            "x": 0,
            "y": 0,
            "width": 600,
            "height": 900,
            "fill_color": "#ffffff",
        },
        text("title", "Alex\nRiver", 40, 25, 260, 30),
        text("role", "Designer", 40, 120, 260, 12),
        text("contact", "alex@example.test", 390, 40, 170, 10),
        {
            "id": "header-rule",
            "type": "shape",
            "shape_type": "rectangle",
            "x": 0,
            "y": 180,
            "width": 600,
            "height": 4,
            "fill_color": "#dddddd",
        },
        {
            "id": "column-rule",
            "type": "shape",
            "shape_type": "rectangle",
            "x": 220,
            "y": 230,
            "width": 1,
            "height": 600,
            "fill_color": "#dddddd",
        },
    ]
    for side, x, width in (("left", 35, 160), ("right", 250, 310)):
        for index, (content, size) in enumerate(
            [
                ("EDUCATION" if side == "left" else "PROFILE", 14),
                ("Source copy stays complete and may wrap in the available column.", 10),
                ("SKILLS" if side == "left" else "EXPERIENCE", 14),
                (
                    "Design\nResearch\nPrototyping"
                    if side == "left"
                    else "Built an accessible product with a small team and measured its impact.",
                    10,
                ),
            ]
        ):
            elements.append(text(f"{side}-{index}", content, x, 230 + index * 140, width, size))
    return {"canvas": {"width": 600, "height": 900}, "elements": elements}


def test_fluid_columns_use_available_width_at_a_common_font_scale():
    source = _source()
    before = deepcopy(source)
    states = generate_fitted_column_states(source, {"width": 1080, "height": 1080})
    assert len(states) >= 2
    design = states[0].design
    by_id = {e["id"]: e for e in design["elements"]}
    scales = [
        by_id[e["id"]]["font_size"] / e["font_size"]
        for e in source["elements"]
        if e["type"] == "text"
    ]
    assert max(scales) - min(scales) < 1e-8
    paragraph = next(e for e in source["elements"] if e["id"] == "right-1")
    assert by_id["right-1"]["width"] > paragraph["width"] * scales[0] * 1.1
    assert check_contract(source, design, {"width": 1080, "height": 1080}) == []
    assert strictly_in_bounds(design)
    assert source == before


def test_explicit_group_members_keep_one_similarity_transform():
    source = _source()
    for element in source["elements"]:
        if element["id"] in {"left-0", "left-1"}:
            element["group_id"] = "education-owner"
        if element["id"] == "left-1":
            element["height"] = 100
    states = generate_fitted_column_states(source, {"width": 1080, "height": 1080})
    assert states
    original = [e for e in source["elements"] if e.get("group_id")]
    changed = {e["id"]: e for e in states[0].design["elements"]}
    scale = changed[original[0]["id"]]["font_size"] / original[0]["font_size"]
    assert math.isclose(
        changed[original[1]["id"]]["y"] - changed[original[0]["id"]]["y"],
        (original[1]["y"] - original[0]["y"]) * scale,
    )
    assert math.isclose(changed[original[1]["id"]]["width"], original[1]["width"] * scale)


@pytest.mark.parametrize(
    "target",
    [
        {"width": 0, "height": 10},
        {"width": float("nan"), "height": 10},
        {"width": True, "height": 10},
    ],
)
def test_invalid_canvas_declines_without_modifying_source(target):
    assert generate_fitted_column_states(_source(), target) == []


def test_missing_column_evidence_and_proportional_resolution_decline():
    source = _source()
    assert generate_fitted_column_states(source, {"width": 300, "height": 450}) == []
    source["elements"] = [e for e in source["elements"] if e["id"] != "column-rule"]
    assert generate_fitted_column_states(source, {"width": 1080, "height": 1080}) == []


def test_section_headings_and_their_body_move_as_one_column_cohort():
    source = _source()
    states = generate_fitted_column_states(source, {"width": 1600, "height": 900})
    four = next(state for state in states if state.name.startswith("fitted-columns/4-"))
    changed = {e["id"]: e for e in four.design["elements"]}
    for side in ("left", "right"):
        for heading, body in ((0, 1), (2, 3)):
            assert math.isclose(changed[f"{side}-{heading}"]["x"], changed[f"{side}-{body}"]["x"])


def test_user_exposed_modern_resume_is_manual_acceptance_only():
    # User explicitly exposed this reserve case. This regression is manual
    # acceptance evidence, never a training row or a blind holdout claim.
    folder = ROOT / "fixtures/open_posters/prepared/doory_modern-resume"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    before = deepcopy(source)
    target = {"width": 1080, "height": 1080}
    with using_resources(folder / "resources/assets", folder / "resources/fonts"):
        states = generate_fitted_column_states(source, target)
        assert states
        design = states[0].design
        assert check_contract(source, design, target) == []
        assert strictly_in_bounds(design)
        assert [e["id"] for e in design["elements"]] == [e["id"] for e in source["elements"]]
        old = {e["id"]: e for e in source["elements"]}
        for element in design["elements"]:
            if element["type"] == "text":
                assert element["content"] == old[element["id"]]["content"]
                assert element["font_family"] == old[element["id"]]["font_family"]
        assert min(e["font_size"] for e in design["elements"] if e["type"] == "text") > 10.5
    assert source == before
