"""Stage-one bounds do not inherit source-bleed exemptions."""

import json
from pathlib import Path

from engine.eval import check_contract
from engine.layout.geometry import bounds
from engine.resize import resize_with_state
from engine.stage_one import strictly_in_bounds


def test_strict_resolution_keeps_foreground_scale_but_contains_authored_decoration():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "input/chameleon_sample_level2_core.json").read_text())["source"]
    target = {"width": 540, "height": 540}
    ordinary = resize_with_state(source, target)
    strict = resize_with_state(source, target, strict_bounds=True)
    assert not strictly_in_bounds(ordinary["design"])
    assert strictly_in_bounds(strict["design"])
    assert check_contract(source, strict["design"], target) == []
    for old, new in zip(source["elements"], strict["design"]["elements"], strict=True):
        if old["type"] in {"text", "image"}:
            for key in {"x", "y", "width", "height", "font_size"} & old.keys():
                assert new[key] == old[key] * 0.5
    assert any(bounds(e).x < 0 for e in ordinary["design"]["elements"])


def test_strict_check_rejects_nonfinite_and_rotated_overflow():
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "a",
                "type": "shape",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 100,
                "shape_type": "rectangle",
                "fill_color": "#ffffff",
            }
        ],
    }
    assert strictly_in_bounds(source)
    source["elements"][0]["rotation"] = 45
    assert not strictly_in_bounds(source)
    source["elements"][0].update(rotation=0, x=float("nan"))
    assert not strictly_in_bounds(source)
