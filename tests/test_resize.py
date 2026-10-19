"""Contract tests for engine.resize.resize().

These encode the hard requirements from RULES.md and the resize() docstring,
run against every sample task x every target canvas. Anything marked
`xfail(strict=True)` is a requirement the shipped baseline (uniform stretch)
knowingly violates: once a real resize() satisfies it, pytest will flag the
now-passing test and you remove the marker. Treat those as the to-do list.
"""

import copy

from engine.main import self_check
from engine.resize import resize

# Max relative aspect-ratio drift allowed for non-cover images (RULES.md: ~5%).
ASPECT_TOLERANCE = 0.05
# Slack for float rounding on canvas bounds, same as main.self_check.
EDGE_SLACK = 0.5


def _in_bounds(el: dict, canvas: dict) -> bool:
    return (
        el["x"] >= -EDGE_SLACK
        and el["y"] >= -EDGE_SLACK
        and el["x"] + el["width"] <= canvas["width"] + EDGE_SLACK
        and el["y"] + el["height"] <= canvas["height"] + EDGE_SLACK
    )


def test_output_canvas_matches_target(source, target_canvas):
    out = resize(source, target_canvas)
    assert out["canvas"] == target_canvas


def test_every_source_id_appears_exactly_once(source, target_canvas):
    out = resize(source, target_canvas)
    out_ids = [el["id"] for el in out["elements"]]
    assert sorted(out_ids) == sorted(el["id"] for el in source["elements"])


def test_required_fields_present(source, target_canvas):
    out = resize(source, target_canvas)
    problems = [p for p in self_check(source, out) if "off-canvas" not in p]
    assert problems == []


def test_source_is_not_mutated(source, target_canvas):
    before = copy.deepcopy(source)
    resize(source, target_canvas)
    assert source == before


def test_content_and_identity_fields_unchanged(source, target_canvas):
    """Stage-one scoring treats text content, font family, asset, type and
    layering as fixed — we move furniture, we don't replace it."""
    out = resize(source, target_canvas)
    out_by_id = {el["id"]: el for el in out["elements"]}
    frozen = ("type", "content", "font_family", "asset_id", "shape_type", "group_id", "z_index")
    for src_el in source["elements"]:
        out_el = out_by_id[src_el["id"]]
        for field in frozen:
            if field in src_el:
                assert out_el.get(field) == src_el[field], f"{src_el['id']}.{field} changed"


def test_font_size_is_at_least_one(source, target_canvas):
    out = resize(source, target_canvas)
    for el in out["elements"]:
        if el["type"] == "text":
            assert el["font_size"] >= 1, f"{el['id']} font_size={el['font_size']}"


def test_elements_inside_source_canvas_stay_inside_target(source, target_canvas):
    """Elements the designer placed fully on-canvas must stay on-canvas.
    Elements that already bleed off the source (e.g. the Level 2 decorative
    circle) are allowed to keep bleeding — that's intentional design."""
    out = resize(source, target_canvas)
    out_by_id = {el["id"]: el for el in out["elements"]}
    for src_el in source["elements"]:
        if _in_bounds(src_el, source["canvas"]):
            assert _in_bounds(out_by_id[src_el["id"]], target_canvas), (
                f"{src_el['id']} pushed off-canvas"
            )


def test_non_cover_images_keep_aspect_ratio(source, target_canvas):
    out = resize(source, target_canvas)
    out_by_id = {el["id"]: el for el in out["elements"]}
    for src_el in source["elements"]:
        if src_el["type"] != "image" or src_el.get("fit") == "cover":
            continue
        out_el = out_by_id[src_el["id"]]
        src_ratio = src_el["width"] / src_el["height"]
        out_ratio = out_el["width"] / out_el["height"]
        drift = abs(out_ratio - src_ratio) / src_ratio
        assert drift <= ASPECT_TOLERANCE, f"{src_el['id']} aspect drift {drift:.1%}"


def test_handles_element_with_missing_asset():
    """Level 3 deliberately references an asset that isn't in brand_kit/.
    resize() works on the JSON only and must not touch (or trip over) files."""
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "ghost",
                "type": "image",
                "x": 10,
                "y": 10,
                "width": 50,
                "height": 50,
                "z_index": 1,
                "asset_id": "does_not_exist.png",
                "fit": "contain",
            }
        ],
    }
    out = resize(source, {"width": 200, "height": 50})
    assert [el["id"] for el in out["elements"]] == ["ghost"]
