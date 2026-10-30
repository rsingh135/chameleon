"""Conservative interband clearance for proven native panel backgrounds."""

from copy import deepcopy

import pytest

from engine.layout.geometry import Box, bounds, intersection, union
from engine.layout.model import Unit
from engine.layout.source_guides import _Band, _Panel, _panel_dimensions, _panel_place


def _panel(plate_y=70, corner=30):
    root = dict(
        id="panel",
        type="shape",
        shape_type="rectangle",
        x=0,
        y=0,
        width=400,
        height=400,
        fill_color="#123456",
        z_index=0,
    )
    previous = dict(
        id="previous",
        type="text",
        x=20,
        y=20,
        width=360,
        height=40,
        content="Previous body",
        font_family="Solstice Sans",
        font_size=20,
        text_align="left",
        line_height=1,
        color="#ffffff",
        z_index=1,
    )
    plate = dict(
        id="plate",
        type="shape",
        shape_type="rectangle",
        x=20,
        y=plate_y,
        width=360,
        height=80,
        fill_color="#ffffff",
        corner_radius=corner,
        z_index=2,
    )
    caption = dict(
        id="caption",
        type="text",
        x=40,
        y=plate_y + 20,
        width=280,
        height=40,
        content="Following caption",
        font_family="Solstice Sans",
        font_size=20,
        text_align="left",
        line_height=1,
        color="#123456",
        z_index=3,
    )
    first = _Band((previous,), bounds(previous), Box(20, 20, 360, 20))
    second = _Band((plate, caption), union([plate, caption]), bounds(plate))
    return _Panel(
        Unit([root], bounds(root)),
        (Unit([previous], bounds(previous)), Unit([plate, caption], second.box)),
        (first, second),
        frozenset({"plate"}),
        True,
        False,
    )


@pytest.mark.parametrize("corner", [0, 30])
def test_compact_panel_keeps_preceding_text_frame_clear_of_next_native_plate(corner):
    panel = _panel(corner=corner)
    original = deepcopy(panel)
    # Source frames are separate. A2em frame has one line of occupied copy;
    # optical packing must retain enough room for the same conservative frame.
    assert intersection(panel.rows[0].box, panel.rows[1].box) == 0
    _, required_height = _panel_dimensions(panel)
    packed = _panel_place(panel, Box(0, 0, 400, required_height), 1)
    assert packed is not None
    changed = {element["id"]: element for element in packed[0]}
    assert intersection(bounds(changed["previous"]), bounds(changed["plate"])) == 0
    assert changed["previous"]["font_size"] == changed["caption"]["font_size"] == 20
    assert changed["previous"]["height"] == 40
    assert changed["plate"]["corner_radius"] == corner
    assert panel == original


def test_original_frame_overlap_is_not_rewritten_as_a_new_clearance_requirement():
    panel = _panel(plate_y=50)
    assert intersection(panel.rows[0].box, panel.rows[1].box) > 0
    _, required_height = _panel_dimensions(panel)
    packed = _panel_place(panel, Box(0, 0, 400, required_height), 1)
    assert packed is not None
    changed = {element["id"]: element for element in packed[0]}
    assert intersection(bounds(changed["previous"]), bounds(changed["plate"])) > 0
    assert changed["previous"]["font_size"] == changed["caption"]["font_size"] == 20
