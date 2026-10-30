"""Target-conditioned plans create source-bound, contract-valid layout candidates."""

import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from engine.eval import check_contract
from engine.layout.visual_plan import realize_plan
from engine.resources import using_resources
from engine.visual_plan import build_plan_menu, validate_decision

ROOT = Path(__file__).resolve().parents[1]


def _document(name):
    if name == "core":
        payload = json.loads((ROOT / "input/chameleon_sample_level2_core.json").read_text("utf-8"))
        return payload["source"], ROOT / "brand_kit/assets", ROOT / "brand_kit/fonts"
    folder = ROOT / "fixtures/open_posters/prepared" / name
    return (
        json.loads((folder / "source.json").read_text("utf-8")),
        folder / "resources/assets",
        folder / "resources/fonts",
    )


def _decision(menu, plan_id):
    return {"schema_version": 1, "menu_digest": menu["menu_digest"], "plan_id": plan_id}


@pytest.mark.parametrize(
    "name,family,target",
    [
        ("core", "brand_message", {"width": 1600, "height": 900}),
        ("doory_rooftop-event-story", "background_caption", {"width": 1600, "height": 900}),
        ("opensuse_trifold_front", "source_panels", {"width": 816, "height": 1145}),
    ],
)
def test_actual_sources_compile_new_layout_options_without_changing_identity(name, family, target):
    source, assets, fonts = _document(name)
    original = deepcopy(source)
    with using_resources(assets, fonts):
        menu = build_plan_menu(source, target)
        assert menu["status"] == "ready"
        assert 2 <= len(menu["plans"]) <= 7
        options = [p for p in menu["plans"] if p["family"] == family]
        assert options, menu
        states = realize_plan(source, target, menu, _decision(menu, options[0]["plan_id"]))
        assert len(states) == 1
        candidate = states[0].design
        assert check_contract(source, candidate, target) == []
        assert [e["id"] for e in candidate["elements"]] == [e["id"] for e in source["elements"]]
        for before, after in zip(source["elements"], candidate["elements"], strict=True):
            for field in (
                "type",
                "content",
                "font_family",
                "color",
                "asset_id",
                "group_id",
                "opacity",
                "z_index",
                "rotation",
                "style_runs",
            ):
                assert before.get(field) == after.get(field)
            if (
                before["type"] == "image"
                and before.get("fit") != "cover"
                and after.get("fit") != "cover"
                and not before.get("source_art")
            ):
                # Native flat-panel paint exceptions are still owned by the
                # existing contract, not granted by a model menu.
                assert math.isclose(
                    before["width"] / before["height"],
                    after["width"] / after["height"],
                    rel_tol=1e-6,
                )
        assert source == original


def test_resolution_and_keep_are_explicit_non_reorganizing_paths():
    source, assets, fonts = _document("core")
    target = {"width": 540, "height": 540}
    with using_resources(assets, fonts):
        menu = build_plan_menu(source, target)
        assert menu["status"] == "resolution"
        assert [p["plan_id"] for p in menu["plans"]] == ["keep"]
        assert realize_plan(source, target, menu, _decision(menu, "keep")) == ()


@pytest.mark.parametrize(
    "change", ["target", "source", "graph", "unknown_plan", "nan", "unsupported"]
)
def test_stale_unknown_or_unbounded_plan_input_is_rejected(change):
    source, assets, fonts = _document("core")
    target = {"width": 1600, "height": 900}
    with using_resources(assets, fonts):
        menu = build_plan_menu(source, target)
        proposal = _decision(menu, menu["plans"][-1]["plan_id"])
        if change == "target":
            target = {"width": 900, "height": 1600}
        elif change == "source":
            source["elements"][-1]["x"] += 1
        elif change == "graph":
            menu["graph_digest"] = "0" * 64
        elif change == "unknown_plan":
            proposal["plan_id"] = "delete-everything"
        elif change == "nan":
            menu["plans"][-1]["parameters"]["share"] = float("nan")
        else:
            proposal["shell_command"] = "untrusted poster text"
        with pytest.raises(ValueError):
            realize_plan(source, target, menu, proposal)


def test_decision_is_closed_and_does_not_trust_model_hashes_or_copy():
    menu = {"menu_digest": "a" * 64, "plans": [{"plan_id": "keep"}]}
    assert validate_decision(_decision(menu, "keep"), menu)["plan_id"] == "keep"
    with pytest.raises(ValueError):
        validate_decision({**_decision(menu, "keep"), "menu_digest": "b" * 64}, menu)


def test_new_background_menu_is_not_just_a_renamed_existing_candidate():
    from engine.layout.backgrounds import generate_background_states

    source, assets, fonts = _document("doory_rooftop-event-story")
    target = {"width": 1600, "height": 900}
    with using_resources(assets, fonts):
        menu = build_plan_menu(source, target)
        option = next(p for p in menu["plans"] if p["family"] == "background_caption")
        state = realize_plan(source, target, menu, _decision(menu, option["plan_id"]))[0]
        assert all(state.design != old.design for old in generate_background_states(source, target))


def test_current_asset_and_font_bytes_are_rebound_even_for_keep(tmp_path):
    source, assets, fonts = _document("core")
    copied_assets, copied_fonts = tmp_path / "assets", tmp_path / "fonts"
    shutil.copytree(assets, copied_assets)
    shutil.copytree(fonts, copied_fonts)
    target = {"width": 540, "height": 540}
    with using_resources(copied_assets, copied_fonts):
        menu = build_plan_menu(source, target)
        (copied_fonts / "new-receipt.json").write_text('{"changed":true}')
        with pytest.raises(ValueError, match="stale"):
            realize_plan(source, target, menu, _decision(menu, "keep"))
        menu = build_plan_menu(source, target)
        asset = copied_assets / source["elements"][0]["asset_id"]
        asset.write_bytes(asset.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="stale"):
            realize_plan(source, target, menu, _decision(menu, "keep"))


def test_compiled_options_are_strictly_in_frame_including_authored_decorations():
    from engine.eval.geometry import canvas_box, contains, rotated_aabb

    source, assets, fonts = _document("core")
    target = {"width": 1600, "height": 900}
    with using_resources(assets, fonts):
        menu = build_plan_menu(source, target)
        for option in menu["plans"][1:]:
            design = realize_plan(source, target, menu, _decision(menu, option["plan_id"]))[
                0
            ].design
            assert all(
                contains(canvas_box(target), rotated_aabb(e), slack=1e-6)
                for e in design["elements"]
            )
