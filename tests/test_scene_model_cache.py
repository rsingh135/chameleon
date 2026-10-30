"""Model bindings follow bytes, and protected source-visible content survives."""

import hashlib
import json
import os
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from engine.scene import analyze_scene
from engine.scene_model import constrain_candidates, validate_model_response


def _png(color):
    buffer = BytesIO()
    Image.new("RGB", (20, 30), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_scene_rebinds_actual_asset_bytes_when_size_and_mtime_are_restored(tmp_path):
    first = _png((10, 20, 30))
    second = next(
        data for channel in range(11, 240) if len(data := _png((channel, 20, 30))) == len(first)
    )
    assert first != second and len(first) == len(second)
    asset = tmp_path / "photo.png"
    asset.write_bytes(first)
    original_stat = asset.stat()
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "photo",
                "type": "image",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 100,
                "asset_id": "photo.png",
                "fit": "fill",
            }
        ],
    }
    original = deepcopy(source)
    before = analyze_scene(source, assets_dir=tmp_path)
    response = {
        "schema_version": 1,
        "source_digest": before["source_digest"],
        "scene_digest": before["scene_digest"],
        "images": [
            {
                "element_id": "photo",
                "asset_sha256": before["images"][0]["asset_sha256"],
                "proposed_role": "product",
                "confidence": 0.99,
            }
        ],
    }
    asset.write_bytes(second)
    os.utime(asset, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert asset.stat().st_size == original_stat.st_size
    assert asset.stat().st_mtime_ns == original_stat.st_mtime_ns
    after = analyze_scene(source, assets_dir=tmp_path)
    assert after["images"][0]["asset_sha256"] == hashlib.sha256(second).hexdigest()
    assert after["scene_digest"] != before["scene_digest"]
    with pytest.raises(ValueError):
        validate_model_response(response, after)
    assert source == original


def test_ordinary_resize_cache_is_invalidated_by_changed_asset_bytes(tmp_path, monkeypatch):
    import engine.resize as resizing

    first = _png((10, 20, 30))
    second = next(
        data for channel in range(11, 240) if len(data := _png((channel, 20, 30))) == len(first)
    )
    asset = tmp_path / "photo.png"
    asset.write_bytes(first)
    original_stat = asset.stat()
    source = {
        "canvas": {"width": 20, "height": 30},
        "elements": [
            {
                "id": "photo",
                "type": "image",
                "x": 0,
                "y": 0,
                "width": 20,
                "height": 30,
                "asset_id": "photo.png",
                "fit": "fill",
            }
        ],
    }
    calls = []

    def run(source_json, width, height, **kwargs):
        calls.append(1)
        return {"design": json.loads(source_json), "state": {"invocation": len(calls)}}

    monkeypatch.setattr(resizing, "_resize_in_context", run)
    original = resizing.resize_with_state(source, source["canvas"], assets_dir=tmp_path)
    assert resizing.resize_with_state(source, source["canvas"], assets_dir=tmp_path) == original
    assert len(calls) == 1
    asset.write_bytes(second)
    os.utime(asset, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    refreshed = resizing.resize_with_state(source, source["canvas"], assets_dir=tmp_path)
    assert len(calls) == 2
    assert refreshed["state"]["invocation"] == 2


def _protected_crop_case():
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "product",
                "type": "image",
                "x": -10,
                "y": -20,
                "width": 100,
                "height": 200,
                "asset_id": "product.png",
                "fit": "fill",
            }
        ],
    }
    profile = {
        "source_digest": "a" * 64,
        "scene_digest": "b" * 64,
        "images": [{"element_id": "product", "asset_sha256": "c" * 64}],
    }
    response = {
        "schema_version": 1,
        "source_digest": "a" * 64,
        "scene_digest": "b" * 64,
        "images": [
            {
                "element_id": "product",
                "asset_sha256": "c" * 64,
                "proposed_role": "product",
                "confidence": 0.99,
            }
        ],
    }
    return source, profile, response


@pytest.mark.parametrize("position", [{"x": -90}, {"x": 10}, {"y": -150}])
def test_protected_authored_crop_cannot_lose_or_exchange_source_visible_pixels(position):
    source, profile, response = _protected_crop_case()
    candidate = deepcopy(source)
    candidate["elements"][0].update(position)
    originals = deepcopy((source, profile, response, candidate))
    remaining, trace = constrain_candidates(source, [candidate], response, profile)
    assert remaining == []
    assert trace["candidates_vetoed"] == 1
    assert (source, profile, response, candidate) == originals


def test_protected_authored_crop_can_scale_and_reveal_more_without_losing_visible_pixels():
    source, profile, response = _protected_crop_case()
    same = deepcopy(source)
    same["canvas"] = {"width": 50, "height": 50}
    same["elements"][0].update(x=-5, y=-10, width=50, height=100)
    more = deepcopy(same)
    more["elements"][0]["x"] = 0
    remaining, trace = constrain_candidates(source, [same, more], response, profile)
    assert remaining == [same, more]
    assert trace["candidates_vetoed"] == 0
