"""Real source semantics survive viewer orchestration and serialized replay."""

import base64
import hashlib
import json
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from engine import design_semantics, semantic_ink
from engine.resources import ResourceContext, get_resources
from viewer import server
from viewer.resize_cache import ResourceBindings, SerializedCache


@pytest.fixture
def resources(tmp_path):
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    (fonts / "font-manifest.json").write_text("[]", encoding="utf-8")
    Image.new("RGBA", (20, 20), (30, 80, 160, 255)).save(assets / "art.png")
    return ResourceContext(assets, fonts)


@pytest.fixture
def payload():
    return {
        "source": {
            "canvas": {"width": 400, "height": 300},
            "elements": [
                {
                    "id": "corner",
                    "type": "image",
                    "asset_id": "art.png",
                    "x": 10,
                    "y": 10,
                    "width": 20,
                    "height": 20,
                },
                {
                    "id": "panel",
                    "type": "shape",
                    "shape_type": "rectangle",
                    "fill_color": "#334455",
                    "x": 80,
                    "y": 80,
                    "width": 240,
                    "height": 100,
                    "group_id": "message",
                },
                {
                    "id": "headline",
                    "type": "text",
                    "content": "A source message",
                    "font_family": "OriginalFace",
                    "font_size": 24,
                    "color": "#ffffff",
                    "x": 100,
                    "y": 100,
                    "width": 200,
                    "height": 40,
                    "group_id": "message",
                },
                {
                    "id": "destination",
                    "type": "text",
                    "content": "https://studio.example/design",
                    "font_family": "OriginalFace",
                    "font_size": 12,
                    "color": "#ffffff",
                    "x": 100,
                    "y": 270,
                    "width": 200,
                    "height": 20,
                },
            ],
        },
        "target_canvas": {"width": 200, "height": 150},
    }


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(sidecar=None, resize_calls=0, describe_calls=0, ink_calls=[])
    monkeypatch.setattr(server, "_RESOURCE_BINDINGS", ResourceBindings())
    monkeypatch.setattr(server, "_SOURCE_PNG_CACHE", SerializedCache(kind="source"))
    monkeypatch.setattr(server, "_RESIZE_RESULT_CACHE", SerializedCache(kind="result"))
    monkeypatch.setattr(server, "_execution_identity", lambda: "frozen-semantic-test")
    monkeypatch.setattr(server, "_visual_model_enabled", lambda: False)

    def resize(source, target, **_kwargs):
        state.resize_calls += 1
        output = deepcopy(source)
        scale = target["width"] / source["canvas"]["width"]
        output["canvas"] = deepcopy(target)
        for element in output["elements"]:
            for key in ("x", "y", "width", "height", "font_size"):
                if key in element:
                    element[key] *= scale
        layout_state = {"name": "test-uniform", "cluster_axes": {}}
        if state.sidecar is not None:
            layout_state["design_semantics"] = (
                state.sidecar(source) if callable(state.sidecar) else deepcopy(state.sidecar)
            )
        return {"design": output, "state": layout_state}

    def describe(source, **kwargs):
        assert server._ENGINE_LOCK.locked()
        state.describe_calls += 1
        return design_semantics.describe_design(source, **kwargs)

    ink = semantic_ink.element_ink_bounds

    def measured_ink(element, *, assets_dir=None, fonts_dir=None):
        assert server._ENGINE_LOCK.locked()
        assert get_resources() == ResourceContext(assets_dir, fonts_dir)
        state.ink_calls.append((element["id"], assets_dir, fonts_dir))
        return ink(element, assets_dir=assets_dir, fonts_dir=fonts_dir)

    def render(design, _attribution=None):
        output = BytesIO()
        Image.new("RGB", tuple(design["canvas"][key] for key in ("width", "height"))).save(
            output, format="PNG"
        )
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(), []

    monkeypatch.setattr(server, "resize_with_state", resize)
    monkeypatch.setattr(server, "_configured_resize", resize)
    monkeypatch.setattr(server, "describe_design", describe)
    monkeypatch.setattr(semantic_ink, "element_ink_bounds", measured_ink)
    monkeypatch.setattr(server, "build_context", lambda *_args: {})
    monkeypatch.setattr(server, "render_plate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server, "score", lambda *_args: {"violations": [], "metrics": {}, "total": 1.0}
    )
    monkeypatch.setattr(server, "build_relationship_graph", lambda *_args: {})
    for name in (
        "analyze_style",
        "analyze_relationships",
        "analyze_proportions",
        "analyze_focal",
        "analyze_rules",
        "analyze_reading_order",
        "conservative_occlusion",
    ):
        monkeypatch.setattr(server, name, lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_render_png", render)
    return state


def test_real_descriptors_bind_every_source_element_and_survive_cache_replay(
    payload, resources, backend
):
    original = deepcopy(payload)
    first = server.perform_resize(payload, resources=resources)
    descriptor = deepcopy(first["report"]["design_semantics"])
    rows = {row["element_id"]: row for row in descriptor["elements"]}
    assert set(rows) == {element["id"] for element in payload["source"]["elements"]}
    assert descriptor["source_canvas"] == original["source"]["canvas"]
    assert descriptor["source_digest"] == design_semantics.semantic_source_digest(
        original["source"]
    )
    assert rows["corner"]["purpose"] == "corner_marker"
    assert rows["corner"]["anchors"]
    assert rows["destination"]["purpose"] == "footer_destination"
    assert rows["headline"]["rigid_group"] == "message"
    assert rows["headline"]["relationships"]
    assert all(row["design_purpose"] and row["provenance"] for row in rows.values())
    assert backend.ink_calls
    assert all(
        (assets, fonts) == (resources.assets_dir, resources.fonts_dir)
        for _, assets, fonts in backend.ink_calls
    )
    ink_calls = list(backend.ink_calls)
    first["report"]["design_semantics"]["elements"][0]["purpose"] = "client mutation"
    repeated = server.perform_resize(payload, resources=resources)
    assert repeated["cache"]["result_hit"] is True
    assert repeated["report"]["design_semantics"] == descriptor
    assert backend.resize_calls == backend.describe_calls == 1
    assert backend.ink_calls == ink_calls
    assert payload == original


@pytest.mark.parametrize("configured", [False, True])
def test_valid_processor_sidecar_is_reused_without_repeating_native_measurement(
    payload, resources, backend, monkeypatch, configured
):
    monkeypatch.setattr(server, "_visual_model_enabled", lambda: configured)
    backend.sidecar = lambda source: design_semantics.describe_design(
        source, assets_dir=resources.assets_dir, fonts_dir=resources.fonts_dir
    )
    result = server.perform_resize(payload, resources=resources)
    assert backend.describe_calls == 0
    assert backend.ink_calls
    measured_ids = [eid for eid, _, _ in backend.ink_calls]
    assert measured_ids.count("corner") == measured_ids.count("destination") == 1
    assert (
        result["report"]["design_semantics"] == result["report"]["layout_state"]["design_semantics"]
    )


@pytest.mark.parametrize("invalid", ["stale_source", "changed_descriptor", "incomplete_coverage"])
def test_invalid_processor_sidecar_is_replaced_with_fresh_source_evidence(
    payload, resources, backend, invalid
):
    descriptor = design_semantics.describe_design(payload["source"])
    if invalid == "stale_source":
        descriptor["source_digest"] = "0" * 64
    elif invalid == "changed_descriptor":
        descriptor["elements"][0]["purpose"] = "unknown"
    else:
        descriptor["elements"].pop()
        body = {key: value for key, value in descriptor.items() if key != "semantic_digest"}
        descriptor["semantic_digest"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    backend.sidecar = descriptor
    result = server.perform_resize(payload, resources=resources)
    fresh = result["report"]["design_semantics"]
    assert backend.describe_calls == 1
    assert len(fresh["elements"]) == len(payload["source"]["elements"])
    assert fresh["source_digest"] == design_semantics.semantic_source_digest(payload["source"])
    assert fresh["elements"][0]["purpose"] == "corner_marker"
    assert fresh != descriptor
