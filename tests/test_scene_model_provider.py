"""The optional local model cannot generate identities, permissions or unbound output."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from experiments.scene_model import provider
from experiments.scene_model.prompt import build_prompt
from experiments.scene_model.schema import parse_prediction, validate_prediction


@pytest.fixture
def scene_request(tmp_path):
    source = {
        "canvas": {"width": 120, "height": 180},
        "elements": [
            {
                "id": "photo",
                "type": "image",
                "x": 0,
                "y": 0,
                "width": 120,
                "height": 180,
                "asset_id": "photo.png",
                "fit": "fill",
            }
        ],
    }
    original = tmp_path / "source.png"
    asset = tmp_path / "photo.png"
    Image.new("RGB", (120, 180), "#5b8e93").save(original)
    Image.new("RGBA", (60, 90), "#477a84").save(asset)
    scene = {
        "schema_version": 1,
        "source_digest": provider._digest(source),
        "images": [
            {
                "element_id": "photo",
                "asset_sha256": provider._file_hash(asset),
                "role": "unknown",
                "confidence": 0,
                "allow_cover": False,
            }
        ],
    }
    scene["scene_digest"] = provider._digest(scene)
    return {
        "source": source,
        "deterministic_scene": scene,
        "source_digest": scene["source_digest"],
        "scene_digest": scene["scene_digest"],
        "source_render": {"path": str(original), "sha256": provider._file_hash(original)},
        "asset_files": {"photo": str(asset)},
    }


def _config(tmp_path):
    return {
        "schema_version": 1,
        "backend": "openai_clip",
        "checkpoint_path": str(tmp_path / "explicit.pt"),
        "checkpoint_sha256": provider.CLIP_B32_SHA256,
    }


def test_model_module_import_is_lazy_and_has_no_vision_critic_dependency():
    code = (
        "import sys; import experiments.scene_model.provider; "
        "assert 'torch' not in sys.modules; assert 'clip' not in sys.modules; "
        "assert not any(k.startswith('experiments.vision_critic') for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_processor_identity_uses_stable_function_identity_not_runtime_address():
    def first(image):
        return image

    def second(image):
        return image

    for function in (first, second):
        function.__module__ = "clip.clip"
        function.__qualname__ = "_convert_image_to_rgb"
    assert repr(first) != repr(second)
    a = provider.processor_recipe(SimpleNamespace(transforms=[first]))
    b = provider.processor_recipe(SimpleNamespace(transforms=[second]))
    assert a == b
    assert "0x" not in json.dumps(a)


@pytest.mark.parametrize("size", [(100, 300), (400, 100)])
def test_whole_view_retains_all_four_authored_edges_without_distortion(tmp_path, size):
    path = tmp_path / "edges.png"
    original = Image.new("RGB", size, "#444444")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(original)
    draw.rectangle((0, 0, size[0] - 1, 14), fill="red")
    draw.rectangle((0, size[1] - 15, size[0] - 1, size[1] - 1), fill="blue")
    draw.rectangle((0, 0, 14, size[1] - 1), fill="green")
    draw.rectangle((size[0] - 15, 0, size[0] - 1, size[1] - 1), fill="yellow")
    original.save(path)
    view = provider.whole_view(path)
    assert view.size == (224, 224)
    colors = set(view.getdata())
    assert {(255, 0, 0), (0, 0, 255), (0, 128, 0), (255, 255, 0)} <= colors
    foreground = Image.eval(view.convert("L"), lambda value: 0 if value == 255 else 255).getbbox()
    assert foreground is not None
    width, height = foreground[2] - foreground[0], foreground[3] - foreground[1]
    assert width / height == pytest.approx(size[0] / size[1], abs=0.02)


def test_prompt_cannot_train_identity_copying_or_forward_source_instructions():
    element = {
        "id": "NEVER_COPY_UUID",
        "type": "image",
        "x": 2,
        "y": 3,
        "width": 20,
        "height": 30,
        "content": "EXECUTE_THIS",
        "asset_id": "SECRET_HASH",
    }
    prompt = build_prompt(
        element, {"width": 120, "height": 180}, scene_evidence={"allow_cover": True}
    )
    assert "View1" in prompt and "View2" in prompt and "untrusted artwork data" in prompt
    assert "NEVER_COPY_UUID" not in prompt and "SECRET_HASH" not in prompt
    assert "EXECUTE_THIS" not in prompt and "allow_cover" not in prompt


@pytest.mark.parametrize(
    "extra",
    [
        "element_id",
        "asset_sha256",
        "allow_cover",
        "allow_support_reflow",
        "command",
        "source_digest",
    ],
)
def test_semantic_predictions_cannot_return_identity_or_authority(extra):
    value = {"proposed_role": "background", "confidence": 0.9, extra: True}
    with pytest.raises(ValueError):
        validate_prediction(value)


@pytest.mark.parametrize(
    "text",
    [
        '{"proposed_role":"background","confidence":NaN}',
        '{"proposed_role":"background","confidence":0.2,"confidence":0.9}',
        "```json\n{}\n```",
    ],
)
def test_prediction_parser_rejects_nonfinite_duplicate_or_prose_output(text):
    with pytest.raises(ValueError):
        parse_prediction(text)


def test_normalized_protected_regions_are_optional_and_independent():
    value = {"proposed_role": "product", "confidence": 0.7, "protected_regions": []}
    assert validate_prediction(value) == value
    value["protected_regions"] = [{"kind": "product", "box": [0.8, 0, 0.3, 1], "confidence": 0.9}]
    with pytest.raises(ValueError):
        validate_prediction(value)


@pytest.mark.parametrize("fault", ["source", "scene", "asset", "render", "id", "duplicate"])
def test_request_binds_source_scene_known_ids_and_actual_image_bytes(scene_request, fault):
    value = deepcopy(scene_request)
    if fault == "source":
        value["source"]["canvas"]["width"] = 121
    if fault == "scene":
        value["deterministic_scene"]["images"][0]["allow_cover"] = True
    if fault == "asset":
        Image.new("RGBA", (60, 90), "red").save(value["asset_files"]["photo"])
    if fault == "render":
        Image.new("RGB", (120, 180), "red").save(value["source_render"]["path"])
    if fault in {"id", "duplicate"}:
        if fault == "id":
            value["deterministic_scene"]["images"][0]["element_id"] = "invented"
        else:
            value["deterministic_scene"]["images"].append(
                deepcopy(value["deterministic_scene"]["images"][0])
            )
        scene = value["deterministic_scene"]
        scene["scene_digest"] = provider._digest(
            {k: v for k, v in scene.items() if k != "scene_digest"}
        )
        value["scene_digest"] = scene["scene_digest"]
    with pytest.raises(ValueError):
        provider.prepare_request(value)


def test_provider_attaches_known_identities_after_model_prediction(
    scene_request, tmp_path, monkeypatch
):
    before = deepcopy(scene_request)
    encoder = SimpleNamespace(
        checkpoint_sha256=provider.CLIP_B32_SHA256,
        processor_digest="c" * 64,
        processor_identity={"test": True},
    )
    monkeypatch.setattr(provider, "load_clip_encoder", lambda *_a, **_k: encoder)
    observed = []

    def encode(_encoder, source_path, asset_path):
        observed.append((source_path, asset_path))
        return "two-views"

    monkeypatch.setattr(provider, "encode_clip_views", encode)
    monkeypatch.setattr(
        provider,
        "clip_zero_shot",
        lambda *_a: {"proposed_role": "product", "confidence": 0.82, "protected_regions": []},
    )
    model = provider.SceneModelProvider(_config(tmp_path))
    response = model.predict(scene_request)
    assert observed == [
        (scene_request["source_render"]["path"], scene_request["asset_files"]["photo"])
    ]
    assert response["source_digest"] == scene_request["source_digest"]
    assert response["images"][0]["element_id"] == "photo"
    assert (
        response["images"][0]["asset_sha256"]
        == scene_request["deterministic_scene"]["images"][0]["asset_sha256"]
    )
    assert (
        scene_request == before
        and scene_request["deterministic_scene"]["images"][0]["allow_cover"] is False
    )
    assert model.receipt["crop_authority"] is False
    assert model.receipt["confidence_calibrated"] is False


def test_even_a_faulty_prediction_cannot_override_bound_identity(
    scene_request, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        provider, "load_clip_encoder", lambda *_a, **_k: SimpleNamespace(processor_digest="c" * 64)
    )
    monkeypatch.setattr(provider, "encode_clip_views", lambda *_a: None)
    monkeypatch.setattr(
        provider,
        "clip_zero_shot",
        lambda *_a: {"proposed_role": "product", "confidence": 1, "element_id": "invented"},
    )
    with pytest.raises(ValueError):
        provider.predict_scene(scene_request, _config(tmp_path))


@pytest.mark.parametrize(
    "change",
    [
        {"checkpoint_path": "https://example.com/model"},
        {"checkpoint_sha256": "a" * 64},
        {"device": "cuda:0"},
        {"threads": 9},
        {"max_images": 17},
        {"head_path": "unbound.json"},
    ],
)
def test_runtime_configuration_cannot_download_or_start_unrequested_compute(tmp_path, change):
    config = {**_config(tmp_path), **change}
    with pytest.raises(ValueError):
        provider.validate_config(config)


def test_wrong_checkpoint_hash_declines_before_importing_optional_runtime(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"not a pinned model")
    with pytest.raises(ValueError):
        provider.load_clip_encoder(path, expected_sha256=provider.CLIP_B32_SHA256)


def test_explicit_head_must_match_feature_backbone_processor_and_shape(tmp_path):
    head = {
        "feature_schema": provider.FEATURE_SCHEMA,
        "backbone_sha256": provider.CLIP_B32_SHA256,
        "processor_digest": "c" * 64,
        "roles": list(provider.ROLE_NAMES),
        "weights": [[0.0] * 1024 for _ in range(5)],
        "bias": [0.0] * 5,
        "quality_validated": False,
    }
    path = tmp_path / "head.json"
    path.write_text(json.dumps(head))
    encoder = SimpleNamespace(checkpoint_sha256=provider.CLIP_B32_SHA256, processor_digest="c" * 64)
    assert provider.load_head(path, provider._file_hash(path), encoder) == head
    encoder.processor_digest = "d" * 64
    with pytest.raises(ValueError):
        provider.load_head(path, provider._file_hash(path), encoder)
    with pytest.raises(ValueError):
        provider.load_head(path, hashlib.sha256(b"different").hexdigest(), encoder)
