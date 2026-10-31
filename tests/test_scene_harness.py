"""Local scene evidence and model proposals cannot change crop authority."""

import argparse
import base64
import hashlib
import json
import sys
from copy import deepcopy

import pytest
from PIL import Image

from engine.resources import DEFAULT_FONTS_DIR
from tools import scene_harness as harness


@pytest.fixture
def document(tmp_path):
    assets = tmp_path / "input/assets"
    assets.mkdir(parents=True)
    Image.new("RGBA", (60, 90), "#346a8b").save(assets / "photo.png")
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
    profile = {
        "schema_version": 1,
        "source_digest": harness._digest(source),
        "scene_digest": "a" * 64,
        "images": [
            {
                "element_id": "photo",
                "role": "unknown",
                "confidence": 0.0,
                "evidence": [],
                "allow_cover": False,
                "support_ids": [],
                "asset_sha256": hashlib.sha256((assets / "photo.png").read_bytes()).hexdigest(),
            }
        ],
    }
    return source, assets, DEFAULT_FONTS_DIR, profile


def _response(profile):
    return {
        "schema_version": 1,
        "source_digest": profile["source_digest"],
        "scene_digest": profile["scene_digest"],
        "images": [
            {
                "element_id": "photo",
                "asset_sha256": profile["images"][0]["asset_sha256"],
                "proposed_role": "background",
                "confidence": 0.9,
                "protected_regions": [
                    {"kind": "face", "box": [0.2, 0.3, 0.4, 0.2], "confidence": 0.8}
                ],
            }
        ],
    }


def test_bound_model_proposal_is_independent_and_never_grants_cover(document):
    _, _, _, profile = document
    response = _response(profile)
    before = deepcopy(profile)
    validated = harness.validate_model_response(response, profile)
    validated["images"][0]["protected_regions"][0]["box"][0] = 0.1
    assert response["images"][0]["protected_regions"][0]["box"][0] == 0.2
    assert profile == before
    assert profile["images"][0]["allow_cover"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "scene",
        "asset",
        "id",
        "duplicate",
        "authority",
        "role",
        "confidence",
        "region",
        "unknown_region",
        "nonfinite",
    ],
)
def test_model_proposals_reject_wrong_binding_or_unbounded_authority(document, fault):
    _, _, _, profile = document
    response = _response(profile)
    image = response["images"][0]
    if fault == "source":
        response["source_digest"] = "b" * 64
    if fault == "scene":
        response["scene_digest"] = "b" * 64
    if fault == "asset":
        image["asset_sha256"] = "b" * 64
    if fault == "id":
        image["element_id"] = "not-in-source"
    if fault == "duplicate":
        response["images"].append(deepcopy(image))
    if fault == "authority":
        image["allow_cover"] = True
    if fault == "role":
        image["proposed_role"] = ["background"]
    if fault == "confidence":
        image["confidence"] = True
    if fault == "region":
        image["protected_regions"][0]["box"] = [0.9, 0.1, 0.2, 0.2]
    if fault == "unknown_region":
        image["protected_regions"][0]["kind"] = "execute"
    if fault == "nonfinite":
        image["confidence"] = float("nan")
    with pytest.raises(ValueError):
        harness.validate_model_response(response, profile)


def test_local_model_command_receives_json_stdin_and_literal_argv():
    code = (
        "import json,sys; v=json.load(sys.stdin); "
        "json.dump({'received':v,'arg':sys.argv[1]},sys.stdout)"
    )
    literal = "a ; $(never-run) > untouched"
    response = harness.run_model([sys.executable, "-c", code, literal], {"source": "poster data"})
    assert response == {"received": {"source": "poster data"}, "arg": literal}


def test_local_model_timeout_and_stdout_are_bounded():
    with pytest.raises(ValueError, match="timeout"):
        harness.run_model([sys.executable, "-c", "import time; time.sleep(3)"], {}, timeout=0.05)
    with pytest.raises(ValueError, match="byte limit"):
        harness.run_model([sys.executable, "-c", f"print('x'*{harness.MAX_MODEL_BYTES + 100})"], {})
    with pytest.raises(ValueError):
        harness.model_command("echo arbitrary shell text")


def test_rejected_model_still_exports_native_deterministic_baseline(
    document, tmp_path, monkeypatch
):
    import engine.resize as resize

    source, assets, fonts, profile = document
    before = deepcopy(source)
    calls = []
    monkeypatch.setattr(harness, "_analyze", lambda *_args: deepcopy(profile))

    def baseline(value, target):
        calls.append(deepcopy(value))
        result = deepcopy(value)
        result["canvas"] = dict(target)
        return {"design": result, "state": {"name": "deterministic-test"}}

    monkeypatch.setattr(resize, "resize_with_state", baseline)
    invalid = tmp_path / "untrusted.json"
    response = _response(profile)
    response["images"][0]["allow_cover"] = True
    invalid.write_text(json.dumps(response), encoding="utf-8")
    out = tmp_path / "evidence"
    report = harness.inspect_scene(
        source, assets, fonts, out, targets=[source["canvas"]], model_response=invalid
    )
    assert report["model_review"]["status"] == "rejected"
    assert report["model_review"]["used_for_resize"] is False
    assert calls == [before]
    assert source == before
    assert report["source_immutable"] and report["resources_immutable"]
    assert report["targets"][0]["status"] == "contract_valid"
    with Image.open(out / "120x180.png") as image:
        assert image.size == (120, 180)
    assert json.loads((out / "scene.json").read_text())["images"][0]["allow_cover"] is False


def test_validated_model_also_remains_review_only(document, tmp_path, monkeypatch):
    source, assets, fonts, profile = document
    monkeypatch.setattr(harness, "_analyze", lambda *_args: deepcopy(profile))
    proposed = tmp_path / "proposal.json"
    proposed.write_text(json.dumps(_response(profile)), encoding="utf-8")
    report = harness.inspect_scene(source, assets, fonts, tmp_path / "out", model_response=proposed)
    assert report["model_review"]["status"] == "validated_proposal"
    assert report["model_review"]["used_for_resize"] is False
    assert report["targets"] == []


def test_protect_policy_passes_validated_prediction_to_resize(document, tmp_path, monkeypatch):
    import engine.resize as resize

    source, assets, fonts, profile = document
    monkeypatch.setattr(harness, "_analyze", lambda *_args: deepcopy(profile))
    calls = []

    def protected(value, target, *, scene_proposal):
        calls.append(deepcopy(scene_proposal))
        return {"design": deepcopy(value), "state": {"name": "test-protect"}}

    monkeypatch.setattr(resize, "resize_with_state", protected)
    proposed = tmp_path / "proposal.json"
    proposed.write_text(json.dumps(_response(profile)), encoding="utf-8")
    report = harness.inspect_scene(
        source,
        assets,
        fonts,
        tmp_path / "out",
        targets=[source["canvas"]],
        model_response=proposed,
        model_policy="protect",
    )
    assert calls == [_response(profile)]
    assert report["model_review"]["used_for_resize"] is True
    assert report["model_review"]["authority"] == "crop_veto_only"
    assert report["targets"][0]["status"] == "contract_valid"


def test_mid_run_implementation_change_is_labeled_incomplete(document, tmp_path, monkeypatch):
    source, assets, fonts, profile = document
    monkeypatch.setattr(harness, "_analyze", lambda *_args: deepcopy(profile))
    revisions = iter([{"engine/scene.py": "a" * 64}, {"engine/scene.py": "b" * 64}])
    monkeypatch.setattr(harness, "_implementation_hashes", lambda: next(revisions))
    report = harness.inspect_scene(source, assets, fonts, tmp_path / "out")
    assert report["implementation_unchanged"] is False
    assert report["implementation_digest"] != report["implementation_after_digest"]
    assert report["source_immutable"] is True


def test_asset_escape_and_input_overwrite_are_rejected(document, tmp_path):
    source, assets, fonts, _ = document
    source["elements"][0]["asset_id"] = "../outside.png"
    with pytest.raises(ValueError):
        harness.resource_hashes(source, assets, fonts)
    p = tmp_path / "original/source.json"
    p.parent.mkdir()
    p.write_text("{}")
    with pytest.raises(ValueError, match="separate"):
        harness.load_input(source_path=p, assets_dir=assets, fonts_dir=fonts, out=p.parent)
    assert p.read_text() == "{}"


def test_portable_json_bundle_uses_exact_existing_resource_transport(document, tmp_path):
    source, assets, _, _ = document
    font = DEFAULT_FONTS_DIR / "SolsticeSans-Regular.ttf"
    payload = {
        "source": source,
        "document": {"snapshot_id": "local", "page_id": "page", "artboard_id": "artboard"},
        "assets": [
            {
                "asset_id": "photo.png",
                "mime_type": "image/png",
                "data_base64": base64.b64encode((assets / "photo.png").read_bytes()).decode(),
            }
        ],
        "fonts": [
            {
                "font_family": "SolsticeSans-Regular",
                "postscript_name": "SolsticeSans-Regular",
                "weight": "regular",
                "style": "normal",
                "data_base64": base64.b64encode(font.read_bytes()).decode(),
            }
        ],
    }
    path = tmp_path / "source-bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    loaded, actual_assets, actual_fonts, receipt = harness.load_input(
        bundle=path, out=tmp_path / "out"
    )
    assert loaded == source
    assert (actual_assets / "photo.png").read_bytes() == (assets / "photo.png").read_bytes()
    assert receipt["fonts"][0]["sha256"] == hashlib.sha256(font.read_bytes()).hexdigest()
    assert (actual_fonts / "font-manifest.json").is_file()
    assert path.read_bytes() == before


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_duplicate_or_nonfinite_json_is_rejected(text):
    with pytest.raises(ValueError):
        harness._parse(text)


def test_target_limits_are_explicit():
    assert harness.target_dimensions("1600x900") == {"width": 1600, "height": 900}
    for value in ("0x10", "8193x1", "8000x8000", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            harness.target_dimensions(value)
