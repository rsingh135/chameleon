"""Three-view request preparation uses synthetic raw inputs, never a model or gold."""

import builtins
import hashlib
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin

from tools import purpose_evidence_request as evidence


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png(size, mode="RGB", color="red", note="unaltered original PNG metadata"):
    stream = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("source-note", note)
    Image.new(mode, size, color).save(stream, format="PNG", pnginfo=metadata)
    return stream.getvalue()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def native_result(element):
    alpha, checker = png((8, 6), "RGBA", (10, 20, 30, 70)), png((8, 6), color="#c0c0c0")
    receipt = {
        "source_element_sha256": sha(evidence._compact(element)),
        "resources_unchanged": True,
        "status": "available",
        "clipped": False,
        "native_single_original_layer": True,
        "frame_or_asset_fallback_used": False,
        "geometry_authority": False,
        "model_executed": False,
        "policy": {"version": "native-element-isolation-v1", "max_thumbnail_axis": 512},
        "implementation_sha256": "f" * 64,
        "alpha_png_sha256": sha(alpha),
        "checker_png_sha256": sha(checker),
    }
    receipt["receipt_digest"] = sha(evidence._compact(receipt))
    return {
        "status": "available",
        "reason": None,
        "alpha_png": alpha,
        "checker_png": checker,
        "receipt": receipt,
    }


@pytest.fixture
def raw_inputs(tmp_path, monkeypatch):
    source = {
        "canvas": {"width": 80, "height": 60},
        "elements": [
            {
                "id": "private-a",
                "type": "shape",
                "x": 2,
                "y": 3,
                "width": 30,
                "height": 20,
                "shape_type": "rectangle",
                "fill_color": "#abcdef",
                "z_index": 8,
            },
            {
                "id": "private-b",
                "type": "text",
                "x": 5,
                "y": 30,
                "width": 60,
                "height": 20,
                "content": "Original caption",
                "font_family": "ExactSans",
                "font_size": 14,
                "color": "#ffffff",
                "z_index": 2,
            },
        ],
    }
    source_path = tmp_path / "source.json"
    write_json(source_path, source)
    source_png = tmp_path / "source.png"
    source_png.write_bytes(png((80, 60)))
    contexts, records = {}, []
    for index, element in enumerate(source["elements"]):
        path = tmp_path / f"context-{index}.png"
        path.write_bytes(png((16, 10), color="blue" if index else "green"))
        contexts[element["id"]] = {"path": str(path), "sha256": sha(path.read_bytes())}
        records.append(
            {
                "element_id": element["id"],
                "element_index": index,
                "frame_xywh": [element[k] for k in ("x", "y", "width", "height")],
                "context_sha256": sha(path.read_bytes()),
            }
        )
    receipt = {
        "schema_version": 1,
        "protocol": "synthetic-native-context-binding-v1",
        "source_json_sha256": sha(source_path.read_bytes()),
        "source_render_sha256": sha(source_png.read_bytes()),
        "records": records,
    }
    receipt_path = tmp_path / "context-receipt.json"
    write_json(receipt_path, receipt)
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    calls = []

    def render(element, *_):
        calls.append(deepcopy(element))
        return native_result(element)

    monkeypatch.setattr(evidence, "_render", render)
    return SimpleNamespace(
        source=source,
        source_path=source_path,
        source_png=source_png,
        contexts=contexts,
        receipt=receipt,
        receipt_path=receipt_path,
        assets=assets,
        fonts=fonts,
        out=tmp_path / "prepared",
        calls=calls,
    )


def prepare(raw, **kwargs):
    return evidence.prepare_purpose_evidence(
        raw.source_path,
        {"path": str(raw.source_png), "sha256": sha(raw.source_png.read_bytes())},
        raw.contexts,
        context_receipt=raw.receipt_path,
        assets_dir=raw.assets,
        fonts_dir=raw.fonts,
        family_id="synthetic-family",
        out=raw.out,
        **kwargs,
    )


def test_request_copies_exact_png_bytes_and_uses_declared_three_view_prompt(raw_inputs):
    raw = raw_inputs
    manifest = prepare(raw, inference_identity={"purpose_views": 3, "adapter_sha256": "a" * 64})
    request_bytes = (raw.out / "request.json").read_bytes()
    request = json.loads(request_bytes)
    assert request["protocol"] == "visual-adapter-label-free-evaluation-v1"
    assert request["task"] == "purpose" and request["purpose_views"] == 3
    assert manifest["request"]["sha256"] == sha(request_bytes)
    assert manifest["ordered_views"] == list(evidence.ORDERED_VIEWS)
    assert manifest["coverage"] == {
        "source_count": 2,
        "admitted_count": 2,
        "excluded_count": 0,
        "excluded": [],
    }
    for row, element in zip(request["records"], raw.source["elements"], strict=True):
        images = row["images"]
        assert len(images) == 3
        assert (raw.out / images[0]["path"]).read_bytes() == raw.source_png.read_bytes()
        assert (raw.out / images[1]["path"]).read_bytes() == Path(
            raw.contexts[element["id"]]["path"]
        ).read_bytes()
        assert "third image" in row["prompt"] and "private-" not in row["prompt"]
        assert row["control"] == "original" and "label" not in row
    assert manifest["records"][0]["paint_order_index"] == 1
    assert manifest["records"][1]["paint_order_index"] == 0
    assert all(len(row["views"]) == 3 and row["alpha_image"] for row in manifest["records"])
    assert (raw.out / "source.json").read_bytes() == raw.source_path.read_bytes()
    assert not manifest["inference_executed"] and not manifest["promotion_eligible"]
    assert not manifest["quality_approved"]
    assert manifest["inference_identity"]["scope"] == "expected_only_no_execution"
    assert len(raw.calls) == 2 and raw.source == json.loads(raw.source_path.read_bytes())
    assert (raw.out / "READY.json").is_file()


@pytest.mark.parametrize(
    "field",
    ["source_json_sha256", "source_render_sha256", "frame", "index", "context_sha256", "duplicate"],
)
def test_context_receipt_must_bind_exact_source_element_and_context(raw_inputs, field):
    raw = raw_inputs
    if field in {"source_json_sha256", "source_render_sha256"}:
        raw.receipt[field] = "0" * 64
    elif field == "frame":
        raw.receipt["records"][0]["frame_xywh"][0] += 1
    elif field == "index":
        raw.receipt["records"][0]["element_index"] = 1
    elif field == "duplicate":
        raw.receipt["records"].append(raw.receipt["records"][0])
    else:
        raw.receipt["records"][0][field] = "0" * 64
    write_json(raw.receipt_path, raw.receipt)
    with pytest.raises(ValueError, match="Context"):
        prepare(raw)
    assert raw.calls == [] and not raw.out.exists()


def test_unavailable_native_view_fails_without_publishing_request_or_falling_back(
    raw_inputs, monkeypatch
):
    monkeypatch.setattr(
        evidence, "_render", lambda *_: {"status": "unavailable", "reason": "clipped"}
    )
    with pytest.raises(ValueError, match="clipped"):
        prepare(raw_inputs)
    assert not (raw_inputs.out / "request.json").exists()
    assert not (raw_inputs.out / "READY.json").exists()


def test_explicit_admission_exclusion_never_invokes_paint_and_is_visible(raw_inputs):
    manifest = prepare(raw_inputs, excluded_element_ids=["private-b"])
    assert len(raw_inputs.calls) == 1 and raw_inputs.calls[0]["id"] == "private-a"
    assert manifest["coverage"]["excluded"] == [
        {"element_id": "private-b", "element_index": 1, "reason": "explicit_admission_exclusion"}
    ]


def test_exclusion_policy_records_missing_third_view_without_two_view_row(raw_inputs, monkeypatch):
    def render(element, *_):
        if element["id"] == "private-a":
            return {"status": "unavailable", "reason": "unsupported effect"}
        return native_result(element)

    monkeypatch.setattr(evidence, "_render", render)
    manifest = prepare(raw_inputs, unavailable_policy="exclude")
    request = json.loads((raw_inputs.out / "request.json").read_bytes())
    assert len(request["records"]) == 1 and len(request["records"][0]["images"]) == 3
    assert manifest["coverage"]["excluded_count"] == 1


def test_no_gold_annotations_or_model_modules_are_read_or_copied(raw_inputs, monkeypatch):
    raw = raw_inputs
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "accelerate"}
        assert not name.endswith(("worker", "train_sft"))
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    prepare(raw)
    raw.out = raw.out.parent / "rejected"
    raw.receipt["records"][0]["gold"] = "synthetic-only-for-rejection"
    write_json(raw.receipt_path, raw.receipt)
    with pytest.raises(ValueError, match="gold"):
        prepare(raw)
    assert not raw.out.exists()


def test_mutated_input_before_publication_is_rejected(raw_inputs, monkeypatch):
    def render(element, *_):
        raw_inputs.source_png.write_bytes(png((80, 60), color="black"))
        return native_result(element)

    monkeypatch.setattr(evidence, "_render", render)
    with pytest.raises(ValueError, match="changed before request publication"):
        prepare(raw_inputs)
    assert not (raw_inputs.out / "READY.json").exists()


def test_existing_or_resource_overlapping_output_is_rejected(raw_inputs):
    raw_inputs.out.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        prepare(raw_inputs)
    raw_inputs.out = raw_inputs.assets / "generated"
    with pytest.raises(ValueError, match="independent"):
        prepare(raw_inputs)


def test_resource_mutation_during_paint_cannot_publish_ready(raw_inputs, monkeypatch):
    note = raw_inputs.fonts / "font-note.json"
    note.write_text("{}", encoding="utf-8")

    def render(element, *_):
        note.write_text('{"changed":true}', encoding="utf-8")
        return native_result(element)

    monkeypatch.setattr(evidence, "_render", render)
    with pytest.raises(RuntimeError, match="changed"):
        prepare(raw_inputs)
    assert not (raw_inputs.out / "READY.json").exists()


def test_native_byte_identity_mismatch_is_rejected(raw_inputs, monkeypatch):
    def render(element, *_):
        result = native_result(element)
        result["checker_png"] = png((8, 6), color="black")
        return result

    monkeypatch.setattr(evidence, "_render", render)
    with pytest.raises(ValueError, match="PNG bytes differ"):
        prepare(raw_inputs)
    assert not (raw_inputs.out / "READY.json").exists()


def test_annotation_field_cannot_hide_in_a_context_receipt(raw_inputs):
    raw_inputs.receipt["records"][0]["purpose"] = "synthetic-not-an-input-feature"
    write_json(raw_inputs.receipt_path, raw_inputs.receipt)
    with pytest.raises(ValueError, match="Context"):
        prepare(raw_inputs)
    assert raw_inputs.calls == [] and not raw_inputs.out.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unavailable"),
        ("clipped", True),
        ("native_single_original_layer", False),
        ("frame_or_asset_fallback_used", True),
        ("geometry_authority", True),
        ("model_executed", True),
    ],
)
def test_self_consistent_native_receipt_cannot_hide_clipping_or_authority(
    raw_inputs, monkeypatch, field, value
):
    def render(element, *_):
        result = native_result(element)
        receipt = result["receipt"]
        receipt[field] = value
        receipt["receipt_digest"] = sha(
            evidence._compact(
                {key: value for key, value in receipt.items() if key != "receipt_digest"}
            )
        )
        return result

    monkeypatch.setattr(evidence, "_render", render)
    with pytest.raises(ValueError, match="Native isolated receipt"):
        prepare(raw_inputs)
    assert not (raw_inputs.out / "READY.json").exists()
