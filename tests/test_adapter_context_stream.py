"""Native pixel crops plus synthetic file envelopes; no model job is launched."""

import io
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from experiments.agentic_graph import adapter_analysis as bridge
from experiments.agentic_graph.adapter_context import (
    build_source_contexts,
    build_source_views,
    context_box,
    sha_file,
    verify_source_views,
)
from experiments.agentic_graph.adapter_stream import handle, main


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def source_png(tmp_path):
    path = tmp_path / "native-original.png"
    image = Image.new("RGB", (100, 200), "white")
    image.putpixel((5, 15), (255, 0, 0))
    image.putpixel((50, 40), (0, 180, 0))
    image.save(path)
    source = {
        "canvas": {"width": 100, "height": 200},
        "elements": [
            {
                "id": "opaque-object",
                "type": "shape",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 200,
                "z_index": 0,
            },
            {
                "id": "arbitrary-copy",
                "type": "text",
                "x": 10,
                "y": 20,
                "width": 80,
                "height": 30,
                "content": "Gather",
                "z_index": 1,
            },
        ],
    }
    return source, {"path": str(path), "sha256": sha_file(path)}


def test_context_crop_matches_training_padding_and_original_pixels_without_overlay(
    source_png, tmp_path
):
    source, render = source_png
    before = Path(render["path"]).read_bytes()
    result = build_source_contexts(source, render, tmp_path / "contexts")
    assert context_box(source["elements"][1], (100, 200)) == (2, 12, 98, 58)
    row = result["receipt"]["records"][1]
    assert row["context_box_pixels"] == [2, 12, 98, 58]
    path = result["context_images"]["arbitrary-copy"]["path"]
    with Image.open(render["path"]) as original, Image.open(path) as crop:
        expected = original.crop((2, 12, 98, 58))
        assert crop.tobytes() == expected.tobytes()
        assert crop.getpixel((3, 3)) == (255, 0, 0)
    assert Path(render["path"]).read_bytes() == before
    assert result["receipt"]["model_executed"] is False
    assert result["receipt"]["supervision_read"] is False
    assert all(row["labels_or_id_overlays_added"] is False for row in result["receipt"]["records"])


def test_bleed_is_clamped_and_completely_invisible_elements_abstain(source_png, tmp_path):
    source, render = deepcopy(source_png)
    source["elements"].extend(
        [
            {"id": "bleed", "type": "shape", "x": -5, "y": 195, "width": 10, "height": 20},
            {"id": "offcanvas", "type": "shape", "x": 200, "y": 0, "width": 10, "height": 20},
        ]
    )
    result = build_source_contexts(source, render, tmp_path / "contexts")
    assert context_box(source["elements"][-2], (100, 200)) == (0, 187, 13, 200)
    assert "bleed" in result["context_images"]
    assert "offcanvas" not in result["context_images"]
    assert result["receipt"]["records"][-1]["abstention_reason"] == "element_outside_source_canvas"
    bundle = bridge.build_adapter_request(source, render, result["context_images"])
    assert bundle["initial_abstentions"] == [
        {"element_id": "offcanvas", "reason": "missing_context_image"}
    ]


def test_wrong_source_hash_or_preview_dimensions_abort_before_scoring(source_png, tmp_path):
    source, render = source_png
    with pytest.raises(ValueError, match="declared bytes"):
        build_source_contexts(source, {**render, "sha256": "0" * 64}, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    wrong = deepcopy(source)
    wrong["canvas"]["width"] = 101
    with pytest.raises(ValueError, match="native canvas"):
        build_source_contexts(wrong, render, tmp_path / "preview")
    with pytest.raises(ValueError, match="expected source identity"):
        build_source_contexts(source, render, tmp_path / "stale", expected_source_digest="0" * 64)


def prepare_job(source_png, tmp_path, *, purpose_views=2, view_builder=None):
    source, render = source_png
    root = tmp_path / "jobs"
    prepared = handle(
        {"operation": "prepare", "job_id": "opaque-job", "source": source, "source_render": render},
        artifact_root=root,
        purpose_views=purpose_views,
        view_builder=view_builder,
    )
    assert prepared["worker_request_ready"] and not prepared["model_executed"]
    job = root / "opaque-job"
    bundle = json.loads((job / "binding.json").read_text())
    evaluation = job / "evaluation"
    records = [
        {
            **deepcopy(row),
            "family_id": "independent",
            "control": "independent",
            "split": "external_evaluation",
        }
        for row in bundle["request"]["records"]
    ]
    dataset = {
        "schema_version": 1,
        "task": "purpose",
        "classes": list(bridge.CLASSES),
        "records": records,
        "provenance": {"request_sha256": bundle["request_sha256"]},
    }
    if purpose_views == 3:
        dataset["purpose_views"] = 3
    write(evaluation / "data/dataset.json", dataset)
    manifest = {
        "repo_id": "test-only/base",
        "revision": "a" * 40,
        "files": {"model.safetensors": {"sha256": "b" * 64}},
    }
    write(evaluation / "base_identity.json", manifest)
    code = evaluation / "code/experiments/visual_finetune/worker.py"
    code.parent.mkdir(parents=True)
    code.write_text("# Synthetic protocol fixture; never executed.\n")
    recipe = {
        "steps": 0,
        "external_evaluation": True,
        "inference_only": True,
        "dataset": "data/dataset.json",
        "dataset_sha256": sha_file(evaluation / "data/dataset.json"),
        "adapter_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
        "base_manifest": "base_identity.json",
        "base_manifest_sha256": sha_file(evaluation / "base_identity.json"),
        "code_sha256": {"worker.py": sha_file(code)},
    }
    if purpose_views == 3:
        recipe["purpose_views"] = 3
    write(evaluation / "recipe.json", recipe)
    return root, job, evaluation, bundle


def fake_completed_output(evaluation, bundle, expected, *, strength=6):
    logits = {key: strength if key == "headline" else 0 for key in bridge.CLASSES}
    maximum = max(logits.values())
    denominator = sum(math.exp(v - maximum) for v in logits.values())
    predictions = [
        {
            "record_id": row["record_id"],
            "prediction": "headline",
            "closed_scores": logits,
            "closed_probabilities": {
                key: math.exp(v - maximum) / denominator for key, v in logits.items()
            },
            "raw_next_token_id": 100,
            "raw_next_token": "headline",
            "raw_next_token_is_allowed": True,
            "input_token_count": 100,
            "image_grid_thw": [[1, 8, 8]] * bundle["request"].get("purpose_views", 2),
            "input_image_sha256": row["input_image_sha256"],
        }
        for row in bundle["records"]
    ]
    write(
        evaluation / "external_adapted.json",
        {"scoring": bridge.SCORING, "metrics": None, "predictions": predictions},
    )
    receipt = {
        **deepcopy(expected),
        "execution_success": True,
        "task": "purpose",
        "request_sha256": bundle["request_sha256"],
        "prediction_file_sha256": sha_file(evaluation / "external_adapted.json"),
        "quality_approved": False,
        "experimental_only": True,
    }
    write(evaluation / "inference_receipt.json", receipt)


def test_transport_freezes_before_inference_and_is_off_unless_operator_diagnostic(
    source_png, tmp_path
):
    root, _job, evaluation, bundle = prepare_job(source_png, tmp_path)
    frozen = handle(
        {"operation": "freeze", "job_id": "opaque-job", "evaluation_dir": str(evaluation)},
        artifact_root=root,
    )
    assert frozen["model_executed"] is False
    fake_completed_output(evaluation, bundle, frozen["prepared_identity"])
    completed = handle({"operation": "complete", "job_id": "opaque-job"}, artifact_root=root)
    assert completed["ok"] and completed["result"]["executed"]
    assert completed["result"]["analysis"]["abstain"]
    diagnostic = handle(
        {"operation": "complete", "job_id": "opaque-job"}, artifact_root=root, diagnostic=True
    )
    assert diagnostic["result"]["analysis"]["observations"]
    assert not diagnostic["result"]["trace"]["deployment_enabled"]


def test_transport_rejects_late_freeze_and_source_pixel_changes(source_png, tmp_path):
    root, _job, evaluation, bundle = prepare_job(source_png, tmp_path)
    (evaluation / "phases.jsonl").write_text('{"phase":"started"}\n')
    with pytest.raises(ValueError, match="before model execution"):
        handle(
            {"operation": "freeze", "job_id": "opaque-job", "evaluation_dir": str(evaluation)},
            artifact_root=root,
        )
    (evaluation / "phases.jsonl").unlink()
    frozen = handle(
        {"operation": "freeze", "job_id": "opaque-job", "evaluation_dir": str(evaluation)},
        artifact_root=root,
    )
    fake_completed_output(evaluation, bundle, frozen["prepared_identity"])
    Image.new("RGB", (100, 200), "black").save(source_png[1]["path"])
    with pytest.raises(ValueError, match="PNG changed"):
        handle(
            {"operation": "complete", "job_id": "opaque-job"}, artifact_root=root, diagnostic=True
        )


@pytest.mark.parametrize("where", ["recipe", "dataset"])
def test_future_three_view_adapter_contract_never_silently_uses_two_images(
    source_png, tmp_path, where
):
    root, _job, evaluation, _bundle = prepare_job(source_png, tmp_path)
    recipe = json.loads((evaluation / "recipe.json").read_text())
    if where == "recipe":
        recipe["purpose_views"] = 3
    else:
        path = evaluation / "data/dataset.json"
        dataset = json.loads(path.read_text())
        dataset["purpose_views"] = 3
        write(path, dataset)
        recipe["dataset_sha256"] = sha_file(path)
    write(evaluation / "recipe.json", recipe)
    with pytest.raises(ValueError, match="two-view"):
        handle(
            {"operation": "freeze", "job_id": "opaque-job", "evaluation_dir": str(evaluation)},
            artifact_root=root,
        )


def test_freeze_rejects_prepared_gold_labels_even_when_dataset_hash_is_updated(
    source_png, tmp_path
):
    root, _job, evaluation, _bundle = prepare_job(source_png, tmp_path)
    dataset_path = evaluation / "data/dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["records"][0]["label"] = "headline"
    write(dataset_path, dataset)
    recipe = json.loads((evaluation / "recipe.json").read_text())
    recipe["dataset_sha256"] = sha_file(dataset_path)
    write(evaluation / "recipe.json", recipe)
    with pytest.raises(ValueError, match="labels"):
        handle(
            {"operation": "freeze", "job_id": "opaque-job", "evaluation_dir": str(evaluation)},
            artifact_root=root,
        )


def test_jsonl_transport_rejects_injected_flags_and_keeps_serving_without_model_imports(
    source_png, tmp_path
):
    source, render = source_png
    packets = [
        {
            "operation": "prepare",
            "job_id": "invalid",
            "source": source,
            "source_render": render,
            "mode": "diagnostic",
            "label": "headline",
        },
        {"operation": "prepare", "job_id": "valid", "source": source, "source_render": render},
    ]
    stdin, stdout = io.StringIO("".join(json.dumps(row) + "\n" for row in packets)), io.StringIO()
    assert main(["--artifact-root", str(tmp_path / "transport")], stdin=stdin, stdout=stdout) == 0
    rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert not rows[0]["ok"] and not rows[0]["model_executed"]
    assert rows[1]["ok"] and rows[1]["record_count"] == 2
    assert not rows[1]["model_executed"] and not rows[1]["deployment_enabled"]


@pytest.fixture
def native_source(tmp_path):
    """Tiny real native source/isolated paint; no model or training artifact."""
    from engine.eval.render import render_image

    source = {
        "canvas": {"width": 100, "height": 200},
        "elements": [
            {
                "id": "opaque-native-shape",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 40,
                "fill_color": "rgba(255,0,0,1)",
                "opacity": 0.6,
            }
        ],
    }
    source_path = tmp_path / "native-source.png"
    render_image(source, max_size=200).save(source_path)
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    font_name = "SolsticeSans-Regular.ttf"
    (fonts / font_name).write_bytes(
        (Path(__file__).resolve().parents[1] / "brand_kit/fonts" / font_name).read_bytes()
    )
    return source, {"path": str(source_path), "sha256": sha_file(source_path)}, assets, fonts


def native_builder(native_source):
    def build(source, render, destination):
        return build_source_views(
            source,
            render,
            destination,
            purpose_views=3,
            assets_dir=native_source[2],
            fonts_dir=native_source[3],
        )

    return build


def test_explicit_three_view_builder_reuses_exact_native_png_bytes_and_resources(
    native_source, tmp_path
):
    from tools.native_element_evidence import ISOLATION_V1, render_isolated_element

    source, render, assets, fonts = native_source
    before = Path(render["path"]).read_bytes()
    direct = render_isolated_element(
        source["elements"][0], assets_dir=assets, fonts_dir=fonts, policy=ISOLATION_V1
    )
    assert direct["status"] == "available", direct["reason"]
    prepared = build_source_views(
        source, render, tmp_path / "views", purpose_views=3, assets_dir=assets, fonts_dir=fonts
    )
    key = source["elements"][0]["id"]
    assert Path(prepared["isolated_images"][key]["path"]).read_bytes() == direct["checker_png"]
    assert (tmp_path / "views/element-0000-alpha.png").read_bytes() == direct["alpha_png"]
    receipt = prepared["isolated_receipts"][key]
    assert receipt["resource_identity"] == direct["receipt"]["resource_identity"]
    assert receipt["implementation_sha256"] == direct["receipt"]["implementation_sha256"]
    assert not receipt["clipped"] and not receipt["geometry_authority"]
    assert Path(render["path"]).read_bytes() == before
    bundle = bridge.build_adapter_request(
        source,
        render,
        prepared["context_images"],
        purpose_views=3,
        isolated_images=prepared["isolated_images"],
        isolated_receipts=prepared["isolated_receipts"],
    )
    verify_source_views(source, render, bundle)
    assert prepared["receipt"]["supervision_read"] is False


def test_unavailable_native_paint_abstains_without_a_frame_or_asset_fallback(
    native_source, tmp_path
):
    source, render, assets, fonts = native_source
    source = deepcopy(source)
    source["elements"][0]["stroke_width"] = 2
    prepared = build_source_views(
        source,
        render,
        tmp_path / "unsupported",
        purpose_views=3,
        assets_dir=assets,
        fonts_dir=fonts,
    )
    assert prepared["isolated_images"] == {}
    bundle = bridge.build_adapter_request(
        source,
        render,
        prepared["context_images"],
        purpose_views=3,
        isolated_images=prepared["isolated_images"],
        isolated_receipts=prepared["isolated_receipts"],
    )
    assert not bundle["request"]["records"]
    assert bundle["initial_abstentions"][0]["reason"] == "native_isolation_unavailable"
    assert not list((tmp_path / "unsupported").glob("*-isolated.png"))
    assert not prepared["receipt"]["model_executed"]


def test_native_view_resources_are_explicit_and_bound_before_preparation(native_source, tmp_path):
    source, render, assets, fonts = native_source
    with pytest.raises(ValueError, match="explicit trusted"):
        build_source_views(source, render, tmp_path / "missing", purpose_views=3)
    with pytest.raises(ValueError, match="resources differ"):
        build_source_views(
            source,
            render,
            tmp_path / "wrong",
            purpose_views=3,
            assets_dir=assets,
            fonts_dir=fonts,
            expected_resource_digest="0" * 64,
        )
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(ValueError, match="Two-view"):
        build_source_views(source, render, tmp_path / "two", assets_dir=assets, fonts_dir=fonts)


def test_three_view_transport_freezes_exact_profile_and_completes_real_native_evidence(
    native_source, tmp_path
):
    root, _job, evaluation, bundle = prepare_job(
        native_source[:2], tmp_path, purpose_views=3, view_builder=native_builder(native_source)
    )
    frozen = handle(
        {"operation": "freeze", "job_id": "opaque-job", "evaluation_dir": str(evaluation)},
        artifact_root=root,
        purpose_views=3,
    )
    fake_completed_output(evaluation, bundle, frozen["prepared_identity"])
    completed = handle(
        {"operation": "complete", "job_id": "opaque-job"},
        artifact_root=root,
        purpose_views=3,
        diagnostic=True,
    )
    assert completed["ok"] and completed["result"]["trace"]["purpose_views"] == 3
    assert completed["result"]["trace"]["model_visual_input_count"] == 3
    assert completed["result"]["trace"]["native_isolation"]
    assert not completed["result"]["trace"]["deployment_enabled"]
    with pytest.raises(ValueError, match="explicit view profile"):
        handle({"operation": "complete", "job_id": "opaque-job"}, artifact_root=root)
    isolated = bundle["request"]["records"][0]["images"][2]["path"]
    Path(isolated).write_bytes(b"changed native evidence")
    with pytest.raises(ValueError, match="PNG changed"):
        handle(
            {"operation": "complete", "job_id": "opaque-job"},
            artifact_root=root,
            purpose_views=3,
            diagnostic=True,
        )


@pytest.mark.parametrize("where", ["recipe", "dataset"])
def test_three_view_transport_rejects_two_view_or_missing_prepared_contract(
    native_source, tmp_path, where
):
    root, _job, evaluation, _bundle = prepare_job(
        native_source[:2], tmp_path, purpose_views=3, view_builder=native_builder(native_source)
    )
    recipe = json.loads((evaluation / "recipe.json").read_text())
    if where == "recipe":
        recipe.pop("purpose_views")
    else:
        path = evaluation / "data/dataset.json"
        dataset = json.loads(path.read_text())
        dataset["purpose_views"] = 2
        write(path, dataset)
        recipe["dataset_sha256"] = sha_file(path)
    write(evaluation / "recipe.json", recipe)
    with pytest.raises(ValueError, match="three-view"):
        handle(
            {"operation": "freeze", "job_id": "opaque-job", "evaluation_dir": str(evaluation)},
            artifact_root=root,
            purpose_views=3,
        )


def test_three_view_graph_uses_actual_native_files_and_explicit_view_callback(
    native_source, tmp_path
):
    from experiments.agentic_graph.adapter_roles import MODEL_KEYS, create_adapter_roles

    pins = {
        "base": {"repo": "test-only/base", "revision": "a" * 40, "model_sha256": "b" * 64},
        "adapter_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "dataset_sha256": "f" * 64,
        "code_sha256": {"worker.py": "0" * 64},
    }

    class Delegate:
        def configuration_identity(self):
            return {"test_only": True}

    def score(bundle, **_kwargs):
        evaluation = tmp_path / "scored"
        fake_completed_output(evaluation, bundle, pins)
        return {
            "prediction_bytes": (evaluation / "external_adapted.json").read_bytes(),
            "receipt": json.loads((evaluation / "inference_receipt.json").read_text()),
            "prepared_identity": pins,
        }

    roles = create_adapter_roles(
        Delegate(),
        purpose_views=3,
        view_builder=lambda source, render: native_builder(native_source)(
            source, render, tmp_path / "graph-views"
        ),
        score_runner=score,
        model_identity={k: pins[k] for k in MODEL_KEYS},
        driver_identity={"test_only": True},
        mode="diagnostic",
    )
    result = roles.analyze(native_source[0], native_source[1], timeout_seconds=60)
    assert result["executed"] and result["valid"] and result["analysis"]["observations"]
    assert result["trace"]["purpose_views"] == 3
    assert not result["trace"]["layout_authority"]


def test_three_view_mode_and_native_resource_paths_are_operator_flags_not_packet_inputs(
    source_png, tmp_path
):
    with pytest.raises(ValueError, match="native view builder"):
        handle(
            {
                "operation": "prepare",
                "job_id": "missing",
                "source": source_png[0],
                "source_render": source_png[1],
            },
            artifact_root=tmp_path / "jobs",
            purpose_views=3,
        )
    with pytest.raises(ValueError, match="native render only"):
        handle(
            {
                "operation": "prepare",
                "job_id": "injected",
                "source": source_png[0],
                "source_render": source_png[1],
                "assets_dir": "not-operator-owned",
                "purpose_views": 3,
            },
            artifact_root=tmp_path / "jobs",
            purpose_views=3,
            view_builder=lambda *_a: pytest.fail("Injected packet must not trigger native paint"),
        )
