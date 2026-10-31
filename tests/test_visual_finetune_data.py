"""Data-boundary checks that do not import Torch or consume GPU memory."""

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from experiments.visual_finetune.data import freeze_dataset, sha, write_json
from experiments.visual_finetune.worker import classification_metrics


def critic_manifest(tmp_path, *, overlap=False):
    images = []
    for i, color in enumerate(("red", "blue", "white")):
        path = tmp_path / f"image-{i}.png"
        Image.new("RGB", (64, 64), color).save(path)
        images.append({"path": path.name, "sha256": sha(path)})
    records = []
    for split, family in (("train", "work-a"), ("validation", "work-a" if overlap else "work-b")):
        records.append(
            {
                "record_id": split,
                "family_id": family,
                "split": split,
                "label": "NEITHER",
                "label_origin": "synthetic_geometry",
                "images": images,
            }
        )
    path = tmp_path / "manifest.json"
    write_json(path, {"records": records})
    return path


def test_family_overlap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        freeze_dataset(critic_manifest(tmp_path, overlap=True), tmp_path / "out", task="critic")


def test_changed_pixels_are_rejected(tmp_path):
    manifest = critic_manifest(tmp_path)
    Image.new("RGB", (64, 64), "black").save(tmp_path / "image-0.png")
    with pytest.raises(ValueError, match="identity"):
        freeze_dataset(manifest, tmp_path / "out", task="critic")


def test_sealed_test_rows_do_not_enter_training_export(tmp_path):
    manifest = critic_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"].append({"split": "test", "secret_label": "never_read"})
    write_json(manifest, payload)
    dataset = freeze_dataset(manifest, tmp_path / "out", task="critic")
    assert len(dataset["records"]) == 2
    assert dataset["counts"]["train"] == {"N": 1}
    assert dataset["families"] == {"train": ["work-a"], "validation": ["work-b"]}
    for row in dataset["records"]:
        assert all(sha(ref["path"]) == ref["sha256"] for ref in row["images"])


def test_constant_choice_does_not_look_like_balanced_success():
    rows = [
        {
            "label": label,
            "prediction": "B",
            "control": "original",
            "raw_next_token_is_allowed": True,
        }
        for label in ("A", "B", "N")
    ]
    metrics = classification_metrics(rows, ["A", "B", "N"])
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert metrics["per_class"]["N"]["recall"] == 0


def test_external_receipt_binds_both_scores_only_when_base_actually_executed(tmp_path):
    from experiments.visual_finetune.worker import score_artifact_hashes

    write_json(tmp_path / "external_adapted.json", {"predictions": ["body"]})
    write_json(tmp_path / "external_base.json", {"predictions": ["support"]})
    assert score_artifact_hashes(tmp_path, base_executed=True) == {
        "prediction_file_sha256": sha(tmp_path / "external_adapted.json"),
        "base_prediction_file_sha256": sha(tmp_path / "external_base.json"),
    }
    assert (
        score_artifact_hashes(tmp_path, base_executed=False)["base_prediction_file_sha256"] is None
    )
    (tmp_path / "external_base.json").unlink()
    with pytest.raises(FileNotFoundError):
        score_artifact_hashes(tmp_path, base_executed=True)


def test_external_scoring_refuses_sealed_answers(tmp_path):
    from tools.evaluate_visual_adapter import prepare

    training = tmp_path / "training"
    training.mkdir()
    write_json(training / "data.json", {"task": "critic", "classes": ["A", "B", "N"]})
    write_json(
        training / "recipe.json",
        {"dataset": str(training / "data.json"), "dataset_sha256": sha(training / "data.json")},
    )
    request = tmp_path / "request.json"
    write_json(
        request,
        {
            "protocol": "visual-adapter-label-free-evaluation-v1",
            "task": "critic",
            "records": [{"record_id": "opaque", "label": "B"}],
        },
    )
    with pytest.raises(ValueError, match="sealed answers"):
        prepare(request, training, tmp_path / "unused-adapter", tmp_path / "out")


@pytest.mark.skipif(os.name != "nt", reason="Windows owned-job lifecycle")
def test_worker_waits_for_owned_job_assignment(tmp_path):
    from experiments.visual_finetune.supervise import run

    executable = Path("C:/Users/edmun/anaconda3/python.exe")
    if not executable.is_file():
        pytest.skip("Measured local lightweight Python is unavailable")
    command = [
        str(executable),
        "-c",
        "import sys; assert sys.stdin.readline().strip() == 'owned-job-assigned'; "
        "print('assigned')",
    ]
    receipt = run(command, tmp_path, cwd=str(tmp_path), max_seconds=10)
    assert receipt["exit_code"] == 0
    assert receipt["owned_kill_on_close_job"] is True
    assert receipt["owned_worker_stopped"] is True
    assert "assigned" in (tmp_path / "stdout.log").read_text()


def test_stale_experiment_code_is_rejected_before_model_import(tmp_path):
    from experiments.visual_finetune.worker import main

    recipe = tmp_path / "recipe.json"
    write_json(recipe, {"code_sha256": {"worker.py": "0" * 64}})
    with pytest.raises(ValueError, match="experiment code"):
        main(recipe)


def test_atomic_checkpoint_includes_complete_state_and_verifies(tmp_path):
    from experiments.visual_finetune.checkpoint import REQUIRED, publish, staging_directory
    from tools.checkpoint_manifest import verify_checkpoint

    destination = tmp_path / "checkpoint-4"
    stage = staging_directory(destination)
    for name in REQUIRED:
        (stage / name).write_bytes(b"io-test-fixture-not-a-model")
    publish(stage, destination)
    assert not stage.exists()
    assert REQUIRED.issubset(verify_checkpoint(destination)["files"])
    (destination / "training-state.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA256"):
        verify_checkpoint(destination)


def test_incomplete_checkpoint_does_not_replace_earlier_save(tmp_path):
    from experiments.visual_finetune.checkpoint import publish, staging_directory

    previous = tmp_path / "checkpoint-4"
    previous.mkdir()
    (previous / "sentinel").write_text("keep")
    destination = tmp_path / "checkpoint-8"
    stage = staging_directory(destination)
    (stage / "adapter_model.safetensors").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        publish(stage, destination)
    assert stage.exists()
    assert not destination.exists()
    assert (previous / "sentinel").read_text() == "keep"


def train_only_inputs(tmp_path):
    manifest = critic_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    row = payload["records"][0]
    payload["records"] = [{**row, "record_id": label, "label": label} for label in ("A", "B", "N")]
    write_json(manifest, payload)
    request = tmp_path / "independent.json"
    write_json(
        request,
        {
            "protocol": "visual-adapter-label-free-evaluation-v1",
            "task": "critic",
            "records": [
                {
                    "record_id": "opaque-probe",
                    "family_id": "work-b",
                    "split": "validation",
                    "images": row["images"],
                }
            ],
        },
    )
    return manifest, request


def test_train_only_keeps_independent_inputs_out_of_gradient_records(tmp_path):
    manifest, request = train_only_inputs(tmp_path)
    dataset = freeze_dataset(
        manifest, tmp_path / "out", task="critic", allow_train_only=True, evaluation_request=request
    )
    assert len(dataset["records"]) == 3
    assert all(row["split"] == "train" for row in dataset["records"])
    assert len(dataset["evaluation_records"]) == 1
    assert "label" not in dataset["evaluation_records"][0]
    assert dataset["provenance"]["evaluation_labels_available_to_trainer"] is False
    assert classification_metrics([{"prediction": "A", "label": None}], ["A", "B", "N"]) is None


def test_train_only_refuses_evaluation_gold_and_family_overlap(tmp_path):
    manifest, request = train_only_inputs(tmp_path)
    payload = json.loads(request.read_text())
    payload["records"][0]["label"] = "A"
    write_json(request, payload)
    with pytest.raises(ValueError, match="held-out answers"):
        freeze_dataset(
            manifest,
            tmp_path / "gold",
            task="critic",
            allow_train_only=True,
            evaluation_request=request,
        )
    del payload["records"][0]["label"]
    payload["records"][0]["family_id"] = "work-a"
    write_json(request, payload)
    with pytest.raises(ValueError, match="disjoint"):
        freeze_dataset(
            manifest,
            tmp_path / "overlap",
            task="critic",
            allow_train_only=True,
            evaluation_request=request,
        )


def test_portable_shells_have_lf_bytes_on_windows(tmp_path):
    from tools.package_visual_training import normalize_shell_scripts

    script = tmp_path / "run.sh"
    script.write_bytes(b"#!/usr/bin/env bash\r\r\nset -euo pipefail\r\n")
    normalize_shell_scripts(tmp_path)
    assert script.read_bytes() == b"#!/usr/bin/env bash\nset -euo pipefail\n"


def test_resident_adapter_is_upcast_before_saved_weights_are_loaded():
    from experiments.visual_finetune.adapter_io import precast_then_load

    calls = []

    class Model:
        base_model = None

        def add_adapter(self, name, config):
            calls.append("add")

        def _cast_adapter_dtype(self, **kwargs):
            calls.append("precast")

        def load_adapter(self, *args, **kwargs):
            assert calls == ["add", "precast"]
            calls.append("load")

        def set_adapter(self, name):
            calls.append("activate")

    model = Model()
    model.base_model = model
    precast_then_load(model, "checkpoint", "reload", object())
    assert calls == ["add", "precast", "load", "activate"]


def test_default_purpose_prompt_stays_two_views_and_third_view_is_explicit():
    from experiments.visual_finetune.data import PURPOSE_PROMPT, build_purpose_prompt, canonical

    context = {
        "bbox_normalized": [0, 0, 1, 1],
        "type": "shape",
        "text_data": "",
        "paint_order_index": 0,
        "element_count": 2,
    }
    assert build_purpose_prompt(context) == PURPOSE_PROMPT + canonical(context)
    assert "third image" in build_purpose_prompt(context, purpose_views=3)
    with pytest.raises(ValueError, match="two or three"):
        build_purpose_prompt(context, purpose_views=4)


def test_wrong_purpose_view_contract_is_rejected_before_adapter_load(tmp_path):
    from tools.evaluate_visual_adapter import prepare

    training = tmp_path / "training"
    training.mkdir()
    write_json(
        training / "data.json", {"task": "purpose", "classes": ["unknown"], "purpose_views": 3}
    )
    write_json(
        training / "recipe.json",
        {"dataset": str(training / "data.json"), "dataset_sha256": sha(training / "data.json")},
    )
    request = tmp_path / "request.json"
    write_json(
        request,
        {
            "protocol": "visual-adapter-label-free-evaluation-v1",
            "task": "purpose",
            "purpose_views": 2,
            "records": [],
        },
    )
    with pytest.raises(ValueError, match="view count"):
        prepare(request, training, tmp_path / "unused", tmp_path / "out")


def test_three_view_blank_controls_cannot_leak_the_isolated_target(tmp_path, monkeypatch):
    from experiments.visual_finetune import data

    rows = []
    for split in ("train", "validation"):
        for family_index in range(2):
            refs = []
            for view in range(3):
                path = tmp_path / f"{split}-{family_index}-{view}.png"
                Image.new("RGB", (16 + view, 20), (30 + 50 * family_index, 20 * view, 200)).save(
                    path
                )
                refs.append({"path": str(path), "sha256": sha(path)})
            for index in range(2):
                rows.append(
                    {
                        "record_id": f"{split}-{family_index}-{index}",
                        "family_id": f"{split}-{family_index}",
                        "split": split,
                        "label": "body",
                        "canonical_label": "body",
                        "images": refs,
                        "prompt": "fixed",
                        "control": "original",
                        "label_origin": "unit_test_fixture",
                    }
                )
    monkeypatch.setattr(data, "purpose_rows", lambda *args, **kwargs: (rows, {}))
    dataset = data.freeze_dataset("unused", tmp_path / "out", task="purpose", purpose_views=3)
    blank = [row for row in dataset["records"] if row["control"] == "both_blank"]
    assert len(blank) == 8
    for row in blank:
        assert len(row["images"]) == 3
        for ref in row["images"]:
            with Image.open(ref["path"]) as picture:
                assert picture.getextrema() == ((127, 127), (127, 127), (127, 127))
    replacements = [row for row in dataset["records"] if row["control"] == "source_replaced"]
    assert len(replacements) == 4
    assert all(
        len(row["images"]) == 3
        and row["label_origin"] == "programmatic_source_mismatch"
        and row["related_input_family"] != row["family_id"]
        for row in replacements
    )


def test_critic_packaging_preserves_declared_rights_and_separates_training_probe(tmp_path):
    from tools.package_visual_training import portable_critic_export

    manifest, request = train_only_inputs(tmp_path)
    notice = tmp_path / "licenses" / "notice.txt"
    notice.parent.mkdir()
    notice.write_text("Unit test provenance fixture; no public-license claim.")
    payload = json.loads(manifest.read_text())
    payload["declared_rights"] = {"status": "project-owned-not-declared-public"}
    payload["files_sha256"] = {"licenses/notice.txt": sha(notice)}
    write_json(manifest, payload)
    destination = tmp_path / "bundle-export"
    portable_critic_export(manifest, request, destination)
    original = json.loads((destination / "provenance/original-training-manifest.json").read_text())
    assert original == payload
    assert sha(destination / "provenance/licenses/notice.txt") == sha(notice)
    exported = json.loads((destination / "manifest.json").read_text())
    assert (
        exported["provenance"]["declared_manifest_metadata"]["declared_rights"]
        == payload["declared_rights"]
    )
    assert exported["provenance"]["original_manifest"]["public_status_inferred"] is False
    probe = json.loads((destination / "fixed-train-probe.json").read_text())
    assert len(probe["records"]) == 3
    assert all("label" not in row for row in probe["records"])
    assert probe["execute_only_after_causal_R1_R2"] is True


def test_purpose_scene_variants_keep_their_three_canonical_families():
    from experiments.visual_finetune.data import purpose_family

    families = []
    for index in range(8):
        source = {
            "source_id": f"scene-{index}",
            "family_id": f"asset-{index % 3}",
            "split": "train",
        }
        binding = {
            "source_id": source["source_id"],
            "family_id": source["family_id"],
            "split": "train",
        }
        families.append(purpose_family(source, receipt=binding, row=binding, frozen=binding))
    assert len(set(families)) == 3
    assert purpose_family({"source_id": "legacy-work", "split": "train"}) == "legacy-work"


def test_purpose_family_binding_rejects_parent_row_receipt_or_split_disagreement():
    from experiments.visual_finetune.data import purpose_family

    source = {"source_id": "scene", "family_id": "asset", "split": "train"}
    with pytest.raises(ValueError, match="family disagrees"):
        purpose_family(source, row={"family_id": "wrong"})
    with pytest.raises(ValueError, match="family disagrees"):
        purpose_family(source, receipt={"family_id": "wrong"})
    with pytest.raises(ValueError, match="family disagrees"):
        purpose_family(source, frozen={"family": "wrong"})
    with pytest.raises(ValueError, match="split disagrees"):
        purpose_family(source, row={"family_id": "asset", "split": "validation"})
    with pytest.raises(ValueError, match="missing"):
        purpose_family(source, receipt={"source_id": "scene"})


def test_coverage_helper_is_copied_hashed_and_verified_without_repo_fallback(tmp_path):
    from experiments.visual_finetune.snapshot import code_path, freeze_coverage_helper, verify_code

    root = Path(__file__).resolve().parents[1]
    code = tmp_path / "code"
    package = code / "experiments/visual_finetune"
    package.mkdir(parents=True)
    hashes = freeze_coverage_helper(root, code)
    assert set(hashes) == {"tools/__init__.py", "tools/purpose_coverage_sampler.py"}
    verify_code(package, hashes)
    helper = code / "tools/purpose_coverage_sampler.py"
    assert sha(helper) == sha(root / "tools/purpose_coverage_sampler.py")
    frozen_copy = tmp_path / "eval-code"
    assert freeze_coverage_helper(code, frozen_copy, expected=hashes) == hashes
    helper.write_bytes(helper.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="frozen recipe"):
        verify_code(package, hashes)
    with pytest.raises(ValueError, match="Unsupported path"):
        code_path(package, "../outside.py")


def test_prepared_dataset_style_arm_preserves_pixels_and_pins_coverage(tmp_path, monkeypatch):
    import sys

    from tools.train_visual_adapter import main

    image = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "white").save(image)
    source = tmp_path / "dataset.json"
    manifest = tmp_path / "base-identity.json"
    write_json(manifest, {"test_fixture": "prepare-only; no model loaded"})
    write_json(
        source,
        {
            "task": "purpose",
            "purpose_views": 3,
            "records": [
                {
                    "record_id": "heldout",
                    "family_id": "heldout-family",
                    "split": "validation",
                    "control": "original",
                    "label": "body",
                    "images": [{"path": image.name, "sha256": sha(image)}] * 3,
                    "prompt": "legacy prompt",
                }
            ],
            "evaluation_records": [],
            "style_prompt_overlay": {"heldout": "same prompt with observed typography"},
            "provenance": {"task_definition_sha256": "a" * 64},
        },
    )
    outputs = []
    for version in ("v2", "observed_typo_v1"):
        out = tmp_path / version
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_visual_adapter",
                "--dataset",
                str(source),
                "--task",
                "purpose",
                "--purpose-views",
                "3",
                "--context-version",
                version,
                "--record-sampling",
                "coverage_family_interleave",
                "--out",
                str(out),
                "--base-manifest",
                str(manifest),
                "--prepare-only",
            ],
        )
        main()
        recipe = json.loads((out / "recipe.json").read_text())
        copied = json.loads((out / "data/dataset.json").read_text())
        assert recipe["context_version"] == version
        assert recipe["task_definition_sha256"] == "a" * 64
        assert recipe["record_sampling"] == "coverage_family_interleave"
        assert "tools/purpose_coverage_sampler.py" in recipe["code_sha256"]
        assert sha(out / "data/source.png") == sha(image)
        outputs.append(copied)
    assert outputs[0]["records"][0]["prompt"] == "legacy prompt"
    assert outputs[1]["records"][0]["prompt"] == "same prompt with observed typography"
    assert outputs[0]["records"][0]["images"] == outputs[1]["records"][0]["images"]


def test_coverage_selection_preserves_label_rng_and_legacy_selection():
    import random

    from experiments.visual_finetune.sampling import choose_training_record
    from tools.purpose_coverage_sampler import ClassCoverageSampler

    rows = [
        {
            "record_id": f"{label}-{i}",
            "label": label,
            "family_id": f"family-{i % 2}",
            "split": "train",
        }
        for label in ("a", "b")
        for i in range(6)
    ]
    pools = {label: [row for row in rows if row["label"] == label] for label in ("a", "b")}
    byid = {row["record_id"]: row for row in rows}

    def sample(covered):
        rng = random.Random(1729)
        coverage = ClassCoverageSampler(rows, seed=1729) if covered else None
        labels, ids = [], []
        for _ in range(48):
            label = rng.choices(["a", "b"], weights=[1, 2], k=1)[0]
            row = choose_training_record(
                rng, pools[label], label, coverage=coverage, record_by_id=byid
            )
            labels.append(label)
            ids.append(row["record_id"])
        return labels, ids, rng.getstate()

    legacy, covered = sample(False), sample(True)
    assert legacy[0] == covered[0]
    assert legacy[2] == covered[2]
    for label in ("a", "b"):
        ids = [
            record_id
            for record_id, selected in zip(covered[1], covered[0], strict=True)
            if selected == label
        ]
        assert len(set(ids[:6])) == 6


def test_coverage_selection_rejects_nontraining_records():
    import random

    from experiments.visual_finetune.sampling import choose_training_record

    class BadCoverage:
        def draw(self, label):
            return "heldout"

    training = {"record_id": "train", "label": "a", "split": "train"}
    with pytest.raises(ValueError, match="non-training"):
        choose_training_record(
            random.Random(1729),
            [training],
            "a",
            coverage=BadCoverage(),
            record_by_id={"heldout": {"label": "a", "split": "validation"}},
        )


def test_contrast_matte_is_explicit_and_prompt_names_the_actual_view(tmp_path):
    from experiments.visual_finetune.data import (
        CONTRAST_MATTE_V1,
        build_purpose_prompt,
        purpose_rows,
    )

    context = {
        "bbox_normalized": [0, 0, 1, 1],
        "type": "text",
        "text_data": "copy",
        "paint_order_index": 0,
        "element_count": 1,
    }
    checker = build_purpose_prompt(context, purpose_views=3)
    matte = build_purpose_prompt(context, purpose_views=3, isolated_view_policy=CONTRAST_MATTE_V1)
    assert "neutral checker" in checker and "plain contrast matte" not in checker
    assert "plain contrast matte" in matte and "neutral checker" not in matte
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "protocol": "semantic-analysis-training-data-v1",
            "human_reviewed": False,
            "isolated_view_policy": CONTRAST_MATTE_V1,
        },
    )
    with pytest.raises(ValueError, match="explicit three-view admission"):
        purpose_rows(manifest, purpose_views=3)


@pytest.mark.parametrize(
    "native_protocol,alpha_field",
    [
        ("semantic-purpose-isolated-layer-evidence-v1", "isolated_alpha_image"),
        ("native-element-isolation-v2-24mp-source", "alpha_image"),
    ],
)
def test_contrast_evidence_binds_retained_alpha_native_provenance_and_output_pixels(
    tmp_path, native_protocol, alpha_field
):
    import hashlib

    from experiments.visual_finetune.data import (
        CONTRAST_COMPOSER_SHA256,
        CONTRAST_MATTE_V1,
        digest,
        verify_contrast_evidence,
    )

    alpha, rgb = tmp_path / "alpha.png", tmp_path / "rgb.png"
    Image.new("RGBA", (8, 8), (200, 30, 10, 255)).save(alpha)
    Image.new("RGB", (8, 8), (200, 30, 10)).save(rgb)

    def ref(path):
        return {"path": path.name, "sha256": sha(path)}

    original = {
        alpha_field: ref(alpha),
        "status": "available",
        "clipped": False,
        "native_single_original_layer": True,
        "label_used_for_paint_choices": False,
        "no_replacement_asset_or_content": True,
        "protocol": native_protocol,
    }
    write_json(tmp_path / "original.json", original)
    composed = {
        "protocol": CONTRAST_MATTE_V1,
        "implementation_sha256": CONTRAST_COMPOSER_SHA256,
        "input_alpha_png_sha256": sha(alpha),
        "output_png_sha256": sha(rgb),
        "input_rgba_unchanged": True,
        "native_redraw": False,
        "model_executed": False,
        "policy": {"selection_uses_labels_or_context": False},
        "size": [8, 8],
    }
    for path, key in ((alpha, "input_rgba_pixel_sha256"), (rgb, "output_rgb_pixel_sha256")):
        with Image.open(path) as image:
            composed[key] = hashlib.sha256(image.tobytes()).hexdigest()
    composed["receipt_digest"] = digest(composed)
    write_json(tmp_path / "composed.json", composed)
    active = {
        **original,
        "retained_alpha": ref(alpha),
        "original_isolation_receipt": ref(tmp_path / "original.json"),
        "contrast_matte_receipt": ref(tmp_path / "composed.json"),
    }
    rgb_ref = ref(rgb)
    hashes = {path.name: sha(path) for path in tmp_path.iterdir() if path.is_file()}
    verify_contrast_evidence(tmp_path, hashes, rgb_ref, active)
    active["clipped"] = True
    with pytest.raises(ValueError, match="native isolation provenance"):
        verify_contrast_evidence(tmp_path, hashes, rgb_ref, active)
    active["clipped"] = False
    Image.new("RGB", (8, 8), "black").save(rgb)
    with pytest.raises(ValueError, match="file changed"):
        verify_contrast_evidence(tmp_path, hashes, rgb_ref, active)
