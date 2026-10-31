"""Synthetic abstention controls preserve the original source-family boundary."""

from __future__ import annotations

import json

import pytest
from PIL import Image
from test_semantic_training_data import inputs  # noqa: F401 - shared explicit fixture

from engine.visual_analysis import bind_analysis
from tools import semantic_analysis_controls as controls
from tools import semantic_training_data as data


@pytest.fixture
def parent(request):
    args, _ = request.getfixturevalue("inputs")
    data.build_dataset(**args, opaque_ids=True)
    return args["out"]


def test_blank_controls_exact_schema_pixels_split_masks_and_parent_immutability(parent, tmp_path):
    before = controls._snapshot(parent)
    out = tmp_path / "controls"
    result = controls.build_controls(dataset=parent, out=out)
    assert result["counts"] == {
        "controls": 8,
        "train": 5,
        "validation": 3,
        "new_independent_scenes": 0,
    }
    assert result["parent_files_sha256_before"] == result["parent_files_sha256_after"] == before
    assert controls._snapshot(parent) == before
    assert result["parent_immutability_verified"] is True
    assert result["diagnostic_only"] is True
    assert result["human_reviewed"] is result["quality_promotion"] is False
    rows = [
        json.loads(line)
        for line in (out / "controls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 8
    for row in rows:
        assert row["diagnostic_only"] is True
        assert row["view_kind"] == "synthetic_blank_view"
        assert row["source_render_parity"] is False
        assert row["supervision_origin"] == "synthetic_abstention_rule"
        assert row["training_eligible"] == (row["split"] == "train")
        assert row["target"] == {"observations": [], "abstain": True}
        receipt = data._read(out / row["receipt"])
        image = out / row["input"]["source_image"]
        with Image.open(image) as blank:
            assert blank.mode == "RGB"
            assert blank.size == tuple(
                receipt["source_canvas"][axis] for axis in ("width", "height")
            )
            assert blank.getcolors() == [(blank.width * blank.height, (255, 255, 255))]
        assert (
            data._sha(image)
            == receipt["blank_render_sha256"]
            == row["input"]["source_render_sha256"]
        )
        assert receipt["original_source_render_sha256"] != receipt["blank_render_sha256"]
        assert receipt["parent_manifest_sha256"] == before["manifest.json"]
        ids = [element["id"] for element in row["input"]["source_summary"]["elements"]]
        assert row["validated_binding"] == bind_analysis(
            row["target"],
            source_digest=receipt["source_digest"],
            source_render_sha256=receipt["blank_render_sha256"],
            known_element_ids=ids,
        )
        parent_receipt = data._read(parent / receipt["parent_receipt"])
        assert receipt["family"] == parent_receipt["family"]
        assert not {"family", "source_id"} & row["input"].keys()
    ready = data._read(out / "READY.json")
    assert ready["manifest_sha256"] == data._sha(out / "manifest.json")
    assert ready["optional_training_policy_required"] is True
    assert all(data._sha(out / path) == value for path, value in result["files_sha256"].items())


def test_fresh_and_nonoverlapping_output_required(parent, tmp_path):
    with pytest.raises(ValueError, match="fresh"):
        controls.build_controls(dataset=parent, out=parent)
    with pytest.raises(ValueError, match="overlap"):
        controls.build_controls(dataset=parent, out=parent / "controls")
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        controls.build_controls(dataset=parent, out=out)


def test_changed_parent_manifest_receipts_fail_before_writing(parent, tmp_path):
    (parent / "analysis.jsonl").write_text("{}\n", encoding="utf-8")
    out = tmp_path / "controls"
    with pytest.raises(ValueError, match="declared receipts"):
        controls.build_controls(dataset=parent, out=out)
    assert not out.exists()


def test_no_original_id_rows_cannot_be_replaced_with_opaque_variants(parent, tmp_path):
    path = parent / "analysis.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    data._jsonl(path, [row for row in rows if row["variant"] == "opaque_ids"])
    manifest = data._read(parent / "manifest.json")
    manifest["files_sha256"]["analysis.jsonl"] = data._sha(path)
    data._write(parent / "manifest.json", manifest)
    with pytest.raises(ValueError, match="eight original-ID"):
        controls.build_controls(dataset=parent, out=tmp_path / "controls")


def test_midbuild_parent_change_never_writes_ready(parent, tmp_path, monkeypatch):
    write = data._jsonl

    def mutate_after_controls(path, rows):
        write(path, rows)
        (parent / "unexpected.txt").write_text("changed", encoding="utf-8")

    monkeypatch.setattr(data, "_jsonl", mutate_after_controls)
    out = tmp_path / "controls"
    with pytest.raises(ValueError, match="parent dataset changed"):
        controls.build_controls(dataset=parent, out=out)
    assert (out / "INCOMPLETE.json").exists()
    assert not (out / "READY.json").exists()
    assert not (out / "manifest.json").exists()


def test_blank_dimensions_pixel_cap_checked_before_writing(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MAX_PIXELS", 100)
    out = tmp_path / "controls"
    with pytest.raises(ValueError, match="bounded full source-canvas"):
        controls.build_controls(dataset=parent, out=out)
    assert not out.exists()
