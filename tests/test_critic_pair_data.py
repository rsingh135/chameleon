"""Temporary shape-only evidence exercises freeze gates, not visual labels."""

from __future__ import annotations

import shutil
from copy import deepcopy

import pytest
from PIL import Image

from engine.design_semantics import semantic_source_digest
from experiments.vision_critic.contracts import canonical_json_sha256
from tools import critic_pair_data as critic
from tools import semantic_training_data as data


def write(path, value):
    data._write(path, value)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    project = tmp_path / "project"
    catalogue_root = project / "fixtures/open_posters"
    annotations = tmp_path / "annotations"
    split_manifest = tmp_path / "split.json"
    entries, splits = [], []
    for index, identity in enumerate(sorted(data.ALLOWED_SOURCES)):
        prepared = catalogue_root / "prepared" / identity
        source = {
            "canvas": {"width": 80 + index, "height": 60 + index},
            "elements": [
                {
                    "id": f"shape-{index}",
                    "type": "shape",
                    "shape_type": "rectangle",
                    "fill_color": "red",
                    "x": index + 1,
                    "y": 10,
                    "width": 20,
                    "height": 10,
                    "z_index": 0,
                }
            ],
        }
        write(prepared / "source.json", source)
        (prepared / "resources/assets").mkdir(parents=True)
        (prepared / "resources/fonts").mkdir(parents=True)
        Image.new("RGB", (80 + index, 60 + index), (index, 20, 40)).save(prepared / "native.png")
        original = catalogue_root / f"originals/{identity}.json"
        write(original, {"original": identity})
        (catalogue_root / "LICENSE").write_text("Fixture licence", encoding="utf-8")
        entries.append(
            {
                "id": identity,
                "split": "reviewed",
                "prepared_path": f"prepared/{identity}",
                "original_path": f"originals/{identity}.json",
                "sha256": data._sha(original),
                "license_id": "Fixture-License",
                "license_path": "LICENSE",
                "license_url": "https://example.test/license",
                "source_url": "https://example.test/" + identity,
                "attribution_text": "Fixture author",
            }
        )
        split = "validation" if index in {0, 1, 6} else "train"
        splits.append({"source_id": identity, "family": identity, "split": split})
        proposal = {
            "schema_version": 1,
            "source_digest_protocol": "canonical-compact-v1",
            "source_digest": semantic_source_digest(source),
            "elements": [
                {
                    "element_id": source["elements"][0]["id"],
                    "purpose": "decoration",
                    "confidence": 0.8,
                    "evidence": "Synthetic test label.",
                }
            ],
            "relations": [],
        }
        write(
            annotations / f"{identity}.json",
            {
                "source_id": identity,
                "source_digest": semantic_source_digest(source),
                "source_render_sha256": data._sha(prepared / "native.png"),
                "supervision_origin": "agent_visual_review",
                "reviewer": "test",
                "reviewed_element_ids": [source["elements"][0]["id"]],
                "uncertain_element_ids": [],
                "purpose_proposal": proposal,
                "analysis_raw": {"observations": [], "abstain": True},
                "limitations": ["Synthetic fixture."],
            },
        )
    write(catalogue_root / "catalogue.json", {"examples": entries})
    write(
        split_manifest,
        {"schema_version": 1, "protocol": "semantic-analysis-family-split-v1", "sources": splits},
    )
    authority = tmp_path / "authority"
    data.build_dataset(
        catalogue=catalogue_root / "catalogue.json",
        annotations=annotations,
        split_manifest=split_manifest,
        out=authority,
    )
    monkeypatch.setattr(critic, "ROOT", project)
    # No text exists in these fixtures; avoid loading any native font resources.
    monkeypatch.setattr("engine.eval.context.get_measurer", lambda *_: object())
    monkeypatch.setattr("engine.eval.contract.get_measurer", lambda *_: object())
    identity = next(row["source_id"] for row in splits if row["split"] == "train")
    source_root = catalogue_root / "prepared" / identity
    root = tmp_path / "evidence"
    shutil.copytree(source_root, root / "source")
    source = data._read(root / "source/source.json")
    target = {"width": 100, "height": 80}

    def artifact(name, design, color):
        write(root / f"{name}.json", design)
        Image.new("RGB", (100, 80), color).save(root / f"{name}.png")
        return {
            "design": f"{name}.json",
            "design_sha256": canonical_json_sha256(design),
            "render": f"{name}.png",
            "render_sha256": data._sha(root / f"{name}.png"),
            "contract_valid": True,
            "contract_violations": [],
            "provenance": {"test": True},
        }

    good = deepcopy(source)
    good["canvas"] = target
    bad = deepcopy(good)
    bad["elements"][0]["x"] = 150
    record = {
        "record_id": "original-record",
        "pair_id": "original-pair",
        "source_id": identity,
        "family_id": identity,
        "split": "train",
        "label": "A",
        "label_origin": "algorithmic_weak",
        "source": {
            "design": "source/source.json",
            "json_sha256": canonical_json_sha256(source),
            "render": "source/native.png",
            "render_sha256": data._sha(root / "source/native.png"),
        },
        "A": artifact("good", good, "green"),
        "B": artifact("bad", bad, "red"),
        "target_canvas": target,
        "rights": {
            "license_id": "Fixture-License",
            "license_url": "https://example.test/license",
            "source_url": "https://example.test/" + identity,
            "acquired_utc": "2026-10-30T12:00:00Z",
            "attribution": "Fixture author",
        },
        "provenance": {"reason": "geometry_perturbation"},
    }
    input_path = root / "input.json"
    write(
        input_path,
        {
            "schema_version": 1,
            "protocol": "critic-pair-evidence-v1",
            "records": [record],
            "resource_roots": {
                identity: {
                    "assets_dir": "source/resources/assets",
                    "fonts_dir": "source/resources/fonts",
                }
            },
        },
    )
    code_path = project / "fixture_producer.py"
    code_path.write_text("# Synthetic test attestation, no native rendering.\n", encoding="utf-8")
    args = {
        "input": input_path,
        "split_manifest": split_manifest,
        "source_manifest": authority / "manifest.json",
        "out": tmp_path / "frozen",
        "generation_receipt": root / "generation-receipt.json",
    }
    write(
        args["generation_receipt"],
        {
            "status": "complete",
            "input": "input.json",
            "input_raw_sha256": data._sha(input_path),
            "input_canonical_sha256": canonical_json_sha256(data._read(input_path)),
            "records": 1,
            "train_sources": [identity],
            "split_sha256": data._sha(split_manifest),
            "frozen_manifest_sha256": data._sha(authority / "manifest.json"),
            "implementation_hashes": {"fixture_producer.py": data._sha(code_path)},
            "implementation_after_hashes": {"fixture_producer.py": data._sha(code_path)},
        },
    )
    return args


def mutate(evidence, operation):
    value = data._read(evidence["input"])
    operation(value)
    write(evidence["input"], value)
    generation = data._read(evidence["generation_receipt"])
    generation.update(
        input_raw_sha256=data._sha(evidence["input"]),
        input_canonical_sha256=canonical_json_sha256(value),
        records=len(value["records"]),
        train_sources=sorted({row["source_id"] for row in value["records"]}),
    )
    write(evidence["generation_receipt"], generation)


def test_exact_native_evidence_swaps_lineage_weak_eligibility_and_loader(evidence):
    from experiments.visual_finetune.data import critic_rows

    before = {
        path: data._sha(path) for path in evidence["input"].parent.rglob("*") if path.is_file()
    }
    result = critic.freeze_pairs(**evidence, swap=True)
    assert result["counts"] == {
        "original_pairs": 1,
        "records_including_swaps": 2,
        "sources": 1,
        "families": 1,
        "labels": {"A": 1, "B": 1},
    }
    first, second = result["records"]
    assert second["images"] == [first["images"][0], first["images"][2], first["images"][1]]
    assert first["original_pair_id"] == second["original_pair_id"]
    assert first["record_id"] != second["record_id"]
    assert all(row["training_eligible"] and row["split"] == "train" for row in result["records"])
    receipt = data._read(evidence["out"] / first["receipt"])
    assert receipt["computed_candidate_checks"]["A"]["valid"] is True
    assert receipt["computed_candidate_checks"]["B"]["valid"] is False
    assert receipt["declared_candidate_metadata"]["B"]["contract_valid"] is True
    assert receipt["computed_candidate_checks"]["B"]["contract_violations"]
    assert result["human_preference"] is result["quality_approved"] is False
    assert {path: data._sha(path) for path in before} == before
    assert all(
        data._sha(evidence["out"] / name) == value for name, value in result["files_sha256"].items()
    )
    loaded, _ = critic_rows(evidence["out"] / "manifest.json")
    assert [row["label"] for row in loaded] == ["A", "B"]
    assert all(
        all(data._sha(image["path"]) == image["sha256"] for image in row["images"])
        for row in loaded
    )


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda v: v["records"][0].update(label="B"), "selects a contract-invalid"),
        (lambda v: v["records"][0].update(split="validation"), "TRAIN"),
        (lambda v: v["records"][0].update(family_id="different"), "TRAIN"),
        (lambda v: v["records"][0].update(source_id="opensuse_banner_we_believe"), "TRAIN"),
        (lambda v: v["records"][0].update(label_origin="human"), "weak"),
        (
            lambda v: v["records"][0]["A"].update(render="../outside.png"),
            "relative|portable|escaped",
        ),
        (lambda v: v["records"][0]["A"].update(render_sha256="0" * 64), "hash differs"),
        (lambda v: v["records"].append(deepcopy(v["records"][0])), "Duplicate"),
        (lambda v: v["records"][0].update(label="N"), "explicit weak"),
    ],
)
def test_fail_closed_admission(evidence, change, match):
    mutate(evidence, change)
    with pytest.raises(ValueError, match=match):
        critic.freeze_pairs(**evidence)
    assert not evidence["out"].exists()


def test_neither_requires_two_invalid_or_explicit_weak_ambiguity(evidence):
    mutate(
        evidence, lambda v: v["records"][0].update(label="N", provenance={"reason": "ambiguity"})
    )
    result = critic.freeze_pairs(**evidence, swap=True)
    assert result["counts"]["labels"] == {"N": 2}
    assert result["records"][0]["original_pair_id"] == result["records"][1]["original_pair_id"]


def test_fresh_output_pixel_and_native_dimensions_gates(evidence, monkeypatch):
    evidence["out"].mkdir()
    with pytest.raises(ValueError, match="fresh"):
        critic.freeze_pairs(**evidence)
    evidence["out"].rmdir()
    monkeypatch.setattr(data, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="pixel budget"):
        critic.freeze_pairs(**evidence)


def test_source_id_relabel_cannot_hide_changed_original(evidence):
    value = data._read(evidence["input"])
    source_path = evidence["input"].parent / value["records"][0]["source"]["design"]
    source = data._read(source_path)
    source["elements"][0]["fill_color"] = "blue"
    write(source_path, source)
    mutate(
        evidence,
        lambda v: v["records"][0]["source"].update(json_sha256=canonical_json_sha256(source)),
    )
    with pytest.raises(ValueError, match="frozen original TRAIN"):
        critic.freeze_pairs(**evidence)


def test_self_hashed_source_png_cannot_replace_original_pixels(evidence):
    path = evidence["input"].parent / "source/native.png"
    with Image.open(path) as old:
        size = old.size
    Image.new("RGB", size, "white").save(path)
    mutate(
        evidence, lambda value: value["records"][0]["source"].update(render_sha256=data._sha(path))
    )
    with pytest.raises(ValueError, match="exact frozen source pixels"):
        critic.freeze_pairs(**evidence)
    assert not evidence["out"].exists()


def test_generation_receipt_is_required_and_binds_input_and_producer(evidence):
    args = {key: value for key, value in evidence.items() if key != "generation_receipt"}
    with pytest.raises(ValueError, match="generation receipt is required"):
        critic.freeze_pairs(**args)
    receipt = data._read(evidence["generation_receipt"])
    stale = deepcopy(receipt)
    stale["input_raw_sha256"] = "0" * 64
    write(evidence["generation_receipt"], stale)
    with pytest.raises(ValueError, match="does not bind"):
        critic.freeze_pairs(**evidence)
    write(evidence["generation_receipt"], receipt)
    producer = critic.ROOT / "fixture_producer.py"
    producer.write_text("# Producer changed after generation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="producer/code bytes differ"):
        critic.freeze_pairs(**evidence)
    assert not evidence["out"].exists()


def test_empty_package_marker_is_valid_code_identity_not_an_image(evidence):
    marker = critic.ROOT / "__init__.py"
    marker.write_bytes(b"")
    receipt = data._read(evidence["generation_receipt"])
    receipt["implementation_hashes"]["__init__.py"] = critic.sha256_file(marker)
    receipt["implementation_after_hashes"]["__init__.py"] = critic.sha256_file(marker)
    write(evidence["generation_receipt"], receipt)
    result = critic.freeze_pairs(**evidence)
    assert result["generation_receipt_verified"] is True
    assert result["input_files_sha256"][str(marker)] == critic.sha256_file(marker)


def test_rights_and_font_resource_substitutions_are_rejected(evidence):
    original = data._read(evidence["input"])
    for key in ("attribution", "license_url"):
        mutate(
            evidence,
            lambda value, key=key: value["records"][0]["rights"].update(
                {key: "https://example.test/wrong" if key == "license_url" else "Wrong author"}
            ),
        )
        with pytest.raises(ValueError, match="License/source authority"):
            critic.freeze_pairs(**evidence)
        mutate(evidence, lambda value: value.update(deepcopy(original)))
    path = evidence["input"].parent / "source/resources/fonts/extra.ttf"
    path.write_bytes(b"not-original-font-bytes")
    with pytest.raises(ValueError, match="asset/font bytes differ"):
        critic.freeze_pairs(**evidence)


def test_real_other_train_source_view_keeps_native_dimensions_and_false_parity(evidence):
    raw = data._read(evidence["input"])
    owner = raw["records"][0]["source_id"]
    manifest = data._read(evidence["source_manifest"])
    other = next(
        row for row in manifest["sources"] if row["split"] == "train" and row["source_id"] != owner
    )
    receipt = data._read(evidence["source_manifest"].parent / other["receipt"])
    source_image = evidence["source_manifest"].parent / (
        "images/" + receipt["source_digest"][:20] + ".png"
    )
    destination = evidence["input"].parent / "displayed-other-source.png"
    shutil.copyfile(source_image, destination)

    def replace_source(value):
        row = value["records"][0]
        row.update(
            label="N",
            control="source_mismatch",
            provenance={
                "reason": "source_mismatch",
                "mismatched_source_id": other["source_id"],
                "displayed_source_parity": False,
            },
        )
        row["source"].update(render=destination.name, render_sha256=data._sha(destination))

    mutate(evidence, replace_source)
    result = critic.freeze_pairs(**evidence, swap=True)
    assert result["generation_receipt_verified"] is True
    assert (
        result["native_png_json_linkage"]
        == "traceable_producer_attestation_not_independent_pixel_proof"
    )
    assert result["counts"]["labels"] == {"N": 2}
    for row in result["records"]:
        assert row["source_render_parity"] is False
        assert row["displayed_source_id"] == other["source_id"]
        assert row["family_id"] == owner
        assert row["control"] == "source_mismatch"
        with Image.open(evidence["out"] / row["images"][0]["path"]) as image:
            assert image.size == tuple(receipt["source_canvas"][key] for key in ("width", "height"))
    frozen_receipt = data._read(evidence["out"] / result["records"][0]["receipt"])
    original_source = data._read(evidence["input"].parent / raw["records"][0]["source"]["design"])
    assert frozen_receipt["displayed_source_canvas"] != original_source["canvas"]
    assert frozen_receipt["displayed_source_legal_files"]


def test_mismatch_rejects_validation_identity_and_forged_parity(evidence):
    original = data._read(evidence["input"])
    validation = next(
        row
        for row in data._read(evidence["split_manifest"])["sources"]
        if row["split"] == "validation"
    )
    for wrong in (
        {
            "reason": "source_mismatch",
            "mismatched_source_id": validation["source_id"],
            "displayed_source_parity": False,
        },
        {
            "reason": "source_mismatch",
            "mismatched_source_id": original["records"][0]["source_id"],
            "displayed_source_parity": False,
        },
        {
            "reason": "source_mismatch",
            "mismatched_source_id": "opensuse_booth_poster",
            "displayed_source_parity": True,
        },
    ):
        mutate(
            evidence,
            lambda value, wrong=wrong: value["records"][0].update(
                label="N", control="source_mismatch", provenance=wrong
            ),
        )
        with pytest.raises(ValueError, match="another TRAIN family"):
            critic.freeze_pairs(**evidence)
    assert not evidence["out"].exists()


def test_train_candidate_json_cannot_disguise_heldout_source_png(evidence):
    raw = data._read(evidence["input"])
    manifest = data._read(evidence["source_manifest"])
    heldout = next(row for row in manifest["sources"] if row["split"] == "validation")
    receipt = data._read(evidence["source_manifest"].parent / heldout["receipt"])
    source_png = evidence["source_manifest"].parent / (
        "images/" + receipt["source_digest"][:20] + ".png"
    )
    target = receipt["source_canvas"]
    for key in ("A", "B"):
        metadata = raw["records"][0][key]
        design_path = evidence["input"].parent / metadata["design"]
        design = data._read(design_path)
        design["canvas"] = target
        write(design_path, design)
        image_path = evidence["input"].parent / metadata["render"]
        if key == "A":
            shutil.copyfile(source_png, image_path)
        else:
            Image.new("RGB", (target["width"], target["height"]), "red").save(image_path)
        metadata.update(
            design_sha256=canonical_json_sha256(design), render_sha256=data._sha(image_path)
        )
    raw["records"][0]["target_canvas"] = target
    mutate(evidence, lambda value: value.update(raw))
    json_hash = raw["records"][0]["A"]["design_sha256"]
    with pytest.raises(ValueError, match="validation/reserved source PNG"):
        critic.freeze_pairs(**evidence)
    assert not evidence["out"].exists()
    image_path = evidence["input"].parent / raw["records"][0]["A"]["render"]
    Image.new("RGB", (target["width"], target["height"]), "orange").save(image_path)
    mutate(
        evidence, lambda value: value["records"][0]["A"].update(render_sha256=data._sha(image_path))
    )
    result = critic.freeze_pairs(**evidence)
    assert result["records"][0]["label"] == "A"
    frozen_receipt = data._read(evidence["out"] / result["records"][0]["receipt"])
    assert frozen_receipt["declared_candidate_metadata"]["A"]["design_sha256"] == json_hash


def test_changed_license_bytes_cannot_reuse_old_catalogue_rights(evidence):
    path = critic.ROOT / "fixtures/open_posters/LICENSE"
    path.write_text("Changed licence text under the same original path", encoding="utf-8")
    with pytest.raises(ValueError, match="license/notice bytes differ from frozen authority"):
        critic.freeze_pairs(**evidence)
    assert not evidence["out"].exists()
