"""Projection-only tests. Fixture pixels/labels are never admitted as real training data."""

from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from engine.design_semantics import semantic_source_digest
from experiments.visual_finetune.data import purpose_rows
from tools import rare_purpose_data as data
from tools.semantic_training_data import ALLOWED_SOURCES


def png():
    stream = BytesIO()
    Image.new("RGBA", (16, 12), (255, 255, 255, 255)).save(stream, format="PNG")
    return stream.getvalue()


def paint(*_args, **_kwargs):
    return {
        "status": "available",
        "reason": None,
        "alpha_png": png(),
        "checker_png": png(),
        "receipt": {
            "status": "available",
            "clipped": False,
            "native_single_original_layer": True,
            "receipt_digest": "helper-test",
        },
    }


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    audit = tmp_path / "audit.json"
    data._write(audit, {"test_fixture": True})
    admitted, inventory = [], {}
    for number, (sid, family) in enumerate(data.SOURCE_FAMILIES.items()):
        count = 11 if number < 2 else 10
        elements = [
            {
                "id": f"e-{i}",
                "type": "shape",
                "x": 2 + number,
                "y": 2,
                "width": 12,
                "height": 10,
                "fill_color": "rgba(255,255,255,1)",
                "shape_type": "rectangle",
            }
            for i in range(count)
        ]
        source = {"canvas": {"width": 64, "height": 48}, "elements": elements}
        source_rel, native_rel = f"sources/{sid}/source.json", f"sources/{sid}/native.png"
        annotation_rel = f"sources/{sid}/annotation.json"
        data._write(package / source_rel, source)
        Image.new("RGB", (64, 48), "blue").save(package / native_rel)
        labels = [
            {
                "element_id": e["id"],
                "purpose": "footer_legal" if i == 0 else "headline",
                "confidence": 0.9,
                "evidence": "Synthetic fixture, no real evidence",
                "placement": "not_applicable",
                "label_origin": "explicit_authored_synthetic_intent",
                "observable": "clear",
                "training_eligible": True,
            }
            for i, e in enumerate(elements)
        ]
        annotation = {
            "source_id": sid,
            "human_reviewed": False,
            "source_digest_protocol": "canonical-compact-v1",
            "source_digest": semantic_source_digest(source),
            "source_render_sha256": data._sha(package / native_rel),
            "elements": labels,
            "relations": [
                {
                    "source_id": f"e-{i}",
                    "target_id": f"e-{i + 1}",
                    "type": "reading_before",
                    "confidence": 0.9,
                    "evidence": "Fixture ordered cluster",
                }
                for i in range(6 if number < 6 else 5)
            ],
        }
        data._write(package / annotation_rel, annotation)
        entry = {
            "source_id": sid,
            "family_id": family,
            "split": "train",
            "source_json": {"path": source_rel, "sha256": data._sha(package / source_rel)},
            "native_png": {"path": native_rel, "sha256": data._sha(package / native_rel)},
            "annotation": {"path": annotation_rel, "sha256": data._sha(package / annotation_rel)},
            "assets_dir": "assets",
            "fonts_dir": "fonts",
        }
        admitted.append((entry, source, annotation, {"reviewer": "test_fixture"}, {}))
        for rel in (source_rel, native_rel, annotation_rel):
            inventory[rel] = data._sha(package / rel)
    monkeypatch.setattr(data, "_admit_package", lambda *_: (inventory, admitted))
    monkeypatch.setattr(data, "_isolate", paint)
    return package, audit, tmp_path / "export", admitted


def test_mask_keeps_legal_label_and_rejects_unplanned_classes():
    label = {"purpose": "footer_legal", "observable": "clear", "training_eligible": True}
    before = deepcopy(label)
    assert data._task_mask(label) == (False, "unsupported_task_taxonomy:footer_legal")
    assert label == before
    with pytest.raises(ValueError, match="Unexpected unsupported"):
        data._task_mask({**label, "purpose": "corner_marker"})
    with pytest.raises(ValueError, match="explicit clear"):
        data._task_mask({**label, "observable": "uncertain"})


def test_source_identity_family_split_and_digest_are_closed(cohort):
    entry, source, annotation, _, _ = cohort[3][0]
    data._validate_scene(entry, source, annotation, (64, 48))
    for changed in (
        {**entry, "family_id": "invented"},
        {**entry, "split": "validation"},
        {**entry, "source_id": "opensuse_banner_we_believe"},
    ):
        with pytest.raises(ValueError, match="frozen eight"):
            data._validate_scene(changed, source, annotation, (64, 48))
    with pytest.raises(ValueError, match="identity"):
        data._validate_scene(entry, source, {**annotation, "source_digest": "0" * 64}, (64, 48))


def test_export_preserves_raw_labels_relations_and_three_families(cohort):
    package, audit, out, admitted = cohort
    original_annotations = [deepcopy(row[2]) for row in admitted]
    manifest = data.export_rare_purpose(package, audit, out)
    assert manifest["counts"] == {
        "sources": 8,
        "train_families": 3,
        "raw_labels": 82,
        "raw_reviewed": 82,
        "task_eligible_train": 74,
        "unsupported_task_masked": 8,
        "isolated_available": 74,
        "eligible_unavailable": 0,
        "raw_relations": 46,
        "validation": 0,
        "test": 0,
    }
    rows = [
        __import__("json").loads(line) for line in (out / "elements.jsonl").read_text().splitlines()
    ]
    legal = [row for row in rows if row["target"]["purpose"] == "footer_legal"]
    assert len(legal) == 8
    assert all(row["raw_review_mask"] and not row["review_mask"] for row in legal)
    assert all(row["input"]["isolated_image"] is None for row in legal)
    assert len({row["family_id"] for row in rows}) == 3
    for row in rows:
        assert set(row["input"]) == {
            "source_image",
            "context_image",
            "isolated_image",
            "bbox_normalized",
            "type",
            "text_data",
            "text_is_untrusted_data",
        }
        with Image.open(out / row["input"]["context_image"]) as image:
            assert max(image.size) <= 256
    assert [row[2] for row in admitted] == original_annotations
    for entry, _, annotation, _, _ in admitted:
        copied = out / "source-package" / entry["annotation"]["path"]
        assert data._read(copied) == annotation
        assert data._sha(copied) == entry["annotation"]["sha256"]
    assert len(ALLOWED_SOURCES) == 8 and set(ALLOWED_SOURCES).isdisjoint(data.SOURCE_FAMILIES)


def test_actual_three_view_loader_accepts_supported_rows(cohort):
    package, audit, out, _ = cohort
    data.export_rare_purpose(package, audit, out)
    rows, receipt = purpose_rows(out / "manifest.json", purpose_views=3)
    assert len(rows) == 74 and all(len(row["images"]) == 3 for row in rows)
    assert {row["family_id"] for row in rows} == set(data.SOURCE_FAMILIES.values())
    assert {row["split"] for row in rows} == {"train"}
    assert receipt["purpose_views"] == 3
    assert all(row["canonical_label"] != "footer_legal" for row in rows)


def test_manifest_hashes_verify_portable_inputs_after_relocation(cohort, tmp_path):
    import shutil

    package, audit, out, _ = cohort
    manifest = data.export_rare_purpose(package, audit, out)
    relocated = tmp_path / "relocated"
    shutil.copytree(out, relocated)
    for rel, expected in manifest["files_sha256"].items():
        assert data._sha(relocated / rel) == expected
    rows, _ = purpose_rows(relocated / "manifest.json", purpose_views=3)
    assert len(rows) == 74


def test_unavailable_layer_is_explicit_and_never_two_view_fallback(cohort, monkeypatch):
    package, audit, out, _ = cohort
    monkeypatch.setattr(
        data,
        "_isolate",
        lambda *_: {
            "status": "unavailable",
            "reason": "ValueError: unsupported paint",
            "receipt": {"status": "unavailable"},
            "alpha_png": None,
            "checker_png": None,
        },
    )
    manifest = data.export_rare_purpose(package, audit, out)
    assert manifest["counts"]["eligible_unavailable"] == 74
    assert data._read(out / "READY.json")["status"] == "review_only_unavailable"
    with pytest.raises(ValueError, match="requires an available"):
        purpose_rows(out / "manifest.json", purpose_views=3)


def test_native_io_failure_halts_without_ready(cohort, monkeypatch):
    package, audit, out, _ = cohort
    monkeypatch.setattr(
        data,
        "_isolate",
        lambda *_: {
            "status": "unavailable",
            "reason": "FileNotFoundError: missing immutable font",
            "receipt": {},
            "alpha_png": None,
            "checker_png": None,
        },
    )
    with pytest.raises(OSError, match="must abort"):
        data.export_rare_purpose(package, audit, out)
    assert not (out / "READY.json").exists()


def test_frozen_package_or_existing_output_cannot_be_overwritten(cohort):
    package, audit, out, _ = cohort
    with pytest.raises(ValueError, match="independent"):
        data.export_rare_purpose(package, audit, package / "child")
    out.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        data.export_rare_purpose(package, audit, out)


def test_three_view_loader_rejects_tampered_receipt(cohort):
    package, audit, out, _ = cohort
    data.export_rare_purpose(package, audit, out)
    path = next((out / "isolated/receipts").rglob("*.json"))
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt changed"):
        purpose_rows(out / "manifest.json", purpose_views=3)


def test_exact_package_gate_rejects_unapproved_bytes(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    data._write(package / "manifest.json", {})
    audit = tmp_path / "audit.json"
    data._write(audit, {})
    with pytest.raises(ValueError, match="exact accepted"):
        data._admit_package(package, audit)


def test_24mp_isolation_requires_explicit_policy_and_preserves_default(cohort, monkeypatch):
    package, audit, out, _ = cohort
    selected = []

    def recorded_paint(element, assets, fonts, policy):
        selected.append(policy)
        return paint(element, assets, fonts)

    monkeypatch.setattr(data, "_isolate", recorded_paint)
    default = data.export_rare_purpose(package, audit, out)
    assert default["isolation_policy"] == data.ISOLATION_V1
    assert selected == [data.ISOLATION_V1] * 74
    selected.clear()
    successor = data.export_rare_purpose(
        package, audit, out.parent / "successor", isolation_policy=data.ISOLATION_V2_24MP
    )
    assert successor["isolation_policy"] == data.ISOLATION_V2_24MP
    assert selected == [data.ISOLATION_V2_24MP] * 74
    with pytest.raises(ValueError, match="explicitly supported"):
        data.export_rare_purpose(package, audit, out.parent / "bad", isolation_policy="unbounded")
