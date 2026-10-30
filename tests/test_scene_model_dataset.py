"""Scene supervision binds actual source pixels and excludes leaked/reserve inputs."""

import json
from copy import deepcopy

import pytest
from PIL import Image

from experiments.scene_model import dataset


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    records = []
    for index, split in enumerate(("train", "validation", "test")):
        directory = tmp_path / f"scene-{index}"
        directory.mkdir()
        asset = directory / "asset.png"
        Image.new("RGB", (32, 48), (index * 50, 100, 200)).save(asset)
        render = directory / "source.png"
        Image.new("RGB", (40, 50), (200, index * 30, 100)).save(render)
        element = {
            "id": f"opaque-id-{index}",
            "type": "image",
            "x": 0,
            "y": 0,
            "width": 40,
            "height": 50,
            "asset_id": "asset.png",
            "fit": "cover",
        }
        source = {"canvas": {"width": 40, "height": 50}, "elements": [element]}
        profile = {
            "schema_version": 1,
            "source_digest": dataset.digest(source),
            "images": [
                {
                    "element_id": element["id"],
                    "asset_sha256": dataset.sha256(asset),
                    "role": "unknown",
                    "allow_cover": False,
                }
            ],
        }
        profile["scene_digest"] = dataset.digest(profile)
        source_render = {"path": str(render), "sha256": dataset.sha256(render)}
        request = {
            "source": source,
            "deterministic_scene": profile,
            "source_digest": profile["source_digest"],
            "scene_digest": profile["scene_digest"],
            "source_render": source_render,
            "asset_files": {element["id"]: str(asset)},
        }
        report = {
            "source_digest": profile["source_digest"],
            "scene_digest": profile["scene_digest"],
            "source_immutable": True,
            "resources_immutable": True,
            "source_render": source_render,
        }
        labels = {
            "schema_version": 1,
            "source_digest": profile["source_digest"],
            "scene_digest": profile["scene_digest"],
            "images": [
                {
                    "element_id": element["id"],
                    "asset_sha256": dataset.sha256(asset),
                    "proposed_role": "background",
                    "confidence": 0.9,
                    "protected_regions": [],
                }
            ],
        }
        for name, value in (
            ("source.json", source),
            ("scene.json", profile),
            ("model-request.json", request),
            ("report.json", report),
            ("labels.json", labels),
        ):
            _write(directory / name, value)
        records.append(
            {
                "record_id": f"record-{index}",
                "source_id": f"source-{index}",
                "family_id": f"family-{index}",
                "split": split,
                "harness_dir": directory.name,
                "labels": str(directory / "labels.json"),
                "element_id": element["id"],
                "supervision_origin": "synthetic_rule",
                "reviewed_by": "synthetic-fixture",
                "rights_accepted": True,
                "source_parity_reviewed": True,
            }
        )
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"schema_version": 1, "records": records})
    return manifest, records


def _load(corpus, **kwargs):
    manifest, _ = corpus
    return dataset.load_records(
        manifest, allowed_root=manifest.parent, allow_synthetic=True, **kwargs
    )


def test_two_real_views_and_semantic_targets_exclude_opaque_id_copy(corpus):
    rows = _load(corpus)
    assert len(rows) == 3
    assert len(rows[0]["images"]) == 2
    response = json.loads(rows[0]["response"])
    assert response["proposed_role"] == "background"
    assert "element_id" not in response and "asset_sha256" not in response
    assert rows[0]["element_id"] not in rows[0]["prompt"]
    assert dataset.dataset_summary(rows)["quality_validated"] is False


def test_synthetic_labels_are_not_accepted_as_real_review(corpus):
    manifest, _ = corpus
    with pytest.raises(ValueError, match="Synthetic"):
        dataset.load_records(manifest, allowed_root=manifest.parent)


@pytest.mark.parametrize("file", ["source.png", "asset.png"])
def test_stale_native_source_or_asset_pixels_are_rejected(corpus, file):
    manifest, _ = corpus
    Image.new("RGB", (32, 48), "red").save(manifest.parent / "scene-0" / file)
    with pytest.raises(ValueError, match="bytes changed"):
        _load(corpus)


@pytest.mark.parametrize("field", ["family_id", "source_id"])
def test_source_families_cannot_leak_across_splits(corpus, field):
    manifest, records = corpus
    records[1][field] = records[0][field]
    _write(manifest, {"schema_version": 1, "records": records})
    with pytest.raises(ValueError, match="crosses experiment splits"):
        _load(corpus)


def test_known_personal_reserve_id_is_rejected_before_any_training(corpus):
    manifest, records = corpus
    records[0]["source_id"] = sorted(dataset._reserves()[0])[0]
    _write(manifest, {"schema_version": 1, "records": records})
    with pytest.raises(ValueError, match="reserve"):
        _load(corpus)


def test_renamed_reserve_source_digest_is_rejected(corpus, monkeypatch):
    manifest, _ = corpus
    source = dataset.read_json(manifest.parent / "scene-0/source.json")
    monkeypatch.setattr(dataset, "_reserves", lambda: (frozenset(), {dataset.digest(source)}))
    with pytest.raises(ValueError, match="Renamed personal reserve"):
        _load(corpus)


def test_rights_or_parity_cannot_be_inferred_from_catalogue_review(corpus):
    manifest, records = corpus
    records[0]["rights_accepted"] = False
    _write(manifest, {"schema_version": 1, "records": records})
    with pytest.raises(ValueError, match="rights"):
        _load(corpus)


def test_record_cannot_invent_model_authority_fields(corpus):
    manifest, records = corpus
    records[0]["allow_cover"] = True
    _write(manifest, {"schema_version": 1, "records": records})
    with pytest.raises(ValueError, match="closed manifest"):
        _load(corpus)


def test_duplicate_image_does_not_inflate_scene_dataset(corpus):
    manifest, records = corpus
    repeated = deepcopy(records[0])
    repeated["record_id"] = "new-name-same-source-image"
    _write(manifest, {"schema_version": 1, "records": [*records, repeated]})
    with pytest.raises(ValueError, match="Duplicate source image"):
        _load(corpus)


def test_manifest_paths_cannot_escape_explicit_local_root(corpus):
    manifest, records = corpus
    records[0]["labels"] = "../elsewhere.json"
    _write(manifest, {"schema_version": 1, "records": records})
    with pytest.raises(ValueError, match="leaves the allowed local root"):
        _load(corpus)
