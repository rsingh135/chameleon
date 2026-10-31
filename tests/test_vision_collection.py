"""Collection evidence must remain immutable, resource-bound and baseline-preserving."""

import json
from copy import deepcopy

import pytest
from PIL import Image

from experiments.vision_critic import collect as module
from experiments.vision_critic.contracts import canonical_json_sha256, load_collection


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # No real reserve source is resized, rendered, labelled or copied by these tests.
    source = {
        "canvas": {"width": 200, "height": 100},
        "elements": [
            {
                "id": "shape",
                "type": "shape",
                "x": 10,
                "y": 10,
                "width": 70,
                "height": 70,
                "shape_type": "rectangle",
                "fill_color": "#336644",
            }
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    manifest = {
        "schema_version": 1,
        "purpose": "synthetic",
        "sources": [
            {
                "source_id": "test",
                "family_id": "test-family",
                "split": "train",
                "source": "source.json",
                "source_parity_verified": False,
                "training_rights_reviewed": False,
                "rights_reference": "fixture only",
                "targets": [{"width": 300, "height": 100}],
            }
        ],
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        module,
        "code_provenance",
        lambda: {"engine_revision": "fixture", "working_tree_digest": "a" * 64, "code_files": {}},
    )
    return path, manifest, source


def snapshot(source, target):
    rows = []
    for index in range(4):
        design = deepcopy(source)
        design["canvas"] = target
        design["elements"][0]["x"] += index * 20
        rows.append(
            {
                "candidate_id": canonical_json_sha256(design),
                "design": design,
                "state": {"topology": "columns" if index == 2 else "row"},
                "diagnostics": {"contract_valid": True, "contract_violations": []},
            }
        )
    admitted = [rows[0]["candidate_id"], rows[1]["candidate_id"]]
    for row in rows:
        row["contract_valid"] = True
        row["admitted"] = row["candidate_id"] in admitted
    return {
        "baseline_candidate_id": rows[0]["candidate_id"],
        "candidates": rows,
        "admitted_finalist_ids": admitted,
    }


def render(design):
    return Image.new("RGBA", tuple(design["canvas"].values()), "white")


def test_immutable_collection_keeps_backend_baseline_and_hashes(inputs, tmp_path):
    path, _, source = inputs
    result = module.collect(
        path, tmp_path / "collection", pool="shadow", snapshot_provider=snapshot, renderer=render
    )
    data = load_collection(tmp_path / "collection")
    case = data["cases"][0]
    assert (
        case["baseline_candidate_id"]
        == snapshot(source, case["target_canvas"])["baseline_candidate_id"]
    )
    assert len(case["candidates"]) == 4
    assert sum(candidate["admitted"] for candidate in case["candidates"]) == 2
    assert not result["production_selection_changed"]
    with pytest.raises(ValueError, match="must not exist"):
        module.collect(path, tmp_path / "collection", snapshot_provider=snapshot, renderer=render)
    assert load_collection(tmp_path / "collection") == data


def test_default_pool_cannot_admit_shadow_only_alternatives(inputs, tmp_path):
    path, _, _ = inputs
    module.collect(path, tmp_path / "collection", snapshot_provider=snapshot, renderer=render)
    case = load_collection(tmp_path / "collection")["cases"][0]
    assert len(case["candidates"]) == 2
    assert all(candidate["admitted"] for candidate in case["candidates"])


def test_failed_render_does_not_publish_or_leave_staging(inputs, tmp_path):
    path, _, _ = inputs
    calls = []

    def broken(design):
        calls.append(design)
        if len(calls) == 2:
            raise RuntimeError("render interrupted")
        return render(design)

    with pytest.raises(RuntimeError, match="interrupted"):
        module.collect(path, tmp_path / "collection", snapshot_provider=snapshot, renderer=broken)
    assert not (tmp_path / "collection").exists()
    assert not list(tmp_path.glob(".vision-collection-*"))


def test_forged_validity_cannot_publish_changed_content(inputs, tmp_path):
    path, _, _ = inputs

    def invalid(source, target):
        result = snapshot(source, target)
        result["candidates"][0]["design"]["elements"][0]["fill_color"] = "#000000"
        return result

    with pytest.raises(ValueError, match="contract-invalid"):
        module.collect(path, tmp_path / "collection", snapshot_provider=invalid, renderer=render)
    assert not (tmp_path / "collection").exists()


def test_duplicate_source_alias_cannot_leak_across_splits(inputs):
    path, manifest, _ = inputs
    duplicate = deepcopy(manifest["sources"][0])
    duplicate.update(source_id="alias", family_id="other", split="test")
    manifest["sources"].append(duplicate)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Identical source"):
        module.read_input(path)


def test_known_reserve_id_cannot_be_collected_under_synthetic_flag(inputs):
    path, manifest, _ = inputs
    catalogue = json.loads(
        (module.ROOT / "fixtures/open_posters/catalogue.json").read_text(encoding="utf-8")
    )
    manifest["sources"][0]["source_id"] = next(
        row["id"] for row in catalogue["examples"] if row["split"] == "reserve"
    )
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reserve"):
        module.read_input(path)


def test_changed_implementation_does_not_publish(inputs, tmp_path, monkeypatch):
    path, _, _ = inputs
    values = iter(({"engine_revision": "a"}, {"engine_revision": "b"}))
    monkeypatch.setattr(module, "code_provenance", lambda: next(values))
    with pytest.raises(ValueError, match="Implementation changed"):
        module.collect(path, tmp_path / "collection", snapshot_provider=snapshot, renderer=render)
    assert not (tmp_path / "collection").exists()


def test_no_image_dimension_substitution(inputs, tmp_path):
    path, _, _ = inputs
    with pytest.raises(ValueError, match="exact canvas"):
        module.collect(
            path,
            tmp_path / "collection",
            snapshot_provider=snapshot,
            renderer=lambda _design: Image.new("RGB", (32, 32)),
        )


@pytest.mark.parametrize("kind", ("font", "image"))
def test_missing_source_resources_cannot_become_training_evidence(inputs, tmp_path, kind):
    path, _, source = inputs
    if kind == "font":
        source["elements"] = [
            {
                "id": "text",
                "type": "text",
                "x": 10,
                "y": 10,
                "width": 100,
                "height": 40,
                "font_size": 12,
                "font_family": "Uninstalled source family 91821",
                "content": "Source copy",
                "color": "#000000",
            }
        ]
    else:
        source["elements"] = [
            {
                "id": "image",
                "type": "image",
                "x": 10,
                "y": 10,
                "width": 30,
                "height": 40,
                "asset_id": "absent-91821.png",
            }
        ]
    (path.parent / "source.json").write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="source (font|image)"):
        module.collect(path, tmp_path / "collection", snapshot_provider=snapshot, renderer=render)
    assert not (tmp_path / "collection").exists()


def test_shadow_snapshot_cannot_publish_an_unadmitted_baseline(inputs, tmp_path):
    path, _, _ = inputs

    def broken(source, target):
        result = snapshot(source, target)
        result["admitted_finalist_ids"].remove(result["baseline_candidate_id"])
        result["candidates"][0]["admitted"] = False
        return result

    with pytest.raises(ValueError, match="admitted baseline"):
        module.collect(
            path, tmp_path / "collection", pool="shadow", snapshot_provider=broken, renderer=render
        )
    assert not (tmp_path / "collection").exists()


@pytest.mark.parametrize("callback", ("snapshot", "renderer"))
def test_callbacks_cannot_mutate_source_evidence(inputs, tmp_path, callback):
    path, _, _ = inputs

    def mutating_snapshot(source, target):
        source["elements"][0]["x"] += 2
        return snapshot(source, target)

    def mutating_renderer(design):
        design["elements"][0]["x"] += 2
        return render(design)

    with pytest.raises(ValueError, match="mutated"):
        module.collect(
            path,
            tmp_path / "collection",
            snapshot_provider=mutating_snapshot if callback == "snapshot" else snapshot,
            renderer=mutating_renderer if callback == "renderer" else render,
        )
    assert not (tmp_path / "collection").exists()


@pytest.mark.parametrize("kind", ("manifest", "source"))
def test_reviewed_bytes_are_bound_before_later_file_reads(inputs, tmp_path, monkeypatch, kind):
    path, _, _ = inputs
    original = module.read_input

    def altered(manifest, max_cases=24, **kwargs):
        result = original(manifest, max_cases, **kwargs)
        target = manifest if kind == "manifest" else manifest.parent / "source.json"
        target.write_bytes(target.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(module, "read_input", altered)
    with pytest.raises(ValueError, match="changed after prerequisite review"):
        module.collect(path, tmp_path / "collection", snapshot_provider=snapshot, renderer=render)
    assert not (tmp_path / "collection").exists()


def test_nonfinite_backend_geometry_cannot_be_published_even_if_marked_valid(inputs, tmp_path):
    path, _, _ = inputs

    def broken(source, target):
        result = snapshot(source, target)
        result["candidates"][0]["design"]["elements"][0]["x"] = float("nan")
        return result

    with pytest.raises(ValueError):
        module.collect(path, tmp_path / "collection", snapshot_provider=broken, renderer=render)
    assert not (tmp_path / "collection").exists()
