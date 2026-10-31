"""Frozen model probes preserve original pixels, identities and sealed labels."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from experiments.vision_critic import independent_probes as probes
from experiments.vision_critic.contracts import (
    canonical_json_sha256,
    parse_json,
    sha256_file,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def collection(tmp_path, monkeypatch):
    root = tmp_path / "collection"
    root.mkdir()
    cases = []
    for index, split in enumerate(("train", "validation", "test")):
        source = {
            "canvas": {"width": 64, "height": 48},
            "elements": [
                {
                    "id": f"headline-{split}",
                    "type": "text",
                    "x": 4,
                    "y": 4,
                    "width": 52,
                    "height": 12,
                    "font_size": 10,
                    "content": f"Café {split}",
                },
                {
                    "id": f"image-{split}",
                    "type": "shape",
                    "x": 12,
                    "y": 20,
                    "width": 32,
                    "height": 24,
                },
            ],
        }
        source_json = root / f"{split}-source.json"
        source_png = root / f"{split}-source.png"
        write_json(source_json, source)
        Image.new("RGB", (64, 48), (50 + index * 50, 60, 100)).save(source_png)
        candidates = []
        for number in range(2):
            design = deepcopy(source)
            design["canvas"] = {"width": 96, "height": 32}
            design["elements"][1]["x"] += number * 6
            design_json = root / f"{split}-candidate-{number}.json"
            design_png = root / f"{split}-candidate-{number}.png"
            write_json(design_json, design)
            Image.new("RGB", (96, 32), (50 + index * 50, number * 80, 150)).save(design_png)
            candidates.append(
                {
                    "candidate_id": f"{split}-native-{number}",
                    "design": design_json.name,
                    "design_sha256": canonical_json_sha256(design),
                    "render": design_png.name,
                    "render_sha256": sha256_file(design_png),
                    "admitted": True,
                    "diagnostics": {"contract_valid": True, "contract_violations": []},
                    "provenance": {"element_ids": [row["id"] for row in source["elements"]]},
                }
            )
        cases.append(
            {
                "case_id": f"case-{split}",
                "source_id": f"source-{split}",
                "family_id": f"family-{split}",
                "split": split,
                "source_parity_verified": False,
                "training_rights_reviewed": False,
                "rights_reference": "synthetic local test only",
                "target_canvas": {"width": 96, "height": 32},
                "source": {
                    "design": source_json.name,
                    "json_sha256": canonical_json_sha256(source),
                    "render": source_png.name,
                    "render_sha256": sha256_file(source_png),
                },
                "baseline_candidate_id": candidates[0]["candidate_id"],
                "admitted_candidate_ids": [row["candidate_id"] for row in candidates],
                "candidates": candidates,
            }
        )
    document = {
        "schema_version": 1,
        "collection_id": "independent-probe-tests",
        "purpose": "synthetic",
        "provenance": {
            "engine_revision": "fixture",
            "working_tree_digest": "a" * 64,
            "policy_version": "fixture-v1",
        },
        "cases": cases,
    }
    write_json(root / "collection.json", document)
    code_root = tmp_path / "code-root"
    code_root.mkdir()
    (code_root / "evaluator.py").write_text("# immutable test implementation\n")
    monkeypatch.setattr(probes, "ROOT", code_root)
    monkeypatch.setattr(probes, "CODE_FILES", ("evaluator.py",))
    monkeypatch.setattr(probes.subprocess, "check_output", lambda *_a, **_k: "fixture-head\n")
    return root, document, code_root


def prepare(collection, tmp_path, **kwargs):
    root, _, _ = collection
    out = tmp_path / "bundle"
    return out, probes.prepare_collection(root, out, **kwargs)


def requests(out, bundle):
    return {
        row["condition"]: (
            row,
            parse_json((out / row["request_path"]).read_text(encoding="utf-8")),
        )
        for row in bundle["probes"]
    }


def rewrite_bundle(path, bundle):
    bundle["bundle_sha256"] = probes.digest(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )
    write_json(path, bundle)


def image_path(out, reference):
    path = Path(reference["path"])
    assert not path.is_absolute(), "Frozen image references must survive relocation"
    return out / path


def test_frozen_probe_matrix_keeps_exact_source_and_candidate_png_bytes(collection, tmp_path):
    root, document, _ = collection
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    out, bundle = prepare(collection, tmp_path)
    assert [case["split"] for case in bundle["cases"]] == ["validation"]
    assert {case["family_id"] for case in bundle["cases"]} == {"family-validation"}
    assert len(bundle["probes"]) == 5
    variants = requests(out, bundle)
    assert set(variants) == {"original", "swapped", "blank_source", "replaced_source", "opaque_ids"}
    original_probe, original = variants["original"]
    selected = document["cases"][1]
    assert (
        image_path(out, original["source_render"]).read_bytes() == before["validation-source.png"]
    )
    assert original_probe["source_sha256"] == selected["source"]["render_sha256"]
    assert original_probe["source_sha256"] == original_probe["original_source_sha256"]
    for rendered, candidate in zip(original["plans"], selected["candidates"], strict=True):
        assert image_path(out, rendered["render"]).read_bytes() == before[candidate["render"]]
    for row in bundle["probes"]:
        request = parse_json((out / row["request_path"]).read_text(encoding="utf-8"))
        assert row["request_digest"] == probes.request_digest(request)
        assert row["menu_digest"] == probes.request_digest(
            {key: value for key, value in request["plan_menu"].items() if key != "menu_digest"}
        )
        assert set(row["known_element_ids"]) == set(row["element_map"])
        assert set(row["element_map"].values()) == {
            element["id"] for element in original["source_summary"]["elements"]
        }
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert probes.load_bundle(out / "bundle.json") == bundle


def test_swap_and_opaque_controls_preserve_the_same_physical_candidate_images(collection, tmp_path):
    out, bundle = prepare(collection, tmp_path)
    variants = requests(out, bundle)
    _, original = variants["original"]
    _, swapped = variants["swapped"]
    assert swapped["source_render"] == original["source_render"]
    assert swapped["plans"] == list(reversed(original["plans"]))
    assert swapped["plan_menu"]["plans"] == list(reversed(original["plan_menu"]["plans"]))
    assert swapped["plan_menu"]["menu_digest"] != original["plan_menu"]["menu_digest"]
    opaque_probe, opaque = variants["opaque_ids"]
    assert opaque["source_render"] == original["source_render"]
    assert [row["render"] for row in opaque["plans"]] == [
        row["render"] for row in original["plans"]
    ]
    original_ids = {row["plan_id"] for row in original["plans"]}
    assert not original_ids.intersection(opaque_probe["plan_map"])
    assert set(opaque_probe["plan_map"].values()) == original_ids
    assert all(key != value for key, value in opaque_probe["element_map"].items())
    assert [row.get("content") for row in opaque["source_summary"]["elements"]] == [
        row.get("content") for row in original["source_summary"]["elements"]
    ]


def test_blank_and_replacement_controls_are_separate_originals_are_unchanged(collection, tmp_path):
    root, document, _ = collection
    out, bundle = prepare(collection, tmp_path)
    variants = requests(out, bundle)
    original_probe, original = variants["original"]
    blank_probe, blank = variants["blank_source"]
    replacement_probe, replacement = variants["replaced_source"]
    assert blank_probe["original_source_sha256"] == original_probe["source_sha256"]
    assert replacement_probe["original_source_sha256"] == original_probe["source_sha256"]
    assert blank["plans"] == replacement["plans"] == original["plans"]
    assert blank["source_summary"] == replacement["source_summary"] == original["source_summary"]
    with Image.open(image_path(out, blank["source_render"])) as image:
        assert image.size == (64, 48)
        assert all(low == high for low, high in image.getextrema())
    assert (
        image_path(out, replacement["source_render"]).read_bytes()
        == (root / document["cases"][0]["source"]["render"]).read_bytes()
    )
    assert replacement_probe["source_sha256"] != original_probe["source_sha256"]
    assert (
        image_path(out, original["source_render"]).read_bytes()
        == (root / "validation-source.png").read_bytes()
    )


def test_source_only_analysis_contains_no_targets_choices_or_swapped_control(collection, tmp_path):
    out, bundle = prepare(collection, tmp_path, modes=("analyze",))
    assert {row["condition"] for row in bundle["probes"]} == {
        "original",
        "blank_source",
        "replaced_source",
        "opaque_ids",
    }
    for probe in bundle["probes"]:
        request = parse_json((out / probe["request_path"]).read_text(encoding="utf-8"))
        assert request["mode"] == "analyze"
        assert not {"target_canvas", "plan_menu", "plans", "source_analysis"}.intersection(request)
        assert probe["plan_map"] == {} and probe["menu_digest"] is None
        assert probe["known_element_ids"]


def test_preparation_never_opens_or_copies_preference_labels(collection, tmp_path, monkeypatch):
    root, _, _ = collection
    label = root / "sealed-test-labels.json"
    label.write_text('{"secret":"DO_NOT_COPY_OR_READ_TEST_LABELS"}')
    read_bytes, read_text = Path.read_bytes, Path.read_text

    def guarded_bytes(path):
        assert "label" not in path.name, f"Opened sealed labels: {path}"
        return read_bytes(path)

    def guarded_text(path, *args, **kwargs):
        assert "label" not in path.name, f"Opened sealed labels: {path}"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    out, bundle = prepare(collection, tmp_path)
    assert all(case["split"] != "test" for case in bundle["cases"])
    assert all(row["split"] != "test" for row in bundle["probes"])
    assert not any("label" in relative for relative in bundle["files"])
    assert not any(
        b"DO_NOT_COPY_OR_READ_TEST_LABELS" in read_bytes(path)
        for path in out.rglob("*")
        if path.is_file()
    )


def test_missing_different_family_replacement_is_reported_without_fabricating_one(
    collection, tmp_path
):
    root, document, _ = collection
    document["cases"] = [document["cases"][1]]
    write_json(root / "collection.json", document)
    _, bundle = prepare(collection, tmp_path)
    assert "replaced_source" not in {row["condition"] for row in bundle["probes"]}
    assert any("no different non-test family replacement" in item for item in bundle["limitations"])


def test_historical_code_snapshot_survives_live_code_changes_but_not_snapshot_tampering(
    collection, tmp_path
):
    _, _, code_root = collection
    out, bundle = prepare(collection, tmp_path)
    (code_root / "evaluator.py").write_text("# changed live implementation\n")
    assert probes.load_bundle(out / "bundle.json") == bundle
    (out / "code" / "evaluator.py").write_text("# changed frozen implementation\n")
    with pytest.raises(ValueError, match="bytes changed"):
        probes.load_bundle(out / "bundle.json")


@pytest.mark.parametrize("artifact", ["request", "image"])
def test_frozen_request_or_image_mutation_is_rejected(collection, tmp_path, artifact):
    out, bundle = prepare(collection, tmp_path)
    request_path = out / bundle["probes"][0]["request_path"]
    path = request_path
    if artifact == "image":
        request = parse_json(request_path.read_text(encoding="utf-8"))
        path = image_path(out, request["source_render"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        probes.load_bundle(out / "bundle.json")


def test_file_map_cannot_escape_the_bundle_even_with_a_recomputed_manifest(collection, tmp_path):
    out, bundle = prepare(collection, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside frozen bundle")
    bundle["files"]["../outside.txt"] = sha256_file(outside)
    rewrite_bundle(out / "bundle.json", bundle)
    with pytest.raises(ValueError, match="relative|escape"):
        probes.load_bundle(out / "bundle.json")


def test_collection_family_leakage_is_rejected_before_output_creation(collection, tmp_path):
    root, document, _ = collection
    document["cases"][1]["family_id"] = document["cases"][0]["family_id"]
    write_json(root / "collection.json", document)
    with pytest.raises(ValueError, match="Family leakage"):
        prepare(collection, tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"max_cases": 0},
        {"max_cases": 21},
        {"max_cases": True},
        {"splits": ()},
        {"splits": ("unknown",)},
        {"modes": ()},
        {"modes": ("plan",)},
        {"modes": ("analyze", "analyze")},
    ],
)
def test_invalid_preparation_options_do_not_create_an_output(collection, tmp_path, options):
    with pytest.raises(ValueError, match="Invalid bounded"):
        prepare(collection, tmp_path, **options)
    assert not (tmp_path / "bundle").exists()


def test_existing_output_directory_is_never_overwritten(collection, tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_bytes(b"earlier run")
    with pytest.raises(FileExistsError):
        probes.prepare_collection(collection[0], out)
    assert sentinel.read_bytes() == b"earlier run"


@pytest.mark.parametrize("field", ["source", "candidate"])
def test_collection_render_change_between_validation_and_copying_is_rejected(
    collection, tmp_path, monkeypatch, field
):
    root, _, _ = collection
    load = probes.load_collection

    def change_after_validation(path):
        document = load(path)
        case = document["cases"][1]
        metadata = case["source"] if field == "source" else case["candidates"][0]
        Image.new("RGB", (64, 48) if field == "source" else (96, 32), "magenta").save(
            root / metadata["render"]
        )
        return document

    monkeypatch.setattr(probes, "load_collection", change_after_validation)
    with pytest.raises(ValueError, match="changed|identity|hash|bytes"):
        prepare(collection, tmp_path)


@pytest.mark.parametrize("field", ["source", "candidate"])
def test_collection_json_change_between_validation_and_copying_is_rejected(
    collection, tmp_path, monkeypatch, field
):
    root, _, _ = collection
    load = probes.load_collection

    def change_after_validation(path):
        document = load(path)
        case = document["cases"][1]
        metadata = case["source"] if field == "source" else case["candidates"][0]
        design_path = root / metadata["design"]
        changed = parse_json(design_path.read_text(encoding="utf-8"))
        changed["elements"][0]["content"] = "Mutated after the collection was checked"
        write_json(design_path, changed)
        return document

    monkeypatch.setattr(probes, "load_collection", change_after_validation)
    with pytest.raises(ValueError, match="changed|identity|hash|bytes"):
        prepare(collection, tmp_path)


def test_frozen_bundle_relocation_keeps_request_and_source_identities(collection, tmp_path):
    out, bundle = prepare(collection, tmp_path)
    relocated = tmp_path / "relocated-bundle"
    assert out.parent == relocated.parent == tmp_path
    out.rename(relocated)
    assert probes.load_bundle(relocated / "bundle.json") == bundle
    assert all(
        image_path(relocated, request["source_render"]).is_file()
        for _, request in requests(relocated, bundle).values()
    )


def test_optional_training_code_is_frozen_only_when_present(collection, tmp_path):
    _, _, code_root = collection
    optional = "experiments/vision_critic/dataset.py"
    _, absent = prepare(collection, tmp_path / "absent")
    key = "code/" + optional
    assert key not in absent["snapshot"]["code_hashes"]
    path = code_root / optional
    path.parent.mkdir(parents=True)
    path.write_text("# local training implementation\n", encoding="utf-8")
    out, present = prepare(collection, tmp_path / "present")
    assert present["snapshot"]["code_hashes"][key] == sha256_file(path)
    assert (out / key).read_bytes() == path.read_bytes()


def test_optional_training_code_cannot_appear_during_freezing(collection, tmp_path, monkeypatch):
    _, _, code_root = collection
    original = probes._copy_artifact

    def introduce_training_code(*args, **kwargs):
        path = code_root / "experiments/vision_critic/dataset.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# appeared during snapshot\n", encoding="utf-8")
        return original(*args, **kwargs)

    monkeypatch.setattr(probes, "_copy_artifact", introduce_training_code)
    with pytest.raises(ValueError, match="Input/code changed"):
        prepare(collection, tmp_path)


def test_missing_mandatory_evaluator_code_is_not_optional(collection, tmp_path):
    _, _, code_root = collection
    (code_root / "evaluator.py").unlink()
    with pytest.raises(FileNotFoundError):
        prepare(collection, tmp_path)
