"""Pure temporary project authority tests; fixture PNGs are not native training data."""

from __future__ import annotations

from copy import deepcopy

import pytest
from PIL import Image

from experiments.vision_critic.contracts import canonical_json_sha256, sha256_file
from tools import project_critic_authority as project
from tools import semantic_training_data as data


def write(path, value):
    data._write(path, value)


def descriptor(root, path):
    source = data._read(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "raw_sha256": data._sha(path),
        "compact_sha256": data._digest(source),
        "spaced_sha256": project._spaced(source),
    }


@pytest.fixture
def package(tmp_path, monkeypatch):
    root, repo = tmp_path / "evidence", tmp_path / "repo"
    root.mkdir()
    repo.mkdir()
    producer = root / "parent-evidence/fit_policy_dataset.py"
    producer.parent.mkdir()
    producer.write_text("# Synthetic test authoring producer\n", encoding="utf-8")
    monkeypatch.setattr(project, "PRODUCER_SHA256", data._sha(producer))
    monkeypatch.setattr(project, "ROOT", repo)
    worker = repo / "worker.py"
    worker.write_text("# Fixture producer attestation, not executed\n", encoding="utf-8")
    (root / "resources/assets").mkdir(parents=True)
    (root / "resources/fonts").mkdir()
    font = root / "resources/fonts/SolsticeSans-Regular.ttf"
    font.write_bytes(b"exact-fixture-face-not-loaded")
    monkeypatch.setattr(project, "FONT_SHA256", data._sha(font))
    write(
        root / "resources/fonts/font-manifest.json",
        [
            {
                "font_family": "Solstice Sans",
                "file": font.name,
                "weight": "regular",
                "style": "normal",
            }
        ],
    )
    notice = root / "licenses/Poppins-OFL.txt"
    notice.parent.mkdir()
    notice.write_text("Synthetic fixture notice; not a licence grant", encoding="utf-8")
    monkeypatch.setattr(project, "OFL_SHA256", data._sha(notice))
    rights = {
        "origin": project.RIGHTS_ORIGIN,
        "public_license_granted": False,
        "authorization_basis": "explicit user instruction for internal training/testing",
        "authorization_reference": "test-session",
        "parent_producer_reference": "experiments/fit_policy_dataset.py@" + project.PRODUCER_SHA256,
        "acquired_utc": "2026-10-30T12:00:00Z",
        "acquired_utc_scope": "local_evidence_package_acquisition",
        "attribution": "Chameleon project-authored synthetic fixture; "
        "bundled Poppins font under OFL-1.1",
    }
    sources, native_sources, parent_rows, parents = [], [], [], {}
    for index, (family, (stem, _)) in enumerate(project.PARENTS.items()):
        identity = family + "__rgba_v1"
        directory = root / "sources" / identity
        directory.mkdir(parents=True)
        original = {
            "canvas": {"width": 80 + index, "height": 60 + index},
            "elements": [
                {
                    "id": "paper",
                    "type": "shape",
                    "shape_type": "rectangle",
                    "fill_color": "#203947",
                    "x": index + 1,
                    "y": 10,
                    "width": 20,
                    "height": 10,
                    "z_index": 0,
                }
            ],
        }
        parent_path = directory / "parent-source.json"
        write(parent_path, original)
        parents[family] = (stem, data._sha(parent_path))
        source = deepcopy(original)
        source["elements"][0]["fill_color"] = "rgba(32,57,71,1)"
        source_path = directory / "source.json"
        write(source_path, source)
        image_path = directory / "source.png"
        Image.new("RGB", (80 + index, 60 + index), (index, 30, 50)).save(image_path)
        audit_path = directory / "source-color-transform.json"
        write(audit_path, project.color_transform_audit(original, source))
        parent_descriptor = {
            **descriptor(root, parent_path),
            "parent_source_id": family,
            "parent_repo_path": "output/fit-policy-training-expanded/synthetic-sources/"
            + stem
            + ".json",
        }
        resources = data._resource_receipts(source, root)
        authority_row = {
            "source_id": identity,
            "family": family,
            "split": "train",
            "parent_source": parent_descriptor,
            "source": descriptor(root, source_path),
            "render": {
                "path": image_path.relative_to(root).as_posix(),
                "sha256": data._sha(image_path),
                "width": 80 + index,
                "height": 60 + index,
            },
            "resource_roots": {"assets_dir": "resources/assets", "fonts_dir": "resources/fonts"},
            "resources": resources,
            "legal_files_sha256": {"licenses/Poppins-OFL.txt": data._sha(notice)},
            "transform": {
                "version": project.TRANSFORM_VERSION,
                "path": audit_path.relative_to(root).as_posix(),
                "sha256": data._sha(audit_path),
            },
            "rights": deepcopy(rights),
        }
        candidates = {}
        for slot, x in (("narrow", 2), ("fit1", 12), ("fit2", 22)):
            old = deepcopy(original)
            old["canvas"] = {"width": 100, "height": 80}
            old["elements"][0]["x"] = x
            new = deepcopy(old)
            new["elements"][0]["fill_color"] = "rgba(32,57,71,1)"
            old_path, new_path = directory / f"parent-{slot}.json", directory / f"{slot}.json"
            write(old_path, old)
            write(new_path, new)
            png = directory / f"{slot}.png"
            Image.new("RGB", (100, 80), (index, x, 90)).save(png)
            candidates[slot] = {
                "parent": {
                    **descriptor(root, old_path),
                    "dataset_design_digest": data._digest(old),
                },
                "derived": descriptor(root, new_path),
                "design": new,
                "transform": project.color_transform_audit(old, new),
                "render": {
                    "path": png.relative_to(root).as_posix(),
                    "sha256": data._sha(png),
                    "dimensions": [100, 80],
                },
            }
            parent_rows.append(
                {
                    "source_id": family,
                    "family": family,
                    "split": "train",
                    "source_digest": project._spaced(original),
                    "design_digest": data._digest(old),
                }
            )
        native_sources.append(
            {
                "source_id": identity,
                "family_id": family,
                "split": "train",
                "source": authority_row["source"],
                "source_render": {
                    "sha256": data._sha(image_path),
                    "dimensions": [80 + index, 60 + index],
                },
                "candidates": candidates,
            }
        )
        sources.append(authority_row)
    monkeypatch.setattr(project, "PARENTS", parents)
    parent_rows.append(
        {
            "source_id": project.OPERATIONS_ID,
            "family": project.OPERATIONS_ID,
            "split": "validation",
            "source_digest": project.OPERATIONS_SPACED,
        }
    )
    parent_path = root / "parent-evidence/dataset.json"
    write(parent_path, {"label_kind": "synthetic_geometry_teacher_v1", "records": parent_rows})
    monkeypatch.setattr(project, "PARENT_DATASET_SHA256", data._sha(parent_path))
    core_path = root / "core-evidence.json"
    write(
        core_path,
        {
            "protocol": "critic-valid-fit-evidence-v3",
            "sources": native_sources,
            "parent_dataset": {"raw_sha256": data._sha(parent_path)},
            "model_executed": False,
            "gpu_used": False,
        },
    )
    native_path = root / "native-generation-receipt.json"
    write(
        native_path,
        {
            "protocol": "native-critic-valid-fit-generation-v3",
            "status": "complete",
            "core_evidence": core_path.name,
            "core_evidence_raw_sha256": data._sha(core_path),
            "implementation_before_hashes": {worker.name: data._sha(worker)},
            "implementation_after_hashes": {worker.name: data._sha(worker)},
        },
    )
    authority_path = root / "project-source-authority.json"
    write(
        authority_path,
        {
            "schema_version": 1,
            "protocol": project.PROTOCOL,
            "parent_dataset": {
                "path": "parent-evidence/dataset.json",
                "sha256": data._sha(parent_path),
            },
            "producer": {
                "path": "parent-evidence/fit_policy_dataset.py",
                "sha256": data._sha(producer),
            },
            "generation_receipt": {"path": native_path.name, "sha256": data._sha(native_path)},
            "sources": sources,
        },
    )
    return authority_path


def refresh(package):
    root = package.parent
    document = data._read(package)
    core_path, native_path = root / "core-evidence.json", root / "native-generation-receipt.json"
    core = data._read(core_path)
    for source in document["sources"]:
        source["source"].update(descriptor(root, root / source["source"]["path"]))
        audit_path = root / source["transform"]["path"]
        audit = data._read(audit_path)
        audit["derived_compact_sha256"] = source["source"]["compact_sha256"]
        write(audit_path, audit)
        source["transform"]["sha256"] = data._sha(audit_path)
        original = next(row for row in core["sources"] if row["source_id"] == source["source_id"])
        original["source"] = deepcopy(source["source"])
    write(core_path, core)
    native = data._read(native_path)
    native["core_evidence_raw_sha256"] = data._sha(core_path)
    write(native_path, native)
    document["generation_receipt"]["sha256"] = data._sha(native_path)
    write(package, document)


def test_exact_project_authority_is_portable_bounded_and_keeps_public_gate(package):
    before = {path: sha256_file(path) for path in package.parent.rglob("*") if path.is_file()}
    allowed = data.ALLOWED_SOURCES
    result = project.load_project_authority(package)
    assert len(result["owners"]) == 3
    assert {row["split"]["family"] for row in result["owners"].values()} == set(project.PARENTS)
    assert all(row["split"]["split"] == "train" for row in result["owners"].values())
    assert all(len(row["candidate_bindings"]) == 3 for row in result["owners"].values())
    assert all(
        not row["receipt"]["rights"]["public_license_granted"] for row in result["owners"].values()
    )
    assert project.OPERATIONS_COMPACT in result["forbidden_source_hashes"]
    assert allowed == data.ALLOWED_SOURCES and len(allowed) == 8
    assert before == {path: sha256_file(path) for path in before}


@pytest.mark.parametrize(
    "field,value", [("x", 30), ("width", 19), ("z_index", 1), ("id", "changed")]
)
def test_geometry_or_identity_cannot_hide_in_rgba_transform(package, field, value):
    document = data._read(package)
    path = package.parent / document["sources"][0]["source"]["path"]
    source = data._read(path)
    source["elements"][0][field] = value
    write(path, source)
    refresh(package)
    with pytest.raises(ValueError, match="other than equivalent HEX"):
        project.load_project_authority(package)


def test_rgba_channels_copy_and_geometry_are_independently_checked():
    parent = {
        "elements": [{"id": "same", "color": "#123456", "content": "Exact copy", "font_size": 12}]
    }
    derived = deepcopy(parent)
    derived["elements"][0]["color"] = "rgba(18,52,86,1)"
    assert len(project.color_transform_audit(parent, derived)["changes"]) == 1
    derived["elements"][0]["color"] = "rgba(19,52,86,1)"
    with pytest.raises(ValueError, match="visible colour"):
        project.color_transform_audit(parent, derived)
    derived["elements"][0].update(color="rgba(18,52,86,1)", content="Altered copy")
    with pytest.raises(ValueError, match="other than equivalent HEX"):
        project.color_transform_audit(parent, derived)


def test_operations_or_unlisted_family_is_never_authorized(package):
    document = data._read(package)
    document["sources"][0].update(
        family=project.OPERATIONS_ID, source_id=project.OPERATIONS_ID + "__rgba_v1"
    )
    write(package, document)
    with pytest.raises(ValueError, match="three exact inherited TRAIN"):
        project.load_project_authority(package)


def test_public_licence_claim_font_notice_and_source_png_substitution_reject(package):
    document = data._read(package)
    document["sources"][0]["rights"]["public_license_granted"] = True
    write(package, document)
    with pytest.raises(ValueError, match="never a public licence"):
        project.load_project_authority(package)
    document["sources"][0]["rights"]["public_license_granted"] = False
    write(package, document)
    (package.parent / "licenses/Poppins-OFL.txt").write_text("replaced notice", encoding="utf-8")
    with pytest.raises(ValueError, match="OFL licence bytes"):
        project.load_project_authority(package)


def test_path_escape_and_native_receipt_change_are_rejected(package):
    document = data._read(package)
    original = deepcopy(document)
    document["producer"]["path"] = "../outside.py"
    write(package, document)
    with pytest.raises(ValueError, match="relative|portable|escaped"):
        project.load_project_authority(package)
    write(package, original)
    native = package.parent / "native-generation-receipt.json"
    native.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash differs"):
        project.load_project_authority(package)


def test_candidate_geometry_cannot_hide_in_colour_derivation(package):
    root = package.parent
    core_path = root / "core-evidence.json"
    core = data._read(core_path)
    candidate = core["sources"][0]["candidates"]["fit1"]
    path = root / candidate["derived"]["path"]
    changed = data._read(path)
    changed["elements"][0]["width"] -= 1
    write(path, changed)
    candidate["derived"] = descriptor(root, path)
    candidate["design"] = changed
    write(core_path, core)
    refresh(package)
    with pytest.raises(ValueError, match="other than equivalent HEX"):
        project.load_project_authority(package)


def test_optional_adapter_freeze_copies_portable_authority_without_public_rights(
    package, tmp_path, monkeypatch
):
    from tools import critic_pair_data as critic

    registry = project.load_project_authority(package)
    monkeypatch.setattr(critic, "ROOT", project.ROOT)
    monkeypatch.setattr("engine.eval.context.get_measurer", lambda *_: object())
    monkeypatch.setattr("engine.eval.contract.get_measurer", lambda *_: object())
    write(project.ROOT / "fixtures/open_posters/catalogue.json", {"examples": []})
    base_manifest, split_path = tmp_path / "base/manifest.json", tmp_path / "split.json"
    write(base_manifest, {"frozen_split": {"sources": []}})
    write(split_path, {"schema_version": 1})
    monkeypatch.setattr(critic, "_authority", lambda *_: ({"frozen_split": {"sources": []}}, {}))
    core = data._read(package.parent / "core-evidence.json")
    original, other = core["sources"][:2]
    authority_doc = data._read(package)
    source_row = authority_doc["sources"][0]

    def candidate(slot):
        value = original["candidates"][slot]
        return {
            "design": value["derived"]["path"],
            "design_sha256": canonical_json_sha256(value["design"]),
            "render": value["render"]["path"],
            "render_sha256": value["render"]["sha256"],
        }

    record = {
        "record_id": "project-pair",
        "pair_id": "project-pair",
        "source_id": original["source_id"],
        "family_id": original["family_id"],
        "split": "train",
        "label": "A",
        "label_origin": "algorithmic_weak",
        "source": {
            "design": original["source"]["path"],
            "json_sha256": canonical_json_sha256(
                registry["owners"][original["source_id"]]["source"]
            ),
            "render": source_row["render"]["path"],
            "render_sha256": source_row["render"]["sha256"],
        },
        "A": candidate("fit1"),
        "B": candidate("narrow"),
        "target_canvas": {"width": 100, "height": 80},
        "rights": source_row["rights"],
        "provenance": {"weak_fit_comparison": True},
    }
    mismatched = deepcopy(record)
    other_row = authority_doc["sources"][1]
    mismatched.update(
        record_id="project-mismatch",
        pair_id="project-mismatch",
        label="N",
        control="source_mismatch",
        provenance={
            "reason": "source_mismatch",
            "mismatched_source_id": other["source_id"],
            "displayed_source_parity": False,
        },
    )
    mismatched["source"].update(
        render=other_row["render"]["path"], render_sha256=other_row["render"]["sha256"]
    )
    input_path = package.parent / "input.json"
    write(
        input_path,
        {
            "schema_version": 1,
            "protocol": "critic-pair-evidence-v1",
            "records": [record, mismatched],
            "resource_roots": {original["source_id"]: source_row["resource_roots"]},
        },
    )
    generation_path = package.parent / "generation-receipt.json"
    worker = project.ROOT / "worker.py"
    write(
        generation_path,
        {
            "status": "complete",
            "input": input_path.name,
            "input_raw_sha256": data._sha(input_path),
            "input_canonical_sha256": canonical_json_sha256(data._read(input_path)),
            "records": 2,
            "train_sources": [original["source_id"]],
            "split_sha256": data._sha(split_path),
            "frozen_manifest_sha256": data._sha(base_manifest),
            "implementation_hashes": {worker.name: data._sha(worker)},
            "implementation_after_hashes": {worker.name: data._sha(worker)},
            "project_source_authority": package.name,
            "project_source_authority_sha256": data._sha(package),
            "parent_native_generation_receipt_sha256": authority_doc["generation_receipt"][
                "sha256"
            ],
        },
    )
    out = tmp_path / "frozen"
    result = critic.freeze_pairs(
        input=input_path,
        source_manifest=base_manifest,
        split_manifest=split_path,
        generation_receipt=generation_path,
        project_source_authority=package,
        out=out,
        swap=True,
    )
    assert result["counts"]["records_including_swaps"] == 4
    assert result["counts"]["labels"] == {"A": 1, "B": 1, "N": 2}
    assert sum(row["source_render_parity"] is False for row in result["records"]) == 2
    assert all(
        row["rights"]["origin"] == project.RIGHTS_ORIGIN
        for row in [data._read(out / entry["path"]) for entry in result["receipts"]]
    )
    assert (out / result["project_source_authority"]["path"]).is_file()
    assert (out / "project-authority/repository-code/worker.py").is_file()
    assert all(sha256_file(out / name) == value for name, value in result["files_sha256"].items())
    relocated = project.load_project_authority(out / result["project_source_authority"]["path"])
    assert len(relocated["owners"]) == 3
