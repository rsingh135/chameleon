"""Artifact integrity, split boundaries and untrusted human-label validation."""

import json
from copy import deepcopy

import pytest

from experiments.vision_critic import contracts as c


def build_collection(root, *, purpose="research", cases=1):
    data = {
        "schema_version": 1,
        "collection_id": "contract-fixture",
        "purpose": purpose,
        "provenance": {
            "engine_revision": "test",
            "working_tree_digest": "0" * 64,
            "policy_version": "fixture-v1",
        },
        "cases": [],
    }
    for index in range(cases):
        folder = root / f"artifacts/case-{index}"
        folder.mkdir(parents=True)
        source = {
            "canvas": {"width": 100, "height": 100},
            "elements": [{"id": "title", "text": f"Source {index}"}],
        }
        (folder / "source.json").write_text(json.dumps(source), encoding="utf-8")
        (folder / "source.png").write_bytes(f"source render {index}".encode())
        case = {
            "case_id": f"case-{index}",
            "source_id": f"source-{index}",
            "family_id": f"family-{index}",
            "split": "train",
            "source_parity_verified": True,
            "training_rights_reviewed": True,
            "rights_reference": "Owned test fixture",
            "target_canvas": {"width": 64, "height": 64},
            "source": {
                "design": f"artifacts/case-{index}/source.json",
                "json_sha256": c.canonical_json_sha256(source),
                "render": f"artifacts/case-{index}/source.png",
                "render_sha256": c.sha256_file(folder / "source.png"),
            },
            "baseline_candidate_id": "candidate-a",
            "admitted_candidate_ids": ["candidate-a"],
            "candidates": [],
        }
        for number, name in enumerate(("a", "b")):
            design = {
                "canvas": dict(case["target_canvas"]),
                "elements": [{"id": "title", "text": f"Source {index}", "x": number}],
            }
            (folder / f"{name}.json").write_text(json.dumps(design), encoding="utf-8")
            (folder / f"{name}.png").write_bytes(f"candidate render {index} {name}".encode())
            case["candidates"].append(
                {
                    "candidate_id": f"candidate-{name}",
                    "design": f"artifacts/case-{index}/{name}.json",
                    "design_sha256": c.canonical_json_sha256(design),
                    "render": f"artifacts/case-{index}/{name}.png",
                    "render_sha256": c.sha256_file(folder / f"{name}.png"),
                    "admitted": name == "a",
                    "diagnostics": {
                        "contract_valid": True,
                        "contract_violations": [],
                        "score": 2 - number,
                        "nested": {"spacing": number},
                    },
                    "provenance": {"element_ids": ["title"]},
                }
            )
        data["cases"].append(case)
    (root / "collection.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def make_label(collection, *, case_index=0, decision="A", annotator="reviewer"):
    case = collection["cases"][case_index]
    a, b = case["candidates"]
    return {
        "schema_version": 1,
        "collection_id": collection["collection_id"],
        "case_id": case["case_id"],
        "pair_id": c.make_pair_id(
            collection["collection_id"], case["case_id"], a["candidate_id"], b["candidate_id"]
        ),
        "candidate_a_id": a["candidate_id"],
        "candidate_b_id": b["candidate_id"],
        "decision": decision,
        "issues": [],
        "annotator": annotator,
        "created_at": "2026-10-29T16:00:00Z",
        "source_render_sha256": case["source"]["render_sha256"],
        "a_render_sha256": a["render_sha256"],
        "b_render_sha256": b["render_sha256"],
    }


def test_load_collection_verifies_original_bytes_and_canonical_json(tmp_path):
    data = build_collection(tmp_path)
    path = tmp_path / data["cases"][0]["candidates"][0]["design"]
    path.write_text(json.dumps(json.loads(path.read_text()), indent=4), encoding="utf-8")
    assert c.load_collection(tmp_path) == data
    assert c.canonical_json_sha256({"b": "é", "a": 1}) == c.canonical_json_sha256(
        {"a": 1, "b": "é"}
    )


@pytest.mark.parametrize("artifact", ["source.png", "a.png", "source.json", "b.json"])
def test_tampered_artifact_is_rejected(tmp_path, artifact):
    build_collection(tmp_path)
    path = tmp_path / "artifacts/case-0" / artifact
    path.write_bytes(b"tampered")
    with pytest.raises(c.ValidationError):
        c.load_collection(tmp_path)


@pytest.mark.parametrize(
    "path", ["../secret", "/absolute/file", "C:\\secret", "a/../b", "a//b", "CON.json"]
)
def test_unsafe_relative_path_is_rejected(tmp_path, path):
    data = build_collection(tmp_path)
    data["cases"][0]["source"]["render"] = path
    with pytest.raises(c.ValidationError):
        c.validate_collection(data)


def test_artifact_symlink_cannot_escape_collection(tmp_path):
    data = build_collection(tmp_path)
    link = tmp_path / "linked.png"
    try:
        link.symlink_to(tmp_path / "artifacts/case-0/a.png")
    except OSError:
        pytest.skip("This host cannot create symlinks")
    data["cases"][0]["source"]["render"] = "linked.png"
    with pytest.raises(c.ValidationError, match="links"):
        c.validate_collection(data, tmp_path)


def test_family_and_source_splits_are_checked_across_unlabeled_cases(tmp_path):
    data = build_collection(tmp_path, cases=2)
    data["cases"][1]["split"] = "test"
    data["cases"][1]["family_id"] = data["cases"][0]["family_id"]
    with pytest.raises(c.ValidationError, match="Family leakage"):
        c.validate_collection(data)
    data["cases"][1]["family_id"] = "different-family"
    data["cases"][1]["source_id"] = data["cases"][0]["source_id"]
    with pytest.raises(c.ValidationError, match="Source identity"):
        c.validate_collection(data)


def test_identical_source_hashes_cannot_hide_behind_new_ids(tmp_path):
    data = build_collection(tmp_path, cases=2)
    data["cases"][1]["split"] = "validation"
    data["cases"][1]["source"] = deepcopy(data["cases"][0]["source"])
    with pytest.raises(c.ValidationError, match="Identical source"):
        c.validate_collection(data)


def test_source_id_cannot_change_original_within_one_split(tmp_path):
    data = build_collection(tmp_path, cases=2)
    data["cases"][1]["family_id"] = data["cases"][0]["family_id"]
    data["cases"][1]["source_id"] = data["cases"][0]["source_id"]
    with pytest.raises(c.ValidationError, match="inconsistent original"):
        c.validate_collection(data)


@pytest.mark.parametrize("mutation", ["baseline", "admitted", "duplicate", "elements", "canvas"])
def test_candidate_contract_mismatches_rejected(tmp_path, mutation):
    data = build_collection(tmp_path)
    case = data["cases"][0]
    if mutation == "baseline":
        case["baseline_candidate_id"] = "unknown"
    elif mutation == "admitted":
        case["admitted_candidate_ids"] = []
    elif mutation == "duplicate":
        case["candidates"][1] = deepcopy(case["candidates"][0])
    elif mutation == "elements":
        case["candidates"][0]["provenance"]["element_ids"] = ["unknown"]
    else:
        case["target_canvas"]["width"] = 99
    with pytest.raises(c.ValidationError):
        c.validate_collection(data, tmp_path)


def test_grounded_label_and_oriented_identity(tmp_path):
    data = build_collection(tmp_path)
    label = make_label(data)
    label["issues"] = [{"candidate": "B", "code": "weak_hierarchy", "element_ids": ["title"]}]
    assert c.validate_label(label, data) == label
    assert c.make_pair_id("collection", "case", "a", "b") != c.make_pair_id(
        "collection", "case", "b", "a"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("pair_id", "0" * 64),
        ("candidate_b_id", "unknown"),
        ("decision", "execute shell"),
        ("a_render_sha256", "0" * 64),
        ("collection_id", "another"),
        ("created_at", "2026-10-29T16:00:00"),
        ("schema_version", True),
    ],
)
def test_invalid_or_stale_labels_rejected(tmp_path, field, value):
    data = build_collection(tmp_path)
    label = make_label(data)
    label[field] = value
    with pytest.raises(c.ValidationError):
        c.validate_label(label, data)


@pytest.mark.parametrize(
    "issue",
    [
        {"candidate": "C", "code": "weak_hierarchy", "element_ids": []},
        {"candidate": "A", "code": "run_command", "element_ids": []},
        {"candidate": "A", "code": "weak_hierarchy", "element_ids": ["nonexistent"]},
        {"candidate": "A", "code": "weak_hierarchy", "element_ids": [], "command": "bad"},
        {"candidate": {}, "code": "weak_hierarchy", "element_ids": []},
    ],
)
def test_invalid_grounded_issue_rejected(tmp_path, issue):
    data = build_collection(tmp_path)
    label = make_label(data)
    label["issues"] = [issue]
    with pytest.raises(c.ValidationError):
        c.validate_label(label, data)


def test_unknown_label_fields_and_invalid_contract_pair_rejected(tmp_path):
    data = build_collection(tmp_path)
    label = make_label(data)
    label["geometry"] = {"x": 4}
    with pytest.raises(c.ValidationError):
        c.validate_label(label, data)
    del label["geometry"]
    data["cases"][0]["candidates"][1]["diagnostics"]["contract_valid"] = False
    with pytest.raises(c.ValidationError, match="contract-valid"):
        c.validate_label(label, data)


def test_nonfinite_and_duplicate_json_keys_are_rejected():
    with pytest.raises(c.ValidationError):
        c.parse_json('{"value":NaN}')
    with pytest.raises(c.ValidationError):
        c.parse_json('{"value":1,"value":2}')


def test_synthetic_parity_bypass_requires_explicit_opt_in_and_cannot_affect_research(tmp_path):
    data = build_collection(tmp_path, purpose="synthetic")
    data["cases"][0]["source_parity_verified"] = False
    label = make_label(data)
    with pytest.raises(c.ValidationError, match="Synthetic labeling"):
        c.validate_label(label, data)
    assert c.validate_label(label, data, allow_synthetic=True) == label
    assert data["cases"][0]["source_parity_verified"] is False
    with pytest.raises(c.ValidationError, match="explicit boolean"):
        c.validate_label(label, data, allow_synthetic="false")
    data["purpose"] = "research"
    with pytest.raises(c.ValidationError, match="Source parity"):
        c.validate_label(label, data, allow_synthetic=True)


def test_public_grounded_issue_contract_is_side_specific_and_does_not_mutate():
    issues = [{"candidate": "A", "code": "poor_spacing", "element_ids": ["a-title"]}]
    valid = c.validate_grounded_issues(issues, {"A": ["a-title"], "B": ["b-title"]})
    assert valid == issues and valid is not issues
    with pytest.raises(c.ValidationError, match="unknown candidate element"):
        c.validate_grounded_issues(issues, {"A": ["different"], "B": ["a-title"]})
