"""Synthetic intent exercises preserve identity; invalid probes stay segregated."""

import json
from copy import deepcopy

import pytest
from PIL import Image

from engine.eval import build_context, check_contract
from experiments.vision_critic import curriculum
from experiments.vision_critic.contracts import canonical_json_sha256, load_collection
from experiments.vision_critic.dataset import load_preferences, sft_records


@pytest.mark.parametrize("family", curriculum.FAMILIES)
@pytest.mark.parametrize("axis", curriculum.AXES)
def test_counterfactual_changes_geometry_only_and_passes_real_contract(family, axis):
    source, intent = curriculum.fixture(family)
    untouched = deepcopy(source)
    reference = curriculum.uniform_reference(source, curriculum.TARGETS[0])
    changed = curriculum.counterfactual(reference, axis)
    assert source == untouched
    assert reference != changed
    assert [element["id"] for element in source["elements"]] == intent["required_element_ids"]
    assert [element["id"] for element in changed["elements"]] == intent["required_element_ids"]
    context = build_context(source)
    assert check_contract(source, reference, curriculum.TARGETS[0], context) == []
    assert check_contract(source, changed, curriculum.TARGETS[0], context) == []
    by_id = {element["id"]: element for element in source["elements"]}
    for element in changed["elements"]:
        for field in ("content", "font_family", "color", "fill_color", "group_id", "type"):
            assert element.get(field) == by_id[element["id"]].get(field)


def test_hard_policy_probes_trigger_real_rejection_codes():
    source, _ = curriculum.fixture("message-left")
    target = curriculum.TARGETS[2]
    reference = curriculum.uniform_reference(source, target)
    context = build_context(source)
    for probe, expected in curriculum.hard_probes(reference).values():
        assert expected in {
            violation.code for violation in check_contract(source, probe, target, context)
        }


def fake_renderer(design):
    # Different mock bytes preserve the source-hash split check. These flat
    # colors test plumbing only; the separate native test renders real content.
    digest = bytes.fromhex(canonical_json_sha256(design))
    return Image.new(
        "RGB", (design["canvas"]["width"], design["canvas"]["height"]), tuple(digest[:3])
    )


def test_bundle_has_counterbalanced_weak_labels_and_no_production_admission(tmp_path):
    destination = tmp_path / "curriculum"
    result = curriculum.create_curriculum(destination, renderer=fake_renderer)
    collection = load_collection(destination / "collection")
    assert result["pair_count"] == 12
    assert result["real_quality_validated"] is False
    assert collection["purpose"] == "synthetic"
    assert all(case["source_parity_verified"] is False for case in collection["cases"])
    assert all(case["admitted_candidate_ids"] == [] for case in collection["cases"])
    records = load_preferences(
        destination / "collection", destination / "weak-labels.jsonl", allow_synthetic=True
    )
    assert sum(record["decision"] == "A" for record in records) == 6
    assert sum(record["decision"] == "B" for record in records) == 6
    assert {record["split"] for record in records} == {"train", "validation", "test"}
    assert all(record["annotator"] == "synthetic-rule-v1" for record in records)
    assert all(json.loads(row["response"])["issues"] for row in sft_records(records))
    manifest = json.loads((destination / "curriculum.json").read_text())
    assert manifest["human_reviewed_preferences"] is False
    assert len(manifest["hard_probes"]) == 9
    assert all(probe["eligible_for_preferences"] is False for probe in manifest["hard_probes"])
    assert not any(
        "hard-probes" in candidate["design"]
        for case in collection["cases"]
        for candidate in case["candidates"]
    )
    with pytest.raises(ValueError, match="already exist"):
        curriculum.create_curriculum(destination, renderer=fake_renderer)


def test_native_source_and_one_pair_render_with_exact_canvas():
    from engine.eval.render import render_image

    source, _ = curriculum.fixture("centered-stack")
    reference = curriculum.uniform_reference(source, curriculum.TARGETS[0])
    changed = curriculum.counterfactual(reference, "reading_order")
    for design in (source, reference, changed):
        image = render_image(design)
        assert image.size == (design["canvas"]["width"], design["canvas"]["height"])
        assert image.convert("RGB").getextrema() != ((255, 255),) * 3
