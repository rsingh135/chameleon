"""Source-relative semantic evidence and strict, non-authoritative proposals.

Fixtures use native frames without resource directories. Native ink measurement
has separate coverage; these tests load no assets, fonts, images or models.
"""

import hashlib
import json
from copy import deepcopy
from itertools import permutations, product

import pytest

from engine import design_semantics as semantics


def shape(identity, x, y, width, height, **extra):
    return {
        "id": identity,
        "type": "shape",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "shape_type": "rectangle",
        "fill_color": "#334455",
        **extra,
    }


def text(identity, x, y, width, height, content, font_size=24, **extra):
    return {
        "id": identity,
        "type": "text",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "content": content,
        "font_size": font_size,
        "font_family": "OriginalFace",
        "color": "#ffffff",
        **extra,
    }


def uniform(source, scale):
    result = deepcopy(source)
    for key in ("width", "height"):
        result["canvas"][key] *= scale
    for element in result["elements"]:
        for key in ("x", "y", "width", "height", "font_size"):
            if key in element:
                element[key] *= scale
    return result


def proposal(source):
    return {
        "schema_version": 1,
        "source_digest_protocol": semantics.SOURCE_DIGEST_PROTOCOL,
        "source_digest": semantics.semantic_source_digest(source),
        "elements": [
            {
                "element_id": element["id"],
                "purpose": "unknown",
                "confidence": 0.0,
                "evidence": "Meaning is uncertain; retain the original element.",
            }
            for element in source["elements"]
        ],
        "relations": [],
    }


def edge(first, second, kind="reading_before"):
    return {
        "source_id": first,
        "target_id": second,
        "type": kind,
        "confidence": 0.7,
        "evidence": "Supported by source arrangement.",
    }


def resign(descriptor):
    body = {key: value for key, value in descriptor.items() if key != "semantic_digest"}
    descriptor["semantic_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return descriptor


@pytest.fixture
def source():
    return {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [
            shape("e-00", 0, 0, 1000, 1000),
            {
                "id": "e-01",
                "type": "image",
                "x": 30,
                "y": 30,
                "width": 60,
                "height": 60,
                "asset_id": "original-unopened.png",
                "fit": "contain",
            },
            shape("e-02", 250, 260, 500, 200, group_id="authored-cluster"),
            text("e-03", 280, 300, 400, 80, "A distinct message", 48, group_id="authored-cluster"),
            text("e-04", 300, 940, 400, 30, "https://studio.example/design", 20),
            shape("e-05", 450, 700, 40, 40),
            text("e-06", 250, 600, 500, 40, "Supporting context", 24, role="body"),
        ],
    }


def test_every_native_element_has_evidence_and_uncertain_purpose_is_not_invented(source):
    result = semantics.describe_design(source)
    rows = {row["element_id"]: row for row in result["elements"]}
    assert set(rows) == {element["id"] for element in source["elements"]}
    assert {row["type"] for row in rows.values()} == {"text", "image", "shape"}
    assert rows["e-05"]["purpose"] == "unknown"
    assert rows["e-05"]["purpose_confidence"] == 0
    assert rows["e-01"]["atomic_asset"] is True
    assert all(row["confidence_kind"] == "heuristic_uncalibrated" for row in rows.values())
    assert result["model_executed"] is False
    assert result["source_digest_protocol"] == semantics.SOURCE_DIGEST_PROTOCOL
    assert result["source_digest"] == semantics.semantic_source_digest(source)
    assert result == semantics.describe_design(source)


@pytest.mark.parametrize(
    "vertical,horizontal", tuple(product(("top", "bottom"), ("left", "right")))
)
@pytest.mark.parametrize("scale", [0.5, 2])
@pytest.mark.parametrize("kind", ["shape", "image"])
def test_corner_affinity_is_source_relative_in_all_orientations_and_scales(
    vertical, horizontal, scale, kind
):
    width, height, margin = 400 * scale, 800 * scale, 20 * scale
    w, h = 28 * scale, 24 * scale
    marker = shape(
        "opaque-042",
        margin if horizontal == "left" else width - margin - w,
        margin if vertical == "top" else height - margin - h,
        w,
        h,
    )
    if kind == "image":
        marker.update(type="image", asset_id="unopened.png")
    source = {"canvas": {"width": width, "height": height}, "elements": [marker]}
    row = semantics.describe_design(source)["elements"][0]
    assert row["purpose"] == "corner_marker"
    assert row["visible_box_provenance"] == "frame_fallback"
    assert len(row["anchors"]) == 1
    anchor = row["anchors"][0]
    assert set(anchor["edges"]) == {vertical, horizontal}
    assert anchor["insets_short_axis"] == pytest.approx({vertical: 0.05, horizontal: 0.05})
    assert anchor["hard"] is False


@pytest.mark.parametrize(
    "content", ["https://quiet.example/visit", "www.quiet.example", "hi@quiet.example"]
)
def test_footer_destination_keeps_bottom_and_center_evidence(content):
    source = {
        "canvas": {"width": 640, "height": 1000},
        "elements": [text("copy-029", 100, 930, 440, 30, content, 20)],
    }
    row = semantics.describe_design(source)["elements"][0]
    assert row["purpose"] == "footer_destination"
    anchor = row["anchors"][0]
    assert anchor["edges"] == ["bottom"]
    assert anchor["horizontal"] == "center"
    assert anchor["source_center_x"] == 0.5
    report = semantics.audit_semantic_preservation(source, uniform(source, 2))
    assert report["coverage"]["footer_anchors"] == 1
    assert report["metrics"]["footer_anchor_retention"] == pytest.approx(1)


def test_plain_headers_and_bottom_copy_do_not_become_semantic_corner_or_footer_claims():
    source = {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [
            text("a", 30, 30, 200, 40, "Heading", 40),
            text("b", 200, 940, 600, 30, "An unclassified reminder", 20),
        ],
    }
    rows = semantics.describe_design(source)["elements"]
    assert rows[0]["purpose"] == "headline" and rows[0]["anchors"] == []
    assert rows[1]["purpose"] == "supporting_copy" and rows[1]["anchors"] == []


def test_authored_groups_remain_hard_and_sources_and_proposals_are_immutable(source):
    before = deepcopy(source)
    description = semantics.describe_design(source)
    original_description = deepcopy(description)
    groups = [relation for relation in description["relations"] if relation["type"] == "same_group"]
    assert groups and all(relation["hard"] is True for relation in groups)
    draft = proposal(source)
    draft_before = deepcopy(draft)
    validated = semantics.validate_semantic_annotations(source, draft)
    semantics.audit_semantic_preservation(source, uniform(source, 0.5), description)
    assert source == before and draft == draft_before and description == original_description
    validated["elements"][0]["evidence"] = "Changed only the returned copy."
    description["source_canvas"]["width"] = 999
    assert source == before and draft == draft_before


def test_model_roles_are_soft_and_do_not_change_native_elements_or_crop_authority(source):
    before = deepcopy(source)
    measured = semantics.describe_design(source)
    draft = proposal(source)
    for row in draft["elements"]:
        row.update(purpose="background", confidence=1.0)
    validated = semantics.validate_semantic_annotations(source, draft)
    assert validated["authority"] == "soft_proposal_only"
    assert "proposal_digest" in validated
    assert not {"allow_crop", "allow_cover", "geometry", "target_canvas"} & validated.keys()
    assert source == before
    assert semantics.describe_design(source) == measured


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_proposal_requires_exact_once_complete_element_coverage(source, mutation):
    draft = proposal(source)
    if mutation == "missing":
        draft["elements"].pop()
    else:
        draft["elements"][-1]["element_id"] = (
            draft["elements"][0]["element_id"] if mutation == "duplicate" else "outside-source"
        )
    with pytest.raises(ValueError):
        semantics.validate_semantic_annotations(source, draft)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("source_digest_protocol", "different-protocol"),
        ("source_digest", "0" * 64),
    ],
)
def test_annotation_schema_source_and_digest_protocol_must_match(source, field, value):
    draft = proposal(source)
    draft[field] = value
    with pytest.raises(ValueError, match="source or protocol"):
        semantics.validate_semantic_annotations(source, draft)


def test_default_spaced_json_digest_cannot_be_substituted_for_compact_protocol(source):
    spaced = hashlib.sha256(
        json.dumps(source, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    assert spaced != semantics.semantic_source_digest(source)
    draft = proposal(source)
    draft["source_digest"] = spaced
    with pytest.raises(ValueError):
        semantics.validate_semantic_annotations(source, draft)


@pytest.mark.parametrize("where", ["element", "relation"])
@pytest.mark.parametrize("value", [True, False, -0.1, 1.1, float("nan"), float("inf")])
def test_element_and_relation_confidences_reject_booleans_nonfinite_and_out_of_range(
    source, where, value
):
    draft = proposal(source)
    draft["relations"] = [edge("e-03", "e-06")]
    row = draft["elements"][0] if where == "element" else draft["relations"][0]
    row["confidence"] = value
    with pytest.raises(ValueError):
        semantics.validate_semantic_annotations(source, draft)


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", []),
        ("purpose", "invented-purpose"),
        ("evidence", ""),
        ("evidence", "x" * 1001),
        ("evidence", {"text": "not a string"}),
    ],
)
def test_purpose_vocabulary_and_evidence_are_bounded(source, field, value):
    draft = proposal(source)
    draft["elements"][0][field] = value
    with pytest.raises(ValueError):
        semantics.validate_semantic_annotations(source, draft)


@pytest.mark.parametrize(
    "where,field,value",
    [
        ("top", "allow_crop", True),
        ("element", "geometry", {"x": 0}),
        ("element", "allow_cover", True),
        ("relation", "hard", True),
    ],
)
def test_proposals_cannot_smuggle_geometry_crop_or_hard_constraints(source, where, field, value):
    draft = proposal(source)
    draft["relations"] = [edge("e-03", "e-06")]
    target = (
        draft
        if where == "top"
        else (draft["elements"][0] if where == "element" else draft["relations"][0])
    )
    target[field] = value
    with pytest.raises(ValueError, match="fields"):
        semantics.validate_semantic_annotations(source, draft)


@pytest.mark.parametrize(
    "relation",
    [
        edge("e-03", "e-03"),
        edge("e-03", "outside"),
        edge("e-03", "e-06", "move_to"),
        edge([], "e-06"),
    ],
)
def test_relations_connect_two_known_distinct_elements_with_a_known_type(source, relation):
    draft = proposal(source)
    draft["relations"] = [relation]
    with pytest.raises(ValueError):
        semantics.validate_semantic_annotations(source, draft)


def test_duplicate_edges_and_reading_cycles_are_rejected(source):
    draft = proposal(source)
    draft["relations"] = [edge("e-03", "e-06"), edge("e-03", "e-06")]
    with pytest.raises(ValueError, match="Duplicate"):
        semantics.validate_semantic_annotations(source, draft)
    draft["relations"] = [edge("e-03", "e-06"), edge("e-06", "e-04"), edge("e-04", "e-03")]
    with pytest.raises(ValueError, match="cycle"):
        semantics.validate_semantic_annotations(source, draft)


def test_relationship_budget_accepts_its_boundary_and_rejects_one_more():
    source = {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [shape(f"node-{index}", 400, 400, 10, 10) for index in range(16)],
    }
    draft = proposal(source)
    kinds = sorted(semantics.PROPOSED_RELATIONS - {"reading_before"})
    ids = [element["id"] for element in source["elements"]]
    edges = [
        edge(first, second, kind) for (first, second), kind in product(permutations(ids, 2), kinds)
    ]
    draft["relations"] = edges[: semantics.MAX_RELATIONS]
    validated = semantics.validate_semantic_annotations(source, draft)
    assert len(validated["relations"]) == semantics.MAX_RELATIONS
    draft["relations"] = edges[: semantics.MAX_RELATIONS + 1]
    with pytest.raises(ValueError, match="budget"):
        semantics.validate_semantic_annotations(source, draft)


def test_uniform_scaling_preserves_all_measured_relationships_without_visual_approval(source):
    report = semantics.audit_semantic_preservation(source, uniform(source, 0.5))
    assert all(value == pytest.approx(1) for value in report["metrics"].values())
    assert report["coverage"]["corner_anchors"] == 1
    assert report["coverage"]["footer_anchors"] == 1
    assert report["coverage"]["explicit_groups"] == 1
    assert report["violations"] == []
    assert report["visual_quality"] == "not_reviewed"


def test_displaced_corner_is_reported_as_lost_affinity(source):
    resized = uniform(source, 0.5)
    resized["elements"][1].update(x=200, y=200)
    report = semantics.audit_semantic_preservation(source, resized)
    assert any(row["code"] == "corner_affinity" for row in report["violations"])
    assert report["metrics"]["corner_anchor_retention"] < 0.5


@pytest.mark.parametrize("axis", ["x", "y"])
def test_footer_center_and_bottom_affinity_are_both_audited(source, axis):
    resized = uniform(source, 0.5)
    resized["elements"][4][axis] -= 150
    report = semantics.audit_semantic_preservation(source, resized)
    assert any(row["code"] == "footer_affinity" for row in report["violations"])
    assert report["metrics"]["footer_anchor_retention"] < 0.5


@pytest.mark.parametrize("deformation", ["position", "frame", "font"])
def test_explicit_group_internal_geometry_and_font_ratios_are_hard(source, deformation):
    resized = uniform(source, 0.5)
    member = resized["elements"][3]
    if deformation == "position":
        member["x"] += 80
    else:
        member["width" if deformation == "frame" else "font_size"] *= 1.8
    report = semantics.audit_semantic_preservation(source, resized)
    assert any(
        row["severity"] == "hard" and row.get("group_id") == "authored-cluster"
        for row in report["violations"]
    )
    assert report["metrics"]["explicit_group_retention"] < 0.5


def test_a_whole_authored_group_can_move_without_internal_deformation(source):
    resized = uniform(source, 0.5)
    for member in resized["elements"]:
        if member.get("group_id") == "authored-cluster":
            member["x"] += 40
            member["y"] += 20
    report = semantics.audit_semantic_preservation(source, resized)
    assert report["metrics"]["explicit_group_retention"] == pytest.approx(1)
    assert not any(row["severity"] == "hard" for row in report["violations"])


def test_singleton_group_does_not_claim_a_multi_element_rigid_relationship():
    source = {
        "canvas": {"width": 100, "height": 100},
        "elements": [shape("single", 40, 40, 10, 10, group_id="one")],
    }
    report = semantics.audit_semantic_preservation(source, uniform(source, 2))
    assert report["coverage"]["explicit_groups"] == 0


@pytest.mark.parametrize("orientation", ["row", "column"])
def test_reading_order_can_reflow_but_not_reverse(orientation):
    source = {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [
            text("a", 250, 200, 400, 60, "First", role="body"),
            text("b", 250, 400, 400, 60, "Second", role="body"),
            text("c", 250, 600, 400, 60, "Third", role="body"),
        ],
    }
    resized = deepcopy(source)
    if orientation == "row":
        resized["canvas"] = {"width": 1200, "height": 600}
        for index, element in enumerate(resized["elements"]):
            element.update(x=20 + 400 * index, y=200, width=300)
    assert (
        semantics.audit_semantic_preservation(source, resized)["metrics"]["reading_order_retention"]
        == 1
    )
    axis = "x" if orientation == "row" else "y"
    values = [element[axis] for element in resized["elements"]][::-1]
    for element, value in zip(resized["elements"], values, strict=True):
        element[axis] = value
    report = semantics.audit_semantic_preservation(source, resized)
    assert report["metrics"]["reading_order_retention"] == 0
    assert sum(row["code"] == "reading_order" for row in report["violations"]) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_version", "design-semantics-v0"),
        ("schema_version", 2),
        ("schema_version", True),
        ("source_digest_protocol", "other-protocol"),
    ],
)
def test_audit_rejects_unknown_sidecar_versions_even_with_a_consistent_body_hash(
    source, field, value
):
    descriptor = semantics.describe_design(source)
    descriptor[field] = value
    resign(descriptor)
    with pytest.raises(ValueError):
        semantics.audit_semantic_preservation(source, uniform(source, 0.5), descriptor)


@pytest.mark.parametrize("mutation", ["body", "digest", "source", "empty"])
def test_changed_or_stale_sidecar_evidence_is_rejected(source, mutation):
    descriptor = semantics.describe_design(source)
    if mutation == "body":
        descriptor["elements"][0]["purpose"] = "unknown"
    elif mutation == "digest":
        descriptor["semantic_digest"] = "0" * 64
    elif mutation == "source":
        source["elements"][0]["fill_color"] = "#abcdef"
    else:
        descriptor = {}
    with pytest.raises(ValueError):
        semantics.audit_semantic_preservation(source, uniform(source, 0.5), descriptor)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_audit_requires_exact_original_element_identity(source, mutation):
    resized = uniform(source, 0.5)
    if mutation == "missing":
        resized["elements"].pop()
    else:
        resized["elements"].append(shape("new-element", 100, 100, 20, 20))
    with pytest.raises(ValueError, match="original element IDs"):
        semantics.audit_semantic_preservation(source, resized)


@pytest.mark.parametrize("value", [True, 0, float("inf")])
def test_invalid_native_font_numbers_are_rejected(source, value):
    source["elements"][3]["font_size"] = value
    with pytest.raises(ValueError):
        semantics.describe_design(source)


@pytest.mark.parametrize("value", [[], "unsupported-native-kind"])
def test_invalid_native_element_types_are_rejected_as_validation_errors(source, value):
    source["elements"][0]["type"] = value
    with pytest.raises(ValueError):
        semantics.describe_design(source)
