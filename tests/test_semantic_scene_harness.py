"""Source-first artifacts preserve identity, provenance and reserve boundaries."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from engine import design_semantics as semantics
from tools import semantic_scene_harness as harness


@pytest.fixture
def case(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    assets, fonts = inputs / "assets", inputs / "fonts"
    assets.mkdir(parents=True)
    fonts.mkdir()
    Image.new("RGBA", (16, 16), "#8cb09a").save(assets / "mark.png")
    source = {
        "canvas": {"width": 120, "height": 180},
        "elements": [
            {
                "id": "bg",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 0,
                "y": 0,
                "width": 120,
                "height": 180,
                "fill_color": "#ffffff",
            },
            {
                "id": "symbol/../literal-id",
                "type": "image",
                "x": 100,
                "y": 4,
                "width": 16,
                "height": 16,
                "asset_id": "mark.png",
                "fit": "contain",
            },
            {
                "id": "headline",
                "type": "text",
                "x": 20,
                "y": 50,
                "width": 80,
                "height": 30,
                "font_family": "Example",
                "font_size": 16,
                "content": "Original headline",
                "color": "#111111",
                "group_id": "message",
            },
            {
                "id": "support",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 20,
                "y": 90,
                "width": 80,
                "height": 30,
                "fill_color": "#709d8b",
                "group_id": "message",
            },
            {
                "id": "footer",
                "type": "text",
                "x": 10,
                "y": 160,
                "width": 90,
                "height": 10,
                "font_family": "Example",
                "font_size": 7,
                "content": "www.example.org",
                "color": "#111111",
            },
        ],
    }
    # Actual rule descriptions and annotation validation; native font work is
    # replaced because these cases test transport/orchestration, not Skia.
    describe = semantics.describe_design
    monkeypatch.setattr(semantics, "describe_design", lambda source, **_: describe(source))
    monkeypatch.setattr(
        semantics,
        "_ink",
        lambda e, *_: {
            "box": [e[k] for k in ("x", "y", "width", "height")],
            "provenance": "frame_fallback",
            "confidence": 0.0,
            "raster_scale": 1.0,
        },
    )
    monkeypatch.setattr(harness, "_reserve_registry", lambda: ({}, "c" * 64))
    monkeypatch.setattr(harness, "_implementation_hashes", lambda: {"code.py": "a" * 64})
    calls = []

    def render(design, path):
        calls.append(deepcopy(design))
        Image.new("RGBA", tuple(design["canvas"][k] for k in ("width", "height")), "#f6f5f0").save(
            path
        )
        return {"path": str(path), "sha256": harness._sha(path)}

    monkeypatch.setattr(harness, "_render", render)
    return source, assets, fonts, tmp_path / "run", calls


def inspect(case, **kwargs):
    source, assets, fonts, out, _ = case
    return harness.inspect_semantic_scene(source, assets, fonts, out, **kwargs)


def proposal_for(source):
    return {
        "schema_version": 1,
        "source_digest_protocol": semantics.SOURCE_DIGEST_PROTOCOL,
        "source_digest": semantics.semantic_source_digest(source),
        "elements": [
            {
                "element_id": e["id"],
                "purpose": "unknown",
                "confidence": 0.0,
                "evidence": "Reviewed uncertainty; abstain.",
            }
            for e in source["elements"]
        ],
        "relations": [],
    }


def test_source_first_exports_all_original_ids_and_actual_bounded_contexts(case):
    source, _, _, out, calls = case
    original = deepcopy(source)
    report = inspect(case)
    request = json.loads((out / "semantic-model-request.json").read_text())
    described = json.loads((out / "semantic-scene.json").read_text())
    assert source == original and calls == [source]
    ids = [e["id"] for e in source["elements"]]
    assert [r["element_id"] for r in described["elements"]] == ids
    assert [r["element_id"] for r in request["element_contexts"]] == ids
    template = semantics.validate_semantic_annotations(source, request["response_template"])
    assert all(r["purpose"] == "unknown" and r["confidence"] == 0 for r in template["elements"])
    assert set(request["response_constraints"]["purposes"]) == semantics.PURPOSES
    kinds = set(request["response_constraints"]["relationship_types"])
    assert kinds == semantics.PROPOSED_RELATIONS
    assert len(harness._canonical(request).encode()) <= harness.MAX_METADATA_BYTES
    for context in request["element_contexts"]:
        assert context["isolated_element"] is False
        path = Path(context["path"])
        assert path.is_relative_to(out) and ".." not in path.relative_to(out).parts
        assert harness._sha(path) == context["sha256"]
        with Image.open(path) as image:
            assert max(image.size) <= harness.CROP_AXIS
            assert image.size == (context["width"], context["height"])
    assert report["source_render"]["sha256"] == harness._sha(out / "source.png")
    assert report["resource_hashes_after"] == report["resource_hashes"]
    assert report["implementation_after_hashes"] == report["implementation_hashes"]
    assert report["report_digest"] == harness._digest(
        {k: v for k, v in report.items() if k != "report_digest"}
    )
    assert report["training_eligible"] is False
    assert report["model_executed"] is False and report["resize_invoked"] is False
    assert described["model_executed"] is False
    assert report["review"]["status"] == "not_supplied"
    assert not (out / "semantic-invariants.json").exists()


def test_supplied_design_is_audited_and_rendered_without_invoking_resize(case, monkeypatch):
    from engine import eval as evaluation
    from engine import resize

    def forbidden(*args, **kwargs):
        raise AssertionError("The inspection harness must never call resize")

    monkeypatch.setattr(resize, "resize_with_state", forbidden)
    monkeypatch.setattr(evaluation, "check_contract", lambda *_args, **_kwargs: [])
    source, _, _, out, calls = case
    target = deepcopy(source)
    for e in target["elements"]:
        for key in ("x", "y", "width", "height", "font_size"):
            if key in e:
                e[key] /= 2
    target["canvas"] = {"width": 60, "height": 90}
    original = deepcopy(target)
    report = inspect(case, design=target)
    assert target == original and calls == [source, target]
    assert report["audit"]["coverage"]["explicit_groups"] == 1
    assert report["audit"]["contract_violations"] == []
    assert report["audit"]["target_canvas"] == target["canvas"]
    assert report["target_render"]["sha256"] == harness._sha(out / "design.png")
    assert (out / "semantic-invariants.json").is_file()
    assert report["visual_quality"] == "not_reviewed"


def test_supplied_design_contract_failures_remain_visible_in_the_audit(case, monkeypatch):
    from engine import eval as evaluation

    monkeypatch.setattr(evaluation, "check_contract", lambda *_args, **_kwargs: ["Changed copy"])
    report = inspect(case, design=deepcopy(case[0]))
    assert report["audit"]["contract_violations"] == ["Changed copy"]
    assert report["visual_quality"] == "not_reviewed"


def test_validated_proposal_and_explicit_reviewer_never_claim_model_execution_or_admission(case):
    source, _, _, out, _ = case
    path = out.parent / "proposal.json"
    path.write_text(json.dumps(proposal_for(source)))
    report = inspect(
        case,
        model_response=path,
        supervision_origin="agent_visual_review",
        reviewed_by="named test reviewer",
    )
    review = json.loads((out / "semantic-review.json").read_text())
    assert review["proposal"]["authority"] == "soft_proposal_only"
    assert review["proposal_binding"] == "source_json_only"
    assert report["review"]["reviewed_by"] == "named test reviewer"
    assert report["training_eligible"] is False
    assert report["review"]["model_executed_by_harness"] is False
    assert report["review"]["vision_trained_by_harness"] is False
    # A supplied layer does not silently replace or relabel rule evidence.
    described = json.loads((out / "semantic-scene.json").read_text())
    assert described["authority"] == "source_evidence_and_uncalibrated_rules"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "stale", "geometry", "cycle"])
def test_bad_proposal_rejects_before_any_output_or_render(case, fault):
    source, _, _, out, calls = case
    proposal = proposal_for(source)
    if fault == "missing":
        proposal["elements"].pop()
    elif fault == "duplicate":
        proposal["elements"][-1] = deepcopy(proposal["elements"][0])
    elif fault == "unknown":
        proposal["elements"][0]["element_id"] = "invented"
    elif fault == "stale":
        proposal["source_digest"] = "0" * 64
    elif fault == "geometry":
        proposal["elements"][0]["x"] = 100
    else:
        proposal["relations"] = [
            {
                "source_id": a,
                "target_id": b,
                "type": "reading_before",
                "confidence": 1.0,
                "evidence": "Invalid cyclic order",
            }
            for a, b in (("headline", "footer"), ("footer", "headline"))
        ]
    with pytest.raises(ValueError):
        inspect(case, model_response=proposal)
    assert not out.exists() and not calls


@pytest.mark.parametrize(
    "origin, reviewer, with_proposal",
    [
        ("human_review", None, True),
        ("agent_visual_review", "named reviewer", False),
        ("invented", None, False),
        ([], None, False),
        ("unreviewed", "\n", False),
    ],
)
def test_unjustified_review_metadata_is_rejected(case, origin, reviewer, with_proposal):
    with pytest.raises(ValueError):
        inspect(
            case,
            model_response=proposal_for(case[0]) if with_proposal else None,
            supervision_origin=origin,
            reviewed_by=reviewer,
        )
    assert not case[3].exists() and not case[4]


@pytest.mark.parametrize("source_id", [None, "renamed-copy", "opensuse_banner_we_believe"])
def test_reserve_content_is_blocked_even_without_its_catalogue_id(case, monkeypatch, source_id):
    digest = semantics.semantic_source_digest(case[0])
    monkeypatch.setattr(
        harness, "_reserve_registry", lambda: ({"opensuse_banner_we_believe": digest}, "c" * 64)
    )
    with pytest.raises(ValueError, match="Reserve source is blocked"):
        inspect(case, source_id=source_id)
    assert not case[3].exists() and not case[4]


@pytest.mark.parametrize("sid", sorted(harness.MANUAL_RESERVES))
def test_opened_reserve_can_only_export_manual_acceptance_evidence(case, monkeypatch, sid):
    digest = semantics.semantic_source_digest(case[0])
    monkeypatch.setattr(harness, "_reserve_registry", lambda: ({sid: digest}, "c" * 64))
    report = inspect(case, manual_acceptance=True)
    assert report["admission"]["matched_reserve_ids"] == [sid]
    assert report["admission"]["split_role"] == "manual_acceptance"
    assert report["training_eligible"] is False
    assert report["admission"]["training_eligible"] is False


def test_other_reserves_and_falsely_claimed_opened_reserve_are_not_admitted(case, monkeypatch):
    digest = semantics.semantic_source_digest(case[0])
    monkeypatch.setattr(
        harness,
        "_reserve_registry",
        lambda: ({"another_reserve": digest, "opensuse_banner_we_believe": "b" * 64}, "c" * 64),
    )
    with pytest.raises(ValueError, match="Reserve source is blocked"):
        inspect(case, manual_acceptance=True)
    with pytest.raises(ValueError, match="Claimed reserve source identity"):
        inspect(case, manual_acceptance=True, source_id="opensuse_banner_we_believe")


@pytest.mark.parametrize("changed", ["source", "resource", "implementation", "catalogue"])
def test_mutation_cannot_produce_a_completed_receipt(case, monkeypatch, changed):
    source, assets, _, out, _ = case
    if changed == "source":
        describe = semantics.describe_design

        def mutating(value, **kwargs):
            result = describe(value, **kwargs)
            value["elements"][0]["fill_color"] = "#000000"
            return result

        monkeypatch.setattr(semantics, "describe_design", mutating)
    elif changed == "resource":
        render = harness._render

        def mutate_resource(value, path):
            result = render(value, path)
            Image.new("RGBA", (16, 16), "#000000").save(assets / "mark.png")
            return result

        monkeypatch.setattr(harness, "_render", mutate_resource)
    elif changed == "implementation":
        values = iter(({"a": "1"}, {"a": "2"}))
        monkeypatch.setattr(harness, "_implementation_hashes", lambda: next(values))
    else:
        values = iter((({}, "a"), ({}, "b")))
        monkeypatch.setattr(harness, "_reserve_registry", lambda: next(values))
    with pytest.raises(RuntimeError, match="changed"):
        inspect(case)
    assert not (out / "report.json").exists()


def test_request_metadata_bound_is_enforced_before_native_render(case, monkeypatch):
    monkeypatch.setattr(harness, "MAX_METADATA_BYTES", 2000)
    with pytest.raises(ValueError, match="Semantic model request.*1 MiB"):
        inspect(case)
    assert not case[3].exists() and not case[4]


@pytest.mark.parametrize("destination", ["assets", "fonts", "parent", "exists"])
def test_output_cannot_overlap_inputs_or_overwrite_existing_artifacts(case, destination):
    source, assets, fonts, out, _ = case
    paths = {
        "assets": assets / "run",
        "fonts": fonts / "run",
        "parent": assets.parent,
        "exists": out,
    }
    if destination == "exists":
        out.mkdir()
        (out / "sentinel").write_text("preserve")
    with pytest.raises(ValueError):
        harness.inspect_semantic_scene(source, assets, fonts, paths[destination])
    if destination == "exists":
        assert (out / "sentinel").read_text() == "preserve"


def test_bad_supplied_design_identity_is_rejected_before_render(case):
    design = deepcopy(case[0])
    design["elements"][0]["id"] = "invented"
    with pytest.raises(ValueError, match="original element IDs"):
        inspect(case, design=design)
    assert not case[3].exists() and not case[4]


def test_out_of_frame_element_is_retained_with_explicit_unavailable_crop(case):
    case[0]["elements"][1]["x"] = -100
    report = inspect(case)
    row = report["element_contexts"][1]
    assert row["status"] == "unavailable"
    assert row["element_id"] == case[0]["elements"][1]["id"]
    assert report["described_element_count"] == len(case[0]["elements"])


def test_wrong_native_dimensions_do_not_produce_completed_evidence(case, monkeypatch):
    monkeypatch.setattr(harness, "_render", lambda _, path: Image.new("RGB", (3, 4)).save(path))
    with pytest.raises(ValueError, match="Native PNG dimensions"):
        inspect(case)
    assert not (case[3] / "report.json").exists()


def test_resource_budget_rejects_before_render(case, monkeypatch):
    monkeypatch.setattr(harness, "MAX_RESOURCE_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="bounded regular files"):
        inspect(case)
    assert not case[3].exists() and not case[4]


def test_cli_exports_source_and_rejects_duplicate_json_keys(case, capsys):
    source, assets, fonts, out, _ = case
    path = out.parent / "source.json"
    path.write_text(json.dumps(source))
    args = [
        "--source",
        str(path),
        "--assets",
        str(assets),
        "--fonts",
        str(fonts),
        "--out",
        str(out),
    ]
    assert harness.main(args) == 0
    assert json.loads(capsys.readouterr().out)["resize_invoked"] is False
    path.write_text('{"canvas":{},"canvas":{}}')
    with pytest.raises(SystemExit) as failure:
        harness.main(args)
    assert failure.value.code == 1
    assert "Duplicate JSON key" in capsys.readouterr().err


def test_cli_does_not_allow_an_input_under_the_output_directory(case, capsys):
    _, assets, fonts, out, _ = case
    with pytest.raises(SystemExit):
        harness.main(
            [
                "--source",
                str(out / "source.json"),
                "--assets",
                str(assets),
                "--fonts",
                str(fonts),
                "--out",
                str(out),
            ]
        )
    assert "separate from every input" in capsys.readouterr().err
    assert not out.exists()
