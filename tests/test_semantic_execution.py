"""Bounded source-purpose admission and honest learned geometry selection."""

import json
from copy import deepcopy

import pytest
from PIL import Image

from engine import fit_policy
from engine.design_semantics import describe_design
from engine.layout.semantic_anchors import generate_semantic_anchor_states
from engine.resources import DEFAULT_FONTS_DIR, using_resources
from engine.semantic_execution import _candidate_id, select_semantic_layout


@pytest.fixture
def scene(tmp_path):
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    Image.new("RGB", (40, 20), "#ff7700").save(assets / "identity.png")
    Image.new("RGB", (60, 100), "#0088aa").save(assets / "hero.png")
    (fonts / "face.ttf").write_bytes((DEFAULT_FONTS_DIR / "SolsticeSans-Regular.ttf").read_bytes())
    (fonts / "font-manifest.json").write_text(
        json.dumps(
            [
                {
                    "font_family": "ExactFace",
                    "file": "face.ttf",
                    "weight": "regular",
                    "style": "normal",
                }
            ]
        )
    )
    source = {
        "canvas": {"width": 100, "height": 300},
        "elements": [
            {
                "id": "paint",
                "type": "shape",
                "shape_type": "rectangle",
                "fill_color": "#112233",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 300,
            },
            {
                "id": "identity",
                "type": "image",
                "asset_id": "identity.png",
                "fit": "fill",
                "x": 72,
                "y": 8,
                "width": 20,
                "height": 10,
            },
            {
                "id": "title",
                "type": "text",
                "content": "Join",
                "font_family": "ExactFace",
                "font_size": 16,
                "color": "#ffffff",
                "text_align": "center",
                "x": 20,
                "y": 60,
                "width": 60,
                "height": 20,
                "group_id": "campaign",
            },
            {
                "id": "hero",
                "type": "image",
                "asset_id": "hero.png",
                "fit": "fill",
                "x": 20,
                "y": 90,
                "width": 60,
                "height": 100,
                "group_id": "campaign",
            },
            {
                "id": "footer",
                "type": "text",
                "content": "www.x.org",
                "font_family": "ExactFace",
                "font_size": 10,
                "color": "#ffffff",
                "text_align": "center",
                "x": 10,
                "y": 275,
                "width": 80,
                "height": 15,
            },
        ],
    }
    target = {"width": 600, "height": 150}
    with using_resources(assets, fonts):
        semantics = describe_design(source, assets_dir=assets, fonts_dir=fonts)
        states = generate_semantic_anchor_states(source, target, semantics)
        assert len(states) >= 2
        yield source, target, semantics, [state.design for state in states]


def test_semantic_admission_rejects_scattered_anchor_and_missing_model_stays_honest(
    scene, tmp_path
):
    source, target, semantics, candidates = scene
    bad = deepcopy(candidates[0])
    bad["elements"][1].update(x=20, y=70)
    all_candidates = candidates + [bad]
    snapshot = {}
    selected = select_semantic_layout(
        source,
        target,
        all_candidates,
        {_candidate_id(row) for row in all_candidates},
        semantics=semantics,
        model_path=tmp_path / "missing.json",
        snapshot=snapshot,
    )
    assert selected is not None
    trace = selected["trace"]
    assert trace["fit_policy"]["executed"] is False
    assert trace["fit_policy"]["status"] == "no_model"
    assert trace["semantic_execution"]["learned_purpose_claim"] is False
    assert _candidate_id(bad) not in snapshot["admitted_finalist_ids"]
    rejected = next(
        row for row in snapshot["candidates"] if row["candidate_id"] == _candidate_id(bad)
    )
    assert "corner_anchor_retention" in rejected["diagnostics"]["failures"]
    assert trace["semantic_execution"]["marketing_pixel_profile_evaluated"] is False


def test_actual_trained_policy_ranks_two_semantically_admitted_geometries(scene, tmp_path):
    source, target, semantics, candidates = scene
    records = []
    # Unit-fixture training exercises the actual optimizer, not real-poster
    # generalization. The separate development dataset supplies that evidence.
    for family, split in (("synthetic-a", "train"), ("synthetic-b", "validation")):
        original = {**source, "fixture_family": family}
        menu = fit_policy.build_fit_menu(
            original,
            target,
            [
                {"plan_id": _candidate_id(row), "design": row, "hard_valid": True}
                for row in candidates
            ],
        )
        for index, candidate in enumerate(candidates):
            records.append(
                {
                    "source_id": family,
                    "family": family,
                    "split": split,
                    "source_digest": fit_policy.source_digest(original),
                    "target_canvas": target,
                    "menu_digest": menu["menu_digest"],
                    "plan_id": _candidate_id(candidate),
                    "features": fit_policy.extract_features(original, target, candidate),
                    "teacher_score": 0.9 if index == 0 else 0.4,
                    "hard_valid": True,
                    "design_digest": fit_policy.digest(candidate),
                }
            )
    path = tmp_path / "trained.json"
    fit_policy.train_model(
        {
            "schema_version": 1,
            "feature_version": fit_policy.FEATURE_VERSION,
            "label_kind": fit_policy.LABEL_KIND,
            "records": records,
        },
        path,
        epochs=40,
    )
    selected = select_semantic_layout(
        source,
        target,
        candidates,
        {_candidate_id(row) for row in candidates},
        semantics=semantics,
        model_path=path,
    )
    assert selected is not None
    receipt = selected["trace"]["fit_policy"]
    assert receipt["executed"] is True
    assert receipt["kind"] == "trained_pairwise_fit_policy"
    assert len(receipt["scores"]) == 2
    assert selected["design"] == candidates[0]
    assert selected["trace"]["semantic_execution"]["quality_improved"] is None


def test_sole_admitted_geometry_does_not_claim_model_choice(scene):
    source, target, semantics, candidates = scene
    selected = select_semantic_layout(
        source, target, candidates[:1], {_candidate_id(candidates[0])}, semantics=semantics
    )
    assert selected is not None
    assert selected["trace"]["fit_policy"]["executed"] is False
    assert selected["trace"]["fit_policy"]["status"] == "no_choice"


def test_group_deformation_and_stale_descriptors_do_not_get_execution_authority(scene):
    source, target, semantics, candidates = scene
    changed = deepcopy(candidates[0])
    changed["elements"][3]["x"] += 10
    assert (
        select_semantic_layout(
            source, target, [changed], {_candidate_id(changed)}, semantics=semantics
        )
        is None
    )
    stale = deepcopy(semantics)
    stale["source_digest"] = "a" * 64
    with pytest.raises(ValueError, match="unchanged source"):
        select_semantic_layout(
            source, target, candidates, {_candidate_id(row) for row in candidates}, semantics=stale
        )


def test_unknown_policy_choice_is_rejected_before_geometry_is_applied(scene, monkeypatch):
    source, target, semantics, candidates = scene
    monkeypatch.setattr(
        fit_policy,
        "choose_fit_plan",
        lambda *_a, **_kw: {
            "decision": {"schema_version": 1, "menu_digest": "a" * 64, "plan_id": "invented"},
            "receipt": {"executed": True},
        },
    )
    assert (
        select_semantic_layout(
            source,
            target,
            candidates,
            {_candidate_id(row) for row in candidates},
            semantics=semantics,
        )
        is None
    )
