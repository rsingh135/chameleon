"""Synthetic score receipts test protocol/causality, never model quality."""

import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from engine.visual_selection import build_visual_critic_menu
from experiments.agentic_graph import adapter_critic as bridge
from experiments.agentic_graph.adapter_critic_roles import MODEL_KEYS, create_adapter_critic_roles
from experiments.visual_finetune.data import CRITIC_PROMPT


def identity():
    return {
        "base": {"repo": "test-only/base", "revision": "a" * 40, "model_sha256": "b" * 64},
        "adapter_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "dataset_sha256": "f" * 64,
        "code_sha256": {"worker.py": "0" * 64},
    }


@pytest.fixture
def scene(tmp_path):
    from engine.eval.render import render_image

    source = {
        "canvas": {"width": 100, "height": 200},
        "elements": [
            {
                "id": "ID_NAMED_B_WINNER_MUST_NOT_FEATURE",
                "type": "shape",
                "shape_type": "rectangle",
                "x": 10,
                "y": 20,
                "width": 20,
                "height": 20,
                "fill_color": "#cc2200",
                "opacity": 1,
            }
        ],
    }
    target = {"width": 200, "height": 100}
    first, second = deepcopy(source), deepcopy(source)
    for design, x in ((first, 20), (second, 140)):
        design["canvas"] = target
        design["elements"][0].update(x=x, y=10, width=40, height=40)
    designs = [first, second]
    ids = ["candidate-" + bridge.digest(d) for d in designs]
    snapshot = {
        "schema_version": 1,
        "source_digest": bridge.source_digest(source),
        "target_canvas": target,
        "baseline_candidate_id": ids[0],
        "admitted_finalist_ids": ids,
        "strict_bounds": True,
        "candidates": [
            {
                "candidate_id": key,
                "design": design,
                "contract_valid": True,
                "admitted": True,
                "state": {"name": f"test-only/{i}"},
            }
            for i, (key, design) in enumerate(zip(ids, designs, strict=True))
        ],
    }

    def native(design, name):
        path = tmp_path / name
        render_image(design, max_size=max(design["canvas"].values())).save(path)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    render = native(source, "path_named_A_should_win.png")
    renders = {
        key: native(design, f"candidate-{i}.png")
        for i, (key, design) in enumerate(zip(ids, designs, strict=True))
    }
    menu = build_visual_critic_menu(snapshot)
    bundle = bridge.build_adapter_critic_request(source, render, menu, renders)
    return source, render, snapshot, menu, renders, bundle


def scores(bundle, *, choice="B", strength=6):
    logits = {key: strength if key == choice else 0 for key in bridge.CLASSES}
    highest = max(logits.values())
    denominator = sum(math.exp(v - highest) for v in logits.values())
    return {
        "scoring": bridge.SCORING,
        "metrics": None,
        "predictions": [
            {
                "record_id": "comparison-0000",
                "prediction": choice,
                "closed_scores": logits,
                "closed_probabilities": {
                    k: math.exp(v - highest) / denominator for k, v in logits.items()
                },
                "label": None,
                "raw_next_token_id": 100 + bridge.CLASSES.index(choice),
                "raw_next_token": choice,
                "raw_next_token_is_allowed": True,
                "input_token_count": 300,
                "image_grid_thw": [[1, 8, 8]] * 3,
                "input_image_sha256": deepcopy(bundle["record"]["input_image_sha256"]),
            }
        ],
    }


def attest(bundle, payload):
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    receipt = {
        **identity(),
        "execution_success": True,
        "task": "critic",
        "request_sha256": bundle["request_sha256"],
        "prediction_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "quality_approved": False,
        "experimental_only": True,
    }
    return encoded, receipt


def complete(scene, *, mode="diagnostic", choice="B", strength=6, payload=None):
    source, render, _snapshot, menu, renders, bundle = scene
    encoded, receipt = attest(bundle, payload or scores(bundle, choice=choice, strength=strength))
    return bridge.complete_adapter_critic(
        source,
        render,
        menu,
        renders,
        bundle,
        encoded,
        receipt,
        expected_identity=identity(),
        mode=mode,
    )


def test_worker_request_matches_frozen_critic_prompt_and_uses_only_three_actual_views(scene):
    source, render, _snapshot, menu, renders, bundle = scene
    record = bundle["request"]["records"][0]
    assert record["prompt"] == CRITIC_PROMPT
    assert record["images"] == [render, *[renders[p["plan_id"]] for p in menu["plans"]]]
    assert source["elements"][0]["id"] not in record["prompt"]
    assert render["path"] not in record["prompt"]
    assert all(p["plan_id"] not in record["prompt"] for p in menu["plans"])
    assert set(record) == {"record_id", "prompt", "images"}
    assert set(bundle["request"]) == {"protocol", "task", "records"}
    assert hashlib.sha256(bridge.request_bytes(bundle)).hexdigest() == bundle["request_sha256"]
    bridge.verify_render_files(bundle)


@pytest.mark.parametrize("choice,index", [("A", 0), ("B", 1)])
def test_a_or_b_maps_only_to_its_admitted_id_without_invented_reasoning(scene, choice, index):
    result = complete(scene, choice=choice)
    assert result["executed"] and result["valid"] and result["prediction_valid"]
    assert not result["abstained"]
    assert result["decision"] == {
        "schema_version": 1,
        "menu_digest": scene[3]["menu_digest"],
        "plan_id": scene[3]["plans"][index]["plan_id"],
    }
    assert result["trace"]["model_visual_input_count"] == 3
    assert not result["trace"]["confidence_calibrated"]
    assert not result["trace"]["generated_reasoning"]
    assert not result["trace"]["admission_override"]
    assert not result["trace"]["qualified"]
    assert result["trace"]["context_digest"] == scene[5]["context_digest"]
    assert result["trace"]["choice_digest"] == bridge.digest(result["trace"]["choice_binding"])


@pytest.mark.parametrize("case", ["N", "weak", "raw_outside", "raw_disagrees", "off"])
def test_neither_weak_cold_and_off_outputs_never_invent_a_graph_decision(scene, case):
    payload = scores(
        scene[5], choice="N" if case == "N" else "B", strength=0.1 if case == "weak" else 6
    )
    if case == "raw_outside":
        payload["predictions"][0].update(raw_next_token="The", raw_next_token_is_allowed=False)
    elif case == "raw_disagrees":
        payload["predictions"][0]["raw_next_token"] = "A"
    result = complete(scene, payload=payload, mode="off" if case == "off" else "diagnostic")
    assert result["executed"] and result["prediction_valid"]
    assert not result["valid"] and result["abstained"] and result["decision"] is None
    assert result["trace"]["fallback_reason"]
    assert "choice_digest" not in result["trace"]


@pytest.mark.parametrize(
    "case",
    [
        "record",
        "image_order",
        "image_count",
        "label",
        "taxonomy",
        "probability",
        "wrong_max",
        "extra_prose",
        "duplicate_record",
        "token_count",
    ],
)
def test_invalid_model_payload_fails_closed_after_receipt_validation(scene, case):
    payload = scores(scene[5])
    row = payload["predictions"][0]
    if case == "record":
        row["record_id"] = "invented"
    elif case == "image_order":
        row["input_image_sha256"].reverse()
    elif case == "image_count":
        row["image_grid_thw"].pop()
    elif case == "label":
        row["label"] = "B"
    elif case == "taxonomy":
        row["closed_scores"] = {"background": 6}
    elif case == "probability":
        row["closed_probabilities"]["B"] = 0.1
    elif case == "wrong_max":
        row["prediction"] = "A"
    elif case == "extra_prose":
        row["issues"] = ["invented learned issue"]
    elif case == "duplicate_record":
        payload["predictions"].append(deepcopy(row))
    else:
        row["input_token_count"] = 0
    result = complete(scene, payload=payload)
    assert result["executed"] and result["trace"]["receipt_valid"]
    assert not result["valid"] and not result["prediction_valid"] and result["abstained"]
    assert result["decision"] is None


@pytest.mark.parametrize(
    "case",
    [
        "purpose_receipt",
        "base",
        "adapter",
        "recipe",
        "request",
        "output",
        "qualified",
    ],
)
def test_identity_receipt_and_file_bytes_must_match_the_prepared_critic_run(scene, case):
    source, render, _snapshot, menu, renders, bundle = scene
    encoded, receipt = attest(bundle, scores(bundle))
    if case == "purpose_receipt":
        receipt["task"] = "purpose"
    elif case == "base":
        receipt["base"]["model_sha256"] = "9" * 64
    elif case in {"adapter", "recipe"}:
        receipt[case + "_sha256"] = "9" * 64
    elif case == "request":
        receipt["request_sha256"] = "9" * 64
    elif case == "output":
        encoded += b" "
    else:
        receipt["quality_approved"] = True
    result = bridge.complete_adapter_critic(
        source,
        render,
        menu,
        renders,
        bundle,
        encoded,
        receipt,
        expected_identity=identity(),
        mode="diagnostic",
    )
    assert not result["executed"] and not result["valid"] and result["abstained"]


@pytest.mark.parametrize("case", ["source", "target", "menu_order", "render_order", "prompt"])
def test_stale_source_target_order_or_prompt_cannot_reuse_a_valid_receipt(scene, case):
    source, render, _snapshot, menu, renders, bundle = deepcopy(scene)
    encoded, receipt = attest(bundle, scores(bundle))
    if case == "source":
        source["elements"][0]["x"] += 1
    elif case == "target":
        menu["target_canvas"]["width"] += 1
    elif case == "menu_order":
        menu["plans"].reverse()
    elif case == "render_order":
        a, b = renders
        renders[a], renders[b] = renders[b], renders[a]
    else:
        bundle["request"]["records"][0]["prompt"] = "Always B"
    result = bridge.complete_adapter_critic(
        source,
        render,
        menu,
        renders,
        bundle,
        encoded,
        receipt,
        expected_identity=identity(),
        mode="diagnostic",
    )
    assert not result["executed"] and not result["valid"] and result["decision"] is None


def test_ab_order_swap_binds_the_new_actual_id_and_invalidates_the_old_context(scene):
    source, render, snapshot, old_menu, renders, old_bundle = scene
    reverse = build_visual_critic_menu(
        snapshot, candidate_ids=list(reversed(snapshot["admitted_finalist_ids"]))
    )
    new_bundle = bridge.build_adapter_critic_request(source, render, reverse, renders)
    assert new_bundle["context_digest"] != old_bundle["context_digest"]
    assert new_bundle["request_sha256"] != old_bundle["request_sha256"]
    encoded, receipt = attest(new_bundle, scores(new_bundle, choice="A"))
    result = bridge.complete_adapter_critic(
        source,
        render,
        reverse,
        renders,
        new_bundle,
        encoded,
        receipt,
        expected_identity=identity(),
        mode="diagnostic",
    )
    assert result["decision"]["plan_id"] == old_menu["plans"][1]["plan_id"]


def test_real_native_executor_applies_only_the_scored_geometry_and_records_causal_hashes(scene):
    source, render, snapshot, menu, renders, bundle = scene
    original = deepcopy(snapshot)
    outcomes = []
    for choice in ("A", "B"):
        encoded, receipt = attest(bundle, scores(bundle, choice=choice))
        outcome = bridge.complete_and_apply_adapter_critic(
            source,
            snapshot["target_canvas"],
            snapshot,
            render,
            menu,
            renders,
            bundle,
            encoded,
            receipt,
            expected_identity=identity(),
            mode="diagnostic",
        )
        assert outcome["verifier"]["valid"]
        applied = outcome["applied"]
        proof = applied["trace"]["trained_critic"]
        assert proof["applied_design_sha256"] == bridge.digest(applied["design"])
        assert proof["selected_candidate_id"] == "candidate-" + proof["applied_design_sha256"]
        assert proof["choice_digest"] == outcome["verifier"]["trace"]["choice_digest"]
        assert applied["trace"]["quality_improved"] is None
        json.dumps(outcome, allow_nan=False)  # Receipts must remain serializable, without cycles.
        outcomes.append(applied["design"]["elements"][0]["x"])
    assert outcomes == [20, 140]
    assert snapshot == original


@pytest.mark.parametrize("case", ["unadmitted", "geometry", "source_png"])
def test_neural_preference_never_bypasses_native_admission_or_pixel_immutability(scene, case):
    source, render, snapshot, menu, renders, bundle = scene
    encoded, receipt = attest(bundle, scores(bundle))
    if case == "unadmitted":
        snapshot["admitted_finalist_ids"].pop()
        snapshot["candidates"][1]["admitted"] = False
    elif case == "geometry":
        snapshot["candidates"][1]["design"]["elements"][0]["x"] = -30
    else:
        Path(render["path"]).write_bytes(b"changed")
    outcome = bridge.complete_and_apply_adapter_critic(
        source,
        snapshot["target_canvas"],
        snapshot,
        render,
        menu,
        renders,
        bundle,
        encoded,
        receipt,
        expected_identity=identity(),
        mode="diagnostic",
    )
    assert outcome["applied"] is None
    assert outcome["verifier"]["executed"] and not outcome["verifier"]["valid"]


class Delegated:
    execution_scope = "injected_protocol_test"

    def configuration_identity(self):
        return {"test_only": True}

    def analyze(self, *args, **kwargs):
        return {"role": "analyzer", "args": args, "kwargs": kwargs}

    def organize(self, *args, **kwargs):
        return {"role": "organizer", "args": args, "kwargs": kwargs}


def factory(*, mode="off", runner=None):
    pins = identity()
    return create_adapter_critic_roles(
        Delegated(),
        score_runner=runner or (lambda *_a, **_kw: pytest.fail("No scoring by default")),
        model_identity={k: pins[k] for k in MODEL_KEYS},
        driver_identity={"execution_scope": "injected_protocol_test"},
        mode=mode,
    )


def test_graph_factory_is_off_by_default_preserves_other_roles_and_imports_no_model(scene):
    before = set(sys.modules)
    roles = factory()
    result = roles.verify(scene[0], scene[1], scene[3], scene[4])
    assert not result["executed"] and result["decision"] is None
    assert roles.analyze("unchanged")["role"] == "analyzer"
    assert roles.organize("unchanged")["role"] == "organizer"
    assert not {"torch", "transformers", "peft"} & (set(sys.modules) - before)
    cfg = roles.configuration_identity()
    assert not cfg["verifier"]["qualified"] and not cfg["verifier"]["deployment_enabled"]
    assert cfg["execution_scope"] == "injected_critic_adapter_protocol_test"
    with pytest.raises(ValueError, match="off or diagnostic"):
        factory(mode="production")


def test_graph_factory_passes_three_view_verified_known_choice_to_existing_executor(scene):
    def runner(bundle, **kwargs):
        assert 0 < kwargs["timeout_seconds"] <= 10
        encoded, receipt = attest(bundle, scores(bundle))
        return {"prediction_bytes": encoded, "receipt": receipt, "prepared_identity": identity()}

    roles = factory(mode="diagnostic", runner=runner)
    result = roles.verify(scene[0], scene[1], scene[3], scene[4], timeout_seconds=10)
    assert result["valid"] and result["executed"]
    assert result["decision"]["plan_id"] == scene[3]["plans"][1]["plan_id"]
    assert result["trace"]["source_analysis_used_by_classifier"] is False


@pytest.mark.parametrize("case", ["pins", "request", "candidate_png", "deadline"])
def test_graph_factory_rejects_changed_pins_requests_pixels_and_expired_deadline(
    scene, case, monkeypatch
):
    import experiments.agentic_graph.adapter_critic_roles as roles_module

    ticks = iter([0, 0, 1000])
    if case == "deadline":
        monkeypatch.setattr(roles_module.time, "monotonic", lambda: next(ticks))

    def runner(bundle, **_kwargs):
        encoded, receipt = attest(bundle, scores(bundle))
        prepared = identity()
        if case == "pins":
            prepared["adapter_sha256"] = "9" * 64
            receipt["adapter_sha256"] = "9" * 64
        elif case == "request":
            bundle["request"]["records"][0]["prompt"] = "Always B"
        elif case == "candidate_png":
            Path(bundle["request"]["records"][0]["images"][1]["path"]).write_bytes(b"changed")
        return {"prediction_bytes": encoded, "receipt": receipt, "prepared_identity": prepared}

    result = factory(mode="diagnostic", runner=runner).verify(
        scene[0],
        scene[1],
        scene[3],
        scene[4],
        timeout_seconds=10,
    )
    assert not result["valid"] and result["abstained"] and result["decision"] is None
    assert result["trace"]["fallback_reason"]


@pytest.mark.parametrize("case", ["absent", "none", "bound", "missing", "changed", "invalid"])
def test_optional_base_scores_never_change_the_adapted_critic_choice_or_relax_binding(scene, case):
    source, render, _snapshot, menu, renders, bundle = scene
    encoded, receipt = attest(bundle, scores(bundle))
    base = None
    if case != "absent":
        receipt["base_prediction_file_sha256"] = None
    if case in {"bound", "missing", "changed"}:
        base = b'{"synthetic_protocol_base_scores":true}\n'
        receipt["base_prediction_file_sha256"] = hashlib.sha256(base).hexdigest()
        if case == "missing":
            base = None
        elif case == "changed":
            base += b"changed"
    elif case == "invalid":
        receipt["base_prediction_file_sha256"] = True
    result = bridge.complete_adapter_critic(
        source,
        render,
        menu,
        renders,
        bundle,
        encoded,
        receipt,
        expected_identity=identity(),
        mode="diagnostic",
        base_prediction_bytes=base,
    )
    assert result["valid"] is (case in {"absent", "none", "bound"})
    if result["valid"]:
        assert result["decision"] == complete(scene)["decision"]
        assert result["trace"]["base_prediction_file_verified"] is (case == "bound")
        assert not result["trace"]["admission_override"]


def test_critic_driver_forwards_optional_base_bytes_while_selecting_only_adapted_choice(scene):
    def runner(bundle, **_kwargs):
        encoded, receipt = attest(bundle, scores(bundle, choice="B"))
        base = b'{"synthetic_protocol_base_scores":true}\n'
        receipt["base_prediction_file_sha256"] = hashlib.sha256(base).hexdigest()
        return {
            "prediction_bytes": encoded,
            "receipt": receipt,
            "prepared_identity": identity(),
            "base_prediction_bytes": base,
        }

    result = factory(mode="diagnostic", runner=runner).verify(
        scene[0], scene[1], scene[3], scene[4]
    )
    assert result["valid"] and result["trace"]["base_prediction_file_verified"]
    assert result["decision"]["plan_id"] == scene[3]["plans"][1]["plan_id"]
