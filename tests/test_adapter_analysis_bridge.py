"""Synthetic protocol envelopes only: no actual model execution/quality proof."""

import hashlib
import json
import math
import sys
from copy import deepcopy

import pytest

from experiments.agentic_graph import adapter_analysis as bridge
from experiments.agentic_graph.adapter_roles import MODEL_KEYS, create_adapter_roles


@pytest.fixture
def scene():
    source = {
        "canvas": {"width": 100, "height": 200},
        "elements": [
            {
                "id": "ID_CONTAINS_BACKGROUND_LABEL",
                "type": "shape",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 200,
                "z_index": 10,
                "role": "HIDDEN_ROLE_MUST_NOT_FEATURE",
                "group_id": "HIDDEN_GROUP_MUST_NOT_FEATURE",
            },
            {
                "id": "ID_CONTAINS_HEADLINE_LABEL",
                "type": "text",
                "content": "Summer gathering",
                "x": 10,
                "y": 20,
                "width": 80,
                "height": 30,
                "z_index": 0,
            },
        ],
    }
    source_render = {"path": "private_path_named_background/source.png", "sha256": "1" * 64}
    crops = {
        element["id"]: {
            "path": f"private_path_named_footer/context-{index}.png",
            "sha256": str(index + 2) * 64,
        }
        for index, element in enumerate(source["elements"])
    }
    return source, source_render, crops


def expected():
    return {
        "base": {"repo": "test-only/base", "revision": "a" * 40, "model_sha256": "b" * 64},
        "adapter_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "dataset_sha256": "f" * 64,
        "code_sha256": {"worker.py": "0" * 64},
    }


def scores(bundle, *, alias="background", strength=6, raw_allowed=True):
    result = {"scoring": bridge.SCORING, "metrics": None, "predictions": []}
    for binding in bundle["records"]:
        logits = {name: strength if name == alias else 0 for name in bridge.CLASSES}
        maximum = max(logits.values())
        denominator = sum(math.exp(v - maximum) for v in logits.values())
        probabilities = {name: math.exp(v - maximum) / denominator for name, v in logits.items()}
        result["predictions"].append(
            {
                "record_id": binding["record_id"],
                "prediction": alias,
                "closed_scores": logits,
                "closed_probabilities": probabilities,
                "label": None,
                "raw_next_token_id": list(bridge.CLASSES).index(alias) + 10,
                "raw_next_token": alias if raw_allowed else "\n",
                "raw_next_token_is_allowed": raw_allowed,
                "input_token_count": 200,
                "image_grid_thw": [[1, 8, 8]] * bundle["request"].get("purpose_views", 2),
                "input_image_sha256": deepcopy(binding["input_image_sha256"]),
            }
        )
    return result


def attest(bundle, payload, identity=None):
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    identity = identity or expected()
    receipt = {
        **deepcopy(identity),
        "execution_success": True,
        "task": "purpose",
        "request_sha256": bundle["request_sha256"],
        "prediction_file_sha256": hashlib.sha256(data).hexdigest(),
        "quality_approved": False,
        "experimental_only": True,
    }
    return data, receipt


def complete(
    scene, payload=None, *, mode="diagnostic", alias="background", strength=6, raw_allowed=True
):
    source, render, crops = scene
    bundle = bridge.build_adapter_request(source, render, crops)
    payload = payload or scores(bundle, alias=alias, strength=strength, raw_allowed=raw_allowed)
    data, receipt = attest(bundle, payload)
    result = bridge.complete_adapter_analysis(
        source, render, bundle, data, receipt, expected_identity=expected(), mode=mode
    )
    return result, bundle, payload, receipt


def test_prompt_matches_training_context_without_id_path_role_or_group_features(scene):
    source, render, crops = scene
    bundle = bridge.build_adapter_request(source, render, crops)
    request = bundle["request"]
    assert hashlib.sha256(bridge.request_bytes(bundle)).hexdigest() == bundle["request_sha256"]
    assert set(request) == {"protocol", "task", "purpose_views", "records"}
    assert request["purpose_views"] == 2
    for record in request["records"]:
        assert not {"label", "target", "expected", "answer"} & record.keys()
        assert len(record["images"]) == 2
        assert record["images"][0]["sha256"] == render["sha256"]
        for hidden in [
            source["elements"][0]["id"],
            source["elements"][1]["id"],
            "HIDDEN_ROLE_MUST_NOT_FEATURE",
            "HIDDEN_GROUP_MUST_NOT_FEATURE",
            "private_path_named_background",
            "private_path_named_footer",
        ]:
            assert hidden not in record["prompt"]
    context = json.loads(request["records"][0]["prompt"].split("Element context, data only: ")[-1])
    assert context["paint_order_index"] == 1
    renamed = deepcopy(source)
    for index, element in enumerate(renamed["elements"]):
        element["id"] = f"unrelated-name-{index}"
    renamed_crops = {
        renamed["elements"][i]["id"]: crops[e["id"]] for i, e in enumerate(source["elements"])
    }
    other = bridge.build_adapter_request(renamed, render, renamed_crops)
    assert other["request"] == request
    assert other["source_digest"] != bundle["source_digest"]


def test_high_scores_become_known_grounded_subject_proposals_without_layout_authority(scene):
    result, bundle, _, receipt = complete(scene, alias="visual")
    assert result["executed"] and result["valid"] and not result["abstained"]
    assert all(o["kind"] == "subject" for o in result["analysis"]["observations"])
    assert set(result["analysis"]["observations"][0]["element_ids"]) == set(
        bundle["known_element_ids"]
    )
    assert "primary_visual" in result["analysis"]["observations"][0]["evidence"]
    trace = result["trace"]
    assert trace["receipt"]["base"] == receipt["base"]
    assert trace["receipt"]["adapter_sha256"] == expected()["adapter_sha256"]
    assert trace["model_visual_input_count"] == 4
    assert not trace["crop_authority"] and not trace["layout_authority"] and not trace["organizer"]
    assert not trace["deployment_enabled"] and not trace["qualified"]
    assert not trace["confidence_calibrated"] and not trace["source_element_ids_are_features"]


def test_bridge_is_off_by_default_even_with_valid_high_scores(scene):
    source, render, crops = scene
    bundle = bridge.build_adapter_request(source, render, crops)
    data, receipt = attest(bundle, scores(bundle))
    result = bridge.complete_adapter_analysis(
        source, render, bundle, data, receipt, expected_identity=expected()
    )
    assert result["executed"] and result["valid"] and result["analysis"]["abstain"]
    assert result["trace"]["fallback_reason"] == "bridge_disabled"


@pytest.mark.parametrize("case", ["low", "unknown", "raw_outside", "raw_disagrees"])
def test_cold_ambiguous_unknown_or_nonclass_raw_tokens_abstain(scene, case):
    result, bundle, payload, _ = complete(
        scene,
        alias="unknown" if case == "unknown" else "background",
        strength=0.1 if case == "low" else 6,
        raw_allowed=case != "raw_outside",
    )
    if case == "raw_disagrees":
        for row in payload["predictions"]:
            row["raw_next_token"] = "headline"
        data, receipt = attest(bundle, payload)
        result = bridge.complete_adapter_analysis(
            scene[0],
            scene[1],
            bundle,
            data,
            receipt,
            expected_identity=expected(),
            mode="diagnostic",
        )
    assert result["executed"] and result["valid"] and result["abstained"]
    assert not result["analysis"]["observations"]
    assert all(row["abstention_reason"] for row in result["trace"]["element_predictions"])


@pytest.mark.parametrize("case", ["adapter", "base", "request", "failed", "old_receipt"])
def test_unpinned_failed_or_pre_bridge_receipts_do_not_claim_verified_execution(scene, case):
    source, render, crops = scene
    bundle = bridge.build_adapter_request(source, render, crops)
    data, receipt = attest(bundle, scores(bundle))
    if case == "adapter":
        receipt["adapter_sha256"] = "9" * 64
    elif case == "base":
        receipt["base"]["model_sha256"] = "9" * 64
    elif case == "request":
        receipt["request_sha256"] = "9" * 64
    elif case == "failed":
        receipt["execution_success"] = False
    else:
        receipt = {"adapter_sha256": expected()["adapter_sha256"], "quality_approved": False}
    result = bridge.complete_adapter_analysis(
        source, render, bundle, data, receipt, expected_identity=expected(), mode="diagnostic"
    )
    assert not result["executed"] and not result["valid"] and result["abstained"]


@pytest.mark.parametrize(
    "case", ["unknown_record", "wrong_images", "one_image", "gold_label", "probability", "taxonomy"]
)
def test_bad_scored_records_are_not_turned_into_observations(scene, case):
    source, render, crops = scene
    bundle = bridge.build_adapter_request(source, render, crops)
    payload = scores(bundle)
    row = payload["predictions"][0]
    if case == "unknown_record":
        row["record_id"] = "invented"
    elif case == "wrong_images":
        row["input_image_sha256"].reverse()
    elif case == "one_image":
        row["image_grid_thw"] = [[1, 8, 8]]
    elif case == "gold_label":
        row["label"] = "background"
    elif case == "probability":
        row["closed_probabilities"]["background"] = 0.1
    else:
        row["closed_scores"] = {"A": 10}
    data, receipt = attest(bundle, payload)
    result = bridge.complete_adapter_analysis(
        source, render, bundle, data, receipt, expected_identity=expected(), mode="diagnostic"
    )
    assert result["executed"]  # Receipt verified; model output still failed protocol.
    assert not result["valid"] and result["abstained"]


def test_missing_context_is_explicit_and_unknown_source_ids_cannot_be_bound(scene):
    source, render, crops = scene
    partial = {next(iter(crops)): next(iter(crops.values()))}
    bundle = bridge.build_adapter_request(source, render, partial)
    assert len(bundle["initial_abstentions"]) == 1
    assert bundle["initial_abstentions"][0]["reason"] == "missing_context_image"
    with pytest.raises(ValueError, match="original source IDs"):
        bridge.build_adapter_request(source, render, {"invented": next(iter(crops.values()))})


class Delegated:
    execution_scope = "injected_provider_test"

    def configuration_identity(self):
        return {"delegated": "test only"}

    def organize(self, *args, **kwargs):
        return {"role": "organizer", "args": args, "kwargs": kwargs}

    def verify(self, *args, **kwargs):
        return {"role": "verifier", "args": args, "kwargs": kwargs}


def factory(scene, *, mode="off", score_runner=None, context_builder=None):
    pin = expected()
    return create_adapter_roles(
        Delegated(),
        context_builder=context_builder or (lambda *_a: scene[2]),
        score_runner=score_runner
        or (lambda *_a, **_kw: pytest.fail("Scoring must not run by default")),
        model_identity={key: pin[key] for key in MODEL_KEYS},
        driver_identity={"execution_scope": "injected_protocol_test", "test_fixture": True},
        mode=mode,
    )


def test_graph_adapter_is_off_by_default_preserves_delegation_and_loads_no_model(scene):
    before = set(sys.modules)
    roles = factory(scene, context_builder=lambda *_a: pytest.fail("Crops must not run by default"))
    from experiments.agentic_graph.providers import LocalModelRoles

    assert isinstance(roles, LocalModelRoles)
    assert not {"torch", "transformers", "peft"} & (set(sys.modules) - before)
    result = roles.analyze(scene[0], scene[1])
    assert not result["executed"] and result["abstained"]
    assert roles.organize("unchanged")["role"] == "organizer"
    assert roles.verify("unchanged")["role"] == "verifier"
    identity = roles.configuration_identity()
    assert identity["execution_scope"] == "injected_adapter_protocol_test"
    assert not identity["analyzer"]["deployment_enabled"]


def test_graph_adapter_diagnostic_uses_its_own_two_images_per_element_receipt(scene):
    def runner(bundle, **_kwargs):
        data, receipt = attest(bundle, scores(bundle, alias="headline"))
        return {"prediction_bytes": data, "receipt": receipt, "prepared_identity": expected()}

    roles = factory(scene, mode="diagnostic", score_runner=runner)
    result = roles.analyze(scene[0], scene[1], timeout_seconds=1)
    assert result["valid"] and result["executed"] and not result["abstained"]
    assert result["trace"]["model_visual_input_count"] == 4
    assert result["trace"]["adapter_is_organizer"] is False
    with pytest.raises(ValueError, match="off or diagnostic"):
        factory(scene, mode="qualified")


def test_classifier_role_context_changes_are_instrumented_and_forwarded_without_geometry(scene):
    outputs = []
    for alias in ("background", "headline"):

        def runner(bundle, expected_alias=alias, **_kwargs):
            data, receipt = attest(bundle, scores(bundle, alias=expected_alias))
            return {"prediction_bytes": data, "receipt": receipt, "prepared_identity": expected()}

        roles = factory(scene, mode="diagnostic", score_runner=runner)
        analyzed = roles.analyze(scene[0], scene[1])
        forwarded = roles.organize(scene[0], scene[1], {}, analysis=analyzed["analysis"])
        assert forwarded["kwargs"]["analysis"] == analyzed["analysis"]
        assert set(analyzed["analysis"]) == {
            "schema_version",
            "source_digest",
            "source_render_sha256",
            "observations",
            "abstain",
        }
        assert analyzed["trace"]["source_analysis_digest"] == bridge.digest(analyzed["analysis"])
        outputs.append(analyzed)
    assert (
        outputs[0]["trace"]["source_analysis_digest"]
        != outputs[1]["trace"]["source_analysis_digest"]
    )
    assert all(output["trace"]["layout_authority"] is False for output in outputs)


def isolated_fixture(scene):
    """Explicitly synthetic native receipts: protocol tests do not paint or score."""
    source, _render, _crops = scene
    images, receipts = {}, {}
    for index, element in enumerate(source["elements"]):
        ref = {"path": f"synthetic-native-{index}.png", "sha256": str(index + 4) * 64}
        implementation, runtime = {"test_only": "f" * 64}, {"synthetic": True}
        receipt = {
            "protocol": "native-element-isolation-v1",
            "policy": {"version": "native-element-isolation-v1"},
            "status": "available",
            "clipped": False,
            "resources_unchanged": True,
            "native_single_original_layer": True,
            "frame_or_asset_fallback_used": False,
            "geometry_authority": False,
            "model_executed": False,
            "source_element_sha256": bridge._native_digest(element),
            "checker_png_sha256": ref["sha256"],
            "alpha_png_sha256": "8" * 64,
            "implementation_hashes": implementation,
            "runtime": runtime,
            "implementation_sha256": bridge._native_digest(
                {"files": implementation, "runtime": runtime}
            ),
            "resource_identity": {"resource_digest": "9" * 64, "hashes": {}},
        }
        receipt["receipt_digest"] = bridge._native_digest(receipt)
        images[element["id"]], receipts[element["id"]] = ref, receipt
    return images, receipts


def three_bundle(scene, images=None, receipts=None):
    source, render, crops = scene
    if images is None or receipts is None:
        images, receipts = isolated_fixture(scene)
    return bridge.build_adapter_request(
        source,
        render,
        crops,
        purpose_views=3,
        isolated_images=images,
        isolated_receipts=receipts,
    )


def test_explicit_three_views_uses_frozen_prompt_source_context_and_actual_isolated_binding(scene):
    source, render, crops = scene
    images, receipts = isolated_fixture(scene)
    bundle = three_bundle(scene, images, receipts)
    assert bundle["request"]["purpose_views"] == 3
    assert all(len(row["images"]) == 3 for row in bundle["request"]["records"])
    for index, element in enumerate(source["elements"]):
        record = bundle["request"]["records"][index]
        assert record["images"] == [render, crops[element["id"]], images[element["id"]]]
        assert "third image" in record["prompt"]
        assert element["id"] not in record["prompt"]
    data, receipt = attest(bundle, scores(bundle, alias="headline"))
    result = bridge.complete_adapter_analysis(
        source, render, bundle, data, receipt, expected_identity=expected(), mode="diagnostic"
    )
    assert result["valid"] and result["executed"]
    assert result["trace"]["purpose_views"] == 3
    assert result["trace"]["model_visual_input_count"] == 6
    assert (
        result["trace"]["native_isolation"][source["elements"][0]["id"]]["receipt_digest"]
        == receipts[source["elements"][0]["id"]]["receipt_digest"]
    )
    assert not result["trace"]["layout_authority"] and not result["trace"]["crop_authority"]


@pytest.mark.parametrize("case", ["missing", "unavailable", "clipped", "receipt_missing"])
def test_three_view_unavailable_evidence_abstains_per_element_without_two_view_fallback(
    scene, case
):
    images, receipts = isolated_fixture(scene)
    key = scene[0]["elements"][0]["id"]
    if case == "missing":
        images.pop(key)
    elif case == "unavailable":
        images.pop(key)
        receipts[key] = {"status": "unavailable", "reason": "unsupported original paint"}
    elif case == "clipped":
        receipts[key]["clipped"] = True
    else:
        receipts.pop(key)
    bundle = three_bundle(scene, images, receipts)
    assert len(bundle["records"]) == 1
    assert bundle["initial_abstentions"][0]["element_id"] == key
    assert len(bundle["request"]["records"][0]["images"]) == 3
    assert key not in {r["element_id"] for r in bundle["records"]}
    data, receipt = attest(bundle, scores(bundle))
    result = bridge.complete_adapter_analysis(
        scene[0], scene[1], bundle, data, receipt, expected_identity=expected(), mode="diagnostic"
    )
    assert result["valid"] and result["trace"]["model_visual_input_count"] == 3


@pytest.mark.parametrize(
    "case", ["source", "checker", "receipt", "implementation", "policy", "unknown_id"]
)
def test_three_view_native_evidence_identity_mismatch_fails_closed(scene, case):
    images, receipts = isolated_fixture(scene)
    key = scene[0]["elements"][0]["id"]
    if case == "source":
        receipts[key]["source_element_sha256"] = "0" * 64
    elif case == "checker":
        images[key]["sha256"] = "0" * 64
    elif case == "receipt":
        receipts[key]["receipt_digest"] = "0" * 64
    elif case == "implementation":
        receipts[key]["implementation_sha256"] = "0" * 64
        receipts[key]["receipt_digest"] = bridge._native_digest(
            {k: v for k, v in receipts[key].items() if k != "receipt_digest"}
        )
    elif case == "policy":
        receipts[key]["policy"] = None
    else:
        images["invented-id"] = images[key]
    with pytest.raises(ValueError):
        three_bundle(scene, images, receipts)


def test_three_view_predictions_cannot_attest_only_two_grids_or_reuse_two_view_binding(scene):
    bundle = three_bundle(scene)
    payload = scores(bundle)
    payload["predictions"][0]["image_grid_thw"].pop()
    data, receipt = attest(bundle, payload)
    result = bridge.complete_adapter_analysis(
        scene[0], scene[1], bundle, data, receipt, expected_identity=expected(), mode="diagnostic"
    )
    assert result["executed"] and not result["valid"] and result["abstained"]
    two = bridge.build_adapter_request(*scene)
    data, receipt = attest(two, scores(two))
    result = bridge.complete_adapter_analysis(
        scene[0], scene[1], bundle, data, receipt, expected_identity=expected(), mode="diagnostic"
    )
    assert not result["executed"] and not result["valid"]


def test_default_two_view_bytes_and_graph_api_require_explicit_version_matched_builders(scene):
    images, receipts = isolated_fixture(scene)
    original = bridge.build_adapter_request(*scene)
    explicit = bridge.build_adapter_request(*scene, purpose_views=2)
    assert bridge.request_bytes(original) == bridge.request_bytes(explicit)
    assert "isolated_images" not in original and "purpose_views" not in original
    with pytest.raises(ValueError, match="Two-view"):
        bridge.build_adapter_request(*scene, isolated_images=images, isolated_receipts=receipts)
    pins = expected()
    roles = create_adapter_roles(
        Delegated(),
        purpose_views=3,
        view_builder=lambda *_a: pytest.fail("No native paint when off"),
        score_runner=lambda *_a, **_kw: pytest.fail("No model score when off"),
        model_identity={k: pins[k] for k in MODEL_KEYS},
        driver_identity={"test_only": True},
    )
    assert roles.configuration_identity()["analyzer"]["purpose_views"] == 3
    assert not roles.analyze(scene[0], scene[1])["executed"]
    with pytest.raises(ValueError, match="version-matched"):
        create_adapter_roles(
            Delegated(),
            purpose_views=3,
            context_builder=lambda *_a: scene[2],
            score_runner=lambda *_a, **_kw: None,
            model_identity={k: pins[k] for k in MODEL_KEYS},
            driver_identity={},
        )


def test_malformed_three_view_record_cannot_crash_or_fall_back_during_completion(scene):
    bundle = three_bundle(scene)
    bundle["request"]["records"][0]["images"] = []
    bundle["request_sha256"] = hashlib.sha256(
        (bridge.canonical(bundle["request"]) + "\n").encode()
    ).hexdigest()
    bundle["binding_digest"] = bridge.digest(
        {key: value for key, value in bundle.items() if key != "binding_digest"}
    )
    data, receipt = attest(bundle, scores(bundle))
    result = bridge.complete_adapter_analysis(
        scene[0], scene[1], bundle, data, receipt, expected_identity=expected(), mode="diagnostic"
    )
    assert not result["executed"] and not result["valid"] and result["abstained"]


@pytest.mark.parametrize("case", ["absent", "none", "bound", "missing", "changed", "invalid"])
def test_optional_base_score_hash_requires_actual_bytes_and_preserves_old_receipts(scene, case):
    bundle = bridge.build_adapter_request(*scene)
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
        receipt["base_prediction_file_sha256"] = "not-a-sha"
    result = bridge.complete_adapter_analysis(
        scene[0],
        scene[1],
        bundle,
        encoded,
        receipt,
        expected_identity=expected(),
        mode="diagnostic",
        base_prediction_bytes=base,
    )
    assert result["valid"] is (case in {"absent", "none", "bound"})
    if result["valid"]:
        assert result["trace"]["base_prediction_file_verified"] is (case == "bound")
        assert not result["trace"]["qualified"] and not result["trace"]["layout_authority"]


def test_graph_driver_forwards_actual_optional_base_bytes_without_changing_observations(scene):
    def runner(bundle, **_kwargs):
        encoded, receipt = attest(bundle, scores(bundle, alias="headline"))
        base = b'{"synthetic_protocol_base_scores":true}\n'
        receipt["base_prediction_file_sha256"] = hashlib.sha256(base).hexdigest()
        return {
            "prediction_bytes": encoded,
            "receipt": receipt,
            "prepared_identity": expected(),
            "base_prediction_bytes": base,
        }

    result = factory(scene, mode="diagnostic", score_runner=runner).analyze(scene[0], scene[1])
    assert result["valid"] and result["trace"]["base_prediction_file_verified"]
    assert result["analysis"]["observations"][0]["kind"] == "subject"
