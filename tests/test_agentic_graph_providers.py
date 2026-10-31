"""Real request protocols and receipt admission without model/GPU execution."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from engine.visual_analysis import bind_analysis
from engine.visual_pipeline import _io
from engine.visual_plan import source_digest
from experiments.agentic_graph import providers as role_provider
from experiments.agentic_graph.providers import LocalModelRoles
from experiments.visual_planner import provider


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setattr(provider, "ROOT", tmp_path)
    path = tmp_path / "source.png"
    Image.new("RGB", (80, 100), "white").save(path)
    source = {
        "canvas": {"width": 80, "height": 100},
        "elements": [
            {
                "id": "title",
                "type": "text",
                "content": "Brand",
                "x": 1,
                "y": 2,
                "width": 60,
                "height": 20,
                "font_size": 14,
            }
        ],
    }
    render = {"path": str(path), "sha256": provider.sha256(path)}
    menu = {
        "schema_version": 1,
        "source_digest": source_digest(source),
        "target_canvas": {"width": 160, "height": 100},
        "plans": [{"plan_id": "keep", "label": "Keep"}],
    }
    menu["menu_digest"] = provider.digest(menu)
    return source, render, menu


def configured(runner):
    return LocalModelRoles(
        {
            "analyzer": ["analyzer"],
            "organizer": ["organizer"],
            "verifier": ["verifier"],
            "timeout_seconds": 60,
        },
        runner=runner,
    )


def response(request):
    prepared = provider.prepare_request(request)
    receipt = {
        "executed": True,
        "request_digest": provider.digest(request),
        "source_digest": prepared["menu"]["source_digest"],
        "source_render_sha256": prepared["images"][0]["sha256"],
        "menu_digest": prepared["menu"]["menu_digest"],
        "target_canvas": prepared["target"],
        "images": deepcopy(prepared["images"]),
        "input_image_count": len(prepared["images"]),
        "source_analysis_digest": (
            provider.digest(request["source_analysis"]) if "source_analysis" in request else None
        ),
    }
    if request["mode"] == "analyze":
        analysis = bind_analysis(
            {
                "observations": [
                    {
                        "kind": "hierarchy",
                        "element_ids": ["title"],
                        "evidence": "Large title above open space",
                    }
                ],
                "abstain": False,
            },
            **provider.analysis_bindings(prepared),
        )
        return {"analysis": analysis, "receipt": receipt}
    return {
        "decision": {
            "schema_version": 1,
            "menu_digest": prepared["menu"]["menu_digest"],
            "plan_id": prepared["plans"][0]["plan_id"],
        },
        "receipt": receipt,
    }


def test_source_only_analysis_and_bound_organizer_feedback(scene):
    source, render, menu = scene
    original = deepcopy(source)
    calls = []

    def run(command, request, timeout):
        calls.append((command, deepcopy(request), timeout))
        return response(request)

    roles = configured(run)
    analyzed = roles.analyze(source, render, timeout_seconds=7)
    assert analyzed["valid"] and analyzed["executed"]
    assert calls[0][0] == ["analyzer"] and 0 < calls[0][2] <= 7
    assert "target_canvas" not in calls[0][1]
    assert "plans" not in calls[0][1] and "repair_feedback" not in calls[0][1]["source_summary"]
    feedback = {"source_digest": menu["source_digest"], "failures": [{"code": "text_too_small"}]}
    organized = roles.organize(source, render, menu, analyzed["analysis"], feedback)
    assert organized["valid"] and organized["executed"]
    assert calls[1][0] == ["organizer"]
    assert calls[1][1]["source_summary"]["repair_feedback"] == feedback
    assert organized["trace"]["receipt"]["request_digest"] == _io._digest(calls[1][1])
    assert organized["trace"]["receipt"]["request_digest"] != calls[1][1]["request_digest"]
    assert source == original


@pytest.mark.parametrize(
    "field",
    [
        "request_digest",
        "source_digest",
        "source_render_sha256",
        "menu_digest",
        "target_canvas",
        "images",
        "input_image_count",
    ],
)
def test_organizer_rejects_every_stale_binding_even_without_analysis(scene, field):
    source, render, menu = scene

    def run(_command, request, _timeout):
        value = response(request)
        value["receipt"][field] = "stale"
        return value

    result = configured(run).organize(source, render, menu)
    assert not result["valid"] and not result["executed"]
    assert result["trace"]["execution_attested"]


def test_verifier_uses_distinct_command_and_all_three_real_images(scene, tmp_path):
    source, render, menu = scene
    menu["plans"] = [
        {"plan_id": "candidate-a", "label": "Alternative1"},
        {"plan_id": "candidate-b", "label": "Alternative2"},
    ]
    menu["menu_digest"] = provider.digest({k: v for k, v in menu.items() if k != "menu_digest"})
    renders = {}
    for index, row in enumerate(menu["plans"]):
        path = tmp_path / f"candidate-{index}.png"
        Image.new("RGB", (160, 100), ["blue", "red"][index]).save(path)
        renders[row["plan_id"]] = {"path": str(path), "sha256": provider.sha256(path)}
    calls = []

    def run(command, request, timeout):
        calls.append((command, request, timeout))
        return response(request)

    result = configured(run).verify(source, render, menu, renders, timeout_seconds=200)
    assert result["valid"] and result["executed"]
    assert calls[0][0] == ["verifier"] and 0 < calls[0][2] <= 60
    assert result["trace"]["receipt"]["input_image_count"] == 3
    assert "repair_feedback" not in calls[0][1]["source_summary"]


@pytest.mark.parametrize("remaining", [0, -1, True, float("nan"), float("inf")])
def test_expired_deadline_never_invokes_a_model(scene, remaining):
    source, render, _menu = scene
    result = configured(lambda *_args: pytest.fail("Provider invoked")).analyze(
        source, render, timeout_seconds=remaining
    )
    assert not result["valid"] and not result["trace"]["invoked"]


def test_unknown_choice_abstention_and_unavailable_provider_are_honest(scene):
    source, render, menu = scene

    def unknown(_command, request, _timeout):
        value = response(request)
        value["decision"]["plan_id"] = "invented"
        return value

    result = configured(unknown).organize(source, render, menu)
    assert result["executed"] and not result["valid"]

    def abstain(_command, request, _timeout):
        value = response(request)
        value["analysis"].update(observations=[], abstain=True)
        return value

    result = configured(abstain).analyze(source, render)
    assert result["executed"] and result["valid"] and result["abstained"]

    def unavailable(*_args):
        raise OSError("Model checkpoint unavailable")

    result = configured(unavailable).analyze(source, render)
    assert not result["executed"] and not result["valid"]
    assert "checkpoint unavailable" in result["trace"]["fallback_reason"]


def test_feedback_and_source_mutation_cannot_escape_request_binding(scene):
    source, render, menu = scene
    roles = configured(lambda *_args: pytest.fail("Provider invoked"))
    for feedback in [{"source_digest": "0" * 64}, {"message": "x" * 2049}]:
        result = roles.organize(source, render, menu, feedback=feedback)
        assert not result["valid"] and not result["trace"]["invoked"]

    def mutate(_command, request, _timeout):
        value = response(request)
        request["target_canvas"]["width"] = 5
        return value

    result = configured(mutate).organize(source, render, menu)
    assert not result["valid"]
    assert "mutated" in result["trace"]["fallback_reason"]


def test_request_preparation_consumes_the_graph_deadline(scene, monkeypatch):
    source, render, _menu = scene
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(role_provider, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    result = configured(lambda *_args: pytest.fail("Provider invoked after deadline")).analyze(
        source, render, timeout_seconds=1
    )
    assert not result["valid"] and not result["trace"]["invoked"]


def test_late_model_response_does_not_become_a_success(scene, monkeypatch):
    source, render, _menu = scene
    ticks = iter([0.0, 0.1, 2.0])
    monkeypatch.setattr(role_provider, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    result = configured(lambda _command, request, _timeout: response(request)).analyze(
        source, render, timeout_seconds=1
    )
    assert result["executed"] and not result["valid"]
    assert "time budget" in result["trace"]["fallback_reason"]


def test_changed_source_pixels_during_inference_invalidate_the_receipt(scene):
    source, render, _menu = scene

    def mutate_pixels(_command, request, _timeout):
        value = response(request)
        Image.new("RGB", (80, 100), "blue").save(render["path"])
        return value

    result = configured(mutate_pixels).analyze(source, render)
    assert not result["valid"] and not result["executed"]
    assert "bound identity" in result["trace"]["fallback_reason"]


def test_configuration_identity_pins_json_files_without_exposing_content(tmp_path):
    secret = "do-not-include-config-content"
    config = tmp_path / "provider.json"
    config.write_text('{"adapter_dir":"adapter-a","private_note":"' + secret + '"}')
    roles = LocalModelRoles(
        {
            "analyzer": ["python", "-m", "local-provider", "--config", str(config)],
            "organizer": ["python", "--config=" + str(config)],
            "verifier": ["verifier"],
        },
        runner=lambda *_args: pytest.fail("Identity inspection invoked inference"),
    )
    before = roles.configuration_identity()
    assert secret not in _io._canonical(before)
    assert before["roles"]["analyzer"]["configuration_files"][0]["sha256"] == provider.sha256(
        config
    )
    assert before["roles"]["organizer"]["configuration_files"][0]["path"] == str(config.resolve())
    assert before["provider_code_sha256"]["experiments/agentic_graph/providers.py"]
    config.write_text('{"adapter_dir":"adapter-b"}')
    after = roles.configuration_identity()
    assert before["digest"] != after["digest"]
    assert (
        before["roles"]["analyzer"]["command_sha256"]
        == after["roles"]["analyzer"]["command_sha256"]
    )
    roles.timeout_seconds = 10
    assert roles.configuration_identity()["digest"] != after["digest"]


def test_configuration_identity_rejects_missing_or_oversized_config(tmp_path):
    path = tmp_path / "provider.json"
    roles = LocalModelRoles(
        {
            "analyzer": ["provider", "--config", str(path)],
            "organizer": ["organizer"],
            "verifier": ["verifier"],
        }
    )
    with pytest.raises(ValueError, match="missing or oversized"):
        roles.configuration_identity()
    with path.open("wb") as stream:
        stream.truncate(role_provider.MAX_IDENTITY_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="missing or oversized"):
        roles.configuration_identity()


def test_configuration_identity_detects_provider_code_changes(tmp_path, monkeypatch):
    code = tmp_path / "provider.py"
    code.write_text("version = 1\n")
    monkeypatch.setattr(role_provider, "ROOT", tmp_path)
    monkeypatch.setattr(role_provider, "_PROVIDER_CODE", ("provider.py",))
    roles = configured(lambda *_args: pytest.fail("Identity inspection invoked inference"))
    before = roles.configuration_identity()
    code.write_text("version = 2\n")
    assert roles.configuration_identity()["digest"] != before["digest"]


def test_configuration_identity_distinguishes_real_runtime_from_injected_provider():
    config = {"analyzer": ["analyzer"], "organizer": ["organizer"], "verifier": ["verifier"]}
    live = LocalModelRoles(config)
    injected = LocalModelRoles(config, runner=lambda *_args: pytest.fail("Provider invoked"))
    live_identity, injected_identity = (
        live.configuration_identity(),
        injected.configuration_identity(),
    )
    assert live.execution_scope == live_identity["execution_scope"] == "supervised_offline_runtime"
    assert (
        injected.execution_scope == injected_identity["execution_scope"] == "injected_provider_test"
    )
    assert live_identity["digest"] != injected_identity["digest"]
