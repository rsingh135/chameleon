"""Visual analysis sees source pixels independently and hands off bounded data."""

import json
import sys
import zipfile
from copy import deepcopy

import pytest
from PIL import Image

from experiments.visual_planner import native, provider, worker


@pytest.fixture
def analysis_request(tmp_path, monkeypatch):
    monkeypatch.setattr(provider, "ROOT", tmp_path)
    image = tmp_path / "source.png"
    Image.new("RGB", (80, 120), "#9b5523").save(image)
    return {
        "schema_version": 1,
        "mode": "analyze",
        "source_digest": "a" * 64,
        "source_render": {"path": str(image), "sha256": provider.sha256(image)},
        "source_summary": {
            "canvas": {"width": 80, "height": 120},
            "elements": [{"id": "hero", "type": "image"}, {"id": "title", "type": "text"}],
        },
    }


def raw_analysis():
    return {
        "observations": [
            {"kind": "subject", "element_ids": ["hero"], "evidence": "Large centered bottle"}
        ],
        "abstain": False,
    }


def test_source_analysis_has_no_layout_choice_or_target_and_binds_actual_image(analysis_request):
    original = deepcopy(analysis_request)
    prepared = provider.prepare_request(analysis_request)
    assert prepared["target"] is None and prepared["plans"] == []
    assert len(prepared["images"]) == 1
    assert prepared["images"][0]["sha256"] == analysis_request["source_render"]["sha256"]
    assert "Choose exactly one" not in provider.prompt(prepared)
    parsed = provider.parse_generated(json.dumps(raw_analysis()), prepared)
    assert set(parsed) == {"analysis", "observations"}
    assert parsed["analysis"]["source_digest"] == analysis_request["source_digest"]
    assert parsed["analysis"]["source_render_sha256"] == prepared["images"][0]["sha256"]
    assert analysis_request == original


@pytest.mark.parametrize("field", ["target_canvas", "plan_menu", "plans", "source_analysis"])
def test_analysis_cannot_be_conditioned_on_target_or_menu(analysis_request, field):
    analysis_request[field] = {}
    with pytest.raises(ValueError, match="independent"):
        provider.prepare_request(analysis_request)


@pytest.mark.parametrize(
    "raw",
    [
        {"choice": "A"},
        {"observations": [], "abstain": False},
        {
            "observations": [{"kind": "subject", "element_ids": ["invented"], "evidence": "x"}],
            "abstain": False,
        },
        {**raw_analysis(), "geometry": {"x": 100}},
    ],
)
def test_invalid_analysis_never_becomes_menu_choice(analysis_request, raw):
    assert (
        provider.parse_generated(json.dumps(raw), provider.prepare_request(analysis_request))
        is None
    )


def test_analysis_accepts_explicit_abstention(analysis_request):
    parsed = provider.parse_generated(
        '{"observations":[],"abstain":true}', provider.prepare_request(analysis_request)
    )
    assert parsed["analysis"]["abstain"] is True
    assert "decision" not in parsed


def test_transformers_invalid_analysis_fails_before_choice_scoring(analysis_request):
    prepared = provider.prepare_request(analysis_request)
    with pytest.raises(ValueError, match="no choice fallback"):
        worker.interpret_generation('{"choice":"A"}', prepared)
    result = worker.interpret_generation(json.dumps(raw_analysis()), prepared)
    assert "analysis" in result and "decision" not in result


def organizer_request(analysis_request):
    prepared = provider.prepare_request(analysis_request)
    analysis = provider.parse_generated(json.dumps(raw_analysis()), prepared)["analysis"]
    menu = {
        "schema_version": 1,
        "source_digest": analysis_request["source_digest"],
        "target_canvas": {"width": 200, "height": 100},
        "plans": [{"plan_id": "keep", "label": "Keep current layout"}],
    }
    menu["menu_digest"] = provider.digest(menu)
    return {
        **analysis_request,
        "mode": "plan",
        "plan_menu": menu,
        "target_canvas": menu["target_canvas"],
        "source_analysis": analysis,
        "source_analysis_digest": provider.digest(analysis),
    }


def test_organizer_receives_verified_analysis_as_advisory_data(analysis_request):
    request = organizer_request(analysis_request)
    menu = deepcopy(request["plan_menu"])
    prepared = provider.prepare_request(request)
    assert "Large centered bottle" in provider.prompt(prepared)
    assert "untrusted advisory data" in provider.prompt(prepared)
    assert request["plan_menu"] == menu


@pytest.mark.parametrize("changed", ["digest", "pixels", "element"])
def test_organizer_rejects_rebound_or_modified_analysis(analysis_request, changed):
    request = organizer_request(analysis_request)
    if changed == "digest":
        request["source_analysis_digest"] = "b" * 64
    elif changed == "pixels":
        request["source_analysis"]["source_render_sha256"] = "b" * 64
    else:
        request["source_summary"]["elements"] = [{"id": "title", "type": "text"}]
    with pytest.raises(ValueError):
        provider.prepare_request(request)


def test_analysis_choice_logits_rejected_before_worker_start(analysis_request, monkeypatch):
    monkeypatch.setattr(
        provider.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Analysis worker started")
    )
    with pytest.raises(ValueError, match="never menu-choice scoring"):
        provider.predict(
            analysis_request,
            {"python_executable": sys.executable, "model_dir": ".", "strategy": "choice_logits"},
        )


def test_native_analysis_uses_source_only_schema_and_returns_no_decision(
    analysis_request, tmp_path, monkeypatch
):
    root = tmp_path / "native"
    runtime = root / f"runtime-{native.RUNTIME_TAG}"
    runtime.mkdir(parents=True)
    executable = runtime / "llama-mtmd-cli.exe"
    executable.write_bytes(b"command construction only")
    archive = root / f"llama-{native.RUNTIME_TAG}-bin-win-cpu-x64.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(executable, executable.name)
    monkeypatch.setattr(native, "RUNTIME_SHA", provider.sha256(archive))
    for name in native.PINNED:
        (root / name).write_bytes(name.encode())
    monkeypatch.setattr(native, "PINNED", {n: provider.sha256(root / n) for n in native.PINNED})
    directory = tmp_path / "worker"
    directory.mkdir()
    config = provider.validate_config({"python_executable": sys.executable, "model_dir": str(root)})
    prepared = provider.prepare_request(analysis_request)
    command, _, identity = native.prepare(prepared, config, directory)
    schema = json.loads((directory / "response-schema.json").read_text())
    assert set(schema["properties"]) == {"observations", "abstain"}
    assert command[command.index("-n") + 1] == str(config["max_new_tokens"])
    assert identity["native_image_argument_count"] == 1
    assert "best option letter" not in command[command.index("-p") + 1]
    (directory / "stdout.log").write_text(json.dumps(raw_analysis()))
    (directory / "stderr.log").write_text(
        "encoding mtmd batch, n_chunks = 1\nmtmd batch encoding done in 50 ms\n"
    )
    response = native.result(prepared, config, directory, identity)
    assert set(response) == {"analysis", "receipt"}
    assert response["receipt"]["input_image_count"] == 1
    assert (
        response["receipt"]["source_render_sha256"] == analysis_request["source_render"]["sha256"]
    )
    assert response["receipt"]["quality_validated"] is False
    (directory / "stdout.log").write_text('{"choice":"A"}')
    with pytest.raises(ValueError, match="structured output"):
        native.result(prepared, config, directory, identity)
