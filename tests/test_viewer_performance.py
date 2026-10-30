"""Exercise the real viewer orchestration with bounded cache reuse and isolation."""

import base64
import http.client
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin

from engine.resources import ResourceContext
from viewer import benchmark, server
from viewer.resize_cache import ResourceBindings, ResourceChangedError, SerializedCache


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch):
    # Each fixture owns its processor/resource lifetime. Never let different
    # fake processors or a shared checkout edit accidentally reuse prior data.
    monkeypatch.setattr(server, "_RESOURCE_BINDINGS", ResourceBindings(), raising=False)
    monkeypatch.setattr(server, "_SOURCE_PNG_CACHE", SerializedCache(kind="source"), raising=False)
    monkeypatch.setattr(
        server, "_RESIZE_RESULT_CACHE", SerializedCache(kind="result"), raising=False
    )
    monkeypatch.setattr(server, "_execution_identity", lambda: "frozen-test-runtime", raising=False)
    monkeypatch.setattr(server, "_visual_model_enabled", lambda: False, raising=False)


@pytest.fixture
def resources(tmp_path):
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    return ResourceContext(assets, fonts)


@pytest.fixture
def payload():
    return {
        "source": {
            "canvas": {"width": 100, "height": 80},
            "elements": [
                {
                    "id": "panel",
                    "type": "shape",
                    "shape_type": "rectangle",
                    "x": 5,
                    "y": 5,
                    "width": 90,
                    "height": 70,
                    "fill_color": "#236789",
                }
            ],
        },
        "target_canvas": {"width": 50, "height": 40},
    }


def _image(url):
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(calls=Counter(), clock=0.0, fail_output=False, block=None)
    monkeypatch.setattr(server, "time", SimpleNamespace(perf_counter=lambda: state.clock))

    def resize(source, target, **_kwargs):
        state.calls["resize"] += 1
        state.clock += 1.0
        if state.block:
            entered, release = state.block
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release processor")
        design = deepcopy(source)
        scale = target["width"] / source["canvas"]["width"]
        design["canvas"] = deepcopy(target)
        for element in design["elements"]:
            for key in ("x", "y", "width", "height"):
                element[key] *= scale
        return {
            "design": design,
            "state": {"name": "test-uniform", "cluster_axes": {}, "sizing_exceptions": []},
        }

    def counted(name, value):
        def call(*_args, **_kwargs):
            state.calls[name] += 1
            return deepcopy(value)

        return call

    def render(design, attribution=None):
        size = design["canvas"]["width"], design["canvas"]["height"]
        kind = "source_png" if size == (100, 80) else "output_png"
        state.calls[kind] += 1
        state.clock += 0.1
        if kind == "output_png" and state.fail_output:
            state.fail_output = False
            raise OSError("injected output render failure")
        metadata = PngImagePlugin.PngInfo()
        for field, name in (("author", "Author"), ("license_id", "License")):
            if attribution and field in attribution:
                metadata.add_text(name, attribution[field])
        output = BytesIO()
        Image.new("RGB", size, (35, 103, 137)).save(output, format="PNG", pnginfo=metadata)
        url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        return url, [f"{kind} warning"]

    monkeypatch.setattr(server, "resize_with_state", resize)
    monkeypatch.setattr(server, "build_context", counted("context", {}))
    monkeypatch.setattr(server, "render_plate", counted("plate", None))
    monkeypatch.setattr(
        server, "score", counted("score", {"violations": [], "metrics": {}, "total": 1.0})
    )
    monkeypatch.setattr(server, "build_relationship_graph", counted("graph", {}))
    for name in (
        "analyze_style",
        "analyze_relationships",
        "analyze_proportions",
        "analyze_focal",
        "analyze_rules",
        "analyze_reading_order",
        "conservative_occlusion",
    ):
        monkeypatch.setattr(server, name, counted(name, {}))
    monkeypatch.setattr(server, "_render_png", render)
    return state


def test_different_targets_share_one_original_native_render(payload, resources, backend):
    first = server.perform_resize(payload, resources=resources)
    second_payload = {**payload, "target_canvas": {"width": 200, "height": 160}}
    second = server.perform_resize(second_payload, resources=resources)
    assert backend.calls["resize"] == 2
    assert backend.calls["source_png"] == 1
    assert backend.calls["output_png"] == 2
    assert first["source_png"] == second["source_png"]
    assert _image(first["output_png"]).size == (50, 40)
    assert _image(second["output_png"]).size == (200, 160)


def test_exact_replay_reuses_completed_work_with_fresh_values_and_elapsed(
    payload, resources, backend
):
    first = server.perform_resize(payload, resources=resources)
    counts = backend.calls.copy()
    first["design"]["elements"][0]["x"] = -999
    first["report"]["metrics"]["caller"] = "changed"
    first["warnings"].append("caller warning")
    second = server.perform_resize(payload, resources=resources)
    assert backend.calls == counts
    assert second["design"]["elements"][0]["x"] == 2.5
    assert second["report"]["metrics"] == {}
    assert "caller warning" not in second["warnings"]
    assert first["elapsed_ms"] > second["elapsed_ms"]
    assert second["elapsed_ms"] == 0.0


def test_attribution_changes_do_not_replay_another_credit_or_license(payload, resources, backend):
    a = {**payload, "attribution": {"author": "Author A", "license_id": "License A"}}
    b = {**payload, "attribution": {"author": "Author B", "license_id": "License B"}}
    first = server.perform_resize(a, resources=resources)
    second = server.perform_resize(b, resources=resources)
    repeated = server.perform_resize(a, resources=resources)
    assert backend.calls["resize"] == backend.calls["source_png"] == 2
    for result, author, license_id in (
        (first, "Author A", "License A"),
        (second, "Author B", "License B"),
        (repeated, "Author A", "License A"),
    ):
        for key in ("source_png", "output_png"):
            with _image(result[key]) as image:
                assert image.info["Author"] == author and image.info["License"] == license_id


def test_mutated_same_path_resource_refuses_old_cached_result(payload, resources, backend):
    import os

    path = resources.assets_dir / "owned.png"
    path.write_bytes(b"old-resource")
    payload["source"]["elements"] = [
        {
            "id": "owned",
            "type": "image",
            "asset_id": "owned.png",
            "x": 0,
            "y": 0,
            "width": 100,
            "height": 80,
        }
    ]
    server.perform_resize(payload, resources=resources)
    before = path.stat()
    path.write_bytes(b"new-resource")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    with pytest.raises(ResourceChangedError, match="Restart"):
        server.perform_resize(payload, resources=resources)
    assert backend.calls["resize"] == 1


def test_failed_output_never_becomes_a_partial_success_and_retry_computes(
    payload, resources, backend
):
    backend.fail_output = True
    with pytest.raises(OSError, match="output render failure"):
        server.perform_resize(payload, resources=resources)
    result = server.perform_resize(payload, resources=resources)
    assert backend.calls["resize"] == 2
    assert result["report"]["violations"] == []
    assert _image(result["output_png"]).size == (50, 40)
    server.perform_resize(payload, resources=resources)
    assert backend.calls["resize"] == 2


def test_legacy_missing_image_keeps_warnings_without_reusing_incomplete_results(
    payload, resources, backend
):
    payload["source"]["elements"] = [
        {
            "id": "missing-original",
            "type": "image",
            "asset_id": "missing.png",
            "x": 0,
            "y": 0,
            "width": 100,
            "height": 80,
        }
    ]
    first = server.perform_resize(payload, resources=resources)
    second = server.perform_resize(payload, resources=resources)
    assert first["warnings"] == second["warnings"]
    assert backend.calls["resize"] == backend.calls["source_png"] == 2
    assert server._SOURCE_PNG_CACHE.info()["entries"] == 0
    assert server._RESIZE_RESULT_CACHE.info()["entries"] == 0
    (resources.assets_dir / "missing.png").write_bytes(b"new-image")
    with pytest.raises(ResourceChangedError, match="Restart"):
        server.perform_resize(payload, resources=resources)
    assert backend.calls["resize"] == 2


def test_concurrent_identical_requests_compute_once_under_existing_engine_lock(
    payload, resources, backend
):
    entered, release = threading.Event(), threading.Event()
    backend.block = entered, release
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(server.perform_resize, deepcopy(payload), resources=resources)
            for _ in range(6)
        ]
        try:
            assert entered.wait(timeout=3)
        finally:
            release.set()
        results = [future.result(timeout=8) for future in futures]
    assert backend.calls["resize"] == 1
    assert backend.calls["source_png"] == backend.calls["output_png"] == 1
    results[0]["design"]["elements"][0]["x"] = -999
    assert all(result["design"]["elements"][0]["x"] == 2.5 for result in results[1:])


def test_model_enabled_requests_bypass_full_result_replay_but_reuse_source_render(
    payload, resources, backend, monkeypatch
):
    monkeypatch.setattr(server, "_visual_model_enabled", lambda: True)

    def configured(source, target, **kwargs):
        backend.calls["configured"] += 1
        return server.resize_with_state(source, target, **kwargs)

    monkeypatch.setattr(server, "_configured_resize", configured, raising=False)
    first = server.perform_resize(payload, resources=resources)
    second = server.perform_resize(payload, resources=resources)
    assert backend.calls["configured"] == 2
    assert backend.calls["resize"] == 2
    assert backend.calls["source_png"] == 1 and backend.calls["output_png"] == 2
    assert first["source_png"] == second["source_png"]
    assert first["cache"]["result_hit"] is second["cache"]["result_hit"] is False


def test_a_warm_cached_request_returns_while_another_target_owns_the_native_engine(
    payload, resources, backend
):
    warm = server.perform_resize(payload, resources=resources)
    entered, release = threading.Event(), threading.Event()
    backend.block = entered, release
    other_target = {**deepcopy(payload), "target_canvas": {"width": 200, "height": 160}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        native_job = pool.submit(server.perform_resize, other_target, resources=resources)
        try:
            assert entered.wait(timeout=3)
            cached_job = pool.submit(server.perform_resize, deepcopy(payload), resources=resources)
            cached = cached_job.result(timeout=2)
            assert not release.is_set() and not native_job.done()
            assert cached["cache"]["result_hit"] is True
            assert cached["design"] == warm["design"]
        finally:
            release.set()
        completed = native_job.result(timeout=8)
    assert completed["design"]["canvas"] == other_target["target_canvas"]
    assert backend.calls["resize"] == 2
    assert backend.calls["source_png"] == 1 and backend.calls["output_png"] == 2


def test_target_mutating_processor_cannot_poison_the_original_request_cache(
    payload, resources, backend, monkeypatch
):
    original_processor = server.resize_with_state
    original_request = deepcopy(payload)

    def bad_processor(source, target):
        target.update(width=60, height=48)
        return original_processor(source, target)

    monkeypatch.setattr(server, "resize_with_state", bad_processor)
    with pytest.raises(RuntimeError, match="target canvas"):
        server.perform_resize(payload, resources=resources)
    assert payload == original_request
    assert backend.calls["context"] == backend.calls["source_png"] == 0
    assert server._SOURCE_PNG_CACHE.info()["entries"] == 0
    assert server._RESIZE_RESULT_CACHE.info()["entries"] == 0

    monkeypatch.setattr(server, "resize_with_state", original_processor)
    correct = server.perform_resize(payload, resources=resources)
    repeated = server.perform_resize(payload, resources=resources)
    assert correct["design"]["canvas"] == original_request["target_canvas"]
    assert repeated["design"] == correct["design"]
    assert backend.calls["resize"] == 2  # One rejected attempt, then one completed computation.


@pytest.mark.parametrize("flip_during", ["preparation", "processing"])
def test_http_method_change_during_preparation_or_processing_never_returns_stale_success(
    payload, resources, monkeypatch, flip_during
):
    identity = {"digest": "a" * 64}
    calls = Counter()
    monkeypatch.setattr(benchmark, "method_receipt", lambda: dict(identity))

    def prepare(_payload):
        calls["prepare"] += 1
        if flip_during == "preparation":
            identity["digest"] = "b" * 64
        return SimpleNamespace(resources=resources, receipt={"snapshot_id": "test-only"})

    def process(_payload, **_kwargs):
        calls["process"] += 1
        if flip_during == "processing":
            identity["digest"] = "b" * 64
        # The HTTP guard must reject before publishing any processing success.
        return {"design": deepcopy(payload["source"]), "output_png": "test-only-output"}

    monkeypatch.setattr(server, "prepare_document", prepare)
    monkeypatch.setattr(server, "perform_resize", process)
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=10)
    try:
        connection.request(
            "POST",
            "/api/document/resize",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 409
        assert "Restart" in body["error"] and "design" not in body
        assert calls["prepare"] == 1
        assert calls["process"] == (0 if flip_during == "preparation" else 1)
    finally:
        connection.close()
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=3)


def test_real_native_replay_keeps_pixels_isolated_between_same_named_asset_packages(tmp_path):
    source = {
        "canvas": {"width": 20, "height": 20},
        "elements": [
            {
                "id": "picture",
                "type": "image",
                "asset_id": "same.png",
                "fit": "contain",
                "x": 0,
                "y": 0,
                "width": 20,
                "height": 20,
            }
        ],
    }
    contexts = []
    colors = ((240, 20, 40, 255), (10, 60, 230, 255))
    for name, color in zip(("red", "blue"), colors, strict=True):
        root = tmp_path / name
        assets, fonts = root / "assets", root / "fonts"
        assets.mkdir(parents=True)
        fonts.mkdir()
        # Image-only native bundles explicitly declare an empty font manifest.
        (fonts / "font-manifest.json").write_text("[]", encoding="utf-8")
        Image.new("RGBA", (20, 20), color).save(assets / "same.png")
        contexts.append(ResourceContext(assets, fonts))
    request = {"source": source, "target_canvas": {"width": 10, "height": 10}}
    red = server.perform_resize(request, resources=contexts[0])
    blue = server.perform_resize(request, resources=contexts[1])
    replay = server.perform_resize(request, resources=contexts[0])
    for result, color in ((red, colors[0]), (blue, colors[1]), (replay, colors[0])):
        with _image(result["output_png"]) as image:
            assert image.size == (10, 10)
            assert image.convert("RGBA").getpixel((5, 5)) == color
        assert result["report"]["violations"] == []
    assert red["output_png"] != blue["output_png"]
    assert replay["output_png"] == red["output_png"]
