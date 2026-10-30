"""Batch identity, equal method settings, isolated failures and real native renders."""

import base64
import http.client
import json
import subprocess
import sys
import textwrap
import threading
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from viewer import benchmark, document_bridge, server


@pytest.fixture
def source():
    return {
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "panel",
                "type": "shape",
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 80,
                "shape_type": "rectangle",
                "fill_color": "#ffffff",
            }
        ],
    }


@pytest.fixture
def batch(source):
    return {
        "scenes": [
            {"id": "one", "family": "cards", "payload": source},
            {
                "id": "two",
                "family": "posters",
                "payload": {"source": source, "target_canvas": {"width": 1, "height": 2}},
            },
        ],
        "targets": [{"width": 120, "height": 80}, {"width": 80, "height": 120}],
    }


def _fake_result(request):
    return {
        "design": {**request["source"], "canvas": request["target_canvas"]},
        "source_png": "native-source",
        "output_png": "native-result",
        "report": {"violations": []},
        "elapsed_ms": 1,
    }


@pytest.fixture
def benchmark_server(monkeypatch):
    # Freeze the receipt for HTTP tests so other agents editing checkout files
    # cannot invalidate these isolated test workers halfway through a request.
    receipt = benchmark.method_receipt()
    monkeypatch.setattr(benchmark, "method_receipt", lambda: deepcopy(receipt))
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance, receipt
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


def _http(instance, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=40)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


def _post(instance, payload, headers=None):
    status, kind, body = _http(
        instance,
        "POST",
        "/api/benchmark/run",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json", **(headers or {})},
    )
    assert "application/json" in kind
    return status, json.loads(body)


def test_same_frozen_targets_for_every_scene_and_inputs_unchanged(batch, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "resource_hashes", lambda *args: {"fonts": {}, "assets": {}})

    def resize(request, **kwargs):
        calls.append(deepcopy(request))
        result = _fake_result(request)
        request["source"]["elements"][0]["x"] = -999
        return result

    monkeypatch.setattr(server, "perform_resize", resize)
    before = deepcopy(batch)
    report = benchmark.run_benchmark(batch)
    assert batch == before
    assert [call["target_canvas"] for call in calls] == batch["targets"] * 2
    assert all(call["source"]["elements"][0]["x"] == 10 for call in calls)
    assert len({row["source_sha256"] for row in report["results"]}) == 1
    assert report["summary"]["contract_passed"] == 4
    assert report["summary"]["visual_review"] == "not_reviewed"
    assert report["summary"]["by_family"]["cards"]["jobs"] == 2
    assert report["method"]["analyzer"] == report["method"]["critic"] == {"enabled": False}
    assert report["comparable"] is True


def test_invalid_later_scene_is_rejected_before_first_resize(batch, monkeypatch):
    batch["scenes"][1]["payload"]["source"]["elements"] = []
    calls = []
    monkeypatch.setattr(server, "perform_resize", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="source.elements"):
        benchmark.run_benchmark(batch)
    assert calls == []


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p.update(method="some-model"), "Only deterministic"),
        (lambda p: p.update(expected_method_digest="x" * 64), "SHA-256"),
        (lambda p: p["scenes"][1].update(id="one"), "unique"),
        (lambda p: p.update(targets=p["targets"] * 3), "target canvases"),
        (lambda p: p.update(scenes=p["scenes"] * 9), "scenes"),
        (lambda p: p["targets"][0].update(width=True), "integer"),
    ],
)
def test_malformed_work_is_bounded_before_render(batch, change, match):
    change(batch)
    with pytest.raises(ValueError, match=match):
        benchmark.validate_benchmark(batch)


def test_changed_method_rejects_next_browser_job_before_preparation(batch, monkeypatch):
    batch["expected_method_digest"] = "0" * 64
    monkeypatch.setattr(benchmark, "method_receipt", lambda: {"digest": "1" * 64})
    monkeypatch.setattr(
        benchmark, "resource_hashes", lambda *args: pytest.fail("Resources must not be read")
    )
    with pytest.raises(ValueError, match="method changed"):
        benchmark.run_benchmark(batch)


def test_changed_code_mid_run_marks_whole_comparison_invalid(batch, monkeypatch):
    receipts = iter([{"digest": "1" * 64}, {"digest": "2" * 64}])
    monkeypatch.setattr(benchmark, "method_receipt", lambda: next(receipts))
    monkeypatch.setattr(benchmark, "resource_hashes", lambda *args: {})
    monkeypatch.setattr(server, "perform_resize", lambda request, **kwargs: _fake_result(request))
    report = benchmark.run_benchmark(batch)
    assert report["comparable"] is False
    assert "rerun all jobs" in report["comparison_warning"]


def test_renderer_failure_does_not_hide_other_scene_outcomes(batch, monkeypatch):
    monkeypatch.setattr(benchmark, "resource_hashes", lambda *args: {})
    calls = 0

    def resize(request, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("native decode failed")
        result = _fake_result(request)
        if calls == 3:
            result["report"]["violations"] = ["outside frame"]
        return result

    monkeypatch.setattr(server, "perform_resize", resize)
    report = benchmark.run_benchmark(batch)
    assert report["summary"] | {} == {
        "jobs": 4,
        "succeeded": 3,
        "failed": 1,
        "contract_passed": 2,
        "visual_review": "not_reviewed",
        "by_family": report["summary"]["by_family"],
    }
    assert report["results"][1]["error"] == "native decode failed"
    assert report["results"][2]["contract_passed"] is False


def test_mutated_resource_bytes_reject_result(batch, monkeypatch):
    hashes = iter([{"asset": "before"}, {"asset": "after"}] * 4)
    monkeypatch.setattr(benchmark, "resource_hashes", lambda *args: next(hashes))
    monkeypatch.setattr(server, "perform_resize", lambda request, **kwargs: _fake_result(request))
    # One source and one target gives one pre/post pair.
    batch["scenes"] = batch["scenes"][:1]
    batch["targets"] = batch["targets"][:1]
    result = benchmark.run_benchmark(batch)["results"][0]
    assert result["status"] == "error"
    assert result["result"] is None
    assert "bytes changed" in result["error"]


def _bundle(color, snapshot):
    image = BytesIO()
    Image.new("RGB", (32, 48), color).save(image, format="PNG")
    return {
        "source": {
            "canvas": {"width": 64, "height": 96},
            "elements": [
                {
                    "id": "photo",
                    "type": "image",
                    "x": 0,
                    "y": 0,
                    "width": 64,
                    "height": 96,
                    "asset_id": "photo.png",
                    "fit": "contain",
                }
            ],
        },
        "document": {"snapshot_id": snapshot, "page_id": "page", "artboard_id": "artboard"},
        "assets": [
            {
                "asset_id": "photo.png",
                "mime_type": "image/png",
                "data_base64": base64.b64encode(image.getvalue()).decode(),
            }
        ],
        "fonts": [],
    }


def test_two_real_native_image_scenes_keep_independent_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(document_bridge, "DOCUMENT_ROOT", tmp_path / "documents")
    payload = {
        "scenes": [
            {"id": "red", "payload": _bundle("red", "red")},
            {"id": "blue", "input_kind": "flat-image", "payload": _bundle("blue", "blue")},
        ],
        "targets": [{"width": 80, "height": 60}],
    }
    before = deepcopy(payload)
    report = benchmark.run_benchmark(payload)
    assert payload == before
    assert report["summary"]["succeeded"] == 2
    rows = report["results"]
    assert [row["input_kind"] for row in rows] == ["editable-design", "flat-image"]
    assert rows[0]["resources_sha256"] != rows[1]["resources_sha256"]
    for row, expected in zip(rows, ((255, 0, 0), (0, 0, 255)), strict=True):
        result = row["result"]
        assert result["design"]["canvas"] == {"width": 80, "height": 60}
        assert [element["id"] for element in result["design"]["elements"]] == ["photo"]
        assert result["design"]["elements"][0]["asset_id"] == "photo.png"
        for key, dimensions in (("source_png", (64, 96)), ("output_png", (80, 60))):
            with Image.open(BytesIO(base64.b64decode(result[key].split(",")[1]))) as image:
                assert image.size == dimensions
                center = (dimensions[0] // 2, dimensions[1] // 2)
                assert image.convert("RGB").getpixel(center) == expected


def test_http_method_and_real_native_resize(benchmark_server, source):
    instance, receipt = benchmark_server
    status, kind, body = _http(instance, "GET", "/api/benchmark/method")
    assert status == 200
    assert "application/json" in kind
    assert json.loads(body) == receipt
    status, report = _post(
        instance,
        {
            "method": benchmark.METHOD_ID,
            "expected_method_digest": receipt["digest"],
            "scenes": [{"id": "native-shape", "payload": source}],
            "targets": [{"width": 120, "height": 80}],
        },
        {"Origin": f"http://127.0.0.1:{instance.server_port}"},
    )
    assert status == 200
    assert report["method"] == receipt
    assert report["comparable"] is True
    row = report["results"][0]
    assert row["status"] == "ok"
    assert row["source_sha256"] == benchmark._digest(source)
    assert row["contract_passed"] is True
    assert [element["id"] for element in row["result"]["design"]["elements"]] == ["panel"]
    for key, dimensions in (("source_png", (100, 100)), ("output_png", (120, 80))):
        png = base64.b64decode(row["result"][key].split(",")[1])
        with Image.open(BytesIO(png)) as image:
            assert image.size == dimensions


@pytest.mark.parametrize("route", ["/api/benchmark/method", "/api/benchmark/run"])
def test_http_cross_origin_benchmark_requests_are_rejected(benchmark_server, batch, route):
    instance, _ = benchmark_server
    headers = {"Origin": "https://unrelated.example", "Content-Type": "application/json"}
    status, kind, body = _http(
        instance,
        "GET" if route.endswith("method") else "POST",
        route,
        json.dumps(batch).encode() if route.endswith("run") else None,
        headers,
    )
    assert status == 403
    assert "application/json" in kind
    assert "same-origin" in json.loads(body)["error"]


@pytest.mark.parametrize("raw", [b"{broken", b"[]", b"null", b""])
def test_http_bad_benchmark_body_is_json_error(benchmark_server, raw):
    instance, _ = benchmark_server
    status, kind, body = _http(
        instance,
        "POST",
        "/api/benchmark/run",
        raw,
        {"Content-Type": "application/json"},
    )
    assert status == 400
    assert "application/json" in kind
    assert json.loads(body)["error"]


def test_http_unsupported_model_and_media_are_rejected(benchmark_server, batch, monkeypatch):
    instance, _ = benchmark_server
    calls = []
    monkeypatch.setattr(server, "perform_resize", lambda *args, **kwargs: calls.append(args))
    batch["method"] = "vision-experimental"
    status, report = _post(instance, batch)
    assert status == 400
    assert "Only deterministic-current" in report["error"]
    # Media type is rejected before reading a body. Sending none also avoids
    # Windows resetting a connection when it closes with unread request bytes.
    status, kind, body = _http(
        instance, "POST", "/api/benchmark/run", headers={"Content-Type": "text/plain"}
    )
    assert status == 415
    assert "application/json" in kind
    report = json.loads(body)
    assert "application/json" in report["error"]
    assert calls == []


@pytest.mark.parametrize("route", ["/api/benchmark/method", "/api/benchmark/run"])
def test_http_changed_files_reject_boot_receipt_before_work(
    benchmark_server, batch, monkeypatch, route
):
    instance, receipt = benchmark_server
    monkeypatch.setattr(benchmark, "method_receipt", lambda: {**receipt, "digest": "0" * 64})
    monkeypatch.setattr(
        benchmark, "run_benchmark", lambda payload: pytest.fail("Changed engine must not run")
    )
    if route.endswith("method"):
        status, _, body = _http(instance, "GET", route)
        report = json.loads(body)
    else:
        status, report = _post(instance, batch)
    assert status == 409
    assert "Restart the server" in report["error"]


@pytest.mark.parametrize("route", ["/api/benchmark/method", "/api/benchmark/run"])
def test_http_unreadable_method_receipt_returns_json_error(
    benchmark_server, batch, monkeypatch, route
):
    instance, _ = benchmark_server

    def fail_receipt():
        raise PermissionError("engine checkout unavailable")

    monkeypatch.setattr(benchmark, "method_receipt", fail_receipt)
    if route.endswith("method"):
        status, _, body = _http(instance, "GET", route)
        report = json.loads(body)
    else:
        status, report = _post(instance, batch)
    assert status == 500
    assert report == {"error": "Benchmark failed: engine checkout unavailable"}


def test_http_unexpected_benchmark_failure_returns_json_and_retry_works(
    benchmark_server, batch, monkeypatch
):
    instance, _ = benchmark_server

    def fail_run(payload):
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(benchmark, "run_benchmark", fail_run)
    status, report = _post(instance, batch)
    assert status == 500
    assert report == {"error": "Benchmark failed: worker unavailable"}
    monkeypatch.setattr(benchmark, "run_benchmark", lambda payload: {"retried": True})
    status, report = _post(instance, batch)
    assert status == 200
    assert report == {"retried": True}


def test_module_cli_and_batch_share_one_server_module_and_engine_lock():
    # Reproduce the real -m entry point in a fresh interpreter, with no socket
    # binding. Importing viewer.server from inside the running CLI previously
    # created a second handler/lock and let Studio and batch render concurrently.
    script = textwrap.dedent(
        """
        import http.server
        import importlib
        import runpy
        import sys

        closed = []

        class FakeServer:
            def __init__(self, address, handler):
                self.server_port = address[1]
                self.handler = handler

            def serve_forever(self):
                canonical = importlib.import_module('viewer.server')
                assert canonical is sys.modules['__main__']
                assert canonical.ViewerHandler is self.handler
                assert canonical.__dict__ is self.handler.do_POST.__globals__
                assert canonical._ENGINE_LOCK is self.handler.do_POST.__globals__['_ENGINE_LOCK']
                import viewer
                assert viewer.server is canonical

                from viewer import benchmark
                calls = []
                def resize(request, **kwargs):
                    calls.append(request)
                    return {'report': {'violations': []}}
                canonical.perform_resize = resize
                benchmark.resource_hashes = lambda *args: {}
                report = benchmark.run_benchmark({
                    'scenes': [{'id': 'one', 'payload': {
                        'canvas': {'width': 10, 'height': 10},
                        'elements': [{'id': 'shape', 'type': 'shape', 'x': 0, 'y': 0,
                            'width': 10, 'height': 10, 'shape_type': 'rectangle',
                            'fill_color': '#ffffff'}]}}],
                    'targets': [{'width': 20, 'height': 10}]})
                assert len(calls) == 1
                assert report['results'][0]['status'] == 'ok'

            def server_close(self):
                closed.append(True)

        http.server.ThreadingHTTPServer = FakeServer
        sys.argv = ['viewer.server', '--port', '65432']
        try:
            runpy.run_module('viewer.server', run_name='__main__', alter_sys=True)
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError('CLI did not exit normally')
        assert closed == [True]
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=benchmark.ROOT,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
