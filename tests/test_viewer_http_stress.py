"""Bounded local HTTP stress, cancellation boundaries and native resource isolation."""

import base64
import http.client
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from viewer import benchmark, document_bridge, server


@pytest.fixture
def live_server(monkeypatch, request):
    if hasattr(request, "param"):
        monkeypatch.setattr(server, "MAX_IN_FLIGHT_RESIZES", request.param, raising=False)
    receipt = benchmark.method_receipt()
    monkeypatch.setattr(benchmark, "method_receipt", lambda: receipt)
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


@pytest.fixture
def shape_task():
    return {
        "source": {
            "canvas": {"width": 16, "height": 16},
            "elements": [
                {
                    "id": "panel",
                    "type": "shape",
                    "shape_type": "rectangle",
                    "fill_color": "#ffffff",
                    "x": 0,
                    "y": 0,
                    "width": 16,
                    "height": 16,
                }
            ],
        },
        "target_canvas": {"width": 32, "height": 16},
    }


def _request(instance, path, payload, timeout=30):
    connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=timeout)
    try:
        connection.request(
            "POST", path, json.dumps(payload), {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_truncated_valid_json_never_reaches_engine(live_server, shape_task, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "perform_resize", lambda payload: calls.append(payload) or {})
    raw = json.dumps(shape_task).encode()
    with socket.create_connection(("127.0.0.1", live_server.server_port), timeout=2) as client:
        headers = (
            f"POST /api/resize HTTP/1.0\r\nHost:127.0.0.1:{live_server.server_port}\r\n"
            f"Content-Type:application/json\r\nContent-Length:{len(raw) + 10}\r\n\r\n"
        ).encode()
        client.sendall(headers + raw)
        client.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(client)
        response.begin()
        assert response.status == 400
        assert "incomplete" in json.loads(response.read())["error"].lower()
    assert calls == []


def test_stalled_body_returns_timeout_and_health_remains_available(live_server, monkeypatch):
    monkeypatch.setattr(server, "BODY_READ_TIMEOUT", 0.1, raising=False)
    with socket.create_connection(("127.0.0.1", live_server.server_port), timeout=2) as client:
        client.settimeout(1)
        client.sendall(
            (
                f"POST /api/resize HTTP/1.0\r\nHost:127.0.0.1:{live_server.server_port}\r\n"
                "Content-Type:application/json\r\nContent-Length:100\r\n\r\n{"
            ).encode()
        )
        response = http.client.HTTPResponse(client)
        response.begin()
        assert response.status == 408
        assert "timed out" in json.loads(response.read())["error"].lower()
    connection = http.client.HTTPConnection("127.0.0.1", live_server.server_port, timeout=2)
    try:
        connection.request("GET", "/api/health")
        assert connection.getresponse().status == 200
    finally:
        connection.close()


def test_trickling_upload_cannot_extend_total_body_deadline(live_server, monkeypatch):
    monkeypatch.setattr(server, "BODY_READ_TIMEOUT", 0.15)
    stop = threading.Event()
    with socket.create_connection(("127.0.0.1", live_server.server_port), timeout=2) as client:
        client.settimeout(1)
        client.sendall(
            (
                f"POST /api/resize HTTP/1.0\r\nHost:127.0.0.1:{live_server.server_port}\r\n"
                "Content-Type:application/json\r\nContent-Length:10000\r\n\r\n{"
            ).encode()
        )

        def trickle():
            while not stop.wait(0.02):
                try:
                    client.sendall(b" ")
                except ConnectionError:
                    return

        sender = threading.Thread(target=trickle)
        sender.start()
        try:
            response = http.client.HTTPResponse(client)
            response.begin()
            assert response.status == 408
            assert "timed out" in json.loads(response.read())["error"].lower()
        finally:
            stop.set()
            sender.join(timeout=1)


@pytest.mark.parametrize("live_server", [1], indirect=True)
def test_busy_engine_rejects_extra_work_then_accepts_retry(live_server, shape_task, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def hold_resize(payload):
        entered.set()
        assert release.wait(timeout=3)
        return {"finished": True}

    monkeypatch.setattr(server, "perform_resize", hold_resize)
    with ThreadPoolExecutor(max_workers=1) as worker:
        first = worker.submit(_request, live_server, "/api/resize", shape_task)
        assert entered.wait(timeout=2)
        try:
            try:
                second = _request(live_server, "/api/resize", shape_task, timeout=0.5)
            except TimeoutError:
                second = None
        finally:
            release.set()
        assert first.result(timeout=3)[0] == 200
    assert second is not None, "An extra request queued behind an already-full processing budget."
    assert second[0] == 503
    assert "busy" in second[1]["error"].lower()
    assert _request(live_server, "/api/resize", shape_task) == (200, {"finished": True})


@pytest.mark.parametrize("input_kind", [[], {}])
def test_malformed_scene_metadata_is_client_error(live_server, shape_task, input_kind):
    status, result = _request(
        live_server,
        "/api/benchmark/run",
        {
            "scenes": [{"id": "one", "input_kind": input_kind, "payload": shape_task}],
            "targets": [shape_task["target_canvas"]],
        },
    )
    assert status == 400
    assert "input_kind" in result["error"]


def _native_text_task(shape_task, descriptor):
    task = deepcopy(shape_task)
    task["source"]["elements"] = [
        {
            "id": "copy",
            "type": "text",
            "x": 0,
            "y": 0,
            "width": 16,
            "height": 16,
            "font_size": 8,
            "font_family": "SolsticeSans",
            "color": "#ffffff",
            "content": "A",
            "native_text_effects": descriptor,
        }
    ]
    return task


@pytest.mark.parametrize("kind", ["outline", "gradient"])
@pytest.mark.parametrize(
    "color", [None, [], {}, "rgb(1)", "transparent"], ids=["null", "list", "object", "rgb", "css"]
)
def test_invalid_native_paint_metadata_is_client_error(
    live_server, shape_task, monkeypatch, kind, color
):
    descriptor = {"version": 1}
    if kind == "outline":
        descriptor[kind] = {"color": color, "width_em": 0.1, "fill": True}
    else:
        descriptor[kind] = {
            "x1": 0,
            "y1": 0,
            "x2": 1,
            "y2": 0,
            "stops": [{"offset": 0, "color": color}, {"offset": 1, "color": "#ffffff"}],
        }
    calls = []
    monkeypatch.setattr(server, "perform_resize", lambda payload: calls.append(payload) or {})
    task = _native_text_task(shape_task, descriptor)
    status, result = _request(live_server, "/api/resize", task)
    assert status == 400
    assert "native_text_effects" in result["error"]
    assert calls == []


@pytest.mark.parametrize(
    "descriptor",
    [
        {"version": 1, "outline": {"color": "#ffffff", "width_em": 0.1, "fill": True}},
        {
            "version": 1,
            "gradient": {
                "x1": 0,
                "y1": 0,
                "x2": 1,
                "y2": 0,
                "stops": [
                    {"offset": 0, "color": "rgba(1,2,3,0.5)"},
                    {"offset": 1, "color": "#ffffff"},
                ],
            },
        },
    ],
    ids=["outline", "gradient"],
)
def test_valid_native_paint_descriptors_reach_processor_unchanged(
    live_server, shape_task, monkeypatch, descriptor
):
    calls = []

    def record(payload):
        calls.append(deepcopy(payload))
        return {"design": payload["source"]}

    monkeypatch.setattr(server, "perform_resize", record)
    task = _native_text_task(shape_task, descriptor)
    status, result = _request(live_server, "/api/resize", task)
    assert status == 200
    assert calls == [task]
    assert result["design"] == task["source"]


def _native_bundle(color, identifier):
    image = BytesIO()
    Image.new("RGB", (32, 48), color).save(image, format="PNG")
    return {
        "source": {
            "canvas": {"width": 64, "height": 96},
            "elements": [
                {
                    "id": identifier,
                    "type": "image",
                    "x": 0,
                    "y": 0,
                    "width": 64,
                    "height": 96,
                    "asset_id": "same-name.png",
                    "fit": "contain",
                }
            ],
        },
        "target_canvas": {"width": 80, "height": 60},
        "document": {"snapshot_id": identifier, "page_id": "page", "artboard_id": "artboard"},
        "assets": [
            {
                "asset_id": "same-name.png",
                "mime_type": "image/png",
                "data_base64": base64.b64encode(image.getvalue()).decode(),
            }
        ],
        "fonts": [],
    }


@pytest.mark.parametrize("round_index", range(4))
def test_concurrent_studio_and_batch_keep_native_pixels_and_ids_isolated(
    live_server, tmp_path, monkeypatch, round_index
):
    monkeypatch.setattr(document_bridge, "DOCUMENT_ROOT", tmp_path / "native-documents")
    original_resize = server.resize_with_state
    active = peak = 0
    guard = threading.Lock()

    def observed_resize(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            return original_resize(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(server, "resize_with_state", observed_resize)
    barrier = threading.Barrier(4)
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255)]

    def submit(index):
        identifier = f"native-photo-{round_index}-{index}"
        task = _native_bundle(colors[index], identifier)
        is_batch = index % 2
        payload = (
            {
                "expected_method_digest": live_server.benchmark_method["digest"],
                "scenes": [{"id": identifier, "payload": task}],
                "targets": [task["target_canvas"]],
            }
            if is_batch
            else task
        )
        barrier.wait(timeout=3)
        status, result = _request(
            live_server,
            "/api/benchmark/run" if is_batch else "/api/document/resize",
            payload,
        )
        assert status == 200, result
        if is_batch:
            row = result["results"][0]
            assert row["status"] == "ok", row.get("error", "missing native result")
            result = row["result"]
        assert result["design"]["canvas"] == task["target_canvas"]
        assert [element["id"] for element in result["design"]["elements"]] == [identifier]
        for key, size in (("source_png", (64, 96)), ("output_png", (80, 60))):
            with Image.open(BytesIO(base64.b64decode(result[key].split(",")[1]))) as image:
                assert image.size == size
                assert image.convert("RGB").getpixel((size[0] // 2, size[1] // 2)) == colors[index]
        return result["document"]["resource_digest"]

    with ThreadPoolExecutor(max_workers=4) as workers:
        resources = list(workers.map(submit, range(4)))
    assert len(set(resources)) == 4
    assert peak == 1
    assert active == 0


def test_startup_hash_failure_releases_bound_socket(monkeypatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def fail():
        raise PermissionError("checkout is unavailable")

    monkeypatch.setattr(benchmark, "method_receipt", fail)
    with pytest.raises(PermissionError, match="checkout"):
        server.create_server(port)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def test_concurrent_first_cache_creation_has_stable_containment(tmp_path):
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255)]
    for round_index in range(20):
        root = tmp_path / f"fresh-cache-{round_index}"
        barrier = threading.Barrier(4)

        def prepare(index, number=round_index, destination=root, gate=barrier):
            payload = _native_bundle(colors[index], f"cache-{number}-{index}")
            gate.wait(timeout=3)
            try:
                return document_bridge.prepare_document(payload, destination)
            except ValueError as exc:
                trace = exc.__traceback__
                while trace is not None:
                    frame = trace.tb_frame
                    if "package_root" in frame.f_locals and "directory" in frame.f_locals:
                        directory = frame.f_locals["directory"]
                        root_before = frame.f_locals["package_root"]
                        return {
                            "error": str(exc),
                            "root": str(root_before),
                            "directory": str(directory),
                            "directory_resolved": str(directory.resolve()),
                            "root_resolved": str(root_before.resolve()),
                        }
                    trace = trace.tb_next
                raise

        with ThreadPoolExecutor(max_workers=4) as workers:
            documents = list(workers.map(prepare, range(4)))
        errors = [document for document in documents if isinstance(document, dict)]
        assert not errors, json.dumps(errors, indent=2)
