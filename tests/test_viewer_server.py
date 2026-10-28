"""Local HTTP contract, request limits and real renderer integration."""

import base64
import http.client
import json
import threading
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from engine.eval.style import analyze_style
from viewer import server


@pytest.fixture
def http_server():
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance.server_port
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


@pytest.fixture
def payload():
    return {
        "source": {
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
        },
        "target_canvas": {"width": 100, "height": 100},
    }


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=40)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


def post(port, payload):
    status, _, body = request(
        port,
        "POST",
        "/api/resize",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    return status, json.loads(body)


def test_health_samples_static_and_font_routes(http_server):
    status, _, body = request(http_server, "GET", "/api/health")
    assert status == 200
    assert json.loads(body) == {"status": "ok"}
    status, _, body = request(http_server, "GET", "/api/samples")
    catalogue = json.loads(body)
    assert status == 200
    assert len(catalogue["samples"]) == 3
    assert all({"id", "label", "task"} <= sample.keys() for sample in catalogue["samples"])
    assert all({"id", "label", "width", "height"} <= item.keys() for item in catalogue["targets"])
    status, kind, body = request(http_server, "GET", "/")
    assert status == 200
    assert "text/html" in kind
    assert b"<html" in body.lower()
    status, _, body = request(http_server, "GET", "/fonts/SolsticeSans-Regular.ttf")
    assert status == 200
    assert body[:4] == b"\x00\x01\x00\x00"


@pytest.mark.parametrize(
    "path",
    ["/../README.md", "/%2e%2e/README.md", "/server.py", "/fonts/../README.md", "/api/unknown"],
)
def test_static_routes_cannot_read_repo_sources(http_server, path):
    status, kind, body = request(http_server, "GET", path)
    assert status == 404
    assert "application/json" in kind
    assert "error" in json.loads(body)


def test_resize_api_returns_agreed_fields(http_server, payload, monkeypatch):
    expected = {
        "design": payload["source"],
        "source_png": "data:image/png;base64,c291cmNl",
        "output_png": "data:image/png;base64,b3V0cHV0",
        "report": {"violations": [], "metrics": {"collision_free": 1}, "total": 1},
        "warnings": [],
        "elapsed_ms": 3.4,
    }
    monkeypatch.setattr(server, "perform_resize", lambda request: expected)
    status, result = post(http_server, payload)
    assert status == 200
    assert result == expected


@pytest.mark.parametrize("raw", [b"{broken", b"[]", b"null", b'{"source":null}'])
def test_malformed_requests_get_json_errors(http_server, raw):
    status, _, body = request(
        http_server, "POST", "/api/resize", raw, {"Content-Type": "application/json"}
    )
    assert status == 400
    assert json.loads(body)["error"]


@pytest.mark.parametrize(
    "width,height",
    [(0, 100), (-1, 100), (100.1, 100), (True, 100), (8193, 1), (8192, 8192)],
)
def test_canvas_limits_are_rejected_before_engine(http_server, payload, width, height):
    payload["target_canvas"].update(width=width, height=height)
    status, result = post(http_server, payload)
    assert status == 400
    assert "target_canvas" in result["error"]


@pytest.mark.parametrize("field,value", [("x", float("nan")), ("width", -1), ("height", "80")])
def test_unsafe_geometry_is_rejected(http_server, payload, field, value):
    payload["source"]["elements"][0][field] = value
    status, result = post(http_server, payload)
    assert status == 400
    assert result["error"]


def test_duplicate_and_missing_ids_are_rejected(http_server, payload):
    payload["source"]["elements"] *= 2
    status, result = post(http_server, payload)
    assert status == 400
    assert "unique" in result["error"]
    payload["source"]["elements"] = [deepcopy(payload["source"]["elements"][0])]
    del payload["source"]["elements"][0]["id"]
    status, result = post(http_server, payload)
    assert status == 400
    assert ".id" in result["error"]


def test_asset_paths_cannot_escape_brand_kit(http_server, payload):
    payload["source"]["elements"][0].update(type="image", asset_id="../../README.md")
    status, result = post(http_server, payload)
    assert status == 400
    assert "asset_id" in result["error"]


def test_request_size_content_type_and_origin_are_bounded(http_server, payload):
    status, _, body = request(
        http_server,
        "POST",
        "/api/resize",
        b"{}",
        {"Content-Type": "application/json", "Content-Length": str(server.MAX_REQUEST_BYTES + 1)},
    )
    assert status == 413
    assert "2 MiB" in json.loads(body)["error"]
    status, _, _ = request(http_server, "POST", "/api/resize", b"{}")
    assert status == 415
    status, _, body = request(
        http_server,
        "POST",
        "/api/resize",
        json.dumps(payload),
        {"Content-Type": "application/json", "Origin": "https://outside.example"},
    )
    assert status == 403
    assert "same-origin" in json.loads(body)["error"]


def test_runtime_failure_is_truthfully_reported(http_server, payload, monkeypatch):
    def fail(request):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(server, "perform_resize", fail)
    status, result = post(http_server, payload)
    assert status == 500
    assert "renderer unavailable" in result["error"]


@pytest.mark.parametrize("sample", ["level2_core", "level3_advanced"])
def test_real_samples_render_and_preserve_style_runs_and_warnings(http_server, sample):
    task_path = server.REPO_ROOT / "input" / f"chameleon_sample_{sample}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    original = deepcopy(task)
    status, result = post(http_server, task)
    assert status == 200, result
    assert result["design"]["canvas"] == task["target_canvas"]
    assert result["report"]["violations"] == []
    assert result["report"]["metrics"]
    assert 0 <= result["report"]["total"] <= 1
    assert result["report"]["style"] == analyze_style(
        task["source"], result["design"], server.ASSETS_DIR
    )
    assert 0 <= result["report"]["style"]["score"] <= 1
    assert result["elapsed_ms"] >= 0
    if sample == "level3_advanced":
        assert any("solstice_holiday_badge_2025.png" in warning for warning in result["warnings"])
    by_id = {element["id"]: element for element in result["design"]["elements"]}
    for element in task["source"]["elements"]:
        assert by_id[element["id"]].get("style_runs") == element.get("style_runs")
    for key in ("source_png", "output_png"):
        prefix, encoded = result[key].split(",", 1)
        assert prefix == "data:image/png;base64"
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        assert image.format == "PNG"
        canvas = task["source"]["canvas"] if key == "source_png" else task["target_canvas"]
        assert image.size == (canvas["width"], canvas["height"])
    assert task == original


def test_download_png_uses_full_target_resolution(http_server, payload):
    payload["target_canvas"] = {"width": 2560, "height": 1440}
    status, result = post(http_server, payload)
    assert status == 200, result
    output_png = base64.b64decode(result["output_png"].split(",", 1)[1])
    source_png = base64.b64decode(result["source_png"].split(",", 1)[1])
    assert Image.open(BytesIO(output_png)).size == (2560, 1440)
    assert Image.open(BytesIO(source_png)).size == (100, 100)
    assert result["report"]["style"]["warnings"] == []
