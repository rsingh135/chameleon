"""Stable Express viewer: precise CORS, strict native output, portable evidence."""

import base64
import http.client
import json
import threading
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from engine.eval.render import render_image
from engine.stage_one import strictly_in_bounds
from tools.package_addon_results import package_result
from viewer import server
from viewer.addon_bridge import addon_origins, source_digest


@pytest.fixture
def http_server():
    instance = server.create_server(0, allowed_addon_origins=["https://viewer.example"])
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance.server_port
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


def request(port, method, path="/api/addon/resize", payload=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=40)
    try:
        connection.request(
            method,
            path,
            json.dumps(payload).encode() if payload is not None else None,
            {"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_addon_preflight_is_scoped_to_its_route_and_exact_origin(http_server):
    headers = {
        "Origin": "https://viewer.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
        "Access-Control-Request-Private-Network": "true",
    }
    status, returned, _ = request(http_server, "OPTIONS", headers=headers)
    assert status == 204
    assert returned["Access-Control-Allow-Origin"] == headers["Origin"]
    assert returned["Access-Control-Allow-Private-Network"] == "true"
    assert "Access-Control-Allow-Credentials" not in returned
    assert request(http_server, "OPTIONS", path="/api/resize", headers=headers)[0] == 403
    for origin in ["https://viewer.example.evil", "http://viewer.example", "null"]:
        status, returned, _ = request(http_server, "OPTIONS", headers={**headers, "Origin": origin})
        assert status == 403 and "Access-Control-Allow-Origin" not in returned


@pytest.mark.parametrize("value", ["https://*.example", "http://localhost", "https://x/a", "null"])
def test_addon_origin_configuration_refuses_wildcards_and_non_origins(value):
    with pytest.raises(ValueError, match="HTTPS origins"):
        addon_origins(value)


def test_addon_rejects_missing_identity_before_engine(http_server, monkeypatch):
    monkeypatch.setattr(server, "perform_addon_resize", lambda _: pytest.fail("Engine invoked"))
    payload = {
        "source": {
            "canvas": {"width": 100, "height": 100},
            "elements": [
                {
                    "id": "plate",
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
    status, _, body = request(
        http_server, "POST", payload=payload, headers={"Origin": "https://viewer.example"}
    )
    assert status == 400
    assert "request_id" in json.loads(body)["error"]


def test_addon_real_response_preserves_identity_and_uses_strict_stage_one(http_server):
    task = json.loads(
        (server.REPO_ROOT / "input/chameleon_sample_level2_core.json").read_text("utf-8")
    )
    task.update(target_canvas={"width": 540, "height": 540}, request_id="source-and-target-42")
    original = deepcopy(task)
    status, headers, body = request(
        http_server, "POST", payload=task, headers={"Origin": "https://viewer.example"}
    )
    assert status == 200, body
    result = json.loads(body)
    assert headers["Access-Control-Allow-Origin"] == "https://viewer.example"
    assert result["request_id"] == task["request_id"]
    assert result["source_digest"] == source_digest(task["source"])
    assert result["report"]["violations"] == []
    assert strictly_in_bounds(result["design"])
    assert task == original
    for name, canvas in [("source", task["source"]["canvas"]), ("output", task["target_canvas"])]:
        with Image.open(BytesIO(base64.b64decode(result[f"{name}_png"].split(",")[1]))) as png:
            assert png.size == (canvas["width"], canvas["height"])


def test_portable_bundle_retains_engine_png_and_refuses_wrong_geometry():
    source = {
        "canvas": {"width": 64, "height": 64},
        "elements": [
            {
                "id": "plate",
                "type": "shape",
                "x": 4,
                "y": 4,
                "width": 56,
                "height": 56,
                "shape_type": "rectangle",
                "fill_color": "#ffffff",
            }
        ],
    }
    task = {"source": source, "target_canvas": source["canvas"]}
    stream = BytesIO()
    render_image(source, max_size=64).save(stream, format="PNG")
    png = stream.getvalue()
    kwargs = {"assets_dir": server.ASSETS_DIR, "fonts_dir": server.FONTS_DIR}
    result = package_result(task, source, png, **kwargs)
    assert base64.b64decode(result["output_png"].split(",")[1]) == png
    changed = deepcopy(source)
    changed["elements"][0]["id"] = "replacement"
    with pytest.raises(ValueError, match="stage-one contract"):
        package_result(task, changed, png, **kwargs)
    stream = BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, format="PNG")
    with pytest.raises(ValueError, match="full-resolution"):
        package_result(task, source, stream.getvalue(), **kwargs)
    stream = BytesIO()
    Image.new("RGBA", (64, 64), "blue").save(stream, format="PNG")
    with pytest.raises(ValueError, match="pixels do not match"):
        package_result(task, source, stream.getvalue(), **kwargs)


def test_addon_rejects_font_substitution_before_engine(http_server, monkeypatch):
    task = json.loads(
        (server.REPO_ROOT / "input/chameleon_sample_level1_warmup.json").read_text("utf-8")
    )
    task["request_id"] = "missing-font"
    next(e for e in task["source"]["elements"] if e["type"] == "text")["font_family"] = "AbsentFace"
    monkeypatch.setattr(server, "perform_addon_resize", lambda _: pytest.fail("Engine invoked"))
    status, _, body = request(
        http_server, "POST", payload=task, headers={"Origin": "https://viewer.example"}
    )
    assert status == 400 and "Font face is unavailable" in json.loads(body)["error"]
    with pytest.raises(ValueError, match="Font face is unavailable"):
        package_result(
            task,
            task["source"],
            b"unused",
            assets_dir=server.ASSETS_DIR,
            fonts_dir=server.FONTS_DIR,
        )
