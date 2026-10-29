"""Native resource bytes, isolation, limits, and the Adobe development HTTP boundary."""

import base64
import hashlib
import http.client
import json
import threading
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from engine.resources import DEFAULT_FONTS_DIR, get_resources, using_resources
from viewer import document_bridge as bridge
from viewer import server


def encoded(data):
    return base64.b64encode(data).decode("ascii")


@pytest.fixture
def payload():
    image = BytesIO()
    Image.new("RGBA", (20, 30), "#16abde").save(image, format="PNG")
    return {
        "source": {
            "canvas": {"width": 200, "height": 160},
            "elements": [
                {
                    "id": "image",
                    "type": "image",
                    "x": 20,
                    "y": 30,
                    "width": 40,
                    "height": 60,
                    "asset_id": "native.png",
                    "fit": "contain",
                    "z_index": 0,
                },
                {
                    "id": "headline",
                    "type": "text",
                    "x": 75,
                    "y": 40,
                    "width": 105,
                    "height": 60,
                    "content": "Native\nposter",
                    "font_family": "SolsticeSans-Regular",
                    "font_size": 20,
                    "color": "#123456",
                    "z_index": 1,
                },
            ],
        },
        "target_canvas": {"width": 400, "height": 320},
        "assets": [
            {
                "asset_id": "native.png",
                "mime_type": "image/png",
                "data_base64": encoded(image.getvalue()),
            }
        ],
        "fonts": [
            {
                "font_family": "SolsticeSans-Regular",
                "postscript_name": "SolsticeSans-Regular",
                "weight": "regular",
                "style": "normal",
                "data_base64": encoded(
                    (DEFAULT_FONTS_DIR / "SolsticeSans-Regular.ttf").read_bytes()
                ),
            }
        ],
        "document": {
            "snapshot_id": "native-capture-1",
            "page_id": "page-1",
            "artboard_id": "board-1",
            "group_tree": [],
            "node_bindings": [],
        },
    }


@pytest.fixture
def http_server(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "DOCUMENT_ROOT", tmp_path / "documents")
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance.server_port
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


def request(port, method="POST", payload=None, headers=None, path="/api/document/resize"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(
            method,
            path,
            json.dumps(payload) if payload is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_original_bytes_exact_alias_and_resource_receipt(payload, tmp_path):
    original = deepcopy(payload)
    prepared = bridge.prepare_document(payload, tmp_path)
    resources = prepared.resources
    assert (resources.assets_dir / "native.png").read_bytes() == base64.b64decode(
        payload["assets"][0]["data_base64"]
    )
    manifest = json.loads((resources.fonts_dir / "font-manifest.json").read_text())
    assert manifest[0]["font_family"] == "SolsticeSans-Regular"
    data = (resources.fonts_dir / manifest[0]["file"]).read_bytes()
    assert data == base64.b64decode(payload["fonts"][0]["data_base64"])
    assert prepared.receipt["fonts"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert "group_tree" not in prepared.receipt
    assert payload == original
    assert bridge.prepare_document(payload, tmp_path).resources == resources


def test_same_names_different_document_bytes_get_separate_caches(payload, tmp_path):
    first = bridge.prepare_document(payload, tmp_path)
    image = BytesIO()
    Image.new("RGBA", (20, 30), "#ff0000").save(image, format="PNG")
    payload["assets"][0]["data_base64"] = encoded(image.getvalue())
    second = bridge.prepare_document(payload, tmp_path)
    assert first.resources != second.resources
    assert (first.resources.assets_dir / "native.png").read_bytes() != image.getvalue()
    assert (second.resources.assets_dir / "native.png").read_bytes() == image.getvalue()
    assert get_resources().assets_dir != second.resources.assets_dir


def test_changed_package_is_not_silently_reused(payload, tmp_path):
    prepared = bridge.prepare_document(payload, tmp_path)
    (prepared.resources.assets_dir / "native.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        bridge.prepare_document(payload, tmp_path)


@pytest.mark.parametrize(
    "name", ["../outside.png", "C:\\outside.png", "CON", "NUL.png", "bad.", ".", ".."]
)
def test_resource_paths_are_portable_and_contained(payload, tmp_path, name):
    payload["assets"][0]["asset_id"] = name
    with pytest.raises(ValueError, match="filename"):
        bridge.prepare_document(payload, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(assets=[]), "Missing native image"),
        (lambda p: p.update(fonts=[]), "Missing exact native font"),
        (lambda p: p["fonts"][0].update(postscript_name="WrongFont"), "exact native PostScript"),
        (
            lambda p: p["fonts"][0].update(font_family="Other", postscript_name="Other"),
            "PostScript name",
        ),
        (lambda p: p["assets"][0].update(data_base64="not base64?"), "valid base64"),
        (
            lambda p: p["fonts"][0].update(data_base64=encoded(b"not font data")),
            "original TrueType",
        ),
        (lambda p: p["assets"][0].update(mime_type="image/jpeg"), "MIME type"),
        (lambda p: p["assets"][0].update(mime_type=[]), "original PNG"),
        (lambda p: p.update(assets=p["assets"] * 2), "Duplicate document asset"),
        (lambda p: p.update(fonts=p["fonts"] * 2), "Duplicate document font"),
        (lambda p: p["fonts"][0].update(weight=[]), "weight/style"),
        (lambda p: p.update(document={}), "snapshot_id"),
    ],
)
def test_invalid_or_missing_native_resources_fail_before_layout(payload, tmp_path, mutation, match):
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        bridge.prepare_document(payload, tmp_path)
    assert not list(tmp_path.iterdir())


def test_native_text_ranges_and_mixed_faces_fail_explicitly(payload, tmp_path):
    text = payload["source"]["elements"][1]
    text["style_runs"] = [{"start": 0, "end": 100, "color": "#ffffff"}]
    with pytest.raises(ValueError, match="inside the text"):
        bridge.prepare_document(payload, tmp_path)
    text["style_runs"] = [{"start": 0, "end": 2, "font_weight": "bold"}]
    with pytest.raises(ValueError, match="mixed font faces"):
        bridge.prepare_document(payload, tmp_path)


def test_image_only_documents_do_not_inherit_brand_fonts(payload, tmp_path):
    payload["source"]["elements"] = payload["source"]["elements"][:1]
    payload["fonts"] = []
    prepared = bridge.prepare_document(payload, tmp_path)
    assert json.loads((prepared.resources.fonts_dir / "font-manifest.json").read_text()) == []


def test_resource_size_limits_precede_materialization(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "MAX_RESOURCE_BYTES", 10)
    with pytest.raises(ValueError, match="8 MiB"):
        bridge.prepare_document(payload, tmp_path)
    monkeypatch.setattr(bridge, "MAX_RESOURCE_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(bridge, "MAX_TOTAL_RESOURCE_BYTES", 10)
    with pytest.raises(ValueError, match="16 MiB"):
        bridge.prepare_document(payload, tmp_path)
    assert not list(tmp_path.iterdir())


def test_real_native_resource_resize_renders_exact_assets(http_server, payload):
    status, headers, raw = request(
        http_server, payload=payload, headers={"Origin": bridge.DOCUMENT_ORIGIN}
    )
    result = json.loads(raw)
    assert status == 200, result
    assert headers["Access-Control-Allow-Origin"] == bridge.DOCUMENT_ORIGIN
    assert result["document"]["snapshot_id"] == "native-capture-1"
    assert result["design"]["canvas"] == payload["target_canvas"]
    assert result["preview_url"] == result["output_png"]
    assert result["warnings"] == []
    source = Image.open(BytesIO(base64.b64decode(result["source_png"].split(",")[1]))).convert(
        "RGBA"
    )
    assert source.size == (200, 160)
    assert source.getpixel((30, 40)) == (22, 171, 222, 255)
    output = Image.open(BytesIO(base64.b64decode(result["output_png"].split(",")[1])))
    assert output.size == (400, 320)
    assert not any("Missing" in warning for warning in result["warnings"])
    assert get_resources().fonts_dir == DEFAULT_FONTS_DIR


def test_exact_development_origin_preflight_is_scoped_to_document_route(http_server):
    status, headers, body = request(
        http_server,
        "OPTIONS",
        headers={
            "Origin": bridge.DOCUMENT_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert status == 204 and body == b""
    assert headers["Access-Control-Allow-Origin"] == bridge.DOCUMENT_ORIGIN
    assert headers["Access-Control-Allow-Private-Network"] == "true"
    assert "Access-Control-Allow-Credentials" not in headers
    assert (
        request(
            http_server, "OPTIONS", headers={"Origin": bridge.DOCUMENT_ORIGIN}, path="/api/resize"
        )[0]
        == 403
    )
    assert (
        request(
            http_server,
            "OPTIONS",
            headers={"Origin": bridge.DOCUMENT_ORIGIN, "Access-Control-Request-Method": "DELETE"},
        )[0]
        == 403
    )


@pytest.mark.parametrize(
    "origin", ["https://outside.example", "https://localhost:5242", "http://localhost:5241", "null"]
)
def test_other_sites_cannot_use_document_bridge(http_server, payload, origin):
    status, headers, _ = request(http_server, payload=payload, headers={"Origin": origin})
    assert status == 403
    assert "Access-Control-Allow-Origin" not in headers


def test_document_transport_size_and_host_are_bounded(http_server):
    assert (
        request(
            http_server,
            payload={},
            headers={"Content-Length": str(bridge.MAX_DOCUMENT_REQUEST_BYTES + 1)},
        )[0]
        == 413
    )
    status, headers, _ = request(
        http_server,
        payload={},
        headers={"Host": "outside.example", "Origin": bridge.DOCUMENT_ORIGIN},
    )
    assert status == 403
    assert "Access-Control-Allow-Origin" not in headers


def test_default_renderer_remains_available_after_document_context(payload, tmp_path):
    prepared = bridge.prepare_document(payload, tmp_path)
    with using_resources(prepared.resources.assets_dir, prepared.resources.fonts_dir):
        png, warnings = server._render_png(payload["source"])
        assert png.startswith("data:image/png;base64,") and warnings == []
    task = json.loads((server.REPO_ROOT / "input/chameleon_sample_level1_warmup.json").read_text())
    png, _ = server._render_png(task["source"])
    assert png.startswith("data:image/png;base64,")
