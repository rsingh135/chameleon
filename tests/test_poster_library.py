"""Library input readiness, origin gates and contained read-only file routes."""

import hashlib
import http.client
import json
import threading

import pytest

from viewer import poster_library as library
from viewer import server


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "CORPUS_ROOT", tmp_path)
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    (gallery / "index.html").write_text("<h1>Poster review</h1>")
    monkeypatch.setattr(library, "GALLERY_ROOT", gallery)
    original = tmp_path / "original.svg"
    original.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    prepared = tmp_path / "prepared" / "example"
    prepared.mkdir(parents=True)
    (prepared / "import-report.json").write_text(json.dumps({"status": "ready"}))
    bundle = {
        "source": {
            "canvas": {"width": 100, "height": 100},
            "elements": [
                {
                    "id": "panel",
                    "type": "shape",
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": 100,
                    "shape_type": "rectangle",
                    "fill_color": "#abcdef",
                }
            ],
        },
        "target_canvas": {"width": 200, "height": 100},
        "assets": [],
        "fonts": [],
        "document": {"snapshot_id": "example", "page_id": "example", "artboard_id": "example"},
    }
    (prepared / "bundle.json").write_text(json.dumps(bundle))
    (prepared / "native.png").write_bytes(b"example-png")
    entries = [
        {
            "id": "example",
            "title": "An editable poster",
            "split": "reserve",
            "author": "Artist",
            "license_id": "CC0-1.0",
            "source_url": "https://example.test/poster",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "original_path": "original.svg",
            "prepared_path": "prepared/example",
            "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        }
    ]
    (tmp_path / "catalogue.json").write_text(json.dumps({"schema_version": 1, "examples": entries}))
    return tmp_path, entries, bundle


@pytest.fixture
def http_server(corpus, monkeypatch):
    def no_automatic_resize(*args, **kwargs):
        raise AssertionError("Browsing a reserve must never run the engine.")

    monkeypatch.setattr(server, "perform_resize", no_automatic_resize)
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance.server_port
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


def request(port, path, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_ready_reserve_is_available_without_automatic_resize(http_server, corpus):
    status, _, raw = request(http_server, "/api/posters")
    data = json.loads(raw)
    assert status == 200
    assert data["examples"][0]["ready"] is True
    assert data["examples"][0]["split"] == "reserve"
    status, _, raw = request(http_server, data["examples"][0]["bundle_url"])
    assert status == 200 and json.loads(raw) == corpus[2]


def test_original_download_preserves_hash_and_disables_active_content(http_server, corpus):
    status, headers, raw = request(http_server, "/api/posters/example/original")
    assert status == 200
    assert hashlib.sha256(raw).hexdigest() == corpus[1][0]["sha256"]
    assert headers["Content-Disposition"] == "attachment"
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    (corpus[0] / "original.svg").write_text("changed")
    assert request(http_server, "/api/posters/example/original")[0] == 400


def test_unsupported_import_is_visible_and_cannot_be_loaded(http_server, corpus):
    (corpus[0] / "prepared/example/import-report.json").write_text('{"status":"unsupported"}')
    status, _, raw = request(http_server, "/api/posters")
    assert status == 200 and json.loads(raw)["examples"][0]["ready"] is False
    assert request(http_server, "/api/posters/example/bundle")[0] == 400


@pytest.mark.parametrize(
    "path",
    [
        "/api/posters/../original",
        "/api/posters/%2e%2e/original",
        "/api/posters/example/unknown",
        "/posters/%2e%2e/original.svg",
        "/posters/../catalogue.json",
    ],
)
def test_poster_routes_do_not_escape_their_roots(http_server, path):
    assert request(http_server, path)[0] in {400, 404}


def test_gallery_and_library_keep_same_origin_gate(http_server):
    status, headers, raw = request(http_server, "/posters/")
    assert status == 200 and b"Poster review" in raw
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert request(http_server, "/api/posters", {"Origin": "https://outside.example"})[0] == 403
    assert request(http_server, "/posters/", {"Host": "outside.example"})[0] == 403


def test_catalogue_rejects_invalid_paths_duplicates_and_active_links(corpus):
    root, entries, _ = corpus
    entries[0]["source_url"] = "javascript:alert(1)"
    (root / "catalogue.json").write_text(json.dumps({"schema_version": 1, "examples": entries}))
    assert library.library_metadata()["examples"][0]["source_url"] is None
    entries[0]["prepared_path"] = "../outside"
    (root / "catalogue.json").write_text(json.dumps({"schema_version": 1, "examples": entries}))
    with pytest.raises(ValueError, match="outside"):
        library.library_metadata()
    (root / "catalogue.json").write_text(json.dumps({"schema_version": 1, "examples": entries * 2}))
    with pytest.raises(ValueError, match="Duplicate"):
        library.catalogue()
