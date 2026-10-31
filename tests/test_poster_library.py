"""Library input readiness, origin gates and contained read-only file routes."""

import hashlib
import http.client
import json
import threading
from html.parser import HTMLParser

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


class GalleryMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []
        self.scripts = []
        self.active_attributes = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.active_attributes.extend(name for name in values if name.startswith("on"))
        if tag == "a":
            self.links.append(values.get("href"))
        elif tag == "img":
            self.images.append(values.get("src"))
        elif tag == "script":
            self.scripts.append(values)


@pytest.mark.parametrize("route", ["/posters", "/posters/", "/posters/index.html"])
def test_clean_checkout_gallery_uses_existing_original_source_routes(http_server, corpus, route):
    (corpus[0] / "gallery/index.html").unlink()
    status, headers, raw = request(http_server, route)
    page = GalleryMarkup()
    page.feed(raw.decode("utf-8"))
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert "/?poster=example" in page.links
    assert "/api/posters/example/original" in page.links
    assert page.images == ["/api/posters/example/preview"]
    assert page.scripts == []
    assert request(http_server, page.images[0])[2] == b"example-png"
    assert (
        request(http_server, "/api/posters/example/original")[2]
        == (corpus[0] / "original.svg").read_bytes()
    )
    assert b"Personal test originals" in raw
    assert b"compare a resized result" in raw
    assert not (corpus[0] / "gallery/index.html").exists()


def test_fallback_gallery_escapes_metadata_and_suppresses_active_urls(http_server, corpus):
    root, entries, _ = corpus
    (root / "gallery/index.html").unlink()
    entries[0].update(
        title='<script>alert("title")</script>',
        attribution_text='<img src="evil">',
        source_note='<b onclick="bad()">Note</b>',
        license_id='" onmouseover="bad()',
        source_url="javascript:alert(1)",
        license_url='https://example.test/license?text=" onmouseover="bad()',
    )
    (root / "catalogue.json").write_text(json.dumps({"schema_version": 1, "examples": entries}))
    status, _, raw = request(http_server, "/posters/")
    page = GalleryMarkup()
    page.feed(raw.decode("utf-8"))
    assert status == 200 and page.scripts == []
    assert page.active_attributes == []
    assert b"&lt;script&gt;" in raw and b"&lt;img" in raw and b"&lt;b" in raw
    assert "javascript:alert(1)" not in page.links
    assert all(link.startswith(("/", "https://")) for link in page.links)
    assert page.images == ["/api/posters/example/preview"]


def test_fallback_gallery_preserves_unsupported_readiness_and_missing_preview(http_server, corpus):
    root, _, _ = corpus
    (root / "gallery/index.html").unlink()
    (root / "prepared/example/import-report.json").write_text('{"status":"unsupported"}')
    (root / "prepared/example/native.png").unlink()
    status, _, raw = request(http_server, "/posters/")
    page = GalleryMarkup()
    page.feed(raw.decode("utf-8"))
    assert status == 200 and not page.images
    assert "/?poster=example" not in page.links
    assert b"Import support is still needed" in raw
    assert b"Source preview is unavailable" in raw


@pytest.mark.parametrize(
    "path",
    ["/posters/missing.html", "/posters/%2e%2e/catalogue.json", "/posters/%2e%2e%5ccatalogue.json"],
)
def test_fallback_gallery_does_not_turn_missing_or_escaped_paths_into_success(
    http_server, corpus, path
):
    (corpus[0] / "gallery/index.html").unlink()
    assert request(http_server, path)[0] in {400, 404}
