"""Studio is the product surface; archived batch fixtures are not public UI."""

import http.client
import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.append(dict(attrs).get("href", ""))


@pytest.mark.parametrize("page", ["index.html", "inspector.html"])
def test_studio_navigation_does_not_advertise_a_batch_page(page):
    links = Links()
    links.feed((ROOT / "viewer" / page).read_text(encoding="utf-8-sig"))
    retired = {"/benchmark", "/benchmark/", "/benchmark.html"}
    assert not any(urlsplit(href).path.casefold() in retired for href in links.hrefs)
    assert "/" in links.hrefs


@pytest.fixture(scope="module")
def local_server():
    from viewer.server import create_server

    instance = create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance.server_port
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


def request(port, method, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, response.getheader("Location"), response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize(
    "path",
    [
        "/benchmark.html",
        "/benchmark",
        "/benchmark/",
        "/./benchmark.html",
        "/%62enchmark.html",
        "/Benchmark.html",
        "/nested/../benchmark.html",
        "/%2e/benchmark.html",
        "/%62enchmark/?view=compare",
    ],
)
def test_legacy_batch_page_addresses_redirect_to_studio(local_server, method, path):
    status, location, body = request(local_server, method, path)
    assert status == 303
    assert location == "/"
    if method == "HEAD":
        assert body == b""


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize(
    "path",
    [
        "/benchmark.js",
        "/benchmark.css",
        "/%62enchmark.js",
        "/Benchmark.CSS",
        "/./benchmark.js",
        "/nested/../%62enchmark.css?version=1",
    ],
)
def test_legacy_batch_script_and_style_are_not_public(local_server, method, path):
    status, _, body = request(local_server, method, path)
    assert status == 404
    if method == "HEAD":
        assert body == b""


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("path", ["/", "/inspector.html", "/api/health"])
def test_studio_and_existing_routes_remain_available(local_server, method, path):
    status, location, body = request(local_server, method, path)
    assert status == 200
    assert location is None
    if method == "HEAD":
        assert body == b""
