"""Early HTTP rejections stay complete on Windows without unbounded body drains."""

import http.client
import json
import threading
from collections import Counter
from copy import deepcopy
from email.message import Message
from types import SimpleNamespace

import pytest

from viewer import benchmark, server


@pytest.fixture
def local_server(monkeypatch):
    calls = Counter()
    receipt = {
        "digest": "a" * 64,
        "options": {
            "request_policy": {
                "studio_timeout_ms": 60000,
                "benchmark_timeout_ms": 120000,
                "model_configured": False,
            }
        },
    }
    monkeypatch.setattr(benchmark, "method_receipt", lambda: deepcopy(receipt))

    def forbidden(*_args, **_kwargs):
        calls["heavy_work"] += 1
        raise AssertionError("Rejected requests must not prepare or process a document")

    monkeypatch.setattr(server, "perform_resize", forbidden)
    monkeypatch.setattr(server, "prepare_document", forbidden)
    monkeypatch.setattr(benchmark, "run_benchmark", forbidden)
    instance = server.create_server(0)
    worker = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield instance, calls
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=3)


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Content-Type": "text/plain"}, 415),
        ({"Content-Type": "application/json", "Origin": "https://outside.example"}, 403),
        ({"Content-Type": "application/json", "Host": "outside.example"}, 403),
    ],
)
def test_twenty_small_rejected_posts_deliver_complete_json_without_transport_resets(
    local_server, headers, expected
):
    instance, calls = local_server
    for _ in range(20):
        connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=5)
        try:
            connection.request("POST", "/api/resize", body=b'{"rejected":true}', headers=headers)
            response = connection.getresponse()
            body = json.loads(response.read())
            assert response.status == expected
            assert body["error"] and "design" not in body
            assert response.getheader("Connection") == "close"
        finally:
            connection.close()
    assert calls["heavy_work"] == 0


def test_a_stalled_rejected_body_does_not_delay_the_error_response(local_server, monkeypatch):
    instance, calls = local_server
    # Widen only the cleanup budget so a one-second client deadline proves the
    # response is sent first, without a fragile sub-250ms scheduling assertion.
    monkeypatch.setattr(server.ViewerHandler, "_rejection_drain_seconds", 3.0)
    connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=1)
    try:
        connection.putrequest("POST", "/api/resize")
        connection.putheader("Origin", "https://outside.example")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "2048")
        connection.endheaders(b"{")
        response = connection.getresponse()
        assert response.status == 403
        assert "same-origin" in json.loads(response.read())["error"]
    finally:
        connection.close()  # Release the cleanup read immediately after the response.
    assert calls["heavy_work"] == 0


def _handler(headers):
    handler = object.__new__(server.ViewerHandler)
    handler.command = "POST"
    handler._body_read_attempted = False
    handler.headers = Message()
    for key, value in headers:
        handler.headers[key] = value
    return handler


@pytest.mark.parametrize(
    "headers",
    [
        [("Content-Length", "2"), ("Content-Length", "2")],
        [("Content-Length", "2"), ("Transfer-Encoding", "chunked")],
        [("Content-Length", "-1")],
        [("Content-Length", "999999999999999999999999")],
    ],
)
def test_cleanup_never_follows_ambiguous_or_unbounded_framing(headers):
    handler = _handler(headers)
    # Accessing connection/read state would fail: framing must reject first.
    handler._discard_rejected_body()


def test_cleanup_has_one_absolute_deadline_even_when_reads_keep_returning_bytes(monkeypatch):
    handler = _handler([("Content-Length", "100")])
    times = iter([0.0, 0.1, 0.2, 0.26])
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: next(times)))
    timeouts, reads = [], []
    handler.connection = SimpleNamespace(
        gettimeout=lambda: 15, settimeout=lambda value: timeouts.append(value)
    )

    def read(count):
        reads.append(count)
        return b"x"

    handler.rfile = SimpleNamespace(read1=read)
    handler._discard_rejected_body()
    assert len(reads) == 2
    assert timeouts[-1] == 15
    assert max(timeouts[:-1]) <= 0.25


def test_cleanup_discards_at_most_64_kib_without_allocating_the_declared_body():
    handler = _handler([("Content-Length", "999999999")])
    requested = []
    handler.connection = SimpleNamespace(gettimeout=lambda: 15, settimeout=lambda _value: None)

    def read(count):
        requested.append(count)
        return b"x" * count

    handler.rfile = SimpleNamespace(read1=read)
    handler._discard_rejected_body()
    assert sum(requested) == 64 * 1024
    assert max(requested) <= 8192


def test_cleanup_does_not_attempt_to_read_an_already_consumed_or_timed_out_body():
    handler = _handler([("Content-Length", "100")])
    handler._body_read_attempted = True
    handler._discard_rejected_body()
