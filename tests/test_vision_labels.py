"""Human annotation HTTP, persistence and integrity checks without models or networks."""

import copy
import hashlib
import http.client
import json
import shutil
import struct
import subprocess
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.vision_critic import label_pairs as ui
from experiments.vision_critic.contracts import canonical_json_sha256, load_collection

UTC = timezone.utc  # noqa: UP017 - local WSL execution supports Python 3.10


def png(width, height, color):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    pixels = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    return (
        ui.PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def collection(tmp_path):
    root = tmp_path / "collection"
    root.mkdir()
    source = {"canvas": {"width": 20, "height": 30}, "elements": [{"id": "headline"}]}
    (root / "source.json").write_text(json.dumps(source), encoding="utf-8")
    (root / "source.png").write_bytes(png(20, 30, (252, 248, 235)))
    candidates = []
    for index in range(4):
        design = {
            "canvas": {"width": 32, "height": 18},
            "elements": [{"id": "headline", "x": index}],
        }
        design_path, image_path = f"candidate-{index}.json", f"candidate-{index}.png"
        (root / design_path).write_text(json.dumps(design), encoding="utf-8")
        image_bytes = png(32, 18, (40 + index * 40, 80, 120))
        (root / image_path).write_bytes(image_bytes)
        candidates.append(
            {
                "candidate_id": f"candidate-{index}",
                "design": design_path,
                "design_sha256": canonical_json_sha256(design),
                "render": image_path,
                "render_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "admitted": index < 2,
                "diagnostics": {
                    "contract_valid": index < 3,
                    "contract_violations": [] if index < 3 else ["off_canvas"],
                },
                "provenance": {
                    "element_ids": ["headline"],
                    "family_name": "hidden-candidate-family",
                },
            }
        )
    data = {
        "schema_version": 1,
        "collection_id": "collection-test",
        "purpose": "research",
        "provenance": {
            "engine_revision": "test-only",
            "working_tree_digest": "0" * 64,
            "policy_version": "test-only",
        },
        "cases": [
            {
                "case_id": "case-one",
                "source_id": "hidden-source-name",
                "family_id": "hidden-source-family",
                "split": "train",
                "source_parity_verified": True,
                "training_rights_reviewed": False,
                "rights_reference": "",
                "target_canvas": {"width": 32, "height": 18},
                "source": {
                    "design": "source.json",
                    "json_sha256": canonical_json_sha256(source),
                    "render": "source.png",
                    "render_sha256": hashlib.sha256((root / "source.png").read_bytes()).hexdigest(),
                },
                "baseline_candidate_id": "candidate-0",
                "admitted_candidate_ids": ["candidate-0", "candidate-1"],
                "candidates": candidates,
            }
        ],
    }
    (root / "collection.json").write_text(json.dumps(data), encoding="utf-8")
    return root


@pytest.fixture
def http_server(collection, tmp_path):
    server = ui.create_server(collection, tmp_path / "labels.jsonl", "Reviewer", port=0)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def request(server, method, path, value=None, headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    try:
        body = (
            raw if raw is not None else (json.dumps(value).encode() if value is not None else None)
        )
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def get_json(server, path):
    status, _, body = request(server, "GET", path)
    assert status == 200
    return json.loads(body)


def label_from(pair):
    return {
        **pair["label_context"],
        "decision": "A",
        "issues": [],
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_headers(server):
    return {
        "Content-Type": "application/json",
        "Origin": f"http://127.0.0.1:{server.server_port}",
        "X-CSRF-Token": server.store.csrf_token,
    }


def test_http_review_serves_real_images_and_blinds_family_and_baseline(http_server):
    status, headers, body = request(http_server, "GET", "/")
    assert status == 200 and b"Which adaptation works better?" in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    pair = get_json(http_server, "/api/pair")
    encoded = json.dumps(pair)
    assert not any(
        word in encoded
        for word in ["hidden-source", "hidden-candidate", "baseline_candidate", "admitted"]
    )
    assert pair["target_canvas"] == {"width": 32, "height": 18}
    for key in ("source_url", "a_url", "b_url"):
        status, headers, data = request(http_server, "GET", pair[key])
        assert status == 200 and headers["Content-Type"] == "image/png"
        assert data.startswith(ui.PNG_SIGNATURE)
    source_data = request(http_server, "GET", pair["source_url"])[2]
    assert struct.unpack(">II", source_data[16:24]) == (20, 30)
    status = get_json(http_server, "/api/status")
    assert status["total"] == 3 and status["completed"] == 0


def test_http_saves_once_and_preserves_conflicting_feedback(http_server):
    pair = get_json(http_server, "/api/pair")
    label = label_from(pair)
    status, _, data = request(http_server, "POST", "/api/label", label, write_headers(http_server))
    assert status == 201 and json.loads(data)["saved"]
    written = http_server.store.labels_path.read_bytes()
    assert written.endswith(b"\n") and json.loads(written)["decision"] == "A"
    assert get_json(http_server, "/api/status")["completed"] == 1
    assert (
        get_json(http_server, "/api/pair")["label_context"]["pair_id"]
        != pair["label_context"]["pair_id"]
    )
    label["decision"] = "B"
    assert request(http_server, "POST", "/api/label", label, write_headers(http_server))[0] == 409
    assert http_server.store.labels_path.read_bytes() == written


def test_resume_skips_physical_pair_even_when_seed_changes_orientation(collection, tmp_path):
    ledger = tmp_path / "labels.jsonl"
    store = ui.AnnotationStore(collection, ledger, "Reviewer", seed=0)
    chosen = store.next_pair()
    store.submit(label_from(chosen))
    first_key = ui._pair_key(chosen["label_context"])
    store.close()
    resumed = ui.AnnotationStore(collection, ledger, "Reviewer", seed=77)
    try:
        assert resumed.status()["completed"] == 1
        assert ui._pair_key(resumed.next_pair()["label_context"]) != first_key
    finally:
        resumed.close()
    other = ui.AnnotationStore(collection, ledger, "Another reviewer")
    try:
        assert other.status()["completed"] == 0
    finally:
        other.close()


def test_pair_pool_is_deterministic_counterbalanced_and_excludes_contract_failures(collection):
    data = load_collection(collection)
    pairs = ui.make_pairs(data, 9, "all-valid")
    assert pairs == ui.make_pairs(data, 9, "all-valid")
    assert len(pairs) == 3 and len(ui.make_pairs(data, 9, "admitted")) == 1
    assert all(
        "candidate-3" not in {pair["a"]["candidate_id"], pair["b"]["candidate_id"]}
        for pair in pairs
    )
    forward = sum(pair["a"]["candidate_id"] < pair["b"]["candidate_id"] for pair in pairs)
    assert abs(forward - (len(pairs) - forward)) <= 1


@pytest.mark.parametrize("change", ["origin", "token", "missing_origin", "host", "content_type"])
def test_write_protections_reject_cross_origin_and_unverified_requests(http_server, change):
    label = label_from(get_json(http_server, "/api/pair"))
    headers = write_headers(http_server)
    if change == "origin":
        headers["Origin"] = "https://evil.example"
    elif change == "token":
        headers["X-CSRF-Token"] = "incorrect"
    elif change == "missing_origin":
        headers.pop("Origin")
    elif change == "host":
        headers["Host"] = "evil.example"
    else:
        headers["Content-Type"] = "text/plain"
    # Repeat real writes to expose Windows TCP resets when rejected bodies are
    # left unread. Every request must receive the intended complete JSON error.
    expected = 415 if change == "content_type" else 403
    for _ in range(8):
        status, response_headers, body = request(http_server, "POST", "/api/label", label, headers)
        assert status == expected
        assert json.loads(body)["error"]
        assert response_headers["Connection"] == "close"
    assert not http_server.store.labels_path.exists()


@pytest.mark.parametrize("authorized", [False, True])
def test_incomplete_request_body_has_bounded_json_error_and_never_commits(
    http_server, monkeypatch, authorized
):
    monkeypatch.setattr(ui, "BODY_READ_SECONDS", 0.15)
    headers = write_headers(http_server)
    if not authorized:
        headers["X-CSRF-Token"] = "invalid"
    connection = http.client.HTTPConnection("127.0.0.1", http_server.server_port, timeout=3)
    try:
        connection.putrequest("POST", "/api/label")
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.putheader("Content-Length", "2048")
        connection.endheaders(b"{")
        started = time.monotonic()
        response = connection.getresponse()
        body = response.read()
        assert time.monotonic() - started < 2
        assert response.status == (408 if authorized else 403)
        assert json.loads(body)["error"]
        assert not http_server.store.labels_path.exists()
    finally:
        connection.close()


def test_body_deadline_is_absolute_even_when_reads_keep_returning_bytes(monkeypatch):
    monkeypatch.setattr(ui, "BODY_READ_SECONDS", 0.2)
    times = iter([0.0, 0.1, 0.25])
    monkeypatch.setattr(ui, "time", SimpleNamespace(monotonic=lambda: next(times)))
    handler = object.__new__(ui.AnnotationHandler)

    class Connection:
        def __init__(self):
            self.timeouts = []

        def gettimeout(self):
            return 10

        def settimeout(self, value):
            self.timeouts.append(value)

    class Trickle:
        reads = 0

        def read1(self, count):
            self.reads += 1
            return b"x"

    handler.connection, handler.rfile = Connection(), Trickle()
    with pytest.raises(TimeoutError, match="deadline"):
        handler._read_body(32)
    assert handler.rfile.reads == 1
    assert handler.connection.timeouts[-1] == 10


@pytest.mark.parametrize(
    "path",
    [
        "/assets/../collection.json",
        "/assets/%2e%2e/source.png",
        "/assets/%5c..%5ccollection.json",
        "/assets/http://example.com/image.png",
        "/collection.json",
        "/static/../label_pairs.py",
    ],
)
def test_only_allowlisted_assets_and_static_files_are_served(http_server, path):
    assert request(http_server, "GET", path)[0] == 404


def test_bad_host_cannot_read_csrf_or_status(http_server):
    assert request(http_server, "GET", "/api/status", headers={"Host": "evil.example"})[0] == 403
    assert request(http_server, "OPTIONS", "/api/label")[0] == 405


@pytest.mark.parametrize(
    "bad", ["hash", "extra", "decision", "annotator", "issue_code", "element_id"]
)
def test_shared_contract_rejects_stale_hashes_and_unknown_feedback(http_server, bad):
    label = label_from(get_json(http_server, "/api/pair"))
    if bad == "hash":
        label["a_render_sha256"] = "f" * 64
    elif bad == "extra":
        label["geometry"] = {"x": 5}
    elif bad == "decision":
        label["decision"] = "generate-new-design"
    elif bad == "annotator":
        label["annotator"] = "Someone else"
    else:
        label["issues"] = [
            {
                "candidate": "A",
                "code": "bad_overlap" if bad == "element_id" else "new-code",
                "element_ids": ["unknown"] if bad == "element_id" else [],
            }
        ]
    assert request(http_server, "POST", "/api/label", label, write_headers(http_server))[0] == 400
    assert not http_server.store.labels_path.exists()


def test_changed_artwork_is_not_served_or_labelled_as_the_old_pair(http_server):
    pair = get_json(http_server, "/api/pair")
    token = pair["a_url"].split("/")[-1][:-4]
    relative, _ = http_server.store.assets[token]
    (http_server.store.root / relative).write_bytes(png(32, 18, (255, 0, 0)))
    assert request(http_server, "GET", pair["a_url"])[0] == 409
    assert (
        request(http_server, "POST", "/api/label", label_from(pair), write_headers(http_server))[0]
        == 409
    )
    assert not http_server.store.labels_path.exists()


def test_request_size_and_duplicate_json_are_rejected(http_server):
    headers = write_headers(http_server)
    assert (
        request(http_server, "POST", "/api/label", raw=b"x" * (ui.MAX_BODY + 1), headers=headers)[0]
        == 413
    )
    assert (
        request(
            http_server,
            "POST",
            "/api/label",
            raw=b'{"decision":"A","decision":"B"}',
            headers=headers,
        )[0]
        == 400
    )
    assert not http_server.store.labels_path.exists()


def test_failed_fsync_does_not_claim_progress_or_damage_prior_feedback(http_server, monkeypatch):
    first = label_from(get_json(http_server, "/api/pair"))
    assert request(http_server, "POST", "/api/label", first, write_headers(http_server))[0] == 201
    prior = http_server.store.labels_path.read_bytes()

    def fail(_descriptor):
        raise OSError("injected disk failure")

    monkeypatch.setattr(ui.os, "fsync", fail)
    second = label_from(get_json(http_server, "/api/pair"))
    assert request(http_server, "POST", "/api/label", second, write_headers(http_server))[0] == 500
    assert get_json(http_server, "/api/status")["completed"] == 1
    assert http_server.store.labels_path.read_bytes() == prior


def test_ledger_lock_and_partial_lines_require_explicit_recovery(collection, tmp_path):
    ledger = tmp_path / "labels.jsonl"
    store = ui.AnnotationStore(collection, ledger, "Reviewer")
    try:
        with pytest.raises(ValueError, match="locked"):
            ui.AnnotationStore(collection, ledger, "Reviewer")
    finally:
        store.close()
    ledger.write_bytes(b'{"partial":')
    with pytest.raises(ValueError, match="partial"):
        ui.AnnotationStore(collection, ledger, "Reviewer")
    assert not ledger.with_name(ledger.name + ".lock").exists()


def test_closed_store_cannot_append_after_another_session_acquires_ledger(collection, tmp_path):
    ledger = tmp_path / "labels.jsonl"
    store = ui.AnnotationStore(collection, ledger, "Reviewer")
    label = label_from(store.next_pair())
    store.close()
    replacement = ui.AnnotationStore(collection, ledger, "Reviewer")
    try:
        with pytest.raises(OSError, match="closed"):
            store.submit(label)
        assert not ledger.exists()
        assert replacement.submit(label)["saved"]
        store.close()  # Idempotent cleanup cannot remove the replacement's lock.
        assert replacement._ledger_lock.exists()
    finally:
        replacement.close()


def test_close_keeps_ledger_ownership_until_inflight_append_finishes(
    collection, tmp_path, monkeypatch
):
    ledger = tmp_path / "labels.jsonl"
    store = ui.AnnotationStore(collection, ledger, "Reviewer")
    label = label_from(store.next_pair())
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original_append = store._append
    failures = []

    def append(value):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("test did not release append")
        original_append(value)

    def save():
        try:
            store.submit(label)
        except BaseException as error:
            failures.append(error)

    def close():
        store.close()
        closed.set()

    monkeypatch.setattr(store, "_append", append)
    writer = threading.Thread(target=save)
    closer = threading.Thread(target=close)
    writer.start()
    try:
        assert entered.wait(2)
        closer.start()
        assert not closed.wait(0.1)
        with pytest.raises(ValueError, match="locked"):
            ui.AnnotationStore(collection, ledger, "Reviewer")
    finally:
        release.set()
        writer.join(timeout=3)
        if closer.ident is not None:
            closer.join(timeout=3)
        store.close()
    assert not failures and closed.is_set()
    assert json.loads(ledger.read_text())["pair_id"] == label["pair_id"]
    assert not store._ledger_lock.exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for browser-state tests")
def test_annotation_frontend_async_state_regressions():
    result = subprocess.run(
        [
            shutil.which("node"),
            "--test",
            str(Path(__file__).with_name("vision_labels_ui.test.cjs")),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_research_parity_is_required_and_synthetic_bypass_is_explicit(collection, tmp_path):
    path = collection / "collection.json"
    data = json.loads(path.read_text())
    data["cases"][0]["source_parity_verified"] = False
    path.write_text(json.dumps(data))
    research = ui.AnnotationStore(
        collection, tmp_path / "research.jsonl", "Reviewer", allow_synthetic=True
    )
    try:
        assert research.status()["total"] == 0
    finally:
        research.close()
    data["purpose"] = "synthetic"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="allow-synthetic"):
        ui.AnnotationStore(collection, tmp_path / "synthetic.jsonl", "Reviewer")
    practice = ui.AnnotationStore(
        collection, tmp_path / "synthetic.jsonl", "automated-ui-test", allow_synthetic=True
    )
    try:
        assert practice.status()["purpose"] == "synthetic"
        assert practice.status()["total"] == 3
        assert practice.submit(label_from(practice.next_pair()))["saved"]
        assert practice.collection["cases"][0]["source_parity_verified"] is False
    finally:
        practice.close()


def test_blank_annotator_or_writing_into_collection_is_rejected(collection, tmp_path):
    with pytest.raises(ValueError, match="annotator"):
        ui.AnnotationStore(collection, tmp_path / "labels.jsonl", "  ")
    with pytest.raises(ValueError, match="outside"):
        ui.AnnotationStore(collection, collection / "labels.jsonl", "Reviewer")


def test_test_split_is_concealed_until_explicit_http_unblinding(collection, tmp_path):
    path = collection / "collection.json"
    data = json.loads(path.read_text())
    heldout = copy.deepcopy(data["cases"][0])
    heldout.update(
        case_id="heldout-case", source_id="heldout-source", family_id="heldout-family", split="test"
    )
    source = {
        "canvas": {"width": 20, "height": 30},
        "elements": [{"id": "headline", "content": "Held-out fixture"}],
    }
    source_image = png(20, 30, (170, 190, 210))
    (collection / "heldout-source.json").write_text(json.dumps(source))
    (collection / "heldout-source.png").write_bytes(source_image)
    heldout["source"] = {
        "design": "heldout-source.json",
        "json_sha256": canonical_json_sha256(source),
        "render": "heldout-source.png",
        "render_sha256": hashlib.sha256(source_image).hexdigest(),
    }
    data["cases"].append(heldout)
    path.write_text(json.dumps(data))

    def serve(include_test):
        instance = ui.create_server(
            collection,
            tmp_path / "holdout-labels.jsonl",
            "Reviewer",
            port=0,
            include_test=include_test,
        )
        thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        return instance, thread

    server, worker = serve(False)
    try:
        state = get_json(server, "/api/status")
        assert state["total"] == 3 and state["held_out_case_count"] == 1
        assert state["include_test"] is False
        assert "heldout-source" not in json.dumps(state)
        assert all(pair["case"]["split"] != "test" for pair in server.store.pairs)
        full_pool = ui.make_pairs(server.store.collection, 0, "all-valid", include_test=True)
        hidden = next(pair for pair in full_pool if pair["case"]["split"] == "test")
        fabricated = label_from({"label_context": server.store._context(hidden)})
        assert request(server, "POST", "/api/label", fabricated, write_headers(server))[0] == 400
        assert not server.store.labels_path.exists()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)

    server, worker = serve(True)
    try:
        state = get_json(server, "/api/status")
        assert state["total"] == 6 and state["held_out_case_count"] == 0
        assert state["include_test"] is True
        for _ in range(6):
            pair = get_json(server, "/api/pair")
            assert (
                request(server, "POST", "/api/label", label_from(pair), write_headers(server))[0]
                == 201
            )
            if pair["label_context"]["case_id"] == "heldout-case":
                break
        else:
            pytest.fail("Explicit test review never exposed the held-out pair")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
