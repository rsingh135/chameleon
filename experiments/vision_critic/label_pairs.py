"""Review source-aware A/B pairs locally and append human labels durably.

python -m experiments.vision_critic.label_pairs --collection DIR --labels FILE \
    --annotator NAME --port 8770

Only collection PNGs and this tool's static UI are served. All-valid pairs are
shadow research comparisons; they do not widen the backend's admitted choices.
One server owns a label ledger at a time. A crash can leave its .lock file;
inspect the stopped process before removing that stale lock manually.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import secrets
import stat
import sys
import threading
import time
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .contracts import ISSUE_CODES, ValidationError, load_collection, make_pair_id, validate_label

STATIC_DIR = Path(__file__).with_name("static")
MAX_BODY = 32 * 1024
BODY_READ_SECONDS = 2.0
MAX_ASSET = 64 * 1024 * 1024
MAX_PAIRS = 10000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ConflictError(ValueError):
    """A pair was already labelled or its immutable collection changed."""


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _safe_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValidationError("Symlinks and reparse points are not allowed")
    return path


def _pair_key(label: dict) -> tuple:
    return (
        label["collection_id"],
        label["case_id"],
        *sorted((label["candidate_a_id"], label["candidate_b_id"])),
        label["annotator"],
    )


def _eligible(candidate: dict) -> bool:
    diagnostic = candidate["diagnostics"]
    return diagnostic.get("contract_valid") is True and diagnostic.get("contract_violations") == []


def make_pairs(
    collection: dict,
    seed: int,
    pair_pool: str,
    *,
    allow_synthetic: bool = False,
    include_test: bool = False,
) -> list[dict]:
    """Stable shuffle with alternating canonical orientation; never auto-label."""
    if pair_pool not in {"all-valid", "admitted"}:
        raise ValidationError("Pair pool must be all-valid or admitted")
    candidates = []
    for case in collection["cases"]:
        if case["split"] == "test" and not include_test:
            continue
        if not case["source_parity_verified"] and not (
            allow_synthetic and collection["purpose"] == "synthetic"
        ):
            continue
        pool = [
            item
            for item in case["candidates"]
            if _eligible(item) and (pair_pool == "all-valid" or item["admitted"])
        ]
        for first, second in itertools.combinations(
            sorted(pool, key=lambda item: item["candidate_id"]), 2
        ):
            key = [seed, case["case_id"], first["candidate_id"], second["candidate_id"]]
            candidates.append((hashlib.sha256(_json_bytes(key)).hexdigest(), case, first, second))
            if len(candidates) > MAX_PAIRS:
                raise ValidationError("Collection exceeds the 10,000-pair annotation limit")
    result = []
    for index, (_, case, first, second) in enumerate(sorted(candidates, key=lambda item: item[0])):
        if (index + seed) % 2:
            first, second = second, first
        result.append(
            {
                "case": case,
                "a": first,
                "b": second,
                "pair_id": make_pair_id(
                    collection["collection_id"],
                    case["case_id"],
                    first["candidate_id"],
                    second["candidate_id"],
                ),
            }
        )
    return result


class AnnotationStore:
    def __init__(
        self,
        collection: Path,
        labels: Path,
        annotator: str,
        *,
        seed: int = 0,
        pair_pool: str = "all-valid",
        allow_synthetic: bool = False,
        include_test: bool = False,
    ):
        self.root = _safe_path(collection)
        self.labels_path = _safe_path(labels)
        annotator = annotator.strip()
        if not annotator or len(annotator) > 80 or any(ord(char) < 32 for char in annotator):
            raise ValidationError("An annotator name of 1–80 printable characters is required")
        if self.labels_path.is_relative_to(self.root):
            raise ValidationError("Keep the labels file outside the immutable collection directory")
        self.annotator = annotator
        self.collection = load_collection(self.root, verify_files=True)
        if self.collection["purpose"] == "synthetic" and not allow_synthetic:
            raise ValidationError("Synthetic collections require explicit --allow-synthetic")
        self.allow_synthetic = allow_synthetic
        self.include_test = include_test
        self.collection_digest = hashlib.sha256(_json_bytes(self.collection)).hexdigest()
        self.pair_pool = pair_pool
        self.pairs = make_pairs(
            self.collection,
            seed,
            pair_pool,
            allow_synthetic=allow_synthetic,
            include_test=include_test,
        )
        self.by_pair = {pair["pair_id"]: pair for pair in self.pairs}
        self.csrf_token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.records = []
        self.completed = set()
        self.assets = {}
        self._ledger_fd = None
        self._ledger_lock = self.labels_path.with_name(self.labels_path.name + ".lock")
        self._write_failed = False
        self._closed = False
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_path(self.labels_path.parent)
        try:
            self._ledger_fd = os.open(
                self._ledger_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as error:
            raise ValidationError(
                "This label ledger is locked by another or interrupted session"
            ) from error
        try:
            os.write(self._ledger_fd, _json_bytes({"pid": os.getpid()}))
            os.fsync(self._ledger_fd)
            self._load_labels()
            for case in self.collection["cases"]:
                self._asset(case["source"]["render"], case["source"]["render_sha256"])
                for candidate in case["candidates"]:
                    self._asset(candidate["render"], candidate["render_sha256"])
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        # A request may still be appending when the threaded server shuts down.
        # Keep ownership of the ledger until that write has completed, and never
        # let a queued request append after another store can acquire the lock.
        with self.lock:
            self._closed = True
            if self._ledger_fd is not None:
                descriptor, self._ledger_fd = self._ledger_fd, None
                identity = os.fstat(descriptor)
                os.close(descriptor)
                try:
                    current = self._ledger_lock.lstat()
                    if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                        self._ledger_lock.unlink()
                except FileNotFoundError:
                    pass

    def _load_labels(self) -> None:
        if not self.labels_path.exists():
            return
        _safe_path(self.labels_path)
        with self.labels_path.open("rb") as stream:
            while line := stream.readline(MAX_BODY + 1):
                if len(line) > MAX_BODY or not line.endswith(b"\n"):
                    raise ValidationError("Label ledger contains a partial or oversized line")
                label = validate_label(
                    json.loads(line, object_pairs_hook=_unique_keys),
                    self.collection,
                    allow_synthetic=self.allow_synthetic,
                )
                key = _pair_key(label)
                if key in self.completed:
                    raise ValidationError(
                        "Label ledger contains duplicate/conflicting physical pairs"
                    )
                self.records.append(label)
                self.completed.add(key)

    def _asset(self, relative: str, expected_hash: str) -> str:
        token = hashlib.sha256(_json_bytes([relative, expected_hash])).hexdigest()
        self.assets[token] = (relative, expected_hash)
        return "/assets/" + token + ".png"

    def asset_bytes(self, token: str) -> bytes:
        if token not in self.assets:
            raise FileNotFoundError("Collection asset not found")
        relative, expected_hash = self.assets[token]
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
            raise ValidationError("Unsafe asset path")
        path = _safe_path(self.root / relative)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValidationError("Asset must remain inside the collection")
        if path.stat().st_size > MAX_ASSET:
            raise ValidationError("PNG exceeds the 64 MiB asset limit")
        data = path.read_bytes()
        if len(data) > MAX_ASSET or not data.startswith(PNG_SIGNATURE):
            raise ValidationError("Collection render is not a bounded PNG")
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ConflictError("Collection artwork changed; reload a verified collection")
        return data

    def _context(self, pair: dict) -> dict:
        case, first, second = pair["case"], pair["a"], pair["b"]
        return {
            "schema_version": 1,
            "pair_id": pair["pair_id"],
            "case_id": case["case_id"],
            "collection_id": self.collection["collection_id"],
            "candidate_a_id": first["candidate_id"],
            "candidate_b_id": second["candidate_id"],
            "annotator": self.annotator,
            "source_render_sha256": case["source"]["render_sha256"],
            "a_render_sha256": first["render_sha256"],
            "b_render_sha256": second["render_sha256"],
        }

    def next_pair(self) -> dict:
        with self.lock:
            for pair in self.pairs:
                context = self._context(pair)
                if _pair_key(context) in self.completed:
                    continue
                case, first, second = pair["case"], pair["a"], pair["b"]
                return {
                    "done": False,
                    "label_context": context,
                    "target_canvas": case["target_canvas"],
                    "source_url": self._asset(
                        case["source"]["render"], case["source"]["render_sha256"]
                    ),
                    "a_url": self._asset(first["render"], first["render_sha256"]),
                    "b_url": self._asset(second["render"], second["render_sha256"]),
                    "element_ids": {
                        "A": first["provenance"].get("element_ids", []),
                        "B": second["provenance"].get("element_ids", []),
                    },
                }
            return {"done": True}

    def status(self) -> dict:
        with self.lock:
            complete = sum(_pair_key(self._context(pair)) in self.completed for pair in self.pairs)
            return {
                "annotator": self.annotator,
                "purpose": self.collection["purpose"],
                "pair_pool": self.pair_pool,
                "include_test": self.include_test,
                "held_out_case_count": sum(
                    case["split"] == "test" for case in self.collection["cases"]
                )
                if not self.include_test
                else 0,
                "total": len(self.pairs),
                "completed": complete,
                "remaining": len(self.pairs) - complete,
                "csrf_token": self.csrf_token,
                "issue_codes": list(ISSUE_CODES),
            }

    def _append(self, label: dict) -> None:
        data = _json_bytes(label) + b"\n"
        if len(data) > MAX_BODY:
            raise ValidationError("Label exceeds the 32 KiB ledger limit")
        _safe_path(self.labels_path)
        descriptor = os.open(
            self.labels_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0),
            0o600,
        )
        original_size = os.fstat(descriptor).st_size
        try:
            pending = memoryview(data)
            while pending:
                written = os.write(descriptor, pending)
                if not written:
                    raise OSError("Label write made no progress")
                pending = pending[written:]
            os.fsync(descriptor)
            if os.name == "posix":
                directory_fd = os.open(
                    self.labels_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            try:
                os.ftruncate(descriptor, original_size)
                os.fsync(descriptor)
            except OSError:
                self._write_failed = True
            raise
        finally:
            os.close(descriptor)

    def submit(self, payload: dict) -> dict:
        with self.lock:
            if self._closed:
                raise OSError("Review session is closed; reopen it before saving feedback")
            if self._write_failed:
                raise OSError("Ledger recovery is required before another label can be saved")
            try:
                fresh = load_collection(self.root, verify_files=True)
            except (OSError, ValueError) as error:
                raise ConflictError("Collection artifacts changed or cannot be verified") from error
            if hashlib.sha256(_json_bytes(fresh)).hexdigest() != self.collection_digest:
                raise ConflictError("Collection metadata changed; start a new verified session")
            label = validate_label(payload, self.collection, allow_synthetic=self.allow_synthetic)
            if label["annotator"] != self.annotator:
                raise ValidationError("Label annotator must match this review session")
            pair = self.by_pair.get(label["pair_id"])
            if pair is None or any(
                label[key] != value for key, value in self._context(pair).items()
            ):
                raise ValidationError("Label does not belong to this session's pair pool")
            key = _pair_key(label)
            if key in self.completed:
                raise ConflictError(
                    "This pair is already labelled; existing feedback was preserved"
                )
            self._append(label)
            self.records.append(label)
            self.completed.add(key)
            return {"saved": True, "pair_id": label["pair_id"]}


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, store: AnnotationStore, port: int):
        self.store = store
        super().__init__(("127.0.0.1", port), AnnotationHandler)

    def server_close(self) -> None:
        super().server_close()
        self.store.close()


class AnnotationHandler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)
        self._body_consumed = False

    def _framed_length(self) -> int | None:
        lengths = self.headers.get_all("Content-Length", [])
        if (
            len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
            or len(lengths[0]) > 10
            or self.headers.get("Transfer-Encoding")
        ):
            return None
        return int(lengths[0])

    def _read_body(self, length: int) -> bytes:
        """Read a bounded body under one absolute deadline, including trickle reads."""
        self._body_consumed = True
        deadline = time.monotonic() + BODY_READ_SECONDS
        original_timeout = self.connection.gettimeout()
        body = bytearray()
        try:
            while len(body) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Label body read deadline reached")
                self.connection.settimeout(remaining)
                chunk = self.rfile.read1(min(8192, length - len(body)))
                if not chunk:
                    raise ValidationError("Incomplete label body")
                body.extend(chunk)
        finally:
            self.connection.settimeout(original_timeout)
        return bytes(body)

    def _discard_rejected_body(self) -> None:
        # Closing HTTP/1.0 with ordinary unread POST bytes can reset Windows TCP
        # before the client receives our 403/415 JSON. Never parse or save them.
        if self.command != "POST" or self._body_consumed:
            return
        length = self._framed_length()
        if length is None:
            return  # Ambiguous framing is never followed or read indefinitely.
        # Preserve the original rejection after the bounded drain.
        with suppress(OSError, ValidationError):
            self._read_body(min(length, MAX_BODY + 1))

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self._discard_rejected_body()
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
                "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            ),
        )
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, value: dict) -> None:
        self._send(status, _json_bytes(value), "application/json; charset=utf-8")

    def _host(self) -> bool:
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        supplied = self.headers.get_all("Host", [])
        if len(supplied) != 1 or supplied[0] not in hosts:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Use this loopback server's address"})
            return False
        return True

    def do_GET(self) -> None:
        if not self._host():
            return
        route = unquote(urlsplit(self.path).path)
        if ".." in route or "\\" in route or "\0" in route:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})
            return
        static = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/static/style.css": ("style.css", "text/css; charset=utf-8"),
            "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        try:
            if route == "/api/pair":
                self._json(HTTPStatus.OK, self.server.store.next_pair())
            elif route == "/api/status":
                self._json(HTTPStatus.OK, self.server.store.status())
            elif route in static:
                name, kind = static[route]
                self._send(HTTPStatus.OK, (STATIC_DIR / name).read_bytes(), kind)
            elif route.startswith("/assets/") and route.endswith(".png"):
                self._send(HTTPStatus.OK, self.server.store.asset_bytes(route[8:-4]), "image/png")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})
        except FileNotFoundError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Collection asset not found"})
        except ConflictError as error:
            self._json(HTTPStatus.CONFLICT, {"error": str(error)})
        except (OSError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:
        if not self._host():
            return
        if urlsplit(self.path).path != "/api/label":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})
            return
        origin = "http://" + self.headers["Host"]
        token = self.headers.get("X-CSRF-Token", "")
        if (
            self.headers.get_all("Origin") != [origin]
            or len(self.headers.get_all("X-CSRF-Token", [])) != 1
            or not token.isascii()
            or len(token) > 256
            or not secrets.compare_digest(token, self.server.store.csrf_token)
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "Same-origin review session required"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Use application/json"})
            return
        length = self._framed_length()
        if length is None:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "One valid Content-Length is required"})
            return
        if not 0 < length <= MAX_BODY:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Label body must be 1–32768 bytes"}
            )
            return
        try:
            body = self._read_body(length)
            payload = json.loads(body, object_pairs_hook=_unique_keys)
            self._json(HTTPStatus.CREATED, self.server.store.submit(payload))
        except ConflictError as error:
            self._json(HTTPStatus.CONFLICT, {"error": str(error)})
        except (ValueError, UnicodeError, RecursionError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except TimeoutError:
            self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "Label body read deadline reached"})
        except OSError:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Label could not be saved durably; feedback was not confirmed"},
            )

    def do_OPTIONS(self) -> None:
        self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "Cross-origin access is disabled"})


def create_server(
    collection: Path,
    labels: Path,
    annotator: str,
    *,
    port: int = 8770,
    seed: int = 0,
    pair_pool: str = "all-valid",
    allow_synthetic: bool = False,
    include_test: bool = False,
) -> AnnotationServer:
    store = AnnotationStore(
        collection,
        labels,
        annotator,
        seed=seed,
        pair_pool=pair_pool,
        allow_synthetic=allow_synthetic,
        include_test=include_test,
    )
    try:
        return AnnotationServer(store, port)
    except BaseException:
        store.close()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-pool", choices=("all-valid", "admitted"), default="all-valid")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic infrastructure practice; never task-training approval",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unblind test review after freezing the training recipe",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        server = create_server(
            args.collection,
            args.labels,
            args.annotator,
            port=args.port,
            seed=args.seed,
            pair_pool=args.pair_pool,
            allow_synthetic=args.allow_synthetic,
            include_test=args.include_test,
        )
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Human review: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
