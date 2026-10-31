"""Bounded caches for completed local resize evidence, never native objects.

Callers validate requests, hold the renderer/engine lock around cache lookup and
computation, and bind resources before and after work. A second lookup inside
that lock coalesces concurrent identical misses. The execution identity must
describe the loaded, frozen implementation/runtime; changed code requires a
restart, not merely a new key for code that has not been loaded.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import stat
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_CACHE_BYTES = 16 * 1024 * 1024
_PNG_PREFIX = "data:image/png;base64,"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_END = b"\x00\x00\x00\x00IEND\xaeB`\x82"
_MISSING_IMAGE = "missing-image"
_REPORT_OBJECTS = (
    "metrics",
    "style",
    "relationships",
    "proportions",
    "focal",
    "rules",
    "reading_order",
    "occlusion",
    "layout_state",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class ResourceChangedError(RuntimeError):
    """A resource path has changed since native objects could have captured it."""


@dataclass(frozen=True)
class ResourceIdentity:
    assets_dir: str
    fonts_dir: str
    digest: str
    _hashes_json: bytes = field(repr=False)
    missing_assets: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing_assets

    def hashes(self) -> dict:
        """Return a fresh receipt; callers cannot mutate the captured identity."""
        return json.loads(self._hashes_json)


class ResourceBindings:
    """Pin observed file bytes without confusing different referenced subsets.

    Font inventories are pinned per root because font managers load whole roots.
    Asset identities are pinned per file, so two sources may use disjoint image
    subsets in the same directory. Pins are never evicted: on exhaustion the
    process must restart rather than forget an old native resource's identity.
    """

    def __init__(self, *, max_paths: int = 16384, max_font_roots: int = 128):
        if max_paths < 1 or max_font_roots < 1:
            raise ValueError("Resource binding limits must be positive")
        self._max_paths = max_paths
        self._max_font_roots = max_font_roots
        self._paths: dict[Path, str] = {}
        self._font_inventories: dict[Path, tuple[str, ...]] = {}
        self._lock = threading.RLock()

    def bind(
        self,
        source: dict,
        assets_dir: str | Path,
        fonts_dir: str | Path,
        *,
        allow_missing: bool = False,
    ) -> ResourceIdentity:
        from tools.scene_harness import resource_hashes

        assets = Path(assets_dir).resolve()
        fonts = Path(fonts_dir).resolve()
        with self._lock:
            image_paths = {}
            missing = set()
            for element in source["elements"]:
                if element["type"] != "image":
                    continue
                name = element.get("asset_id")
                logical, resolved = _image_paths(assets, name)
                image_paths[name] = logical, resolved
                try:
                    info = resolved.stat()
                except FileNotFoundError:
                    missing.add(name)
                else:
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError(
                            "Image assets must be regular files inside their directory"
                        )
            checked = source
            if allow_missing and missing:
                checked = {
                    **source,
                    "elements": [
                        element
                        for element in source["elements"]
                        if element["type"] != "image" or element["asset_id"] not in missing
                    ],
                }
            # Missing fonts/directories and all remaining resources stay strict.
            hashes = resource_hashes(checked, assets, fonts)
            if allow_missing:
                hashes["assets"].update({name: _MISSING_IMAGE for name in missing})
            inventory = tuple(sorted(hashes["fonts"]))
            previous = self._font_inventories.get(fonts)
            if previous is not None and previous != inventory:
                raise ResourceChangedError(
                    "Font inventory changed. Restart the server or use a new immutable package."
                )
            paths = {}
            for name, digest in hashes["assets"].items():
                logical, resolved = image_paths[name]
                # Preserve the logical alias as well: a missing path later
                # appearing as an in-root symlink must not evade its old pin.
                paths[logical] = paths[resolved] = digest
            paths.update(
                {
                    (fonts / relative).resolve(): digest
                    for relative, digest in hashes["fonts"].items()
                }
            )
            for path, digest in paths.items():
                if path in self._paths and self._paths[path] != digest:
                    raise ResourceChangedError(
                        "Resource bytes changed at a previously loaded path. "
                        "Restart the server or use a new immutable package."
                    )
            if (
                len(self._paths.keys() | paths.keys()) > self._max_paths
                or previous is None
                and len(self._font_inventories) >= self._max_font_roots
            ):
                raise ResourceChangedError("Resource identity budget is full. Restart the server.")
            self._paths.update(paths)
            self._font_inventories[fonts] = inventory
        identity = {"assets_dir": str(assets), "fonts_dir": str(fonts), "hashes": hashes}
        return ResourceIdentity(
            str(assets), str(fonts), _digest(identity), _canonical(hashes), tuple(sorted(missing))
        )


def _image_paths(root: Path, name: object) -> tuple[Path, Path]:
    """Reject unsafe names even when the final image does not exist yet."""
    if (
        not isinstance(name, str)
        or not name
        or any(ord(char) < 32 for char in name)
        or ":" in name
        or PureWindowsPath(name).drive
        or PureWindowsPath(name).root
    ):
        raise ValueError("Image assets require safe relative paths")
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    if PurePosixPath(normalized).is_absolute() or any(
        part in {"", ".", ".."} or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
        for part in parts
    ):
        raise ValueError("Image asset traversal and absolute paths are not allowed")
    logical = root.joinpath(*parts)
    try:
        resolved = logical.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("Image asset path cannot be resolved safely") from exc
    if not resolved.is_relative_to(root):
        raise ValueError("Image asset path leaves its resource directory")
    return logical, resolved


def _content_key(source, target, resources, execution_identity, attribution, kind):
    if not isinstance(resources, ResourceIdentity):
        raise TypeError("A bound resource identity is required")
    if not execution_identity:
        raise ValueError("The frozen execution identity is required")
    return _digest(
        {
            "protocol": "viewer-resize-cache-v1",
            "kind": kind,
            "source": source,
            "target": target,
            "attribution": attribution,
            "resources": {
                "assets_dir": resources.assets_dir,
                "fonts_dir": resources.fonts_dir,
                "digest": resources.digest,
            },
            "execution": execution_identity,
        }
    )


def source_key(source, *, resources, execution_identity, attribution=None) -> str:
    """Source rendering is target-independent, but includes PNG attribution."""
    return _content_key(source, None, resources, execution_identity, attribution, "source")


def result_key(
    source,
    target,
    *,
    resources,
    execution_identity,
    attribution=None,
    model_enabled=False,
) -> str | None:
    """Model-enabled requests bypass result reuse until their identity is frozen."""
    if model_enabled or isinstance(resources, ResourceIdentity) and not resources.complete:
        return None
    return _content_key(source, target, resources, execution_identity, attribution, "result")


def _warnings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _png(value):
    if not isinstance(value, str) or not value.startswith(_PNG_PREFIX):
        return False
    try:
        data = base64.b64decode(value[len(_PNG_PREFIX) :], validate=True)
    except (ValueError, binascii.Error):
        return False
    # Admission is only for the native renderer's completed outputs. This
    # framing check rejects truncated values; it does not replace PNG decoding.
    return len(data) >= 45 and data.startswith(_PNG_SIGNATURE) and data.endswith(_PNG_END)


def _successful(value, kind):
    if not isinstance(value, dict) or "error" in value:
        return False
    if value.get("status") in ("error", "failed", "interrupted", "running"):
        return False
    if kind == "source":
        return _png(value.get("png")) and _warnings(value.get("warnings"))
    design, report = value.get("design"), value.get("report")
    if not isinstance(design, dict) or not isinstance(report, dict):
        return False
    canvas = design.get("canvas")
    return (
        isinstance(canvas, dict)
        and all(type(canvas.get(key)) is int and canvas[key] > 0 for key in ("width", "height"))
        and isinstance(design.get("elements"), list)
        and bool(design["elements"])
        and report.get("violations") == []
        and all(isinstance(report.get(key), dict) for key in _REPORT_OBJECTS)
        and type(report.get("total")) in (int, float)
        and _png(value.get("source_png"))
        and _png(value.get("output_png"))
        and _warnings(value.get("warnings"))
    )


class SerializedCache:
    """A byte- and entry-bounded LRU holding only complete serialized successes.

    No caller-owned dictionary, image, font manager or renderer is retained.
    Each get decodes a new JSON value. Compression is useful for base64 PNGs;
    the uncompressed per-entry bound also limits decode allocations on a hit.
    Two default instances retain at most 32 MiB of encoded entries in total.
    """

    def __init__(
        self,
        *,
        kind: str,
        max_bytes: int = DEFAULT_CACHE_BYTES,
        max_entries: int = 16,
        compress: bool = True,
        max_entry_bytes: int | None = None,
    ):
        if kind not in {"source", "result"}:
            raise ValueError("Cache kind must be source or result")
        if max_bytes < 1 or max_entries < 1:
            raise ValueError("Cache limits must be positive")
        if max_entry_bytes is not None and max_entry_bytes < 1:
            raise ValueError("Entry limit must be positive")
        self._kind = kind
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._max_entry_bytes = max_entry_bytes or max_bytes
        self._compress = compress
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._stored_bytes = 0
        self._hits = self._misses = 0
        self._lock = threading.RLock()

    def get(self, key: str | None) -> dict | None:
        with self._lock:
            data = self._entries.get(key) if key is not None else None
            if data is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
        return json.loads(zlib.decompress(data) if self._compress else data)

    def put_success(self, key: str | None, value: object) -> bool:
        if not isinstance(key, str) or not key:
            return False
        if isinstance(value, dict) and self._kind == "result":
            value = {name: item for name, item in value.items() if name != "elapsed_ms"}
        try:
            raw = _canonical(value)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return False
        if len(raw) > self._max_entry_bytes or not _successful(value, self._kind):
            return False
        data = zlib.compress(raw, level=1) if self._compress else raw
        if len(data) > self._max_bytes:
            return False
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._stored_bytes -= len(old)
            while self._entries and (
                len(self._entries) >= self._max_entries
                or self._stored_bytes + len(data) > self._max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._stored_bytes -= len(evicted)
            self._entries[key] = data
            self._stored_bytes += len(data)
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._stored_bytes = 0

    def info(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "stored_bytes": self._stored_bytes,
                "max_bytes": self._max_bytes,
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }
