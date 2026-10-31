"""Bounded native-document resource transport for the local Adobe add-on.

The browser sends bytes, never filesystem paths or remote URLs. Each package is
stored in an immutable, content-addressed directory so renderer caches cannot
confuse two documents that use the same asset or font name.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PureWindowsPath

from PIL import Image

from engine.resources import ResourceContext

MAX_DOCUMENT_REQUEST_BYTES = 24 * 1024 * 1024
MAX_RESOURCE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 16 * 1024 * 1024
MAX_RESOURCE_PIXELS = 20_000_000
MAX_CACHE_BYTES = 256 * 1024 * 1024
MAX_CACHE_PACKAGES = 64
DOCUMENT_ORIGIN = "https://localhost:5241"
DOCUMENT_ROOT = Path(__file__).resolve().parents[1] / "output" / "documents"
_WRITE_LOCK = threading.Lock()
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
_PACKAGE_NAME = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PreparedDocument:
    resources: ResourceContext
    receipt: dict


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{name} must be a nonempty string of at most 200 characters.")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must not contain control characters.")
    return value


def _decode(value: object, name: str) -> bytes:
    if not isinstance(value, str) or len(value) > 4 * ((MAX_RESOURCE_BYTES + 2) // 3):
        raise ValueError(f"{name} must contain at most 8 MiB of base64 resource data.")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{name} must be valid base64.") from exc
    if not data or len(data) > MAX_RESOURCE_BYTES:
        raise ValueError(f"{name} must contain between 1 byte and 8 MiB.")
    return data


def _entries(payload: dict, key: str) -> list[dict]:
    entries = payload.get(key, [])
    if not isinstance(entries, list) or len(entries) > 200:
        raise ValueError(f"{key} must contain at most 200 resource objects.")
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{key} must contain resource objects.")
    return entries


def _image(data: bytes, mime: object, name: str) -> None:
    allowed = {"image/png": "PNG", "image/jpeg": "JPEG"}
    if not isinstance(mime, str) or mime not in allowed:
        raise ValueError(f"{name} must be an original PNG or JPEG image.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != allowed[mime]:
                raise ValueError(f"{name} image bytes do not match their MIME type.")
            if image.width * image.height > MAX_RESOURCE_PIXELS:
                raise ValueError(f"{name} exceeds the 20,000,000 pixel image limit.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"{name} must be a still image.")
            image.verify()
    except (OSError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{name} is not a valid supported image.") from exc
    # Header/CRC verification alone does not validate compressed pixel data.
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (OSError, SyntaxError, ValueError, EOFError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{name} is not a valid supported image.") from exc


def _font(data: bytes, expected: str) -> str:
    import skia

    if len(data) < 12 or data[:4] not in {b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"}:
        raise ValueError(f"{expected}: provide the original TrueType or OpenType font file.")
    face = skia.Typeface.MakeFromData(data)
    if face is None or face.getPostScriptName() != expected:
        raise ValueError(f"{expected}: font file PostScript name must match the native document.")
    return ".otf" if data[:4] == b"OTTO" else ".ttf"


def prepare_document(payload: dict, root: Path | None = None) -> PreparedDocument:
    """Validate exact resources and materialize a package after design validation."""
    source = payload["source"]
    elements = source["elements"]
    metadata = payload.get("document")
    if not isinstance(metadata, dict):
        raise ValueError("document must identify the captured native page.")
    receipt = {
        key: _text(metadata.get(key), f"document.{key}")
        for key in ("snapshot_id", "page_id", "artboard_id")
    }
    files: dict[str, bytes] = {}
    assets: dict[str, dict] = {}
    fonts: dict[tuple[str, str, str], dict] = {}
    for entry in _entries(payload, "assets"):
        name = _text(entry.get("asset_id"), "asset_id")
        if (
            not _SAFE_FILENAME.fullmatch(name)
            or name in {".", ".."}
            or name.endswith(".")
            or PureWindowsPath(name).is_reserved()
        ):
            raise ValueError("asset_id must be a simple portable filename.")
        if name.casefold() in {asset.casefold() for asset in assets}:
            raise ValueError(f"Duplicate document asset: {name}")
        data = _decode(entry.get("data_base64"), name)
        _image(data, entry.get("mime_type"), name)
        files[f"assets/{name}"] = data
        assets[name] = {"asset_id": name, "sha256": hashlib.sha256(data).hexdigest()}
    font_manifest = []
    for entry in _entries(payload, "fonts"):
        family = _text(entry.get("font_family"), "font_family")
        postscript = _text(entry.get("postscript_name"), "postscript_name")
        if family != postscript:
            raise ValueError("Document font_family must be the exact native PostScript alias.")
        weight, style = entry.get("weight"), entry.get("style")
        if weight not in ("regular", "bold") or style not in ("normal", "italic"):
            raise ValueError("Document font weight/style must be regular|bold and normal|italic.")
        key = family, weight, style
        if key in fonts:
            raise ValueError(f"Duplicate document font face: {family}")
        data = _decode(entry.get("data_base64"), family)
        suffix = _font(data, postscript)
        digest = hashlib.sha256(data).hexdigest()
        filename = digest + suffix
        files[f"fonts/{filename}"] = data
        item = {"font_family": family, "file": filename, "weight": weight, "style": style}
        font_manifest.append(item)
        fonts[key] = {**item, "postscript_name": postscript, "sha256": digest}
    if sum(len(data) for data in files.values()) > MAX_TOTAL_RESOURCE_BYTES:
        raise ValueError("Document resources must contain at most 16 MiB of decoded data.")
    for element in elements:
        if element["type"] == "image" and element["asset_id"] not in assets:
            raise ValueError(f"Missing native image bytes: {element['asset_id']}")
        if element["type"] == "text":
            key = (
                element["font_family"],
                element.get("font_weight", "regular"),
                element.get("font_style", "normal"),
            )
            if key not in fonts:
                raise ValueError(f"Missing exact native font bytes: {key[0]}")
            for run in element.get("style_runs", []):
                if run["end"] > len(element["content"]):
                    raise ValueError("Native style_runs ranges must be inside the text content.")
                if any(
                    run.get(k, element.get(k, default)) != element.get(k, default)
                    for k, default in (("font_weight", "regular"), ("font_style", "normal"))
                ):
                    raise ValueError("Native mixed font faces are outside the supported renderer.")
    font_manifest.sort(key=lambda item: (item["font_family"], item["weight"], item["style"]))
    files["fonts/font-manifest.json"] = json.dumps(font_manifest, sort_keys=True).encode("utf-8")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    package_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    with _WRITE_LOCK:
        # First imports can create this root concurrently. Resolve and verify
        # it under the same lock as creation: Windows path resolution while a
        # sibling thread creates the missing parents can transiently disagree.
        package_root = (DOCUMENT_ROOT if root is None else Path(root)).resolve()
        directory = package_root / package_digest
        if not directory.resolve().is_relative_to(package_root):
            raise ValueError("Document resource cache path escaped its package root.")
        packages = (
            [path for path in package_root.iterdir() if _PACKAGE_NAME.fullmatch(path.name)]
            if package_root.exists()
            else []
        )
        if not directory.exists():
            used = sum(
                path.stat().st_size
                for package in packages
                for path in package.rglob("*")
                if path.is_file()
            )
            if (
                len(packages) >= MAX_CACHE_PACKAGES
                or used + sum(map(len, files.values())) > MAX_CACHE_BYTES
            ):
                raise ValueError(
                    "Native document cache is full (64 packages / 256 MiB). "
                    "Clear output/documents while the local server is stopped, then retry."
                )
        for name, data in files.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Document resource cache path escaped its package.")
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError(
                        "Document resource cache changed; use a clean package directory."
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        (directory / "assets").mkdir(parents=True, exist_ok=True)
    receipt.update(
        resource_digest=package_digest,
        assets=list(assets.values()),
        fonts=list(fonts.values()),
    )
    return PreparedDocument(ResourceContext(directory / "assets", directory / "fonts"), receipt)
