"""Read the curated poster library without accepting arbitrary local paths."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "fixtures" / "open_posters"
GALLERY_ROOT = ROOT / "output" / "poster-audit"
MAX_BUNDLE_BYTES = 24 * 1024 * 1024
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,99}\Z")


def contained(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\0" in relative:
        raise ValueError("Invalid library path.")
    rel = Path(relative.replace("\\", "/"))
    if rel.is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError("Invalid library path.")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Library path is outside the collection.")
    return path


def catalogue() -> list[dict]:
    path = CORPUS_ROOT / "catalogue.json"
    if not path.is_file():
        return []
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Poster catalogue is too large.")
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("examples") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(entries, list)
        or len(entries) > 100
    ):
        raise ValueError("Invalid poster catalogue.")
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict) or not _ID.fullmatch(str(entry.get("id", ""))):
            raise ValueError("Invalid poster identifier.")
        if entry["id"] in ids or entry.get("split") not in {"reviewed", "reserve"}:
            raise ValueError("Duplicate poster or invalid collection split.")
        ids.add(entry["id"])
    return entries


def entry_for(poster_id: str) -> dict:
    if not _ID.fullmatch(poster_id):
        raise FileNotFoundError("Poster not found.")
    return next((entry for entry in catalogue() if entry["id"] == poster_id), None) or _missing()


def _missing():
    raise FileNotFoundError("Poster not found.")


def prepared_path(entry: dict, name: str) -> Path:
    folder = contained(CORPUS_ROOT, entry["prepared_path"])
    return contained(folder, name)


def import_status(entry: dict) -> dict:
    report = prepared_path(entry, "import-report.json")
    if not report.is_file():
        return {"status": "unprepared", "failures": []}
    if report.stat().st_size > 1024 * 1024:
        raise ValueError("Import report is too large.")
    result = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Invalid poster import report.")
    return result


def _web_link(value: object) -> str | None:
    return value if isinstance(value, str) and urlsplit(value).scheme in {"https", "http"} else None


def library_metadata() -> dict:
    examples = []
    for entry in catalogue():
        status = import_status(entry)
        poster_id = entry["id"]
        ready = status.get("status") == "ready" and prepared_path(entry, "bundle.json").is_file()
        examples.append(
            {
                "id": poster_id,
                "title": entry.get("title", poster_id),
                "split": entry["split"],
                "author": entry.get("author", "See source attribution"),
                "attribution_text": entry.get(
                    "attribution_text", entry.get("author", "See source attribution")
                ),
                "license_id": entry.get("license_id", "See source license"),
                "original_sha256": entry.get("sha256"),
                "source_url": _web_link(entry.get("source_url")),
                "license_url": _web_link(entry.get("license_url")),
                "ready": ready,
                "import_status": status.get("status", "unprepared"),
                "bundle_url": f"/api/posters/{poster_id}/bundle" if ready else None,
                "original_url": f"/api/posters/{poster_id}/original",
                "preview_url": f"/api/posters/{poster_id}/preview",
            }
        )
    return {"examples": examples, "gallery_url": "/posters/"}


def read_bundle(poster_id: str) -> dict:
    entry = entry_for(poster_id)
    if import_status(entry).get("status") != "ready":
        raise ValueError("This poster is not ready for resizing. See its import notes.")
    path = prepared_path(entry, "bundle.json")
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Poster bundle exceeds the local request limit.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"source", "assets", "fonts", "document"} <= data.keys():
        raise ValueError("Invalid poster resource bundle.")
    return data


def original_file(poster_id: str) -> Path:
    entry = entry_for(poster_id)
    path = contained(CORPUS_ROOT, entry["original_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Poster original no longer matches its recorded source hash.")
    return path


def preview_file(poster_id: str) -> Path:
    return prepared_path(entry_for(poster_id), "native.png")


def gallery_file(relative: str) -> Path:
    path = contained(GALLERY_ROOT, relative or "index.html")
    if path.suffix.lower() not in {".html", ".png", ".jpg", ".jpeg", ".json", ".zip", ".css"}:
        raise FileNotFoundError("Gallery file not found.")
    if not path.is_file():
        raise FileNotFoundError("Gallery has not been generated. Run tools.poster_audit.")
    return path
