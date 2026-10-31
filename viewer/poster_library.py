"""Read the curated poster library without accepting arbitrary local paths."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "fixtures" / "open_posters"
GALLERY_ROOT = ROOT / "output" / "poster-audit"
MAX_BUNDLE_BYTES = 24 * 1024 * 1024
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,99}\Z")


@dataclass(frozen=True)
class _CataloguePage:
    """Read-only HTTP asset with the same read interface as persisted gallery files."""

    content: bytes
    name: str = "index.html"

    def is_file(self):
        return True

    def read_bytes(self):
        return self.content


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
                "source_note": entry.get("source_note"),
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


def _catalogue_gallery() -> _CataloguePage:
    """List original inputs without depending on generated evaluation artifacts."""
    entries = {entry["id"]: entry for entry in catalogue()}
    examples = library_metadata()["examples"]
    groups = []
    for split, heading in (
        ("reviewed", "Reviewed examples"),
        ("reserve", "Personal test originals"),
    ):
        cards = []
        for item in examples:
            if item["split"] != split:
                continue
            identity = item["id"]
            title = html.escape(str(item["title"]))
            credit = html.escape(str(item["attribution_text"]))
            license_name = html.escape(str(item["license_id"]))
            preview = prepared_path(entries[identity], "native.png")
            image = (
                f'<img src="/api/posters/{identity}/preview" alt="Source preview: {title}" '
                'loading="lazy" width="320" height="300">'
                if preview.is_file()
                else '<p class="unavailable">Source preview is unavailable.</p>'
            )
            action = (
                f'<a class="open" href="/?poster={identity}">Open in Studio</a>'
                if item["ready"]
                else '<p class="status">Import support is still needed.</p>'
            )
            links = [f'<a href="/api/posters/{identity}/original">Editable original</a>']
            for key, label in (("source_url", "Source"), ("license_url", "License")):
                if item[key]:
                    links.append(
                        f'<a href="{html.escape(item[key], quote=True)}" '
                        f'target="_blank" rel="noopener noreferrer">{label}</a>'
                    )
            note = (
                f'<p class="note">{html.escape(str(item["source_note"]))}</p>'
                if item["source_note"]
                else ""
            )
            cards.append(
                f'<article>{image}<div class="details"><h3>{title}</h3>'
                f'<p class="credit">{credit} · {license_name}</p>{note}{action}'
                f'<p class="links">{" · ".join(links)}</p></div></article>'
            )
        if cards:
            groups.append(
                f'<section><h2>{heading}</h2><div class="grid">{"".join(cards)}</div></section>'
            )
    content = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Poster library · Chameleon</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f6f6;color:#242424;
font:14px/1.5 system-ui,sans-serif}a{color:#305d8f;text-underline-offset:3px}
a:focus-visible{outline:3px solid #1473e6;outline-offset:4px}
header{padding:18px 28px;background:#fff;border-bottom:1px solid #dedede}
header a{color:#242424;font-size:18px;font-weight:650;text-decoration:none}
main{max-width:1480px;margin:auto;padding:28px}h1{font-size:30px;letter-spacing:-.7px;
margin:0 0 8px}h2{font-size:20px;margin:32px 0 16px}h3{font-size:16px;margin:0 0 6px}
.intro{max-width:72ch;color:#575757}.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:20px}
article{min-width:0;background:#fff;border:1px solid #dedede;border-radius:8px;overflow:hidden}
article img{display:block;width:100%;height:300px;object-fit:contain;padding:16px;
background:#ededed}.details{padding:18px}.credit,.note,.status,.links{font-size:13px;
overflow-wrap:anywhere}.credit{color:#626262}.open{display:inline-block;margin:8px 0;
padding:8px 12px;color:#fff;background:#303030;border-radius:5px;text-decoration:none}
.unavailable{display:grid;place-items:center;min-height:180px;color:#686868}
@media(max-width:600px){header{padding:16px}main{padding:20px 16px}h1{font-size:26px}}
</style></head><body><header><a href="/">Chameleon Studio</a></header><main>
<h1>Poster library</h1><p class="intro">Browse original designs and their source previews.
Open a design in Studio to choose a canvas and compare a resized result.
Source and license information accompanies each design.</p>"""
    content += (
        "".join(groups) if groups else "<p>No poster sources are available in this checkout.</p>"
    )
    content += "</main></body></html>"
    return _CataloguePage(content.encode("utf-8"))


def gallery_file(relative: str) -> Path | _CataloguePage:
    path = contained(GALLERY_ROOT, relative or "index.html")
    if path.suffix.lower() not in {".html", ".png", ".jpg", ".jpeg", ".json", ".zip", ".css"}:
        raise FileNotFoundError("Gallery file not found.")
    if not path.is_file():
        if path == (GALLERY_ROOT / "index.html").resolve():
            return _catalogue_gallery()
        raise FileNotFoundError("Gallery has not been generated. Run tools.poster_audit.")
    return path
