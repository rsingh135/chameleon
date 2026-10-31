"""Package ten untouched personal-test inputs; never call the resize engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or PureWindowsPath(relative).drive:
        raise ValueError("Invalid source path")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes corpus")
    return path


def pack_reserve(manifest_path: Path, destination: Path) -> dict:
    manifest_path, destination = Path(manifest_path).resolve(), Path(destination).resolve()
    root = manifest_path.parent
    catalogue = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [entry for entry in catalogue["examples"] if entry["split"] == "reserve"]
    if len(entries) != 10 or len({entry["sha256"] for entry in entries}) != 10:
        raise ValueError("The personal-test pack requires ten distinct reserve originals")
    files: dict[str, bytes] = {}

    def add(relative, expected=None):
        path = _inside(root, relative)
        data = path.read_bytes()
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {relative}")
        files[path.relative_to(root).as_posix()] = data

    credits = []
    for entry in entries:
        add(entry["original_path"], entry["sha256"])
        if entry.get("prepared_reference_path"):
            add(entry["prepared_reference_path"], entry["prepared_reference_sha256"])
        for asset in entry.get("asset_requirements", []):
            add(asset["path"], asset["sha256"])
        reference = entry.get("native_reference")
        if reference:
            add(reference["path"], reference["sha256"])
        for face in entry.get("font_requirements", []):
            add(face["path"], face["sha256"])
            add(face["license_path"])
        prepared = entry["prepared_path"]
        report = json.loads(
            _inside(root, f"{prepared}/import-report.json").read_text(encoding="utf-8")
        )
        if report.get("status") != "ready":
            raise ValueError(f"Reserve import is not ready: {entry['id']}")
        for name in ("bundle.json", "source.json", "import-report.json", "native.png"):
            add(f"{prepared}/{name}")
        credit = entry.get("attribution_text", entry["author"])
        credits.append(
            f"- {entry['title']}: {credit}\n  {entry['source_url']}\n  {entry['license_id']}"
        )
        if entry.get("source_note"):
            credits.append(f"  Source limitation: {entry['source_note']}")
    # Keep the full signed licenses, authors, asset provenance and notices together.
    for path in sorted((root / "licenses").rglob("*")):
        if path.is_file():
            add(path.relative_to(root).as_posix())
    files["catalogue.json"] = json.dumps(
        {"schema_version": 1, "examples": entries}, indent=2
    ).encode()
    readme = (
        """# Ten personal-test designs

These are ten distinct reserve inputs. They were imported and checked for source
support, but were not resized or used to tune Chameleon's layout algorithm.
They include banners, a card, a slide and reference sheets as well as posters.

Start Chameleon's local studio at http://127.0.0.1:8765/. Use **Import JSON** and
select `prepared/<example>/bundle.json`; it contains its editable elements,
images, exact font faces and attribution. Choose a target size and resize.
The studio's "personal tests" source menu exposes the same inputs.

Author originals remain under `originals/`. SVG originals can be edited in a
vector editor. Doory JSON originals retain the author's complete page data;
the Chameleon bundle explicitly selects page 0 for multipage designs. Separately
adapted open-font references are under `adapted/`; consult each import report.
Complex non-text vector artwork is a movable image layer in Chameleon, while
the retained original keeps its vector paths. Text remains editable.

Keep the included licenses, notices and credits with copies and derivatives.
The native previews are source references, not successful resize claims.
Doory's historical font bytes are unknown; the matching families use pinned faces.

## Credits

"""
        + "\n\n".join(credits)
        + "\n"
    )
    files["README.md"] = readme.encode("utf-8")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files["SHA256SUMS.json"] = json.dumps(hashes, indent=2).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    return {
        "reserve_count": len(entries),
        "files": len(files),
        "path": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("fixtures/open_posters/catalogue.json")
    )
    parser.add_argument("--out", type=Path, default=Path("output/poster-audit/reserve-10.zip"))
    args = parser.parse_args()
    print(json.dumps(pack_reserve(args.manifest, args.out), indent=2))


if __name__ == "__main__":
    main()
