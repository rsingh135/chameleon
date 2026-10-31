"""Verify the portable source corpus without resizing the reserve examples."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "open_posters"
CATALOGUE = json.loads((ROOT / "catalogue.json").read_text(encoding="utf-8"))


def checked(relative, expected=None):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert path.is_file(), relative
    data = path.read_bytes()
    assert data, relative
    if expected:
        assert hashlib.sha256(data).hexdigest() == expected, relative
    return path


def test_eighteen_distinct_works_and_separate_personal_test_set():
    entries = CATALOGUE["examples"]
    assert CATALOGUE["schema_version"] == 1
    assert Counter(entry["split"] for entry in entries) == {"reviewed": 8, "reserve": 10}
    assert len({entry["id"] for entry in entries}) == 18
    assert len({entry["sha256"] for entry in entries}) == 18
    for entry in entries:
        checked(entry["original_path"], entry["sha256"])
        assert entry["source_url"].startswith("https://")
        assert entry["source_commit"]
        checked(entry["license_path"])
        if "notice_path" in entry:
            checked(entry["notice_path"])


def test_associated_images_and_independent_author_references():
    for entry in CATALOGUE["examples"]:
        for asset in entry.get("asset_requirements", []):
            checked(asset["path"], asset["sha256"])
            checked(asset["provenance_path"])
        reference = entry.get("native_reference")
        if reference:
            checked(reference["path"], reference["sha256"])
            checked(reference["license_path"])


def test_pinned_open_font_bytes_and_postscript_names():
    manifest = json.loads((ROOT / "fonts/source-manifest.json").read_text(encoding="utf-8"))
    faces = manifest["faces"]
    assert len(faces) >= 30
    known = set()
    for face in faces:
        path = checked(face["path"], face["sha256"])
        checked(face["license_path"])
        if face.get("source_path"):
            checked(face["source_path"], face["source_sha256"])
        with TTFont(path) as font:
            names = {record.toUnicode() for record in font["name"].names if record.nameID == 6}
            assert face["postscript_name"] in names
        known.add((face["path"], face["sha256"], face["postscript_name"]))
    for entry in CATALOGUE["examples"]:
        for face in entry.get("font_requirements", []):
            assert (face["path"], face["sha256"], face["postscript_name"]) in known


def test_open_font_adaptations_are_separate_and_disclosed():
    adapted = [entry for entry in CATALOGUE["examples"] if "prepared_reference_path" in entry]
    assert len(adapted) == 2
    for entry in adapted:
        checked(entry["prepared_reference_path"], entry["prepared_reference_sha256"])
        assert entry["prepared_reference_path"] != entry["original_path"]
        assert entry["adaptations"]
        assert entry["adaptation_note"]


def test_required_doory_credit_is_kept_with_each_work():
    required = "Doory by Siam, creator and maintainer of Doory, https://github.com/shaongitbd/doory"
    entries = [entry for entry in CATALOGUE["examples"] if entry["id"].startswith("doory_")]
    assert len(entries) == 10
    assert all(entry["attribution_text"] == required for entry in entries)
    assert all(entry["license_id"] == "LicenseRef-Doory-AAL" for entry in entries)
