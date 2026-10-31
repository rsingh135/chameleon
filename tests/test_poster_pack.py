import hashlib
import json
from zipfile import ZipFile

import pytest

from tools.poster_pack import pack_reserve


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "originals").mkdir(parents=True)
    (root / "licenses").mkdir()
    (root / "licenses/NOTICE").write_text("Retain this credit", encoding="utf-8")
    entries = []
    for index in range(11):
        case = f"case{index}"
        data = json.dumps({"original": case}).encode()
        (root / f"originals/{case}.json").write_bytes(data)
        prepared = root / "prepared" / case
        prepared.mkdir(parents=True)
        for name in ("bundle.json", "source.json", "import-report.json"):
            (prepared / name).write_text('{"status":"ready"}', encoding="utf-8")
        (prepared / "native.png").write_bytes(b"fixture native preview")
        (prepared / "landscape.png").write_bytes(b"must not enter input backup")
        entries.append(
            {
                "id": case,
                "title": case,
                "author": "Author",
                "split": "reserve" if index < 10 else "reviewed",
                "original_path": f"originals/{case}.json",
                "sha256": hashlib.sha256(data).hexdigest(),
                "prepared_path": f"prepared/{case}",
                "source_url": "https://example.org",
                "license_id": "CC0-1.0",
            }
        )
    manifest = root / "catalogue.json"
    manifest.write_text(json.dumps({"schema_version": 1, "examples": entries}), encoding="utf-8")
    return manifest, entries


def test_backup_is_reproducible_and_contains_only_ten_reserve_inputs(corpus, tmp_path):
    manifest, _ = corpus
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    assert pack_reserve(manifest, first)["reserve_count"] == 10
    pack_reserve(manifest, second)
    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        names = archive.namelist()
        assert not any("case10" in name or "landscape.png" in name for name in names)
        assert "licenses/NOTICE" in names
        assert len([name for name in names if name.endswith("bundle.json")]) == 10
        hashes = json.loads(archive.read("SHA256SUMS.json"))
        assert all(
            hashlib.sha256(archive.read(name)).hexdigest() == value
            for name, value in hashes.items()
        )


def test_unready_import_and_changed_original_cannot_be_packaged(corpus, tmp_path):
    manifest, entries = corpus
    report = manifest.parent / "prepared/case0/import-report.json"
    report.write_text('{"status":"unsupported"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not ready"):
        pack_reserve(manifest, tmp_path / "backup.zip")
    report.write_text('{"status":"ready"}', encoding="utf-8")
    (manifest.parent / entries[0]["original_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        pack_reserve(manifest, tmp_path / "backup.zip")
