"""The stage-one archive contains only declared bytes and never claims a run."""

from __future__ import annotations

import hashlib
import json
from zipfile import ZipFile

import pytest

from tools import package_submission as packaging


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    names = (
        *packaging.REQUIRED_FILES,
        *(f"brand_kit/fonts/{name}" for name in packaging.FONT_NAMES),
        "engine/layout/example.py",
        "renderer/NOTICE.txt",
        "brand_kit/fonts/licenses/OFL.txt",
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + "\nexact original bytes\r\n").encode())
    return root


def build(root, name="candidate"):
    destination = root / "output/submission" / name
    return destination, packaging.package_submission(destination, root=root)


def test_archive_exact_inventory_hashes_provenance_and_determinism(project):
    first, receipt = build(project)
    second, second_receipt = build(project, "again")
    assert receipt == second_receipt
    assert (first / packaging.ARCHIVE_NAME).read_bytes() == (
        second / packaging.ARCHIVE_NAME
    ).read_bytes()
    with ZipFile(first / packaging.ARCHIVE_NAME) as archive:
        manifest_bytes = archive.read(packaging.MANIFEST_NAME)
        manifest = json.loads(manifest_bytes)
        assert set(archive.namelist()) == set(manifest["files"]) | {packaging.MANIFEST_NAME}
        assert archive.read("output/.gitkeep") == b""
        assert manifest["docker_verified"] is False
        for name, reference in manifest["files"].items():
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == reference["sha256"]
            assert len(data) == reference["bytes"]
            if name != "output/.gitkeep":
                assert data == (project / name).read_bytes()
        for name in ("renderer/NOTICE.txt", "brand_kit/fonts/licenses/OFL.txt"):
            assert name in manifest["files"]
        assert receipt["manifest"]["sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    assert (
        receipt["archive"]["sha256"]
        == hashlib.sha256((first / packaging.ARCHIVE_NAME).read_bytes()).hexdigest()
    )
    assert receipt["docker_verified"] is False


def test_excludes_runtime_config_reports_weights_and_caches(project):
    excluded = (
        "engine/__pycache__/cache.pyc",
        "engine/reports/private.md",
        "engine/.env",
        "engine/configs/runtime.json",
        "renderer/debug-report.json",
        "brand_kit/.secret",
        "brand_kit/runtime/token.txt",
        "models/adapter.safetensors",
        "output/run.txt",
        "experiments/private.py",
        "input/my-own-scene.json",
        "docs/internal.md",
    )
    for name in excluded:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not a submission input", encoding="utf-8")
    destination, _ = build(project)
    with ZipFile(destination / packaging.ARCHIVE_NAME) as archive:
        assert not set(excluded) & set(archive.namelist())


def test_optional_host_verifier_has_its_package_marker(project):
    for name in ("tools/__init__.py", "tools/verify_docker.py"):
        path = project / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"")
    destination, _ = build(project)
    with ZipFile(destination / packaging.ARCHIVE_NAME) as archive:
        assert archive.read("tools/verify_docker.py") == b""
        assert archive.read("tools/__init__.py") == b""
    (project / "tools/__init__.py").unlink()
    with pytest.raises(ValueError, match="Missing required"):
        build(project, "missing-marker")


@pytest.mark.parametrize(
    "missing",
    [
        "Dockerfile",
        ".dockerignore",
        "engine/main.py",
        "models/fit_policy/default.json",
        "input/chameleon_sample_level3_advanced.json",
    ],
)
def test_missing_required_input_refused_before_output(project, missing):
    (project / missing).unlink()
    with pytest.raises(ValueError, match="Missing required"):
        build(project)
    assert not (project / "output/submission/candidate").exists()


def test_existing_destination_never_overwritten(project):
    destination, receipt = build(project)
    before = (destination / packaging.ARCHIVE_NAME).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        packaging.package_submission(destination, root=project)
    assert (destination / packaging.ARCHIVE_NAME).read_bytes() == before
    assert json.loads((destination / "receipt.json").read_bytes()) == receipt


@pytest.mark.parametrize("relative", ["elsewhere", "output/submission", "output/../bad"])
def test_destination_stays_under_declared_submission_root(project, relative):
    with pytest.raises(ValueError, match="fresh child"):
        packaging.package_submission(relative, root=project)


def test_external_symlink_refused(project, tmp_path):
    secret = tmp_path / "secret.py"
    secret.write_text("do not copy", encoding="utf-8")
    link = project / "engine/leak.py"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("Platform does not permit test symlinks")
    with pytest.raises(ValueError, match="symlinks/reparse"):
        build(project)


def test_mutation_during_zip_write_has_no_success_receipt(project, monkeypatch):
    original = packaging._zip_member
    changed = False

    def mutate(archive, name, data):
        nonlocal changed
        original(archive, name, data)
        if not changed:
            changed = True
            (project / "engine/main.py").write_bytes(b"changed")

    monkeypatch.setattr(packaging, "_zip_member", mutate)
    with pytest.raises(ValueError, match="changed during freeze"):
        build(project)
    assert not (project / "output/submission/candidate/receipt.json").exists()


@pytest.mark.parametrize("budget", ["MAX_TOTAL_BYTES", "MAX_FILE_BYTES", "MAX_FILES"])
def test_byte_and_file_budgets_are_enforced_before_freeze(project, monkeypatch, budget):
    monkeypatch.setattr(packaging, budget, 12)
    expected = "file-count budget" if budget == "MAX_FILES" else "byte budget"
    with pytest.raises(ValueError, match=expected):
        build(project)
    assert not (project / "output/submission/candidate").exists()


def test_new_source_during_zip_write_has_no_success_receipt(project, monkeypatch):
    original = packaging._zip_member

    def add_source(archive, name, data):
        original(archive, name, data)
        (project / "engine/new_runtime.py").write_bytes(b"new source")

    monkeypatch.setattr(packaging, "_zip_member", add_source)
    with pytest.raises(ValueError, match="inventory changed"):
        build(project)
    assert not (project / "output/submission/candidate/receipt.json").exists()
