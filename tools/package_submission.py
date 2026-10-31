"""Freeze a small, byte-exact offline-engine ZIP; this does not validate Docker.

python -m tools.package_submission --out output/submission/release-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "chameleon-stage-one-package-v1"
ARCHIVE_NAME = "chameleon-stage-one.zip"
MANIFEST_NAME = "SUBMISSION_MANIFEST.json"
MAX_FILES = 2048
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
TASK_NAMES = (
    "chameleon_sample_level1_warmup.json",
    "chameleon_sample_level2_core.json",
    "chameleon_sample_level3_advanced.json",
)
REQUIRED_FILES = (
    "Dockerfile",
    "requirements.txt",
    "chameleon_design_schema_v0.9.json",
    "README.md",
    "RULES.md",
    "FAQ.md",
    ".dockerignore",
    "engine/__init__.py",
    "engine/main.py",
    "engine/resize.py",
    "renderer/__init__.py",
    "renderer/converter/__init__.py",
    "renderer/converter/convert.py",
    "models/fit_policy/default.json",
    "brand_kit/README.md",
    "brand_kit/assets/solstice_bg_gradient.png",
    "brand_kit/assets/solstice_bottle.png",
    "brand_kit/assets/solstice_logo_lockup.png",
    *(f"input/{name}" for name in TASK_NAMES),
)
TREE_ROOTS = ("engine", "renderer", "brand_kit")
FONT_NAMES = (
    "SolsticeSans-Bold.ttf",
    "SolsticeSans-BoldItalic.ttf",
    "SolsticeSans-Italic.ttf",
    "SolsticeSans-Regular.ttf",
    "SolsticeSerif-Bold.ttf",
    "SolsticeSerif-Italic.ttf",
    "SolsticeSerif-Regular.ttf",
)
LEGAL_PREFIXES = (
    "license",
    "licence",
    "copying",
    "notice",
    "authors",
    "attribution",
    "copyright",
    "ofl",
    "third_party",
    "third-party",
    "provenance",
    "readme",
)
ASSET_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"})
FONT_SUFFIXES = frozenset({".ttf", ".otf", ".woff", ".woff2"})
EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        "cache",
        "checkpoints",
        "configs",
        "experiments",
        "logs",
        "output",
        "reports",
        "runtime",
        "secrets",
        "tmp",
    }
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _plain_path(path, root):
    """Do not follow either POSIX links or Windows junction/reparse points."""
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            metadata = cursor.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"Missing required package input: {relative.as_posix()}") from exc
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        ):
            raise ValueError(f"Package inputs cannot be symlinks/reparse points: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Package input escaped its source root")


def _legal(path):
    return path.name.lower().startswith(LEGAL_PREFIXES)


def _tree_files(directory, root):
    _plain_path(directory, root)
    if not directory.is_dir():
        raise ValueError(f"Missing required package directory: {directory.name}")
    for child in sorted(directory.iterdir()):
        _plain_path(child, root)
        if child.name.startswith(".") or child.name.lower() in EXCLUDED_DIRS:
            continue
        if child.is_dir():
            yield from _tree_files(child, root)
        elif child.is_file():
            relative = child.relative_to(root)
            if relative.parts[0] in {"engine", "renderer"}:
                included = child.suffix == ".py" or _legal(child)
            else:
                included = (
                    child.suffix.lower() in ASSET_SUFFIXES | FONT_SUFFIXES
                    or child.name == "font-manifest.json"
                    or _legal(child)
                )
            if included:
                yield child


def _source_paths(root):
    required = (*REQUIRED_FILES, *(f"brand_kit/fonts/{name}" for name in FONT_NAMES))
    paths = {}
    for relative in required:
        path = root / relative
        _plain_path(path, root)
        if not path.is_file():
            raise ValueError(f"Missing required package input: {relative}")
        paths[relative] = path
    for relative in TREE_ROOTS:
        for path in _tree_files(root / relative, root):
            paths[path.relative_to(root).as_posix()] = path
    verifier = root / "tools/verify_docker.py"
    if verifier.exists() or verifier.is_symlink():
        for relative in ("tools/__init__.py", "tools/verify_docker.py"):
            path = root / relative
            _plain_path(path, root)
            if not path.is_file():
                raise ValueError(f"Missing optional verifier dependency: {relative}")
            paths[relative] = path
    if len(paths) > MAX_FILES:
        raise ValueError("Package file-count budget exceeded")
    return paths


def _snapshot(root):
    paths = _source_paths(root)
    contents, total = {}, 0
    for relative, path in sorted(paths.items()):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Package input exceeds byte budget: {relative}")
        data = path.read_bytes()
        total += len(data)
        if len(data) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("Package input byte budget exceeded")
        contents[relative] = data
    return contents


def _zip_member(archive, name, data):
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = ZIP_STORED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, data)


def package_submission(out, *, root=ROOT):
    """Package declared local inputs into a fresh output/submission child."""
    root = Path(root).resolve(strict=True)
    requested = Path(out)
    destination = (requested if requested.is_absolute() else root / requested).resolve()
    allowed = root / "output/submission"
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ValueError("Submission output must be a fresh child of output/submission")
    if destination.exists():
        raise ValueError("Submission output already exists; choose a fresh directory")
    contents = _snapshot(root)
    source_hashes = {name: _sha(data) for name, data in contents.items()}
    contents["output/.gitkeep"] = b""
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "offline stage-one engine; optional models and development reports excluded",
        "entrypoint": ["python", "-m", "engine.main"],
        "files": {
            name: {"sha256": _sha(data), "bytes": len(data)}
            for name, data in sorted(contents.items())
        },
        "manifest_self_hash": "Stored outside this ZIP in receipt.json",
        "docker_verified": False,
    }
    manifest_bytes = _json_bytes(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    archive_path = destination / ARCHIVE_NAME
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as archive:
        for name, data in sorted({**contents, MANIFEST_NAME: manifest_bytes}.items()):
            _zip_member(archive, name, data)
    current_paths = _source_paths(root)
    if set(current_paths) != set(source_hashes):
        raise ValueError("Package source inventory changed during freeze")
    for name, expected in source_hashes.items():
        path = current_paths[name]
        _plain_path(path, root)
        if not path.is_file() or _sha(path.read_bytes()) != expected:
            raise ValueError(f"Package source changed during freeze: {name}")
    receipt = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "packaged_not_docker_verified",
        "archive": {
            "path": ARCHIVE_NAME,
            "sha256": _sha(archive_path.read_bytes()),
            "bytes": archive_path.stat().st_size,
        },
        "manifest": {"path": MANIFEST_NAME, "sha256": _sha(manifest_bytes)},
        "payload_file_count": len(contents),
        "source_files_unchanged": True,
        "docker_verified": False,
    }
    (destination / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = package_submission(args.out)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Submission package failed: {exc}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
