"""Create a deterministic Adobe Express private-distribution ZIP from built files.

Only the stable viewer allowlist ships. Native sandbox development code and
node_modules are excluded even when they are present in dist/.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "manifest.json",
    "ui/index.html",
    "ui/style.css",
    "ui/main.js",
    "shared/viewer.js",
    "samples/chameleon_sample_level1_warmup.json",
    "samples/chameleon_sample_level2_core.json",
    "samples/chameleon_sample_level3_advanced.json",
)


def package(destination: Path, results_dir: Path | None = None) -> Path:
    manifest = json.loads((ROOT / "dist/manifest.json").read_text(encoding="utf-8"))
    if "experimentalApis" in manifest.get("requirements", {}):
        raise ValueError("The submission manifest must not declare experimental APIs")
    if any("documentSandbox" in entry for entry in manifest["entryPoints"]):
        raise ValueError("The stage-one viewer must not load the native development sandbox")
    main = (ROOT / "dist/ui/main.js").read_text(encoding="utf-8")
    if "apiProxy" in main or "express-document-sdk" in main:
        raise ValueError("Experimental/native APIs cannot be part of the viewer entrypoint")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"chameleon-viewer-{manifest['version']}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(FILES):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, (ROOT / "dist" / name).read_bytes())
    if archive.stat().st_size >= 50 * 1024 * 1024:
        raise ValueError("Adobe private-distribution ZIP must be smaller than 50 MiB")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or tuple(sorted(bundle.namelist())) != tuple(sorted(FILES)):
            raise ValueError("The submission archive failed verification")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (destination / f"{archive.name}.sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    shutil.copyfile(ROOT / "assets/icon-144.png", destination / "icon-144.png")
    (destination / "release-notes.txt").write_text(
        "Chameleon Design Viewer 1.0.0. Compare original and resized engine renders, "
        "inspect element geometry, download unchanged design JSON and full-resolution PNG, "
        "and add the rendered PNG to Adobe Express. Open an engine-result bundle without "
        "localhost access, or connect to the local offline engine for new canvas sizes. "
        "Uses stable Adobe UI SDK APIs; native editable reflow remains "
        "a separate development build.\n",
        encoding="utf-8",
    )
    if results_dir is not None:
        results = sorted(results_dir.glob("*.chameleon.json"))
        if not results:
            raise ValueError("The result directory contains no .chameleon.json engine bundles")
        result_output = destination / "engine-results"
        result_output.mkdir(exist_ok=True)
        receipts = []
        for path in results:
            evidence = json.loads(path.read_text(encoding="utf-8"))
            if evidence.get("schema_version") != 1 or not all(
                name in evidence for name in ("source", "target_canvas", "design", "output_png")
            ):
                raise ValueError(f"{path.name} is not an engine-result bundle")
            shutil.copyfile(path, result_output / path.name)
            receipts.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        (result_output / "SHA256SUMS").write_text("\n".join(receipts) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"archive": str(archive.resolve()), "bytes": archive.stat().st_size, "sha256": digest}
        )
    )
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "submission")
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    package(args.out, args.results_dir)
