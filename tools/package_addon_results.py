"""Package actual engine JSON/PNG artifacts for the stable Express viewer.

Usage: python -m tools.package_addon_results --outputs <engine-output-directory>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from io import BytesIO
from pathlib import Path

from PIL import Image

from engine.eval import check_contract
from engine.eval.render import render_image
from engine.resources import using_resources
from engine.stage_one import strictly_in_bounds
from viewer.addon_bridge import source_digest, validate_design_fonts

ROOT = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def package_result(task: dict, design: dict, output_png: bytes, *, assets_dir, fonts_dir) -> dict:
    """Validate the supplied result and preserve its exported PNG bytes exactly."""
    from viewer.server import _render_png

    source, target = task["source"], task["target_canvas"]
    with using_resources(assets_dir, fonts_dir):
        validate_design_fonts(source, fonts_dir)
        violations = check_contract(source, design, target)
        if violations or not strictly_in_bounds(design):
            raise ValueError(
                "Result violates the stage-one contract: " + "; ".join(map(str, violations))
            )
        with Image.open(BytesIO(output_png)) as image:
            if image.format != "PNG" or image.size != (target["width"], target["height"]):
                raise ValueError("Output must be a full-resolution PNG matching the target canvas.")
            image.load()
            expected = render_image(design, max_size=max(target.values()))
            if image.convert("RGBA").tobytes() != expected.convert("RGBA").tobytes():
                raise ValueError(
                    "Output PNG pixels do not match a native render of the output JSON."
                )
        source_preview, warnings = _render_png(source)
    source_png = base64.b64decode(source_preview.split(",", 1)[1])
    digest = source_digest(source)
    return {
        "schema_version": 1,
        "request_id": "engine-result-" + _sha(output_png)[:24],
        "source": source,
        "source_digest": digest,
        "target_canvas": target,
        "design": design,
        "source_png": source_preview,
        "output_png": _png(output_png),
        "report": {"violations": []},
        "warnings": warnings,
        "provenance": {
            "source_sha256": digest,
            "source_png_sha256": _sha(source_png),
            "output_png_sha256": _sha(output_png),
            "output_json_sha256": _sha(
                json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ),
            "renderer": "vendored cr-renderer; output PNG pixels verified against output JSON",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "input")
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "output/addon-results")
    parser.add_argument("--assets", type=Path, default=ROOT / "brand_kit/assets")
    parser.add_argument("--fonts", type=Path, default=ROOT / "brand_kit/fonts")
    parser.add_argument(
        "--docker-report", type=Path, help="Use the verified run's exact Linux image"
    )
    args = parser.parse_args()
    if args.docker_report is not None:
        report = json.loads(args.docker_report.read_text(encoding="utf-8"))
        image = report.get("image_id", "")
        if (
            report.get("status") != "passed"
            or report.get("network_mode") != "none"
            or not report.get("workspace_matches_image")
            or not image.startswith("sha256:")
            or len(image) != 71
            or any(char not in "0123456789abcdef" for char in image[7:])
            or Path(report.get("artifacts_directory", "")).resolve() != args.outputs.resolve()
        ):
            parser.error("Use a passing offline receipt for these output artifacts.")
        if (
            args.assets.resolve() != ROOT / "brand_kit/assets"
            or args.fonts.resolve() != ROOT / "brand_kit/fonts"
        ):
            parser.error(
                "Docker receipt mode uses the brand resources packaged in the verified image."
            )
        try:
            inputs = args.inputs.resolve().relative_to(ROOT).as_posix()
            outputs = args.outputs.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            parser.error(
                "Docker receipt mode requires input/output artifacts inside this repository."
            )
        args.out.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,source={ROOT},target=/source,readonly",
            "--mount",
            f"type=bind,source={args.out.resolve()},target=/bundles",
            "--env",
            "PYTHONPATH=/source",
            "--entrypoint",
            "python",
            image,
            "-m",
            "tools.package_addon_results",
            "--inputs",
            f"/source/{inputs}",
            "--outputs",
            f"/source/{outputs}",
            "--out",
            "/bundles",
            "--assets",
            "/app/brand_kit/assets",
            "--fonts",
            "/app/brand_kit/fonts",
        ]
        return subprocess.run(command, check=False).returncode
    tasks = sorted(args.inputs.glob("*.json"))
    if not tasks:
        parser.error("No task JSON files found in --inputs.")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in tasks:
        task = json.loads(path.read_text(encoding="utf-8"))
        design = json.loads((args.outputs / f"{path.stem}_output.json").read_text(encoding="utf-8"))
        png = (args.outputs / f"{path.stem}_output.png").read_bytes()
        bundle = package_result(task, design, png, assets_dir=args.assets, fonts_dir=args.fonts)
        target = args.out / f"{path.stem}.chameleon.json"
        target.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
