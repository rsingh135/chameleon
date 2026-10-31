"""Build the Linux submission and verify real sample artifacts without networking.

    python -m tools.verify_docker --out output/docker-compliance/run-20261030

Each invocation requires a fresh output directory so prior artifacts cannot make
a failed run look successful. Docker Desktop must expose a Linux engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TREES = ("engine", "renderer", "brand_kit")
RUNTIME_FILES = (
    "chameleon_design_schema_v0.9.json",
    "requirements.txt",
    "models/fit_policy/default.json",
)


def runtime_file_hashes(root: Path) -> dict[str, str]:
    """The complete runtime COPY contract, excluding generated Python caches."""
    paths = [root / name for name in RUNTIME_FILES]
    for folder in RUNTIME_TREES:
        paths.extend(
            path
            for path in (root / folder).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def verify_packaged_files(packaged: dict[str, str], *, root: Path = ROOT) -> None:
    expected = runtime_file_hashes(root)
    if packaged == expected:
        return
    missing = sorted(expected.keys() - packaged.keys())
    unexpected = sorted(packaged.keys() - expected.keys())
    changed = sorted(
        name for name in expected.keys() & packaged.keys() if expected[name] != packaged[name]
    )
    raise RuntimeError(
        "Image differs from submission files: "
        f"missing={missing[:5]}, unexpected={unexpected[:5]}, changed={changed[:5]}"
    )


def verify_output_schemas(artifacts: Path, rows: list[dict], *, root: Path = ROOT) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("Install jsonschema==4.26.0 on the host before verification") from exc
    schema = json.loads((root / "chameleon_design_schema_v0.9.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for row in rows:
        name = row["output_json"]
        if Path(name).name != name:
            raise RuntimeError(f"Invalid output artifact name: {name}")
        design = json.loads((artifacts / name).read_text(encoding="utf-8"))
        error = next(validator.iter_errors(design), None)
        if error is not None:
            raise RuntimeError(f"Official schema rejected {name}: {error.message}")
        row["schema_valid"] = True


VALIDATOR = r"""
import hashlib
import json
from pathlib import Path
from PIL import Image
from engine.eval import check_contract
from engine.eval.render import render_image
from engine.stage_one import strictly_in_bounds

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = []
tasks = sorted(Path("/app/input").glob("*.json"))
assert tasks, "No input tasks were mounted"
for task_path in tasks:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    result_path = Path("/app/output") / (task_path.stem + "_output.json")
    png_path = result_path.with_suffix(".png")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    violations = check_contract(task["source"], result, task["target_canvas"])
    assert not violations, str(violations)
    assert strictly_in_bounds(result), "Output exceeds strict rotated bounds"
    with Image.open(png_path) as image:
        dimensions = list(image.size)
        assert image.format == "PNG", "Preview is not a PNG"
        assert dimensions == [task["target_canvas"][key] for key in ("width", "height")]
        image.verify()
    expected = render_image(result, max_size=max(result["canvas"].values()))
    with Image.open(png_path) as image:
        assert image.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes(), \
            "PNG does not match native rendering of output JSON"
    rows.append({
        "task": task_path.name,
        "task_sha256": sha(task_path),
        "output_json": result_path.name,
        "output_json_sha256": sha(result_path),
        "output_png": png_path.name,
        "output_png_sha256": sha(png_path),
        "png_dimensions": dimensions,
        "element_count": len(result["elements"]),
        "contract_valid": True,
        "strict_bounds_valid": True,
        "native_render_matches": True,
    })
files = [Path("/app/chameleon_design_schema_v0.9.json"), Path("/app/requirements.txt")]
for folder in ("engine", "renderer", "brand_kit", "models"):
    files.extend(path for path in Path("/app", folder).rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts
                 and path.suffix != ".pyc")
packaged = {str(path.relative_to("/app")): sha(path) for path in sorted(files)}
print(json.dumps({"tasks": rows, "packaged_file_sha256": packaged}, sort_keys=True))
"""


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8").strip()


def logged(command: list[str], path: Path) -> int:
    print(f"Running: {subprocess.list2cmdline(command)}", flush=True)
    with path.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="chameleon-engine:compliance")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args(argv)
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = destination / "artifacts"
    artifacts.mkdir()
    container = "chameleon-offline-" + uuid4().hex[:16]
    receipt = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "image_tag": args.image,
        "platform": "linux/amd64",
        "container": container,
        "network_mode": "none",
        "artifacts_directory": str(artifacts),
    }
    try:
        if not args.skip_build:
            command = [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--progress",
                "plain",
                "-t",
                args.image,
                ".",
            ]
            receipt["build_command"] = command
            if logged(command, destination / "build.log"):
                raise RuntimeError("Docker build failed; see build.log")
        image_id, operating_system, architecture = capture(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}} {{.Os}} {{.Architecture}}",
                args.image,
            ]
        ).split()
        if (operating_system, architecture) != ("linux", "amd64"):
            raise RuntimeError("Submission image is not Linux AMD64")
        receipt["image_id"] = image_id
        mounts = [
            "--mount",
            f"type=bind,source={ROOT / 'input'},target=/app/input,readonly",
            "--mount",
            f"type=bind,source={artifacts},target=/app/output",
        ]
        command = [
            "docker",
            "run",
            "--name",
            container,
            "--network",
            "none",
            "--env",
            "PYTHONUNBUFFERED=1",
            *mounts,
            image_id,
        ]
        receipt["run_command"] = command
        run_exit_code = logged(command, destination / "run.log")
        receipt["run_exit_code"] = run_exit_code
        inspection = json.loads(
            capture(
                [
                    "docker",
                    "inspect",
                    "--format",
                    '{"network_mode":{{json .HostConfig.NetworkMode}},'
                    '"image_id":{{json .Image}},"exit_code":{{.State.ExitCode}}}',
                    container,
                ]
            )
        )
        receipt["container_inspection"] = inspection
        if inspection["network_mode"] != "none" or inspection["image_id"] != image_id:
            raise RuntimeError("Container did not use the required offline image configuration")
        if run_exit_code:
            raise RuntimeError("Offline engine failed; see run.log")
        with (destination / "validation.log").open("w", encoding="utf-8") as stream:
            validation_process = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    *mounts,
                    "--entrypoint",
                    "python",
                    image_id,
                    "-c",
                    VALIDATOR,
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=stream,
                text=True,
                encoding="utf-8",
                check=False,
            )
        if validation_process.returncode:
            raise RuntimeError("Artifact validation failed; see validation.log")
        validation = json.loads(validation_process.stdout)
        receipt.update(validation)
        receipt["workspace_matches_image"] = False
        verify_packaged_files(validation["packaged_file_sha256"])
        receipt["workspace_matches_image"] = True
        verify_output_schemas(artifacts, validation["tasks"])
        receipt["status"] = "passed"
        print(f"PASS: {len(validation['tasks'])} tasks; real JSON/PNG; network none", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        receipt["error"] = str(exc)
        print(f"FAIL: {exc}", flush=True)
        return 1
    finally:
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        (destination / "report.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(
            ["docker", "rm", container],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
