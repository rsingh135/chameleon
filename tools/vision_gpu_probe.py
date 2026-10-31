"""Check GPU execution before spending time on a vision-model experiment.

Run inside the intended experiment environment on an allocated GPU. This probes
one small matrix multiplication; it does not download or load model weights.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import sysconfig
from pathlib import Path


def build_prerequisites() -> tuple[dict, int]:
    include = Path(sysconfig.get_path("include"))
    header = include / "Python.h"
    compiler = shutil.which("cc") or shutil.which("gcc")
    missing = []
    if not header.is_file():
        missing.append(str(header))
    if not compiler:
        missing.append("cc/gcc compiler")
    report = {
        "build_prerequisites_passed": not missing,
        "python_header": str(header),
        "compiler": compiler,
        "missing": missing,
        "model_training_validated": False,
        "scope": "Compiler/header availability only; no CUDA/model execution",
    }
    if missing:
        package = f"python{sys.version_info.major}.{sys.version_info.minor}-dev"
        report["error"] = (
            f"Missing runtime kernel build prerequisites. On Ubuntu install {package} "
            "and build-essential, then rerun. No system packages were installed."
        )
    return report, 0 if not missing else 5


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "peft", "trl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def probe(allow_cpu: bool, min_vram_gib: float) -> tuple[dict, int]:
    report = {
        "python": platform.python_version(),
        "packages": package_versions(),
        "probe_passed": False,
        "model_training_validated": False,
        "scope": "small tensor operation only; validate model/data compatibility separately",
    }
    try:
        import torch
    except (ImportError, OSError) as error:
        report["error"] = (
            f"PyTorch cannot load in this environment: {type(error).__name__}: {error}"
        )
        return report, 2

    try:
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"] and not allow_cpu:
            report["error"] = (
                "CUDA is unavailable. Check the GPU allocation and CUDA-enabled PyTorch."
            )
            return report, 2
        if not report["cuda_available"] and min_vram_gib:
            report["error"] = (
                "A VRAM minimum requires a CUDA device; CPU fallback cannot satisfy it."
            )
            return report, 3

        device = "cuda:0" if report["cuda_available"] else "cpu"
        report["device"] = device
        dtype = torch.float32
        if report["cuda_available"]:
            gpu = torch.cuda.get_device_properties(0)
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            report["gpu_name"] = gpu.name
            report["compute_capability"] = [gpu.major, gpu.minor]
            report["total_vram_gib"] = round(total_memory / 1024**3, 3)
            report["free_vram_gib"] = round(free_memory / 1024**3, 3)
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
            if free_memory / 1024**3 < min_vram_gib:
                report["error"] = f"Less than {min_vram_gib:g} GiB of free GPU memory is available."
                return report, 3
            if report["bf16_supported"]:
                dtype = torch.bfloat16

        with torch.inference_mode():
            matrix = torch.ones((512, 512), device=device, dtype=dtype)
            result = matrix @ matrix
            if report["cuda_available"]:
                torch.cuda.synchronize()
            report["dtype"] = str(dtype)
            report["observed_value"] = float(result[0, 0].item())
            report["probe_passed"] = report["observed_value"] == 512.0
        if not report["probe_passed"]:
            report["error"] = "The tensor operation returned an unexpected result."
            return report, 4
        return report, 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report, 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow a CPU-only plumbing check")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Check compiler/Python.h only; does not import torch",
    )
    parser.add_argument("--min-vram-gib", type=float, default=0, help="Required free GPU GiB")
    args = parser.parse_args()
    if args.min_vram_gib < 0:
        parser.error("--min-vram-gib must be nonnegative")
    report, status = (
        build_prerequisites() if args.build_only else probe(args.allow_cpu, args.min_vram_gib)
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
