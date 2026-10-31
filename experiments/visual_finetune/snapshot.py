"""Freeze the single approved stdlib sampler dependency alongside experiment code."""

import shutil
from pathlib import Path

from .data import sha

FROZEN_TOOL_KEYS = frozenset({"tools/__init__.py", "tools/purpose_coverage_sampler.py"})


def code_path(package_dir, name):
    package_dir = Path(package_dir).resolve()
    if Path(name).name == name:
        return package_dir / name
    if name in FROZEN_TOOL_KEYS:
        return package_dir.parents[1] / name
    raise ValueError("Unsupported path in isolated experiment code identity")


def verify_code(package_dir, expected):
    for name, digest in expected.items():
        if sha(code_path(package_dir, name)) != digest:
            raise ValueError("Isolated experiment code differs from its frozen recipe")


def freeze_coverage_helper(source_root, code_root, expected=None):
    source_root, code_root = Path(source_root), Path(code_root)
    helper = source_root / "tools/purpose_coverage_sampler.py"
    if expected is not None:
        if not FROZEN_TOOL_KEYS.issubset(expected):
            raise ValueError("Frozen coverage identity is incomplete")
        for name in FROZEN_TOOL_KEYS:
            if sha(source_root / name) != expected[name]:
                raise ValueError("Original frozen coverage helper changed")
    target = code_root / "tools"
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_bytes(b"")
    shutil.copyfile(helper, target / helper.name)
    return {name: sha(code_root / name) for name in sorted(FROZEN_TOOL_KEYS)}
