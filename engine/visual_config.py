"""Local operator configuration and honest visual-execution identity.

No model library is imported, no subprocess is started, and no weight binary is
hashed here. The provider verifies actual model bytes before loading them. Full
model-response caching is disabled independently by the production pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_KEY = "CHAMELEON_VISUAL_PLANNER_CONFIG"
PROVIDER_MODULE = "experiments.visual_planner.provider"
MAX_JSON_BYTES = 4 * 1024 * 1024


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Visual configuration must contain finite JSON values") from exc


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path, maximum=MAX_JSON_BYTES):
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("A visual configuration/code resource is missing or oversized")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("A visual configuration/code resource exceeds its byte budget")
    return hashlib.sha256(data).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate visual configuration JSON key")
        result[key] = value
    return result


def _read(path):
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Visual configuration file is missing or oversized")
    try:
        result = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _x: (_ for _ in ()).throw(ValueError("Nonfinite visual JSON")),
        )
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid visual configuration JSON") from exc
    _canonical(result)
    return result


def _path(value, base):
    if not isinstance(value, str) or not value or "://" in value or any(ord(c) < 32 for c in value):
        raise ValueError("Visual operator paths must be explicit local paths")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _artifact_root(value, base):
    project = ROOT.resolve()
    output = (project / "output").resolve()
    target = _path(value, base)
    if not output.is_relative_to(project) or not target.is_relative_to(output):
        raise ValueError("Visual artifacts must stay under the project output directory")
    return str(target)


def _runtime(path):
    if not path.is_file() or not re.fullmatch(
        r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", path.name, re.I
    ):
        raise ValueError(
            "Visual provider requires an existing local Python executable, not a shell"
        )
    same = path == Path(sys.executable).resolve()
    return {
        "path": str(path),
        "executable_sha256": _sha(path, 64 * 1024 * 1024),
        "python_version": sys.version if same else None,
        "version_scope": "current interpreter" if same else "external version recorded by provider",
    }


def _provider_identity(command, base):
    if isinstance(command, str):
        try:
            command = json.loads(command)
        except (ValueError, RecursionError) as exc:
            raise ValueError("Visual provider command must be a JSON argument array") from exc
    if (
        not isinstance(command, list)
        or len(command) != 5
        or any(not isinstance(v, str) or not v or any(ord(c) < 32 for c in v) for v in command)
    ):
        raise ValueError(
            "Visual provider command must be a fixed local Python/module argument array"
        )
    if command[1:4] != ["-m", PROVIDER_MODULE, "--config"]:
        raise ValueError("Only the known local visual provider module is executable")
    executable = _path(command[0], base)
    config_path = _path(command[4], base)
    runtime = _runtime(executable)
    config = _read(config_path)
    if not isinstance(config, dict):
        raise ValueError("Visual provider configuration must be a JSON object")
    if config.get("backend", "transformers") not in {"transformers", "llama_cpp"}:
        raise ValueError("Unsupported local visual provider backend")
    if config.get("strategy", "generate_then_score") not in {
        "generate_then_score",
        "choice_logits",
    }:
        raise ValueError("Unsupported local visual decision strategy")
    # The provider starts at the project root. Require absolute nested runtime
    # paths so the identity cannot describe a different file from that runtime.
    for key in ("model_dir", "python_executable"):
        if not isinstance(config.get(key), str) or not Path(config[key]).expanduser().is_absolute():
            raise ValueError("Visual provider model/runtime paths must be absolute")
    model = _path(config.get("model_dir"), config_path.parent)
    if not model.is_dir():
        raise ValueError("Visual provider model directory is missing")
    package = ROOT / "experiments/visual_planner"
    code = {p.relative_to(ROOT).as_posix(): _sha(p) for p in sorted(package.rglob("*.py"))}
    if not (package / "provider.py").is_file() or not code:
        raise ValueError("Visual provider implementation is missing")
    manifest_path = model / "verified-model.json"
    manifest_sha, pins = None, None
    if manifest_path.is_file():
        manifest = _read(manifest_path)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise ValueError("Visual model manifest is malformed")
        pins = {k: manifest.get(k) for k in ("repo_id", "revision", "weight_sha256", "files")}
        manifest_sha = _sha(manifest_path)
    elif config.get("backend") == "llama_cpp" and (package / "native.py").is_file():
        # Native code contains the exact public GGUF/projector/runtime pins and
        # validates their actual bytes. Reading pins is not binary verification.
        source = (package / "native.py").read_text(encoding="utf-8")
        values = sorted(set(re.findall(r"[0-9a-f]{64}", source)))
        if not values:
            raise ValueError("Native visual provider has no declared model/runtime pins")
        pins = {
            "provider_native_code_sha256": _sha(package / "native.py"),
            "declared_sha256_values": values,
        }
    else:
        raise ValueError("Visual model manifest is missing")
    worker = config.get("python_executable")
    identity = {
        "command": [str(executable), "-m", PROVIDER_MODULE, "--config", str(config_path)],
        "provider_configuration_path": str(config_path),
        "provider_configuration_sha256": _sha(config_path),
        "provider_configuration": config,
        "code_sha256": code,
        "runtime": runtime,
        "worker_runtime": _runtime(_path(worker, config_path.parent))
        if worker is not None
        else None,
        "model": {
            "directory": str(model),
            "manifest_sha256": manifest_sha,
            "declared_pins": pins,
            "binary_bytes_verified": False,
            "verification_scope": (
                "Manifest/provider-code pins only; provider verifies actual model bytes before load"
            ),
        },
    }
    return identity["command"], identity


def resolve_config(config=None):
    """Return (validated operator config or None, identity), without side effects."""
    value = config if config is not None else os.environ.get(ENVIRONMENT_KEY)
    if value is None or value == "":
        return None, {"enabled": False, "digest": _digest({"enabled": False})}
    base, file_hash = ROOT.resolve(), None
    if isinstance(value, (str, Path)):
        path = _path(str(value), base)
        value, file_hash, base = _read(path), _sha(path), path.parent
    if not isinstance(value, dict):
        raise ValueError("Visual operator configuration must be an object or local JSON path")
    _canonical(value)
    required = {"schema_version", "enabled"}
    allowed = required | {"provider_command", "timeout_seconds", "artifact_root", "critic"}
    if not required <= value.keys() or value.keys() - allowed:
        raise ValueError("Visual operator configuration has missing or unknown fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported visual operator configuration version")
    if type(value["enabled"]) is not bool or type(value.get("critic", True)) is not bool:
        raise ValueError("Visual enabled and critic settings must be booleans")
    timeout = value.get("timeout_seconds", 240)
    if type(timeout) is not int or not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise ValueError("Visual timeout_seconds must be an integer from 1 to 600")
    artifact = _artifact_root(
        value.get("artifact_root", str(ROOT / "output/visual-planner-live")), base
    )
    result = {
        "schema_version": 1,
        "enabled": value["enabled"],
        "timeout_seconds": timeout,
        "artifact_root": artifact,
        "critic": value.get("critic", True),
    }
    identity = {"enabled": result["enabled"], "configuration_file_sha256": file_hash}
    if result["enabled"]:
        command, provider = _provider_identity(value.get("provider_command"), base)
        result["provider_command"] = command
        identity.update(provider)
    identity["configuration_sha256"] = _digest(result)
    identity["digest"] = _digest(identity)
    return deepcopy(result), deepcopy(identity)


def visual_model_enabled(config=None):
    """Invalid explicitly configured models stay enabled for cache-bypass purposes."""
    try:
        resolved, _ = resolve_config(config)
        return resolved is not None and resolved["enabled"]
    except (ValueError, OSError, TypeError):
        return config is not None or bool(os.environ.get(ENVIRONMENT_KEY))


def visual_execution_identity(config=None):
    try:
        resolved, identity = resolve_config(config)
        return {
            "enabled": resolved is not None and resolved["enabled"],
            "valid": True,
            "identity": identity["digest"],
            "pins": identity,
        }
    except (ValueError, OSError, TypeError) as exc:
        return {
            "enabled": config is not None or bool(os.environ.get(ENVIRONMENT_KEY)),
            "valid": False,
            "identity": None,
            "error": str(exc)[:400],
        }
