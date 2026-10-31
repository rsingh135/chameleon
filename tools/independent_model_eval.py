"""Collect bounded local model transcripts against a frozen evaluation bundle.

Dry-run is the default. Method identity is operator-declared metadata, not an
attestation that an executable loaded particular weights. This collector does
not judge responses, train models, download weights, or unseal test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

MAX_OUTPUT_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 24 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
_EXEC_GATE = """import os,subprocess,sys,time
while not os.path.isfile(sys.argv[1]): time.sleep(0.01)
try:
    if os.name == 'nt':
        sys.exit(subprocess.call(sys.argv[2:], stdin=sys.stdin, stdout=sys.stdout,
                                 stderr=sys.stderr, shell=False))
    else:
        os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
except OSError as error:
    print('Could not execute declared command: '+str(error), file=sys.stderr, flush=True)
    sys.exit(127)
"""


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _parse(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("JSON must contain finite values")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def _read_file(path, maximum):
    path = Path(path)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        or info.st_size > maximum
    ):
        raise ValueError(f"Expected a regular file within the byte limit: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"File exceeded its byte limit: {path.name}")
    return data


def _load_bundle(path):
    from experiments.vision_critic.independent_probes import load_bundle

    return load_bundle(path)


def _request_digest(value):
    from experiments.vision_critic.independent_probes import request_digest

    return request_digest(value)


def _write_json(path, value):
    Path(path).write_bytes(_canonical(value) + b"\n")


def _number(value, name, maximum):
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be finite and in (0,{maximum}]")


def _command(value):
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 64
        or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in value)
        or sum(len(arg.encode("utf-8")) for arg in value) > 64 * 1024
    ):
        raise ValueError("command-json must be a nonempty bounded JSON argv list")
    return list(value)


def _snapshots(out):
    paths = (
        Path(__file__).resolve(),
        ROOT / "experiments/vision_critic/independent_probes.py",
        ROOT / "engine/visual_pipeline.py",
    )
    hashes = {}
    for path in paths:
        data = _read_file(path, MAX_REQUEST_BYTES)
        relative = path.relative_to(ROOT).as_posix()
        target = out / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return hashes


def _verify_frozen(bundle_path, expected_bundle, identity_path, identity_sha, code_hashes):
    current = _load_bundle(bundle_path)
    if current["bundle_sha256"] != expected_bundle:
        raise ValueError("Frozen bundle identity changed")
    if hashlib.sha256(_read_file(identity_path, MAX_OUTPUT_BYTES)).hexdigest() != identity_sha:
        raise ValueError("Declared method identity file changed during collection")
    for relative, expected in code_hashes.items():
        if hashlib.sha256(_read_file(ROOT / relative, MAX_REQUEST_BYTES)).hexdigest() != expected:
            raise ValueError(f"Collector code changed during collection: {relative}")
    return current


def _run_process(argv, request_path, response_path, timeout, cwd):
    # These helpers use only the standard library. Keep the private session's
    # leader unreaped until its descendants have been stopped on POSIX; Windows
    # keeps exact process ownership in a kernel Job Object.
    from engine.visual_pipeline import _wait_provider_without_reaping, _WindowsProviderJob

    gate = response_path.parent / "execution-gate"
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
    )
    process, job, started = None, None, False
    failures, readers = [], []
    stop_lock = threading.Lock()
    stopped = False

    def stop():
        nonlocal stopped
        with stop_lock:
            if stopped or process is None:
                return
            if job:
                job.terminate()
            elif os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
            stopped = True

    def receive(pipe, destination, name):
        size = 0
        try:
            while chunk := pipe.read(8192):
                remaining = MAX_OUTPUT_BYTES - size
                destination.write(chunk[:remaining])
                size += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    failures.append(f"Declared command exceeded its {name} byte limit")
                    stop()
                    break
        except OSError as error:
            failures.append(f"Could not read {name}: {error}")

    with (
        request_path.open("rb") as request,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        try:
            job = _WindowsProviderJob() if os.name == "nt" else None
            process = subprocess.Popen(
                [sys.executable, "-c", _EXEC_GATE, str(gate), *argv],
                stdin=request,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=environment,
                shell=False,
                start_new_session=os.name == "posix",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            started = True
            if job:
                try:
                    job.assign(process)
                except OSError:
                    # The gate has not been opened; no declared command or
                    # descendant can have started outside the owned Job Object.
                    process.kill()
                    process.wait(timeout=5)
                    stopped = True
                    raise
            for pipe, destination, name in (
                (process.stdout, stdout, "stdout"),
                (process.stderr, stderr, "stderr"),
            ):
                reader = threading.Thread(
                    target=receive, args=(pipe, destination, name), daemon=True
                )
                reader.start()
                readers.append(reader)
            gate.write_bytes(b"start\n")
            _wait_provider_without_reaping(process, timeout)
        except subprocess.TimeoutExpired:
            failures.append("Declared command exceeded its request timeout")
        except (OSError, ValueError) as error:
            failures.append(f"Declared command failed: {error}")
        finally:
            try:
                stop()
                if process is not None:
                    process.wait(timeout=5)
                for reader in readers:
                    reader.join(timeout=2)
                if any(reader.is_alive() for reader in readers):
                    failures.append("Declared command output did not close after termination")
            finally:
                if process is not None:
                    for pipe in (process.stdout, process.stderr):
                        if pipe:
                            pipe.close()
                if job:
                    job.close()
        if process is not None and process.returncode and not failures:
            failures.append(f"Declared command exited with code {process.returncode}")
        stdout.seek(0)
        stderr.seek(0)
        return started, stdout.read(), stderr.read(), failures


def collect(
    bundle_path,
    method_name,
    method_identity_path,
    protocol,
    command,
    out,
    *,
    execute=False,
    max_probes=20,
    request_timeout=120,
    max_total_minutes=10,
):
    """Create one immutable run directory; return its plan or completion summary."""
    if type(max_probes) is not int or not 1 <= max_probes <= 200:
        raise ValueError("max-probes must be an integer in [1,200]")
    _number(request_timeout, "request-timeout", 900)
    _number(max_total_minutes, "max-total-minutes", 60)
    if type(execute) is not bool or protocol not in {"provider", "critic"}:
        raise ValueError("An explicit execution boolean and provider/critic protocol are required")
    if (
        not isinstance(method_name, str)
        or not method_name.strip()
        or len(method_name) > 200
        or any(ord(character) < 32 for character in method_name)
    ):
        raise ValueError("method-name must be nonempty bounded text")
    command = _command(command)
    bundle_path, identity_path, out = (
        Path(bundle_path).resolve(),
        Path(method_identity_path).resolve(),
        Path(out).absolute(),
    )
    bundle = _load_bundle(bundle_path)
    identity_bytes = _read_file(identity_path, MAX_OUTPUT_BYTES)
    identity = _parse(identity_bytes.decode("utf-8"))
    if not isinstance(identity, dict) or not identity:
        raise ValueError("method-identity must be a nonempty JSON object")
    identity_sha = hashlib.sha256(identity_bytes).hexdigest()
    if out.resolve().is_relative_to(bundle_path.parent):
        raise ValueError("Transcript output must be outside the frozen bundle")
    probes = bundle["probes"][:max_probes]
    if not probes:
        raise ValueError("Frozen bundle contains no probes")
    plan = {
        "schema_version": 1,
        "bundle_sha256": bundle["bundle_sha256"],
        "method_name": method_name,
        "method_identity": identity,
        "method_identity_sha256": _digest(identity),
        "method_identity_scope": "operator_declared_metadata_not_attested_loaded_weights",
        "protocol": protocol,
        "command": command,
        "execute": execute,
        "bundle_probe_count": len(bundle["probes"]),
        "planned_probe_ids": [probe["probe_id"] for probe in probes],
        "probe_limit_truncated": len(probes) < len(bundle["probes"]),
        "limits": {
            "max_probes": max_probes,
            "request_timeout_seconds": request_timeout,
            "max_total_minutes": max_total_minutes,
            "stdout_stderr_byte_limit": MAX_OUTPUT_BYTES,
        },
    }
    # Verification and all validation precede publication of any output path.
    out.mkdir(parents=True, exist_ok=False)
    code_hashes = _snapshots(out)
    (out / "method-identity.json").write_bytes(identity_bytes)
    manifest = {
        **plan,
        "bundle_path": str(bundle_path),
        "method_identity_file_sha256": identity_sha,
        "collector_code_hashes": code_hashes,
        "status": "collecting" if execute else "dry_run",
    }
    _write_json(out / "plan.json", plan)
    _write_json(out / "run-manifest.json", manifest)
    if not execute:
        return plan
    deadline = time.monotonic() + max_total_minutes * 60
    completed = executed = failed = 0
    with (out / "transcripts.jsonl").open("x", encoding="utf-8") as transcript:
        for index, probe in enumerate(probes):
            row = {
                "probe_id": probe["probe_id"],
                "bundle_sha256": bundle["bundle_sha256"],
                "request_digest": probe["request_digest"],
                "method_name": method_name,
                "method_identity": deepcopy(identity),
                "method_identity_sha256": _digest(identity),
                "protocol": protocol,
                "executed": False,
                "fallback": False,
                "response": None,
                "raw_response": None,
                "error": None,
            }
            errors = []
            try:
                _verify_frozen(
                    bundle_path, bundle["bundle_sha256"], identity_path, identity_sha, code_hashes
                )
                request_path = (bundle_path.parent / probe["request_path"]).resolve()
                if not request_path.is_relative_to(bundle_path.parent):
                    raise ValueError("Probe request leaves the frozen bundle")
                request = _parse(_read_file(request_path, MAX_REQUEST_BYTES).decode("utf-8"))
                if _request_digest(request) != probe["request_digest"]:
                    raise ValueError("Probe request digest changed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Collection total time budget exhausted before this probe")
                job_dir = out / "jobs" / f"probe-{index:04d}"
                job_dir.mkdir(parents=True)
                frozen_request, response_path = job_dir / "request.json", job_dir / "response.json"
                _write_json(frozen_request, request)
                argv = [
                    arg.replace("{request}", str(frozen_request)).replace(
                        "{response}", str(response_path)
                    )
                    for arg in command
                ]
                started, stdout, stderr, failures = _run_process(
                    argv,
                    frozen_request,
                    response_path,
                    min(request_timeout, remaining),
                    bundle_path.parent,
                )
                row["executed"] = started
                errors.extend(failures)
                (job_dir / "stdout.txt").write_bytes(stdout)
                (job_dir / "stderr.txt").write_bytes(stderr)
                raw = stdout.decode("utf-8", errors="replace")
                selected = (
                    _read_file(response_path, MAX_OUTPUT_BYTES).decode("utf-8")
                    if response_path.exists()
                    else raw
                )
                row["raw_response"] = (selected if protocol == "critic" else raw) or None
                try:
                    parsed = _parse(selected)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    row["response"] = parsed
                    row["fallback"] = parsed.get("fallback") is True
                    if "fallback" in parsed and type(parsed["fallback"]) is not bool:
                        errors.append("Response fallback flag must be boolean")
                elif protocol == "provider":
                    errors.append("Provider response is not a JSON object")
            except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
                errors.append(str(error))
            finally:
                try:
                    _verify_frozen(
                        bundle_path,
                        bundle["bundle_sha256"],
                        identity_path,
                        identity_sha,
                        code_hashes,
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append(f"Post-probe verification failed: {error}")
            row["error"] = "; ".join(dict.fromkeys(errors))[:4000] or None
            transcript.write(_canonical(row).decode("utf-8") + "\n")
            transcript.flush()
            completed += 1
            executed += row["executed"]
            failed += row["error"] is not None
    manifest.update(
        status="completed", transcript_rows=completed, executed_rows=executed, error_rows=failed
    )
    _write_json(out / "run-manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--method-identity", type=Path, required=True)
    parser.add_argument("--protocol", choices=("provider", "critic"), required=True)
    parser.add_argument("--command-json", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-probes", type=int, default=20)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--max-total-minutes", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        report = collect(
            args.bundle,
            args.method_name,
            args.method_identity,
            args.protocol,
            _parse(args.command_json),
            args.out,
            execute=args.execute,
            max_probes=args.max_probes,
            request_timeout=args.request_timeout,
            max_total_minutes=args.max_total_minutes,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Independent transcript collection failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
