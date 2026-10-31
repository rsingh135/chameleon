"""Pull one published checkpoint into an immutable, verified off-VM backup.

The VM must already be RUNNING/READY with a nonexpired guard allowance. This
helper never starts, stops or deletes an instance, and never stores credentials.
Copy the reviewed backup and smoke helpers to remote-root before invoking it.
Local verification proves file integrity, not model quality or useful training.
Use WSL/Linux for transfers: POSIX timeouts kill the CLI process group, including
rsync/SSH children. The Windows fallback bounds only the immediate CLI process.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path, PurePosixPath

# Support both `python tools/brev_sync.py` and package imports in offline tests.
_PREFIX = f"{__package__}." if __package__ else ""
backup = importlib.import_module(_PREFIX + "brev_backup")
guard = importlib.import_module(_PREFIX + "brev_guard")
smoke = importlib.import_module(_PREFIX + "vision_lora_smoke")
MARKER = "CHAMELEON_SNAPSHOT="
DEFAULT_BACKUP_ROOT = guard.DEFAULT_CONFIG.parent / "snapshots"


class SyncError(RuntimeError):
    """A checkpoint could not be transferred and independently verified."""


def run_transfer_command(command: list[str], timeout: float) -> str:
    """Bound a POSIX CLI process group; retain a parent-only Windows fallback."""
    if os.name != "posix":
        return guard.run_command(command, timeout)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as error:
        raise guard.GuardError(f"Cannot execute Brev CLI: {error}") from error
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException as error:
        # A child rsync/SSH must not outlive a timed-out/interrupted CLI parent.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # Cleanup is bounded even if a descendant changed its own session.
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        if isinstance(error, subprocess.TimeoutExpired):
            raise guard.GuardError(
                f"Brev process group timed out after {timeout:g} seconds"
            ) from error
        raise
    if process.returncode:
        raise guard.GuardError(f"Brev command failed with exit code {process.returncode}")
    return output


def remote_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or any(ord(character) < 32 for character in value):
        raise SyncError("Remote paths must be absolute, normalized POSIX paths")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or value.startswith("//")
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise SyncError("Remote paths must be absolute, normalized POSIX paths")
    return path


def checkpoint_source(root: PurePosixPath, source: PurePosixPath) -> None:
    if (
        root == PurePosixPath("/")
        or not source.is_relative_to(root)
        or source.parent.name != "published"
        or not re.fullmatch(r"checkpoint-[0-9]{6}", source.name)
        or any(part.startswith(".") for part in source.relative_to(root).parts)
    ):
        raise SyncError("Source must be one published/checkpoint-NNNNNN leaf under remote-root")


def remote_command(root: PurePosixPath, source: PurePosixPath) -> str:
    code = "\n".join(
        [
            "import json, sys",
            "from pathlib import Path",
            f"root = Path({str(root)!r})",
            f"source = Path({str(source)!r})",
            "sys.path.insert(0, str(root))",
            "from tools.vision_lora_smoke import verify_checkpoint",
            "from tools.brev_backup import snapshot",
            "verify_checkpoint(source)",
            "receipt = snapshot(source, root / '.export-snapshots')",
            f"print({MARKER!r} + json.dumps(receipt, sort_keys=True), flush=True)",
        ]
    )
    return "python3 -c " + shlex.quote(code)


def parse_receipt(output: str, root: PurePosixPath, source: PurePosixPath) -> dict:
    markers = [line[len(MARKER) :] for line in output.splitlines() if line.startswith(MARKER)]
    if len(markers) != 1 or len(markers[0]) > 64 * 1024:
        raise SyncError("Expected exactly one bounded remote snapshot receipt")
    try:
        receipt = json.loads(markers[0])
    except json.JSONDecodeError as error:
        raise SyncError("Remote snapshot receipt is invalid JSON") from error
    if not isinstance(receipt, dict):
        raise SyncError("Remote snapshot receipt must be an object")
    generation = receipt.get("generation")
    digest = receipt.get("manifest_sha256")
    if not isinstance(generation, str) or not backup.GENERATION_PATTERN.fullmatch(generation):
        raise SyncError("Invalid remote snapshot generation identity")
    if not isinstance(digest, str) or not backup.HASH_PATTERN.fullmatch(digest):
        raise SyncError("Invalid remote manifest SHA-256")
    expected = str(root / ".export-snapshots" / generation)
    if receipt.get("generation_dir") != expected or receipt.get("source") != str(source):
        raise SyncError("Remote snapshot path/source does not match the requested checkpoint")
    return receipt


def plain_local_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (*reversed(absolute.parents), absolute):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            raise SyncError("Local backup paths must not contain links or reparse points")
    return absolute


def sync_checkpoint(
    config,
    *,
    remote_root: str,
    remote_source: str,
    backup_root: Path,
    brev: str = "brev",
    command_timeout: float = 120.0,
    runner=run_transfer_command,
    now: Callable[[], datetime] = lambda: datetime.now(guard.UTC),
) -> dict:
    if not math.isfinite(command_timeout) or not 0 < command_timeout <= 120:
        raise SyncError("Command timeout must be greater than zero and at most 120 seconds")
    root, source = remote_path(remote_root), remote_path(remote_source)
    checkpoint_source(root, source)
    local_root = plain_local_path(backup_root)

    def bounded(command: list[str], timeout: float) -> str:
        remaining = (config.deadline - now()).total_seconds()
        if remaining <= 0:
            raise SyncError(
                "Saved billing deadline reached; transfer will not start another command"
            )
        return runner(command, min(timeout, command_timeout, remaining))

    controller = guard.Guard(config, brev=brev, command_timeout=command_timeout, runner=bounded)

    def require_running() -> None:
        status = controller.status()
        if status not in {"RUNNING", "READY"}:
            raise SyncError(
                f"Transfer requires an already RUNNING/READY instance; observed {status}"
            )

    require_running()
    output = controller.command("exec", config.instance_name, remote_command(root, source))
    receipt = parse_receipt(output, root, source)
    require_running()  # Reverify identity/state immediately before using the name for SCP.
    generation = receipt["generation"]
    final = local_root / generation
    if final.exists() or final.is_symlink():
        raise SyncError(
            "Backup generation already exists; no existing generation will be overwritten"
        )
    local_root.mkdir(parents=True, exist_ok=True)
    plain_local_path(local_root)
    incoming = local_root / (".brev-incoming-" + uuid.uuid4().hex)
    incoming.mkdir()
    # Current Brev copy accepts two paths and detects directories itself.
    # -r belongs to its internal SCP fallback, not the Brev command's flags.
    controller.command("copy", f"{config.instance_name}:{receipt['generation_dir']}", str(incoming))
    # Brev/SCP versions differ in whether they copy the directory or its contents.
    candidates = [
        candidate
        for candidate in (incoming, incoming / generation)
        if (candidate / "receipt.json").is_file()
    ]
    if len(candidates) != 1:
        raise SyncError(
            f"Incomplete or ambiguous transfer; unverified files retained in {incoming}"
        )
    candidate = candidates[0]
    observed = backup.verify(candidate)
    for field in ("generation", "manifest_sha256", "source", "file_count", "byte_count"):
        if observed[field] != receipt.get(field):
            raise SyncError(f"Copied snapshot differs from remote receipt: {field}")
    smoke.verify_checkpoint(candidate / "data")
    # These are exclusively created local staging paths; previous generations are untouched.
    if final.exists() or final.is_symlink():
        raise SyncError("Backup generation already exists; publication refused")
    candidate.rename(final)
    if os.name == "posix":
        descriptor = os.open(local_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    verified = backup.verify(final)
    return {
        "verified": True,
        "scope": "one complete checkpoint, byte integrity only; no model quality claim",
        "instance_id": config.instance_id,
        "remote_source": str(source),
        "generation": generation,
        "generation_dir": str(final),
        "manifest_sha256": verified["manifest_sha256"],
        "file_count": verified["file_count"],
        "byte_count": verified["byte_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=guard.DEFAULT_CONFIG)
    parser.add_argument("--brev", default="brev")
    parser.add_argument("--remote-root", default="/home/ubuntu/workspace/chameleon")
    parser.add_argument(
        "--remote-source",
        required=True,
        help="Absolute path to one complete published/checkpoint-NNNNNN leaf",
    )
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        result = sync_checkpoint(
            guard.load_config(args.config),
            remote_root=args.remote_root,
            remote_source=args.remote_source,
            backup_root=args.backup_root,
            brev=args.brev,
            command_timeout=args.command_timeout,
        )
    except (guard.GuardError, SyncError, OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"verified": False, "error": str(error)}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"verified": False, "error": "Transfer interrupted"}), file=sys.stderr)
        return 130
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
