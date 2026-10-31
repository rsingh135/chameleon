"""Enforce a saved Brev stop deadline through the authenticated local CLI.

This is a best-effort controller, not a hard spending guarantee. It must stay
running on an available controller host. API outages, host sleep and shared
organization usage can defeat estimates. Stopped storage still costs money.
No credentials are accepted or stored, and this tool never deletes instances.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# The standalone controller also runs on Ubuntu 22.04's Python 3.10.
UTC = timezone.utc  # noqa: UP017 -- datetime.UTC requires Python 3.11.
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "local_backups/brev/guard.json"
Runner = Callable[[list[str], float], str]


class GuardError(Exception):
    """An operation could not be verified."""


class IdentityError(GuardError):
    """Readable control-plane data conflicts with the configured target."""


def utc_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamps must be ISO strings with a UTC offset or Z")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamps must include a UTC offset or Z")
    return parsed.astimezone(UTC)


def iso_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Config:
    instance_id: str
    instance_name: str
    org_id: str
    org_name: str
    started_at: datetime
    deadline: datetime
    padded_hourly_usd: float = 2.0
    reserve_usd: float = 10.0

    def __post_init__(self) -> None:
        for name in ("instance_id", "instance_name", "org_id", "org_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a nonempty identifier without outer whitespace")
            if value.startswith("-") or any(ord(character) < 32 for character in value):
                raise ValueError(f"Invalid {name}")
        for value in (self.started_at, self.deadline):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Timestamps must be timezone aware")
        if self.deadline <= self.started_at:
            raise ValueError("The deadline must be after started_at")
        for name in ("padded_hourly_usd", "reserve_usd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.padded_hourly_usd == 0:
            raise ValueError("padded_hourly_usd must be positive")

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "instance_id": self.instance_id,
            "instance_name": self.instance_name,
            "org_id": self.org_id,
            "org_name": self.org_name,
            "started_at": iso_time(self.started_at),
            "deadline": iso_time(self.deadline),
            "padded_hourly_usd": self.padded_hourly_usd,
            "reserve_usd": self.reserve_usd,
            "reserve_note": "Planning reserve only; no live balance API or account cap.",
        }

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported or missing guard config schema_version")
        return cls(
            instance_id=data["instance_id"],
            instance_name=data["instance_name"],
            org_id=data["org_id"],
            org_name=data["org_name"],
            started_at=utc_time(data["started_at"]),
            deadline=utc_time(data["deadline"]),
            padded_hourly_usd=data["padded_hourly_usd"],
            reserve_usd=data["reserve_usd"],
        )


def run_command(command: list[str], timeout: float) -> str:
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise GuardError(f"Brev command timed out after {timeout:g} seconds") from error
    except OSError as error:
        raise GuardError(f"Cannot execute Brev CLI: {error}") from error
    if result.returncode:
        # Do not echo CLI output: login prompts or debug output can contain private data.
        raise GuardError(f"Brev command failed with exit code {result.returncode}")
    return result.stdout


class Guard:
    def __init__(
        self,
        config: Config,
        *,
        brev: str = "brev",
        command_timeout: float = 30.0,
        attempts: int = 5,
        poll_seconds: float = 5.0,
        runner: Runner = run_command,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not brev or not math.isfinite(command_timeout) or command_timeout <= 0:
            raise ValueError("A Brev executable and positive command timeout are required")
        if attempts < 1 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("Attempts and poll interval must be positive")
        self.config = config
        self.brev = brev
        self.command_timeout = command_timeout
        self.attempts = attempts
        self.poll_seconds = poll_seconds
        self.runner = runner
        self.sleep = sleep

    def command(self, *arguments: str, scoped: bool = True) -> str:
        command = [self.brev]
        if scoped:
            # The live CLI resolves --org as a name, not an ID. verify_org()
            # separately binds that name to the configured immutable org ID.
            command += ["--org", self.config.org_name]
        return self.runner(command + list(arguments), self.command_timeout)

    def json_command(self, *arguments: str, scoped: bool = True):
        try:
            return json.loads(self.command(*arguments, scoped=scoped))
        except json.JSONDecodeError as error:
            raise GuardError("Brev returned invalid JSON") from error

    def verify_org(self) -> None:
        organizations = self.json_command("ls", "orgs", "--json", scoped=False)
        if not isinstance(organizations, list) or not all(
            isinstance(item, dict) for item in organizations
        ):
            raise GuardError("Unexpected Brev organization JSON shape")
        matches = [item for item in organizations if item.get("id") == self.config.org_id]
        if len(matches) != 1 or matches[0].get("name") != self.config.org_name:
            raise IdentityError("Configured organization ID/name could not be verified")

    def status(self) -> str:
        self.verify_org()
        payload = self.json_command("ls", "--json")
        if not isinstance(payload, dict) or not isinstance(payload.get("workspaces"), list):
            raise GuardError("Unexpected Brev workspace JSON shape")
        workspaces = payload["workspaces"]
        if not all(isinstance(item, dict) for item in workspaces):
            raise GuardError("Unexpected Brev workspace record")
        matches = [item for item in workspaces if item.get("id") == self.config.instance_id]
        if len(matches) != 1:
            raise IdentityError("Exact configured instance ID was not found uniquely")
        target = matches[0]
        if target.get("name") != self.config.instance_name:
            raise IdentityError("Configured instance ID has an unexpected name")
        status = target.get("status")
        if not isinstance(status, str) or not status.strip():
            raise GuardError("Configured instance has no usable control-plane status")
        return status.strip().upper()

    def report(self, now: datetime, action: str, status: str | None = None) -> dict:
        elapsed = max(0.0, (now - self.config.started_at).total_seconds())
        return {
            "action": action,
            "instance_id": self.config.instance_id,
            "instance_name": self.config.instance_name,
            "org_id": self.config.org_id,
            "observed_at": iso_time(now),
            "started_at": iso_time(self.config.started_at),
            "deadline": iso_time(self.config.deadline),
            "deadline_reached": now >= self.config.deadline,
            "seconds_until_deadline": max(0.0, (self.config.deadline - now).total_seconds()),
            "status": status,
            "confirmed_stopped": status == "STOPPED",
            "estimated_elapsed_usd": round(elapsed / 3600 * self.config.padded_hourly_usd, 4),
            "padded_hourly_usd": self.config.padded_hourly_usd,
            "reserve_usd": self.config.reserve_usd,
            "estimate_note": "Estimate only, excludes unknown shared usage; storage continues.",
            "errors": [],
        }

    def stop(self, now: datetime) -> dict:
        report = self.report(now, "stop")
        for attempt in range(self.attempts):
            try:
                report["status"] = self.status()
                if report["status"] == "STOPPED":
                    report["confirmed_stopped"] = True
                    return report
            except IdentityError as error:
                report["errors"].append(str(error))
                report["identity_mismatch"] = True
                return report
            except GuardError as error:
                # Listing/network failure must not disable the saved deadline.
                # The fallback still uses only the fixed ID and configured org name.
                report["errors"].append(str(error))
            try:
                self.command("stop", self.config.instance_id)
            except GuardError as error:
                report["errors"].append(str(error))
            report["stop_attempts"] = attempt + 1
            self.sleep(self.poll_seconds)
            try:
                report["status"] = self.status()
                if report["status"] == "STOPPED":
                    report["confirmed_stopped"] = True
                    return report
            except IdentityError as error:
                report["errors"].append(str(error))
                report["identity_mismatch"] = True
                return report
            except GuardError as error:
                report["errors"].append(str(error))
        report["errors"].append("Stop was not confirmed; compute billing may continue")
        return report

    def check(self, now: datetime) -> dict:
        if now >= self.config.deadline:
            return self.stop(now)
        report = self.report(now, "check")
        try:
            report["status"] = self.status()
            report["confirmed_stopped"] = report["status"] == "STOPPED"
        except GuardError as error:
            report["errors"].append(str(error))
            report["identity_mismatch"] = isinstance(error, IdentityError)
        return report


def save_new_config(path: Path, config: Config, guard: Guard) -> None:
    if path.exists():
        raise FileExistsError("Guard config already exists; its deadline will not be reset")
    guard.status()  # Verify both fixed identities before arming this new allowance.
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(config.as_dict(), output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def load_config(path: Path) -> Config:
    return Config.from_dict(json.loads(path.read_text(encoding="utf-8")))


def report_exit_code(report: dict) -> int:
    if report["action"] == "stop" or report["deadline_reached"]:
        return 0 if report["confirmed_stopped"] else 2
    return 2 if report["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--brev", default="brev", help="Brev executable (not a shell command)")
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument(
        "--attempts", type=int, default=5, help="Stop attempts per enforcement cycle"
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="Verify and save a new fixed deadline; never overwrites"
    )
    init.add_argument("--instance-id", required=True)
    init.add_argument("--instance-name", required=True)
    init.add_argument("--org-id", required=True)
    init.add_argument("--org-name", required=True)
    init.add_argument("--started-at", required=True, help="Absolute ISO timestamp with UTC offset")
    init.add_argument("--deadline", required=True, help="Absolute ISO timestamp with UTC offset")
    init.add_argument("--padded-hourly-usd", type=float, default=2.0)
    init.add_argument(
        "--reserve-usd", type=float, default=10.0, help="Planning note, not a live cap"
    )
    commands.add_parser(
        "check", help="Inspect state; enforce stop if the saved deadline has passed"
    )
    commands.add_parser("stop", help="Request and confirm stop now, without deleting data")
    watch = commands.add_parser(
        "watch", help="Enforce deadline while this controller host remains available"
    )
    watch.add_argument(
        "--wait-for-start",
        action="store_true",
        help=(
            "Keep watching an initially STOPPED instance until RUNNING/READY has been seen "
            "or the fixed deadline is reached; arm before starting compute"
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            config = Config(
                args.instance_id,
                args.instance_name,
                args.org_id,
                args.org_name,
                utc_time(args.started_at),
                utc_time(args.deadline),
                args.padded_hourly_usd,
                args.reserve_usd,
            )
        else:
            config = load_config(args.config)
        guard = Guard(
            config,
            brev=args.brev,
            command_timeout=args.command_timeout,
            attempts=args.attempts,
            poll_seconds=args.poll_seconds,
        )
        if args.command == "init":
            save_new_config(args.config, config, guard)
            print(json.dumps({"action": "armed", "config": str(args.config), **config.as_dict()}))
            return 0
        running_seen = False
        wait_for_start = getattr(args, "wait_for_start", False)
        while True:
            now = datetime.now(UTC)
            report = guard.stop(now) if args.command == "stop" else guard.check(now)
            running_seen = running_seen or report["status"] in {"RUNNING", "READY"}
            awaiting_start = wait_for_start and not running_seen and not report["deadline_reached"]
            if args.command == "watch":
                report["awaiting_start"] = awaiting_start
            print(json.dumps(report), flush=True)
            result = report_exit_code(report)
            if args.command != "watch" or (report["confirmed_stopped"] and not awaiting_start):
                return result
            if report.get("identity_mismatch"):
                return 2
            # Retry failed enforcement cycles indefinitely until confirmed or interrupted.
            remaining = (config.deadline - datetime.now(UTC)).total_seconds()
            if remaining > 0:
                guard.sleep(min(guard.poll_seconds, remaining))
            elif report["deadline_reached"]:
                guard.sleep(guard.poll_seconds)
    except (GuardError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"action": "error", "error": str(error)}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Guard interrupted; stop is NOT confirmed and compute billing may continue.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
