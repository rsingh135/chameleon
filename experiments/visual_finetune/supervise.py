"""Bound one owned worker; never trim or terminate a different application."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
import signal
import subprocess
import time
from contextlib import closing, suppress
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc  # noqa: UP017 -- the approved cloud environment uses Python 3.10.


class _Status(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong)
        for name in (
            "total",
            "available",
            "page_total",
            "page_available",
            "virtual_total",
            "virtual_available",
            "extended",
        )
    ]


class _Memory(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t)
        for name in (
            "peak",
            "rss",
            "paged_peak",
            "paged",
            "nonpaged_peak",
            "nonpaged",
            "pagefile",
            "pagefile_peak",
            "private",
        )
    ]


def memory(handle):
    status = _Status()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    counters = _Memory()
    counters.cb = ctypes.sizeof(counters)
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Memory), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    get_memory(handle, ctypes.byref(counters), counters.cb)
    return {
        "rss_mb": counters.rss / 2**20,
        "peak_rss_mb": counters.peak / 2**20,
        "available_mb": status.available / 2**20,
    }


def trim_owned(worker):
    handle = wintypes.HANDLE(int(worker._handle))
    get_pid = ctypes.windll.kernel32.GetProcessId
    get_pid.argtypes, get_pid.restype = [wintypes.HANDLE], wintypes.DWORD
    if get_pid(handle) != worker.pid:
        raise RuntimeError("Owned process handle and PID disagree")
    trim = ctypes.windll.psapi.EmptyWorkingSet
    trim.argtypes, trim.restype = [wintypes.HANDLE], wintypes.BOOL
    return bool(trim(handle))


def _run_windows(
    command,
    out,
    *,
    cwd,
    environment=None,
    max_seconds=240,
    max_rss_mb=4096,
    minimum_available_mb=256,
    trim=True,
    deadline_utc="2026-10-31T06:31:00+00:00",
):
    if os.name != "nt":
        raise RuntimeError("This measured supervisor currently targets the Windows GPU worker")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = datetime.fromisoformat(deadline_utc)
    peak, minimum, low = 0.0, float("inf"), 0
    status, trims = "exited", 0
    from engine.visual_pipeline import _WindowsProviderJob

    job = _WindowsProviderJob()
    job_class_sha = hashlib.sha256(inspect.getsource(_WindowsProviderJob).encode()).hexdigest()
    environment = dict(os.environ if environment is None else environment)
    environment["VISUAL_ADAPTER_REQUIRE_OWNED_JOB"] = "1"
    with (
        closing(job),
        (out / "stdout.log").open("wb") as stdout,
        (out / "stderr.log").open("wb") as stderr,
    ):
        worker = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        handle = wintypes.HANDLE(int(worker._handle))
        try:
            job.assign(worker)
            worker.stdin.write(b"owned-job-assigned\n")
            worker.stdin.close()
            while worker.poll() is None:
                sample = memory(handle)
                peak = max(peak, sample["peak_rss_mb"], sample["rss_mb"])
                minimum = min(minimum, sample["available_mb"])
                if trim and sample["available_mb"] < 512 and sample["rss_mb"] > 64:
                    before = sample
                    succeeded = trim_owned(worker)
                    sample = memory(handle)
                    minimum = min(minimum, sample["available_mb"])
                    trims += 1
                    with (out / "memory-trims.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "seconds": time.monotonic() - started,
                                    "owned_pid": worker.pid,
                                    "success": succeeded,
                                    "before": before,
                                    "after": sample,
                                }
                            )
                            + "\n"
                        )
                low = low + 1 if sample["available_mb"] < minimum_available_mb else 0
                if low >= 3 or peak > max_rss_mb:
                    raise MemoryError("Owned worker exceeded its host-memory safety budget")
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("Owned worker exceeded its run deadline")
                if datetime.now(UTC) >= deadline or (out / "STOP").exists():
                    raise TimeoutError("Owned worker reached the sprint deadline or stop signal")
                time.sleep(0.5)
        except BaseException as exc:
            status = type(exc).__name__ + ": " + str(exc)
            job.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        finally:
            job.close()
    receipt = {
        "status": status,
        "exit_code": worker.returncode,
        "seconds": time.monotonic() - started,
        "peak_rss_mb": peak,
        "minimum_available_mb": minimum,
        "minimum_available_guard_mb": minimum_available_mb,
        "maximum_owned_rss_mb": max_rss_mb,
        "owned_pid": worker.pid,
        "owned_worker_stopped": worker.poll() is not None,
        "owned_kill_on_close_job": True,
        "job_helper_class_sha256": job_class_sha,
        "owned_working_set_trims": trims,
        "command": command,
        "absolute_deadline_utc": deadline_utc,
    }
    (out / "supervision.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def _linux_memory(pid):
    status = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, *_ = line.split()
                status[key] = float(value) / 1024
    except FileNotFoundError:
        pass
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = float(line.split()[1]) / 1024
    if available is None:
        raise RuntimeError("Linux memory supervision requires MemAvailable")
    return {
        "rss_mb": status.get("VmRSS:", 0),
        "peak_rss_mb": status.get("VmHWM:", 0),
        "available_mb": available,
    }


def _linux_parent_death(parent_pid):
    # The cloud entry point is a standalone, single-threaded supervisor. Arrange
    # kernel cleanup before model import even if that supervisor is killed.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        os._exit(126)


def _run_posix(
    command,
    out,
    *,
    cwd,
    environment=None,
    max_seconds=240,
    max_rss_mb=4096,
    minimum_available_mb=256,
    trim=True,
    deadline_utc="2026-10-31T06:31:00+00:00",
):
    if not Path("/proc/meminfo").is_file() or not hasattr(os, "WNOWAIT"):
        raise RuntimeError("Portable cloud supervision currently requires Linux /proc and waitid")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ if environment is None else environment)
    environment["VISUAL_ADAPTER_REQUIRE_OWNED_JOB"] = "1"
    started, peak, minimum, low = time.monotonic(), 0.0, float("inf"), 0
    status = "exited"
    deadline = datetime.fromisoformat(deadline_utc)
    with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
        parent_pid = os.getpid()
        worker = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=lambda: _linux_parent_death(parent_pid),
        )
        try:
            worker.stdin.write(b"owned-job-assigned\n")
            worker.stdin.close()
            # Retain an exited leader unreaped until its private group is killed. Its PID
            # cannot be reused for an unrelated process group during cleanup.
            while os.waitid(os.P_PID, worker.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                sample = _linux_memory(worker.pid)
                peak = max(peak, sample["peak_rss_mb"], sample["rss_mb"])
                minimum = min(minimum, sample["available_mb"])
                low = low + 1 if sample["available_mb"] < minimum_available_mb else 0
                if low >= 3 or peak > max_rss_mb:
                    raise MemoryError("Owned worker exceeded its host-memory safety budget")
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("Owned worker exceeded its run deadline")
                if datetime.now(UTC) >= deadline or (out / "STOP").exists():
                    raise TimeoutError("Owned worker reached the sprint deadline or stop signal")
                time.sleep(0.5)
        except BaseException as exc:
            status = type(exc).__name__ + ": " + str(exc)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
    receipt = {
        "status": status,
        "exit_code": worker.returncode,
        "seconds": time.monotonic() - started,
        "peak_rss_mb": peak,
        "minimum_available_mb": minimum if minimum != float("inf") else None,
        "minimum_available_guard_mb": minimum_available_mb,
        "maximum_owned_rss_mb": max_rss_mb,
        "owned_pid": worker.pid,
        "owned_worker_stopped": True,
        "owned_private_process_group": True,
        "owned_parent_death_signal": True,
        "owned_working_set_trims": 0,
        "working_set_trim_applicable": False,
        "command": command,
        "absolute_deadline_utc": deadline_utc,
    }
    (out / "supervision.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def run(command, out, **kwargs):
    """Use the measured Windows job or an unreaped, private Linux process group."""
    if os.name == "nt":
        return _run_windows(command, out, **kwargs)
    return _run_posix(command, out, **kwargs)
