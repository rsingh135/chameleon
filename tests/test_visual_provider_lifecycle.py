"""Real process regressions: supervisor exit must not leak its owned worker."""

import os
import select
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from engine import visual_pipeline as pipeline


class ChildHandle:
    """Hold an exact child identity while the provider's supervisor exits."""

    def __init__(self, pid):
        self.pid = pid
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            self.kernel.WaitForSingleObject.restype = wintypes.DWORD
            self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            self.kernel.TerminateProcess.restype = wintypes.BOOL
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel.CloseHandle.restype = wintypes.BOOL
            self.handle = self.kernel.OpenProcess(0x100000 | 0x0001, False, pid)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
        elif hasattr(os, "pidfd_open"):
            self.handle = os.pidfd_open(pid)
        else:
            pytest.skip("Exact process-identity probe requires Windows handles or Linux pidfds")

    def exited(self):
        if os.name == "nt":
            return self.kernel.WaitForSingleObject(self.handle, 0) == 0
        return bool(select.select([self.handle], [], [], 0)[0])

    def close(self):
        # Regression cleanup targets this exact owned worker, never a PID search.
        if not self.exited():
            if os.name == "nt":
                self.kernel.TerminateProcess(self.handle, 1)
            else:
                signal.pidfd_send_signal(self.handle, signal.SIGKILL)
        if os.name == "nt":
            self.kernel.CloseHandle(self.handle)
        else:
            os.close(self.handle)


def wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Owned subprocess did not reach its expected state")
        time.sleep(0.01)


@pytest.mark.parametrize("outcome", ["crash", "success", "timeout"])
def test_owned_child_stops_even_after_supervisor_exit_and_unrelated_process_survives(
    tmp_path, outcome
):
    pid_file, gate = tmp_path / "owned-pid.txt", tmp_path / "continue.txt"
    supervisor = "\n".join(
        [
            "import json,os,subprocess,sys,time",
            "from pathlib import Path",
            "json.load(sys.stdin)",
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
            "stdin=subprocess.DEVNULL,stderr=subprocess.DEVNULL)",
            f"Path({str(pid_file)!r}).write_text(str(child.pid))",
            f"while not Path({str(gate)!r}).exists(): time.sleep(0.01)",
            {
                "crash": "os._exit(7)",
                "success": "print(json.dumps({'ok':True}),flush=True)",
                "timeout": "time.sleep(60)",
            }[outcome],
        ]
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                pipeline.run_provider,
                [sys.executable, "-c", supervisor],
                {"test": "read request before spawning"},
                1.5 if outcome == "timeout" else 6,
            )
            try:
                wait_until(pid_file.exists)
                probe = ChildHandle(int(pid_file.read_text()))
                assert not probe.exited()
                started = time.monotonic()
                gate.write_text("continue")
                if outcome == "success":
                    assert future.result(timeout=5) == {"ok": True}
                else:
                    error = "timeout" if outcome == "timeout" else "exited with code 7"
                    with pytest.raises(ValueError, match=error):
                        future.result(timeout=5)
                wait_until(probe.exited)
                assert time.monotonic() - started < 4
                assert unrelated.poll() is None
            finally:
                gate.write_text("continue")
    finally:
        if probe:
            probe.close()
        unrelated.kill()
        unrelated.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object assignment boundary")
def test_windows_assignment_failure_terminates_supervisor_before_delivering_request(
    tmp_path, monkeypatch
):
    received = tmp_path / "received.txt"
    processes = []

    def reject(_self, process):
        processes.append(process)
        raise OSError("Simulated Job Object assignment failure")

    monkeypatch.setattr(pipeline._WindowsProviderJob, "assign", reject)
    program = (
        "import json,sys;from pathlib import Path;json.load(sys.stdin);"
        f"Path({str(received)!r}).write_text('request delivered')"
    )
    with pytest.raises(ValueError, match="owned visual-provider process lifetime"):
        pipeline.run_provider([sys.executable, "-c", program], {"start": True}, timeout=5)
    assert processes and processes[0].poll() is not None
    assert not received.exists()
