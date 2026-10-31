"""Bounded transcript collection with benign Python fixtures, never model inference."""

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools import independent_model_eval as collector

ROW_FIELDS = {
    "probe_id",
    "bundle_sha256",
    "request_digest",
    "method_name",
    "method_identity",
    "method_identity_sha256",
    "protocol",
    "executed",
    "fallback",
    "response",
    "raw_response",
    "error",
}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    requests = []
    for index in range(2):
        value = {"probe": f"probe-{index}", "content": "original café"}
        path = root / f"request-{index}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        requests.append(
            {
                "probe_id": value["probe"],
                "request_path": path.name,
                "request_digest": hashlib.sha256(encoded).hexdigest(),
            }
        )
    bundle = {"bundle_sha256": "a" * 64, "probes": requests}
    bundle_path = root / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    expected = {path: path.read_bytes() for path in root.iterdir()}
    checks = []

    def load(path):
        checks.append(Path(path))
        if any(path.read_bytes() != data for path, data in expected.items()):
            raise ValueError("Frozen bundle bytes changed")
        return json.loads(bundle_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(collector, "_load_bundle", load)
    monkeypatch.setattr(
        collector,
        "_request_digest",
        lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )

    def snapshot(out):
        # Isolate collector unit tests from the probe builder's evolving bundle
        # schema while retaining real code-copy/hash checks and subprocess work.
        relative = "tools/independent_model_eval.py"
        data = Path(collector.__file__).read_bytes()
        target = out / "code" / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(data)
        return {relative: hashlib.sha256(data).hexdigest()}

    monkeypatch.setattr(collector, "_snapshots", snapshot)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        '{"kind":"benign-test-fixture","revision":"déclaré"}\n', encoding="utf-8"
    )
    return bundle_path, identity_path, checks


def run(frozen, tmp_path, command, *, protocol="provider", execute=True, **options):
    bundle, identity, _ = frozen
    options.setdefault("request_timeout", 5)
    out = tmp_path / "run"
    report = collector.collect(
        bundle, "benign fixture", identity, protocol, command, out, execute=execute, **options
    )
    rows_path = out / "transcripts.jsonl"
    rows = (
        [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        if rows_path.exists()
        else []
    )
    for row in rows:
        assert set(row) == ROW_FIELDS
    return report, rows, out


def test_default_cli_dry_run_never_starts_a_process_and_snapshots_declared_identity(
    frozen, tmp_path, monkeypatch, capsys
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Dry run must not start a process")

    monkeypatch.setattr(collector.subprocess, "Popen", forbidden)
    bundle, identity, _ = frozen
    out = tmp_path / "dry"
    assert (
        collector.main(
            [
                "--bundle",
                str(bundle),
                "--method-name",
                "declared only",
                "--method-identity",
                str(identity),
                "--protocol",
                "provider",
                "--command-json",
                '["not-a-model"]',
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    manifest = json.loads((out / "run-manifest.json").read_text(encoding="utf-8"))
    assert report["execute"] is False and manifest["status"] == "dry_run"
    assert (
        manifest["method_identity_scope"]
        == "operator_declared_metadata_not_attested_loaded_weights"
    )
    assert manifest["method_identity_sha256"] == collector._digest(
        json.loads(identity.read_text(encoding="utf-8"))
    )
    assert (
        manifest["method_identity_file_sha256"] == hashlib.sha256(identity.read_bytes()).hexdigest()
    )
    assert (out / "method-identity.json").read_bytes() == identity.read_bytes()
    assert (out / "code/tools/independent_model_eval.py").read_bytes() == Path(
        collector.__file__
    ).read_bytes()
    assert not (out / "transcripts.jsonl").exists()


def test_stdin_request_and_literal_file_placeholders_collect_in_frozen_order(frozen, tmp_path):
    program = tmp_path / "benign_provider.py"
    program.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "request=json.load(sys.stdin)\n"
        "assert request==json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "print(json.dumps({'trace':request['probe']}),flush=True)\n"
        "Path(sys.argv[2]).write_text(json.dumps({'probe':request['probe'],"
        "'fallback':False,'offline':os.environ['HF_HUB_OFFLINE']}),encoding='utf-8')\n",
        encoding="utf-8",
    )
    report, rows, _out = run(
        frozen, tmp_path, [sys.executable, str(program), "{request}", "{response}"]
    )
    assert report["executed_rows"] == report["transcript_rows"] == 2
    assert report["error_rows"] == 0
    assert [row["probe_id"] for row in rows] == ["probe-0", "probe-1"]
    for row in rows:
        assert row["executed"] is True and row["fallback"] is False and row["error"] is None
        assert row["response"] == {"probe": row["probe_id"], "fallback": False, "offline": "1"}
        assert json.loads(row["raw_response"]) == {"trace": row["probe_id"]}
    assert len(frozen[2]) == 5  # Initial validation plus before/after each attempted probe.


def test_critic_response_file_is_preserved_as_raw_text_without_fabricating_validity(
    frozen, tmp_path
):
    command = [
        sys.executable,
        "-c",
        "import sys;from pathlib import Path;"
        'Path(sys.argv[1]).write_bytes(b\'```json\\n{\\"choice\\":\\"B\\"}\\n```\');'
        "print('trace only')",
        "{response}",
    ]
    _report, rows, _out = run(frozen, tmp_path, command, protocol="critic", max_probes=1)
    assert rows[0]["raw_response"] == '```json\n{"choice":"B"}\n```'
    assert rows[0]["response"] is None
    assert rows[0]["error"] is None and rows[0]["executed"] is True
    assert rows[0]["fallback"] is False


def test_nonzero_exit_and_explicit_fallback_never_become_success(frozen, tmp_path):
    command = [
        sys.executable,
        "-c",
        "import json,sys;print(json.dumps({'fallback':True}));sys.exit(7)",
    ]
    report, rows, _out = run(frozen, tmp_path, command, max_probes=1)
    assert report["error_rows"] == 1
    assert rows[0]["executed"] is True and rows[0]["fallback"] is True
    assert "exited with code 7" in rows[0]["error"]


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_captured_streams_are_byte_capped_and_overflow_is_an_error(frozen, tmp_path, stream):
    command = [
        sys.executable,
        "-c",
        f"import sys;sys.{stream}.buffer.write(b'x'*({collector.MAX_OUTPUT_BYTES}+8192));"
        f"sys.{stream}.flush()",
    ]
    _report, rows, out = run(frozen, tmp_path, command, max_probes=1, request_timeout=5)
    assert rows[0]["executed"] is True
    assert f"{stream} byte limit" in rows[0]["error"]
    assert (out / f"jobs/probe-0000/{stream}.txt").stat().st_size == collector.MAX_OUTPUT_BYTES
    if stream == "stdout":
        assert len(rows[0]["raw_response"].encode()) == collector.MAX_OUTPUT_BYTES


def test_start_failure_remains_an_unexecuted_row(frozen, tmp_path, monkeypatch):
    def failed(*_args, **_kwargs):
        raise FileNotFoundError("injected process startup failure")

    monkeypatch.setattr(collector.subprocess, "Popen", failed)
    _report, rows, _out = run(frozen, tmp_path, ["benign-placeholder"], max_probes=1)
    assert rows[0]["executed"] is False and rows[0]["fallback"] is False
    assert rows[0]["response"] is None and "startup failure" in rows[0]["error"]


@pytest.mark.parametrize("changed", ["bundle", "identity"])
def test_frozen_input_changes_are_reported_and_later_probes_are_not_executed(
    frozen, tmp_path, changed
):
    bundle, identity, _checks = frozen
    target = bundle.parent / "request-0.json" if changed == "bundle" else identity
    command = [
        sys.executable,
        "-c",
        "import sys;from pathlib import Path;Path(sys.argv[1]).write_text('{}');print('{}')",
        str(target),
    ]
    report, rows, _out = run(frozen, tmp_path, command)
    assert rows[0]["executed"] is True and "Post-probe verification failed" in rows[0]["error"]
    assert rows[1]["executed"] is False and rows[1]["error"]
    assert report["executed_rows"] == 1 and report["error_rows"] == 2


def test_request_timeout_preserves_failure_and_stops_only_the_owned_process_tree(frozen, tmp_path):
    from test_visual_provider_lifecycle import ChildHandle, wait_until

    pid_file = tmp_path / "owned-child.txt"
    command = [
        sys.executable,
        "-c",
        "import subprocess,sys,time;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "Path(sys.argv[1]).write_text(str(child.pid));time.sleep(60)",
        str(pid_file),
    ]
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    handle = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Exercise cleanup of a running descendant, not cold Python startup
            # speed. Windows can need over a second to launch the gated chain.
            future = pool.submit(run, frozen, tmp_path, command, max_probes=1, request_timeout=5)
            wait_until(pid_file.exists, seconds=10)
            handle = ChildHandle(int(pid_file.read_text()))
            _report, rows, _out = future.result(timeout=15)
        wait_until(handle.exited)
        assert rows[0]["executed"] is True and rows[0]["fallback"] is False
        assert "request timeout" in rows[0]["error"]
        assert unrelated.poll() is None
    finally:
        if handle:
            handle.close()
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_total_budget_exhaustion_retains_unexecuted_probe_rows(frozen, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(collector.time, "monotonic", lambda: clock[0])

    def consume(*_args, **_kwargs):
        clock[0] += 61
        return True, b"{}", b"", []

    monkeypatch.setattr(collector, "_run_process", consume)
    report, rows, _out = run(frozen, tmp_path, ["test-fixture"], max_total_minutes=1)
    assert rows[0]["executed"] is True and rows[0]["error"] is None
    assert rows[1]["executed"] is False and "total time budget" in rows[1]["error"]
    assert report["transcript_rows"] == 2


def test_probe_cap_is_explicit_and_run_directories_cannot_be_reused(frozen, tmp_path):
    report, rows, out = run(frozen, tmp_path, ["not-executed"], execute=False, max_probes=1)
    assert rows == [] and report["planned_probe_ids"] == ["probe-0"]
    assert report["bundle_probe_count"] == 2 and report["probe_limit_truncated"] is True
    before = (out / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(frozen, tmp_path, ["not-executed"], execute=False)
    assert (out / "plan.json").read_bytes() == before


@pytest.mark.parametrize(
    "options",
    [
        {"max_probes": 0},
        {"max_probes": 201},
        {"max_probes": True},
        {"request_timeout": 901},
        {"request_timeout": float("nan")},
        {"max_total_minutes": 61},
        {"max_total_minutes": True},
    ],
)
def test_invalid_resource_limits_fail_before_creating_any_output(frozen, tmp_path, options):
    with pytest.raises(ValueError):
        run(frozen, tmp_path, ["not-executed"], **options)
    assert not (tmp_path / "run").exists()


def test_failed_initial_bundle_verification_creates_no_output(frozen, tmp_path):
    bundle, _identity, _checks = frozen
    bundle.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="bundle bytes"):
        run(frozen, tmp_path, ["not-executed"])
    assert not (tmp_path / "run").exists()


def test_bundle_code_snapshot_covers_supervised_provider_receipt_sources():
    from experiments.vision_critic.independent_probes import CODE_FILES

    supervised_sources = {
        "experiments/visual_planner/provider.py",
        "experiments/visual_planner/worker.py",
        "experiments/visual_planner/download_model.py",
        "experiments/visual_planner/native.py",
        "engine/visual_analysis.py",
    }
    assert supervised_sources <= set(CODE_FILES)
