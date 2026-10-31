"""Safety properties of the Brev deadline controller; no real CLI/network calls."""

import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta

import pytest

from tools import brev_guard as guard

UTC = guard.UTC
NOW = datetime(2026, 10, 29, 16, tzinfo=UTC)


@pytest.fixture
def config():
    return guard.Config(
        "instance-123", "chameleon", "org-123", "our-org", NOW, NOW + timedelta(hours=2)
    )


def orgs():
    # A different active org must not alter the configured command scope.
    return json.dumps(
        [
            {"id": "org-other", "name": "other-org", "is_active": True},
            {"id": "org-123", "name": "our-org", "is_active": False},
        ]
    )


def workspace(status="RUNNING", instance_id="instance-123", name="chameleon"):
    return json.dumps({"workspaces": [{"id": instance_id, "name": name, "status": status}]})


class FakeCLI:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, command, timeout):
        self.calls.append((command, timeout))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def make_guard(config, responses, **kwargs):
    fake = FakeCLI(responses)
    instance = guard.Guard(config, runner=fake, sleep=lambda _: None, **kwargs)
    return instance, fake


def stop_calls(fake):
    return [command for command, _ in fake.calls if "stop" in command]


def test_before_deadline_is_read_only_and_scopes_to_fixed_org(config):
    instance, fake = make_guard(config, [orgs(), workspace()])
    report = instance.check(NOW + timedelta(hours=1))
    assert report["action"] == "check"
    assert report["estimated_elapsed_usd"] == 2.0
    assert guard.report_exit_code(report) == 0
    assert fake.calls[1][0] == ["brev", "--org", "our-org", "ls", "--json"]
    assert not stop_calls(fake)


def test_wrong_id_does_not_stop_an_instance_with_the_same_name(config):
    instance, fake = make_guard(config, [orgs(), workspace(instance_id="wrong-id")])
    report = instance.check(config.deadline)
    assert report["identity_mismatch"]
    assert not report["confirmed_stopped"]
    assert guard.report_exit_code(report) == 2
    assert not stop_calls(fake)


@pytest.mark.parametrize(
    "response",
    [
        json.dumps([{"id": "wrong-org", "name": "our-org", "is_active": True}]),
        json.dumps([{"id": "org-123", "name": "wrong-name", "is_active": True}]),
    ],
)
def test_org_identity_mismatch_prevents_stop(config, response):
    instance, fake = make_guard(config, [response])
    report = instance.stop(NOW)
    assert report["identity_mismatch"]
    assert not stop_calls(fake)


def test_deadline_submits_stop_and_requires_final_stopped_state(config):
    instance, fake = make_guard(
        config, [orgs(), workspace(), "Stopping", orgs(), workspace("stopped")]
    )
    report = instance.check(config.deadline)
    assert report["confirmed_stopped"]
    assert report["stop_attempts"] == 1
    assert stop_calls(fake) == [["brev", "--org", "our-org", "stop", "instance-123"]]
    assert guard.report_exit_code(report) == 0


def test_stop_success_without_confirmation_is_failure(config):
    instance, fake = make_guard(
        config, [orgs(), workspace(), "Stopping", orgs(), workspace("STOPPING")], attempts=1
    )
    report = instance.stop(NOW)
    assert stop_calls(fake)
    assert not report["confirmed_stopped"]
    assert guard.report_exit_code(report) == 2
    assert "billing may continue" in report["errors"][-1]


def test_transient_list_failure_at_deadline_still_stops_fixed_id(config):
    instance, fake = make_guard(
        config, [guard.GuardError("network"), "", orgs(), workspace("STOPPED")]
    )
    report = instance.check(config.deadline)
    assert report["confirmed_stopped"]
    assert stop_calls(fake) == [["brev", "--org", "our-org", "stop", "instance-123"]]


def test_malformed_json_does_not_disable_expired_deadline(config):
    instance, fake = make_guard(config, ["not json", "", orgs(), workspace("STOPPED")])
    assert instance.check(config.deadline)["confirmed_stopped"]
    assert len(stop_calls(fake)) == 1


def test_transient_failure_before_deadline_does_not_stop_early(config):
    instance, fake = make_guard(config, [guard.GuardError("network")])
    report = instance.check(NOW)
    assert guard.report_exit_code(report) == 2
    assert not stop_calls(fake)


def test_stop_retries_after_timeout_and_then_confirms(config):
    instance, fake = make_guard(
        config,
        [
            orgs(),
            workspace(),
            guard.GuardError("timeout"),
            orgs(),
            workspace(),
            orgs(),
            workspace(),
            "",
            orgs(),
            workspace("STOPPED"),
        ],
        attempts=2,
    )
    report = instance.stop(NOW)
    assert report["confirmed_stopped"]
    assert report["stop_attempts"] == 2
    assert len(stop_calls(fake)) == 2


def test_already_stopped_does_not_resubmit(config):
    instance, fake = make_guard(config, [orgs(), workspace("STOPPED")])
    assert instance.check(config.deadline)["confirmed_stopped"]
    assert not stop_calls(fake)


def test_init_refuses_overwrite_and_restart_keeps_absolute_deadline(config, tmp_path):
    path = tmp_path / "guard.json"
    instance, fake = make_guard(config, [orgs(), workspace()])
    guard.save_new_config(path, config, instance)
    original = path.read_bytes()
    reloaded = guard.load_config(path)
    assert reloaded.deadline == config.deadline
    assert reloaded.started_at == NOW
    with pytest.raises(FrozenInstanceError):
        reloaded.deadline = NOW + timedelta(days=1)
    with pytest.raises(FileExistsError, match="will not be reset"):
        guard.save_new_config(path, replace(config, deadline=NOW + timedelta(days=1)), instance)
    assert path.read_bytes() == original
    assert len(fake.calls) == 2


def test_init_does_not_write_if_identity_unverified(config, tmp_path):
    path = tmp_path / "guard.json"
    instance, _ = make_guard(config, [orgs(), workspace(instance_id="wrong")])
    with pytest.raises(guard.IdentityError):
        guard.save_new_config(path, config, instance)
    assert not path.exists()


def test_subprocess_is_bounded_without_shell_or_interactive_login(monkeypatch):
    observed = {}

    def timeout(command, **kwargs):
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(guard.subprocess, "run", timeout)
    with pytest.raises(guard.GuardError, match="timed out after 7"):
        guard.run_command(["brev", "ls", "--json"], 7)
    assert observed["timeout"] == 7
    assert observed["stdin"] == subprocess.DEVNULL
    assert not observed.get("shell", False)


def test_subprocess_error_does_not_print_potential_secrets(monkeypatch):
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "secret stdout", "secret stderr"
        ),
    )
    with pytest.raises(guard.GuardError) as error:
        guard.run_command(["brev", "ls"], 5)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("value", ["2026-10-29T16:00:00", "not-a-date"])
def test_rejects_naive_or_invalid_time(value):
    with pytest.raises(ValueError):
        guard.utc_time(value)


def test_rejects_invalid_budget_and_deadline(config):
    with pytest.raises(ValueError):
        replace(config, deadline=NOW)
    with pytest.raises(ValueError):
        replace(config, padded_hourly_usd=float("nan"))
    with pytest.raises(ValueError):
        replace(config, reserve_usd=-1)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "workspaces": [
                {"id": "instance-123", "name": "chameleon", "status": "RUNNING"},
                {"id": "instance-123", "name": "chameleon", "status": "STOPPED"},
            ]
        },
        {"workspaces": [{"id": "instance-123", "name": "renamed", "status": "RUNNING"}]},
    ],
)
def test_conflicting_identity_records_do_not_trigger_a_stop(config, payload):
    instance, fake = make_guard(config, [orgs(), json.dumps(payload)])
    report = instance.stop(NOW)
    assert report["identity_mismatch"]
    assert not stop_calls(fake)


def test_watch_waits_for_saved_deadline_and_retries_unconfirmed_stop(config, monkeypatch, capsys):
    responses = [
        orgs(),
        workspace(),  # Before deadline: observe without mutation.
        orgs(),
        workspace(),
        "",
        orgs(),
        workspace("STOPPING"),  # First cycle unconfirmed.
        orgs(),
        workspace("STOPPED"),  # Next cycle confirms without another request.
    ]
    fake = FakeCLI(responses)
    sleeps = []
    real_guard = guard.Guard

    def make_controller(config, **kwargs):
        return real_guard(config, runner=fake, sleep=sleeps.append, **kwargs)

    times = iter(
        [
            config.deadline - timedelta(seconds=10),
            config.deadline - timedelta(seconds=2),
            config.deadline,
            config.deadline + timedelta(seconds=5),
            config.deadline + timedelta(seconds=10),
        ]
    )

    class Clock:
        @staticmethod
        def now(timezone):
            assert timezone is UTC
            return next(times)

    monkeypatch.setattr(guard, "load_config", lambda _: config)
    monkeypatch.setattr(guard, "Guard", make_controller)
    monkeypatch.setattr(guard, "datetime", Clock)
    assert guard.main(["--attempts", "1", "watch"]) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["action"] for item in reports] == ["check", "stop", "stop"]
    assert [item["confirmed_stopped"] for item in reports] == [False, False, True]
    assert {item["deadline"] for item in reports} == {guard.iso_time(config.deadline)}
    assert sleeps == [2, 5, 5]
    assert stop_calls(fake) == [["brev", "--org", "our-org", "stop", "instance-123"]]


def test_nonstring_config_timestamp_is_rejected(config):
    data = config.as_dict()
    data["deadline"] = 1234
    with pytest.raises(ValueError, match="ISO strings"):
        guard.Config.from_dict(data)


def test_wait_for_start_monitors_stopped_starting_and_stopping_until_running_stops(
    config, monkeypatch, capsys
):
    statuses = ["STOPPED", "STARTING", "STOPPING", "STOPPED", "READY", "STOPPING", "STOPPED"]
    fake = FakeCLI([value for status in statuses for value in (orgs(), workspace(status))])
    sleeps = []
    real_guard = guard.Guard

    def make_controller(config, **kwargs):
        return real_guard(config, runner=fake, sleep=sleeps.append, **kwargs)

    class Clock:
        @staticmethod
        def now(timezone):
            return NOW

    monkeypatch.setattr(guard, "load_config", lambda _: config)
    monkeypatch.setattr(guard, "Guard", make_controller)
    monkeypatch.setattr(guard, "datetime", Clock)
    assert guard.main(["--poll-seconds", "7", "watch", "--wait-for-start"]) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [report["status"] for report in reports] == statuses
    assert [report["awaiting_start"] for report in reports] == [True] * 4 + [False] * 3
    assert sleeps == [7] * 6
    assert not stop_calls(fake)


def test_wait_for_start_still_finishes_at_expired_deadline_if_never_started(
    config, monkeypatch, capsys
):
    fake = FakeCLI([orgs(), workspace("STOPPED")])
    real_guard = guard.Guard

    def make_controller(config, **kwargs):
        return real_guard(
            config, runner=fake, sleep=lambda _: pytest.fail("Unexpected wait"), **kwargs
        )

    class Clock:
        @staticmethod
        def now(timezone):
            return config.deadline

    monkeypatch.setattr(guard, "load_config", lambda _: config)
    monkeypatch.setattr(guard, "Guard", make_controller)
    monkeypatch.setattr(guard, "datetime", Clock)
    assert guard.main(["watch", "--wait-for-start"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["deadline_reached"]
    assert report["confirmed_stopped"]
    assert not report["awaiting_start"]
    assert not stop_calls(fake)


def test_default_watch_preserves_immediate_stopped_confirmation(config, monkeypatch, capsys):
    fake = FakeCLI([orgs(), workspace("STOPPED")])
    real_guard = guard.Guard

    def make_controller(config, **kwargs):
        return real_guard(
            config, runner=fake, sleep=lambda _: pytest.fail("Unexpected wait"), **kwargs
        )

    class Clock:
        @staticmethod
        def now(timezone):
            return NOW

    monkeypatch.setattr(guard, "load_config", lambda _: config)
    monkeypatch.setattr(guard, "Guard", make_controller)
    monkeypatch.setattr(guard, "datetime", Clock)
    assert guard.main(["watch"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["confirmed_stopped"]
    assert not report["awaiting_start"]


def test_watch_does_not_sleep_after_status_check_consumes_remaining_allowance(
    config, monkeypatch, capsys
):
    fake = FakeCLI([orgs(), workspace("RUNNING"), orgs(), workspace("STOPPED")])
    sleeps = []
    real_guard = guard.Guard

    def make_controller(config, **kwargs):
        return real_guard(config, runner=fake, sleep=sleeps.append, **kwargs)

    times = iter([config.deadline - timedelta(seconds=1), config.deadline, config.deadline])

    class Clock:
        @staticmethod
        def now(timezone):
            return next(times)

    monkeypatch.setattr(guard, "load_config", lambda _: config)
    monkeypatch.setattr(guard, "Guard", make_controller)
    monkeypatch.setattr(guard, "datetime", Clock)
    assert guard.main(["watch"]) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["action"] for item in reports] == ["check", "stop"]
    assert not sleeps
