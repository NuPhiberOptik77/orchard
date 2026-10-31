import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from orchard import cli, dispatcher
from orchard.cli import _redact_entries, _stats_summary
from orchard.config import Worker
from orchard.job import JobSpec

FULL_CAPS = {"machine": "x86_64", "ram_gb": 8, "disk_free_gb": 34, "idle_seconds": 99999}


def test_redact_entries_masks_values():
    entries = [{"id": "a", "spec": {"env": ["HF_TOKEN"], "env_values": {"HF_TOKEN": "s3cret"}}}]
    redacted = _redact_entries(entries)
    assert redacted[0]["spec"]["env_values"] == {"HF_TOKEN": "***"}
    assert redacted[0]["spec"]["env"] == ["HF_TOKEN"]
    assert entries[0]["spec"]["env_values"] == {"HF_TOKEN": "s3cret"}
    assert redacted[0] is not entries[0]


def test_redact_entries_noop_without_env():
    entries = [{"id": "b", "spec": {"cmd": ["true"]}}]
    assert _redact_entries(entries) == entries


def test_offload_target_picks_free_worker(monkeypatch):
    w = Worker(name="pro", host="10.0.0.9")
    monkeypatch.setattr(cli, "load_workers", lambda: [w])
    monkeypatch.setattr(cli.engine, "remote_caps", lambda worker: dict(FULL_CAPS))
    monkeypatch.setattr(cli.dispatcher, "idle_hold", lambda caps, cfg: None)
    target, why = cli._offload_target(JobSpec(name="d", cmd=["pytest"]), None, {})
    assert target is w
    assert why == ""


def test_offload_target_skips_unreachable(monkeypatch):
    w = Worker(name="pro", host="10.0.0.9")
    monkeypatch.setattr(cli, "load_workers", lambda: [w])
    monkeypatch.setattr(cli.engine, "remote_caps", lambda worker: None)
    target, why = cli._offload_target(JobSpec(name="d", cmd=["x"]), None, {})
    assert target is None
    assert "unreachable" in why


def test_offload_target_honors_capability_and_hold(monkeypatch):
    w = Worker(name="pro", host="10.0.0.9")
    monkeypatch.setattr(cli, "load_workers", lambda: [w])
    monkeypatch.setattr(cli.engine, "remote_caps", lambda worker: dict(FULL_CAPS, ram_gb=0))
    target, why = cli._offload_target(JobSpec(name="d", cmd=["x"]), None, {})
    assert target is None
    assert "RAM" in why

    monkeypatch.setattr(cli.engine, "remote_caps", lambda worker: dict(FULL_CAPS))
    monkeypatch.setattr(cli.dispatcher, "idle_hold", lambda caps, cfg: "owner active on worker")
    target, why = cli._offload_target(JobSpec(name="d", cmd=["x"]), None, {})
    assert target is None
    assert "owner active" in why
    records = [
        {"ts": 10, "event": "submitted", "job": "a", "worker": "pro"},
        {"ts": 40, "event": "held", "job": "a", "worker": "pro", "reason": "worker busy"},
        {"ts": 52, "event": "held", "job": "a", "worker": "pro", "reason": "worker busy"},
        {"ts": 70, "event": "launched", "job": "a", "worker": "pro"},
        {"ts": 200, "event": "finished", "job": "a", "worker": "pro", "state": "succeeded"},
        {"ts": 20, "event": "submitted", "job": "b", "worker": "pro"},
        {"ts": 30, "event": "held", "job": "b", "worker": "pro", "reason": "worker in use (idle 12s < 183s)"},
        {"ts": 90, "event": "launched", "job": "b", "worker": "local"},
        {"ts": 400, "event": "failed", "job": "b", "worker": "local", "detail": "boom"},
        {"ts": 30, "event": "adopted", "job": "c", "worker": "pro"},
    ]
    summary = _stats_summary(records)
    assert summary["runs"] == {"pro": 2, "local": 1}
    assert summary["finished"] == {"done": 1, "failed": 1}
    assert summary["waits"] == [60.0, 70.0]
    assert summary["holds"]["worker busy"]["count"] == 2
    assert "worker in use (idle Ns < Ns)" in summary["holds"]



def _isolate_control_plane(tmp_path, monkeypatch, installed=False, agent=(False, None)):
    monkeypatch.setattr(cli.dispatcher, "DISPATCH_PID", tmp_path / "dispatch.pid")
    monkeypatch.setattr(cli.dispatcher, "TICK_LOCK", tmp_path / "tick.lock")
    monkeypatch.setattr(cli.dispatcher, "agent_installed", lambda: installed)
    monkeypatch.setattr(cli.dispatcher, "agent_state", lambda: agent)


def test_doctor_flags_dispatcher_killed_without_cleanup(tmp_path, monkeypatch):
    _isolate_control_plane(tmp_path, monkeypatch)
    (tmp_path / "dispatch.pid").write_text("4000000")
    monkeypatch.setattr(cli, "load_workers", lambda: [])
    monkeypatch.setattr(cli, "STATE_DIR", tmp_path)
    monkeypatch.setattr(cli.store, "BLOB_DIR", tmp_path / "blobs")
    monkeypatch.setattr(cli.dispatcher, "load_entries", lambda: [{"id": "j1", "state": "queued"}])
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "dispatch died without cleaning up (pid 4000000" in result.output
    assert "not supervised" in result.output
    assert "1 queued job(s) will not start" in result.output


def test_doctor_says_when_other_cpu_cant_be_read(tmp_path, monkeypatch):
    # A worker with no ps can't honour max_other_cpu_pct, and the dispatcher
    # ignores it there, so doctor says so rather than letting it look enforced.
    _isolate_control_plane(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "STATE_DIR", tmp_path)
    monkeypatch.setattr(cli.store, "BLOB_DIR", tmp_path / "blobs")
    monkeypatch.setattr(cli.dispatcher, "load_entries", lambda: [])
    monkeypatch.setattr(cli, "load_workers", lambda: [Worker(name="pro", host="10.0.0.9")])
    monkeypatch.setattr(cli.transport, "file_sha256_remote", lambda w, path: None)
    unknown = "other CPU unknown (no ps), max_other_cpu_pct ignored"
    for caps, shown in (
        (dict(FULL_CAPS, max_other_cpu_pct=25, other_cpu_pct=None), True),
        (dict(FULL_CAPS, max_other_cpu_pct=25, other_cpu_pct=3.0), False),
        (dict(FULL_CAPS, max_other_cpu_pct=None, other_cpu_pct=None), False),
    ):
        monkeypatch.setattr(cli.engine, "remote_caps", lambda worker, caps=caps: caps)
        result = CliRunner().invoke(cli.app, ["doctor"])
        assert (unknown in result.output) == shown, result.output


def test_watch_hands_off_to_launchd(tmp_path, monkeypatch):
    _isolate_control_plane(tmp_path, monkeypatch, installed=True, agent=(True, 4242))
    monkeypatch.delenv(cli.dispatcher.SUPERVISED_ENV, raising=False)
    pids = iter([None, None, 4242])
    monkeypatch.setattr(cli.dispatcher, "running_pid", lambda: next(pids))
    monkeypatch.setattr(cli.dispatcher, "start_agent", lambda: None)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    result = CliRunner().invoke(cli.app, ["dispatch", "--watch"])
    assert result.exit_code == 0
    assert "under launchd" in result.output and "pid 4242" in result.output
    assert not (tmp_path / "dispatch.pid").exists()


def test_handoff_falls_back_to_this_process_when_launchd_fails(monkeypatch):
    monkeypatch.setattr(cli.dispatcher, "release_tick_lock", lambda: None)
    monkeypatch.setattr(cli.dispatcher, "running_pid", lambda: None)
    monkeypatch.setattr(cli.dispatcher, "start_agent", lambda: "Bootstrap failed: 5")
    assert cli._hand_off_to_launchd() is False


def test_watch_stops_cleanly_on_sigterm(tmp_path, monkeypatch):
    _isolate_control_plane(tmp_path, monkeypatch)
    monkeypatch.delenv(cli.dispatcher.SUPERVISED_ENV, raising=False)

    def tick():
        assert (tmp_path / "dispatch.pid").read_text() == str(os.getpid())
        os.kill(os.getpid(), signal.SIGTERM)  # arrives mid-tick: the tick must finish first
        return ["j1: started on local"]

    monkeypatch.setattr(cli.dispatcher, "tick", tick)
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        result = CliRunner().invoke(cli.app, ["dispatch", "--watch", "--foreground", "--interval", "0.01"])
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)
    assert result.exit_code == 0, result.output
    assert "j1: started on local" in result.output
    assert "dispatch stopped (SIGTERM)" in result.output
    assert not (tmp_path / "dispatch.pid").exists()
    assert not (tmp_path / "tick.lock").exists()


def test_main_adds_homebrew_to_a_gui_path(monkeypatch):
    # Orchard.app launches dispatch with this PATH; uv is in /opt/homebrew/bin (2026-10-29).
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    seen = {}
    monkeypatch.setattr(cli, "app", lambda: seen.setdefault("path", os.environ["PATH"]))
    cli.main()
    dirs = seen["path"].split(":")
    assert dirs[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    assert "/opt/homebrew/bin" in dirs


def _queued_preempted(monkeypatch, tmp_path, kill):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(cli, "load_workers", lambda: [Worker(name="pro", host="10.0.0.9")])
    kills = []
    monkeypatch.setattr(cli.engine, "kill_remote", lambda w, rjob: kills.append(rjob) or kill)
    entry = dispatcher.enqueue(JobSpec(name="demo", cmd=["true"]), "demo-q", Path("/tmp/p"), Path("/tmp/s"),
                               worker="pro")
    entry.update(state="queued", rhome="/Users/x", attempts=1, preempted_attempt=1)
    dispatcher.save_entry(entry)
    return kills


@pytest.mark.parametrize("kill", [True, False, None])
def test_cancel_kills_a_preempted_run_that_may_still_be_going(monkeypatch, tmp_path, kill):
    # Requeued after a preempt whose kill reply was lost: the run can still be
    # going on the worker, so cancel stops it or leaves it to the dispatcher.
    # False is the agent finding no run: it has ended, so nothing is left.
    kills = _queued_preempted(monkeypatch, tmp_path, kill)
    result = CliRunner().invoke(cli.app, ["cancel", "demo-q"])
    assert result.exit_code == 0, result.output
    rjob = "/Users/x/.orchard/jobs/demo-q"
    assert kills == [rjob]
    entry = dispatcher.load_entries()[0]
    assert entry["state"] == "cancelled"
    if kill is not None:
        assert not entry.get("orphan")
        assert "dispatcher will retry" not in result.output
    else:
        assert entry["orphan"] == {"worker": "pro", "rjob": rjob, "rhome": "/Users/x", "attempt": 1,
                                   "preempted": True}
        assert "dispatcher will retry" in result.output


def test_cancel_stands_when_the_tick_records_the_end_of_the_run_it_stopped(monkeypatch, tmp_path):
    # The kill ends a local run, and a tick can record it failed before cancel
    # saves: it was still cancelled, not "already failed".
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    entry = dispatcher.enqueue(JobSpec(name="demo", cmd=["true"]), "demo-l", Path("/tmp/p"), tmp_path / "s",
                               worker="local")
    entry.update(state="running", attempts=1)
    dispatcher.save_entry(entry)
    run = subprocess.Popen(["sleep", "30"], start_new_session=True)
    monkeypatch.setattr(cli.engine, "local_status", lambda sdir: {"pgid": run.pid})

    def terminate(pgid, grace):
        disk = dispatcher.load_entries()[0]
        disk.update(state="failed", note="exit 143")
        dispatcher.save_entry(disk)

    monkeypatch.setattr(cli.agent, "_terminate_group", terminate)
    try:
        result = CliRunner().invoke(cli.app, ["cancel", "demo-l"])
    finally:
        run.kill()
        run.wait()
    assert result.exit_code == 0, result.output
    assert "cancelled: demo-l (was running)" in result.output
    assert dispatcher.load_entries()[0]["state"] == "cancelled"


def test_cancel_stops_a_run_started_while_it_acts(monkeypatch, tmp_path):
    # A dispatch tick can start the job between cancel's read and its save.
    kills = _queued_preempted(monkeypatch, tmp_path, True)
    entry = dispatcher.load_entries()[0]
    entry.update(preempted_attempt=None)
    dispatcher.save_entry(entry)
    states, real = [], cli._stop_for_cancel

    def stop(entry, state):
        states.append(state)
        if len(states) == 1:
            fresh = dict(entry, state="running", worker="pro", rhome="/Users/x", attempts=2)
            dispatcher.save_entry(fresh)
        return real(entry, state)

    monkeypatch.setattr(cli, "_stop_for_cancel", stop)
    result = CliRunner().invoke(cli.app, ["cancel", "demo-q"])
    assert result.exit_code == 0, result.output
    assert states == ["queued", "running"]
    assert kills == ["/Users/x/.orchard/jobs/demo-q"]
    entry = dispatcher.load_entries()[0]
    assert entry["state"] == "cancelled"
    assert entry["note"] == "cancelled by user (was running)"


def _cancel_remote_run_that_ends(monkeypatch, tmp_path, kill):
    """A remote running job; the tick records it done during cancel's kill, which answers `kill`."""
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(cli, "load_workers", lambda: [Worker(name="pro", host="10.0.0.9")])
    entry = dispatcher.enqueue(JobSpec(name="demo", cmd=["true"]), "demo-r", Path("/tmp/p"), tmp_path / "s",
                               worker="pro")
    entry.update(state="running", rhome="/Users/x", attempts=1)
    dispatcher.save_entry(entry)

    def kill_remote(w, rjob):
        disk = dispatcher.load_entries()[0]
        disk.update(state="done", note="succeeded (exit 0)")
        dispatcher.save_entry(disk)
        return kill

    monkeypatch.setattr(cli.engine, "kill_remote", kill_remote)


@pytest.mark.parametrize("kill, want", [(True, "cancelled"), (False, "done")])
def test_cancel_counts_a_remote_run_as_stopped_only_if_it_signalled_one(monkeypatch, tmp_path, kill, want):
    # The agent finding no run (False) means it ended by itself: the tick's
    # record of it stands. A run the kill ended was cancelled.
    _cancel_remote_run_that_ends(monkeypatch, tmp_path, kill)
    result = CliRunner().invoke(cli.app, ["cancel", "demo-r"])
    assert result.exit_code == 0, result.output
    assert dispatcher.load_entries()[0]["state"] == want, result.output


def _dead_pgid():
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()
    return proc.pid


def _local_running(monkeypatch, tmp_path, status, record_done):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    entry = dispatcher.enqueue(JobSpec(name="demo", cmd=["true"]), "demo-l", Path("/tmp/p"), tmp_path / "s",
                               worker="local")
    entry.update(state="running", attempts=1)
    dispatcher.save_entry(entry)

    def local_status(sdir):
        if record_done:
            disk = dispatcher.load_entries()[0]
            disk.update(state="done", note="succeeded (exit 0)")
            dispatcher.save_entry(disk)
        return status

    monkeypatch.setattr(cli.engine, "local_status", local_status)
    stopped = []
    monkeypatch.setattr(cli.agent, "_terminate_group", lambda pgid, grace: stopped.append(pgid))
    return stopped


def test_cancel_leaves_a_local_run_that_already_ended(monkeypatch, tmp_path):
    # Its status says it ended (its pgid may be another group's by now), and
    # the tick has yet to record it: nothing is signalled or cancelled.
    pgid = _dead_pgid()
    stopped = _local_running(monkeypatch, tmp_path, {"state": "succeeded", "exit_code": 0, "pgid": pgid}, False)
    result = CliRunner().invoke(cli.app, ["cancel", "demo-l"])
    assert result.exit_code == 0, result.output
    assert "demo-l is already succeeded" in result.output
    assert stopped == [] and dispatcher.load_entries()[0]["state"] == "running"


def test_cancel_counts_a_local_group_as_stopped_only_if_it_was_there(monkeypatch, tmp_path):
    # The group was gone before cancel signalled it, so the tick's "done" stands.
    pgid = _dead_pgid()
    _local_running(monkeypatch, tmp_path, {"state": "running", "pgid": pgid}, True)
    result = CliRunner().invoke(cli.app, ["cancel", "demo-l"])
    assert result.exit_code == 0, result.output
    assert dispatcher.load_entries()[0]["state"] == "done", result.output


def test_doctor_flags_a_job_held_long_behind_a_preempted_run(tmp_path, monkeypatch):
    # Waiting out a preempted run restarts held_since at each answer; the
    # warning times the wait from the preempt instead.
    from test_dispatcher import _age, _long_paused

    _long_paused(tmp_path, monkeypatch, kill=True, alive=[True] * 10)
    workers = [Worker(name="pro", host="10.0.0.9")]
    dispatcher.tick(workers=workers)
    _age("demo-p", preempt_sent_at=4200)
    dispatcher.tick(workers=workers)
    entry = dispatcher.load_entries()[0]
    assert entry["state"] == "queued" and time.time() - entry["held_since"] < 60
    _isolate_control_plane(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "load_workers", lambda: [])
    monkeypatch.setattr(cli, "STATE_DIR", tmp_path)
    monkeypatch.setattr(cli.store, "BLOB_DIR", tmp_path / "blobs")
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert "warn  demo-p held " in result.output, result.output
