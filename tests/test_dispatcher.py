import json
import os
import subprocess
import sys
import time
from pathlib import Path

from orchard import dispatcher, events
from orchard.config import Worker
from orchard.job import JobSpec


def make_job():
    return JobSpec(name="demo", cmd=["python", "-c", "print(1)"], requirements="six==1.17.0")


def test_enqueue_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    entry = dispatcher.enqueue(make_job(), "demo-1", Path("/tmp/proj"), Path("/tmp/state/demo-1"), worker="pro")
    assert entry["state"] == "queued"
    entries = dispatcher.load_entries()
    assert len(entries) == 1
    assert entries[0]["spec"]["cmd"] == ["python", "-c", "print(1)"]
    spec = dispatcher.spec_of(entries[0])
    assert spec.requirements == "six==1.17.0"
    assert spec.name == "demo"


def test_save_entry_updates_state(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    entry = dispatcher.enqueue(make_job(), "demo-2", Path("/tmp/p"), Path("/tmp/s"), worker="local")
    entry["state"] = "running"
    dispatcher.save_entry(entry)
    assert dispatcher.load_entries()[0]["state"] == "running"


def test_load_entries_skips_corrupt(tmp_path, monkeypatch):
    queue = tmp_path / "queue"
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", queue)
    queue.mkdir(parents=True)
    (queue / "bad.json").write_text("{not json")
    dispatcher.enqueue(make_job(), "demo-3", Path("/tmp/p"), Path("/tmp/s"), worker="local")
    entries = dispatcher.load_entries()
    assert [e["id"] for e in entries] == ["demo-3"]


def test_tick_holds_when_worker_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: None)
    dispatcher.enqueue(make_job(), "demo-4", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("unreachable" in line for line in lines)
    entries = dispatcher.load_entries()
    assert entries[0]["state"] == "queued"
    assert entries[0]["attempts"] == 0


def test_tick_falls_back_to_local_after_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: None)
    monkeypatch.setattr(
        dispatcher, "load_config", lambda: {**dispatcher.DISPATCH_DEFAULTS, "fallback_after_s": 0}
    )
    monkeypatch.setattr(dispatcher, "_local_fallback", lambda job: (True, ""))
    dispatcher.enqueue(make_job(), "demo-fb", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("falling back to local" in line for line in lines)
    entry = dispatcher.load_entries()[0]
    assert entry["worker"] == "local"
    assert entry["state"] == "queued"


def test_tick_keeps_held_when_not_local_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: None)
    monkeypatch.setattr(
        dispatcher, "load_config", lambda: {**dispatcher.DISPATCH_DEFAULTS, "fallback_after_s": 0}
    )
    monkeypatch.setattr(dispatcher, "_local_fallback", lambda job: (False, "missing tags: x86_64"))
    dispatcher.enqueue(make_job(), "demo-fb2", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    entry = dispatcher.load_entries()[0]
    assert entry["worker"] == "pro"
    assert entry["state"] == "queued"
    assert "x86_64" in entry["note"]


def test_lost_contact_requeues_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_status", lambda worker, rdir: None)
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: None)
    monkeypatch.setattr(dispatcher, "_local_fallback", lambda job: (True, ""))
    monkeypatch.setattr(dispatcher.engine, "launch_local", lambda *a, **k: (None, None))
    entry = dispatcher.enqueue(make_job(), "demo-lost", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    entry.update(state="running", rhome="/Users/x", misses=dispatcher.DISPATCH_DEFAULTS["max_misses"])
    dispatcher.save_entry(entry)
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("requeued locally" in line for line in lines)
    entry = dispatcher.load_entries()[0]
    assert entry["worker"] == "local"
    assert entry["state"] == "running"
    assert entry["orphan"] == {
        "worker": "pro",
        "rjob": "/Users/x/.orchard/jobs/demo-lost",
        "rhome": "/Users/x",
        "attempt": 0,
    }


def test_orphan_cancelled_when_worker_reachable(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_status", lambda worker, rdir: None)
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: {"machine": "x86_64"})
    monkeypatch.setattr(dispatcher, "_local_fallback", lambda job: (True, ""))
    monkeypatch.setattr(dispatcher.engine, "launch_local", lambda *a, **k: (None, None))
    calls = []
    monkeypatch.setattr(dispatcher.engine, "kill_remote", lambda w, rjob: calls.append((w.name, rjob)) or True)
    entry = dispatcher.enqueue(make_job(), "demo-orphan", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    entry.update(state="running", rhome="/Users/x", misses=dispatcher.DISPATCH_DEFAULTS["max_misses"])
    dispatcher.save_entry(entry)
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("cancelled orphaned remote job" in line for line in lines)
    assert calls == [("pro", "/Users/x/.orchard/jobs/demo-orphan")]
    entry = dispatcher.load_entries()[0]
    assert entry["orphan"] is None
    assert "orphan cancelled" in entry["note"]


ORPHAN = {"worker": "pro", "rjob": "/Users/x/.orchard/jobs/demo-o", "rhome": "/Users/x", "attempt": 1}


def _orphaned(tmp_path, monkeypatch, remote, state, **fields):
    """A job requeued locally after lost contact, whose worker now answers with `remote`."""
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: {"machine": "x86_64"})
    monkeypatch.setattr(dispatcher.engine, "remote_status", lambda worker, rjob: remote)
    kills = []
    monkeypatch.setattr(dispatcher.engine, "kill_remote", lambda w, rjob: kills.append(rjob) or True)
    collected = []

    def collect(worker, job_id, rjob, into, cwd, artifacts=None):
        collected.append((into, artifacts))
        (into / "status.json").write_text(json.dumps(remote))
        return remote

    monkeypatch.setattr(dispatcher.engine, "collect_remote", collect)
    sdir, cwd = tmp_path / "state", tmp_path / "proj"
    sdir.mkdir(exist_ok=True)
    cwd.mkdir(exist_ok=True)
    entry = dispatcher.enqueue(make_job(), "demo-o", cwd, sdir, worker="local")
    entry.update(state=state, attempts=2, orphan=dict(ORPHAN), note="lost contact with worker", **fields)
    dispatcher.save_entry(entry)
    return sdir, cwd, kills, collected


def test_orphan_success_adopted_and_rerun_stopped(tmp_path, monkeypatch):
    remote = {"state": "succeeded", "exit_code": 0, "attempt": 1}
    sdir, cwd, kills, collected = _orphaned(tmp_path, monkeypatch, remote, "running")
    job_json = sdir / "job.json"
    job_json.write_text(json.dumps({
        "id": "demo-o", "attempt": 2, "cmd": ["sleep", "30"], "venv": None, "workdir": str(cwd),
        "job_dir": str(sdir), "timeout_s": 60, "preempt": False, "env": {},
    }))
    proc = subprocess.Popen(
        [sys.executable, dispatcher.agent.__file__, "exec", str(job_json)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    try:
        deadline = time.time() + 10
        while not (dispatcher.engine.local_status(sdir) or {}).get("pgid") and time.time() < deadline:
            time.sleep(0.1)
        entry = dispatcher.load_entries()[0]
        entry["local_pid"] = proc.pid
        dispatcher.save_entry(entry)
        lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    finally:
        proc.kill()
    assert any("adopted result from pro (local rerun stopped)" in line for line in lines), (lines, dispatcher.load_entries()[0])
    assert proc.poll() is not None
    assert kills == []
    assert collected == [(sdir, None)]
    entry = dispatcher.load_entries()[0]
    assert (entry["state"], entry["worker"], entry["rhome"], entry["orphan"]) == ("done", "pro", "/Users/x", None)
    assert json.loads((sdir / "status.json").read_text())["attempt"] == 1
    assert json.loads((sdir / "rerun" / "status.json").read_text())["attempt"] == 2
    recs = events.load()
    assert [r for r in recs if r["event"] == "finished"][0]["attempt"] == 1
    orphan = [r for r in recs if r["event"] == "orphan"][0]
    assert (orphan["outcome"], orphan["duplicate"]) == ("adopted", True)


def test_orphan_success_adopted_before_rerun_starts(tmp_path, monkeypatch):
    remote = {"state": "succeeded", "exit_code": 0, "attempt": 1}
    sdir, _cwd, _kills, collected = _orphaned(tmp_path, monkeypatch, remote, "queued")
    monkeypatch.setattr(dispatcher, "start_queued", lambda *a, **k: [])
    dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    entry = dispatcher.load_entries()[0]
    assert (entry["state"], entry["worker"]) == ("done", "pro")
    assert "local rerun not started" in entry["note"]
    assert collected == [(sdir, None)]
    orphan = [r for r in events.load() if r["event"] == "orphan"][0]
    assert orphan["duplicate"] is False


def test_orphan_after_finished_rerun_kept_as_duplicate(tmp_path, monkeypatch):
    remote = {"state": "succeeded", "exit_code": 0, "attempt": 1}
    sdir, cwd, kills, collected = _orphaned(tmp_path, monkeypatch, remote, "done")
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("result kept beside the rerun" in line for line in lines)
    assert kills == []
    assert collected == [(sdir / "orphan", cwd / ".orchard-out" / "demo-o" / "orphan")]
    entry = dispatcher.load_entries()[0]
    assert (entry["state"], entry["worker"], entry["orphan"]) == ("done", "local", None)
    orphan = [r for r in events.load() if r["event"] == "orphan"][0]
    assert (orphan["outcome"], orphan["rerun"], orphan["duplicate"]) == ("kept", "done", True)


def test_orphan_failure_leaves_rerun_running(tmp_path, monkeypatch):
    remote = {"state": "failed", "exit_code": 1, "attempt": 1}
    sdir, cwd, kills, collected = _orphaned(tmp_path, monkeypatch, remote, "running", local_pid=os.getpid())
    (sdir / "status.json").write_text(json.dumps({"state": "running", "attempt": 2}))
    monkeypatch.setattr(dispatcher, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(dispatcher, "_stop_local_rerun", lambda *a: (_ for _ in ()).throw(AssertionError))
    dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert collected == [(sdir / "orphan", cwd / ".orchard-out" / "demo-o" / "orphan")]
    entry = dispatcher.load_entries()[0]
    assert (entry["state"], entry["worker"], entry["orphan"]) == ("running", "local", None)
    assert "orphan failed on pro" in entry["note"]


def test_orphan_from_other_attempt_ignored(tmp_path, monkeypatch):
    remote = {"state": "succeeded", "exit_code": 0, "attempt": 3}
    _sdir, _cwd, kills, collected = _orphaned(tmp_path, monkeypatch, remote, "done")
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("ignored stale orphan result" in line for line in lines)
    assert (kills, collected) == ([], [])
    entry = dispatcher.load_entries()[0]
    assert entry["orphan"] is None
    assert "attempt 3, expected 1" in entry["note"]


def test_local_pid_liveness_marks_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher, "_pid_alive", lambda pid: False)
    sdir = tmp_path / "state"
    sdir.mkdir()
    (sdir / "status.json").write_text(json.dumps({"state": "running"}))
    entry = dispatcher.enqueue(make_job(), "demo-dead", Path("/tmp/p"), sdir, worker="local")
    entry.update(state="running", local_pid=999999)
    dispatcher.save_entry(entry)
    lines = dispatcher.tick(workers=[])
    assert any("process gone" in line for line in lines)
    assert dispatcher.load_entries()[0]["state"] == "failed"


def test_local_pid_alive_keeps_running(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher, "_pid_alive", lambda pid: True)
    sdir = tmp_path / "state"
    sdir.mkdir()
    (sdir / "status.json").write_text(json.dumps({"state": "running", "elapsed_s": 5}))
    entry = dispatcher.enqueue(make_job(), "demo-alive", Path("/tmp/p"), sdir, worker="local")
    entry.update(state="running", local_pid=os.getpid())
    dispatcher.save_entry(entry)
    dispatcher.tick(workers=[])
    assert dispatcher.load_entries()[0]["state"] == "running"


def test_pid_alive_detects_dead_process():
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    proc.wait()
    assert dispatcher._pid_alive(proc.pid) is False
    assert dispatcher._pid_alive(os.getpid()) is True


def test_lost_contact_not_runnable_locally_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_status", lambda worker, rdir: None)
    monkeypatch.setattr(dispatcher, "_local_fallback", lambda job: (False, "missing tags: x86_64"))
    entry = dispatcher.enqueue(make_job(), "demo-lost2", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    entry.update(state="running", rhome="/Users/x/.orchard", misses=dispatcher.DISPATCH_DEFAULTS["max_misses"])
    dispatcher.save_entry(entry)
    dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    entry = dispatcher.load_entries()[0]
    assert entry["state"] == "failed"
    assert "x86_64" in entry["note"]


def test_tick_holds_when_worker_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: None)
    dispatcher.enqueue(make_job(), "demo-5", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    entry = dispatcher.load_entries()[0]
    entry["state"] = "running"
    dispatcher.save_entry(entry)
    dispatcher.enqueue(make_job(), "demo-6", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("worker busy" in line for line in lines)
    states = {e["id"]: e["state"] for e in dispatcher.load_entries()}
    assert states["demo-6"] == "queued"


def test_running_pid_reads_own_process(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "DISPATCH_PID", tmp_path / "dispatch.pid")
    monkeypatch.setattr(dispatcher, "_proc_command", lambda pid: "python -m orchard dispatch --watch")
    dispatcher.DISPATCH_PID.write_text(str(__import__("os").getpid()))
    assert dispatcher.running_pid() == __import__("os").getpid()


def test_running_pid_rejects_reused_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "DISPATCH_PID", tmp_path / "dispatch.pid")
    dispatcher.DISPATCH_PID.write_text(str(os.getpid()))
    # the real command line of this pytest process is not a dispatcher
    assert dispatcher.running_pid() is None
    assert dispatcher.recorded_pid() == os.getpid()


def test_running_pid_trusts_liveness_when_ps_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "DISPATCH_PID", tmp_path / "dispatch.pid")
    monkeypatch.setattr(dispatcher, "_proc_command", lambda pid: None)
    dispatcher.DISPATCH_PID.write_text(str(os.getpid()))
    assert dispatcher.running_pid() == os.getpid()


def test_heartbeat_dates_last_tick(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "DISPATCH_PID", tmp_path / "dispatch.pid")
    assert dispatcher.last_tick_age() is None
    dispatcher.DISPATCH_PID.write_text("1")
    os.utime(dispatcher.DISPATCH_PID, (time.time() - 900, time.time() - 900))
    assert dispatcher.last_tick_age() > 800
    dispatcher.heartbeat()
    assert dispatcher.last_tick_age() < 60


def test_agent_plist_supervises_and_keeps_path(tmp_path):
    plist = dispatcher.agent_plist(tmp_path / "bin" / "orchard", "/custom/bin:/usr/bin")
    assert plist["ProgramArguments"] == [str(tmp_path / "bin" / "orchard"), "dispatch", "--watch"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    env_vars = plist["EnvironmentVariables"]
    assert env_vars[dispatcher.SUPERVISED_ENV] == "1"
    dirs = env_vars["PATH"].split(":")
    assert dirs[:2] == ["/custom/bin", "/usr/bin"]
    assert "/opt/homebrew/bin" in dirs and dirs.count("/usr/bin") == 1


def test_merged_path_appends_missing_base_dirs():
    dirs = dispatcher.merged_path("/usr/bin:/bin:/usr/sbin:/sbin").split(":")
    assert dirs[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    assert "/opt/homebrew/bin" in dirs and dirs.count("/usr/bin") == 1
    assert dispatcher.merged_path("") == ":".join(dispatcher.BASE_PATH)


def test_rotate_log_truncates_in_place(tmp_path, monkeypatch):
    log = tmp_path / "orchard-dispatch.log"
    monkeypatch.setattr(dispatcher, "DISPATCH_LOG", log)
    monkeypatch.setattr(dispatcher, "LOG_MAX_BYTES", 10)
    log.write_text("x" * 20)
    dispatcher.rotate_log()
    assert log.read_text() == ""
    assert (tmp_path / "orchard-dispatch.log.1").read_text() == "x" * 20


def test_running_pid_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "DISPATCH_PID", tmp_path / "dispatch.pid")
    assert dispatcher.running_pid() is None


def test_running_pid_none_when_garbage_or_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "DISPATCH_PID", tmp_path / "dispatch.pid")
    dispatcher.DISPATCH_PID.write_text("not-a-pid")
    assert dispatcher.running_pid() is None
    dispatcher.DISPATCH_PID.write_text("4000000")
    assert dispatcher.running_pid() is None


def test_idle_hold_reasons():
    cfg = dispatcher.DISPATCH_DEFAULTS
    assert "in use" in dispatcher.idle_hold({"idle_seconds": 10, "power": "ac", "thermal_limit": 100}, cfg)
    assert dispatcher.idle_hold({"idle_seconds": 999, "power": "ac", "thermal_limit": 100}, cfg) is None
    assert "battery" in dispatcher.idle_hold({"idle_seconds": 999, "power": "battery", "thermal_limit": 100}, cfg)
    assert "thermal" in dispatcher.idle_hold({"idle_seconds": 999, "power": "ac", "thermal_limit": 30}, cfg)


def test_idle_hold_worker_busy_with_interactive_job():
    cfg = dispatcher.DISPATCH_DEFAULTS
    caps = {"idle_seconds": 9999, "power": "ac", "thermal_limit": 100, "busy_jobs": ["interactive-1"]}
    assert "interactive-1" in dispatcher.idle_hold(caps, cfg)


def test_launch_errors_fail_after_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(
        dispatcher.engine,
        "remote_caps",
        lambda worker: {
            "machine": "x86_64",
            "idle_seconds": 9999,
            "power": "ac",
            "thermal_limit": 100,
            "ram_gb": 8,
            "disk_free_gb": 30,
        },
    )
    monkeypatch.setattr(dispatcher.engine, "remote_job_running", lambda worker, jid: False)
    monkeypatch.setattr(
        dispatcher, "load_config", lambda: {**dispatcher.DISPATCH_DEFAULTS, "max_launch_errors": 2}
    )

    def boom(*a, **k):
        raise RuntimeError("venv sync exploded")

    monkeypatch.setattr(dispatcher.engine, "launch_remote", boom)
    dispatcher.enqueue(make_job(), "demo-err", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    workers = [Worker(name="pro", host="10.0.0.9")]
    lines = dispatcher.tick(workers=workers)
    assert any("launch error (1/2)" in line for line in lines)
    assert dispatcher.load_entries()[0]["state"] == "queued"
    lines = dispatcher.tick(workers=workers)
    assert any("FAILED" in line for line in lines)
    assert dispatcher.load_entries()[0]["state"] == "failed"


def test_adopts_already_running_job(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(
        dispatcher.engine,
        "remote_caps",
        lambda worker: {
            "machine": "x86_64",
            "idle_seconds": 0,
            "power": "ac",
            "thermal_limit": 100,
            "ram_gb": 8,
            "disk_free_gb": 30,
            "busy_jobs": ["demo-adopt"],
        },
    )
    monkeypatch.setattr(dispatcher.engine, "remote_job_running", lambda worker, jid: True)
    monkeypatch.setattr(dispatcher.engine, "remote_home", lambda worker: "/Users/x")

    def boom(*a, **k):
        raise AssertionError("must not relaunch an already-running job")

    monkeypatch.setattr(dispatcher.engine, "launch_remote", boom)
    dispatcher.enqueue(make_job(), "demo-adopt", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("adopted running job" in line for line in lines)
    entry = dispatcher.load_entries()[0]
    assert entry["state"] == "running"
    assert entry["rhome"] == "/Users/x"
    assert "adopted" in entry["note"]


def test_max_attempts_fails_job(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher, "load_config", lambda: dict(dispatcher.DISPATCH_DEFAULTS))
    entry = dispatcher.enqueue(make_job(), "demo-att", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    entry["attempts"] = dispatcher.DISPATCH_DEFAULTS["max_attempts"]
    dispatcher.save_entry(entry)
    lines = dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert any("FAILED after" in line for line in lines)
    assert dispatcher.load_entries()[0]["state"] == "failed"


def test_hold_resets_launch_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "remote_caps", lambda worker: None)
    entry = dispatcher.enqueue(make_job(), "demo-flap", Path("/tmp/p"), Path("/tmp/s"), worker="pro")
    entry["launch_errors"] = 4
    dispatcher.save_entry(entry)
    dispatcher.tick(workers=[Worker(name="pro", host="10.0.0.9")])
    assert dispatcher.load_entries()[0]["launch_errors"] == 0


def test_tick_lock_exclusive_and_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "TICK_LOCK", tmp_path / "tick.lock")
    assert dispatcher.acquire_tick_lock() is True
    assert dispatcher.acquire_tick_lock() is True
    assert dispatcher.TICK_LOCK.read_text() == str(os.getpid())
    dispatcher.TICK_LOCK.write_text("1")
    assert dispatcher.acquire_tick_lock() is False
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    proc.wait()
    dispatcher.TICK_LOCK.write_text(str(proc.pid))
    assert dispatcher.acquire_tick_lock() is True
    assert dispatcher.TICK_LOCK.read_text() == str(os.getpid())
    dispatcher.release_tick_lock()
    assert not dispatcher.TICK_LOCK.exists()


def test_idle_hold_owner_pause_wins():
    cfg = dispatcher.DISPATCH_DEFAULTS
    caps = {"idle_seconds": 9999, "power": "ac", "thermal_limit": 100, "policy_paused": True}
    assert "owner" in dispatcher.idle_hold(caps, cfg)


def test_log_holds_logs_once_per_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "EVENTS_PATH", tmp_path / "events.jsonl")
    dispatcher._hold_logged.clear()
    entry = {"id": "j1", "state": "queued", "worker": "pro", "note": "worker busy"}
    dispatcher._log_holds([entry], set())
    dispatcher._log_holds([entry], set())
    assert [r["event"] for r in events.load()] == ["held"]
    entry["note"] = "worker in use (idle 12s < 183s)"
    dispatcher._log_holds([entry], set())
    assert len(events.load()) == 2
    entry["state"] = "running"
    dispatcher._log_holds([entry], set())
    entry["state"] = "queued"
    dispatcher._log_holds([entry], set())
    assert len(events.load()) == 3


def test_log_holds_skips_stale_and_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "EVENTS_PATH", tmp_path / "events.jsonl")
    dispatcher._hold_logged.clear()
    stale = {"id": "j2", "state": "queued", "worker": "pro", "note": "worker busy"}
    blank = {"id": "j3", "state": "queued", "worker": "pro", "note": None}
    dispatcher._log_holds([stale, blank], {"j2"})
    assert events.load() == []


def test_local_start_held_when_air_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "air_busy", lambda *a: True)
    monkeypatch.setattr(dispatcher.engine, "air_load", lambda: 9.5)
    launched = []
    monkeypatch.setattr(
        dispatcher.engine, "launch_local", lambda *a, **k: (launched.append(1), (None, 0))[1]
    )
    dispatcher.enqueue(make_job(), "demo-air", Path("/tmp/p"), Path("/tmp/s"), worker="local")
    lines = dispatcher.tick(workers=[])
    assert any("air busy" in line for line in lines)
    entry = dispatcher.load_entries()[0]
    assert entry["state"] == "queued"
    assert entry["held_since"] is not None
    assert launched == []


def test_local_start_runs_when_air_free(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "air_busy", lambda *a: False)
    monkeypatch.setattr(dispatcher.engine, "launch_local", lambda *a, **k: (Path("/tmp/venv"), 123))
    dispatcher.enqueue(make_job(), "demo-air2", Path("/tmp/p"), Path("/tmp/s"), worker="local")
    lines = dispatcher.tick(workers=[])
    assert any("started locally" in line for line in lines)
    entry = dispatcher.load_entries()[0]
    assert entry["state"] == "running"
    assert entry["local_pid"] == 123


def test_local_start_fallback_exempt_from_air_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "air_busy", lambda *a: True)
    monkeypatch.setattr(dispatcher.engine, "launch_local", lambda *a, **k: (None, 7))
    entry = dispatcher.enqueue(make_job(), "demo-air3", Path("/tmp/p"), Path("/tmp/s"), worker="local")
    entry["note"] = "worker unreachable; falling back to local after 15m"
    dispatcher.save_entry(entry)
    lines = dispatcher.tick(workers=[])
    assert any("started locally" in line for line in lines)
    assert dispatcher.load_entries()[0]["state"] == "running"


def test_local_start_runs_after_air_busy_backstop(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher.engine, "air_busy", lambda *a: True)
    monkeypatch.setattr(dispatcher.engine, "air_load", lambda: 9.5)
    monkeypatch.setattr(dispatcher.engine, "launch_local", lambda *a, **k: (None, 9))
    entry = dispatcher.enqueue(make_job(), "demo-air4", Path("/tmp/p"), Path("/tmp/s"), worker="local")
    entry["held_since"] = time.time() - 2000
    dispatcher.save_entry(entry)
    lines = dispatcher.tick(workers=[])
    assert any("started locally" in line for line in lines)
