import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import agent

STOCK_PYTHON = "/usr/bin/python3"


@pytest.mark.skipif(not Path(STOCK_PYTHON).exists(), reason="stock python3 not present")
def test_agent_runs_under_stock_python(tmp_path):
    result = subprocess.run(
        [STOCK_PYTHON, str(Path(agent.__file__)), "running"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr


def _exec(tmp_path, cmd, **fields):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"id": "j1", "cmd": cmd, "venv": None, "workdir": str(tmp_path), "job_dir": str(job_dir),
           "timeout_s": None, "preempt": False, "env": {}, **fields}
    (job_dir / "job.json").write_text(json.dumps(job))
    subprocess.run(
        [sys.executable, str(Path(agent.__file__)), "exec", str(job_dir / "job.json")],
        capture_output=True,
        timeout=60,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    return json.loads((job_dir / "status.json").read_text())


def test_exec_records_attempt(tmp_path):
    status = _exec(tmp_path, ["sh", "-c", 'echo "$ORCHARD_ATTEMPT" > attempt.txt'], attempt=3)
    assert (status["state"], status["attempt"]) == ("succeeded", 3)
    assert (tmp_path / "attempt.txt").read_text().strip() == "3"


def test_exec_timeout_stops_group(tmp_path):
    # A leader killed by SIGTERM stays a zombie until reaped; macOS then
    # answers the group probe with EPERM, which used to crash the agent.
    status = _exec(tmp_path, ["sleep", "30"], timeout_s=1)
    assert status["state"] == "timed_out"


def test_running_marker_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    proc = subprocess.Popen(["/bin/sh", "-c", "sleep 30"], start_new_session=True)
    try:
        agent.write_running_marker("job-1", proc.pid)
        assert agent.running_jobs() == ["job-1"]
        agent.clear_running_marker("job-1")
        assert agent.running_jobs() == []
    finally:
        proc.kill()
        proc.wait()


def test_running_marker_prunes_dead_group(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    agent.write_running_marker("dead-1", proc.pid)
    assert agent.running_jobs() == []


def test_running_marker_ignores_garbage(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    agent.write_running_marker("job-2", 0)
    path = agent._marker_path("job-2")
    with open(path, "w") as f:
        f.write("{not json")
    assert agent.running_jobs() == []


def test_cmd_running(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    proc = subprocess.Popen(["/bin/sh", "-c", "sleep 30"], start_new_session=True)
    try:
        agent.write_running_marker("job-9", proc.pid)
        assert agent.cmd_running([]) == 0
        assert capsys.readouterr().out.strip() == "job-9"
        assert agent.cmd_running(["job-9"]) == 0
        assert capsys.readouterr().out.strip() == "yes"
        assert agent.cmd_running(["nope"]) == 0
        assert capsys.readouterr().out.strip() == "no"
    finally:
        proc.kill()
        proc.wait()


def test_parse_hid_idle():
    text = '    | |   "HIDIdleTime" = 1500000000\n'
    assert agent.parse_hid_idle(text) == 1.5


def test_parse_hid_idle_missing():
    assert agent.parse_hid_idle("nothing here") is None


def test_parse_power():
    assert agent.parse_power("Now drawing from 'AC Power'") == "ac"
    assert agent.parse_power("Now drawing from 'Battery Power'") == "battery"
    assert agent.parse_power("garbage") == "unknown"


def test_parse_thermal_limit():
    assert agent.parse_thermal_limit("CPU_Scheduler_Limit \t= 100") == 100
    assert agent.parse_thermal_limit("No thermal warning level has been recorded") is None


def test_collect_caps_shape():
    caps = agent.collect_caps()
    assert caps["machine"] in {"arm64", "x86_64"}
    assert caps["cores"] > 0
    assert caps["ram_gb"] > 0
    assert caps["disk_free_gb"] > 0
    assert caps["bench"] > 0
    assert isinstance(caps["tags"], list)
    assert caps["machine"] in caps["tags"]


def test_capability_tags_include_platform():
    tags = agent.capability_tags("14.5")
    assert agent.os.uname().machine in tags
    assert "macos14" in tags
    assert "os:macos" in tags


def test_capability_tags_linux_has_no_mac_tags(monkeypatch):
    monkeypatch.setattr(agent, "_xcode_path", lambda: "/Applications/Xcode.app/Contents/Developer")
    tags = agent.capability_tags("ubuntu24.04", family="linux")
    assert "os:linux" in tags
    assert not any(t.startswith("macos") or t == "os:macos" for t in tags)
    assert "xcode" not in tags and "clt" not in tags


def test_os_family(monkeypatch):
    class Uname:
        def __init__(self, sysname):
            self.sysname = sysname

    for sysname, family in (("Darwin", "macos"), ("Linux", "linux"), ("FreeBSD", "freebsd")):
        monkeypatch.setattr(agent.os, "uname", lambda s=sysname: Uname(s))
        assert agent.os_family() == family


def test_parse_meminfo_total_gb():
    text = "MemTotal:       10169532 kB\nMemFree:         9001234 kB\n"
    assert agent.parse_meminfo_total_gb(text) == 10.41
    assert agent.parse_meminfo_total_gb("") == 0.0


def test_parse_os_release():
    text = 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nID=ubuntu\nVERSION_ID="24.04"\n'
    assert agent.parse_os_release(text) == "ubuntu24.04"
    assert agent.parse_os_release("") == "linux"


def write_host_status(path, now, age=5.0, **over):
    data = {"ts": "x", "idle_seconds": 600, "power": "ac", "battery_pct": 90, "disk_free_gb": 120.5}
    data.update(over)
    path.write_text(json.dumps(data))
    os.utime(path, (now - age, now - age))
    return path


def write_raw(path, now, text):
    path.write_text(text)
    os.utime(path, (now, now))
    return path


def test_read_host_status_fresh(tmp_path):
    now = 1.9e9
    path = write_host_status(tmp_path / "host.json", now)
    assert agent.read_host_status(str(path), now=now) == {
        "idle_seconds": 600.0, "power": "ac", "battery_pct": 90, "disk_free_gb": 120.5, "host_status": "ok",
    }


@pytest.mark.parametrize("setup, problem", [
    (lambda p, now: None, "missing"),
    (lambda p, now: write_host_status(p, now, age=120), "stale"),
    (lambda p, now: write_host_status(p, now, age=-120), "stale"),
    (lambda p, now: write_raw(p, now, "{not json"), "unreadable"),
    (lambda p, now: write_raw(p, now, "[1, 2]"), "unreadable"),
    (lambda p, now: write_host_status(p, now, idle_seconds=None), "incomplete"),
    (lambda p, now: write_host_status(p, now, idle_seconds=-1), "incomplete"),
    (lambda p, now: write_host_status(p, now, idle_seconds=True), "incomplete"),
    (lambda p, now: write_host_status(p, now, idle_seconds="600"), "incomplete"),
    (lambda p, now: write_host_status(p, now, power="mains"), "incomplete"),
    (lambda p, now: write_host_status(p, now, disk_free_gb=None), "incomplete"),
])
def test_read_host_status_fails_closed(tmp_path, setup, problem):
    now = 1.9e9
    path = tmp_path / "host.json"
    setup(path, now)
    status = agent.read_host_status(str(path), now=now)
    assert status["host_status"].startswith(problem)
    assert status["idle_seconds"] == 0.0
    assert status["power"] == "unknown"
    assert status["disk_free_gb"] is None


def test_read_host_status_rejects_infinite_idle(tmp_path):
    now = 1.9e9
    path = write_raw(tmp_path / "host.json", now, '{"idle_seconds": Infinity, "power": "ac", "disk_free_gb": 10}')
    assert agent.read_host_status(str(path), now=now)["host_status"] == "incomplete"


def test_worker_disk_free_gb_takes_host_minimum(tmp_path):
    free = agent.disk_free_gb(str(tmp_path))
    assert agent.worker_disk_free_gb(str(tmp_path)) == free
    assert agent.worker_disk_free_gb(str(tmp_path), {"disk_free_gb": None}) == free
    assert agent.worker_disk_free_gb(str(tmp_path), {"disk_free_gb": 0.5}) == min(free, 0.5)


def fake_linux(monkeypatch, tmp_path, host_status):
    files = {
        "/etc/os-release": 'ID=ubuntu\nVERSION_ID="24.04"\n',
        "/proc/meminfo": "MemTotal:       10169532 kB\n",
    }
    monkeypatch.setattr(agent, "os_family", lambda: "linux")
    monkeypatch.setattr(agent, "_read_text", lambda path: files.get(path, ""))
    monkeypatch.setattr(agent, "_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no mac probes")))
    monkeypatch.setattr(agent, "_xcode_path", lambda: (_ for _ in ()).throw(AssertionError("no xcode probe")))
    monkeypatch.setattr(agent, "cached_bench", lambda python: 1234)
    monkeypatch.setattr(agent, "HOST_STATUS_PATH", str(tmp_path / "host-status.json"))
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path))
    if host_status is not None:
        write_host_status(tmp_path / "host-status.json", agent.time.time(), **host_status)


def test_collect_caps_linux_reads_host_status(monkeypatch, tmp_path):
    fake_linux(monkeypatch, tmp_path, {"idle_seconds": 900, "power": "battery", "disk_free_gb": 0.25})
    caps = agent.collect_caps()
    assert caps["os"] == "ubuntu24.04"
    assert "os:linux" in caps["tags"]
    assert not any(t.startswith("macos") for t in caps["tags"])
    assert caps["cores"] == (os.cpu_count() or 0)
    assert caps["ram_gb"] == 10.41
    assert caps["idle_seconds"] == 900.0
    assert caps["power"] == "battery"
    assert caps["thermal_limit"] is None
    assert caps["disk_free_gb"] == 0.25
    assert caps["host_status"] == "ok"


def test_collect_caps_linux_fails_closed_without_host_status(monkeypatch, tmp_path):
    fake_linux(monkeypatch, tmp_path, None)
    caps = agent.collect_caps()
    assert caps["host_status"] == "missing"
    assert caps["idle_seconds"] == 0.0
    assert caps["power"] == "unknown"
    assert caps["disk_free_gb"] == agent.disk_free_gb(os.path.expanduser("~"))


def test_sample_preemption_linux_uses_host_status(monkeypatch, tmp_path):
    fake_linux(monkeypatch, tmp_path, None)
    sample_ = agent.sample_preemption_state(dict(agent.DEFAULT_POLICY), 0)
    assert sample_["host_status"] == "missing"
    assert agent.decide_preemption(sample_, dict(agent.DEFAULT_POLICY), False) == ("pause", "host status missing")


def test_collect_caps_macos_has_no_host_status():
    if agent.os_family() != "macos":
        pytest.skip("macOS only")
    caps = agent.collect_caps()
    assert caps["host_status"] is None
    assert "os:macos" in caps["tags"]


def test_benchmark_falls_back_on_bad_python():
    value = agent.benchmark("/nonexistent/python", iterations=1000)
    assert value > 0


def test_benchmark_uses_python_path(tmp_path):
    script = tmp_path / "fake-python"
    script.write_text("#!/bin/sh\necho 12345\n")
    script.chmod(0o755)
    assert agent.benchmark(str(script)) == 12345


def test_cached_bench_reuses_value(tmp_path):
    script = tmp_path / "fake-python"
    script.write_text("#!/bin/sh\necho 999\n")
    script.chmod(0o755)
    assert agent.cached_bench(str(script), cache_dir=str(tmp_path)) == 999
    script.write_text("#!/bin/sh\necho 111\n")
    assert agent.cached_bench(str(script), cache_dir=str(tmp_path)) == 999


def test_cached_bench_normalizes_missing_interpreter(tmp_path):
    value = agent.cached_bench("/nonexistent/python", cache_dir=str(tmp_path))
    assert value > 0
    data = json.loads((tmp_path / "bench.json").read_text())
    assert data["python"] == agent.sys.executable


def test_wrap_full_speed_by_default():
    wrapped = agent._wrap(["echo", "hi"], policy=agent.DEFAULT_POLICY)
    joined = " ".join(wrapped)
    assert "nice" not in joined
    assert "taskpolicy" not in joined


def test_wrap_background_qos_opt_in():
    policy = {**agent.DEFAULT_POLICY, "background_qos": True}
    wrapped = agent._wrap(["echo", "hi"], policy=policy)
    assert any("nice" in part for part in wrapped)


def sample(**over):
    base = {
        "idle_seconds": 600,
        "power": "ac",
        "battery_pct": 80,
        "thermal_limit": 100,
        "other_cpu_pct": None,
    }
    base.update(over)
    return base


def test_parse_battery_pct():
    text = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=7077987)\t9%; AC attached; not charging present: true"
    assert agent.parse_battery_pct(text) == 9
    assert agent.parse_battery_pct("no battery here") is None


def test_decide_low_battery_pauses_even_on_ac():
    action, reason = agent.decide_preemption(sample(battery_pct=9), agent.DEFAULT_POLICY, paused=False)
    assert action == "pause"
    assert "battery low" in reason
    assert agent.decide_preemption(sample(battery_pct=25), agent.DEFAULT_POLICY, paused=False) == (None, None)


def test_decide_owner_pause_flag():
    policy = {**agent.DEFAULT_POLICY, "paused": True}
    action, reason = agent.decide_preemption(sample(), policy, paused=False)
    assert action == "pause"
    assert "owner" in reason
    assert agent.decide_preemption(sample(), policy, paused=True) == (None, None)


def test_decide_pause_when_user_active():
    action, reason = agent.decide_preemption(sample(idle_seconds=10), agent.DEFAULT_POLICY, paused=False)
    assert action == "pause"
    assert "active" in reason


def test_decide_runs_when_idle():
    assert agent.decide_preemption(sample(), agent.DEFAULT_POLICY, paused=False) == (None, None)


def test_decide_host_status_problem_pauses_and_blocks_resume():
    policy = dict(agent.DEFAULT_POLICY, pause_if_user_active=False, pause_on_battery=False)
    assert agent.decide_preemption(sample(host_status="stale (90s old)"), policy, False) == (
        "pause", "host status stale (90s old)"
    )
    assert agent.decide_preemption(sample(host_status="missing"), policy, True) == (None, None)
    assert agent.decide_preemption(sample(host_status="ok"), policy, False) == (None, None)
    assert agent.decide_preemption(sample(host_status="ok"), policy, True) == ("resume", None)


def test_decide_no_pause_when_idle_unknown():
    assert agent.decide_preemption(sample(idle_seconds=None), agent.DEFAULT_POLICY, paused=False) == (None, None)


def test_decide_resume_respects_margin():
    threshold = agent.DEFAULT_POLICY["idle_threshold_s"]
    just_over = sample(idle_seconds=threshold + 10)
    assert agent.decide_preemption(just_over, agent.DEFAULT_POLICY, paused=True) == (None, None)
    well_over = sample(idle_seconds=threshold + agent.RESUME_MARGIN_S + 10)
    assert agent.decide_preemption(well_over, agent.DEFAULT_POLICY, paused=True)[0] == "resume"


def test_decide_battery_pauses():
    action, reason = agent.decide_preemption(sample(power="battery"), agent.DEFAULT_POLICY, paused=False)
    assert action == "pause"
    assert "battery" in reason


def test_decide_thermal_hysteresis():
    policy = agent.DEFAULT_POLICY
    assert agent.decide_preemption(sample(thermal_limit=40), policy, paused=False)[0] == "pause"
    assert agent.decide_preemption(sample(thermal_limit=55), policy, paused=True) == (None, None)
    assert agent.decide_preemption(sample(thermal_limit=65), policy, paused=True)[0] == "resume"


def test_decide_other_cpu_off_by_default():
    assert agent.decide_preemption(sample(other_cpu_pct=90), agent.DEFAULT_POLICY, paused=False) == (None, None)


def test_decide_other_cpu_opt_in():
    policy = {**agent.DEFAULT_POLICY, "max_other_cpu_pct": 25}
    assert agent.decide_preemption(sample(other_cpu_pct=90), policy, paused=False)[0] == "pause"


def garbage(path, age_days, size_gb, protected=False, now=1e9):
    return {"path": path, "mtime": now - age_days * 86400, "size": int(size_gb * 1e9), "protected": protected}


def test_select_garbage_by_age():
    entries = [garbage("old", 20, 0.5), garbage("new", 1, 0.5)]
    picked = agent.select_garbage(entries, 1e9, 14, 3, 10)
    assert [e["path"] for e in picked] == ["old"]


def test_select_garbage_disk_pressure_picks_oldest_first():
    entries = [garbage("a", 1, 2), garbage("b", 2, 2), garbage("c", 3, 2)]
    picked = agent.select_garbage(entries, 1e9, 14, 3, 1)
    assert [e["path"] for e in picked] == ["c"]


def test_select_garbage_protects_running_jobs():
    entries = [garbage("run", 30, 5, protected=True)]
    assert agent.select_garbage(entries, 1e9, 14, 3, 0.1) == []


def test_select_garbage_nothing_when_plenty_free():
    entries = [garbage("recent", 1, 0.1)]
    assert agent.select_garbage(entries, 1e9, 14, 3, 50) == []
