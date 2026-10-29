import json
import os
import subprocess
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
