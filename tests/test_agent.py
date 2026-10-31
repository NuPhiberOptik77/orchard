import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
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


def test_exec_records_its_own_exit(tmp_path):
    status = _exec(tmp_path, ["sh", "-c", "exit 3"], attempt=2)
    assert (status["state"], status["exit_code"]) == ("failed", 3)
    ended = json.loads((tmp_path / "job" / "exit.json").read_text())
    assert (ended["exit_code"], ended["attempt"]) == (3, 2)


def test_exec_wrapper_exits_the_way_the_command_did(tmp_path):
    (tmp_path / "signal").mkdir()
    status = _exec(tmp_path / "signal", ["sh", "-c", "kill -TERM $$"])
    assert status["exit_code"] == -signal.SIGTERM
    assert json.loads((tmp_path / "signal" / "job" / "exit.json").read_text())["exit_code"] == -signal.SIGTERM
    (tmp_path / "missing").mkdir()
    status = _exec(tmp_path / "missing", ["orchard-no-such-command"])
    assert (status["state"], status["exit_code"]) == ("failed", 127)


def test_wrap_keeps_ignored_signals_ignored(tmp_path):
    # A remote agent starts under nohup: its jobs got SIGHUP ignored before
    # `wrap` existed, and still do. Signals it didn't ignore are the default.
    probe = "import signal; print(signal.getsignal(signal.SIGHUP) == signal.SIG_IGN, " \
            "signal.getsignal(signal.SIGTERM) == signal.SIG_DFL)"
    out = subprocess.run(
        [sys.executable, str(Path(agent.__file__)), "wrap", str(tmp_path / "exit.json"), "1", "--",
         sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=30,
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),
    )
    assert out.stdout.split() == ["True", "True"]


def test_exec_runs_commands_the_way_a_shell_does(tmp_path):
    # A script with no #! line runs under /bin/sh, as it did under caffeinate.
    (tmp_path / "plain").mkdir()
    script = tmp_path / "plain" / "run-me"
    script.write_text("echo ran > ran.txt\n")
    script.chmod(0o755)
    status = _exec(tmp_path / "plain", [str(script)])
    assert (status["state"], status["exit_code"]) == ("succeeded", 0)
    assert (tmp_path / "plain" / "ran.txt").read_text().strip() == "ran"
    # A file that isn't executable fails 126, like in a shell.
    (tmp_path / "noexec").mkdir()
    script = tmp_path / "noexec" / "run-me"
    script.write_text("echo ran\n")
    script.chmod(0o644)
    status = _exec(tmp_path / "noexec", [str(script)])
    assert (status["state"], status["exit_code"]) == ("failed", 126)


def test_exec_survives_eperm_on_pause(tmp_path, monkeypatch):
    # macOS answers EPERM when every process in the group is an unreaped
    # zombie, which used to crash the agent and leave the job running forever.
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(agent, "load_policy", lambda: dict(agent.DEFAULT_POLICY))
    monkeypatch.setattr(agent, "sample_preemption_state", lambda policy, pgid: {})
    monkeypatch.setattr(agent, "decide_preemption", lambda sample, policy, paused: ("pause", "test"))
    real_killpg = os.killpg

    def killpg(pgid, sig):
        if sig == signal.SIGSTOP:
            raise PermissionError(1, "Operation not permitted")
        real_killpg(pgid, sig)

    monkeypatch.setattr(agent.os, "killpg", killpg)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"id": "j1", "cmd": ["sh", "-c", "sleep 1"], "venv": None, "workdir": str(tmp_path),
           "job_dir": str(job_dir), "timeout_s": None, "preempt": True, "env": {}, "attempt": 1}
    (job_dir / "job.json").write_text(json.dumps(job))
    assert agent.cmd_exec([str(job_dir / "job.json")]) == 0
    assert json.loads((job_dir / "status.json").read_text())["state"] == "succeeded"


def _preempted(tmp_path, monkeypatch, shell, actions, killpg_errors):
    """Run shell as a preemptible job in-process.

    `actions` answers each preemption decision in turn (the last one repeats),
    or is a function of whether the job is paused that answers it.
    `killpg_errors` maps a signal to how many killpg calls with it raise EPERM.
    Returns cmd_exec's code, the final status and the signals killpg was sent.
    """
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(agent, "POLL_S", 0.2)
    monkeypatch.setattr(agent, "load_policy", lambda: dict(agent.DEFAULT_POLICY))
    monkeypatch.setattr(agent, "sample_preemption_state", lambda policy, pgid: {})
    answers = list(actions) if not callable(actions) else None

    def decide(sample, policy, paused):
        if answers is None:
            return actions(paused)
        return answers.pop(0) if len(answers) > 1 else answers[0]

    monkeypatch.setattr(agent, "decide_preemption", decide)
    real_killpg = os.killpg
    sent = []

    def killpg(pgid, sig):
        sent.append(sig)
        if killpg_errors.get(sig):
            killpg_errors[sig] -= 1
            raise PermissionError(1, "Operation not permitted")
        real_killpg(pgid, sig)

    monkeypatch.setattr(agent.os, "killpg", killpg)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"id": "j1", "cmd": ["sh", "-c", shell], "venv": None, "workdir": str(tmp_path),
           "job_dir": str(job_dir), "timeout_s": None, "preempt": True, "env": {}, "attempt": 1}
    (job_dir / "job.json").write_text(json.dumps(job))
    rc = agent.cmd_exec([str(job_dir / "job.json")])
    return rc, json.loads((job_dir / "status.json").read_text()), sent


def test_exec_survives_eperm_on_resume(tmp_path, monkeypatch):
    rc, status, sent = _preempted(tmp_path, monkeypatch, "sleep 1", [("resume", "test")], {signal.SIGCONT: 1})
    assert (rc, status["state"]) == (0, "succeeded")
    assert sent.count(signal.SIGCONT) >= 2


def _wait_for_state(job_dir, state, limit=10):
    deadline = time.time() + limit
    while time.time() < deadline:
        try:
            if json.loads((job_dir / "status.json").read_text()).get("state") == state:
                return True
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    return False


def _wait_for_text(path, limit=10):
    deadline = time.time() + limit
    while time.time() < deadline:
        try:
            text = path.read_text().strip()
            if text:
                return text
        except OSError:
            pass
        time.sleep(0.05)
    raise AssertionError("%s never written" % path)


def _ended(pid):
    """True once pid is gone or a zombie (a container's init may never reap it)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        return Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


def _command_line(pid):
    if sys.platform.startswith("linux"):
        return Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\0", b" ").decode()
    return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout


def test_job_killed_while_paused_ends_at_once(tmp_path, monkeypatch):
    # Once the command runs, the pause leaves `wrap` running, so a command
    # SIGKILLed while paused is recorded at once instead of reading as paused
    # until the job resumes.
    job_dir = tmp_path / "job"

    def kill_command():
        _wait_for_state(job_dir, "paused")
        os.kill(int(_wait_for_text(job_dir / "command.pid")), signal.SIGKILL)

    def decide(paused):
        started = (job_dir / "command.pid").exists()
        return ("pause", "test") if started and not paused else (None, None)

    killer = threading.Thread(target=kill_command, daemon=True)
    killer.start()
    started = time.time()
    rc, status, sent = _preempted(tmp_path, monkeypatch, "exec sleep 30", decide, {})
    killer.join()
    assert time.time() - started < 5
    assert (rc, status["state"], status["exit_code"]) == (1, "failed", -signal.SIGKILL)
    # One pause (two stops), then the agent's check that nothing of the group is left.
    assert sent == [signal.SIGSTOP, signal.SIGSTOP, 0]
    ended = json.loads((tmp_path / "job" / "exit.json").read_text())
    assert ended["exit_code"] == -signal.SIGKILL


def test_pause_before_the_command_starts_holds(tmp_path, monkeypatch):
    # The first check comes right after the agent starts `wrap`, which may not
    # have started the command yet. wrap then stays stopped with the rest of
    # the group: let go, it would start the command unpaused.
    lines = tmp_path / "lines"
    action = [("pause", "test")]
    seen = {}

    def count():
        try:
            return len(lines.read_text().splitlines())
        except OSError:
            return 0

    def drive():
        try:
            _wait_for_state(tmp_path / "job", "paused")
            before = count()
            time.sleep(2)
            seen["paused"] = (before, count())
            action[0] = ("resume", "test")
            deadline = time.time() + 10
            while count() < before + 3 and time.time() < deadline:
                time.sleep(0.1)
            seen["resumed"] = count()
        finally:
            action[0] = ("resume", "test")
            (tmp_path / "stop").write_text("")

    driver = threading.Thread(target=drive, daemon=True)
    driver.start()
    rc, status, sent = _preempted(tmp_path, monkeypatch, "while [ ! -f stop ]; do echo x >> lines; sleep 0.2; done",
                                  lambda paused: action[0], {})
    driver.join()
    assert sent[0] == signal.SIGSTOP
    before, after = seen["paused"]
    assert after == before
    assert seen["resumed"] >= before + 3
    assert (rc, status["state"]) == (0, "succeeded")


def test_exec_survives_a_failed_status_write(tmp_path, monkeypatch):
    # A full disk while the job runs must not kill the agent: it would leave
    # the job reading "running" forever.
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    real_write = agent._atomic_write_json
    writes = []

    def write(path, data):
        writes.append(data.get("state"))
        # The first two writes (before and after the start) are not the loop's.
        if len(writes) > 2 and data.get("state") == "running":
            raise OSError(errno.ENOSPC, "No space left on device")
        real_write(path, data)

    monkeypatch.setattr(agent, "_atomic_write_json", write)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"id": "j1", "cmd": ["sh", "-c", "sleep 1"], "venv": None, "workdir": str(tmp_path),
           "job_dir": str(job_dir), "timeout_s": None, "preempt": False, "env": {}, "attempt": 1}
    (job_dir / "job.json").write_text(json.dumps(job))
    assert agent.cmd_exec([str(job_dir / "job.json")]) == 0
    assert writes[2] == "running" and writes[-1] == "succeeded"
    assert json.loads((job_dir / "status.json").read_text())["state"] == "succeeded"


def test_call_command_passes_on_an_early_signal(tmp_path):
    # A signal that reached `wrap` before the command started is sent on to
    # its process group once the command has, so a stop request between the
    # two is not lost. It runs in a session of its own, since the group gets it.
    script = tmp_path / "run-me"
    script.write_text("exec sleep 30\n")
    script.chmod(0o755)
    code = (
        "import signal, sys, time\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import agent\n"
        "signal.signal(signal.SIGTERM, lambda signum, frame: None)\n"
        "started = time.time()\n"
        "print(agent._call_command(['sleep', '30'], [signal.SIGTERM]),\n"
        "      agent._call_command([sys.argv[2]], [signal.SIGTERM]), time.time() - started < 10)\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code, os.path.dirname(agent.__file__), str(script)],
                         capture_output=True, text=True, timeout=30, start_new_session=True)
    assert out.stdout.split() == [str(-signal.SIGTERM)] * 2 + ["True"], out.stderr
    assert agent._call_command(["sh", "-c", "exit 3"], []) == 3


def test_call_command_survives_a_failed_start(monkeypatch):
    # Popen raises SubprocessError, not OSError, when the child dies between
    # fork and exec. The command never started: 126, or the early signal.
    def popen(*a, **k):
        raise subprocess.SubprocessError("Exception occurred in preexec_fn.")

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    assert agent._call_command(["true"], []) == 126
    assert agent._call_command(["true"], [signal.SIGTERM]) == -signal.SIGTERM


def test_exec_drops_the_last_attempts_files_before_starting(tmp_path, monkeypatch):
    # Until this attempt's pid is written, a kill or pause must find no
    # process rather than the last attempt's.
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(agent, "load_policy", lambda: dict(agent.DEFAULT_POLICY))
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    stale = [job_dir / "pid", job_dir / "exit.json", Path(agent._started_path(str(job_dir)))]
    for path in stale:
        path.write_text("1")
    seen = []
    real_popen = agent.subprocess.Popen

    def popen(argv, **kw):
        if "wrap" in argv:
            seen.append([path.exists() for path in stale])
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
        return real_popen(argv, **kw)

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    job = {"id": "j1", "cmd": ["true"], "venv": None, "workdir": str(tmp_path),
           "job_dir": str(job_dir), "timeout_s": None, "preempt": False, "env": {}, "attempt": 2}
    (job_dir / "job.json").write_text(json.dumps(job))
    assert agent.cmd_exec([str(job_dir / "job.json")]) == 1
    assert seen == [[False, False, False]]
    assert json.loads((job_dir / "status.json").read_text())["state"] == "failed"


def test_wrap_passes_on_a_signal_that_came_before_the_command(tmp_path):
    # SIGTERM reaching wrap before the command starts is sent on once it has,
    # and wrap then records it and ends the way the command did.
    code = (
        "import os, signal, subprocess, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import agent\n"
        "real = subprocess.Popen\n"
        "def popen(*args, **kwargs):\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    return real(*args, **kwargs)\n"
        "subprocess.Popen = popen\n"
        "sys.exit(agent.cmd_wrap([sys.argv[2], '1', '--', 'sleep', '30']))\n"
    )
    exit_path = tmp_path / "exit.json"
    started = time.time()
    proc = subprocess.run([sys.executable, "-I", "-c", code, os.path.dirname(agent.__file__), str(exit_path)],
                          timeout=10, start_new_session=True)
    assert time.time() - started < 10
    assert proc.returncode == -signal.SIGTERM
    ended = json.loads(exit_path.read_text())
    assert (ended["exit_code"], ended["signal"]) == (-signal.SIGTERM, signal.SIGTERM)


def test_kill_ends_a_real_job_through_its_fences(tmp_path, monkeypatch):
    # The pid file names `wrap`, which leads the job's group, and the status
    # holds its start time, so `kill` gets past its fences and ends the job.
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"id": "j1", "cmd": ["sh", "-c", "exec sleep 30"], "venv": None, "workdir": str(tmp_path),
           "job_dir": str(job_dir), "timeout_s": None, "preempt": False, "env": {}, "attempt": 1}
    (job_dir / "job.json").write_text(json.dumps(job))
    proc = subprocess.Popen([sys.executable, str(Path(agent.__file__)), "exec", str(job_dir / "job.json")],
                            env={**os.environ, "HOME": str(tmp_path)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = None
    try:
        _wait_for_text(job_dir / "command.pid")
        deadline = time.time() + 10
        while not json.loads((job_dir / "status.json").read_text()).get("pgid") and time.time() < deadline:
            time.sleep(0.05)
        pid = int((job_dir / "pid").read_text())
        status = json.loads((job_dir / "status.json").read_text())
        assert status["pid_start"] == agent.process_start(pid)
        if sys.platform.startswith("linux"):
            assert status["pid_start"] is not None
        assert os.getpgid(pid) == pid == status["pgid"]
        assert " wrap " in _command_line(pid)
        monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / ".orchard"))
        assert agent.cmd_signal([str(job_dir)], signal.SIGTERM, "kill", cont_first=True) == 0
        assert proc.wait(timeout=15) == 1
        assert json.loads((job_dir / "status.json").read_text())["exit_code"] == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if pid:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


def test_kill_ends_a_command_that_outlived_wrap(tmp_path, monkeypatch):
    # SIGKILL can't be shrugged off, so wrap can die before its command. The
    # command keeps wrap's group, which `kill` signals.
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    job_dir = tmp_path / "home" / "jobs" / "job-s"
    job_dir.mkdir(parents=True)
    wrap = subprocess.Popen([sys.executable, "-I", str(Path(agent.__file__)), "wrap", str(job_dir / "exit.json"),
                             "1", "--", "sleep", "30"], start_new_session=True)
    try:
        command = int(_wait_for_text(job_dir / "command.pid"))
        (job_dir / "pid").write_text(str(wrap.pid))
        (job_dir / "status.json").write_text(json.dumps({
            "state": "running", "pgid": wrap.pid, "boot_id": agent.boot_id(),
            "pid_start": agent.process_start(wrap.pid)}))
        agent.write_running_marker("job-s", wrap.pid)
        wrap.kill()
        wrap.wait()
        assert not _ended(command)
        assert agent.cmd_signal([str(job_dir)], signal.SIGTERM, "kill", cont_first=True) == 0
        deadline = time.time() + 5
        while not _ended(command) and time.time() < deadline:
            time.sleep(0.05)
        assert _ended(command)
    finally:
        try:
            os.killpg(wrap.pid, signal.SIGKILL)
        except OSError:
            pass


def test_agent_ends_a_command_that_outlived_wrap(tmp_path):
    # wrap SIGKILLed mid-job: the agent records the job and ends what is left
    # of its group instead of leaving the command running unwatched.
    started = time.time()
    status = _exec(tmp_path, ["sh", "-c", 'echo $$ > cmd.pid; kill -KILL "$PPID"; exec sleep 30'])
    assert time.time() - started < 30
    assert (status["state"], status["exit_code"]) == ("failed", -signal.SIGKILL)
    assert _ended(int((tmp_path / "cmd.pid").read_text()))


def test_pause_lets_wrap_run_only_once_the_command_started(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    job_dir = tmp_path / "home" / "jobs" / "job-p"
    job_dir.mkdir(parents=True)
    (job_dir / "pid").write_text("4242")
    (job_dir / "status.json").write_text(json.dumps({"state": "running", "pgid": 4242}))
    calls = []
    monkeypatch.setattr(agent.os, "killpg", lambda pgid, sig: calls.append(("group", pgid, sig)))
    monkeypatch.setattr(agent.os, "kill", lambda pid, sig: calls.append(("leader", pid, sig)))
    assert agent.cmd_signal([str(job_dir)], signal.SIGSTOP, "pause") == 0
    # Stopped twice: macOS can miss a process that is forking during the first.
    assert calls == [("group", 4242, signal.SIGSTOP)] * 2
    calls.clear()
    (job_dir / "command.pid").write_text("4243")
    assert agent.cmd_signal([str(job_dir)], signal.SIGSTOP, "pause") == 0
    assert calls == [("group", 4242, signal.SIGSTOP)] * 2 + [("leader", 4242, signal.SIGCONT)]


def test_keep_awake_watches_wrap(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.shutil, "which", lambda name: "/usr/bin/caffeinate" if name == "caffeinate" else None)
    seen = {}

    def popen(argv, **kwargs):
        seen.update(kwargs, argv=argv)
        return "proc"

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    assert agent._keep_awake(4242) == "proc"
    assert seen["argv"] == ["/usr/bin/caffeinate", "-is", "-w", "4242"]
    assert seen["start_new_session"] is True
    monkeypatch.setattr(agent.shutil, "which", lambda name: None)
    assert agent._keep_awake(4242) is None


def test_exec_keeps_awake_for_wrap(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    watched = []
    monkeypatch.setattr(agent, "_keep_awake", watched.append)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"id": "j1", "cmd": ["sh", "-c", "exit 0"], "venv": None, "workdir": str(tmp_path),
           "job_dir": str(job_dir), "timeout_s": None, "preempt": False, "env": {}, "attempt": 1}
    (job_dir / "job.json").write_text(json.dumps(job))
    assert agent.cmd_exec([str(job_dir / "job.json")]) == 0
    assert watched == [int((job_dir / "pid").read_text())]


def _settle_case(tmp_path, monkeypatch, status, ended):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    job_dir = tmp_path / "job"
    job_dir.mkdir(exist_ok=True)
    (job_dir / "status.json").write_text(json.dumps(status))
    if ended is not None:
        (job_dir / "exit.json").write_text(json.dumps(ended))
    return job_dir


def test_settle_finishes_a_run_whose_agent_died(tmp_path, monkeypatch):
    running = {"id": "j1", "state": "paused", "attempt": 2, "started_at": 100.0, "pause_reason": "user active"}
    job_dir = _settle_case(tmp_path, monkeypatch, running, {"exit_code": 0, "attempt": 2, "ended_at": 160.0})
    status = agent.settle_status(str(job_dir))
    assert (status["state"], status["exit_code"], status["elapsed_s"]) == ("succeeded", 0, 60.0)
    assert status["pause_reason"] is None
    assert json.loads((job_dir / "status.json").read_text())["state"] == "succeeded"
    (job_dir / "status.json").write_text(json.dumps(running))
    (job_dir / "exit.json").write_text(json.dumps({"exit_code": 4, "attempt": 2, "ended_at": 160.0}))
    assert agent.settle_status(str(job_dir))["state"] == "failed"


def test_settle_leaves_runs_it_cannot_vouch_for(tmp_path, monkeypatch):
    running = {"id": "j1", "state": "running", "attempt": 2, "started_at": 100.0}
    # No exit record: the job died with its agent, so it has to run again.
    job_dir = _settle_case(tmp_path, monkeypatch, running, None)
    assert agent.settle_status(str(job_dir))["state"] == "running"
    # A record from an earlier attempt says nothing about this one.
    _settle_case(tmp_path, monkeypatch, running, {"exit_code": 0, "attempt": 1, "ended_at": 160.0})
    assert agent.settle_status(str(job_dir))["state"] == "running"
    # A finished status is already final.
    done = dict(running, state="timed_out")
    _settle_case(tmp_path, monkeypatch, done, {"exit_code": 0, "attempt": 2, "ended_at": 160.0})
    assert agent.settle_status(str(job_dir))["state"] == "timed_out"
    # While the group is alive, the record can't be this run's last word.
    _settle_case(tmp_path, monkeypatch, running, {"exit_code": 0, "attempt": 2, "ended_at": 160.0})
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
    try:
        agent.write_running_marker("j1", proc.pid)
        assert agent.settle_status(str(job_dir))["state"] == "running"
    finally:
        proc.kill()
        proc.wait()
    assert agent.settle_status(str(job_dir))["state"] == "succeeded"


def _outlive_agent(tmp_path, shell, ready, stop_agent, stop_job=None):
    """Start shell as a job, stop its agent once `ready` exists, let the job end, then settle."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"id": "j1", "cmd": ["sh", "-c", shell], "venv": None, "workdir": str(tmp_path),
           "job_dir": str(job_dir), "timeout_s": None, "preempt": False, "env": {}, "attempt": 1}
    (job_dir / "job.json").write_text(json.dumps(job))
    env = {**os.environ, "HOME": str(tmp_path)}
    script = str(Path(agent.__file__))
    proc = subprocess.Popen([sys.executable, script, "exec", str(job_dir / "job.json")], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    pgid = None
    while (pgid is None or not ready.exists()) and time.time() < deadline:
        try:
            pgid = json.loads((job_dir / "status.json").read_text()).get("pgid")
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    assert pgid and ready.exists()
    # The job runs in its own session, so the agent can die without it.
    os.kill(proc.pid, stop_agent)
    proc.wait()
    if stop_job is not None:
        os.killpg(pgid, stop_job)
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.05)
    assert json.loads((job_dir / "status.json").read_text())["state"] == "running"
    out = subprocess.run([sys.executable, script, "settle", str(job_dir)], env=env, capture_output=True,
                         timeout=30, check=True)
    return json.loads(out.stdout), json.loads((job_dir / "exit.json").read_text())


def test_job_that_outlives_its_agent_settles(tmp_path):
    shell = "touch ready; sleep 1; echo done > out/result.txt"
    status, _ = _outlive_agent(tmp_path, shell, tmp_path / "ready", signal.SIGKILL)
    assert (status["state"], status["exit_code"]) == ("succeeded", 0)
    assert (tmp_path / "out" / "result.txt").read_text().strip() == "done"


def test_job_stopped_by_a_shutdown_is_not_settled(tmp_path):
    # A shutdown SIGTERMs every process. A job that catches it and exits 0
    # didn't finish, so its run stays open for the dispatcher to run again.
    shell = "trap 'exit 0' TERM; touch ready; sleep 5 & wait"
    status, ended = _outlive_agent(tmp_path, shell, tmp_path / "ready", signal.SIGTERM, signal.SIGTERM)
    assert (ended["exit_code"], ended["signal"]) == (0, signal.SIGTERM)
    assert status["state"] == "running"


def test_running_marker_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
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


def test_running_marker_from_earlier_boot_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
    try:
        monkeypatch.setattr(agent, "boot_id", lambda: "boot-a")
        agent.write_running_marker("job-3", proc.pid)
        assert agent.running_jobs() == ["job-3"]
        # After a restart the same pgid can belong to another process.
        monkeypatch.setattr(agent, "boot_id", lambda: "boot-b")
        assert agent.running_jobs() == []
        assert not os.path.exists(agent._marker_path("job-3"))
        # Markers written before boot ids, or with no id to compare, go by the pgid.
        monkeypatch.setattr(agent, "boot_id", lambda: None)
        agent.write_running_marker("job-4", proc.pid)
        monkeypatch.setattr(agent, "boot_id", lambda: "boot-b")
        assert agent.running_jobs() == ["job-4"]
    finally:
        proc.kill()
        proc.wait()


def test_boot_id_is_stable():
    first = agent.boot_id()
    assert first and first == agent.boot_id()


def _job_with_status(home, job_id, state, age_s):
    job_dir = os.path.join(home, "jobs", job_id)
    os.makedirs(job_dir)
    status_path = os.path.join(job_dir, "status.json")
    with open(status_path, "w") as f:
        json.dump({"state": state}, f)
    stamp = time.time() - age_s
    os.utime(status_path, (stamp, stamp))
    os.utime(job_dir, (stamp, stamp))


def test_gc_frees_runs_that_died_with_their_machine(tmp_path, monkeypatch):
    home = str(tmp_path / "home")
    monkeypatch.setattr(agent, "ORCHARD_HOME", home)
    _job_with_status(home, "dead", "running", 3 * 86400)
    # Died an hour ago: kept, so the dispatcher can resume it from ckpt/.
    _job_with_status(home, "recent", "running", 3600)
    _job_with_status(home, "fresh", "running", 5)
    _job_with_status(home, "live", "paused", 3 * 86400)
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
    try:
        agent.write_running_marker("live", proc.pid)
        protected = {os.path.basename(e["path"]): e["protected"] for e in agent._gc_entries(time.time(), 2)}
    finally:
        proc.kill()
        proc.wait()
    assert protected == {"dead": False, "recent": True, "fresh": True, "live": True}


def test_kill_skips_a_pid_from_an_earlier_boot(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    job_dir = tmp_path / "home" / "jobs" / "job-k"
    job_dir.mkdir(parents=True)
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
    try:
        (job_dir / "pid").write_text(str(proc.pid))
        (job_dir / "status.json").write_text(json.dumps({"state": "running", "boot_id": "boot-a"}))
        # After a restart the old pid may be someone else's process.
        monkeypatch.setattr(agent, "boot_id", lambda: "boot-b")
        assert agent.cmd_signal([str(job_dir)], signal.SIGTERM, "kill", cont_first=True) == 1
        assert proc.poll() is None
        monkeypatch.setattr(agent, "boot_id", lambda: "boot-a")
        assert agent.cmd_signal([str(job_dir)], signal.SIGTERM, "kill", cont_first=True) == 0
        assert proc.wait(timeout=5) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_kill_skips_a_reused_pid(tmp_path, monkeypatch):
    # `wsl --terminate` keeps the boot id but hands pids out again.
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    job_dir = tmp_path / "home" / "jobs" / "job-r"
    job_dir.mkdir(parents=True)
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
    try:
        (job_dir / "pid").write_text(str(proc.pid))
        (job_dir / "status.json").write_text(json.dumps({"state": "running", "pid_start": 111}))
        agent.write_running_marker("job-r", proc.pid)
        monkeypatch.setattr(agent, "process_start", lambda pid: 222)
        assert agent.cmd_signal([str(job_dir)], signal.SIGTERM, "kill", cont_first=True) == 1
        assert proc.poll() is None
        assert not os.path.exists(agent._marker_path("job-r"))
        # A start time it can't read (macOS) goes by the pid, as before.
        monkeypatch.setattr(agent, "process_start", lambda pid: None)
        assert agent.cmd_signal([str(job_dir)], signal.SIGTERM, "kill", cont_first=True) == 0
        assert proc.wait(timeout=5) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_running_marker_from_a_restarted_distro_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
    try:
        monkeypatch.setattr(agent, "process_start", lambda pid: 111)
        agent.write_running_marker("job-5", proc.pid)
        assert agent.running_jobs() == ["job-5"]
        monkeypatch.setattr(agent, "process_start", lambda pid: None)
        assert agent.running_jobs() == ["job-5"]
        monkeypatch.setattr(agent, "process_start", lambda pid: 222)
        assert agent.running_jobs() == []
        assert not os.path.exists(agent._marker_path("job-5"))
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_process_start_reads_proc():
    start = agent.process_start(os.getpid())
    assert isinstance(start, int) and start == agent.process_start(os.getpid())
    assert agent.process_start(1) != start


def test_process_start_unknown_off_linux(monkeypatch):
    monkeypatch.setattr(agent, "_read_text", lambda path: "")
    assert agent.process_start(os.getpid()) is None


def test_process_start_counts_fields_after_the_last_paren(monkeypatch):
    # A command name may hold ")" and spaces; field n reads n here.
    fields = " ".join(str(n) for n in range(3, 53))
    monkeypatch.setattr(agent, "_read_text", lambda path: "42 (a) b) c) %s\n" % fields)
    assert agent.process_start(42) == 22


def _signal_job(tmp_path, monkeypatch, error, own=None):
    """cmd_signal a job whose group answers killpg with `error`; True if its marker is kept."""
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    job_dir = tmp_path / "home" / "jobs" / "job-e"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "pid").write_text(str(os.getpid()))
    (job_dir / "status.json").write_text(json.dumps({"state": "running"}))
    agent.write_running_marker("job-e", os.getpid())

    def killpg(pgid, sig):
        raise error

    monkeypatch.setattr(agent.os, "killpg", killpg)
    monkeypatch.setattr(agent, "_group_has_own_process", lambda pgid: own)
    assert agent.cmd_signal([str(job_dir)], signal.SIGSTOP, "pause") == 1
    return os.path.exists(agent._marker_path("job-e"))


def test_signal_survives_eperm(tmp_path, monkeypatch):
    # macOS answers EPERM for a group of unreaped zombies: nothing to signal.
    # Its agent reaps them and clears the marker.
    assert _signal_job(tmp_path, monkeypatch, PermissionError(1, "Operation not permitted"), own=True)
    # ps unavailable: keep it, as before.
    assert _signal_job(tmp_path, monkeypatch, PermissionError(1, "Operation not permitted"), own=None)


def test_signal_eperm_from_another_users_group_clears_the_marker(tmp_path, monkeypatch):
    # macOS has no start-time fence: a pgid reused by another user's group
    # answers EPERM too, and must not keep the worker busy forever.
    assert not _signal_job(tmp_path, monkeypatch, PermissionError(1, "Operation not permitted"), own=False)


def test_signal_esrch_clears_the_marker(tmp_path, monkeypatch):
    assert not _signal_job(tmp_path, monkeypatch, ProcessLookupError(3, "No such process"), own=True)


def test_group_alive_on_eperm_asks_whose_group_it_is(monkeypatch):
    def killpg(pgid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(agent.os, "killpg", killpg)
    for own, alive in ((True, True), (None, True), (False, False)):
        monkeypatch.setattr(agent, "_group_has_own_process", lambda pgid, own=own: own)
        assert agent._group_alive(4242) is alive


def test_group_has_own_process_reads_ps(monkeypatch):
    uid = os.getuid()
    rows = {"ps": "  4242   %d\n  4242   %d\n  777   %d\n" % (uid + 1, uid + 2, uid)}
    monkeypatch.setattr(agent, "_run", lambda cmd, **k: rows["ps"])
    assert agent._group_has_own_process(4242) is False
    assert agent._group_has_own_process(777) is True
    rows["ps"] = ""
    assert agent._group_has_own_process(4242) is None


def _fake_proc(monkeypatch, stats):
    """Pretend to be Linux with /proc/<pid>/stat as given ({pid: (state, pgrp)})."""
    monkeypatch.setattr(agent, "os_family", lambda: "linux")
    real_listdir = os.listdir
    monkeypatch.setattr(agent.os, "listdir",
                        lambda path: [str(p) for p in stats] + ["self"] if path == "/proc" else real_listdir(path))
    monkeypatch.setattr(agent, "_read_text", lambda path: "%s (a) b) %s 1 %s 0 0" % (
        path.split("/")[2], *stats[int(path.split("/")[2])]) if path.split("/")[2].isdigit() else "")


def test_group_of_zombies_is_gone_on_linux(monkeypatch):
    # Linux answers killpg(pgid, 0) for zombies, and a PID 1 that doesn't reap
    # (a container without an init) leaves them there.
    _fake_proc(monkeypatch, {10: ("Z", 10), 11: ("Z", 10), 12: ("S", 99)})
    monkeypatch.setattr(agent.os, "killpg", lambda pgid, sig: None)
    assert agent._group_has_live_member(10) is False
    assert agent._group_alive(10) is False
    assert agent._group_has_live_member(99) is True
    assert agent._group_alive(99) is True
    assert agent._group_has_live_member(5) is None
    assert agent._group_alive(5) is True


def test_terminate_stops_waiting_for_a_group_of_zombies(monkeypatch):
    _fake_proc(monkeypatch, {10: ("Z", 10)})
    sent = []
    monkeypatch.setattr(agent.os, "killpg", lambda pgid, sig: sent.append(sig))
    started = time.time()
    agent._terminate_group(10, grace=5)
    assert time.time() - started < 1
    assert signal.SIGKILL not in sent


def test_terminate_wakes_a_paused_group(tmp_path):
    # A stopped process only acts on SIGTERM once it runs again: without the
    # SIGCONT it is SIGKILLed after the grace and never gets to clean up.
    flag = tmp_path / "terminated"
    code = ("import signal, sys, time, pathlib\n"
            "def bye(*a):\n    pathlib.Path(%r).write_text('x'); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, bye)\n"
            "print('ready', flush=True)\n"
            "time.sleep(30)\n" % str(flag))
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, start_new_session=True)
    try:
        assert proc.stdout.readline().strip() == b"ready"
        os.killpg(proc.pid, signal.SIGSTOP)
        reaper = threading.Thread(target=proc.wait, daemon=True)
        reaper.start()
        agent._terminate_group(proc.pid, grace=5)
        reaper.join(5)
        assert flag.exists()
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_cmd_running(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "home"))
    proc = subprocess.Popen(["/bin/sh", "-c", "exec sleep 30"], start_new_session=True)
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
    assert caps["machine"] in {"arm64", "x86_64", "aarch64"}
    assert caps["cores"] > 0
    assert caps["ram_gb"] > 0
    assert caps["disk_free_gb"] > 0
    assert caps["bench"] > 0
    assert isinstance(caps["tags"], list)
    assert caps["machine"] in caps["tags"]


def test_capability_tags_include_platform():
    tags = agent.capability_tags("14.5", family="macos")
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


def test_collect_caps_carries_what_decide_preemption_reads(monkeypatch, tmp_path):
    # The dispatcher holds a worker a job would pause on at once, so caps say
    # everything decide_preemption checks. The agent's own process group is
    # seconds old and reads high, so it doesn't count as other CPU.
    fake_linux(monkeypatch, tmp_path, {"idle_seconds": 900, "power": "ac", "battery_pct": 37})
    rows = "%d 90.0\n4242 4.5\n4243 2.0\n" % os.getpgrp()
    monkeypatch.setattr(agent, "_run", lambda argv, **k: rows if argv[0] == "ps" else "")
    monkeypatch.setattr(agent, "running_jobs", lambda: ["j7", "j8"])
    (tmp_path / "policy.json").write_text(json.dumps(
        {"max_other_cpu_pct": 25, "pause_on_low_battery_pct": 30, "paused": True}))
    caps = agent.collect_caps()
    assert caps["battery_pct"] == 37
    assert caps["pause_on_low_battery_pct"] == 30
    assert caps["other_cpu_pct"] == 6.5
    assert caps["max_other_cpu_pct"] == 25
    assert caps["policy_paused"] is True
    assert caps["busy_jobs"] == ["j7", "j8"]
    assert caps["busy_job"] == "j7"

    (tmp_path / "policy.json").unlink()
    monkeypatch.setattr(agent, "_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ps unasked")))
    monkeypatch.setattr(agent, "running_jobs", lambda: [])
    caps = agent.collect_caps()
    assert caps["other_cpu_pct"] is None
    assert caps["max_other_cpu_pct"] is None
    assert caps["pause_on_low_battery_pct"] == agent.DEFAULT_POLICY["pause_on_low_battery_pct"]
    assert caps["policy_paused"] is False
    assert (caps["busy_jobs"], caps["busy_job"]) == ([], None)


def test_sampler_leaves_out_the_agent_and_the_job(monkeypatch, tmp_path):
    fake_linux(monkeypatch, tmp_path, {"idle_seconds": 900, "power": "ac"})
    policy = {**agent.DEFAULT_POLICY, "max_other_cpu_pct": 25}
    me = os.getpgrp()
    for other, action in ((3.0, None), (30.0, "pause")):
        rows = "%d 80.0\n4242 70.0\n4243 %s\n" % (me, other)
        monkeypatch.setattr(agent, "_run", lambda argv, rows=rows, **k: rows if argv[0] == "ps" else "")
        sample_ = agent.sample_preemption_state(policy, 4242)
        assert sample_["other_cpu_pct"] == other
        assert agent.decide_preemption(sample_, policy, False)[0] == action


def test_other_cpu_is_unknown_without_ps(monkeypatch):
    # A minimal image without procps: no rows is not 0% busy, and an unknown
    # reading holds nothing.
    monkeypatch.setattr(agent, "_run", lambda *a, **k: "")
    assert agent.other_cpu_pct(4242) is None
    policy = {**agent.DEFAULT_POLICY, "max_other_cpu_pct": 25}
    sample_ = {"idle_seconds": 900, "power": "ac", "other_cpu_pct": None}
    assert agent.decide_preemption(sample_, policy, False) == (None, None)


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
    assert agent._wrap(["x"], {}) == ["x"]


def test_wrap_background_qos_opt_in():
    policy = {**agent.DEFAULT_POLICY, "background_qos": True}
    wrapped = agent._wrap(["echo", "hi"], policy=policy)
    assert any("nice" in part for part in wrapped)
    # caffeinate runs beside the job now (see _keep_awake), never in front.
    assert wrapped[-2:] == ["echo", "hi"]
    assert not any("caffeinate" in part for part in wrapped)


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
