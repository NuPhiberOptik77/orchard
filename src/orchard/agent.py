"""Orchard worker agent.

Single file, stdlib only, Python >= 3.9, so it runs under stock /usr/bin/python3
on any Mac before any bootstrap has happened.

Usage:
  python3 agent.py status [bench_python]
  python3 agent.py exec <job.json>
  python3 agent.py jobstatus <job_dir>
  python3 agent.py pause <job_dir>
  python3 agent.py resume <job_dir>
  python3 agent.py kill <job_dir>
  python3 agent.py gc [max_age_days] [min_free_gb]
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

TERMINAL_STATES = ("succeeded", "failed", "timed_out", "killed")

ORCHARD_HOME = os.path.expanduser("~/.orchard")


def _running_dir():
    return os.path.join(ORCHARD_HOME, "state", "running")


def _marker_path(job_id):
    return os.path.join(_running_dir(), "%s.json" % job_id)


def _group_alive(pid):
    try:
        os.killpg(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def write_running_marker(job_id, pgid):
    """Record that a job's process group is alive (best effort, atomic)."""
    try:
        os.makedirs(_running_dir(), exist_ok=True)
        tmp = _marker_path(job_id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"job_id": job_id, "pgid": pgid, "started_at": time.time()}, f)
        os.replace(tmp, _marker_path(job_id))
    except OSError:
        pass


def clear_running_marker(job_id):
    try:
        os.unlink(_marker_path(job_id))
    except OSError:
        pass


def running_jobs():
    """Job ids whose process groups are alive; stale markers are pruned."""
    try:
        names = os.listdir(_running_dir())
    except OSError:
        return []
    alive = []
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(_running_dir(), name)
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = None
        pid = (data or {}).get("pgid")
        if pid and _group_alive(pid):
            alive.append(data.get("job_id") or name[:-5])
        else:
            try:
                os.unlink(path)
            except OSError:
                pass
    return sorted(alive)

DEFAULT_POLICY = {
    "paused": False,
    "pause_if_user_active": True,
    "idle_threshold_s": 180,
    "pause_on_battery": True,
    "thermal_pause_below": 50,
    "thermal_resume_at": 60,
    "max_other_cpu_pct": None,
    "background_qos": False,
    "pause_on_low_battery_pct": 20,
}
RESUME_MARGIN_S = 30


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout
        )
        return out.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def _sysctl(name):
    return _run(["sysctl", "-n", name]).strip()


def parse_hid_idle(ioreg_text):
    m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', ioreg_text)
    if not m:
        return None
    return int(m.group(1)) / 1e9


def hid_idle_seconds():
    return parse_hid_idle(_run(["ioreg", "-c", "IOHIDSystem"]))


def parse_power(pmset_batt_text):
    if "AC Power" in pmset_batt_text:
        return "ac"
    if "Battery Power" in pmset_batt_text:
        return "battery"
    return "unknown"


def battery_state():
    return parse_power(_run(["pmset", "-g", "batt"]))


def parse_battery_pct(pmset_batt_text):
    m = re.search(r"(\d+)%", pmset_batt_text)
    return int(m.group(1)) if m else None


def battery_pct():
    return parse_battery_pct(_run(["pmset", "-g", "batt"]))


def parse_thermal_limit(pmset_therm_text):
    m = re.search(r"CPU_Scheduler_Limit\s*=\s*(\d+)", pmset_therm_text)
    return int(m.group(1)) if m else None


def thermal_limit():
    return parse_thermal_limit(_run(["pmset", "-g", "therm"]))


def disk_free_gb(path):
    st = os.statvfs(path)
    return round(st.f_bavail * st.f_frsize / 1e9, 2)


BENCH_CODE = (
    "import time\n"
    "n = 300000\n"
    "t = time.perf_counter()\n"
    "x = 0.0\n"
    "for i in range(n):\n"
    "    x = x * 1.0000001 + 0.5\n"
    "dt = time.perf_counter() - t\n"
    "print(round(n / dt))\n"
)


def benchmark(python_path=None, iterations=300000):
    if python_path:
        out = _run([python_path, "-c", BENCH_CODE], timeout=60)
        try:
            value = float(out.strip())
            if value > 0:
                return round(value)
        except ValueError:
            pass
    t0 = time.perf_counter()
    x = 0.0
    for i in range(iterations):
        x = x * 1.0000001 + 0.5
    dt = time.perf_counter() - t0
    return round(iterations / dt, 0)


def other_cpu_pct(own_pgid):
    total = 0.0
    own = 0.0
    for line in _run(["ps", "-axo", "pgid=,pcpu="]).splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pgid, cpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        total += cpu
        if pgid == own_pgid:
            own += cpu
    return round(total - own, 1)


def _xcode_path():
    try:
        proc = subprocess.run(
            ["/usr/bin/xcode-select", "-p"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def capability_tags(os_version=None):
    tags = {os.uname().machine}
    if os_version is None:
        os_version = _run(["sw_vers", "-productVersion"]).strip()
    try:
        tags.add("macos%s" % os_version.split(".")[0])
    except IndexError:
        pass
    for tool in ("swift", "clang", "git", "make"):
        if shutil.which(tool):
            tags.add(tool)
    if os.path.exists(os.path.join(ORCHARD_HOME, "tools", "uv")):
        tags.add("uv")
    path = _xcode_path()
    if path and path.endswith(".app/Contents/Developer"):
        tags.add("xcode")
    elif path:
        tags.add("clt")
    return tags


BENCH_TTL_S = 86400


def cached_bench(python_path, cache_dir=None, ttl=BENCH_TTL_S):
    if not python_path or not os.path.exists(python_path):
        python_path = sys.executable
    key = python_path
    cache_dir = cache_dir or os.path.join(ORCHARD_HOME, "state")
    path = os.path.join(cache_dir, "bench.json")
    now = time.time()
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("python") == key and now - float(data.get("ts", 0)) < ttl:
            return data["value"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    value = benchmark(python_path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        _atomic_write_json(path, {"python": key, "value": value, "ts": now})
    except OSError:
        pass
    return value


def collect_caps(bench_python=None):
    os_version = _run(["sw_vers", "-productVersion"]).strip()
    busy = running_jobs()
    return {
        "hostname": os.uname().nodename,
        "machine": os.uname().machine,
        "os": os_version,
        "tags": sorted(capability_tags(os_version)),
        "python": sys.version.split()[0],
        "cores": int(_sysctl("hw.ncpu") or 0),
        "ram_gb": round(int(_sysctl("hw.memsize") or 0) / 1e9, 2),
        "disk_free_gb": disk_free_gb(os.path.expanduser("~")),
        "idle_seconds": hid_idle_seconds(),
        "power": battery_state(),
        "thermal_limit": thermal_limit(),
        "has_taskpolicy": bool(shutil.which("taskpolicy")),
        "bench": cached_bench(bench_python),
        "bench_python": bench_python or sys.executable,
        "policy_paused": bool(load_policy().get("paused")),
        "busy_jobs": busy,
        "busy_job": busy[0] if busy else None,
    }


def load_policy():
    policy = dict(DEFAULT_POLICY)
    path = os.path.join(ORCHARD_HOME, "policy.json")
    try:
        with open(path) as f:
            policy.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass
    return policy


def sample_preemption_state(policy, own_pgid):
    sample = {
        "idle_seconds": hid_idle_seconds(),
        "power": battery_state(),
        "battery_pct": battery_pct(),
        "thermal_limit": thermal_limit(),
        "other_cpu_pct": None,
    }
    if policy.get("max_other_cpu_pct") is not None:
        sample["other_cpu_pct"] = other_cpu_pct(own_pgid)
    return sample


def decide_preemption(sample, policy, paused):
    if policy.get("paused"):
        if paused:
            return (None, None)
        return ("pause", "paused by owner")
    blockers = []
    idle = sample.get("idle_seconds")
    margin = RESUME_MARGIN_S if paused else 0
    if policy.get("pause_if_user_active") and idle is not None:
        if idle < policy.get("idle_threshold_s", 180) + margin:
            blockers.append("user active (idle %.0fs)" % idle)
    if policy.get("pause_on_battery") and sample.get("power") == "battery":
        blockers.append("on battery")
    low_pct = policy.get("pause_on_low_battery_pct")
    pct = sample.get("battery_pct")
    if low_pct is not None and pct is not None and pct <= low_pct:
        blockers.append("battery low (%d%%)" % pct)
    limit = sample.get("thermal_limit")
    if limit is not None:
        floor = policy.get("thermal_resume_at", 60) if paused else policy.get("thermal_pause_below", 50)
        if limit < floor:
            blockers.append("thermal throttle (limit %s)" % limit)
    max_other = policy.get("max_other_cpu_pct")
    other = sample.get("other_cpu_pct")
    if max_other is not None and other is not None:
        threshold = max_other * 0.8 if paused else max_other
        if other > threshold:
            blockers.append("other CPU %.1f%% > %s%%" % (other, max_other))
    if paused:
        return ("resume", None) if not blockers else (None, None)
    if blockers:
        return ("pause", blockers[0])
    return (None, None)


def _atomic_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_status(job_dir):
    path = os.path.join(job_dir, "status.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _wrap(cmd, policy=None):
    """Wrap the job command. Jobs run at full priority by default; preemption
    (pause-on-user-activity etc.) is what protects the worker. Setting
    `background_qos: true` in policy.json restores nice + background QoS,
    which costs ~2.4x throughput on this hardware when measured on the Pro."""
    if policy is None:
        policy = load_policy()
    parts = []
    caffeinate = shutil.which("caffeinate")
    if caffeinate:
        parts += [caffeinate, "-is"]
    if policy.get("background_qos"):
        nice = shutil.which("nice") or "/usr/bin/nice"
        parts += [nice, "-n", "10"]
        taskpolicy = shutil.which("taskpolicy")
        if taskpolicy:
            parts += [taskpolicy, "-b"]
    return parts + list(cmd)


def _terminate_group(pgid, grace=10.0):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _log_line(log, text):
    log.write("orchard: %s\n" % text)
    log.flush()


def cmd_exec(argv):
    if len(argv) != 1:
        print("usage: agent.py exec <job.json>", file=sys.stderr)
        return 2
    with open(argv[0]) as f:
        job = json.load(f)

    job_dir = job["job_dir"]
    workdir = job["workdir"]
    os.makedirs(os.path.join(workdir, "out"), exist_ok=True)
    ckpt_dir = os.path.join(job_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(job_dir, "log.txt")
    status_path = os.path.join(job_dir, "status.json")
    pid_path = os.path.join(job_dir, "pid")

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
    venv = job.get("venv")
    cmd = list(job["cmd"])
    if venv:
        env["VIRTUAL_ENV"] = venv
        env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
        if cmd and cmd[0] == "python":
            cmd = [os.path.join(venv, "bin", "python")] + cmd[1:]
    env["ORCHARD_JOB_ID"] = str(job.get("id"))
    env["ORCHARD_CKPT_DIR"] = ckpt_dir
    resuming = bool(os.listdir(ckpt_dir))
    if resuming:
        env["ORCHARD_RESUME"] = "1"

    preempt = bool(job.get("preempt"))
    wrapped = _wrap(cmd, load_policy())
    timeout = job.get("timeout_s")

    started = time.time()
    status = {
        "id": job.get("id"),
        "state": "running",
        "cmd": cmd,
        "workdir": workdir,
        "venv": venv,
        "pid": None,
        "pgid": None,
        "resumed": resuming,
        "pause_reason": None,
        "paused_since": None,
        "started_at": started,
        "ended_at": None,
        "elapsed_s": 0,
        "active_s": 0,
        "exit_code": None,
    }
    _atomic_write_json(status_path, status)

    log = open(log_path, "a")
    _log_line(log, "starting job %s (resume=%s)" % (job.get("id"), resuming))
    try:
        proc = subprocess.Popen(
            wrapped, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log.close()
        status.update(state="failed", ended_at=time.time(), exit_code=-1, error=str(exc))
        _atomic_write_json(status_path, status)
        return 1

    pgid = os.getpgid(proc.pid)
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    write_running_marker(str(job.get("id")), pgid)
    status["pid"] = proc.pid
    status["pgid"] = pgid
    _atomic_write_json(status_path, status)

    state = "succeeded"
    exit_code = None
    paused = False
    paused_since = None
    active_s = 0.0
    last_tick = time.time()
    last_poll = 0.0
    while True:
        now = time.time()
        dt = now - last_tick
        last_tick = now
        rc = proc.poll()
        if rc is not None:
            exit_code = rc
            state = "succeeded" if rc == 0 else "failed"
            break
        if not paused:
            active_s += dt
        if timeout and active_s > timeout:
            _log_line(log, "timeout after %.0fs of active time" % active_s)
            _terminate_group(pgid)
            exit_code = proc.wait()
            state = "timed_out"
            break
        if now - last_poll > 5:
            if preempt:
                policy = load_policy()
                sample = sample_preemption_state(policy, pgid)
                action, reason = decide_preemption(sample, policy, paused)
                if action == "pause":
                    try:
                        os.killpg(pgid, signal.SIGSTOP)
                    except ProcessLookupError:
                        time.sleep(0.5)
                        continue
                    paused = True
                    paused_since = now
                    status.update(state="paused", pause_reason=reason, paused_since=paused_since)
                    _log_line(log, "paused: %s" % reason)
                elif action == "resume":
                    try:
                        os.killpg(pgid, signal.SIGCONT)
                    except ProcessLookupError:
                        time.sleep(0.5)
                        continue
                    paused = False
                    paused_since = None
                    status.update(state="running", pause_reason=None, paused_since=None)
                    _log_line(log, "resumed")
            status.update(elapsed_s=round(now - started, 1), active_s=round(active_s, 1))
            _atomic_write_json(status_path, status)
            last_poll = now
        time.sleep(0.5)

    log.close()
    if paused:
        try:
            os.killpg(pgid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    clear_running_marker(str(job.get("id")))
    status.update(
        state=state,
        pause_reason=None,
        paused_since=None,
        ended_at=time.time(),
        elapsed_s=round(time.time() - started, 1),
        active_s=round(active_s, 1),
        exit_code=exit_code,
    )
    _atomic_write_json(status_path, status)
    return 0 if state == "succeeded" else 1


def _read_pid(job_dir):
    pid_path = os.path.join(job_dir, "pid")
    if not os.path.exists(pid_path):
        return None
    with open(pid_path) as f:
        return int(f.read().strip())


def cmd_signal(argv, sig, label, cont_first=False):
    if len(argv) != 1:
        print("usage: agent.py %s <job_dir>" % label, file=sys.stderr)
        return 2
    pid = _read_pid(argv[0])
    if pid is None:
        print("no pid file in %s" % argv[0], file=sys.stderr)
        return 1
    try:
        pgid = os.getpgid(pid)
        if cont_first:
            os.killpg(pgid, signal.SIGCONT)
        os.killpg(pgid, sig)
    except ProcessLookupError:
        clear_running_marker(os.path.basename(argv[0].rstrip("/")))
        print("process group not running", file=sys.stderr)
        return 1
    if sig in (signal.SIGTERM, signal.SIGKILL):
        clear_running_marker(os.path.basename(argv[0].rstrip("/")))
    return 0


def cmd_running(argv):
    ids = running_jobs()
    if argv:
        print("yes" if argv[0] in ids else "no")
    else:
        print(",".join(ids))
    return 0


def cmd_jobstatus(argv):
    if len(argv) != 1:
        print("usage: agent.py jobstatus <job_dir>", file=sys.stderr)
        return 2
    status = read_status(argv[0])
    if status is None:
        print("no status.json in %s" % argv[0], file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2))
    return 0


def select_garbage(entries, now, max_age_days, min_free_gb, free_gb):
    picked = []
    leftover = []
    for entry in entries:
        if entry.get("protected"):
            continue
        if now - entry["mtime"] > max_age_days * 86400:
            picked.append(entry)
        else:
            leftover.append(entry)
    freed_gb = sum(e["size"] for e in picked) / 1e9
    if free_gb + freed_gb < min_free_gb:
        for entry in sorted(leftover, key=lambda e: e["mtime"]):
            if free_gb + freed_gb >= min_free_gb:
                break
            picked.append(entry)
            freed_gb += entry["size"] / 1e9
    return picked


def _dir_size(path):
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _gc_entries(now):
    entries = []
    live_workdirs = []
    inbox = os.path.join(ORCHARD_HOME, "inbox")
    if os.path.isdir(inbox):
        for name in os.listdir(inbox):
            path = os.path.join(inbox, name)
            if os.path.isfile(path):
                st = os.stat(path)
                entries.append({"path": path, "mtime": st.st_mtime, "size": st.st_size, "protected": False})
    jobs_dir = os.path.join(ORCHARD_HOME, "jobs")
    if os.path.isdir(jobs_dir):
        for name in os.listdir(jobs_dir):
            path = os.path.join(jobs_dir, name)
            if not os.path.isdir(path):
                continue
            status = read_status(path) or {}
            protected = status.get("state") in ("running", "paused")
            if protected and status.get("workdir"):
                live_workdirs.append(status["workdir"])
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = now
            status_path = os.path.join(path, "status.json")
            if os.path.exists(status_path):
                mtime = max(mtime, os.path.getmtime(status_path))
            entries.append({"path": path, "mtime": mtime, "size": _dir_size(path), "protected": protected})
    builds_dir = os.path.join(ORCHARD_HOME, "builds")
    if os.path.isdir(builds_dir):
        for name in os.listdir(builds_dir):
            path = os.path.join(builds_dir, name)
            if not os.path.isdir(path):
                continue
            protected = any(w.startswith(path + os.sep) for w in live_workdirs)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = now
            entries.append({"path": path, "mtime": mtime, "size": _dir_size(path), "protected": protected})
    return entries


def cmd_gc(argv):
    max_age = float(argv[0]) if len(argv) > 0 else 14.0
    min_free = float(argv[1]) if len(argv) > 1 else 3.0
    os.makedirs(ORCHARD_HOME, exist_ok=True)
    now = time.time()
    picked = select_garbage(_gc_entries(now), now, max_age, min_free, disk_free_gb(ORCHARD_HOME))
    freed = 0
    for entry in picked:
        try:
            if os.path.isdir(entry["path"]):
                shutil.rmtree(entry["path"])
            else:
                os.remove(entry["path"])
            freed += entry["size"]
        except OSError:
            pass
    print(json.dumps({
        "removed": len(picked),
        "freed_mb": round(freed / 1e6, 1),
        "free_gb": disk_free_gb(ORCHARD_HOME),
    }))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "status":
        print(json.dumps(collect_caps(rest[0] if rest else None), indent=2))
        return 0
    if cmd == "exec":
        return cmd_exec(rest)
    if cmd == "jobstatus":
        return cmd_jobstatus(rest)
    if cmd == "pause":
        return cmd_signal(rest, signal.SIGSTOP, "pause")
    if cmd == "resume":
        return cmd_signal(rest, signal.SIGCONT, "resume")
    if cmd == "kill":
        return cmd_signal(rest, signal.SIGTERM, "kill", cont_first=True)
    if cmd == "gc":
        return cmd_gc(rest)
    if cmd == "running":
        return cmd_running(rest)
    print("unknown command: %s" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
