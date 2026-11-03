"""Orchard worker agent.

Single file, stdlib only, Python >= 3.9, so it runs under stock /usr/bin/python3
on any Mac before any bootstrap has happened. It also runs on Linux (including
WSL), where idle, power and host disk come from a status file the host writes;
see read_host_status.

Usage:
  python3 agent.py status [bench_python]
  python3 agent.py exec <job.json>
  python3 agent.py jobstatus <job_dir>
  python3 agent.py pause <job_dir>
  python3 agent.py resume <job_dir>
  python3 agent.py kill <job_dir>
  python3 agent.py terminate <job_dir>
  python3 agent.py settle <job_dir>
  python3 agent.py wrap <exit.json> <attempt> -- <cmd...>
  python3 agent.py gc [max_age_days] [min_free_gb]
  python3 agent.py replica-state <key> [manifest]
  python3 agent.py replica-apply <key> [token]
  python3 agent.py replica-begin <key> <job.json> <digest>
  python3 agent.py replica-release <key> <job>
  python3 agent.py replica-meta <key> <json>
  python3 agent.py replica-pack <key>   (paths as JSON on stdin, tar.gz on stdout)
  python3 agent.py replica-down <key>
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

TERMINAL_STATES = ("succeeded", "failed", "timed_out", "killed")

ORCHARD_HOME = os.path.expanduser("~/.orchard")
HOST_STATUS_PATH = os.path.join(ORCHARD_HOME, "host-status.json")
HOST_STATUS_MAX_AGE_S = 60
HOST_POWER_STATES = ("ac", "battery", "unknown")
POLL_S = 5  # how often a running job checks for preemption and rewrites status.json
KILL_GRACE_S = 60  # how long a job asked to end may ignore SIGTERM before it gets SIGKILL


def _running_dir():
    return os.path.join(ORCHARD_HOME, "state", "running")


def _marker_path(job_id):
    return os.path.join(_running_dir(), "%s.json" % job_id)


def _group_alive(pid):
    try:
        os.killpg(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS's answer for a group of unreaped zombies, and everywhere for a
        # group of another user's processes: a number reused since the marker
        # was written. Only a member of our own still counts.
        return _group_has_own_process(pid) is not False
    except OSError:
        return True
    return _group_has_live_member(pid) is not False


def _group_has_live_member(pgid):
    """Whether pgid has a member that isn't a zombie; None when /proc can't say.

    Linux answers killpg(pgid, 0) for a group of zombies too, and under a PID 1
    that doesn't reap (a container without an init) they stay. The state is
    the first field after the command name's last ")" and the group the third.
    """
    if os_family() != "linux":
        return None
    try:
        pids = [name for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return None
    seen = False
    for name in pids:
        try:
            fields = _read_text("/proc/%s/stat" % name).rsplit(")", 1)[1].split()
            if int(fields[2]) != int(pgid):
                continue
        except (IndexError, ValueError):
            continue
        seen = True
        if fields[0] not in ("Z", "X") or _has_live_thread(name):
            return True
    return False if seen else None


def _has_live_thread(pid):
    """Whether a process listed as a zombie still runs other threads.

    /proc/<pid>/stat describes the main thread, which reads Z once it has
    exited (pthread_exit) even while the process's other threads run on.
    """
    try:
        tids = os.listdir("/proc/%s/task" % pid)
    except OSError:
        return False
    for tid in tids:
        try:
            state = _read_text("/proc/%s/task/%s/stat" % (pid, tid)).rsplit(")", 1)[1].split()[0]
        except IndexError:
            continue
        if state not in ("Z", "X"):
            return True
    return False


def _group_has_own_process(pgid):
    """Whether one of pgid's members runs as us; None when ps can't say."""
    found = False
    uid = os.getuid()
    for line in _run(["ps", "-axo", "pgid=,uid="]).splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        found = True
        try:
            if int(parts[0]) == int(pgid) and int(parts[1]) == uid:
                return True
        except ValueError:
            continue
    return False if found else None


def boot_id():
    """An id that changes at every boot, or None if unknown."""
    if os_family() == "linux":
        value = _read_text("/proc/sys/kernel/random/boot_id").strip()
    else:
        value = _sysctl("kern.bootsessionuuid")
    return value or None


def process_start(pid):
    """When pid started, in clock ticks since the kernel booted, or None.

    `wsl --terminate` restarts a distro without a new boot_id, and pids are
    handed out again, so a pid alone can name another process afterwards.
    Linux only; the field after the command name's last ")" is the 3rd, so
    starttime, the 22nd, is the 20th after it.
    """
    try:
        return int(_read_text("/proc/%d/stat" % int(pid)).rsplit(")", 1)[1].split()[19])
    except (IndexError, TypeError, ValueError):
        return None


def _same_process(pid, started):
    """False only when pid is known to have started at another time than `started`."""
    if started is None:
        return True
    now = process_start(pid)
    return now is None or now == started


def write_running_marker(job_id, pgid):
    """Record that a job's process group is alive (best effort, atomic)."""
    try:
        os.makedirs(_running_dir(), exist_ok=True)
        tmp = _marker_path(job_id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"job_id": job_id, "pgid": pgid, "started_at": time.time(), "boot_id": boot_id(),
                       "pid_start": process_start(pgid)}, f)
        os.replace(tmp, _marker_path(job_id))
    except OSError:
        pass


def clear_running_marker(job_id):
    try:
        os.unlink(_marker_path(job_id))
    except OSError:
        pass


def running_jobs():
    """Job ids whose process groups are alive; stale markers are pruned.

    A marker from an earlier boot is stale even if its pgid is in use again:
    after a restart that number can belong to an unrelated process. So is one
    whose group leader started at another time (a WSL distro restarted).
    """
    try:
        names = os.listdir(_running_dir())
    except OSError:
        return []
    alive = []
    boot = boot_id()
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
        marker_boot = (data or {}).get("boot_id")
        same_boot = not (boot and marker_boot) or marker_boot == boot
        if pid and same_boot and _same_process(pid, data.get("pid_start")) and _group_alive(pid):
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


def _read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def os_family():
    """`macos` or `linux`; anything else is reported as-is and matches no job by default."""
    sysname = os.uname().sysname
    return {"Darwin": "macos", "Linux": "linux"}.get(sysname, sysname.lower())


def parse_meminfo_total_gb(meminfo_text):
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", meminfo_text, re.M)
    return round(int(m.group(1)) * 1024 / 1e9, 2) if m else 0.0


def parse_os_release(os_release_text):
    fields = {}
    for line in os_release_text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip().strip("\"'")
    return (fields.get("ID") or "linux") + fields.get("VERSION_ID", "")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_host_status(path=None, now=None, max_age=HOST_STATUS_MAX_AGE_S):
    """Idle time, power and free disk reported by the host OS.

    On Linux nothing in the guest can see the owner's keyboard or the charger,
    so the host writes them to HOST_STATUS_PATH (contrib/windows-wsl has the
    Windows writer). Fails closed: a missing, stale or malformed file reads as
    "user active, power unknown" and `host_status` names the problem, which
    holds new jobs and pauses running ones. Age comes from the file's mtime on
    this side, so the host's clock never matters.
    """
    path = path or HOST_STATUS_PATH
    now = time.time() if now is None else now
    closed = {"idle_seconds": 0.0, "power": "unknown", "battery_pct": None, "disk_free_gb": None}
    try:
        age = now - os.path.getmtime(path)
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return dict(closed, host_status="missing")
    except (OSError, ValueError):
        return dict(closed, host_status="unreadable")
    if abs(age) > max_age:
        return dict(closed, host_status="stale (%.0fs old)" % age)
    if not isinstance(data, dict):
        return dict(closed, host_status="unreadable")
    idle = data.get("idle_seconds")
    power = data.get("power")
    disk = data.get("disk_free_gb")
    if not (_finite_number(idle) and idle >= 0 and power in HOST_POWER_STATES
            and _finite_number(disk) and disk >= 0):
        return dict(closed, host_status="incomplete")
    pct = data.get("battery_pct")
    return {
        "idle_seconds": float(idle),
        "power": power,
        "battery_pct": int(pct) if _finite_number(pct) else None,
        "disk_free_gb": round(float(disk), 2),
        "host_status": "ok",
    }


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


def worker_disk_free_gb(path, host=None):
    """Free space for jobs. Under WSL the guest disk is a file on the host's C:,
    so the host's free space caps what statvfs reports."""
    free = disk_free_gb(path)
    host_free = (host or {}).get("disk_free_gb")
    return min(free, host_free) if host_free is not None else free


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


def other_cpu_pct(job_pgid=None):
    """CPU used by everything but this agent and the job, in percent; None if unknown.

    ps reports each process's CPU time over its lifetime, so a process seconds
    old can read 50-100% (on Linux; less on macOS, but not 0). This agent, the
    shell that started it and ps itself are all that young, and all in the
    agent's process group, so that group is left out with the job's.
    """
    mine = {os.getpgrp(), job_pgid}
    total = 0.0
    rows = 0
    for line in _run(["ps", "-axo", "pgid=,pcpu="]).splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pgid, cpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        rows += 1
        if pgid not in mine:
            total += cpu
    # No ps (a minimal image without procps) prints nothing; 0% would read idle.
    return round(total, 1) if rows else None


def _xcode_path():
    try:
        proc = subprocess.run(
            ["/usr/bin/xcode-select", "-p"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def capability_tags(os_version=None, family=None):
    family = family or os_family()
    tags = {os.uname().machine, "os:%s" % family}
    if family == "macos":
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
    if family == "macos":
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


def collect_caps(bench_python=None, bench=True):
    family = os_family()
    home = os.path.expanduser("~")
    policy = load_policy()
    if family == "macos":
        os_version = _run(["sw_vers", "-productVersion"]).strip()
        cores = int(_sysctl("hw.ncpu") or 0)
        ram_gb = round(int(_sysctl("hw.memsize") or 0) / 1e9, 2)
        disk = disk_free_gb(home)
        idle, power, thermal, host_status = hid_idle_seconds(), battery_state(), thermal_limit(), None
        pct = battery_pct()
    else:
        host = read_host_status()
        os_version = parse_os_release(_read_text("/etc/os-release"))
        cores = os.cpu_count() or 0
        ram_gb = parse_meminfo_total_gb(_read_text("/proc/meminfo"))
        disk = worker_disk_free_gb(home, host)
        idle, power, thermal, host_status = host["idle_seconds"], host["power"], None, host["host_status"]
        pct = host["battery_pct"]
    # The dispatcher holds a worker that a job would pause on at once, so it
    # gets what decide_preemption reads too.
    max_other = policy.get("max_other_cpu_pct")
    busy = running_jobs()
    return {
        "hostname": os.uname().nodename,
        "machine": os.uname().machine,
        "os": os_version,
        "tags": sorted(capability_tags(os_version, family)),
        "python": sys.version.split()[0],
        "cores": cores,
        "ram_gb": ram_gb,
        "disk_free_gb": disk,
        "idle_seconds": idle,
        "power": power,
        "battery_pct": pct,
        "pause_on_low_battery_pct": policy.get("pause_on_low_battery_pct"),
        "other_cpu_pct": other_cpu_pct() if max_other is not None else None,
        "max_other_cpu_pct": max_other,
        "thermal_limit": thermal,
        "host_status": host_status,
        "has_taskpolicy": bool(shutil.which("taskpolicy")),
        "bench": cached_bench(bench_python) if bench else None,
        "bench_python": bench_python or sys.executable,
        "policy_paused": bool(policy.get("paused")),
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
    if os_family() == "macos":
        sample = {
            "idle_seconds": hid_idle_seconds(),
            "power": battery_state(),
            "battery_pct": battery_pct(),
            "thermal_limit": thermal_limit(),
            "other_cpu_pct": None,
        }
    else:
        host = read_host_status()
        sample = {
            "idle_seconds": host["idle_seconds"],
            "power": host["power"],
            "battery_pct": host["battery_pct"],
            "thermal_limit": None,
            "other_cpu_pct": None,
            "host_status": host["host_status"],
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
    host = sample.get("host_status")
    if host is not None and host != "ok":
        blockers.append("host status %s" % host)
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
    if policy.get("background_qos"):
        nice = shutil.which("nice") or "/usr/bin/nice"
        parts += [nice, "-n", "10"]
        taskpolicy = shutil.which("taskpolicy")
        if taskpolicy:
            parts += [taskpolicy, "-b"]
    return parts + list(cmd)


def _keep_awake(pid):
    """Keep a Mac from sleeping until pid exits (`caffeinate -is -w`).

    It runs beside the job, not in its process group, so `wrap` leads the
    group (see _stop_group) and a job's `kill 0` can't end it early.
    """
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        return None
    try:
        return subprocess.Popen([caffeinate, "-is", "-w", str(pid)], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return None


def _started_path(job_dir):
    """Where `wrap` writes its command's pid once the command has started."""
    return os.path.join(job_dir, "command.pid")


def _kill_request_path(job_dir):
    """Where `kill` and `terminate` note that the job was asked to end."""
    return os.path.join(job_dir, "kill-requested")


def _stop_group(pgid, leader, started):
    """Pause a job's process group, except its leader, `wrap`, once the command runs.

    wrap only waits for the command, so letting it run costs nothing. A
    command killed while paused (SIGKILL, the OOM killer) is then reaped and
    recorded at once, and the job ends instead of reading as paused until it
    resumes.

    Until wrap has started the command it stays stopped with the rest:
    let go, it would start the command after the stop and the job would run
    on unpaused. wrap writes `started` right after the command starts, and a
    stopped wrap can't write it, so checking for it after the stop is safe.

    macOS can miss a process that is forking or execing when the group is
    stopped, and it then runs on unpaused, so the group is stopped twice.
    A kill that lands between the two stops would leave the job frozen in
    its SIGTERM handler, so a job asked to end is let go again.
    Raises what killpg raises.
    """
    os.killpg(pgid, signal.SIGSTOP)
    time.sleep(0.05)
    os.killpg(pgid, signal.SIGSTOP)
    if os.path.exists(_kill_request_path(os.path.dirname(started))):
        os.killpg(pgid, signal.SIGCONT)
        return
    if not os.path.exists(started):
        return
    try:
        os.kill(leader, signal.SIGCONT)
    except OSError:
        pass


def _terminate_group(pgid, grace=10.0):
    # macOS answers EPERM, not ESRCH, for a group whose members are all
    # unreaped zombies, so both mean nothing is left to stop.
    gone = (ProcessLookupError, PermissionError)
    try:
        os.killpg(pgid, signal.SIGTERM)
        os.killpg(pgid, signal.SIGCONT)  # a paused member gets SIGTERM only once it runs
    except gone:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except gone:
            return
        if _group_has_live_member(pgid) is False:
            return
        time.sleep(0.25)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except gone:
        pass


def _close_log(log):
    # Closing flushes, which a full disk can fail; the job is recorded anyway.
    try:
        log.close()
    except OSError:
        pass


def _log_line(log, text):
    # A full disk must not kill the agent before it signals or records the job.
    try:
        log.write("orchard: %s\n" % text)
        log.flush()
    except OSError:
        pass


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
    exit_path = os.path.join(job_dir, "exit.json")
    started_path = _started_path(job_dir)
    kill_path = _kill_request_path(job_dir)

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
    for extra in reversed(job.get("path_prepend") or []):
        env["PATH"] = str(extra) + os.pathsep + env.get("PATH", "")
    venv = job.get("venv")
    cmd = list(job["cmd"])
    if venv:
        env["VIRTUAL_ENV"] = venv
        env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
        if cmd and cmd[0] == "python":
            cmd = [os.path.join(venv, "bin", "python")] + cmd[1:]
    env["ORCHARD_JOB_ID"] = str(job.get("id"))
    attempt = job.get("attempt")
    if attempt is not None:
        env["ORCHARD_ATTEMPT"] = str(attempt)
    env["ORCHARD_CKPT_DIR"] = ckpt_dir
    resuming = bool(os.listdir(ckpt_dir))
    if resuming:
        env["ORCHARD_RESUME"] = "1"

    preempt = bool(job.get("preempt"))
    # The job records its own start and exit (see cmd_wrap), so drop an
    # earlier attempt's. Its pid goes too: until this attempt's is written,
    # a kill or pause must find no process rather than the last attempt's.
    # So does a request to end the last attempt.
    for path in (exit_path, started_path, pid_path, kill_path):
        try:
            os.unlink(path)
        except OSError:
            pass
    recorded = [sys.executable, "-I", os.path.abspath(__file__), "wrap", exit_path,
                "" if attempt is None else str(attempt), "--"] + cmd
    wrapped = _wrap(recorded, load_policy())
    timeout = job.get("timeout_s")

    started = time.time()
    status = {
        "id": job.get("id"),
        "attempt": attempt,
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
        # This agent clears and reads the kill-requested file, so settle can
        # say whether a kill reached the run. Older agents' runs lack it.
        "kill_marker": True,
        # What really ran it, for receipts.
        "machine": os.uname().machine,
        "os_family": os_family(),
        # How busy the machine was at the start, for consensus.
        "load_per_core": round(os.getloadavg()[0] / (os.cpu_count() or 1), 2),
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
        _close_log(log)
        status.update(state="failed", ended_at=time.time(), exit_code=-1, error=str(exc))
        _atomic_write_json(status_path, status)
        return 1

    _keep_awake(proc.pid)
    pgid = os.getpgid(proc.pid)
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    write_running_marker(str(job.get("id")), pgid)
    status["pid"] = proc.pid
    status["pgid"] = pgid
    status["boot_id"] = boot_id()
    status["pid_start"] = process_start(proc.pid)
    _atomic_write_json(status_path, status)

    state = "succeeded"
    exit_code = None
    paused = False
    paused_since = None
    active_s = 0.0
    last_tick = time.time()
    last_poll = 0.0
    kill_requested = None
    hard_killed = False
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
        # `kill` sends SIGCONT, which ends a pause behind this loop's back. A
        # job asked to end isn't paused again, and gets SIGKILL if it is
        # still running KILL_GRACE_S later.
        if kill_requested is None and os.path.exists(kill_path):
            kill_requested = now
            _log_line(log, "asked to end")
            if paused:
                try:
                    os.killpg(pgid, signal.SIGCONT)
                except OSError:
                    pass
                paused = False
                paused_since = None
                status.update(state="running", pause_reason=None, paused_since=None)
        if kill_requested is not None and not hard_killed and now - kill_requested > KILL_GRACE_S:
            _log_line(log, "still running %.0fs after it was asked to end; killing" % (now - kill_requested))
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
            hard_killed = True
        if now - last_poll > POLL_S:
            if preempt and kill_requested is None:
                policy = load_policy()
                sample = sample_preemption_state(policy, pgid)
                action, reason = decide_preemption(sample, policy, paused)
                # EPERM is macOS's answer for an all-zombie group, as in
                # _terminate_group: the job has ended, so poll it again.
                if action == "pause":
                    try:
                        _stop_group(pgid, proc.pid, started_path)
                    except (ProcessLookupError, PermissionError):
                        time.sleep(0.5)
                        continue
                    paused = True
                    paused_since = now
                    status.update(state="paused", pause_reason=reason, paused_since=paused_since)
                    _log_line(log, "paused: %s" % reason)
                elif action == "resume":
                    try:
                        os.killpg(pgid, signal.SIGCONT)
                    except (ProcessLookupError, PermissionError):
                        time.sleep(0.5)
                        continue
                    paused = False
                    paused_since = None
                    status.update(state="running", pause_reason=None, paused_since=None)
                    _log_line(log, "resumed")
            status.update(elapsed_s=round(now - started, 1), active_s=round(active_s, 1))
            try:
                _atomic_write_json(status_path, status)
            except OSError:
                pass  # a full disk must not kill the agent; the next poll tries again
            last_poll = now
        time.sleep(0.5)

    _close_log(log)
    # What's left of the group is ended, not left behind: a command that
    # outlived wrap (wrap SIGKILLed), or members still stopped by a pause.
    if _group_alive(pgid):
        _terminate_group(pgid)
    status.update(
        state=state,
        pause_reason=None,
        paused_since=None,
        ended_at=time.time(),
        elapsed_s=round(time.time() - started, 1),
        active_s=round(active_s, 1),
        exit_code=exit_code,
        # Whether `kill` or `terminate` reached this run. A preempted run whose
        # kill reply was lost is run again only if it was; one that ended by
        # itself first is finished as it ended.
        kill_requested=os.path.exists(kill_path),
    )
    # The final status goes first: once the marker is gone the dispatcher
    # can start the next attempt, which must not find this one still running.
    try:
        _atomic_write_json(status_path, status)
    finally:
        clear_running_marker(str(job.get("id")))
    return 0 if state == "succeeded" else 1


def cmd_wrap(argv):
    """Run a job's command, then record how it ended in exit.json.

    The agent runs every job through this, so a job that ends after its agent
    died (a crash, a full disk) still leaves its exit code, and `settle` can
    finish it instead of the dispatcher running it again. It shrugs off the
    signals the agent sends the job's process group, so it outlives the
    command, and then exits the way the command did.

    A signal it shrugged off goes in the record too. A shutdown or a restart
    signals every process, the agent included, and a command it stopped
    didn't end by itself, so `settle` leaves that run to be run again.
    """
    if len(argv) < 4 or argv[2] != "--":
        print("usage: agent.py wrap <exit.json> <attempt> -- <cmd...>", file=sys.stderr)
        return 2
    exit_path, attempt, cmd = argv[0], argv[1], argv[3:]
    got = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        # A handler, not SIG_IGN, so the command gets the default back on
        # exec. One ignored already (nohup) stays ignored for the command.
        if signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, lambda signum, frame: got.append(signum))
    rc = _call_command(cmd, got, _started_path(os.path.dirname(exit_path)))
    record = {"exit_code": rc, "attempt": int(attempt) if attempt.isdigit() else None, "ended_at": time.time()}
    if got:
        record["signal"] = got[0]
    try:
        _atomic_write_json(exit_path, record)
    except OSError:
        pass
    if rc < 0:
        try:
            signal.signal(-rc, signal.SIG_DFL)
        except (OSError, ValueError):
            pass  # SIGKILL can't be handled, and needs no reset
        os.kill(os.getpid(), -rc)
        return 128 - rc
    return rc


def _call_command(cmd, got=(), started=None):
    """Run cmd and return its exit code, the shell's 127 or 126 if it can't start.

    A file with no #! line runs under /bin/sh, as caffeinate's execvp ran it
    before every job went through `wrap`. Once it starts, its pid goes in
    `started` (see _stop_group). A signal in `got` came before the command
    could get it, so the process group is sent the first one then.
    """
    try:
        try:
            child = subprocess.Popen(cmd)
        except OSError as exc:
            path = cmd[0] if "/" in cmd[0] else shutil.which(cmd[0])
            if exc.errno != errno.ENOEXEC or not path:
                raise
            child = subprocess.Popen(["/bin/sh", path] + cmd[1:])
    except OSError as exc:
        print("orchard: cannot run %s: %s" % (cmd[0], exc), file=sys.stderr)
        return 127 if exc.errno in (errno.ENOENT, errno.ENOTDIR) else 126
    except subprocess.SubprocessError as exc:
        print("orchard: cannot run %s: %s" % (cmd[0], exc), file=sys.stderr)
        return -got[0] if got else 126
    if started:
        try:
            with open(started, "w") as f:
                f.write(str(child.pid))
        except OSError:
            pass
    if got:
        try:
            os.killpg(os.getpgrp(), got[0])
        except OSError:
            pass
    return child.wait()


def settle_status(job_dir):
    """The job's status, made final from exit.json when its agent died first.

    status.json says running or paused until the agent writes how the job
    ended. If the agent died and the job ran on to the end, its process group
    is gone and cmd_wrap's exit.json holds the exit code, which then becomes
    the final status. A record from another attempt, one whose command was
    signalled (a shutdown stopping it), or a group still alive, leaves the
    status as it is.
    """
    status = read_status(job_dir)
    if not status or status.get("state") not in ("running", "paused"):
        return status
    try:
        with open(os.path.join(job_dir, "exit.json")) as f:
            ended = json.load(f)
    except (OSError, ValueError):
        return status
    if not isinstance(ended, dict) or ended.get("attempt") != status.get("attempt") or ended.get("signal"):
        return status
    if str(status.get("id")) in running_jobs():
        return status
    exit_code = ended.get("exit_code")
    ended_at = ended.get("ended_at") or time.time()
    status.update(
        state="succeeded" if exit_code == 0 else "failed",
        pause_reason=None,
        paused_since=None,
        ended_at=ended_at,
        elapsed_s=round(ended_at - (status.get("started_at") or ended_at), 1),
        exit_code=exit_code,
    )
    # A run started by an older agent may have had a kill that left no file,
    # so nothing is said and the dispatcher takes the kill to have reached it.
    if status.get("kill_marker"):
        status["kill_requested"] = os.path.exists(_kill_request_path(job_dir))
    try:
        _atomic_write_json(os.path.join(job_dir, "status.json"), status)
    except OSError:
        pass  # the caller still gets the final status
    return status


def cmd_settle(argv):
    if len(argv) != 1:
        print("usage: agent.py settle <job_dir>", file=sys.stderr)
        return 2
    status = settle_status(argv[0])
    if status is None:
        print("no status.json in %s" % argv[0], file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2))
    return 0


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
    # A pid file with no status.json is a leftover (the start command removes
    # both, and exec writes status.json first), so its pid proves nothing.
    status = read_status(argv[0])
    if status is None:
        clear_running_marker(os.path.basename(argv[0].rstrip("/")))
        print("process group not running (no status.json beside its pid file)", file=sys.stderr)
        return 1
    # After a restart the pid can belong to an unrelated process.
    started_boot = status.get("boot_id")
    boot = boot_id()
    if started_boot and boot and started_boot != boot:
        clear_running_marker(os.path.basename(argv[0].rstrip("/")))
        print("process group not running (the machine restarted)", file=sys.stderr)
        return 1
    if not _same_process(pid, status.get("pid_start")):
        clear_running_marker(os.path.basename(argv[0].rstrip("/")))
        print("process group not running (its pid now belongs to another process)", file=sys.stderr)
        return 1
    # The group, not the pid: a command can outlive a SIGKILLed wrap, and its
    # group keeps wrap's number, which no new process can take meanwhile. The
    # marker stays until the group is gone (running_jobs prunes it then), so
    # the worker doesn't read as free while a command shrugs off SIGTERM.
    pgid = status.get("pgid") or pid
    if sig in (signal.SIGTERM, signal.SIGKILL):
        # A group that is gone needs no new request (a kill already on record
        # stays), and an unwritable folder must not keep the dispatcher asking
        # about a run that is over.
        if not _group_alive(pgid):
            clear_running_marker(os.path.basename(argv[0].rstrip("/")))
            print("process group not running", file=sys.stderr)
            return 1
        # Read by the job's agent (cmd_exec) and by a pause in flight (_stop_group),
        # which only check that it exists. Without it, a run this ends would
        # read as having ended by itself and be finished rather than rerun, so
        # no signal goes out: exit 3 tells the dispatcher nothing happened and
        # to ask again. An empty file is enough, so a full disk with inodes
        # left still records it, and one already there needs no write.
        kill_path = _kill_request_path(argv[0])
        if not os.path.exists(kill_path):
            try:
                os.close(os.open(kill_path, os.O_CREAT | os.O_WRONLY, 0o644))
            except OSError as exc:
                if not os.path.exists(kill_path):
                    print("could not record the kill request (%s); not signalled" % exc, file=sys.stderr)
                    return 3
    try:
        if cont_first:
            os.killpg(pgid, signal.SIGCONT)
        if sig == signal.SIGSTOP:
            _stop_group(pgid, pid, _started_path(argv[0]))
        else:
            os.killpg(pgid, sig)
    except ProcessLookupError:
        clear_running_marker(os.path.basename(argv[0].rstrip("/")))
        print("process group not running", file=sys.stderr)
        return 1
    except PermissionError:
        # macOS, for a group of unreaped zombies: its agent reaps them and
        # clears the marker. A group with none of our processes is another
        # user's, its number reused, and the marker is stale.
        if not _group_alive(pgid):
            clear_running_marker(os.path.basename(argv[0].rstrip("/")))
        print("process group not running", file=sys.stderr)
        return 1
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
        if now - entry["mtime"] > entry.get("max_age_days", max_age_days) * 86400:
            picked.append(entry)
        else:
            leftover.append(entry)
    freed_gb = sum(e["size"] for e in picked) / 1e9
    if free_gb + freed_gb < min_free_gb:
        for entry in sorted(leftover, key=lambda e: e["mtime"]):
            if free_gb + freed_gb >= min_free_gb:
                break
            if entry.get("recent"):
                continue
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


def _gc_entries(now, max_age_days=14.0):
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
        live = set(running_jobs())
        for name in os.listdir(jobs_dir):
            path = os.path.join(jobs_dir, name)
            if not os.path.isdir(path):
                continue
            status = read_status(path) or {}
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = now
            status_path = os.path.join(path, "status.json")
            status_mtime = None
            if os.path.exists(status_path):
                status_mtime = os.path.getmtime(status_path)
                mtime = max(mtime, status_mtime)
            # A run that says "running" with no live process died with its
            # machine (a restart, a WSL shutdown) and will never finish. The
            # dispatcher may still resume it from its ckpt/, so it is kept as
            # long as a finished job would be, but not forever.
            protected = status.get("state") in ("running", "paused") and (
                name in live or (status_mtime is not None and now - status_mtime < max_age_days * 86400)
            )
            if protected and status.get("workdir"):
                live_workdirs.append(status["workdir"])
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
    replicas_dir = os.path.join(ORCHARD_HOME, "replicas")
    if os.path.isdir(replicas_dir):
        for name in os.listdir(replicas_dir):
            path = os.path.join(replicas_dir, name)
            if not REPLICA_KEY_RE.match(name) or not os.path.isdir(path):
                continue
            protected = bool(_replica_busy(path)) or any(w.startswith(path + os.sep) for w in live_workdirs)
            mtime = 0.0
            for part in (path, os.path.join(path, "meta.json"), os.path.join(path, "used"),
                         os.path.join(path, "current")):
                try:
                    mtime = max(mtime, os.path.getmtime(part))
                except OSError:
                    pass
            mtime = mtime or now
            # A replica is meant to stay warm between uses, so it outlives job
            # folders; low free space still takes it, oldest first, but not
            # one used in the last day.
            entries.append({"path": path, "mtime": mtime, "size": _dir_size(path), "protected": protected,
                            "replica": True, "max_age_days": max(max_age_days, REPLICA_MIN_AGE_DAYS),
                            "recent": now - mtime < REPLICA_RECENT_S})
    return entries


def _gc_free_gb():
    host = read_host_status() if os_family() != "macos" else None
    return worker_disk_free_gb(ORCHARD_HOME, host)


def cmd_gc(argv):
    max_age = float(argv[0]) if len(argv) > 0 else 14.0
    min_free = float(argv[1]) if len(argv) > 1 else 3.0
    os.makedirs(ORCHARD_HOME, exist_ok=True)
    now = time.time()
    picked = select_garbage(_gc_entries(now, max_age), now, max_age, min_free, _gc_free_gb())
    freed = 0
    for entry in picked:
        if entry.get("replica"):
            # A sync or a run may start between the scan and here.
            with _replica_lock(entry["path"], block=False, create=False) as got:
                if got and not _replica_busy(entry["path"]) and _rmtree_force(entry["path"]):
                    freed += entry["size"]
            continue
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
        "free_gb": _gc_free_gb(),
    }))
    return 0


# Living replicas. A replica is a warm copy of one project, kept in
# ~/.orchard/replicas/<key>/: work/ (the tree), manifest.json (path ->
# [kind, exe flag or link target, sha256, size, mtime_ns] for every file the
# Air ships), meta.json, and current (the job using it, from replica-begin
# until the Air has collected its results and released it). The Air sends
# only what changed; these verbs apply it under a lock and say what the tree
# is now. The digest of a manifest is the same tree digest receipts use.

REPLICA_KEY_RE = re.compile(r"^[0-9a-f]{12}$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TOKEN_RE = re.compile(r"^[0-9a-f]{6,32}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
REPLICA_META_FIELDS = ("project", "prep_digest", "prep_job", "env_key", "venv")
REPLICA_STARTING_S = 60  # how long a job just begun counts as running before its agent says so
REPLICA_COLLECT_S = 600  # how long a finished job holds its replica for the Air to collect
REPLICA_RECENT_S = 86400  # low disk spares a replica used this recently
REPLICA_INCOMING_MAX_S = 3600  # an update left this long was abandoned by its sender
REPLICA_MIN_AGE_DAYS = 14.0
REPLICA_USAGE = {
    "replica-state": "<key> [manifest]",
    "replica-apply": "<key> [token]",
    "replica-begin": "<key> <job.json> <digest>",
    "replica-release": "<key> <job>",
    "replica-meta": "<key> <json>",
    "replica-pack": "<key>",
    "replica-down": "<key>",
}


def _replica_dir(key):
    if not isinstance(key, str) or not REPLICA_KEY_RE.match(key):
        raise ValueError("bad replica key: %r" % (key,))
    return os.path.join(ORCHARD_HOME, "replicas", key)


@contextlib.contextmanager
def _replica_lock(rep, block=True, create=True):
    """Hold the replica's lock; yields whether it was taken. Without create a
    missing replica is not made, and yields None."""
    import fcntl
    lock_path = os.path.join(rep, ".lock")
    while True:
        if create:
            os.makedirs(rep, exist_ok=True)
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except FileNotFoundError:
            yield None
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
            got = True
        except OSError:
            got = False
        try:
            same = os.path.samestat(os.fstat(fd), os.stat(lock_path))
        except OSError:
            same = False
        if got and not same:
            # replica-down removed the folder while this waited for it.
            os.close(fd)
            continue
        break
    try:
        yield got
    finally:
        os.close(fd)


@contextlib.contextmanager
def _machine_lock():
    """One replica-begin at a time on this machine, so two cannot both find
    it free."""
    import fcntl
    state_dir = os.path.join(ORCHARD_HOME, "state")
    os.makedirs(state_dir, exist_ok=True)
    fd = os.open(os.path.join(state_dir, "replica-begin.lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _read_json(path, default):
    try:
        with open(path) as f:
            value = json.load(f)
    except (OSError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replica_entry(path, prev=None):
    """The manifest entry for path, reusing prev's hash while its size, mtime
    and exe bit are unchanged; None for anything but a file or a link."""
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        entry = ["L", os.readlink(path), None, st.st_size, st.st_mtime_ns]
        return prev if prev == entry else entry
    if not stat.S_ISREG(st.st_mode):
        return None
    exe = "x" if st.st_mode & 0o111 else "-"
    if prev and prev[0] == "F" and prev[1] == exe and prev[3] == st.st_size and prev[4] == st.st_mtime_ns:
        return prev
    return ["F", exe, _sha256_file(path), st.st_size, st.st_mtime_ns]


def manifest_digest(manifest):
    """The digest receipts.tree_digest gives the tree this manifest describes.
    A "?" entry is a file whose content is not known (one that could not be
    read, or one an interrupted update was changing)."""
    h = hashlib.sha256()
    for arc in sorted(manifest):
        entry = manifest[arc]
        if entry[0] == "L":
            h.update(("L\0%s\0%s\n" % (arc, entry[1])).encode())
        elif entry[0] == "?":
            h.update(("?\0%s\n" % arc).encode())
        else:
            h.update(("F\0%s\0%s\0%s\n" % (arc, entry[1], entry[2])).encode())
    return h.hexdigest()


def _match_case(folder, name, listing):
    """Give folder's entry the spelling name when it is there in another case:
    on a case-insensitive disk a renamed folder or file keeps its old spelling
    otherwise. listing caches folder contents across calls."""
    names = listing.get(folder)
    if names is None:
        names = listing[folder] = set(os.listdir(folder))
    if name in names:
        return
    for other in names:
        if other.lower() == name.lower():
            os.rename(os.path.join(folder, other), os.path.join(folder, name))
            names.discard(other)
            break
    names.add(name)


def _replica_path(work, arc, make_parents=False, listing=None):
    """Where arc lives under work. Refuses absolute paths, "..", and anything
    that would resolve outside work. With make_parents, a file or link in the
    way of a parent folder is replaced by the folder, and a parent spelled in
    another case takes arc's spelling."""
    parts = arc.split("/") if isinstance(arc, str) else []
    if not parts or "\0" in arc or any(p in ("", ".", "..") for p in parts):
        raise ValueError("unsafe path in replica: %r" % (arc,))
    if make_parents:
        listing = {} if listing is None else listing
        cur = work
        for part in parts[:-1]:
            parent, cur = cur, os.path.join(cur, part)
            if os.path.islink(cur) or (os.path.lexists(cur) and not os.path.isdir(cur)):
                os.unlink(cur)
                listing.pop(parent, None)
            if os.path.isdir(cur):
                _match_case(parent, part, listing)
            else:
                os.mkdir(cur)
                listing.pop(parent, None)
    path = os.path.join(work, *parts)
    real_work = os.path.realpath(work)
    parent = os.path.realpath(os.path.dirname(path))
    if parent != real_work and not parent.startswith(real_work + os.sep):
        raise ValueError("path leaves the replica: %r" % (arc,))
    return path


def _rmtree_force(path):
    """Remove path even when a run left read-only folders in it. True once it
    is gone."""
    def writable(folder):
        try:
            if not os.path.islink(folder):
                os.chmod(folder, stat.S_IMODE(os.lstat(folder).st_mode) | stat.S_IRWXU)
        except OSError:
            pass
    if os.path.isdir(path) and not os.path.islink(path):
        writable(path)
        for dirpath, dirnames, _filenames in os.walk(path):
            for name in dirnames:
                writable(os.path.join(dirpath, name))
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.lexists(path):
        try:
            os.unlink(path)
        except OSError:
            pass
    return not os.path.lexists(path)


def _remove_path(path):
    if os.path.lexists(path) and not _rmtree_force(path):
        raise OSError(errno.EACCES, "could not remove", path)


def _prune_empty(work, folder):
    while folder != work and folder.startswith(work + os.sep):
        try:
            os.rmdir(folder)
        except OSError:
            return
        folder = os.path.dirname(folder)


def _replica_refresh(rep, manifest):
    """Re-stat what the manifest lists and rehash what changed. A file a
    check wrote to, or someone edited in the copy, gets its real entry; a
    missing one is dropped. Either way the Air sees the difference and
    resends it. Returns (drifted paths, whether the manifest changed)."""
    work = os.path.join(rep, "work")
    drifted = []
    changed = False
    for arc in sorted(manifest):
        old = manifest[arc]
        try:
            new = replica_entry(_replica_path(work, arc), old)
        except (OSError, ValueError):
            new = None
        if new is old:
            continue
        changed = True
        if new is None:
            del manifest[arc]
            drifted.append(arc)
            continue
        manifest[arc] = new
        if new[:3] != old[:3]:
            drifted.append(arc)
    return drifted, changed


def _replica_recover(rep, manifest):
    """Clear up after an update that was cut short (its sender killed, the
    machine off): the paths it was changing are rehashed on the next refresh
    and its half-written files go. True if there was one; the caller removes
    the journal once the manifest is written."""
    journal = os.path.join(rep, "applying.json")
    if not os.path.lexists(journal):
        return False
    work = os.path.join(rep, "work")
    for arc in _read_json(journal, []):
        if not isinstance(arc, str):
            continue
        try:
            present = os.path.lexists(_replica_path(work, arc))
        except ValueError:
            continue
        if present:
            manifest[arc] = ["?", None, None, 0, 0]
        else:
            manifest.pop(arc, None)
    for dirpath, _dirnames, filenames in os.walk(work):
        for name in filenames:
            if name.startswith(".orchard-incoming-"):
                try:
                    os.unlink(os.path.join(dirpath, name))
                except OSError:
                    pass
    return True


def _replica_job(rep):
    """(job id, phase) of the job holding this replica: "running", "starting"
    (begun, its agent not up yet) or "collecting" (finished, its results not
    yet fetched and released). (None, None) when it is free."""
    current = os.path.join(rep, "current")
    try:
        with open(current) as f:
            job_id = f.read().strip()
        begun = os.path.getmtime(current)
    except OSError:
        return None, None
    if not JOB_ID_RE.match(job_id):
        return None, None
    if job_id in running_jobs():
        return job_id, "running"
    job_dir = os.path.join(ORCHARD_HOME, "jobs", job_id)
    status = read_status(job_dir) or {}
    now = time.time()
    if status.get("state") in TERMINAL_STATES:
        try:
            ended = os.path.getmtime(os.path.join(job_dir, "status.json"))
        except OSError:
            ended = begun
        return (job_id, "collecting") if now - ended < REPLICA_COLLECT_S else (None, None)
    if now - begun < REPLICA_STARTING_S:
        return job_id, "starting"
    return None, None


def _replica_busy(rep):
    """The job holding this replica, or None."""
    return _replica_job(rep)[0]


def _machine_busy(rep):
    """A job running on this machine outside rep. A replica run takes the
    whole machine, like any other job."""
    for job_id in running_jobs():
        status = read_status(os.path.join(ORCHARD_HOME, "jobs", job_id)) or {}
        if status.get("state") not in TERMINAL_STATES:
            return job_id
    replicas_dir = os.path.dirname(rep)
    try:
        names = os.listdir(replicas_dir)
    except OSError:
        names = []
    for name in names:
        other = os.path.join(replicas_dir, name)
        if other != rep and REPLICA_KEY_RE.match(name):
            job_id, phase = _replica_job(other)
            if phase == "starting":
                return job_id
    return None


def _replica_write(rep, manifest, meta):
    _atomic_write_json(os.path.join(rep, "manifest.json"), manifest)
    meta["digest"] = manifest_digest(manifest)
    meta["files"] = len(manifest)
    meta["last_used"] = time.time()
    _atomic_write_json(os.path.join(rep, "meta.json"), meta)


def _replica_files(rep):
    return (_read_json(os.path.join(rep, "manifest.json"), {}),
            _read_json(os.path.join(rep, "meta.json"), {}))


def _replica_load(rep):
    """The manifest and meta, brought up to date with what is on disk, and
    the paths that had drifted."""
    manifest, meta = _replica_files(rep)
    recovered = _replica_recover(rep, manifest)
    drifted, changed = _replica_refresh(rep, manifest)
    if changed or recovered:
        _replica_write(rep, manifest, meta)
    if recovered:
        os.unlink(os.path.join(rep, "applying.json"))
    return manifest, meta, drifted


def cmd_replica_state(argv):
    rep = _replica_dir(argv[0])
    want_manifest = argv[1:] == ["manifest"]
    work = os.path.join(rep, "work")
    out = {"exists": False, "busy": None, "drifted": [], "meta": {}}
    manifest = {}
    if os.path.isdir(work):
        with _replica_lock(rep, create=False) as got:
            if got and os.path.isdir(work):
                manifest, meta, drifted = _replica_load(rep)
                out.update(exists=True, busy=_replica_busy(rep), drifted=drifted, meta=meta)
    out["digest"] = manifest_digest(manifest)
    out["files"] = len(manifest)
    # Enough for the Air to choose between replicas and match a check's tags
    # in the same round trip.
    out["caps"] = collect_caps(bench=False)
    # Job paths in a replica must be spelled exactly as the agent spells them.
    out["home"] = ORCHARD_HOME
    if want_manifest:
        out["manifest"] = manifest
    print(json.dumps(out))
    return 0


def _replica_extract(work, tgz, wanted, manifest):
    got = set()
    listing = {}
    with tarfile.open(tgz, "r:gz") as tar:
        for member in tar:
            arc = member.name
            if arc not in wanted or arc in got:
                raise ValueError("unexpected path in replica update: %r" % (arc,))
            path = _replica_path(work, arc, make_parents=True, listing=listing)
            if os.path.lexists(path):
                _match_case(os.path.dirname(path), os.path.basename(path), listing)
            if os.path.isdir(path) and not os.path.islink(path):
                _remove_path(path)
            if member.issym():
                if os.path.lexists(path):
                    os.unlink(path)
                os.symlink(member.linkname, path)
                manifest[arc] = replica_entry(path)
            elif member.isreg():
                src = tar.extractfile(member)
                tmp = os.path.join(os.path.dirname(path), ".orchard-incoming-%d" % os.getpid())
                h = hashlib.sha256()
                try:
                    with open(tmp, "wb") as out:
                        for chunk in iter(lambda: src.read(1 << 20), b""):
                            h.update(chunk)
                            out.write(chunk)
                    os.chmod(tmp, member.mode & 0o777)
                    os.replace(tmp, path)
                except BaseException:
                    if os.path.lexists(tmp):
                        os.unlink(tmp)
                    raise
                st = os.lstat(path)
                manifest[arc] = ["F", "x" if st.st_mode & 0o111 else "-", h.hexdigest(), st.st_size, st.st_mtime_ns]
            else:
                raise ValueError("unsupported entry in replica update: %r" % (arc,))
            got.add(arc)
    return got


def _drop_stale_incoming(rep):
    """Updates whose senders gave up before applying them."""
    now = time.time()
    for name in os.listdir(rep):
        if name.startswith("incoming") and name.endswith((".json", ".tgz")):
            path = os.path.join(rep, name)
            try:
                if now - os.path.getmtime(path) > REPLICA_INCOMING_MAX_S:
                    os.unlink(path)
            except OSError:
                pass


def cmd_replica_apply(argv):
    rep = _replica_dir(argv[0])
    token = argv[1] if len(argv) > 1 else ""
    if token and not TOKEN_RE.match(token):
        raise ValueError("bad update token: %r" % (token,))
    # Each sync has its own names, so two Airs updating one replica cannot
    # apply each other's files.
    suffix = "-" + token if token else ""
    work = os.path.join(rep, "work")
    incoming_json = os.path.join(rep, "incoming%s.json" % suffix)
    incoming_tgz = os.path.join(rep, "incoming%s.tgz" % suffix)
    journal = os.path.join(rep, "applying.json")
    with _replica_lock(rep):
        try:
            busy = _replica_busy(rep)
            if busy:
                print(json.dumps({"error": "busy", "busy": busy}))
                return 3
            _drop_stale_incoming(rep)
            incoming = _read_json(incoming_json, {})
            os.makedirs(work, exist_ok=True)
            manifest, meta, _drifted = _replica_load(rep)
            base = manifest_digest(manifest)
            if incoming.get("base") != base:
                print(json.dumps({"error": "base", "digest": base}))
                return 4
            deleted = list(incoming.get("deleted") or [])
            wanted = set(incoming.get("updates") or [])
            # Written before any file changes: if this is killed part way, the
            # next look at the replica rehashes these paths instead of
            # trusting entries that may no longer be true.
            _atomic_write_json(journal, deleted + sorted(wanted))
            done = False
            try:
                for arc in deleted:
                    path = _replica_path(work, arc)
                    _remove_path(path)
                    manifest.pop(arc, None)
                    _prune_empty(work, os.path.dirname(path))
                if wanted:
                    missing = wanted - _replica_extract(work, incoming_tgz, wanted, manifest)
                    if missing:
                        raise ValueError("update is missing %d file(s), e.g. %r" % (len(missing), sorted(missing)[0]))
                done = True
            except Exception as exc:  # whatever stopped it, the copy and its manifest must still agree
                print(json.dumps({"error": str(exc) or type(exc).__name__}))
                return 6
            finally:
                if not done:
                    _replica_recover(rep, manifest)
                    _replica_refresh(rep, manifest)
                _replica_write(rep, manifest, meta)
                os.unlink(journal)
        finally:
            for path in (incoming_json, incoming_tgz):
                try:
                    os.unlink(path)
                except OSError:
                    pass
    if incoming.get("digest") and meta["digest"] != incoming["digest"]:
        print(json.dumps({"error": "digest", "digest": meta["digest"]}))
        return 5
    print(json.dumps({"digest": meta["digest"], "files": len(manifest)}))
    return 0


def _replica_clear_out(work, manifest):
    """Empty out/ of everything but the files the tree itself has there, so a
    run starts with out/ as a fresh copy would have it."""
    out = os.path.join(work, "out")
    if os.path.islink(out) or (os.path.lexists(out) and not os.path.isdir(out)):
        if "out" not in manifest:
            os.unlink(out)
        return
    for dirpath, dirnames, filenames in os.walk(out, topdown=False):
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            full = os.path.join(dirpath, name)
            if os.path.relpath(full, work).replace(os.sep, "/") not in manifest:
                os.unlink(full)
        if dirpath != out:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def _touch(path):
    with open(path, "a"):
        pass
    os.utime(path, None)


def cmd_replica_begin(argv):
    """Start a job in the replica, if it still holds the tree the Air checked
    (digest): prep, an earlier run or a person may have changed it since."""
    rep = _replica_dir(argv[0])
    job_json, expected = argv[1], argv[2]
    if not DIGEST_RE.match(expected):
        raise ValueError("bad digest: %r" % (expected,))
    job = _read_json(job_json, {})
    job_id = str(job.get("id") or "")
    work = os.path.join(rep, "work")
    if not JOB_ID_RE.match(job_id) or job.get("workdir") != work or not job.get("job_dir"):
        print("replica-begin: job.json does not belong to this replica", file=sys.stderr)
        return 2
    job_dir = job["job_dir"]
    with _machine_lock(), _replica_lock(rep, create=False) as got:
        if not got or not os.path.isdir(work) or not os.path.exists(os.path.join(rep, "manifest.json")):
            print(json.dumps({"error": "missing"}))
            return 4
        busy = _replica_busy(rep) or _machine_busy(rep)
        if busy:
            print(json.dumps({"error": "busy", "busy": busy}))
            return 3
        manifest, _meta, drifted = _replica_load(rep)
        digest = manifest_digest(manifest)
        if digest != expected:
            print(json.dumps({"error": "moved", "digest": digest, "drifted": drifted[:20]}))
            return 4
        _replica_clear_out(work, manifest)
        os.makedirs(job_dir, exist_ok=True)
        for name in ("status.json", "pid", "command.pid", "kill-requested"):
            try:
                os.unlink(os.path.join(job_dir, name))
            except OSError:
                pass
        _touch(os.path.join(rep, "used"))
        with open(os.path.join(rep, "current"), "w") as f:
            f.write(job_id)
        with open(os.path.join(job_dir, "agent.out"), "wb") as out:
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "exec", job_json],
                cwd=ORCHARD_HOME, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    print("started")
    return 0


def cmd_replica_release(argv):
    """Free the replica once the Air has collected what the job left in it."""
    rep = _replica_dir(argv[0])
    job_id = argv[1]
    released = False
    with _replica_lock(rep, create=False) as got:
        current = os.path.join(rep, "current")
        if got:
            try:
                with open(current) as f:
                    same = f.read().strip() == job_id
            except OSError:
                same = False
            if same and _replica_job(rep)[1] in (None, "collecting"):
                os.unlink(current)
                released = True
    print(json.dumps({"released": released}))
    return 0


def cmd_replica_meta(argv):
    rep = _replica_dir(argv[0])
    try:
        fields = json.loads(argv[1])
    except ValueError:
        fields = None
    if not isinstance(fields, dict) or set(fields) - set(REPLICA_META_FIELDS):
        print("replica-meta: fields must be a JSON object of %s" % ", ".join(REPLICA_META_FIELDS), file=sys.stderr)
        return 2
    with _replica_lock(rep, create=False) as got:
        if not got or not os.path.isdir(os.path.join(rep, "work")):
            print("replica-meta: no such replica", file=sys.stderr)
            return 4
        meta = _read_json(os.path.join(rep, "meta.json"), {})
        meta.update(fields)
        _atomic_write_json(os.path.join(rep, "meta.json"), meta)
    print(json.dumps(meta))
    return 0


def cmd_replica_pack(argv):
    rep = _replica_dir(argv[0])
    work = os.path.join(rep, "work")
    arcs = _read_json(sys.stdin.fileno(), [])
    with _replica_lock(rep, create=False) as got:
        if not got or not os.path.isdir(work):
            print("replica-pack: no such replica", file=sys.stderr)
            return 4
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as tar:
            for arc in arcs:
                path = _replica_path(work, arc)
                if os.path.lexists(path) and not (os.path.isdir(path) and not os.path.islink(path)):
                    tar.add(path, arcname=arc, recursive=False)
    return 0


def cmd_replica_down(argv):
    rep = _replica_dir(argv[0])
    with _replica_lock(rep, create=False) as got:
        if not got:
            print(json.dumps({"removed": False}))
            return 0
        busy = _replica_busy(rep)
        if busy:
            print(json.dumps({"error": "busy", "busy": busy}))
            return 3
        gone = _rmtree_force(rep)
    if not gone:
        print(json.dumps({"error": "some files could not be removed", "left": rep}))
        return 7
    print(json.dumps({"removed": True}))
    return 0


REPLICA_VERBS = {
    "replica-state": (cmd_replica_state, (1, 2)),
    "replica-apply": (cmd_replica_apply, (1, 2)),
    "replica-begin": (cmd_replica_begin, (3,)),
    "replica-release": (cmd_replica_release, (2,)),
    "replica-meta": (cmd_replica_meta, (2,)),
    "replica-pack": (cmd_replica_pack, (1,)),
    "replica-down": (cmd_replica_down, (1,)),
}


def cmd_replica(verb, argv):
    func, counts = REPLICA_VERBS[verb]
    if len(argv) not in counts:
        print("usage: agent.py %s %s" % (verb, REPLICA_USAGE[verb]), file=sys.stderr)
        return 2
    try:
        return func(argv)
    except ValueError as exc:
        print("%s: %s" % (verb, exc), file=sys.stderr)
        return 2


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
    if cmd == "terminate":
        return cmd_signal(rest, signal.SIGKILL, "terminate")
    if cmd == "gc":
        return cmd_gc(rest)
    if cmd == "running":
        return cmd_running(rest)
    if cmd == "settle":
        return cmd_settle(rest)
    if cmd == "wrap":
        return cmd_wrap(rest)
    if cmd in REPLICA_VERBS:
        return cmd_replica(cmd, rest)
    print("unknown command: %s" % cmd, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
