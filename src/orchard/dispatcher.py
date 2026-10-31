"""Background queue and scheduler.

Entries live in ~/.orchard/state/queue/<job_id>.json. A tick adopts running
jobs (poll status, collect finished ones, preempt long pauses) and starts
queued jobs once their worker is reachable, eligible and idle.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from . import agent, engine, env, events, placement
from .config import ORCHARD_HOME, QUEUE_DIR, STATE_DIR, Worker, load_workers
from .job import JobSpec

DISPATCH_DEFAULTS = {
    "poll_interval_s": 15,
    "idle_threshold_s": engine.IDLE_THRESHOLD_S,
    "kill_paused_after_s": 1800,
    "kill_grace_s": 120,
    "max_misses": 40,
    "hold_on_battery": True,
    "fallback_after_s": 900,
    "max_launch_errors": 6,
    "max_attempts": 20,
    "hold_on_air_busy": True,
    "air_busy_load": engine.AIR_BUSY_LOAD,
    "air_busy_max_hold_s": 1800,
    "stall_polls": 4,
}

DISPATCH_PID = STATE_DIR / "dispatch.pid"
TICK_LOCK = STATE_DIR / "tick.lock"

# launchd keeps `dispatch --watch` alive. Unsupervised, it lives in the process
# coalition of whatever started it: force-quitting that app (or quitting the
# session) killed it silently on 2026-10-24, and nothing restarted it.
AGENT_LABEL = "io.orchard.dispatch"
AGENT_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{AGENT_LABEL}.plist"
DISPATCH_LOG = Path.home() / "Library" / "Logs" / "orchard-dispatch.log"
SUPERVISED_ENV = "ORCHARD_DISPATCH_SUPERVISED"
LOG_MAX_BYTES = 5_000_000
BASE_PATH = [
    str(Path.home() / ".local" / "bin"),
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
]


def recorded_pid() -> int | None:
    """Pid written by the last dispatch --watch, alive or not."""
    try:
        return int(DISPATCH_PID.read_text().strip())
    except (OSError, ValueError):
        return None


def running_pid() -> int | None:
    """Pid of the live dispatch --watch; None if missing, dead, or reused by another program."""
    pid = recorded_pid()
    if pid is None or not _is_dispatcher(pid):
        return None
    return pid


def _proc_command(pid: int) -> str | None:
    """Command line of pid; "" when there is no such process, None when ps itself failed."""
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _is_dispatcher(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmd = _proc_command(pid)
    if cmd is None:
        return True
    # Whole words, so an editor open on orchard/dispatcher.py doesn't count.
    words = cmd.split()
    return "dispatch" in words and any(w == "orchard" or w.endswith("/orchard") for w in words)


def heartbeat() -> None:
    """Touch the pid file each tick, so its mtime dates the last tick (or the death)."""
    try:
        os.utime(DISPATCH_PID)
    except OSError:
        pass


def last_tick_age() -> float | None:
    try:
        return time.time() - DISPATCH_PID.stat().st_mtime
    except OSError:
        return None


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


def _agent_target() -> str:
    return f"gui/{os.getuid()}/{AGENT_LABEL}"


def agent_installed() -> bool:
    return AGENT_PLIST.exists()


def agent_state() -> tuple[bool, int | None]:
    """(loaded in launchd, pid launchd reports for it or None)."""
    try:
        out = _launchctl("print", _agent_target())
    except (OSError, subprocess.SubprocessError):
        return False, None
    if out.returncode != 0:
        return False, None
    m = re.search(r"^\s*pid = (\d+)", out.stdout, re.M)
    return True, int(m.group(1)) if m else None


def merged_path(path: str) -> str:
    """`path` plus each BASE_PATH dir it lacks, appended so its own order still wins."""
    dirs = [d for d in path.split(":") if d]
    dirs += [d for d in BASE_PATH if d not in dirs]
    return ":".join(dirs)


def agent_plist(launcher: Path, path: str) -> dict:
    return {
        "Label": AGENT_LABEL,
        "ProgramArguments": [str(launcher), "dispatch", "--watch"],
        # Local jobs inherit the dispatcher's environment, so keep the
        # installing shell's PATH (launchd's default is /usr/bin:/bin:...).
        "EnvironmentVariables": {SUPERVISED_ENV: "1", "PATH": merged_path(path)},
        "WorkingDirectory": str(Path.home()),
        "RunAtLoad": True,
        # Restart after a crash or kill; a SIGTERM stop (menu bar, logout) exits 0 and stays stopped.
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Standard",
        "StandardOutPath": str(DISPATCH_LOG),
        "StandardErrorPath": str(DISPATCH_LOG),
    }


def install_agent(launcher: Path) -> None:
    AGENT_PLIST.parent.mkdir(parents=True, exist_ok=True)
    DISPATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENT_PLIST, "wb") as fh:
        plistlib.dump(agent_plist(launcher, os.environ.get("PATH", "")), fh)
    _launchctl("bootout", _agent_target())
    out = None
    for _attempt in range(5):
        out = _launchctl("bootstrap", f"gui/{os.getuid()}", str(AGENT_PLIST))
        if out.returncode == 0:
            return
        time.sleep(1)  # bootout finishes asynchronously
    raise RuntimeError(out.stderr.strip() or f"launchctl bootstrap exited {out.returncode}")


def uninstall_agent() -> bool:
    _launchctl("bootout", _agent_target())
    existed = AGENT_PLIST.exists()
    AGENT_PLIST.unlink(missing_ok=True)
    return existed


def start_agent() -> str | None:
    """Start the launchd job unless it is running; returns an error message or None."""
    loaded, _pid = agent_state()
    if loaded:
        out = _launchctl("kickstart", _agent_target())
    else:
        out = _launchctl("bootstrap", f"gui/{os.getuid()}", str(AGENT_PLIST))
    if out.returncode != 0:
        return out.stderr.strip() or f"launchctl exited {out.returncode}"
    return None


def open_log() -> None:
    """Point stdout/stderr at the launchd log in append mode, so truncating it in place is safe."""
    DISPATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(DISPATCH_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


def rotate_log() -> None:
    try:
        if DISPATCH_LOG.stat().st_size > LOG_MAX_BYTES:
            shutil.copyfile(DISPATCH_LOG, DISPATCH_LOG.with_suffix(".log.1"))
            os.truncate(DISPATCH_LOG, 0)
    except OSError:
        pass


def load_config() -> dict:
    cfg = dict(DISPATCH_DEFAULTS)
    path = ORCHARD_HOME / "dispatch.json"
    try:
        cfg.update(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


def entry_path(job_id: str) -> Path:
    return QUEUE_DIR / f"{job_id}.json"


def save_entry(entry: dict) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    entry["updated_at"] = time.time()
    path = entry_path(entry["id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entry, indent=2) + "\n")
    tmp.replace(path)


def _save(entry: dict) -> bool:
    """Save a tick's change unless `orchard cancel` cancelled the job meanwhile.

    dispatch --watch holds the tick lock for life, so cancel writes the entry
    directly, often while a tick waits on ssh. The cancel stands: the entry is
    reloaded from disk, keeping only an orphan this tick recorded, and this
    returns False so the caller can stop what it started.
    """
    try:
        disk = json.loads(entry_path(entry["id"]).read_text())
    except (OSError, json.JSONDecodeError):
        disk = None
    if disk is None or disk.get("state") != "cancelled" or entry.get("state") == "cancelled":
        save_entry(entry)
        return True
    orphan = entry.get("orphan")
    entry.clear()
    entry.update(disk)
    if orphan and not entry.get("orphan"):
        entry["orphan"] = orphan
        save_entry(entry)
    return False


def enqueue(job: JobSpec, job_id: str, cwd: Path, sdir: Path, worker: str) -> dict:
    entry = {
        "id": job_id,
        "spec": dict(job.__dict__),
        "cwd": str(cwd),
        "sdir": str(sdir),
        "worker": worker,
        "state": "queued",
        "attempts": 0,
        "misses": 0,
        "rhome": None,
        "local_pid": None,
        "orphan": None,
        "note": None,
        "paused_since": None,
        "held_since": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    save_entry(entry)
    return entry


def load_entries() -> list[dict]:
    if not QUEUE_DIR.exists():
        return []
    entries = []
    for path in sorted(QUEUE_DIR.glob("*.json")):
        try:
            entries.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(entries, key=lambda e: e.get("created_at", 0))


def spec_of(entry: dict) -> JobSpec:
    return JobSpec(**entry["spec"])


_hold_logged: dict[str, str] = {}


def _log_holds(entries: list[dict], stale: set[str]) -> None:
    """Log a "held" event when a queued job's reason appears or changes.

    Runs once per tick; the in-memory dedup dict means a job held for hours
    logs one event per distinct reason (a dispatcher restart re-logs once,
    which doubles as a restart marker in the log).
    """
    for entry in entries:
        job_id = entry["id"]
        if job_id in stale:
            continue
        if entry.get("state") == "queued":
            note = entry.get("note") or ""
            if note and _hold_logged.get(job_id) != note:
                _hold_logged[job_id] = note
                events.append("held", job=job_id, worker=entry.get("worker"), reason=note)
        else:
            _hold_logged.pop(job_id, None)


def tick(workers: list[Worker] | None = None) -> list[str]:
    cfg = load_config()
    entries = load_entries()
    if workers is None:
        workers = load_workers()
    worker_map = {w.name: w for w in workers}
    caps_cache: dict[str, dict | None] = {}

    def caps_of(worker: Worker) -> dict | None:
        if worker.name not in caps_cache:
            caps_cache[worker.name] = engine.remote_caps(worker)
        return caps_cache[worker.name]

    lines: list[str] = []
    busy: set[str] = set()
    stale: set[str] = set()

    def _fresh(entry: dict) -> bool:
        """Sync the entry from disk; False when its state changed (e.g. cancelled)."""
        try:
            disk = json.loads(entry_path(entry["id"]).read_text())
        except (OSError, json.JSONDecodeError):
            return True
        if disk.get("state") != entry.get("state"):
            stale.add(entry["id"])
            return False
        entry.clear()
        entry.update(disk)
        return True

    for entry in entries:
        if entry.get("state") != "running":
            continue
        if not _fresh(entry):
            continue
        try:
            lines += poll_running(entry, worker_map, cfg)
        except Exception as exc:
            entry["note"] = f"dispatch error: {exc}"
            _save(entry)
            lines.append(f"{entry['id']}: dispatch error: {exc}")
        # A job stranded on a worker that stopped answering doesn't hold that
        # worker's queue, or its other jobs could never reach their fallback.
        if entry.get("state") == "running" and entry.get("misses", 0) <= cfg["max_misses"]:
            busy.add(entry["worker"])

    for entry in entries:
        if entry.get("state") != "queued":
            continue
        if not _fresh(entry):
            continue
        if entry.get("worker") in busy:
            entry["note"] = f"waiting for {entry['worker']} (busy)"
            _save(entry)
            lines.append(f"{entry['id']}: held (worker busy)")
            continue
        tried: list[str] = []
        try:
            lines += start_queued(entry, worker_map, cfg, caps_of, tried)
        except Exception as exc:
            lines += _note_launch_error(entry, cfg, exc)
        # A launch that was tried keeps its worker for the rest of the tick, even
        # one that cancel stopped or that errored: its run may still be ending there.
        if tried:
            busy.add(tried[0])
        elif entry.get("state") == "running":
            busy.add(entry["worker"])

    for entry in entries:
        # Synced first: the loops above make ssh calls, which give `orchard
        # cancel` time to rewrite the entry.
        if not entry.get("orphan") or not _fresh(entry):
            continue
        orphan = entry.get("orphan")
        if not orphan:
            continue
        worker = worker_map.get(orphan["worker"])
        if worker is None:
            entry["orphan"] = None
            _save(entry)
            continue
        if caps_of(worker) is None:
            continue
        try:
            lines += settle_orphan(entry, worker)
        except Exception as exc:
            lines.append(f"{entry['id']}: orphan check failed: {exc}")

    _log_holds(entries, stale)
    return lines


def idle_hold(caps: dict, cfg: dict) -> str | None:
    if caps.get("policy_paused"):
        return "worker paused by owner"
    busy = caps.get("busy_jobs") or []
    if busy:
        return f"worker running {busy[0]}"
    host = caps.get("host_status")
    if host is not None and host != "ok":
        return f"worker host status {host}"
    idle = caps.get("idle_seconds")
    if idle is None and placement.worker_os(caps) != placement.DEFAULT_OS:
        return "worker idle time unknown"
    threshold = cfg["idle_threshold_s"] + engine.RESUME_MARGIN_S
    if idle is not None and idle < threshold:
        return f"worker in use (idle {idle:.0f}s < {threshold}s)"
    if cfg.get("hold_on_battery", True) and caps.get("power") == "battery":
        return "worker on battery"
    # The agent's own pause rules, which an older agent doesn't report: a job
    # started past them would pause at its first check.
    low, pct = caps.get("pause_on_low_battery_pct"), caps.get("battery_pct")
    if low is not None and pct is not None and pct <= low:
        return f"worker battery low ({pct:.0f}%)"
    thermal = caps.get("thermal_limit")
    if thermal is not None and thermal < engine.THERMAL_PAUSE_BELOW:
        return f"worker thermal throttle (limit {thermal})"
    max_other, other = caps.get("max_other_cpu_pct"), caps.get("other_cpu_pct")
    if max_other is not None and other is not None and other > max_other:
        return f"worker busy with other work (CPU {other:.1f}% > {max_other}%)"
    return None


def _local_fallback(job: JobSpec) -> tuple[bool, str]:
    caps = agent.collect_caps(env.ensure_bench_python())
    caps["platform_ok"] = True
    probs = placement.problems(job, caps)
    if probs:
        return False, "; ".join(probs)
    return True, ""


def _hold_or_fallback(entry: dict, job: JobSpec, cfg: dict, reason: str) -> list[str]:
    now = time.time()
    entry["launch_errors"] = 0
    if entry.get("held_since") is None:
        entry["held_since"] = now
    fallback_after = cfg.get("fallback_after_s")
    waited = now - entry["held_since"]
    if fallback_after is not None and waited >= fallback_after:
        ok, why = _local_fallback(job)
        if ok:
            preempted, rhome = entry.get("preempted_attempt"), entry.get("rhome")
            if preempted is not None and rhome and not entry.get("orphan"):
                # The preempted run may still be alive on a worker that can't be
                # asked now. It is settled as an orphan once the worker answers:
                # stopped if it still runs, and never adopted.
                entry["orphan"] = {
                    "worker": entry["worker"],
                    "rjob": engine.remote_job_dir(rhome, entry["id"]),
                    "rhome": rhome,
                    "attempt": preempted,
                    "preempted": True,
                }
            entry.update(
                worker="local",
                held_since=None,
                preempted_attempt=None,
                note=f"{reason}; falling back to local after {waited / 60:.0f}m",
            )
            if not _save(entry):
                return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
            events.append(
                "fallback",
                job=entry["id"],
                worker="local",
                detail=f"after {waited / 60:.0f}m held: {reason}",
            )
            return [f"{entry['id']}: falling back to local ({reason})"]
        entry["note"] = f"{reason}; not runnable locally ({why})"
    else:
        entry["note"] = reason
    _save(entry)
    return [f"{entry['id']}: held ({entry['note']})"]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _child_gone(pid: int) -> bool:
    """True once pid has exited, reaping it if it is this process's child."""
    try:
        if os.waitpid(int(pid), os.WNOHANG)[0]:
            return True
    except ChildProcessError:
        pass
    return not _pid_alive(pid)


def acquire_tick_lock() -> bool:
    """O_EXCL lock so only one dispatcher processes the queue at a time."""
    TICK_LOCK.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(2):
        try:
            fd = os.open(TICK_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(TICK_LOCK.read_text().strip())
            except (OSError, ValueError):
                pid = None
            if pid == os.getpid():
                return True
            if pid and _pid_alive(pid):
                return False
            TICK_LOCK.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return True
    return False


def release_tick_lock() -> None:
    try:
        if TICK_LOCK.read_text().strip() == str(os.getpid()):
            TICK_LOCK.unlink(missing_ok=True)
    except OSError:
        pass


def _note_launch_error(entry: dict, cfg: dict, exc: Exception) -> list[str]:
    cap = cfg.get("max_launch_errors", 6)
    count = entry.get("launch_errors", 0) + 1
    entry["launch_errors"] = count
    if count >= cap:
        entry.update(state="failed", note=f"failed after {count} launch errors: {exc}")
        if not _save(entry):
            return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
        events.append("failed", job=entry["id"], worker=entry.get("worker"), detail=str(exc))
        return [f"{entry['id']}: FAILED ({count} launch errors: {exc})"]
    entry["note"] = f"launch error ({count}/{cap}): {exc}"
    _save(entry)
    return [f"{entry['id']}: launch error ({count}/{cap}): {exc}"]


def start_queued(
    entry: dict,
    worker_map: dict[str, Worker],
    cfg: dict,
    caps_of: Callable[[Worker], dict | None],
    tried: list[str] | None = None,
) -> list[str]:
    """Launch or hold one queued job. A launch it tries adds the worker to `tried`."""
    worker_name = entry["worker"]
    job = spec_of(entry)
    sdir = Path(entry["sdir"])
    cwd = Path(entry["cwd"])

    if worker_name == "local":
        failed = _out_of_attempts(entry, cfg)
        if failed:
            return failed
        note = entry.get("note") or ""
        exempt = "falling back" in note or "lost contact" in note
        if cfg.get("hold_on_air_busy", True) and not exempt:
            now = time.time()
            held_for = 0.0 if entry.get("held_since") is None else now - entry["held_since"]
            if held_for < cfg.get("air_busy_max_hold_s", 1800) and engine.air_busy(
                cfg.get("air_busy_load")
            ):
                if entry.get("held_since") is None:
                    entry["held_since"] = now
                entry["note"] = f"air busy (load {engine.air_load():.1f})"
                _save(entry)
                return [f"{entry['id']}: held ({entry['note']})"]
        attempt = entry.get("attempts", 0) + 1
        if tried is not None:
            tried.append("local")
        venv, pid = engine.launch_local(job, entry["id"], sdir, cwd, attempt=attempt)
        entry.update(
            state="running",
            attempts=attempt,
            misses=0,
            local_pid=pid,
            held_since=None,
            note="running locally",
        )
        if not _save(entry):
            _stop_local(pid, sdir)
            return [f"{entry['id']}: cancelled while starting; stopped the local run"]
        return [f"{entry['id']}: started locally"]

    worker = worker_map.get(worker_name)
    if worker is None:
        entry["note"] = f"worker {worker_name!r} not configured"
        _save(entry)
        return [f"{entry['id']}: held ({entry['note']})"]

    caps = caps_of(worker)
    if caps is None:
        return _hold_or_fallback(entry, job, cfg, "worker unreachable")

    if entry.get("preempted_attempt") is not None:
        # track_paused stopped this attempt. Its marker stays until the whole
        # group is gone, and a job that traps SIGTERM can take the agent's grace
        # to exit, so adopting it would collect the dying run. Wait for it to
        # end, then launch the next attempt, which resumes from its checkpoint.
        # While the worker says it is alive the job never falls back, or the
        # command would run twice at once.
        alive = engine.remote_job_alive(worker, entry["id"])
        if alive is None:
            return _hold_or_fallback(entry, job, cfg, f"waiting for the preempted run on {worker.name} to end")
        if alive:
            return _wait_preempted(entry, worker, cfg)
        entry.update(preempted_attempt=None, preempt_sent_at=None, terminate_sent_at=None)
    elif engine.remote_job_running(worker, entry["id"]):
        rhome = engine.remote_home(worker)
        entry.update(
            state="running",
            misses=0,
            rhome=rhome,
            held_since=None,
            launch_errors=0,
            note=f"adopted running job on {worker.name} after dispatcher restart",
        )
        if not _save(entry):
            return _stop_cancelled_start(entry, worker, rhome, entry.get("attempts", 0))
        events.append("adopted", job=entry["id"], worker=worker.name)
        return [f"{entry['id']}: adopted running job on {worker.name}"]

    # Checked after the wait above, so a job is never failed while its
    # preempted run is still alive on the worker.
    failed = _out_of_attempts(entry, cfg)
    if failed:
        return failed

    probs = placement.problems(job, {**caps, "platform_ok": True})
    if probs:
        entry["note"] = "; ".join(probs)
        _save(entry)
        return [f"{entry['id']}: held ({entry['note']})"]

    hold = idle_hold(caps, cfg)
    if hold:
        return _hold_or_fallback(entry, job, cfg, hold)

    attempt = entry.get("attempts", 0) + 1
    # The ssh calls above give `orchard cancel` time to cancel the job, which
    # is then not launched. A cancel during the launch is caught after it.
    if not _save(entry):
        return [f"{entry['id']}: cancelled meanwhile; not launched"]
    if tried is not None:
        tried.append(worker_name)
    try:
        rhome, _rjob = engine.launch_remote(worker, job, entry["id"], sdir, cwd, attempt=attempt)
    except Exception as exc:
        return _note_launch_error(entry, cfg, exc)
    entry.update(
        state="running",
        attempts=attempt,
        misses=0,
        rhome=rhome,
        held_since=None,
        launch_errors=0,
        note=f"running on {worker.name}",
    )
    if not _save(entry):
        return _stop_cancelled_start(entry, worker, rhome, attempt)
    return [f"{entry['id']}: launched on {worker.name}"]


def _out_of_attempts(entry: dict, cfg: dict) -> list[str] | None:
    """Fail a job that has used up max_attempts; None while it has some left.

    Preemptions don't count: they are the worker's owner taking it back, not
    the job failing.
    """
    attempts = entry.get("attempts", 0)
    if attempts - entry.get("preemptions", 0) < cfg.get("max_attempts", 20):
        return None
    entry.update(state="failed", note=f"failed after {attempts} attempts")
    if not _save(entry):
        return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
    events.append("failed", job=entry["id"], worker=entry.get("worker"), detail=f"{attempts} attempts")
    return [f"{entry['id']}: FAILED after {attempts} attempts"]


def _wait_preempted(entry: dict, worker: Worker, cfg: dict) -> list[str]:
    """One tick of a job whose preempted run is still alive on its worker.

    track_paused sent it SIGTERM. Once the run outlives kill_grace_s, the
    agent's `terminate` SIGKILLs its group, again each grace while it lingers.
    """
    now = time.time()
    entry["launch_errors"] = 0
    # The worker just answered, so a fallback (once it stops answering) waits
    # its full time from here, not from when the wait began.
    entry["held_since"] = now
    if entry.get("preempt_sent_at") is None:
        entry["preempt_sent_at"] = now
    last = entry.get("terminate_sent_at") or entry["preempt_sent_at"]
    if entry.get("rhome") and now - last >= cfg.get("kill_grace_s", 120):
        rjob = engine.remote_job_dir(entry["rhome"], entry["id"])
        # Only True is a SIGKILL sent; False means no run was signalled.
        if engine.terminate_remote(worker, rjob) is True:
            entry["terminate_sent_at"] = now
    entry["note"] = f"waiting for the preempted run on {worker.name} to end"
    if entry.get("terminate_sent_at"):
        entry["note"] += "; SIGKILL sent"
    if not _save(entry):
        return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
    return [f"{entry['id']}: held ({entry['note']})"]


def _stop_cancelled_start(entry: dict, worker: Worker, rhome: str | None, attempt: int) -> list[str]:
    """Stop a run that started on `worker` while `orchard cancel` cancelled its job.

    One that can't be stopped now (no answer, or its agent hasn't written a pid
    yet) is recorded as an orphan and stopped once the worker answers.
    """
    job_id = entry["id"]
    rjob = engine.remote_job_dir(rhome, job_id) if rhome else None
    if rjob and engine.kill_remote(worker, rjob):
        return [f"{job_id}: cancelled while starting; stopped the run on {worker.name}"]
    if rjob and not entry.get("orphan"):
        entry["orphan"] = {"worker": worker.name, "rjob": rjob, "rhome": rhome, "attempt": attempt}
        save_entry(entry)
    return [f"{job_id}: cancelled while starting; will stop the run on {worker.name} once it can"]


def _fenced(entry: dict, status: dict | None) -> dict | None:
    """A job's status, or None when an earlier attempt wrote it.

    Each launch clears status.json first, but the previous attempt's agent can
    still write one as it exits. Read as "no status yet", it is never taken for
    the new attempt finishing. A later attempt than the entry records is
    believed (its launch was not saved, then adopted), and recorded.
    """
    if status is None:
        return None
    attempt, expected = status.get("attempt"), entry.get("attempts", 0)
    if attempt is None or attempt == expected:
        return status
    if attempt < expected:
        return None
    entry["attempts"] = attempt
    return status


def poll_running(entry: dict, worker_map: dict[str, Worker], cfg: dict) -> list[str]:
    worker_name = entry["worker"]
    job_id = entry["id"]

    if worker_name == "local":
        status = _fenced(entry, engine.local_status(Path(entry["sdir"])))
        pid = entry.get("local_pid")
        if (
            pid
            and status is not None
            and status.get("state") in ("running", "paused")
            and not _pid_alive(pid)
        ):
            entry.update(
                state="failed",
                local_pid=None,
                note="local job process gone (status stuck running)",
            )
            if not _save(entry):
                return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
            events.append("failed", job=job_id, worker="local", detail="process gone (status stuck running)")
            return [f"{job_id}: FAILED (local job process gone)"]
        return finish_or_track(entry, status, worker_map, cfg)

    worker = worker_map.get(worker_name)
    if worker is None:
        entry["note"] = f"worker {worker_name!r} not configured"
        _save(entry)
        return [f"{job_id}: held (worker not configured)"]

    rjob = engine.remote_job_dir(entry["rhome"], job_id)
    status = _fenced(entry, engine.remote_status(worker, rjob))
    if status is None:
        entry["misses"] = entry.get("misses", 0) + 1
        if entry["misses"] > cfg["max_misses"]:
            waiting = f"lost contact with {worker_name}; waiting for it"
            if (entry.get("note") or "").startswith(waiting):
                return _wait_stranded(entry, worker, rjob)
            job = spec_of(entry)
            ok, why = _local_fallback(job)
            if ok:
                orphan = None
                if entry.get("rhome"):
                    orphan = {
                        "worker": worker_name,
                        "rjob": engine.remote_job_dir(entry["rhome"], job_id),
                        "rhome": entry["rhome"],
                        "attempt": entry.get("attempts"),
                    }
                    # A preempt of this attempt was sent, its reply lost: the
                    # run ending is that preemption (settle_orphan).
                    preempting = entry.get("preempting_attempt")
                    if preempting is not None and preempting == entry.get("attempts"):
                        orphan["preempted"] = True
                entry.update(
                    state="queued",
                    worker="local",
                    misses=0,
                    rhome=None,
                    orphan=orphan,
                    preempting_attempt=None,
                    note="lost contact with worker; falling back to local",
                )
                if not _save(entry):
                    return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
                events.append("requeued", job=job_id, worker="local", detail=f"lost contact with {worker_name}")
                return [f"{job_id}: lost contact with {worker_name}; requeued locally"]
            # Only this worker can run it, and it may still be running or have
            # finished there: keep the job running and wait for the worker.
            entry["note"] = f"{waiting} (not runnable locally: {why})"
            _save(entry)
            events.append("held", job=job_id, worker=worker_name, reason=entry["note"])
            return [f"{job_id}: lost contact with {worker_name}; waiting for it (not runnable locally)"]
        _save(entry)
        return [f"{job_id}: no status from {worker_name} ({entry['misses']}/{cfg['max_misses']})"]

    entry["misses"] = 0
    if status.get("state") in ("running", "paused"):
        lost = _lost_run(entry, status, worker, rjob, worker_map, cfg)
        if lost is not None:
            return lost
    return finish_or_track(entry, status, worker_map, cfg)


def _lost_run(
    entry: dict, status: dict, worker: Worker, rjob: str, worker_map: dict[str, Worker], cfg: dict
) -> list[str] | None:
    """Requeue a job whose worker answers but whose agent died mid-run.

    The agent rewrites status.json every few seconds while a job runs, paused
    or not. A restart, or a WSL VM shutting down, kills it without a final
    status, and the job would read "running" forever and keep the worker busy.
    So a status that stops changing for `stall_polls` polls is checked against
    the worker's running markers. Only a clear "not running" requeues it;
    nothing is killed, because after a restart its pid can belong to another
    process.
    """
    job_id = entry["id"]
    progress = [status.get("attempt"), status.get("elapsed_s")]
    if entry.get("last_progress") != progress:
        entry.update(last_progress=progress, stalled_polls=0)
        return None
    entry["stalled_polls"] = entry.get("stalled_polls", 0) + 1
    if entry["stalled_polls"] < cfg["stall_polls"]:
        return None
    if engine.remote_job_alive(worker, job_id) is not False:
        # Alive, or no clear answer: ask again after another quiet spell.
        entry["stalled_polls"] = 0
        return None
    # It may have finished between the two reads, or run on to the end after
    # its agent died: then the job's own exit record settles its status, and
    # it is collected now instead of running twice. The settled status is used
    # as it comes back, because the agent may have failed to write it. For an
    # agent too old to settle, the plain status decides.
    settled = engine.remote_settle(worker, rjob)
    if settled is None:
        # No clear answer, and requeueing would wipe an out/ it may still hold.
        entry["stalled_polls"] = 0
        return None
    status = settled if isinstance(settled, dict) else engine.remote_status(worker, rjob)
    if status is None:
        return None
    if status.get("state") not in ("running", "paused"):
        entry["stalled_polls"] = 0
        return finish_or_track(entry, status, worker_map, cfg)
    entry.update(
        state="queued",
        misses=0,
        paused_since=None,
        last_progress=None,
        stalled_polls=0,
        preempting_attempt=None,
        note=f"{worker.name} lost the job mid-run; resumes from checkpoint",
    )
    if not _save(entry):
        return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
    events.append("requeued", job=job_id, worker=worker.name, detail="agent gone mid-run")
    return [f"{job_id}: {worker.name} lost the job mid-run; requeued"]


def _wait_stranded(entry: dict, worker: Worker, rjob: str) -> list[str]:
    """One tick of a lost-contact job that only its worker can run.

    The worker's status is collected normally once it answers again. A worker
    that answers but has no status.json for the job (its folder was cleaned up,
    or it never really started) will never report one, so the job fails then.
    """
    job_id = entry["id"]
    if engine.remote_status_missing(worker, rjob):
        entry.update(state="failed", note=f"{worker.name} answers again but has no status for the job")
        if not _save(entry):
            return [f"{entry['id']}: cancelled meanwhile; left cancelled"]
        events.append("failed", job=job_id, worker=worker.name, detail="lost contact; no status after it came back")
        return [f"{job_id}: FAILED ({worker.name} has no status for the job)"]
    _save(entry)
    return [f"{job_id}: waiting for {worker.name} (lost contact; not runnable locally)"]


def finish_or_track(
    entry: dict, status: dict | None, worker_map: dict[str, Worker], cfg: dict
) -> list[str]:
    job_id = entry["id"]
    if status is None:
        entry["misses"] = entry.get("misses", 0) + 1
        _save(entry)
        return [f"{job_id}: waiting for status"]
    state = status.get("state", "unknown")
    if state in agent.TERMINAL_STATES:
        preempting = entry.get("preempting_attempt")
        if (
            preempting is not None
            and status.get("attempt", preempting) == preempting
            and status.get("kill_requested", True)
        ):
            # track_paused's kill reached the run but its reply was lost: the
            # run ending (killed, or exiting cleanly from a SIGTERM trap) is
            # the preemption, not the job finishing. A run the kill never
            # reached ended by itself and is finished as it ended. An agent
            # too old to record that is taken to have been reached.
            return _requeue_preempted(entry, "preempted run ended (its kill reply was lost)")
        return finish(entry, status, worker_map)
    if state == "paused":
        return track_paused(entry, status, worker_map, cfg)
    elapsed = status.get("elapsed_s") or 0
    active = status.get("active_s") or 0
    entry["note"] = f"{state} on {entry['worker']} (elapsed {elapsed:.0f}s, active {active:.0f}s)"
    _save(entry)
    return [f"{job_id}: {entry['note']}"]


def finish(entry: dict, status: dict, worker_map: dict[str, Worker]) -> list[str]:
    job_id = entry["id"]
    state = status.get("state")
    worker_name = entry["worker"]
    if worker_name != "local":
        worker = worker_map.get(worker_name)
        if worker is not None:
            engine.collect_remote(
                worker,
                job_id,
                engine.remote_job_dir(entry["rhome"], job_id),
                Path(entry["sdir"]),
                Path(entry["cwd"]),
            )
            _keep_final_status(Path(entry["sdir"]), status)
    entry.update(
        state="done" if state == "succeeded" else "failed",
        note=f"{state} (exit {status.get('exit_code')})",
        paused_since=None,
    )
    if not _save(entry):
        return [f"{job_id}: {state} on {worker_name}, but it was cancelled meanwhile; left cancelled"]
    events.append(
        "finished",
        job=job_id,
        worker=worker_name,
        state=state,
        exit_code=status.get("exit_code"),
        elapsed_s=status.get("elapsed_s"),
        attempt=status.get("attempt"),
    )
    return [f"{job_id}: {state} on {worker_name}"]


def _keep_final_status(into: Path, status: dict) -> None:
    """Write the status a job was finished from over the copy collect_remote pulled.

    The worker's status.json still says running when its agent died and
    `settle` couldn't rewrite it, and `orchard jobs` reads the copy.
    """
    try:
        (into / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    except OSError:
        pass


ORPHAN_ASKS = 20  # passes (5 min at the default poll) an orphan with no clear status is asked about


def settle_orphan(entry: dict, worker: Worker) -> list[str]:
    """Resolve the remote job a lost-contact fallback left behind, once its worker answers.

    The worker may have finished the job while out of contact, so its status is
    read before anything is killed. A success is adopted if the local rerun has
    not finished: the rerun is stopped and the worker's result becomes the job's.
    Any other finished orphan is kept beside the rerun's result (sdir/orphan and
    .orchard-out/<id>/orphan), and the `orphan` event's duplicate flag records
    whether the command also ran again. The attempt number fences stale results: a
    status written by a different attempt is never adopted.
    """
    job_id = entry["id"]
    orphan = entry["orphan"]
    rjob = orphan["rjob"]
    expected = orphan.get("attempt")
    # A run that went on to the end after its agent died still reads "running";
    # its own exit record settles it, so it is adopted instead of cancelled and
    # run twice. An agent too old to settle answers False and the plain status
    # decides. No clear answer (a timeout, a status.json that won't parse) is
    # asked about again for ORPHAN_ASKS passes, and so is a run with no pid yet
    # (its agent may still be starting). The last pass reads the plain status.
    settled = engine.remote_settle(worker, rjob)
    asks = orphan.get("asks", 0) + 1
    status = settled if isinstance(settled, dict) else None
    if settled is False or (settled is None and asks >= ORPHAN_ASKS):
        status = engine.remote_status(worker, rjob)
    if status is None:
        if not engine.remote_status_missing(worker, rjob):
            if asks < ORPHAN_ASKS:
                _save_orphan(entry, None, orphan={**orphan, "asks": asks})
                return [f"{job_id}: no clear answer about the orphan on {worker.name}; asking again"]
            # The agent's `kill` skips its boot and pid fences when it can't
            # read the status, so an orphan nothing vouches for is let go.
            _save_orphan(entry, f"orphan left alone (no readable status on {worker.name})", orphan=None)
            events.append("orphan", job=job_id, worker=worker.name, outcome="dropped", attempt=expected)
            return [f"{job_id}: no readable status for the orphan on {worker.name}; left it alone"]
        status = {}
    state = status.get("state")
    attempt = status.get("attempt", expected)

    if state not in agent.TERMINAL_STATES:
        ok = engine.kill_remote(worker, rjob)
        if ok is None:
            return [f"{job_id}: could not reach {worker.name} to cancel the orphan; trying again"]
        if ok is False and not status.get("pid") and asks < ORPHAN_ASKS:
            _save_orphan(entry, None, orphan={**orphan, "asks": asks})
            return [f"{job_id}: the orphan on {worker.name} has no pid yet; asking again"]
        _save_orphan(entry, f"orphan cancelled ({'ok' if ok else 'no-op'})", orphan=None)
        events.append("orphan", job=job_id, worker=worker.name, outcome="cancelled", attempt=expected)
        return [f"{job_id}: cancelled orphaned remote job on {worker.name}"]

    if attempt is not None and expected is not None and attempt != expected:
        _save_orphan(entry, f"orphan ignored (attempt {attempt}, expected {expected})", orphan=None)
        events.append("orphan", job=job_id, worker=worker.name, outcome="stale", state=state, attempt=attempt)
        return [f"{job_id}: ignored stale orphan result on {worker.name}"]

    # The ssh calls above give `orchard cancel` time to change the job, so
    # adopt-or-keep is decided against what is on disk now.
    try:
        disk = json.loads(entry_path(job_id).read_text())
    except (OSError, json.JSONDecodeError):
        disk = None
    if disk is not None:
        entry.clear()
        entry.update(disk)
    sdir, cwd = Path(entry["sdir"]), Path(entry["cwd"])
    rerun = entry.get("state")
    local_rerun = entry.get("worker") == "local" and rerun in ("queued", "running")
    # Every launch bumps attempts, so a count past the orphan's means the
    # command started again. Records without an attempt go by the state.
    if expected is None:
        ran_again = rerun not in ("queued", "cancelled")
    else:
        ran_again = entry.get("attempts", 0) > expected
    # A preempted run's exit is the preemption ending, even a clean one, unless
    # the kill never reached it (as in finish_or_track).
    preempted = orphan.get("preempted") and status.get("kill_requested", True)
    if state == "succeeded" and local_rerun and not preempted:
        if rerun == "running":
            _stop_local_rerun(entry, sdir)
        engine.collect_remote(worker, job_id, rjob, sdir, cwd)
        _keep_final_status(sdir, status)
        stopped = "local rerun stopped" if rerun == "running" else "local rerun not started"
        if not _save_orphan(
            entry,
            f"orphan succeeded on {worker.name} while out of contact (collected into {sdir})",
            state="done",
            worker=worker.name,
            rhome=orphan.get("rhome"),
            local_pid=None,
            orphan=None,
            paused_since=None,
            note=f"succeeded on {worker.name} while out of contact (exit {status.get('exit_code')}); {stopped}",
        ):
            events.append(
                "orphan",
                job=job_id,
                worker=worker.name,
                outcome="collected",
                state=state,
                attempt=attempt,
                rerun=rerun,
                duplicate=ran_again,
            )
            return [f"{job_id}: orphan succeeded on {worker.name}; collected, job left {entry.get('state')}"]
        events.append(
            "finished",
            job=job_id,
            worker=worker.name,
            state=state,
            exit_code=status.get("exit_code"),
            elapsed_s=status.get("elapsed_s"),
            attempt=attempt,
        )
        events.append(
            "orphan",
            job=job_id,
            worker=worker.name,
            outcome="adopted",
            state=state,
            attempt=attempt,
            rerun=rerun,
            duplicate=ran_again,
        )
        return [f"{job_id}: adopted result from {worker.name} ({stopped})"]

    side = sdir / "orphan"
    side.mkdir(parents=True, exist_ok=True)
    engine.collect_remote(worker, job_id, rjob, side, cwd, artifacts=cwd / ".orchard-out" / job_id / "orphan")
    _keep_final_status(side, status)
    _save_orphan(entry, f"orphan {state} on {worker.name} while out of contact (kept in {side})", orphan=None)
    events.append(
        "orphan",
        job=job_id,
        worker=worker.name,
        outcome="kept",
        state=state,
        attempt=attempt,
        rerun=rerun,
        duplicate=ran_again,
    )
    return [f"{job_id}: orphan {state} on {worker.name}; result kept beside the rerun"]


def _save_orphan(entry: dict, suffix: str | None, **changes) -> bool:
    """Save what settle_orphan decided without undoing a change made while it ran.

    Its ssh calls can take a while, and `orchard cancel` may rewrite the entry
    in that time. Then only the orphan field and `suffix` (added to the note)
    are applied to what is on disk, and this returns False.
    """
    try:
        disk = json.loads(entry_path(entry["id"]).read_text())
    except (OSError, json.JSONDecodeError):
        disk = None
    applied = disk is None or disk.get("state") == entry.get("state")
    if not applied:
        entry.clear()
        entry.update(disk)
        changes = {k: v for k, v in changes.items() if k == "orphan"}
    if suffix and "note" not in changes:
        changes["note"] = f"{entry.get('note') or ''}; {suffix}"
    entry.update(changes)
    save_entry(entry)
    return applied


def _stop_local_rerun(entry: dict, sdir: Path) -> None:
    """Stop a running local rerun and move its log and status to sdir/rerun."""
    _stop_local(entry.get("local_pid"), sdir)
    keep = sdir / "rerun"
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("log.txt", "status.json"):
        if (sdir / name).exists():
            (sdir / name).replace(keep / name)


def _stop_local(pid: int | None, sdir: Path) -> None:
    """Stop the local run started from sdir: its process group, then its agent."""
    deadline = time.time() + 10
    # A run launched this tick may not have written its pgid yet.
    while True:
        pgid = (engine.local_status(sdir) or {}).get("pgid")
        if pgid:
            agent._terminate_group(pgid, grace=5.0)
            break
        if not pid or _child_gone(pid) or time.time() > deadline:
            break
        time.sleep(0.25)
    while pid and not _child_gone(pid) and time.time() < deadline:
        time.sleep(0.25)
    if pid and not _child_gone(pid):
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass


def track_paused(entry: dict, status: dict, worker_map: dict[str, Worker], cfg: dict) -> list[str]:
    job_id = entry["id"]
    paused_since = status.get("paused_since") or entry.get("paused_since") or time.time()
    entry["paused_since"] = paused_since
    reason = status.get("pause_reason") or "unknown"
    if time.time() - paused_since > cfg["kill_paused_after_s"]:
        worker_name = entry["worker"]
        worker = worker_map.get(worker_name) if worker_name != "local" else None
        if worker is not None:
            # Recorded before the kill, so that if its reply is lost, the run's
            # final status still reads as this preemption (finish_or_track).
            entry["preempting_attempt"] = entry.get("attempts", 0)
            if not _save(entry):
                return [f"{job_id}: cancelled meanwhile; left cancelled"]
            if engine.kill_remote(worker, engine.remote_job_dir(entry["rhome"], job_id)) is None:
                # Requeued now, the job could fall back to local while the paused
                # run waits on the worker, untracked, to resume. Ask again next poll.
                entry["note"] = f"paused: {reason}; could not reach {worker_name} to preempt; retrying"
                _save(entry)
                return [f"{job_id}: could not reach {worker_name} to preempt; retrying"]
        return _requeue_preempted(entry, f"preempted after {cfg['kill_paused_after_s']}s paused")
    entry["note"] = f"paused: {reason}"
    _save(entry)
    return [f"{job_id}: paused ({reason})"]


def _requeue_preempted(entry: dict, line: str) -> list[str]:
    """Requeue a job whose run was preempted; the next attempt resumes from its checkpoint.

    start_queued waits for the stopped run to end before launching it, and
    preemptions don't count toward max_attempts.
    """
    job_id = entry["id"]
    entry.update(
        state="queued",
        paused_since=None,
        misses=0,
        preempting_attempt=None,
        preempted_attempt=entry.get("attempts", 0),
        preempt_sent_at=time.time(),
        terminate_sent_at=None,
        preemptions=entry.get("preemptions", 0) + 1,
        note="preempted after long pause; resumes from checkpoint",
    )
    if not _save(entry):
        return [f"{job_id}: cancelled meanwhile; left cancelled"]
    events.append("requeued", job=job_id, worker=entry["worker"], detail="preempted after long pause")
    return [f"{job_id}: {line}; requeued"]
