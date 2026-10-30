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
    return "orchard" in cmd and "dispatch" in cmd


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
            save_entry(entry)
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
            save_entry(entry)
            lines.append(f"{entry['id']}: held (worker busy)")
            continue
        try:
            lines += start_queued(entry, worker_map, cfg, caps_of)
        except Exception as exc:
            lines += _note_launch_error(entry, cfg, exc)
        if entry.get("state") == "running":
            busy.add(entry["worker"])

    for entry in entries:
        orphan = entry.get("orphan")
        if not orphan:
            continue
        worker = worker_map.get(orphan["worker"])
        if worker is None:
            entry["orphan"] = None
            save_entry(entry)
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
    thermal = caps.get("thermal_limit")
    if thermal is not None and thermal < engine.THERMAL_PAUSE_BELOW:
        return f"worker thermal throttle (limit {thermal})"
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
            entry.update(
                worker="local",
                held_since=None,
                note=f"{reason}; falling back to local after {waited / 60:.0f}m",
            )
            save_entry(entry)
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
    save_entry(entry)
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
        save_entry(entry)
        events.append("failed", job=entry["id"], worker=entry.get("worker"), detail=str(exc))
        return [f"{entry['id']}: FAILED ({count} launch errors: {exc})"]
    entry["note"] = f"launch error ({count}/{cap}): {exc}"
    save_entry(entry)
    return [f"{entry['id']}: launch error ({count}/{cap}): {exc}"]


def start_queued(
    entry: dict, worker_map: dict[str, Worker], cfg: dict, caps_of: Callable[[Worker], dict | None]
) -> list[str]:
    worker_name = entry["worker"]
    job = spec_of(entry)
    sdir = Path(entry["sdir"])
    cwd = Path(entry["cwd"])

    attempts = entry.get("attempts", 0)
    if attempts >= cfg.get("max_attempts", 20):
        entry.update(state="failed", note=f"failed after {attempts} attempts")
        save_entry(entry)
        events.append("failed", job=entry["id"], worker=entry.get("worker"), detail=f"{attempts} attempts")
        return [f"{entry['id']}: FAILED after {attempts} attempts"]

    if worker_name == "local":
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
                save_entry(entry)
                return [f"{entry['id']}: held ({entry['note']})"]
        venv, pid = engine.launch_local(job, entry["id"], sdir, cwd, attempt=entry.get("attempts", 0) + 1)
        entry.update(
            state="running",
            attempts=entry.get("attempts", 0) + 1,
            misses=0,
            local_pid=pid,
            held_since=None,
            note="running locally",
        )
        save_entry(entry)
        return [f"{entry['id']}: started locally"]

    worker = worker_map.get(worker_name)
    if worker is None:
        entry["note"] = f"worker {worker_name!r} not configured"
        save_entry(entry)
        return [f"{entry['id']}: held ({entry['note']})"]

    caps = caps_of(worker)
    if caps is None:
        return _hold_or_fallback(entry, job, cfg, "worker unreachable")

    if engine.remote_job_running(worker, entry["id"]):
        entry.update(
            state="running",
            misses=0,
            rhome=engine.remote_home(worker),
            held_since=None,
            launch_errors=0,
            note=f"adopted running job on {worker.name} after dispatcher restart",
        )
        save_entry(entry)
        events.append("adopted", job=entry["id"], worker=worker.name)
        return [f"{entry['id']}: adopted running job on {worker.name}"]

    probs = placement.problems(job, {**caps, "platform_ok": True})
    if probs:
        entry["note"] = "; ".join(probs)
        save_entry(entry)
        return [f"{entry['id']}: held ({entry['note']})"]

    hold = idle_hold(caps, cfg)
    if hold:
        return _hold_or_fallback(entry, job, cfg, hold)

    try:
        rhome, _rjob = engine.launch_remote(
            worker, job, entry["id"], sdir, cwd, attempt=entry.get("attempts", 0) + 1
        )
    except Exception as exc:
        return _note_launch_error(entry, cfg, exc)
    entry.update(
        state="running",
        attempts=entry.get("attempts", 0) + 1,
        misses=0,
        rhome=rhome,
        held_since=None,
        launch_errors=0,
        note=f"running on {worker.name}",
    )
    save_entry(entry)
    return [f"{entry['id']}: launched on {worker.name}"]


def poll_running(entry: dict, worker_map: dict[str, Worker], cfg: dict) -> list[str]:
    worker_name = entry["worker"]
    job_id = entry["id"]

    if worker_name == "local":
        status = engine.local_status(Path(entry["sdir"]))
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
            save_entry(entry)
            events.append("failed", job=job_id, worker="local", detail="process gone (status stuck running)")
            return [f"{job_id}: FAILED (local job process gone)"]
        return finish_or_track(entry, status, worker_map, cfg)

    worker = worker_map.get(worker_name)
    if worker is None:
        entry["note"] = f"worker {worker_name!r} not configured"
        save_entry(entry)
        return [f"{job_id}: held (worker not configured)"]

    rjob = engine.remote_job_dir(entry["rhome"], job_id)
    status = engine.remote_status(worker, rjob)
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
                entry.update(
                    state="queued",
                    worker="local",
                    misses=0,
                    rhome=None,
                    orphan=orphan,
                    note="lost contact with worker; falling back to local",
                )
                save_entry(entry)
                events.append("requeued", job=job_id, worker="local", detail=f"lost contact with {worker_name}")
                return [f"{job_id}: lost contact with {worker_name}; requeued locally"]
            # Only this worker can run it, and it may still be running or have
            # finished there: keep the job running and wait for the worker.
            entry["note"] = f"{waiting} (not runnable locally: {why})"
            save_entry(entry)
            events.append("held", job=job_id, worker=worker_name, reason=entry["note"])
            return [f"{job_id}: lost contact with {worker_name}; waiting for it (not runnable locally)"]
        save_entry(entry)
        return [f"{job_id}: no status from {worker_name} ({entry['misses']}/{cfg['max_misses']})"]

    entry["misses"] = 0
    if status.get("state") in ("running", "paused"):
        lost = _lost_run(entry, status, worker, rjob, cfg)
        if lost:
            return lost
    return finish_or_track(entry, status, worker_map, cfg)


def _lost_run(entry: dict, status: dict, worker: Worker, rjob: str, cfg: dict) -> list[str] | None:
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
    # It may have finished between the two reads.
    status = engine.remote_status(worker, rjob)
    if status is None or status.get("state") not in ("running", "paused"):
        return None
    entry.update(
        state="queued",
        misses=0,
        paused_since=None,
        last_progress=None,
        stalled_polls=0,
        note=f"{worker.name} lost the job mid-run; resumes from checkpoint",
    )
    save_entry(entry)
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
        save_entry(entry)
        events.append("failed", job=job_id, worker=worker.name, detail="lost contact; no status after it came back")
        return [f"{job_id}: FAILED ({worker.name} has no status for the job)"]
    save_entry(entry)
    return [f"{job_id}: waiting for {worker.name} (lost contact; not runnable locally)"]


def finish_or_track(
    entry: dict, status: dict | None, worker_map: dict[str, Worker], cfg: dict
) -> list[str]:
    job_id = entry["id"]
    if status is None:
        entry["misses"] = entry.get("misses", 0) + 1
        save_entry(entry)
        return [f"{job_id}: waiting for status"]
    state = status.get("state", "unknown")
    if state in agent.TERMINAL_STATES:
        return finish(entry, status, worker_map)
    if state == "paused":
        return track_paused(entry, status, worker_map, cfg)
    elapsed = status.get("elapsed_s") or 0
    active = status.get("active_s") or 0
    entry["note"] = f"{state} on {entry['worker']} (elapsed {elapsed:.0f}s, active {active:.0f}s)"
    save_entry(entry)
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
    entry.update(
        state="done" if state == "succeeded" else "failed",
        note=f"{state} (exit {status.get('exit_code')})",
        paused_since=None,
    )
    save_entry(entry)
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
    status = engine.remote_status(worker, rjob) or {}
    state = status.get("state")
    expected = orphan.get("attempt")
    attempt = status.get("attempt", expected)
    note = entry.get("note") or ""

    if state not in agent.TERMINAL_STATES:
        ok = engine.kill_remote(worker, rjob)
        entry.update(orphan=None, note=f"{note}; orphan cancelled ({'ok' if ok else 'no-op'})")
        save_entry(entry)
        events.append("orphan", job=job_id, worker=worker.name, outcome="cancelled", attempt=expected)
        return [f"{job_id}: cancelled orphaned remote job on {worker.name}"]

    if attempt is not None and expected is not None and attempt != expected:
        entry.update(orphan=None, note=f"{note}; orphan ignored (attempt {attempt}, expected {expected})")
        save_entry(entry)
        events.append("orphan", job=job_id, worker=worker.name, outcome="stale", state=state, attempt=attempt)
        return [f"{job_id}: ignored stale orphan result on {worker.name}"]

    sdir, cwd = Path(entry["sdir"]), Path(entry["cwd"])
    rerun = entry.get("state")
    local_rerun = entry.get("worker") == "local" and rerun in ("queued", "running")
    # Every launch bumps attempts, so a count past the orphan's means the
    # command started again. Records without an attempt go by the state.
    if expected is None:
        ran_again = rerun not in ("queued", "cancelled")
    else:
        ran_again = entry.get("attempts", 0) > expected
    if state == "succeeded" and local_rerun:
        if rerun == "running":
            _stop_local_rerun(entry, sdir)
        engine.collect_remote(worker, job_id, rjob, sdir, cwd)
        stopped = "local rerun stopped" if rerun == "running" else "local rerun not started"
        entry.update(
            state="done",
            worker=worker.name,
            rhome=orphan.get("rhome"),
            local_pid=None,
            orphan=None,
            paused_since=None,
            note=f"succeeded on {worker.name} while out of contact (exit {status.get('exit_code')}); {stopped}",
        )
        save_entry(entry)
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
    entry.update(orphan=None, note=f"{note}; orphan {state} on {worker.name} while out of contact (kept in {side})")
    save_entry(entry)
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


def _stop_local_rerun(entry: dict, sdir: Path) -> None:
    """Stop a running local rerun and move its log and status to sdir/rerun."""
    pid = entry.get("local_pid")
    deadline = time.time() + 10
    # A rerun launched this tick may not have written its pgid yet.
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
    keep = sdir / "rerun"
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("log.txt", "status.json"):
        if (sdir / name).exists():
            (sdir / name).replace(keep / name)


def track_paused(entry: dict, status: dict, worker_map: dict[str, Worker], cfg: dict) -> list[str]:
    job_id = entry["id"]
    paused_since = status.get("paused_since") or entry.get("paused_since") or time.time()
    entry["paused_since"] = paused_since
    reason = status.get("pause_reason") or "unknown"
    if time.time() - paused_since > cfg["kill_paused_after_s"]:
        worker_name = entry["worker"]
        if worker_name != "local":
            worker = worker_map.get(worker_name)
            if worker is not None:
                engine.kill_remote(worker, engine.remote_job_dir(entry["rhome"], job_id))
        entry.update(
            state="queued",
            paused_since=None,
            misses=0,
            note="preempted after long pause; resumes from checkpoint",
        )
        save_entry(entry)
        events.append("requeued", job=job_id, worker=worker_name, detail="preempted after long pause")
        return [f"{job_id}: preempted after {cfg['kill_paused_after_s']}s paused; requeued"]
    entry["note"] = f"paused: {reason}"
    save_entry(entry)
    return [f"{job_id}: paused ({reason})"]
