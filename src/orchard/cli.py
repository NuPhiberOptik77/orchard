from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import typer

from . import agent, dispatcher, engine, env, events, placement, store, transport
from .config import DEFAULT_WORKER_HOST, JOBS_DIR, STATE_DIR, Worker, init_config, load_workers
from .job import JobSpec, capture_env, load_job_file, new_job_id

app = typer.Typer(no_args_is_help=True, help="Orchard — personal compute spillover between Macs.")


def print_result(status: dict) -> None:
    state = status.get("state", "unknown")
    dur = status.get("elapsed_s")
    code = status.get("exit_code")
    print(f"result: {state} in {dur}s (exit {code})")


def find_jobs(prefix: str) -> list[Path]:
    base = JOBS_DIR
    if not base.exists():
        return []
    return [d for d in sorted(base.iterdir()) if d.name.startswith(prefix)]


def worker_rows() -> list[dict]:
    rows = [{"name": "local", "caps": agent.collect_caps(env.ensure_bench_python()), "error": None}]
    for w in load_workers():
        caps = engine.remote_caps(w)
        rows.append({"name": w.name, "caps": caps, "error": None if caps else "unreachable"})
    return rows


def _redact_entries(entries: list[dict]) -> list[dict]:
    """Mask forwarded env values so tokens never leak into printed output."""
    redacted = []
    for entry in entries:
        entry = copy.deepcopy(entry)
        spec = entry.get("spec")
        if isinstance(spec, dict) and spec.get("env_values"):
            spec["env_values"] = {name: "***" for name in spec["env_values"]}
        redacted.append(entry)
    return redacted


def jobs_rows(limit: Optional[int] = None) -> list[dict]:
    base = JOBS_DIR
    if not base.exists():
        return []
    queued = {e["id"]: e for e in dispatcher.load_entries()}
    rows = []
    for d in sorted(base.iterdir(), reverse=True):
        status = agent.read_status(str(d)) or {}
        placed = {}
        p = d / "placement.json"
        if p.exists():
            try:
                placed = json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
        entry = queued.get(d.name, {})
        rows.append(
            {
                "id": d.name,
                "state": status.get("state") or entry.get("state", "unknown"),
                "worker": placed.get("worker") or entry.get("worker") or "?",
                "kind": placed.get("kind"),
                "elapsed_s": status.get("elapsed_s"),
                "note": entry.get("note"),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


@app.command()
def init(
    host: str = typer.Option(DEFAULT_WORKER_HOST, help="SSH host of the worker Mac"),
    name: str = typer.Option("pro", help="worker name"),
    user: Optional[str] = typer.Option(None, help="SSH user (default: same as local)"),
    weight: float = typer.Option(0.2, help="scheduling weight (benchmark-relative)"),
    force: bool = typer.Option(False, "--force", help="overwrite existing config"),
) -> None:
    """Create the Orchard config with a worker entry."""
    worker = Worker(name=name, host=host, user=user, weight=weight, tags=[])
    path = init_config(worker, force=force)
    print(f"wrote {path}")
    print("next: orchard workers    (probe the worker)")


@app.command()
def workers(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    """Probe local and configured workers and print capabilities."""
    rows = worker_rows()
    if json_out:
        print(json.dumps(rows, indent=2))
        return
    header = (
        f"{'name':10} {'machine':8} {'os':8} {'cores':>5} {'ram_gb':>6} {'disk_free':>9} "
        f"{'bench':>10} {'idle_s':>7} {'power':>7}  tags"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        name, c = row["name"], row["caps"]
        if row["error"]:
            print(f"{name:10} {row['error']}")
            continue
        idle = c.get("idle_seconds")
        idle_s = f"{idle:.0f}" if idle is not None else "-"
        print(
            f"{name:10} {c.get('machine', '?'):8} {c.get('os', '?'):8} {c.get('cores', 0):>5} "
            f"{c.get('ram_gb', 0):>6} {c.get('disk_free_gb', 0):>9} {c.get('bench', 0):>10.0f} "
            f"{idle_s:>7} {c.get('power', '?'):>7}  {','.join(c.get('tags') or [])}"
        )


@app.command()
def status(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    """Aggregate status: workers, queue, recent jobs, dispatcher process."""
    payload = {
        "workers": worker_rows(),
        "queue": _redact_entries(dispatcher.load_entries()),
        "jobs": jobs_rows(limit=10),
        "dispatch_pid": dispatcher.running_pid(),
    }
    if json_out:
        print(json.dumps(payload, indent=2))
        return
    for row in payload["workers"]:
        name, c = row["name"], row["caps"]
        if row["error"]:
            print(f"{name:10} {row['error']}")
            continue
        idle = c.get("idle_seconds")
        idle_s = f"idle {idle / 60:.0f}m" if idle is not None else "idle ?"
        print(
            f"{name:10} {c.get('machine', '?'):8} {idle_s:10} {c.get('power', '?'):7} "
            f"{c.get('disk_free_gb', 0):>5} GB free  bench {c.get('bench', 0):>10.0f}"
        )
    counts: dict[str, int] = {}
    for e in payload["queue"]:
        counts[e.get("state", "?")] = counts.get(e.get("state", "?"), 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "empty"
    pid = payload["dispatch_pid"]
    print(f"queue:    {summary}")
    print(f"dispatch: running (pid {pid})" if pid else "dispatch: stopped")
    for job in payload["jobs"][:5]:
        dur = job["elapsed_s"]
        dur_s = f"{dur}s" if dur is not None else "-"
        print(f"  {job['id']:46} {job['state']:10} {job['worker']:8} {dur_s}")


@app.command()
def run(
    cmd: Optional[list[str]] = typer.Argument(None, help="command to run (after --)"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="job YAML file"),
    where: str = typer.Option("auto", "--where", "-w", help="auto | local | worker name"),
    background: bool = typer.Option(False, "--background", "-b", help="queue for a worker"),
    now: bool = typer.Option(False, "--now", help="with --background, run immediately instead of queueing"),
    name: Optional[str] = typer.Option(None, "--name", help="job name"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before the agent kills the job"),
    python: Optional[str] = typer.Option(None, "--python", help="Python spec (e.g. 3.12)"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
) -> None:
    """Submit a job: `orchard run -- python train.py`."""
    cwd = Path.cwd()

    if file is not None:
        job = load_job_file(file)
    else:
        job = JobSpec(name=name or (cmd[0] if cmd else "job"), cmd=cmd or [])

    if name:
        job.name = name
    if timeout:
        job.timeout_s = timeout
    if python:
        job.python = python
    elif file is None:
        job.python = env.capture_python(cwd)
    if tag:
        job.tags = list(tag)
    if env_names:
        job.env = list(dict.fromkeys(list(job.env) + list(env_names)))
    if job.env:
        job.env_values, missing = capture_env(job.env)
        for name in missing:
            print(f"note: env var {name} is not set here; it will not be forwarded")

    if not job.cmd:
        print("error: provide a command after -- or --file", file=sys.stderr)
        raise typer.Exit(2)

    if job.kind == "run":
        if job.requirements_file:
            job.requirements = Path(cwd / job.requirements_file).read_text()
        else:
            job.requirements, notes = env.capture_requirements(cwd)
            for note in notes:
                print(f"note: {note}")

    submit(job, where=where, background=background, now=now, cwd=cwd)


def _offload_target(job: JobSpec, worker: Optional[str], cfg: dict) -> tuple[Optional[Worker], str]:
    """First worker that is reachable, capable (platform/disk/mem) and currently free."""
    why = "no workers configured"
    candidates = load_workers()
    if worker:
        candidates = [w for w in candidates if w.name == worker]
        why = f"worker {worker!r} not configured"
    for w in candidates:
        caps = engine.remote_caps(w)
        if caps is None:
            why = f"{w.name} unreachable"
            continue
        probs = placement.problems(job, {**caps, "platform_ok": True})
        if probs:
            why = f"{w.name}: {'; '.join(probs)}"
            continue
        if job.requirements:
            ok, note = env.resolve_for_platform(
                job.requirements, job.python, env.platform_tag(caps["machine"])
            )
            if not ok:
                why = f"{w.name}: {note}"
                continue
        hold = dispatcher.idle_hold(caps, cfg)
        if hold:
            why = f"{w.name}: {hold}"
            continue
        return w, ""
    return None, why


@app.command()
def offload(
    cmd: list[str] = typer.Argument(..., help="command to run offloaded (after --)"),
    worker: Optional[str] = typer.Option(None, "--worker", "-w", help="preferred worker"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before the agent kills the job"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
) -> None:
    """Run a command on the Pro when it is free and capable, else locally.

    Drop-in for agent workflows: never fails because of the worker — falls back
    to a local run with the reason printed. Prints the job log tail and exits
    with the job's exit code.
    """
    cwd = Path.cwd()
    job = JobSpec(name=cmd[0] if cmd else "job", cmd=list(cmd))
    if timeout:
        job.timeout_s = timeout
    job.python = env.capture_python(cwd)
    if env_names:
        job.env = list(dict.fromkeys(env_names))
    if job.env:
        job.env_values, missing = capture_env(job.env)
        for name in missing:
            print(f"note: env var {name} is not set here; it will not be forwarded")
    job.requirements, notes = env.capture_requirements(cwd)
    for note in notes:
        print(f"note: {note}")

    job_id = new_job_id(job.name)
    sdir = JOBS_DIR / job_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "spec.json").write_text(json.dumps(job.__dict__, indent=2) + "\n")

    target, why = _offload_target(job, worker, dispatcher.load_config())
    events.append(
        "submitted",
        job=job_id,
        worker=target.name if target else "local",
        background=False,
        kind=job.kind,
    )
    if target is not None:
        print(f"offload: {target.name}")
        try:
            engine.check_shipping_size(cwd, job.kind)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise typer.Exit(2)
        code = run_remote(target, job, job_id, sdir, cwd)
        log = sdir / "log.txt"
        if log.exists():
            lines = log.read_text(errors="replace").splitlines()
            if lines:
                print(f"--- log tail ({job_id}) ---")
                print("\n".join(lines[-40:]))
    else:
        print(f"offload: local ({why})")
        code = run_local(job, job_id, sdir, cwd)
    raise typer.Exit(code)


@app.command()
def build(
    cmd: list[str] = typer.Argument(..., help="build command (after --)"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    where: str = typer.Option("auto", "--where", "-w", help="auto | local | worker name"),
    background: bool = typer.Option(False, "--background", "-b", help="queue for a worker"),
    now: bool = typer.Option(False, "--now", help="with --background, run immediately instead of queueing"),
    name: Optional[str] = typer.Option(None, "--name", help="job name"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before the agent kills the job"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
) -> None:
    """Submit a build job: `orchard build -t x86_64 -t swift -- swift test`.

    Build jobs ship the working tree (tracked files inside a git repo), skip
    Python environment sync, and keep `.build` as a persistent per-project cache
    on the worker, so repeat builds are incremental.
    """
    cwd = Path.cwd()
    job = JobSpec(name=name or (cmd[0] if cmd else "build"), cmd=list(cmd), kind="build", tags=list(tag))
    if timeout:
        job.timeout_s = timeout
    if env_names:
        job.env = list(env_names)
    if job.env:
        job.env_values, missing = capture_env(job.env)
        for var in missing:
            print(f"note: env var {var} is not set here; it will not be forwarded")
    submit(job, where=where, background=background, now=now, cwd=cwd)


def submit(job: JobSpec, where: str, background: bool, now: bool, cwd: Path) -> None:
    job_id = new_job_id(job.name)
    sdir = JOBS_DIR / job_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "spec.json").write_text(json.dumps(job.__dict__, indent=2) + "\n")

    local_caps = agent.collect_caps(env.ensure_bench_python())
    candidates = [placement.Candidate("local", "local", local_caps, 1.0)]
    worker_map: dict[str, Worker] = {}
    for w in load_workers():
        caps = engine.remote_caps(w)
        if caps is None:
            print(f"worker {w.name}: unreachable (skipped)")
            continue
        if job.kind == "run":
            ok, note = env.resolve_for_platform(job.requirements, job.python, env.platform_tag(caps["machine"]))
            caps["platform_ok"] = ok
            caps["platform_note"] = note
        else:
            caps["platform_ok"] = True
            caps["platform_note"] = ""
        candidates.append(placement.Candidate(w.name, "remote", caps, w.weight))
        worker_map[w.name] = w

    if where not in ("auto", "local") and where not in worker_map:
        print(f"error: worker {where!r} is not configured or unreachable", file=sys.stderr)
        raise typer.Exit(2)

    try:
        decision = placement.choose(job, candidates, where=where, background=background)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(2)
    print(f"placement: {decision.candidate.name} ({decision.candidate.kind})")
    for reason in decision.reasons:
        print(f"  - {reason}")
    placement_info = {
        "worker": decision.candidate.name,
        "kind": decision.candidate.kind,
        "reasons": decision.reasons,
    }
    (sdir / "placement.json").write_text(json.dumps(placement_info, indent=2) + "\n")

    will_remote = decision.candidate.kind == "remote"
    if not will_remote and background and not now:
        will_remote = decision.candidate.name != "local"
    if will_remote:
        try:
            engine.check_shipping_size(cwd, job.kind)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise typer.Exit(2)

    if background and not now:
        worker_name = decision.candidate.name if decision.candidate.kind != "local" else "local"
        entry = dispatcher.enqueue(job, job_id, cwd, sdir, worker=worker_name)
        events.append("submitted", job=entry["id"], worker=worker_name, background=True, kind=job.kind)
        print(f"queued: {entry['id']} (worker: {entry['worker']})")
        if dispatcher.running_pid() is None:
            print("warning: dispatch is not running; the job waits until it starts (orchard doctor)", file=sys.stderr)
            print("next: orchard dispatch --watch")
        raise typer.Exit(0)

    events.append(
        "submitted",
        job=job_id,
        worker=decision.candidate.name,
        background=False,
        kind=job.kind,
    )
    if decision.candidate.kind == "local":
        code = run_local(job, job_id, sdir, cwd)
    else:
        code = run_remote(worker_map[decision.candidate.name], job, job_id, sdir, cwd)
    raise typer.Exit(code)


def run_local(job: JobSpec, job_id: str, sdir: Path, cwd: Path) -> int:
    _venv, job_json = engine.prepare_local_job(job, job_id, sdir, cwd)
    print(f"running locally: {' '.join(job.cmd)}")
    proc = subprocess.run([sys.executable, str(Path(agent.__file__)), "exec", str(job_json)])
    status = agent.read_status(str(sdir)) or {}
    print_result(status)
    events.append("finished", job=job_id, worker="local", state=status.get("state"), exit_code=proc.returncode)
    return proc.returncode


def run_remote(worker: Worker, job: JobSpec, job_id: str, sdir: Path, cwd: Path) -> int:
    _rhome, rjob = engine.launch_remote(worker, job, job_id, sdir, cwd)
    print("Ctrl-C detaches; the job keeps running on the worker")
    try:
        engine.watch_remote(worker, rjob)
    except KeyboardInterrupt:
        print("detached; the job continues on the worker")
        return 0
    status = engine.collect_remote(worker, job_id, rjob, sdir, cwd)
    print_result(status)
    events.append(
        "finished",
        job=job_id,
        worker=worker.name,
        state=status.get("state"),
        exit_code=status.get("exit_code"),
        elapsed_s=status.get("elapsed_s"),
    )
    return 0 if status.get("state") == "succeeded" else 1


def _code_stamp() -> float:
    base = Path(__file__).parent
    return max((p.stat().st_mtime for p in base.glob("*.py")), default=0.0)


@app.command()
def cancel(job: str = typer.Argument(..., help="job id or unique prefix")) -> None:
    """Cancel a queued or running background job."""
    matches = [e for e in dispatcher.load_entries() if e["id"].startswith(job)]
    if not matches:
        print(f"error: no queue entry matching {job!r}", file=sys.stderr)
        raise typer.Exit(1)
    if len(matches) > 1:
        print(
            f"error: {len(matches)} entries match: {', '.join(m['id'] for m in matches)}",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    entry = matches[0]
    state = entry.get("state")
    if state in ("done", "failed", "cancelled"):
        print(f"{entry['id']} is already {state}")
        raise typer.Exit(0)

    if state == "running" and entry.get("worker") == "local":
        status = engine.local_status(Path(entry["sdir"])) or {}
        pgid = status.get("pgid")
        if pgid:
            try:
                agent._terminate_group(pgid, grace=5.0)
            except Exception as exc:
                print(f"warning: could not kill local process group: {exc}", file=sys.stderr)
    elif state == "running":
        worker = next((w for w in load_workers() if w.name == entry.get("worker")), None)
        rjob = engine.remote_job_dir(entry["rhome"], entry["id"]) if entry.get("rhome") else None
        killed = False
        if worker is not None and rjob:
            try:
                killed = engine.kill_remote(worker, rjob)
            except Exception:
                killed = False
        if not killed and worker is not None and rjob:
            entry["orphan"] = {"worker": worker.name, "rjob": rjob}
            print("warning: could not kill remote job now; dispatcher will retry")

    entry.update(
        state="cancelled",
        local_pid=None,
        paused_since=None,
        held_since=None,
        note=f"cancelled by user (was {state})",
    )
    dispatcher.save_entry(entry)
    events.append("cancelled", job=entry["id"], worker=entry.get("worker"), detail=f"was {state}")
    print(f"cancelled: {entry['id']} (was {state})")


def _fmt_dur(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _dir_size(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
                count += 1
            except OSError:
                pass
    return total, count


def _stats_summary(records: list[dict]) -> dict:
    submitted: dict[str, float] = {}
    starts: dict[str, float] = {}
    waits: list[float] = []
    runs: dict[str, int] = {}
    finished: dict[str, int] = {}
    holds: dict[str, dict] = {}

    for r in records:
        ev = r.get("event")
        job = r.get("job")
        ts = r.get("ts") or 0
        if ev == "submitted" and job and job not in submitted:
            submitted[job] = ts
        elif ev in ("launched", "adopted") and job:
            worker = r.get("worker") or "?"
            runs[worker] = runs.get(worker, 0) + 1
            starts.setdefault(job, ts)
        elif ev == "finished":
            state = r.get("state") or "?"
            key = "done" if state == "succeeded" else state
            finished[key] = finished.get(key, 0) + 1
        elif ev in ("failed", "cancelled"):
            finished[ev] = finished.get(ev, 0) + 1
        elif ev == "held":
            reason = re.sub(r"\d+", "N", r.get("reason") or "")
            info = holds.setdefault(reason, {"count": 0, "last": 0})
            info["count"] += 1
            info["last"] = max(info["last"], ts)

    for job, ts in submitted.items():
        start = starts.get(job)
        if start is not None and start >= ts:
            waits.append(start - ts)
    waits.sort()
    return {"runs": runs, "finished": finished, "waits": waits, "holds": holds}


@app.command()
def stats(
    days: int = typer.Option(7, "--days", "-d", help="window in days"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Summarize recent activity: runs per machine, wait times, holds by reason."""
    now = time.time()
    records = [r for r in events.load() if (r.get("ts") or 0) >= now - days * 86400]
    summary = _stats_summary(records)
    if json_out:
        print(json.dumps(summary, indent=2, default=str))
        return
    if not records:
        print(f"no events in the last {days}d (log: {events.EVENTS_PATH})")
        return
    print(f"window: last {days}d ({len(records)} events)")
    runs = summary["runs"]
    if runs:
        print("runs started: " + " · ".join(f"{w} {n}" for w, n in sorted(runs.items(), key=lambda kv: -kv[1])))
    fin = summary["finished"]
    if fin:
        print("finished: " + " · ".join(f"{n} {s}" for s, n in sorted(fin.items())))
    waits = summary["waits"]
    if waits:
        p50 = waits[min(len(waits) - 1, int(0.5 * (len(waits) - 1) + 0.5))]
        p90 = waits[min(len(waits) - 1, int(0.9 * (len(waits) - 1) + 0.5))]
        print(f"queue wait (submit→start): p50 {_fmt_dur(p50)} · p90 {_fmt_dur(p90)} · max {_fmt_dur(waits[-1])}")
    holds = summary["holds"]
    if holds:
        total = sum(v["count"] for v in holds.values())
        print(f"holds ({total}):")
        for reason, info in sorted(holds.items(), key=lambda kv: -kv[1]["count"])[:6]:
            label = reason if len(reason) <= 70 else reason[:67] + "..."
            print(f"  {info['count']:5}x  {label} (last {_fmt_dur(now - info['last'])} ago)")
    recent = [r for r in records if r.get("job")]
    if recent:
        print("recent:")
        for r in recent[-12:]:
            stamp = time.strftime("%m-%d %H:%M", time.localtime(r.get("ts") or 0))
            job = (r.get("job") or "")[:14]
            detail = r.get("reason") or r.get("detail") or r.get("state") or ""
            detail = next((ln for ln in detail.splitlines() if ln.strip()), "")[:70]
            print(f"  {stamp}  {r.get('event', '?'):9} {job:14} {r.get('worker') or '':6} {detail}")


@app.command()
def doctor() -> None:
    """Check the control plane, workers, and local resources."""
    problems = 0
    print("control plane")
    pid = dispatcher.running_pid()
    installed = dispatcher.agent_installed()
    loaded, agent_pid = dispatcher.agent_state()
    start_hint = "orchard dispatch --watch" if installed else "orchard dispatch --install"
    if pid:
        print(f"  ok    dispatch running (pid {pid}, {'launchd' if pid == agent_pid else 'unsupervised'})")
        age = dispatcher.last_tick_age()
        if age is not None and age > 600:
            print(f"  warn  dispatch last ticked {_fmt_dur(age)} ago; hung? log: {dispatcher.DISPATCH_LOG}")
    else:
        problems += 1
        stale = dispatcher.recorded_pid()
        if stale is not None:
            last = time.strftime("%m-%d %H:%M", time.localtime(dispatcher.DISPATCH_PID.stat().st_mtime))
            print(
                f"  FAIL  dispatch died without cleaning up (pid {stale}, last tick {last}): killed by a signal, "
                f"e.g. force quit, or the app or session that started it quit; start: {start_hint}"
            )
        else:
            print(f"  FAIL  dispatch stopped (clean exit: menu bar, SIGTERM or Ctrl-C); start: {start_hint}")
    if not installed:
        print("  warn  dispatch is not supervised: it dies with the app or session that started it")
        print("        fix: orchard dispatch --install")
    elif not loaded:
        print(f"  warn  launchd job {dispatcher.AGENT_LABEL} is installed but not loaded")
        print("        fix: orchard dispatch --install")
    elif pid and pid != agent_pid:
        print(f"  warn  dispatch pid {pid} runs outside launchd and dies with its parent")
        print("        launchd takes over when it exits")
    if dispatcher.TICK_LOCK.exists():
        try:
            lock_pid = int(dispatcher.TICK_LOCK.read_text().strip())
        except (OSError, ValueError):
            lock_pid = None
        if lock_pid and not dispatcher._pid_alive(lock_pid):
            print(f"  warn  stale tick lock (pid {lock_pid} dead); self-heals on next tick")
        elif lock_pid and lock_pid != pid:
            print(f"  info  tick lock held by pid {lock_pid}")

    print("workers")
    local_sha = store.sha256_bytes(Path(agent.__file__).read_bytes())
    for w in load_workers():
        caps = engine.remote_caps(w)
        if caps is None:
            problems += 1
            print(f"  FAIL  {w.name}: unreachable ({w.host})")
            continue
        idle = caps.get("idle_seconds")
        idle_txt = f"idle {_fmt_dur(idle)}" if isinstance(idle, (int, float)) else "idle ?"
        disk = caps.get("disk_free_gb") or 0
        mark = "ok  " if disk >= 3 else "warn"
        line = f"  {mark}  {w.name}: {idle_txt} · {caps.get('power', '?')} · {disk:.1f} GB free"
        if caps.get("policy_paused"):
            line += " · paused by owner"
        print(line)
        if transport.file_sha256_remote(w, env.AGENT_REMOTE_PATH) != local_sha:
            print(f"  warn  {w.name}: agent outdated; uploads automatically on next launch")

    print("local")
    free_gb = shutil.disk_usage(Path.home()).free / 1e9
    if free_gb >= 10:
        print(f"  ok    Air disk: {free_gb:.0f} GB free")
    else:
        print(f"  warn  Air disk: only {free_gb:.0f} GB free (fallback runs may fail)")
    size, count = _dir_size(store.BLOB_DIR)
    print(f"  info  blob cache: {_fmt_bytes(size)} ({count} files)")
    bench = STATE_DIR / "bench.json"
    if bench.exists():
        print(f"  info  bench cache: {_fmt_dur(time.time() - bench.stat().st_mtime)} old")

    print("queue")
    entries = dispatcher.load_entries()
    queued = [e for e in entries if e.get("state") == "queued"]
    running = [e for e in entries if e.get("state") == "running"]
    terminal = len(entries) - len(queued) - len(running)
    print(f"  info  {len(running)} running · {len(queued)} queued · {terminal} terminal")
    if queued and not pid:
        problems += 1
        print(f"  FAIL  {len(queued)} queued job(s) will not start while dispatch is stopped")
    now = time.time()
    for e in queued:
        held = e.get("held_since")
        if held and now - held > 3600:
            print(f"  warn  {e['id']} held {_fmt_dur(now - held)}: {e.get('note')}")

    raise typer.Exit(1 if problems else 0)


class _DispatchStop(Exception):
    """Raised by the SIGTERM/SIGHUP handler between ticks."""


def _stamped(line: str) -> None:
    print(f"{time.strftime('%m-%d %H:%M:%S')} {line}")


def _hand_off_to_launchd() -> bool:
    """Start dispatch under launchd instead of in this process. False means run here."""
    dispatcher.release_tick_lock()  # still held after a code-change re-exec
    existing = dispatcher.running_pid()
    if existing and existing != os.getpid():
        print(f"error: dispatch already running (pid {existing})", file=sys.stderr)
        raise typer.Exit(1)
    err = dispatcher.start_agent()
    if err is None:
        deadline = time.time() + 10
        while time.time() < deadline:
            pid = dispatcher.running_pid()
            if pid and pid != os.getpid():
                print(f"dispatch running under launchd ({dispatcher.AGENT_LABEL}, pid {pid})")
                print(f"log: {dispatcher.DISPATCH_LOG}")
                return True
            time.sleep(0.5)
        err = "the launchd job did not start within 10s"
    print(
        f"warning: {err}; running dispatch in this process instead, where it stops if its parent "
        f"app or session quits (see {dispatcher.DISPATCH_LOG})",
        file=sys.stderr,
    )
    return False


@app.command()
def dispatch(
    watch: bool = typer.Option(False, "--watch", "-w", help="keep processing the queue until interrupted"),
    interval: Optional[float] = typer.Option(None, "--interval", help="seconds between ticks while watching"),
    foreground: bool = typer.Option(
        False, "--foreground", help="with --watch: run in this process even when the launchd job is installed"
    ),
    install: bool = typer.Option(False, "--install", help="install and start the launchd job that runs dispatch"),
    uninstall: bool = typer.Option(False, "--uninstall", help="stop and remove the launchd job"),
) -> None:
    """Process the background queue: start queued jobs, track and collect running ones.

    With the launchd job installed (--install), `dispatch --watch` starts that job
    and returns, so the dispatcher never belongs to a session or app that can quit.
    """
    if install:
        launcher = Path(__file__).resolve().parents[2] / "bin" / "orchard"
        try:
            dispatcher.install_agent(launcher)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            print(f"error: cannot install the launchd job: {exc}", file=sys.stderr)
            raise typer.Exit(1)
        print(f"installed {dispatcher.AGENT_PLIST}; launchd keeps dispatch running (log: {dispatcher.DISPATCH_LOG})")
        other = dispatcher.running_pid()
        _loaded, agent_pid = dispatcher.agent_state()
        if other and other != agent_pid:
            print(f"note: dispatch pid {other} runs outside launchd; the launchd job takes over when it exits")
        return
    if uninstall:
        existed = dispatcher.uninstall_agent()
        print(f"removed {dispatcher.AGENT_PLIST}" if existed else "launchd job was not installed")
        return
    if not watch:
        existing = dispatcher.running_pid()
        if existing and existing != os.getpid():
            print(f"error: dispatch --watch is running (pid {existing})", file=sys.stderr)
            raise typer.Exit(1)
        if not dispatcher.acquire_tick_lock():
            print("error: another dispatch tick is already running", file=sys.stderr)
            raise typer.Exit(1)
        try:
            lines = dispatcher.tick()
        finally:
            dispatcher.release_tick_lock()
        print("\n".join(lines) if lines else "queue: empty")
        return
    supervised = os.environ.get(dispatcher.SUPERVISED_ENV) == "1"
    if not supervised and not foreground and dispatcher.agent_installed() and _hand_off_to_launchd():
        return
    period = interval or dispatcher.load_config()["poll_interval_s"]
    stop: dict = {"signal": None, "in_tick": False}

    def on_signal(signum: int, _frame: object) -> None:
        # Finish a tick in progress (a half-done launch would orphan a job); stop at once otherwise.
        stop["signal"] = signal.Signals(signum).name
        if not stop["in_tick"]:
            raise _DispatchStop

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    if supervised:
        dispatcher.open_log()
    sys.stdout.reconfigure(line_buffering=True)
    try:
        if supervised:
            # Wait out a dispatcher started elsewhere instead of exiting and being relaunched every 30 s.
            waiting_for = None
            while True:
                existing = dispatcher.running_pid()
                if existing in (None, os.getpid()) and dispatcher.acquire_tick_lock():
                    break
                if existing != waiting_for:
                    _stamped(f"waiting for dispatch pid {existing or '(tick lock)'} to exit")
                    waiting_for = existing
                time.sleep(period)
        else:
            existing = dispatcher.running_pid()
            if existing:
                print(f"error: dispatch already running (pid {existing})", file=sys.stderr)
                raise typer.Exit(1)
            if not dispatcher.acquire_tick_lock():
                print("error: another dispatch tick is already running", file=sys.stderr)
                raise typer.Exit(1)
        dispatcher.DISPATCH_PID.parent.mkdir(parents=True, exist_ok=True)
        dispatcher.DISPATCH_PID.write_text(str(os.getpid()))
        _stamped(f"dispatch started (pid {os.getpid()}, {'launchd' if supervised else 'unsupervised'})")
        events.append("dispatch_started", pid=os.getpid(), supervised=supervised)
        stamp = _code_stamp()
        shown: set[str] = set()
        while True:
            stop["in_tick"] = True
            lines = dispatcher.tick()
            stop["in_tick"] = False
            for line in lines:
                if line not in shown:  # "held (worker busy)" repeats every tick
                    _stamped(line)
            shown = set(lines)
            dispatcher.heartbeat()
            if supervised:
                dispatcher.rotate_log()
            if stop["signal"]:
                raise _DispatchStop
            if _code_stamp() > stamp:
                _stamped("orchard code changed; restarting dispatch with the new code")
                sys.stdout.flush()
                dispatcher.DISPATCH_PID.unlink(missing_ok=True)
                os.execv(sys.executable, [sys.executable, "-m", "orchard", *sys.argv[1:]])
            time.sleep(period)
    except KeyboardInterrupt:
        print("dispatch stopped")
    except _DispatchStop:
        # Exit 0 so launchd's KeepAlive leaves a deliberate stop (menu bar, logout) stopped.
        _stamped(f"dispatch stopped ({stop['signal']})")
        events.append("dispatch_stopped", pid=os.getpid(), signal=stop["signal"])
    finally:
        dispatcher.release_tick_lock()
        if dispatcher.recorded_pid() == os.getpid():
            dispatcher.DISPATCH_PID.unlink(missing_ok=True)


@app.command()
def queue(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    """Show background queue entries."""
    entries = dispatcher.load_entries()
    if json_out:
        print(json.dumps(_redact_entries(entries), indent=2))
        return
    if not entries:
        print("queue is empty")
        return
    header = f"{'id':46} {'state':8} {'worker':8} {'att':>3}  note"
    print(header)
    print("-" * len(header))
    for e in entries:
        print(
            f"{e['id']:46} {e.get('state', '?'):8} {e.get('worker', '?'):8} "
            f"{e.get('attempts', 0):>3}  {e.get('note') or ''}"
        )


@app.command()
def jobs(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    """List submitted jobs."""
    rows = jobs_rows()
    if json_out:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("no jobs yet")
        return
    for job in rows:
        dur = job["elapsed_s"]
        dur_s = f"{dur}s" if dur is not None else "-"
        print(f"{job['id']:46} {job['state']:10} {job['worker']:8} {dur_s}")


@app.command()
def discover(seconds: float = typer.Option(4.0, "--seconds", "-s", help="listen duration")) -> None:
    """Listen for SSH services advertised over Bonjour (mDNS) on the LAN."""
    try:
        proc = subprocess.Popen(
            ["dns-sd", "-B", "_ssh._tcp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("error: dns-sd not found (macOS only)", file=sys.stderr)
        raise typer.Exit(1)
    time.sleep(seconds)
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    found = []
    for line in out.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) < 7 or parts[1] != "Add":
            continue
        found.append(" ".join(parts[6:]))
    if not found:
        print("no _ssh._tcp services found (Remote Login must be enabled on the other Mac)")
        return
    for name in found:
        print(f"found: {name}")
    print("hint: orchard init --host <name>.local")


@app.command()
def logs(
    job_id: str = typer.Argument(..., help="job id (or unique prefix)"),
    lines: int = typer.Option(40, "--lines", "-n"),
) -> None:
    """Show the tail of a job's log."""
    matches = find_jobs(job_id)
    if not matches:
        print(f"no job matching {job_id!r}", file=sys.stderr)
        raise typer.Exit(1)
    log = matches[-1] / "log.txt"
    if not log.exists():
        print("no log yet", file=sys.stderr)
        raise typer.Exit(1)
    print("\n".join(log.read_text().splitlines()[-lines:]))


SELFTEST_MAIN = """\
import os
import pathlib

token = os.environ.get("ORCHARD_SELFTEST_TOKEN", "<unset>")
out = pathlib.Path("out")
out.mkdir(exist_ok=True)
(out / "result.txt").write_text(f"selftest ok token={token}\\n")
print("selftest job ran")
"""


def _selftest_run(cwd: Path, args: list[str], token: str, timeout: float) -> tuple[int, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["ORCHARD_SELFTEST_TOKEN"] = token
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "orchard", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _selftest_artifact(path: Path | None, expected: str) -> str | None:
    if path is None or not path.exists():
        return f"artifact missing ({path})"
    text = path.read_text()
    if expected not in text:
        return f"artifact mismatch: {text.strip()!r}"
    return None


@app.command()
def selftest(
    worker: Optional[str] = typer.Option(None, "--worker", help="worker for the remote leg"),
    local_only: bool = typer.Option(False, "--local-only", help="skip the remote leg"),
    keep: bool = typer.Option(False, "--keep", help="keep the temp project for inspection"),
    timeout: int = typer.Option(300, "--timeout", help="seconds per leg"),
) -> None:
    """End-to-end canary: run a job locally and on the worker, assert artifacts."""
    proj = Path(tempfile.mkdtemp(prefix="orchard-selftest-"))
    (proj / "main.py").write_text(SELFTEST_MAIN)
    failures: list[str] = []
    try:
        start = time.time()
        code, out = _selftest_run(
            proj,
            ["run", "-w", "local", "--env", "ORCHARD_SELFTEST_TOKEN", "--", "python", "main.py"],
            "local",
            timeout,
        )
        err = _selftest_artifact(proj / "out" / "result.txt", "token=local")
        if code == 0 and err is None:
            print(f"selftest: local  OK ({time.time() - start:.1f}s)")
        else:
            failures.append("local")
            print(f"selftest: local  FAIL (exit {code}{'; ' + err if err else ''})")
            print(out[-1500:])

        workers = load_workers()
        target = None
        if not local_only:
            if worker:
                target = next((w for w in workers if w.name == worker), None)
                if target is None:
                    failures.append("remote")
                    print(f"selftest: remote FAIL (worker {worker!r} not configured)")
            elif workers:
                target = workers[0]
            if target is None and not failures:
                print("selftest: remote SKIPPED (no worker configured)")
            elif target is not None:
                caps = engine.remote_caps(target)
                if caps is None:
                    print(f"selftest: remote SKIPPED (worker {target.name} unreachable)")
                elif caps.get("policy_paused"):
                    print(f"selftest: remote SKIPPED (worker {target.name} paused by owner)")
                else:
                    start = time.time()
                    code, out = _selftest_run(
                        proj,
                        [
                            "run",
                            "-w",
                            target.name,
                            "--timeout",
                            str(timeout),
                            "--env",
                            "ORCHARD_SELFTEST_TOKEN",
                            "--",
                            "python",
                            "main.py",
                        ],
                        "remote",
                        timeout + 60,
                    )
                    out_root = proj / ".orchard-out"
                    art = None
                    if out_root.exists():
                        dirs = sorted(
                            (d for d in out_root.iterdir() if d.is_dir()),
                            key=lambda d: d.stat().st_mtime,
                        )
                        if dirs:
                            art = dirs[-1] / "out" / "result.txt"
                    err = _selftest_artifact(art, "token=remote")
                    if code == 0 and err is None:
                        print(f"selftest: remote OK ({time.time() - start:.1f}s)")
                    else:
                        failures.append("remote")
                        print(f"selftest: remote FAIL (exit {code}{'; ' + err if err else ''})")
                        print(out[-1500:])
    finally:
        if keep:
            print(f"kept: {proj}")
        else:
            shutil.rmtree(proj, ignore_errors=True)
    if failures:
        print(f"selftest: FAILED ({', '.join(failures)})")
        raise typer.Exit(1)
    print("selftest: all good")


def main() -> None:
    # The menu-bar app starts dispatch with the GUI's bare PATH (/usr/bin:/bin:/usr/sbin:/sbin).
    # require_uv() and every local job (they inherit this environment) need Homebrew's dirs.
    os.environ["PATH"] = dispatcher.merged_path(os.environ.get("PATH", ""))
    app()


if __name__ == "__main__":
    main()
