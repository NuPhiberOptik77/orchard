"""Run one command on chosen machines, as child `orchard run` processes.

Consensus, many worlds and speculation all need "run this on that machine
and tell me what happened". Going through a child `orchard run -w <target>`
keeps placement, environment sync, collection and receipts exactly as they
are for a plain run; this module only names the job so it can find it
afterwards, and reads back its status, log and receipt.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import threading
from pathlib import Path

from . import agent, engine, env, placement, receipts
from .config import JOBS_DIR, STATE_DIR, load_workers
from .job import JobSpec, capture_env, slugify

LOG_DIR = STATE_DIR / "fleet"


def job_like_run(cmd: list[str], cwd: Path, tags: list[str] | None = None,
                 env_names: list[str] | None = None, timeout: int | None = None) -> JobSpec:
    """The JobSpec `orchard run -- cmd` would build in cwd, so work keys match."""
    job = JobSpec(name=cmd[0] if cmd else "job", cmd=list(cmd))
    job.python = env.capture_python(cwd)
    job.tags = list(tags or [])
    job.timeout_s = timeout
    if env_names:
        job.env = list(dict.fromkeys(env_names))
        job.env_values, _missing = capture_env(job.env)
    job.requirements, _notes = env.capture_requirements(cwd)
    receipts.attach_proof(job, cwd)
    return job


def capable_targets(job: JobSpec) -> list[str]:
    """Every reachable machine that could run job: "local" first, then workers."""
    found = []
    local_caps = agent.collect_caps(env.ensure_bench_python())
    if not placement.problems(job, local_caps):
        found.append("local")
    for w in load_workers():
        caps = engine.remote_caps(w)
        if caps is None:
            continue
        if job.kind == "run":
            caps["platform_ok"], caps["platform_note"] = env.resolve_for_worker(job.requirements, job.python, caps)
        if not placement.problems(job, caps):
            found.append(w.name)
    return found


def new_token() -> str:
    return secrets.token_hex(3)


def job_name(prefix: str, token: str, target: str, index: int = 0) -> str:
    # Job ids keep 32 characters of the name, so the unique part goes first.
    return "-".join([prefix, token, str(index), slugify(target)])


def find_job(name: str) -> Path | None:
    """The newest job run under exactly this name."""
    base = JOBS_DIR
    if not base.exists():
        return None
    exact = re.compile(re.escape(slugify(name)) + r"-\d{8}-\d{6}-[0-9a-f]{4}")
    matches = sorted(d for d in base.iterdir() if exact.fullmatch(d.name))
    return matches[-1] if matches else None


def result_of(sdir: Path, target: str | None = None) -> dict:
    """What a finished job did: state, machine, load, log tail, receipt."""
    status = agent.read_status(str(sdir)) or {}
    if target is None:
        try:
            target = json.loads((sdir / "placement.json").read_text()).get("worker")
        except (OSError, json.JSONDecodeError):
            target = None
    log = ""
    try:
        log = (sdir / "log.txt").read_text(errors="replace")
    except OSError:
        pass
    return {
        "target": target,
        "job": sdir.name,
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "elapsed_s": status.get("elapsed_s"),
        "machine": status.get("machine"),
        "os_family": status.get("os_family"),
        "load_per_core": status.get("load_per_core"),
        "log_tail": "\n".join(log.splitlines()[-40:]),
        "sealed": (sdir / "receipt.json").exists(),
    }


def child_argv(target: str, name: str, cmd: list[str], tags: list[str] | None = None,
               env_names: list[str] | None = None, timeout: int | None = None,
               background: bool = False, no_local: bool = False) -> list[str]:
    argv = [sys.executable, "-m", "orchard", "run", "-w", target, "--name", name]
    if background:
        argv.append("-b")
    if no_local:
        argv.append("--no-local")
    if timeout:
        argv += ["--timeout", str(timeout)]
    for tag in tags or []:
        argv += ["-t", tag]
    for var in env_names or []:
        argv += ["--env", var]
    return argv + ["--", *cmd]


def child_env() -> dict[str, str]:
    """This environment, with PYTHONPATH set so a child imports this same
    Orchard however this one was started."""
    out = dict(os.environ)
    src = str(Path(__file__).resolve().parent.parent)
    out["PYTHONPATH"] = os.pathsep.join(p for p in (src, out.get("PYTHONPATH")) if p)
    return out


def run_on(target: str, cmd: list[str], cwd: Path, name: str, tags: list[str] | None = None,
           env_names: list[str] | None = None, timeout: int | None = None) -> dict:
    """Run cmd on target (a worker name, "local" or "auto") and wait for it."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    child_log = LOG_DIR / f"{slugify(name)}.log"
    with open(child_log, "wb") as out:
        proc = subprocess.run(child_argv(target, name, cmd, tags, env_names, timeout),
                              cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=child_env())
    sdir = find_job(name)
    if sdir is None:
        lines = child_log.read_text(errors="replace").strip().splitlines()
        return {"target": target, "job": None, "state": "unavailable", "exit_code": proc.returncode,
                "why": lines[-1] if lines else f"orchard run exited {proc.returncode}"}
    found = result_of(sdir, None if target == "auto" else target)
    if found["state"] is None:
        found["state"] = "unavailable"
        found["why"] = f"no status (orchard run exited {proc.returncode})"
    return found


def run_spread(plans: list[dict], stop=None) -> list[dict]:
    """Run plans ({target, cmd, cwd, name, ...run_on kwargs}) at once, one at a
    time per target, in plan order within a target. stop(result) -> True skips
    plans not yet started. Results come back in plan order."""
    results: list[dict | None] = [None] * len(plans)
    halt = threading.Event()
    by_target: dict[str, list[int]] = {}
    for i, plan in enumerate(plans):
        by_target.setdefault(plan["target"], []).append(i)

    def lane(indexes: list[int]) -> None:
        for i in indexes:
            plan = dict(plans[i])
            if halt.is_set():
                results[i] = {"target": plan["target"], "job": None, "state": "skipped", "why": "a winner was found"}
                continue
            results[i] = run_on(plan.pop("target"), plan.pop("cmd"), plan.pop("cwd"), plan.pop("name"), **plan)
            results[i]["target"] = results[i].get("target") or plans[i]["target"]
            if stop is not None and stop(results[i]):
                halt.set()

    threads = [threading.Thread(target=lane, args=(idx,), daemon=True) for idx in by_target.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r or {"state": "unavailable"} for r in results]
