"""Agent-native commands: receipt, consensus, fork, speculate, replica, live.

They live apart from cli.py so the original commands stay readable; importing
this module registers them on the same Typer app. See docs/agent-native.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from . import consensus, fleet, receipts
from .cli import app, find_jobs


def _out_dir(receipt: dict) -> Path | None:
    project = receipt.get("project")
    if not project:
        return None
    name = (receipt.get("worker") or {}).get("name")
    if name == "local":
        return Path(project) / "out"
    return Path(project) / ".orchard-out" / receipt["job"] / "out"


def _when(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "-"


@app.command()
def receipt(
    job: str = typer.Argument(..., help="job id or prefix"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
    deep: bool = typer.Option(False, "--deep", help="also check the log and out/ still match"),
) -> None:
    """Show a job's sealed receipt: what ran, where, on which exact inputs."""
    matches = find_jobs(job)
    if not matches:
        print(f"no job matching {job!r}", file=sys.stderr)
        raise typer.Exit(1)
    sdir = matches[-1]
    try:
        rec = json.loads((sdir / "receipt.json").read_text())
    except (OSError, json.JSONDecodeError):
        print(f"{sdir.name}: no receipt (still running, from before receipts, or never collected)", file=sys.stderr)
        raise typer.Exit(1)
    verified = receipts.verify(rec)
    if deep:
        problems = receipts.deep_check(rec, sdir, _out_dir(rec))
    else:
        problems = [] if verified else ["seal does not verify (receipt edited, or sealed under another key)"]
    if json_out:
        print(json.dumps({"receipt": rec, "verified": verified, "problems": problems}, indent=2))
    else:
        w = rec.get("worker") or {}
        work = rec.get("work") or {}
        print(f"job:      {rec.get('job')}")
        print(f"state:    {rec.get('state')} (exit {rec.get('exit_code')}) in {rec.get('elapsed_s')}s")
        print(f"ran on:   {w.get('name')} ({w.get('machine')}, {w.get('os_family')})")
        print(f"command:  {' '.join(work.get('cmd') or [])}")
        print(f"tree:     {rec.get('tree')} ({rec.get('files')} files)")
        print(f"work key: {rec.get('work_key')}")
        if work.get("env"):
            print(f"env:      {', '.join(work['env'])} (values hashed, not stored)")
        print(f"started:  {_when(rec.get('started_at'))}  sealed: {_when(rec.get('sealed_at'))}")
        print(f"seal:     {'verified' if verified else 'DOES NOT VERIFY'}")
        for problem in problems:
            print(f"problem:  {problem}")
        if deep and not problems:
            print("deep:     log and out/ match the receipt")
    if problems:
        raise typer.Exit(1)


def _run_line(kind: str, run: dict) -> str:
    load = run.get("load_per_core")
    where = f"{run.get('machine') or '?'}/{run.get('os_family') or '?'}"
    extra = f", load {load}" if load is not None else ""
    tail = f"  {run['job']}" if run.get("job") else f"  ({run.get('why', '')})"
    return f"  {kind:<6} {str(run.get('target')):<10} {str(run.get('state')):<11} ({where}{extra}){tail}"


@app.command("consensus")
def consensus_cmd(
    cmd: Optional[list[str]] = typer.Argument(None, help="command to check (after --)"),
    where: str = typer.Option("auto", "--where", "-w", help="where the first run goes: auto | local | worker name"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before each run is killed"),
    from_job: Optional[str] = typer.Option(None, "--job", help="use this finished job as the first run"),
    offenders: bool = typer.Option(False, "--offenders", help="list the failures most often labelled noise"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Rerun a failure across machines and label it: pass, flaky, real,
    load-sensitive, platform-only or machine-only. Exit 0 pass, 1 real, 2 other."""
    if offenders:
        rows = consensus.offenders()
        if json_out:
            print(json.dumps([{"signature": s, "labels": dict(c)} for s, c in rows], indent=2))
        for sig, seen in rows:
            print(f"{sum(seen.values()):>4}  {consensus.describe_memory(seen).split(': ', 1)[1]}  {sig}")
        if not rows and not json_out:
            print("no failure has been labelled noise yet")
        return
    cwd = Path.cwd()
    token = fleet.new_token()
    if from_job:
        matches = find_jobs(from_job)
        if not matches:
            print(f"no job matching {from_job!r}", file=sys.stderr)
            raise typer.Exit(1)
        try:
            spec = json.loads((matches[-1] / "spec.json").read_text())
        except (OSError, json.JSONDecodeError):
            print(f"{matches[-1].name}: no spec.json to rerun from", file=sys.stderr)
            raise typer.Exit(1)
        cmd = cmd or spec.get("cmd")
        tag = tag or spec.get("tags") or []
        env_names = env_names or spec.get("env") or []
        timeout = timeout or spec.get("timeout_s")
        first = fleet.result_of(matches[-1])
    if not cmd:
        print("error: provide a command after -- or --job", file=sys.stderr)
        raise typer.Exit(2)
    run_opts = {"tags": tag, "env_names": env_names, "timeout": timeout}
    # Progress lines would break --json output.
    say = (lambda _msg: None) if json_out else print
    if not from_job:
        say(f"consensus: first run ({where})")
        first = fleet.run_on(where, cmd, cwd, fleet.job_name("consensus", token, where), **run_opts)
    if not consensus.usable(first):
        print(f"consensus: the first run did not finish ({first.get('state')}: {first.get('why', '')})", file=sys.stderr)
        raise typer.Exit(2)
    reruns: list[dict] = []
    if not consensus.passed(first):
        job = fleet.job_like_run(cmd, cwd, tag, env_names, timeout)
        targets = fleet.capable_targets(job)
        if first.get("target") not in targets:
            targets.insert(0, first.get("target"))
        say(f"consensus: failed on {first.get('target')}; rerunning on {', '.join(targets)}")
        plans = [{"target": t, "cmd": cmd, "cwd": cwd, "name": fleet.job_name("consensus", token, t, i), **run_opts}
                 for i, t in enumerate(targets, 1)]
        reruns = fleet.run_spread(plans)
    label, reason = consensus.classify(first, reruns)
    sig = consensus.signature(first.get("log_tail") or "") if label != "pass" else None
    seen = consensus.recall(sig)
    verdict = {
        "label": label,
        "reason": reason,
        "signature": sig,
        "signature_id": consensus.signature_id(sig),
        "cmd": list(cmd),
        "runs": [{k: r.get(k) for k in ("target", "job", "state", "machine", "os_family", "load_per_core")}
                 for r in [first] + reruns],
    }
    if label != "pass":
        consensus.remember(verdict)
    if json_out:
        print(json.dumps({**verdict, "seen_before": dict(seen)}, indent=2))
    else:
        print(f"consensus: {label}")
        print(f"  why:   {reason}")
        print(_run_line("first", first))
        for run in reruns:
            print(_run_line("rerun", run))
        if sig:
            print(f"  error: {sig}")
        note = consensus.describe_memory(seen)
        if note:
            print(f"  memory: {note}")
    raise typer.Exit(consensus.exit_code(label))
