"""Agent-native commands: receipt, consensus, fork, speculate, replica, live.

They live apart from cli.py so the original commands stay readable; importing
this module registers them on the same Typer app. See docs/agent-native.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

from . import receipts
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
