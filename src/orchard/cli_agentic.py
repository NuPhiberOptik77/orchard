"""Agent-native commands: receipt, consensus, fork, speculate, replica, mind, live.

They live apart from cli.py so the original commands stay readable; importing
this module registers them on the same Typer app. See docs/agent-native.md.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import typer

from . import consensus, fleet, live, mind, placement, receipts, replica, speculate, transport, worlds
from .job import slugify
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


@app.command()
def fork(
    cmd: list[str] = typer.Argument(..., help="check to run in every world (after --)"),
    patch: list[Path] = typer.Option([], "--patch", "-p", help="a unified diff to try (repeatable)"),
    baseline: bool = typer.Option(False, "--baseline", help="also run the unpatched tree"),
    first_only: bool = typer.Option(False, "--first", help="start no more worlds once one patch passes"),
    keep: bool = typer.Option(False, "--keep", help="keep every world folder (default: only the winner's)"),
    on: list[str] = typer.Option([], "--on", help="only use these machines (repeatable)"),
    share: list[str] = typer.Option([], "--share", help="also link this ignored top-level folder (repeatable)"),
    no_share: bool = typer.Option(False, "--no-share", help="link nothing; for patches that change dependencies"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before each world is killed"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Try several patches at once, each in its own copy of the tree, and name
    the first that passes. Exit 0 with a winner, 1 without, 2 if none ran."""
    if not patch:
        print("error: give at least one --patch", file=sys.stderr)
        raise typer.Exit(2)
    missing = [str(p) for p in patch if not p.is_file()]
    if missing:
        print(f"error: no such patch: {', '.join(missing)}", file=sys.stderr)
        raise typer.Exit(2)
    root = Path.cwd()
    token = fleet.new_token()
    say = (lambda _msg: None) if json_out else print
    shared = () if no_share else tuple(dict.fromkeys(worlds.SHARED_DIRS + tuple(share)))
    entries = ([{"label": "baseline", "patch": None}] if baseline else []) + \
        [{"label": str(p), "patch": p.resolve()} for p in patch]
    base = worlds.WORLDS_DIR / f"fork-{token}"
    for i, w in enumerate(entries):
        w["dir"] = base / f"{i}-{slugify(Path(w['label']).name)}"
        worlds.copy_tree(root, w["dir"], shared)
        if w["patch"] is not None:
            ok, err = worlds.apply_patch(w["dir"], w["patch"])
            if not ok:
                w.update(state="patch does not apply", why=err.splitlines()[0] if err else "")
    ready = [w for w in entries if "state" not in w]
    targets = list(on) or fleet.capable_targets(fleet.job_like_run(cmd, root, tag, env_names, timeout))
    if not targets:
        print("error: no reachable machine can run this check", file=sys.stderr)
        raise typer.Exit(2)
    run_opts = {"tags": tag, "env_names": env_names, "timeout": timeout}
    plans = []
    for (i, w), target in zip(enumerate(ready), worlds.assign(len(ready), targets)):
        w["name"] = fleet.job_name("fork", token, target, i)
        plans.append({"target": target, "cmd": cmd, "cwd": w["dir"], "name": w["name"], **run_opts})
    say(f"fork: {len(ready)} world(s) on {', '.join(sorted(set(p['target'] for p in plans)))}")
    base_prefix = next((slugify(w["name"]) + "-" for w in ready if w["patch"] is None), None)

    def stop(result: dict) -> bool:
        is_base = base_prefix is not None and (result.get("job") or "").startswith(base_prefix)
        return first_only and consensus.passed(result) and not is_base

    for w, result in zip(ready, fleet.run_spread(plans, stop=stop)):
        w.update({k: result.get(k) for k in ("target", "job", "state", "why")})
    decided = worlds.verdict(entries)
    winner = next((w for w in entries if w["label"] == decided["winner"]), None)
    stat = worlds.diff_stat(winner["patch"], root) if winner else ""
    for w in entries:
        if not keep and w is not winner:
            shutil.rmtree(w["dir"], ignore_errors=True)
    if base.exists() and not any(base.iterdir()):
        base.rmdir()
    if json_out:
        rows = [{"label": w["label"], "patch": str(w["patch"]) if w["patch"] else None,
                 **{k: w.get(k) for k in ("target", "job", "state", "why")},
                 "world": str(w["dir"]) if w["dir"].exists() else None} for w in entries]
        print(json.dumps({**decided, "stat": stat, "worlds": rows}, indent=2))
    else:
        for w in entries:
            tail = w.get("job") or f"({w.get('why') or ''})"
            print(f"  {w['label']:<24} {str(w.get('target') or '-'):<10} {str(w.get('state')):<20} {tail}")
        print(f"fork: {decided['summary']}")
        if stat:
            print("\n".join("  " + line for line in stat.splitlines()))
        if winner:
            print(f"  world: {winner['dir']}")
    if winner:
        raise typer.Exit(0)
    raise typer.Exit(1 if ready else 2)


@app.command()
def prove(
    cmd: Optional[list[str]] = typer.Argument(None, help="command to prove (after --)"),
    on: list[str] = typer.Option([], "--on", help="run only on these machines (repeatable)"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before each run is killed"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Run a command on every reachable machine, whatever its OS, and record
    where it passes. Afterwards `-t os:proven` lets it run on any of those.
    Exit 0 passed everywhere it ran, 1 failed somewhere, 2 nothing finished."""
    if not cmd:
        print("error: provide a command after --", file=sys.stderr)
        raise typer.Exit(2)
    cwd = Path.cwd()
    token = fleet.new_token()
    say = (lambda _msg: None) if json_out else print
    tags = list(dict.fromkeys([*tag, placement.OS_TAG_PREFIX + placement.ANY_OS]))
    targets = list(on) or fleet.capable_targets(fleet.job_like_run(cmd, cwd, tags, env_names, timeout))
    if not targets:
        print("prove: no reachable machine can run this", file=sys.stderr)
        raise typer.Exit(2)
    say(f"prove: running on {', '.join(targets)}")
    plans = [{"target": t, "cmd": cmd, "cwd": cwd, "name": fleet.job_name("prove", token, t, i),
              "tags": tags, "env_names": env_names, "timeout": timeout} for i, t in enumerate(targets)]
    runs = fleet.run_spread(plans)
    done = [r for r in runs if consensus.usable(r)]
    passed_on = sorted({r.get("os_family") or "?" for r in done if consensus.passed(r)})
    failing = [r for r in done if not consensus.passed(r)]
    failed_on = sorted({r.get("os_family") or "?" for r in failing})
    label = reason = sig = None
    if failing and passed_on:
        # A split between machines is a consensus verdict; remember it like one.
        label, reason = consensus.classify(failing[0], [r for r in done if r is not failing[0]])
        sig = consensus.signature(failing[0].get("log_tail") or "")
    allowed = sorted({placement.DEFAULT_OS, *receipts.proven_oses(cmd, cwd)})
    out = {
        "cmd": list(cmd),
        "passed_on": passed_on,
        "failed_on": failed_on,
        "label": label,
        "reason": reason,
        "proven_allows": allowed,
        "runs": [{k: r.get(k) for k in ("target", "job", "state", "machine", "os_family", "why")} for r in runs],
    }
    if label:
        consensus.remember({"label": label, "reason": reason, "signature": sig,
                            "signature_id": consensus.signature_id(sig), "cmd": list(cmd),
                            "runs": [{k: r.get(k) for k in ("target", "job", "state", "machine", "os_family",
                                                            "load_per_core")} for r in done]})
    if json_out:
        print(json.dumps(out, indent=2))
    else:
        for run in runs:
            print(_run_line("run", run))
        if not done:
            print("prove: no run finished")
        elif not failing:
            print(f"prove: passes on {', '.join(passed_on)}")
        elif not passed_on:
            print(f"prove: fails everywhere it ran ({', '.join(failed_on)})")
        else:
            print(f"prove: passes on {', '.join(passed_on)}; fails on {', '.join(failed_on)} ({label}: {reason})")
        print(f"  -t os:proven may now run it on: {', '.join(allowed)}")
    if not done:
        raise typer.Exit(2)
    raise typer.Exit(1 if failing else 0)


speculate_app = typer.Typer(no_args_is_help=True, help="Run likely checks before anyone asks for them.")
app.add_typer(speculate_app, name="speculate")


def _check_line(c: dict) -> str:
    extra = [f"-t {t}" for t in c["tags"]] + [f"--env {e}" for e in c["env"]]
    if c.get("timeout"):
        extra.append(f"--timeout {c['timeout']}")
    return " ".join(c["cmd"]) + (f"  ({' '.join(extra)})" if extra else "")


def _print_report(report: dict) -> None:
    if report.get("error"):
        print(f"  error      {report['error']} (will retry)")
    for row in report["checks"]:
        detail = row.get("job") or row.get("why") or ""
        print(f"  {row['state']:10} {' '.join(row['cmd'])}  {detail}".rstrip())
    for job in report["cancelled"]:
        print(f"  cancelled  {job} (stale)")


@speculate_app.command("add")
def speculate_add(
    cmd: Optional[list[str]] = typer.Argument(None, help="check to speculate (after --)"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before the run is killed"),
) -> None:
    """Add a check this project is likely to be asked for."""
    if not cmd:
        print("error: provide a command after --", file=sys.stderr)
        raise typer.Exit(2)
    check = speculate.make_check(cmd, tag, env_names, timeout)
    print(f"speculate: {speculate.add(Path.cwd(), check)} {_check_line(check)}")


@speculate_app.command("remove")
def speculate_remove(cmd: Optional[list[str]] = typer.Argument(None, help="command to stop speculating")) -> None:
    """Stop speculating a command (every tag and env variant of it)."""
    if not cmd:
        print("error: provide a command after --", file=sys.stderr)
        raise typer.Exit(2)
    n = speculate.remove(Path.cwd(), cmd)
    print(f"speculate: removed {n} check{'s' if n != 1 else ''}")
    raise typer.Exit(0 if n else 1)


@speculate_app.command("list")
def speculate_list(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    """Show this project's checks and whether the tree as it is now is answered."""
    cwd = Path.cwd()
    checks = speculate.load(cwd)
    rows, stale = speculate.survey(cwd, checks) if checks else ([], [])
    if json_out:
        print(json.dumps({"checks": [{**c, **r} for r, c in rows], "stale": [e["id"] for e in stale]}, indent=2))
        return
    if not checks:
        print("no checks yet: orchard speculate add -- <cmd>, or orchard speculate learn")
        return
    for row, c in rows:
        print(f"  {row['state']:10} {_check_line(c)}  {row.get('job') or ''}".rstrip())
    if stale:
        print(f"  {len(stale)} stale speculative job(s); orchard speculate run cancels them")


@speculate_app.command("learn")
def speculate_learn(
    top: int = typer.Option(3, "--top", help="how many of the most-run checks to take"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Add the checks most often run in this project, read from receipts."""
    cwd = Path.cwd()
    found = speculate.learn(cwd, top)
    have = speculate.load(cwd)
    for check in found:
        check = {k: v for k, v in check.items() if k != "runs"}
        if not any(speculate._same(old, check) for old in have):  # keep a timeout already set
            speculate.add(cwd, check)
    if json_out:
        print(json.dumps(found, indent=2))
        return
    if not found:
        print("speculate: no receipts for this project yet; run some checks first")
        return
    for check in found:
        print(f"  learned  {_check_line(check)}  ({check['runs']} runs)")


@speculate_app.command("run")
def speculate_run(
    limit: int = typer.Option(speculate.MAX_IN_FLIGHT, "--max", help="most speculative jobs in flight for this project"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Queue every check the tree as it is now has no receipt for, and cancel
    this project's speculative jobs that went stale. Never touches other jobs."""
    cwd = Path.cwd()
    if not speculate.load(cwd):
        print("error: no checks yet: orchard speculate add -- <cmd>, or orchard speculate learn", file=sys.stderr)
        raise typer.Exit(2)
    report = speculate.run(cwd, limit=limit)
    if json_out:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    raise typer.Exit(1 if any(r["state"] == "not queued" for r in report["checks"]) else 0)


@speculate_app.command("watch")
def speculate_watch(
    settle: float = typer.Option(5.0, "--settle", min=0, help="seconds the tree must hold still before a run"),
    interval: float = typer.Option(1.0, "--interval", min=0.1, help="seconds between looks at the tree"),
    limit: int = typer.Option(speculate.MAX_IN_FLIGHT, "--max", help="most speculative jobs in flight for this project"),
) -> None:
    """Speculate now, then each time the tree settles after a change. Ctrl-C stops."""
    cwd = Path.cwd()
    if not speculate.load(cwd):
        print("error: no checks yet: orchard speculate add -- <cmd>, or orchard speculate learn", file=sys.stderr)
        raise typer.Exit(2)

    def say(report: dict) -> None:
        print(time.strftime("%H:%M:%S"), "speculate:")
        _print_report(report)
        sys.stdout.flush()

    try:
        speculate.watch(cwd, settle, interval, limit, say=say)
    except KeyboardInterrupt:
        print("speculate: stopped")


@speculate_app.command("stats")
def speculate_stats(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    """How often speculation paid off in this project, and what it cost."""
    s = speculate.stats(Path.cwd())
    if json_out:
        print(json.dumps(s, indent=2))
        return
    if not s["speculated"]:
        print("speculate: no speculative runs have finished in this project yet")
        return
    print(f"speculate: {s['asked_for']} of {s['speculated']} speculative runs were asked for "
          f"(hit rate {s['hit_rate']:.0%})")
    print(f"  agent wait saved: {s['saved_s']}s; CPU spent on runs nobody asked for (yet): {s['wasted_s']}s")


replica_app = typer.Typer(no_args_is_help=True, help="Living replicas: a warm copy of the project on each body.")
app.add_typer(replica_app, name="replica")


def _replica_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except replica.ReplicaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(2)
    except (transport.RemoteError, subprocess.TimeoutExpired) as exc:
        print(f"error: a body went quiet ({exc})", file=sys.stderr)
        raise typer.Exit(2)


def _need_cmd(cmd: Optional[list[str]]) -> list[str]:
    if not cmd:
        print("error: provide a command after --", file=sys.stderr)
        raise typer.Exit(2)
    return cmd


@replica_app.command("up")
def replica_up(
    worker: str = typer.Option(..., "--worker", "-w", help="body to hold the replica"),
    role: Optional[str] = typer.Option(None, "--role", help="day (acts) or dusk (remembers the last green tree)"),
    prep: Optional[str] = typer.Option(None, "--prep", help="command that readies a fresh copy, e.g. 'npm ci' (\"\" clears)"),
    prep_key: list[str] = typer.Option([], "--prep-key", help="file or glob whose change reruns prep (repeatable; default: lockfiles)"),
    augment: list[str] = typer.Option([], "--augment", help="tag this replica gains from tools it installs in tools/bin (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to prep (repeatable)"),
) -> None:
    """Create or refresh a replica on a body: copy the project, build its env, run prep (Dawn)."""
    raise typer.Exit(_replica_call(replica.up, Path.cwd(), worker, role, prep, prep_key, augment, env_names))


@replica_app.command("run")
def replica_run(
    cmd: Optional[list[str]] = typer.Argument(None, help="command to run (after --)"),
    worker: Optional[str] = typer.Option(None, "--worker", "-w", help="run on this replica only"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before the run is killed"),
    reuse: bool = typer.Option(False, "--reuse", help="answer from a sealed passing receipt for this exact tree and check"),
    as_is: bool = typer.Option(False, "--as-is", help="run on the body's copy as it is, without sending this tree (needs -w)"),
    bless: bool = typer.Option(False, "--bless", help="when it passes, move the dusk to this tree"),
    no_local: bool = typer.Option(False, "--no-local", help="never fall back to running on this Mac"),
    target_file: Optional[Path] = typer.Option(None, "--target-file", hidden=True),
    result_file: Optional[Path] = typer.Option(None, "--result-file", hidden=True),
    pooled: bool = typer.Option(False, "--pooled", hidden=True),
    sight: bool = typer.Option(False, "--sight", help="when it fails, look again on the same body and on a second body: code, body or chance"),
) -> None:
    """Send only what changed to the nearest free replica and run there (Day)."""
    cmd = _need_cmd(cmd)
    raise typer.Exit(_replica_call(replica.run, cmd, Path.cwd(), worker, tag, env_names, timeout, reuse, as_is,
                                   bless, no_local, target_file, result_file, pooled, sight))


@replica_app.command("check")
def replica_check(
    split: Optional[str] = typer.Option(None, "--split", help="command template with {i} and {n}, e.g. 'pytest --shard {i}/{n}'"),
    shards: int = typer.Option(0, "--shards", min=0, help="how many shards --split makes"),
    no_local: bool = typer.Option(False, "--no-local", help="never fall back to running on this Mac"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to every check (repeatable)"),
    sight: bool = typer.Option(False, "--sight", help="when it fails, look again on the same body and on a second body: code, body or chance"),
) -> None:
    """Run every speculate check (or every shard) across the day replicas at once.
    When all pass on one tree, the dusk moves to it."""
    raise typer.Exit(_replica_call(replica.check, Path.cwd(), split, shards, no_local, env_names, sight))


@replica_app.command("compare")
def replica_compare(
    cmd: Optional[list[str]] = typer.Argument(None, help="command to run (after --)"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before the run is killed"),
) -> None:
    """Run a check on your tree and on the dusk's last green tree at once: is it your change?"""
    cmd = _need_cmd(cmd)
    raise typer.Exit(_replica_call(replica.compare, Path.cwd(), cmd, tag, env_names, timeout))


@replica_app.command("blame")
def replica_blame(
    cmd: Optional[list[str]] = typer.Argument(None, help="failing command (after --)"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to the job (repeatable)"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before each run is killed"),
) -> None:
    """Find the fewest changed files that make a check fail, testing mixed trees across every day replica."""
    cmd = _need_cmd(cmd)
    raise typer.Exit(_replica_call(replica.blame, Path.cwd(), cmd, tag, env_names, timeout))


@replica_app.command("list")
def replica_list(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    """Show this project's replicas: role, whether each holds your tree, drift, pending prep."""
    rows = _replica_call(replica.list_rows, Path.cwd())
    if json_out:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("no replicas yet: orchard replica up -w <worker>")
        return
    for r in rows:
        notes = []
        if r.get("augment"):
            notes.append("augmented: " + ", ".join(r["augment"]))
        if r.get("busy"):
            notes.append(f"busy: {r['busy']}")
        if r.get("drifted"):
            notes.append(f"drifted: {len(r['drifted'])} file(s)")
        if r.get("dawn_pending"):
            notes.append("prep pending")
        if r["role"] == "dusk" and "blessed" in r:
            notes.append("last green" if r["blessed"] else "not blessed")
        elif r.get("digest"):
            notes.append("holds your tree" if r.get("current") else "behind your tree")
        tree = (r.get("digest") or "")[:12] or "-"
        print(f"  {r['name']:16} {r['role']:5} {r['status']:8} {tree:12}  {'; '.join(notes)}".rstrip())


@replica_app.command("role")
def replica_role(
    name: str = typer.Argument(..., help="replica's worker name"),
    role: str = typer.Argument(..., help="day or dusk"),
) -> None:
    """Make a replica a day replica or the dusk."""
    _replica_call(replica.set_role, Path.cwd(), name, role)


@replica_app.command("down")
def replica_down(name: str = typer.Argument(..., help="replica's worker name")) -> None:
    """Remove a replica's copy from its body and forget it here."""
    _replica_call(replica.down, Path.cwd(), name)


# ---------------------------------------------------------------- mind: one memory, one plan, every body
# Opt-in: nothing above reads or writes mind's memory. Output names a body by
# the first 8 hex of its id; --bodies adds which worker each id is here.

mind_app = typer.Typer(no_args_is_help=True,
                       help="One mind: one memory, one plan, every body. Opt-in; nothing else changes.")
app.add_typer(mind_app, name="mind")


def _mind_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except mind.MindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(2)
    except (transport.RemoteError, subprocess.TimeoutExpired) as exc:
        print(f"error: a body went quiet ({exc})", file=sys.stderr)
        raise typer.Exit(2)


def _mind_json(obj, bodies: bool = False) -> None:
    ids = mind.body_ids(mind.load_records())
    out = mind.public(obj, ids)
    if bodies and isinstance(out, dict):
        out["bodies"] = mind.bodies_view(ids)
    print(json.dumps(out, indent=2, sort_keys=True))


def _ms(ms) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}s"


def _body(b) -> str:
    return f"body {b[:8]}" if b else ""


def _print_bodies() -> None:
    for b, target in sorted(mind.bodies_view(mind.body_ids(mind.load_records())).items()):
        print(f"  body {b} = {target}")


_MIND_ENDS = {
    "failed": "failed. `orchard mind why <step>` says why; `orchard mind retry <step>` reopens it",
    "stranded": "stranded: no body can run a step. `orchard mind why <step>` says why",
    "usage": "no body has reported for this goal yet. Check `orchard workers`, then run again",
    "tree_changed": "the tree changed under this goal. Run again for the new tree",
    "detached": "detached. Launched steps keep running; the next `orchard mind run` adopts them",
}


def _print_run(out: dict, bodies: bool) -> None:
    head = "plan only, nothing written" if out["plan_only"] else ("from memory" if out["from_memory"] else "")
    print(f"mind: goal {out['goal'][:12]}, {len(out['steps'])} step(s){', ' + head if head else ''}")
    why = {st["step"]: st["why"] for st in out["plan"]["steps"]}
    for s in out["steps"]:
        note = "(memory)" if s["from_memory"] else ""
        print(f"  {s['step'][:12]} {s['state']:8} {_ms(s['elapsed_ms']):>8}  {s['cmd']}  {_body(s['body'])} {note}"
              .rstrip())
        if s["state"] in ("pending", "stranded", "failed"):
            for w in why.get(s["step"], []):
                print(f"      {w}")
    if out.get("sync"):
        for when, t in out["sync"].items():
            print(f"mind: sync {when}: pulled {t['pulled']}, pushed {t['pushed']}, bad {t['bad']}, "
                  f"unreachable {t['unreachable']}")
    if bodies:
        _print_bodies()
    state = out["state"]
    if out["plan_only"]:
        print(f"mind: plan only: {state}. Nothing was written and no body was reached")
    elif state == "passed":
        print(f"mind: all passed. {out['coord_msgs']} coordination messages, {len(out['recover_s'])} recoveries")
    else:
        print(f"mind: {_MIND_ENDS.get(state, state)}")


@mind_app.command("run")
def mind_run(
    cmd: Optional[list[str]] = typer.Argument(None, help="command to run (after --)"),
    step: list[str] = typer.Option([], "--step", help="another step, one shell-quoted string (repeatable)"),
    use_speculate: bool = typer.Option(False, "--speculate", help="add this project's speculate checks as steps"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="required worker capability tag (repeatable)"),
    env_names: list[str] = typer.Option([], "--env", help="forward this env var to every step (repeatable)"),
    timeout: int = typer.Option(mind.DEFAULT_TIMEOUT_S, "--timeout", min=1, help="seconds before a step is killed"),
    quiet_after: int = typer.Option(mind.QUIET_DEFAULT_S, "--quiet-after",
                                    help=f"seconds a body may stay silent before its step counts as lost "
                                         f"(at least {mind.QUIET_MIN_S})"),
    trust: list[str] = typer.Option([], "--trust", help="also accept results under this key id (repeatable; "
                                                         "see `orchard mind key`)"),
    reuse: bool = typer.Option(False, "--reuse", help="lift sealed passing receipts for this exact tree and check"),
    fresh: bool = typer.Option(False, "--fresh", help="run every step again, even ones memory already answers"),
    plan_only: bool = typer.Option(False, "--plan-only", help="show the plan from memory; write and reach nothing"),
    sync_too: bool = typer.Option(False, "--sync", help="swap memory with every worker before and after"),
    bodies: bool = typer.Option(False, "--bodies", help="also show which worker each body id is here"),
    as_json: bool = typer.Option(False, "--json", help="print the session summary as JSON"),
) -> None:
    """Run a goal (one or more steps) across every body until it passes, fails or strands.

    Exits 0 passed, 1 failed, 2 usage, 3 stranded, 4 tree changed, 130 detached."""
    workers = _mind_call(mind.workers_here)

    def say(msg: str) -> None:
        line = msg if bodies else mind.hide_names(msg, workers)
        print(f"mind: {line}", file=sys.stderr if as_json else sys.stdout, flush=True)

    out = _mind_call(mind.run_goal, Path.cwd(), cmd, step, use_speculate=use_speculate, tags=tag,
                     env_names=env_names, timeout=timeout, quiet_s=quiet_after, trust=trust, reuse=reuse,
                     fresh=fresh, plan_only=plan_only, sync_too=sync_too, say=say, workers=workers)
    by_worker = out.pop("sync_workers", None)
    if as_json:
        if bodies and by_worker is not None:
            out["sync_workers"] = by_worker
        _mind_json(out, bodies)
    else:
        _print_run(out, bodies)
    raise typer.Exit(out["exit"])


@mind_app.command("plan")
def mind_plan(
    goal: Optional[str] = typer.Argument(None, help="goal id or prefix (default: this project's newest)"),
    bodies: bool = typer.Option(False, "--bodies", help="also show which worker each body id is here"),
    as_json: bool = typer.Option(False, "--json", help="print the plan as JSON"),
) -> None:
    """Show the plan memory gives a goal right now. Reads only; writes nothing."""
    gid = _mind_call(mind.resolve_goal, Path.cwd(), goal)
    records = mind.load_records()
    p = mind.plan(records, gid)
    if as_json:
        _mind_json(p, bodies)
        return
    g = mind.goal_of(records, gid)
    cmds = {s["step_id"]: shlex.join(s["cmd"]) for s in (g["body"]["steps"] if g else [])}
    print(f"mind: goal {gid[:12]} {p['state']} (plan {p['digest'][:12]})")
    for st in p["steps"]:
        print(f"  {st['step'][:12]} {st['state']:8} gen {st['gen']}  {cmds.get(st['step'], '?')}  "
              f"{_body(st['body'])}".rstrip())
        for w in st["why"]:
            print(f"      {w}")
    for b, queue in sorted(p["queues"].items()):
        print(f"  next on body {b[:8]}: {', '.join(s[:12] for s in queue)}")
    if bodies:
        _print_bodies()


@mind_app.command("status")
def mind_status(as_json: bool = typer.Option(False, "--json", help="print the goals as JSON")) -> None:
    """This project's goals in memory here, newest first, and whether a session is working on each."""
    rows = _mind_call(mind.status_rows, Path.cwd())
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        print("mind: no goal for this project in memory here")
        return
    for r in rows:
        counts = ", ".join(f"{n} {s}" for s, n in r["steps"].items())
        notes = [n for n, on in (("watched", r["watching"]), ("this tree", r["this_tree"])) if on]
        print(f"  {r['goal'][:12]} {r['state']:8} {r['total']} step(s): {counts}  {'; '.join(notes)}".rstrip())


@mind_app.command("why")
def mind_why(
    step: str = typer.Argument(..., help="step id prefix, or the step's command text"),
    goal: Optional[str] = typer.Option(None, "--goal", help="goal id or prefix (default: this project's newest)"),
    bodies: bool = typer.Option(False, "--bodies", help="also show which worker each body id is here"),
    as_json: bool = typer.Option(False, "--json", help="print the explanation as JSON"),
) -> None:
    """Why a step is where it is: what each body can run, and every attempt so far."""
    gid = _mind_call(mind.resolve_goal, Path.cwd(), goal)
    ex = _mind_call(mind.explain, gid, step)
    if as_json:
        _mind_json(ex, bodies)
        return
    print(f"mind: step {ex['step'][:12]} {ex['state']} at gen {ex['gen']}: {ex['cmd']}")
    for w in ex["why"]:
        print(f"  {w}")
    for b in ex["bodies"]:
        fit = "can run it" if b["fits"] else "cannot run it: " + "; ".join(b["problems"])
        print(f"  body {b['body'][:8]} (bench {b['bench']}) {fit}")
    for h in ex["history"]:
        line = f"  gen {h['gen']} {h['kind']}"
        if h["kind"] == "lost":
            line += f" on {_body(h['body'])}: {h['why']}" + (" (held)" if h["hold"] else "")
        elif h["kind"] == "result":
            line += f" {h['state']} in {_ms(h['elapsed_ms'])} on {_body(h['body'])}"
        elif h["kind"] == "launch":
            line += f" on {_body(h['body'])}"
        print(line)
    for u in ex["untrusted"]:
        print(f"  gen {u['gen']} {u['state']} under key {u['key']} ({u['status']}); not trusted by this goal")
    if bodies:
        _print_bodies()


@mind_app.command("memory")
def mind_memory(
    records: bool = typer.Option(False, "--records", help="list every record with its status and sources"),
    as_json: bool = typer.Option(False, "--json", help="print as JSON"),
) -> None:
    """What memory holds here: segments, records by kind, status and key, and how they arrived."""
    v = _mind_call(mind.memory_view, records)
    if as_json:
        print(json.dumps(v, indent=2, sort_keys=True))
        return
    kinds = ", ".join(f"{n} {k}" for k, n in v["kinds"].items()) or "none"
    status = ", ".join(f"{n} {s}" for s, n in v["status"].items()) or "none"
    arrivals = ", ".join(f"{n} {s}" for s, n in v["arrivals"].items()) or "none"
    print(f"mind: {v['segments']} segment(s), {v['records']} record(s): {kinds}")
    print(f"  status: {status}")
    print(f"  arrived by: {arrivals}")
    for k in v["by_key"]:
        print(f"  key {k['key']}: {k['records']} record(s){' (this install)' if k['mine'] else ''}")
    for e in v.get("entries", []):
        print(f"  {e['id'][:12]} {e['kind']:7} key {e['key'][:8]} {e['status']:12} {','.join(e['sources'])}")


@mind_app.command("sync")
def mind_sync(
    worker: Optional[str] = typer.Argument(None, help="only this worker (default: every worker)"),
    bodies: bool = typer.Option(False, "--bodies", help="also show each worker's own counts"),
    as_json: bool = typer.Option(False, "--json", help="print the counts as JSON"),
) -> None:
    """Swap memory segments with workers. Workers only relay: nothing on them is trusted."""
    workers = _mind_call(mind.workers_here)
    if worker is not None:
        if worker not in workers:
            print(f"error: no worker named {worker!r}; see `orchard workers`", file=sys.stderr)
            raise typer.Exit(2)
        workers = {worker: workers[worker]}
    res = _mind_call(mind.sync, workers, Counter())
    totals = mind.sync_totals(res)
    if as_json:
        out = {"workers": len(res), "totals": totals}
        if bodies:
            out["by_worker"] = res
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"mind: synced with {len(res)} worker(s): pulled {totals['pulled']}, pushed {totals['pushed']}, "
              f"bad {totals['bad']}, unreachable {totals['unreachable']}")
        if bodies:
            for name, got in res.items():
                print(f"  {name}: " + ", ".join(f"{k} {n}" for k, n in got.items()))
    raise typer.Exit(1 if totals["unreachable"] else 0)


@mind_app.command("export")
def mind_export(
    file: Path = typer.Argument(..., help="file to write (an existing directory gets one file per segment)"),
) -> None:
    """Write every memory segment into one file another install can import."""
    n = _mind_call(mind.export_memory if file.is_dir() else mind.export_file, file)
    print(f"mind: exported {n} segment(s) to {file}")


@mind_app.command("import")
def mind_import(
    file: Path = typer.Argument(..., help="file from `orchard mind export` (or a directory of segments)"),
    as_json: bool = typer.Option(False, "--json", help="print the counts as JSON"),
) -> None:
    """Store the segments of an export here. A segment whose bytes do not match its name is rejected;
    a record under this install's key that does not verify is kept as forged and never trusted."""
    res = _mind_call(mind.import_memory if file.is_dir() else mind.import_file, file)
    rejected = res.get("rejected", res.get("skipped", 0))
    if as_json:
        print(json.dumps(res, indent=2, sort_keys=True))
    else:
        print(f"mind: imported {res['segments']} segment(s), {res['new']} new; {rejected} rejected; "
              f"{res['forged']} forged record(s)")
    raise typer.Exit(1 if rejected else 0)


@mind_app.command("retry")
def mind_retry(
    step: str = typer.Argument(..., help="step id prefix, or the step's command text"),
    goal: Optional[str] = typer.Option(None, "--goal", help="goal id or prefix (default: this project's newest)"),
) -> None:
    """Reopen a failed or stranded step; the next `orchard mind run` of the same goal runs it again."""
    gid = _mind_call(mind.resolve_goal, Path.cwd(), goal)
    res = _mind_call(mind.retry_step, gid, step)
    print(f"mind: step {res['step'][:12]} reopened at gen {res['gen']} ({res['state']})")


@mind_app.command("key")
def mind_key() -> None:
    """Print this install's key id, for another install's `mind run --trust`."""
    print(_mind_call(mind.key_id))


@app.command("live")
def live_cmd(
    cmd: Optional[list[str]] = typer.Argument(None, help="check to run (after --); {path} and {unit} are filled in"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="seconds before a check is killed"),
    link: list[str] = typer.Option([], "--link", help="ignored folder to link into the shadow copy (repeatable; .venv, venv and node_modules always are)"),
    no_speculate: bool = typer.Option(False, "--no-speculate", help="never queue the registered checks the model names"),
) -> None:
    """Check a file each time a function in it is finished, while an AI is still writing it.

    Reads JSON lines on stdin ({"path", "text"}, {"path", "reset"}, {"path", "done"}, {"think"}),
    prints one JSON line per check, and exits 1 at the first failure."""
    cmd = _need_cmd(cmd)
    try:
        code = live.serve(cmd, Path.cwd(), timeout=timeout or None, links=link, speculate_on=not no_speculate)
    except ValueError as exc:  # the project is over the shipping size cap
        typer.echo(f"orchard live: {exc}", err=True)
        code = 2
    raise typer.Exit(code)
